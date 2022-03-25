from collections import defaultdict
from math import pi
from typing import Callable, Tuple

import click
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pydd.analysis import calculate_match_unnormd_fft, get_match_pads, loglikelihood_fft
from pydd.binary import (
    DAY,
    DynamicDress,
    HOUR,
    MONTH,
    MSUN,
    Phi_to_c,
    VacuumBinary,
    WEEK,
    YR,
    convert,
    t_to_c,
)
from pydd.utils import get_target_pbh_dynamicdress
from scipy.optimize import minimize_scalar
from tqdm.auto import tqdm


Array = jnp.ndarray
M_1_RANGE = (1e-1 * MSUN, 10 ** 2.5 * MSUN)
M_2_RANGE = (1e-4 * MSUN, 1e-1 * MSUN)


def fit_v(
    dd: DynamicDress,
    S_n: Callable[[Array], Array],
    fs: Array,
    pad_low: Array,
    pad_high: Array,
) -> VacuumBinary:
    def fun(x):
        vb = VacuumBinary(x * MSUN, dd.Phi_c, dd.tT_c, dd.dL, dd.f_c)
        nll = -loglikelihood_fft(vb, dd, fs, pad_low, pad_high, S_n)
        return nll

    bracket = (dd.M_chirp / MSUN, dd.M_chirp / MSUN * (1 + 5e-3))
    res = minimize_scalar(fun, bracket=bracket)
    assert res.success
    return VacuumBinary(res.x * MSUN, dd.Phi_c, dd.tT_c, dd.dL, dd.f_c)


def get_statistics(
    m_1: float,
    m_2: float,
    t_obs: float,
    snr_thresh: float,
    S_n: Callable[[Array], Array],
    f_range_n: Tuple[float, float],
    n_f: int,
) -> dict:
    results = {}
    if m_2 / m_1 > 10 ** (-2.5):
        return defaultdict(lambda: jnp.nan)

    dd, f_range_d = get_target_pbh_dynamicdress(
        m_1, m_2, t_obs, snr_thresh, S_n, f_range_n
    )
    f_l = max(f_range_d[0], f_range_n[0])
    f_h = min(f_range_d[1], f_range_n[1])

    # System with no spike
    vb_0 = convert(dd, VacuumBinary)

    # Fit vacuum system
    fs = jnp.linspace(f_l, f_h, n_f)
    pad_low, pad_high = get_match_pads(fs)
    vb = fit_v(dd, S_n, fs, pad_low, pad_high)

    match = calculate_match_unnormd_fft(vb, dd, fs, pad_low, pad_high, S_n)

    # Save results
    results["M_chirp_MSUN"] = dd.M_chirp / MSUN
    results["M_chirp_MSUN_v"] = vb.M_chirp / MSUN
    results["t_in_band"] = t_to_c(f_l, dd) - t_to_c(f_h, dd)
    results["f_range_d"] = f_range_d
    results["dL"] = dd.dL
    results["N_in_band"] = (Phi_to_c(f_l, dd) - Phi_to_c(f_h, dd)) / (2 * pi)
    results["N_v_0_in_band"] = (Phi_to_c(f_l, vb_0) - Phi_to_c(f_h, vb_0)) / (2 * pi)
    results["N_v_in_band"] = (Phi_to_c(f_l, vb) - Phi_to_c(f_h, vb)) / (2 * pi)
    results["snr"] = snr_thresh
    results["snr_loss_frac"] = jnp.clip(
        (results["snr"] - jnp.sqrt(match)) / results["snr"], 0, 1
    )

    return results


def _parse_args(
    detector: str, t_obs: float, t_obs_units: str
) -> Tuple[Callable[[Array], Array], Tuple[float, float], float, float]:
    # Set noise and other globals based on detector
    if detector == "et":
        from pydd.noise import S_n_et as S_n, f_range_et as f_range_n
    elif detector == "ce":
        from pydd.noise import S_n_ce as S_n, f_range_ce as f_range_n
    elif detector == "aLIGO":
        from pydd.noise import S_n_aLIGO as S_n, f_range_aLIGO as f_range_n
    else:
        raise ValueError("invalid detector")

    snr_thresh = 12.0

    if t_obs_units not in ("SECOND", "HOUR", "DAY", "WEEK", "MONTH", "YR"):
        raise ValueError("invalid t_obs units")

    t_obs = t_obs * eval(t_obs_units)
    return S_n, f_range_n, snr_thresh, t_obs


@click.command()
@click.option("-d", "--detector", type=str, help="detector name")
@click.option("-t", "--t_obs", type=float, help="observing time")
@click.option(
    "-u",
    "--t-obs-units",
    type=str,
    help="units of t_obs ('SECOND', 'HOUR', 'DAY', 'WEEK', 'MONTH' or 'YR')",
)
@click.option("-n", "--n-f", default=10_000, help="number of frequency points")
@click.option("-n1", "--n-m1", default=5, help="number of points for m1 grid")
@click.option("-n2", "--n-m2", default=4, help="number of points for m2 grid")
@click.option(
    "-s", "--suffix", default="-test", help="suffix for saving plots and results"
)
def run(
    detector: str,
    t_obs: float,
    t_obs_units: str,
    n_f: int,
    n_m1: int,
    n_m2: int,
    suffix: str,
):
    S_n, f_range_n, snr_thresh, t_obs = _parse_args(detector, t_obs, t_obs_units)
    print(
        f"running discoverability analysis for {detector} with t_obs = {t_obs} {t_obs_units}"
    )

    m_1_g = jnp.geomspace(*M_1_RANGE, n_m1)
    m_2_g = jnp.geomspace(*M_2_RANGE, n_m2)
    m_1_mg, m_2_mg = jnp.meshgrid(
        jnp.geomspace(*M_1_RANGE, n_m1), jnp.geomspace(*M_2_RANGE, n_m2)
    )
    m_1s = m_1_mg.flatten()
    m_2s = m_2_mg.flatten()
    results = []
    for m_1, m_2 in tqdm(list(zip(m_1s, m_2s))):
        results.append(get_statistics(m_1, m_2, t_obs, snr_thresh, S_n, f_range_n, n_f))

    results = {
        k: jnp.array([r[k] for r in results]).reshape((len(m_2_g), len(m_1_g)))
        for k in results[0]
    }

    timestr = f"{t_obs:g}{t_obs_units.lower()}"
    base_path = f"vac-fit-{detector}-{timestr}{suffix}"
    results_path = f"vacuum_fits/{base_path}.npz"
    jnp.savez(results_path, m_1_mg=m_1_mg, m_2_mg=m_2_mg, **results)
    print(f"fitting complete, saved results to {results_path}")
