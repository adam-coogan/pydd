import pickle
from typing import Callable, Optional, Tuple

import click
import dynesty
from dynesty import plotting as dyplot
import jax
import jax.numpy as jnp
from matplotlib.figure import Figure
import numpy as np
from pydd.analysis import get_match_pads, loglikelihood_fft
from pydd.binary import (
    DAY,
    DynamicDress,
    HOUR,
    MONTH,
    MSUN,
    PC,
    WEEK,
    YR,
    get_f_isco,
    get_m_1,
    GAMMA_S_PBH,
)
from pydd.utils import get_target_pbh_dynamicdress

Array = jnp.ndarray


def get_r6T(rho_6):
    return rho_6 / (1e16 * MSUN / PC ** 3)


def get_r6(rho_6T):
    return 1e16 * MSUN / PC ** 3 * rho_6T


def get_q(M_chirp, rho_6):
    # Invert get_rho_6_pbh
    Mc_MSUN = M_chirp / MSUN
    a = 1.396e13 * MSUN / PC ** 3
    m1_MSUN = (rho_6 / a) ** (4 / 3)
    return (
        2 * 3 ** (1 / 3) * Mc_MSUN ** 5
        + 2 ** (1 / 3)
        * Mc_MSUN ** (10 / 3)
        * (9 * m1_MSUN ** (5 / 2) + jnp.sqrt(81 * m1_MSUN ** 5 - 12 * Mc_MSUN ** 5))
        ** (2 / 3)
    ) / (
        6 ** (2 / 3)
        * (
            m1_MSUN ** (15 / 2)
            * Mc_MSUN ** 5
            * (9 * m1_MSUN ** (5 / 2) + jnp.sqrt(81 * m1_MSUN ** 5 - 12 * Mc_MSUN ** 5))
        )
        ** (1 / 3)
    )


def unpack_2d(x: Array, dd_d: DynamicDress) -> DynamicDress:
    rho_6T, Mc_MSUN = x
    rho_6 = get_r6(rho_6T)
    M_chirp = Mc_MSUN * MSUN
    q = get_q(M_chirp, rho_6)
    f_c = get_f_isco(get_m_1(M_chirp, q))
    return DynamicDress(
        GAMMA_S_PBH, rho_6, M_chirp, q, dd_d.Phi_c, dd_d.tT_c, dd_d.dL, f_c
    )


def ptform_et_2d_1yr(u: Array, dd_d: DynamicDress) -> Array:
    assert u.shape == (2,)
    central = jnp.array([get_r6T(dd_d.rho_6), dd_d.M_chirp / MSUN])
    delta = jnp.array([1e-3, 4.8e-7]) / 20
    v_low = central - delta * jnp.array([2.0, 1.0])
    v_high = central + delta * jnp.array([3.0, 1.0])
    return v_low + (v_high - v_low) * u


def ptform_et_2d_1month(u: Array, dd_d: DynamicDress) -> Array:
    assert u.shape == (2,)
    central = jnp.array([get_r6T(dd_d.rho_6), dd_d.M_chirp / MSUN])
    delta = jnp.array([1e-3, 4.8e-7]) / 20
    v_low = central - delta * jnp.array([2.0, 1.0])
    v_high = central + delta * jnp.array([3.0, 1.0])
    return v_low + (v_high - v_low) * u


def ptform_et_2d_1week(u: Array, dd_d: DynamicDress) -> Array:
    assert u.shape == (2,)
    M_chirp_MSUN = dd_d.M_chirp / MSUN
    v_low = jnp.array([0.001, M_chirp_MSUN - 2e-7])
    v_high = jnp.array([0.002, M_chirp_MSUN + 2e-8])
    return v_low + (v_high - v_low) * u


def ptform_et_2d_1day(u: Array, dd_d: DynamicDress) -> Array:
    assert u.shape == (2,)
    M_chirp_MSUN = dd_d.M_chirp / MSUN
    v_low = jnp.array([0.0005, M_chirp_MSUN - 5e-7])
    v_high = jnp.array([0.005, M_chirp_MSUN + 7e-8])
    return v_low + (v_high - v_low) * u


def ptform_ce_2d_1yr(u: Array, dd_d: DynamicDress) -> Array:
    assert u.shape == (2,)
    central = jnp.array([get_r6T(dd_d.rho_6), dd_d.M_chirp / MSUN])
    delta = jnp.array([1e-3, 4.8e-7]) / 15
    v_low = central - delta * jnp.array([2.0, 1.0])
    v_high = central + delta * jnp.array([3.0, 1.0])
    return v_low + (v_high - v_low) * u


def ptform_ce_2d_1month(u: Array, dd_d: DynamicDress) -> Array:
    assert u.shape == (2,)
    central = jnp.array([get_r6T(dd_d.rho_6), dd_d.M_chirp / MSUN])
    delta = jnp.array([1e-3, 4.8e-7]) / 15
    v_low = central - delta * jnp.array([2.0, 1.0])
    v_high = central + delta * jnp.array([3.0, 1.0])
    return v_low + (v_high - v_low) * u


def ptform_ce_2d_1week(u: Array, dd_d: DynamicDress) -> Array:
    assert u.shape == (2,)
    M_chirp_MSUN = dd_d.M_chirp / MSUN
    v_low = jnp.array([0.001, M_chirp_MSUN - 2e-7])
    v_high = jnp.array([0.002, M_chirp_MSUN + 2e-8])
    return v_low + (v_high - v_low) * u


def ptform_ce_2d_1day(u: Array, dd_d: DynamicDress) -> Array:
    assert u.shape == (2,)
    M_chirp_MSUN = dd_d.M_chirp / MSUN
    v_low = jnp.array([0.0001, M_chirp_MSUN - 5e-7])
    v_high = jnp.array([0.008, M_chirp_MSUN + 7e-8])
    return v_low + (v_high - v_low) * u


# def ptform_aLIGO_2d_1yr(u: Array, dd_d: DynamicDress) -> Array:
#     raise NotImplementedError()


def plot_2d_results(
    results,
    dd_d: DynamicDress,
    title: str,
    fig_path: Optional[str],
    quantiles_2d: list[float] = [0.6827, 0.9545, 0.9973],
) -> Tuple[Figure, np.ndarray]:
    labels = (r"$\rho_6$ [$10^{16}$ M$_\odot$/pc$^3$]", r"$\mathcal{M}$ [M$_\odot$]")
    truths = (dd_d.rho_6 / (1e16 * MSUN / PC ** 3), dd_d.M_chirp / MSUN)
    fig, axes = dyplot.cornerplot(
        results,
        labels=labels,
        quantiles_2d=quantiles_2d,
        smooth=0.015,
        truths=truths,
        show_titles=True,
        title_kwargs={"fontsize": 12},
        title_fmt=".4f",
    )
    fig.suptitle(title)
    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path)

    return fig, axes


def run_ns_2d(
    dd_d: DynamicDress,
    t_obs: float,
    fs: Array,
    S_n: Callable[[Array], Array],
    detector: str,
    suffix: str = "-test",
) -> Tuple[Figure, np.ndarray]:
    if detector == "et":  # done
        if t_obs == 1 * YR:
            ptform = lambda u: ptform_et_2d_1yr(u, dd_d)
        elif t_obs == 1 * DAY:
            ptform = lambda u: ptform_et_2d_1day(u, dd_d)
        elif t_obs == 1 * WEEK:
            ptform = lambda u: ptform_et_2d_1week(u, dd_d)
        elif t_obs == 1 * MONTH:
            ptform = lambda u: ptform_et_2d_1month(u, dd_d)
        else:
            raise ValueError("no prior for the specified t_obs")
        title = "Einstein Telescope, "
    elif detector == "ce":  # done
        if t_obs == 1 * YR:
            ptform = lambda u: ptform_ce_2d_1yr(u, dd_d)
        elif t_obs == 1 * DAY:
            ptform = lambda u: ptform_ce_2d_1day(u, dd_d)
        elif t_obs == 1 * WEEK:
            ptform = lambda u: ptform_ce_2d_1week(u, dd_d)
        elif t_obs == 1 * MONTH:
            ptform = lambda u: ptform_ce_2d_1month(u, dd_d)
        else:
            raise ValueError("no prior for the specified t_obs")
        title = "Cosmic Explorer, "
    elif detector == "aLIGO":  # not started
        ptform = lambda u: ptform_aLIGO_2d_1yr(u, dd_d)
        title = "aLIGO, "
    else:
        raise ValueError("invalid detector")

    if t_obs >= HOUR and t_obs < DAY:
        title += f"{t_obs / HOUR:g} hours"
        timestr = f"{t_obs / HOUR:g}hr"
    elif t_obs >= DAY and t_obs < WEEK:
        title += f"{t_obs / DAY:g} days"
        timestr = f"{t_obs / DAY:g}day"
    elif t_obs >= WEEK and t_obs < MONTH:
        title += f"{t_obs / WEEK:g} weeks"
        timestr = f"{t_obs / WEEK:g}week"
    elif t_obs >= MONTH and t_obs < YR:
        title += f"{t_obs / MONTH:g} months"
        timestr = f"{t_obs / MONTH:g}month"
    elif t_obs >= YR:
        title += f"{t_obs / YR:g} years"
        timestr = f"{t_obs / YR:g}yr"
    else:
        title += f"{t_obs:g} seconds"
        timestr = f"{t_obs:g}s"

    pad_low, pad_high = get_match_pads(fs)

    @jax.jit
    def get_ll_fft(x):
        dd_h = unpack_2d(x, dd_d)
        return loglikelihood_fft(dd_h, dd_d, fs, pad_low, pad_high, S_n)

    print("setup complete, starting sampler")
    sampler = dynesty.NestedSampler(get_ll_fft, ptform, 2, nlive=2000)
    sampler.run_nested()
    results = sampler.results
    print("sampling complete")

    base_path = f"ns-{detector}-{timestr}{suffix}-2d"
    results_path = f"ns/{base_path}.pkl"
    fig_path = f"figures/{base_path}.pdf"

    with open(results_path, "wb") as output:
        pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)
    print(f"sampling complete, saved results to {results_path}")

    fig, axes = plot_2d_results(results, dd_d, title, fig_path)
    print(f"plotting complete, saved figure to {fig_path}")

    return fig, axes


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
@click.option(
    "-s", "--suffix", default="-test", help="suffix for saving plots and results"
)
def run(detector: str, t_obs: float, t_obs_units: str, n_f: int, suffix: str):
    # Set noise and other globals based on detector
    if detector == "et":
        from pydd.noise import S_n_et as S_n, f_range_et as f_range_n
    elif detector == "ce":
        from pydd.noise import S_n_ce as S_n, f_range_ce as f_range_n
    elif detector == "aLIGO":
        from pydd.noise import S_n_aLIGO as S_n, f_range_aLIGO as f_range_n
    elif detector == "LISA":
        from pydd.noise import S_n_LISA as S_n, f_range_LISA as f_range_n
    else:
        raise ValueError("invalid detector")

    snr_thresh = 15.0 if detector == "LISA" else 12.0

    if t_obs_units not in ("SECOND", "HOUR", "DAY", "WEEK", "MONTH", "YR"):
        raise ValueError("invalid t_obs units")

    print(f"running analysis for {detector} with t_obs = {t_obs} {t_obs_units}")
    t_obs = t_obs * eval(t_obs_units)
    dd_d, f_range_d = get_target_pbh_dynamicdress(
        1 * MSUN, 1e-3 * MSUN, t_obs, snr_thresh, S_n, f_range_n
    )
    print("rho_6T:", dd_d.rho_6 / (1e16 * MSUN / PC**3))
    fs = jnp.linspace(*f_range_d, n_f)
    run_ns_2d(dd_d, t_obs, fs, S_n, detector, suffix)


if __name__ == "__main__":
    run()  # type: ignore
