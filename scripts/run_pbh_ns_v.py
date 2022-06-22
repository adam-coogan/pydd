import os
import pickle
from typing import Callable, Optional, Tuple

import click
import dynesty
from dynesty import plotting as dyplot
from dynesty.results import Results
import jax
import jax.numpy as jnp
from matplotlib.figure import Figure
import numpy as np
from pydd.analysis import get_match_pads, loglikelihood_fft
from pydd.binary import (
    DAY,
    DynamicDress,
    VacuumBinary,
    HOUR,
    MONTH,
    MSUN,
    PC,
    WEEK,
    YR,
    get_f_isco,
    get_m_1,
)
from pydd.utils import get_target_pbh_dynamicdress

Array = jnp.ndarray


def unpack_v(x: Array, dd_d: DynamicDress) -> VacuumBinary:
    return VacuumBinary(x[0] * MSUN, dd_d.Phi_c, dd_d.tT_c, dd_d.dL, dd_d.f_c)


# Hand-crafted priors for (m1, m2) = (1, 1e-3)
def ptform_et_v_1day(
    u: Array,
    dd_d: DynamicDress,
    dMc_low: float = -1e-6,
    dMc_high: float = 7e-7,
) -> Array:
    raise NotImplemented()


def ptform_et_v_1week(
    u: Array,
    dd_d: DynamicDress,
    dMc_low: float = -1e-6,
    dMc_high: float = 2e-6,
) -> Array:
    assert u.shape == (1,)
    M_chirp_MSUN = dd_d.M_chirp / MSUN
    v_low = jnp.array([M_chirp_MSUN + dMc_low])
    v_high = jnp.array([M_chirp_MSUN + dMc_high])
    return v_low + (v_high - v_low) * u


def ptform_et_v_1month(
    u: Array,
    dd_d: DynamicDress,
    dMc_low: float = -1.5e-6,
    dMc_high: float = 7e-7,
) -> Array:
    raise NotImplemented()


# def ptform_et_v_1yr(u: Array, dd_d: DynamicDress) -> Array:
#     assert u.shape == (4,)
#     central = jnp.array(
#         [
#             dd_d.gamma_s,
#             dd_d.rho_6 / (1e16 * MSUN / PC**3),
#             dd_d.M_chirp / MSUN,
#             jnp.log10(dd_d.q),
#         ]
#     )
#     delta = jnp.array([6e-2, 1e-3, 4.8e-7, 5e-1]) / 5
#     v_low = central - delta * jnp.array([3, 2, 1, 3.0])
#     v_high = central + delta * jnp.array([3, 3, 1, 5.0])
#     return v_low + (v_high - v_low) * u


def ptform_ce_v_1day(
    u: Array,
    dd_d: DynamicDress,
    dMc_low: float = -7e-7,
    dMc_high: float = 7e-7,
) -> Array:
    raise NotImplemented()


def ptform_ce_v_1week(
    u: Array,
    dd_d: DynamicDress,
    dMc_low: float = -0.5e-6,
    dMc_high: float = 4e-6,
) -> Array:
    assert u.shape == (1,)
    M_chirp_MSUN = dd_d.M_chirp / MSUN
    v_low = jnp.array([M_chirp_MSUN + dMc_low])
    v_high = jnp.array([M_chirp_MSUN + dMc_high])
    return v_low + (v_high - v_low) * u


def ptform_ce_v_1month(
    u: Array,
    dd_d: DynamicDress,
    dMc_low: float = -1.5e-6,
    dMc_high: float = 7e-7,
) -> Array:
    raise NotImplemented()


def ptform_ce_v_1yr(
    u: Array,
    dd_d: DynamicDress,
    dMc_low: float = -1e-7,
    dMc_high: float = 2e-7,
) -> Array:
    raise NotImplemented()


# def ptform_ce_v_1yr(u: Array, dd_d: DynamicDress) -> Array:
#     assert u.shape == (4,)
#     central = jnp.array(
#         [
#             dd_d.gamma_s,
#             dd_d.rho_6 / (1e16 * MSUN / PC ** 3),
#             dd_d.M_chirp / MSUN,
#             jnp.log10(dd_d.q),
#         ]
#     )
#     delta = jnp.array([6e-2, 1e-3, 4.8e-7, 5e-1])
#     v_low = central - delta * jnp.array([3, 2, 1, 3.0])
#     v_high = central + delta * jnp.array([3, 3, 1, 1.0])
#     return v_low + (v_high - v_low) * u


# def ptform_aLIGO_v_1yr(u: Array, dd_d: DynamicDress) -> Array:
#     assert u.shape == (4,)
#     central = jnp.array(
#         [
#             dd_d.gamma_s,
#             dd_d.rho_6 / (1e16 * MSUN / PC**3),
#             dd_d.M_chirp / MSUN,
#             jnp.log10(dd_d.q),
#         ]
#     )
#     delta = jnp.array([1.2e-4, 5e-4, 1.6e-8, 8e-4])
#     v_low = central - 20 * jnp.array([4, 3, 6, 4]) * delta
#     v_high = central + 20 * jnp.array([4, 3, 6, 4]) * delta
#     return v_low + (v_high - v_low) * u


def plot_v_results(
    results,
    dd_d: DynamicDress,
    title: str,
    fig_path: Optional[str],
    quantiles_2d: list[float] = [0.6827, 0.9545, 0.9973],
) -> Tuple[Figure, np.ndarray]:
    labels = (r"$\mathcal{M}$ [M$_\odot$]",)
    truths = (dd_d.M_chirp / MSUN,)
    fig, axes = dyplot.cornerplot(
        results,
        labels=labels,
        quantiles_2d=quantiles_2d,
        smooth=0.015,
        truths=truths,
        show_titles=True,
        title_kwargs={"fontsize": 12},
        title_fmt=".4f",
        span=(1,),
    )
    fig.suptitle(title)
    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path)

    return fig, axes


def run_ns_v(
    dd_d: DynamicDress,
    t_obs: float,
    fs: Array,
    S_n: Callable[[Array], Array],
    detector: str,
    base_path: str,
    dMc_low: float,
    dMc_high: float,
) -> Tuple[Figure, np.ndarray]:
    """
    Runs nested sampling for a vacuum system given a signal system dd_s. Saves
    results and posterior plots.
    """
    if detector == "et":  # done
        if t_obs == 1 * DAY:
            ptform = lambda u: ptform_et_v_1day(u, dd_d, dMc_low, dMc_high)
        elif t_obs == 1 * WEEK:
            ptform = lambda u: ptform_et_v_1week(u, dd_d, dMc_low, dMc_high)
        elif t_obs == 1 * MONTH:
            ptform = lambda u: ptform_et_v_1month(u, dd_d, dMc_low, dMc_high)
        else:
            raise ValueError("no prior for the specified t_obs")
        title = "Einstein Telescope, "
    elif detector == "ce":  # done
        if t_obs == 1 * YR:
            ptform = lambda u: ptform_ce_v_1yr(u, dd_d, dMc_low, dMc_high)
        elif t_obs == 1 * DAY:
            ptform = lambda u: ptform_ce_v_1day(u, dd_d, dMc_low, dMc_high)
        elif t_obs == 1 * WEEK:
            ptform = lambda u: ptform_ce_v_1week(u, dd_d, dMc_low, dMc_high)
        elif t_obs == 1 * MONTH:
            ptform = lambda u: ptform_ce_v_1month(u, dd_d, dMc_low, dMc_high)
        else:
            raise ValueError("no prior for the specified t_obs")
        title = "Cosmic Explorer, "
    else:
        raise ValueError("invalid detector")

    if t_obs >= HOUR and t_obs < DAY:
        title += f"{t_obs / HOUR:g} hours"
    elif t_obs >= DAY and t_obs < WEEK:
        title += f"{t_obs / DAY:g} days"
    elif t_obs >= WEEK and t_obs < MONTH:
        title += f"{t_obs / WEEK:g} weeks"
    elif t_obs >= MONTH and t_obs < YR:
        title += f"{t_obs / MONTH:g} months"
    elif t_obs >= YR:
        title += f"{t_obs / YR:g} years"
    else:
        title += f"{t_obs:g} seconds"

    pad_low, pad_high = get_match_pads(fs)

    @jax.jit
    def get_ll_fft(x):
        vb = unpack_v(x, dd_d)
        return loglikelihood_fft(vb, dd_d, fs, pad_low, pad_high, S_n)

    print("setup complete, starting sampler")
    sampler = dynesty.NestedSampler(get_ll_fft, ptform, 1, nlive=1000)
    sampler.run_nested()
    results = sampler.results
    print("sampling complete")

    results_path = f"ns/{base_path}.pkl"
    fig_path = f"figures/{base_path}.pdf"

    with open(results_path, "wb") as output:
        pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)
    print(f"sampling complete, saved results to {results_path}")

    fig, axes = plot_v_results(results, dd_d, title, fig_path)
    print(f"plotting complete, saved figure to {fig_path}")

    return fig, axes


@click.command()
@click.option("-d", "--detector", type=str, help="detector name")
@click.option("-t", "--t_obs", default=1.0, help="observing time")
@click.option(
    "-u",
    "--t-obs-units",
    default="WEEK",
    help="units of t_obs ('SECOND', 'HOUR', 'DAY', 'WEEK', 'MONTH' or 'YR')",
)
@click.option("-n", "--n-f", default=10_000, help="number of frequency points")
@click.option(
    "-s", "--suffix", default="-test", help="suffix for saving plots and results"
)
@click.option("-m1", default=1.0)
@click.option("-m2", default=1e-3)
@click.option("--dmc-low", default=-1.5e-6)
@click.option("--dmc-high", default=7e-7)
@click.option("--savedir", default="")
def run(
    detector: str,
    t_obs: float,
    t_obs_units: str,
    n_f: int,
    suffix: str,
    m1: float,
    m2: float,
    dmc_low: float,
    dmc_high: float,
    savedir: str,
):
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

    t_obs = t_obs * eval(t_obs_units)

    # Get path for saving results
    if t_obs >= HOUR and t_obs < DAY:
        timestr = f"{t_obs / HOUR:g}hr"
    elif t_obs >= DAY and t_obs < WEEK:
        timestr = f"{t_obs / DAY:g}day"
    elif t_obs >= WEEK and t_obs < MONTH:
        timestr = f"{t_obs / WEEK:g}week"
    elif t_obs >= MONTH and t_obs < YR:
        timestr = f"{t_obs / MONTH:g}month"
    elif t_obs >= YR:
        timestr = f"{t_obs / YR:g}yr"
    else:
        timestr = f"{t_obs:g}s"
    base_path = os.path.join(savedir, f"ns-m1={m1}-m2={m2}-{detector}-{timestr}{suffix}-v")

    print(f"running analysis for {detector} with t_obs = {t_obs} {t_obs_units}")
    print(f"base_path = {base_path}")
    dd_d, f_range_d = get_target_pbh_dynamicdress(
        m1 * MSUN, m2 * MSUN, t_obs, snr_thresh, S_n, f_range_n
    )

    print("rho_6T:", dd_d.rho_6 / (1e16 * MSUN / PC**3))
    fs = jnp.linspace(*f_range_d, n_f)
    run_ns_v(dd_d, t_obs, fs, S_n, detector, base_path, dmc_low, dmc_high)


if __name__ == "__main__":
    run()  # type: ignore
