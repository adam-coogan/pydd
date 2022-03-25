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
)
from pydd.utils import get_target_pbh_dynamicdress

Array = jnp.ndarray


def unpack_4d(x: Array, dd_d: DynamicDress) -> DynamicDress:
    gamma_s, rho_6T, Mc_MSUN, log10_q = x
    rho_6 = rho_6T * 1e16 * MSUN / PC ** 3
    M_chirp = Mc_MSUN * MSUN
    q = 10 ** log10_q

    m_1 = get_m_1(M_chirp, q)
    f_c = get_f_isco(m_1)
    return DynamicDress(gamma_s, rho_6, M_chirp, q, dd_d.Phi_c, dd_d.tT_c, dd_d.dL, f_c)


def ptform_et_4d_1day(u: Array, dd_d: DynamicDress) -> Array:
    assert u.shape == (4,)
    M_chirp_MSUN = dd_d.M_chirp / MSUN
    v_low = jnp.array([2.05, 0.0, M_chirp_MSUN - 1e-6, -4.5])
    v_high = jnp.array([2.5, 0.04, M_chirp_MSUN + 7e-7, -2.5])
    return v_low + (v_high - v_low) * u


def ptform_et_4d_1week(u: Array, dd_d: DynamicDress) -> Array:
    assert u.shape == (4,)
    M_chirp_MSUN = dd_d.M_chirp / MSUN
    v_low = jnp.array([2.05, 0.0, M_chirp_MSUN - 1.5e-6, -4.0])
    v_high = jnp.array([2.5, 0.02, M_chirp_MSUN + 7e-7, -2.5])
    return v_low + (v_high - v_low) * u


def ptform_et_4d_1month(u: Array, dd_d: DynamicDress) -> Array:
    assert u.shape == (4,)
    M_chirp_MSUN = dd_d.M_chirp / MSUN
    v_low = jnp.array([2.05, 0.0001, M_chirp_MSUN - 1.5e-6, -4.0])
    v_high = jnp.array([2.4, 0.003, M_chirp_MSUN + 7e-7, -2.5])
    return v_low + (v_high - v_low) * u


def ptform_et_4d_1yr(u: Array, dd_d: DynamicDress) -> Array:
    assert u.shape == (4,)
    central = jnp.array(
        [
            dd_d.gamma_s,
            dd_d.rho_6 / (1e16 * MSUN / PC ** 3),
            dd_d.M_chirp / MSUN,
            jnp.log10(dd_d.q),
        ]
    )
    delta = jnp.array([6e-2, 1e-3, 4.8e-7, 5e-1]) / 5
    v_low = central - delta * jnp.array([3, 2, 1, 3.0])
    v_high = central + delta * jnp.array([3, 3, 1, 5.0])
    return v_low + (v_high - v_low) * u


def ptform_ce_4d_1day(u: Array, dd_d: DynamicDress) -> Array:
    assert u.shape == (4,)
    M_chirp_MSUN = dd_d.M_chirp / MSUN
    v_low = jnp.array([2.05, 0.0, M_chirp_MSUN - 7e-7, -4.0])
    v_high = jnp.array([2.5, 0.04, M_chirp_MSUN + 7e-7, -2.5])
    return v_low + (v_high - v_low) * u


def ptform_ce_4d_1week(u: Array, dd_d: DynamicDress) -> Array:
    assert u.shape == (4,)
    M_chirp_MSUN = dd_d.M_chirp / MSUN
    v_low = jnp.array([2.05, 0.0, M_chirp_MSUN - 1e-6, -4.0])
    v_high = jnp.array([2.5, 0.02, M_chirp_MSUN + 1e-6, -2.5])
    return v_low + (v_high - v_low) * u


def ptform_ce_4d_1month(u: Array, dd_d: DynamicDress) -> Array:
    assert u.shape == (4,)
    M_chirp_MSUN = dd_d.M_chirp / MSUN
    v_low = jnp.array([2.05, 0.0001, M_chirp_MSUN - 1.5e-6, -4.0])
    v_high = jnp.array([2.4, 0.004, M_chirp_MSUN + 7e-7, -2.5])
    return v_low + (v_high - v_low) * u


def ptform_ce_4d_1yr(u: Array, dd_d: DynamicDress) -> Array:
    assert u.shape == (4,)
    M_chirp_MSUN = dd_d.M_chirp / MSUN
    v_low = jnp.array([2.05, 0.0004, M_chirp_MSUN - 1e-7, -4.0])
    v_high = jnp.array([2.4, 0.0045, M_chirp_MSUN + 2e-7, -2.5])
    return v_low + (v_high - v_low) * u


# def ptform_ce_4d_1yr(u: Array, dd_d: DynamicDress) -> Array:
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


def ptform_aLIGO_4d_1yr(u: Array, dd_d: DynamicDress) -> Array:
    assert u.shape == (4,)
    central = jnp.array(
        [
            dd_d.gamma_s,
            dd_d.rho_6 / (1e16 * MSUN / PC ** 3),
            dd_d.M_chirp / MSUN,
            jnp.log10(dd_d.q),
        ]
    )
    delta = jnp.array([1.2e-4, 5e-4, 1.6e-8, 8e-4])
    v_low = central - 20 * jnp.array([4, 3, 6, 4]) * delta
    v_high = central + 20 * jnp.array([4, 3, 6, 4]) * delta
    return v_low + (v_high - v_low) * u


def plot_4d_results(
    results,
    dd_d: DynamicDress,
    title: str,
    fig_path: Optional[str],
    quantiles_2d: list[float] = [0.6827, 0.9545, 0.9973],
) -> Tuple[Figure, np.ndarray]:
    labels = (
        r"$\gamma_s$",
        r"$\rho_6$ [$10^{16}$ M$_\odot$/pc$^3$]",
        r"$\mathcal{M}$ [M$_\odot$]",
        r"$\log_{10} q$",
    )
    truths = (
        dd_d.gamma_s,
        dd_d.rho_6 / (1e16 * MSUN / PC ** 3),
        dd_d.M_chirp / MSUN,
        jnp.log10(dd_d.q),
    )
    fig, axes = dyplot.cornerplot(
        results,
        labels=labels,
        quantiles_2d=quantiles_2d,
        smooth=0.015,
        truths=truths,
        show_titles=True,
        title_kwargs={"fontsize": 12},
        title_fmt=".4f",
        span=(1, 1, 1, 1),
    )
    fig.suptitle(title)
    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path)

    return fig, axes


def run_ns_4d(
    dd_d: DynamicDress,
    t_obs: float,
    fs: Array,
    S_n: Callable[[Array], Array],
    detector: str,
    suffix: str = "-test",
) -> Tuple[Figure, np.ndarray]:
    if detector == "et":  # done
        if t_obs == 1 * YR:
            ptform = lambda u: ptform_et_4d_1yr(u, dd_d)
        elif t_obs == 1 * DAY:
            ptform = lambda u: ptform_et_4d_1day(u, dd_d)
        elif t_obs == 1 * WEEK:
            ptform = lambda u: ptform_et_4d_1week(u, dd_d)
        elif t_obs == 1 * MONTH:
            ptform = lambda u: ptform_et_4d_1month(u, dd_d)
        else:
            raise ValueError("no prior for the specified t_obs")
        title = "Einstein Telescope, "
    elif detector == "ce":  # done
        if t_obs == 1 * YR:
            ptform = lambda u: ptform_ce_4d_1yr(u, dd_d)
        elif t_obs == 1 * DAY:
            ptform = lambda u: ptform_ce_4d_1day(u, dd_d)
        elif t_obs == 1 * WEEK:
            ptform = lambda u: ptform_ce_4d_1week(u, dd_d)
        elif t_obs == 1 * MONTH:
            ptform = lambda u: ptform_ce_4d_1month(u, dd_d)
        else:
            raise ValueError("no prior for the specified t_obs")
        title = "Cosmic Explorer, "
    elif detector == "aLIGO":  # not started
        if t_obs == 1 * YR:
            ptform = lambda u: ptform_aLIGO_4d_1yr(u, dd_d)
        # elif t_obs == 1 * DAY:
        #     ptform = lambda u: ptform_aLIGO_4d_1day(u, dd_d)
        # elif t_obs == 1 * WEEK:
        #     ptform = lambda u: ptform_aLIGO_4d_1week(u, dd_d)
        # elif t_obs == 1 * MONTH:
        #     ptform = lambda u: ptform_aLIGO_4d_1month(u, dd_d)
        else:
            raise ValueError("no prior for the specified t_obs")
        title = "aLIGO, "
    else:
        raise ValueError("invalid detector")

    # TODO: move to run and simplify
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
        dd_h = unpack_4d(x, dd_d)
        return loglikelihood_fft(dd_h, dd_d, fs, pad_low, pad_high, S_n)

    print("setup complete, starting sampler")
    # sampler = dynesty.DynamicNestedSampler(get_ll_fft, ptform, 4, nlive=2000)
    # sampler.run_nested(nlive_init=500, nlive_batch=100)
    sampler = dynesty.NestedSampler(get_ll_fft, ptform, 4, nlive=2000)
    sampler.run_nested()
    results = sampler.results
    print("sampling complete")

    base_path = f"ns-{detector}-{timestr}{suffix}-4d"
    results_path = f"ns/{base_path}.pkl"
    fig_path = f"figures/{base_path}.pdf"

    with open(results_path, "wb") as output:
        pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)
    print(f"sampling complete, saved results to {results_path}")

    fig, axes = plot_4d_results(results, dd_d, title, fig_path)
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
    print("rho_6T:", dd_d.rho_6 / (1e16 * MSUN / PC ** 3))
    fs = jnp.linspace(*f_range_d, n_f)
    run_ns_4d(dd_d, t_obs, fs, S_n, detector, suffix)


if __name__ == "__main__":
    run()  # type: ignore


# # %%
# labels = (
#     r"$\gamma_s$",
#     r"$\rho_6$ [$10^{16}$ M$_\odot$/pc$^3$]",
#     r"$\mathcal{M}$ [M$_\odot$]",
#     r"$\log_{10} q$",
# )
# # quantiles_2d = [1 - np.exp(-(x ** 2) / 2) for x in [1, 2, 3]]
# quantiles_2d = [0.6827, 0.9545, 0.9973]  # more typical for Bayesian analyses
# truths = (
#     DD_D.gamma_s,
#     DD_D.rho_6 / (1e16 * MSUN / PC ** 3),
#     DD_D.M_chirp / MSUN,
#     jnp.log10(DD_D.q),
# )
#
# # %%
# with open(f"ns/ns-ce-4d.pkl", "rb") as infile:
#     results = pickle.load(infile)
#
# fig, axes = dyplot.cornerplot(
#     results,
#     labels=labels,
#     quantiles_2d=quantiles_2d,
#     smooth=0.015,
#     truths=truths,
#     show_titles=True,
#     title_kwargs={"fontsize": 12},
#     title_fmt=".4f",
# )
# fig.suptitle("Cosmic Explorer")
# fig.tight_layout()
# fig.savefig(f"figures/ns-ce.pdf")
#
# # %%
# with open(f"ns/ns-et-4d.pkl", "rb") as infile:
#     results = pickle.load(infile)
#
# fig, axes = dyplot.cornerplot(
#     results,
#     labels=labels,
#     quantiles_2d=quantiles_2d,
#     smooth=0.015,
#     truths=truths,
#     show_titles=True,
#     title_kwargs={"fontsize": 12},
#     title_fmt=".4f",
# )
# fig.suptitle("Einstein Telescope")
# fig.tight_layout()
# fig.savefig(f"figures/ns-et.pdf")
#
# # %%
# with open(f"ns/ns-ce-et-target-4d.pkl", "rb") as infile:
#     results = pickle.load(infile)
#
# fig, axes = dyplot.cornerplot(
#     results,
#     labels=labels,
#     quantiles_2d=quantiles_2d,
#     smooth=0.015,
#     truths=truths,
#     show_titles=True,
#     title_kwargs={"fontsize": 12},
#     title_fmt=".4f",
# )
# fig.suptitle("Cosmic Explorer, Einstein Telescope target")
# fig.tight_layout()
# fig.savefig(f"figures/ns-ce-et-target.pdf")
#
# # %%
# with open(f"ns/ns-et-4d.pkl", "rb") as infile:
#     results_et = pickle.load(infile)
#
# with open(f"ns/ns-ce-et-target-4d.pkl", "rb") as infile:
#     results_ce_et_target = pickle.load(infile)
#
# # %%
# fig, axes = dyplot.cornerplot(
#     results_et,
#     labels=labels,
#     quantiles_2d=quantiles_2d,
#     smooth=0.015,
#     truths=truths,
#     show_titles=True,
#     title_kwargs={"fontsize": 12},
#     title_fmt=".4f",
# )
#
# # %%
# from scipy.ndimage import gaussian_filter as norm_kde
#
#
# def get_HDI_thresholds(x, cred_level=[0.68268, 0.95450, 0.99730]):
#     """
#     Taken from swyft.plot.plot
#     """
#     x = x.flatten()
#     x = np.sort(x)[::-1]  # Sort backwards
#     total_mass = x.sum()
#     enclosed_mass = np.cumsum(x)
#     idx = [np.argmax(enclosed_mass >= total_mass * f) for f in cred_level]
#     levels = np.array(x[idx])
#     return levels
#
#
# def my_corner(results, axes, color, bins_1d=50, bins_2d=40, smooth=0.015):
#     span = [0.999999426697 for i in range(2)]
#
#     samples = results["samples"].T
#     ndim = samples.shape[0]
#     weights = np.exp(results["logwt"] - results["logz"][-1])
#
#     for i, x in enumerate(samples):
#         ax = axes[i, i]
#         ax.hist(x, weights=weights, bins=bins_1d, histtype="step", color=color)
#
#         for j, y in enumerate(samples):
#             ax = axes[i, j]
#             if j == i:
#                 continue
#             if j > i:
#                 ax.axis("off")
#                 continue
#             counts, xbins, ybins = np.histogram2d(y, x, weights=weights, bins=bins_2d)
#             levels = sorted(get_HDI_thresholds(counts))
#             ax.contour(
#                 counts.T,
#                 extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()],
#                 levels=levels,
#                 colors=[color],
#             )
#
#
# # %%
# ndim = 4
# fig, axes = plt.subplots(ndim, ndim, figsize=(3 * ndim, 3 * ndim))
#
# my_corner(results_et, axes, "r")
# my_corner(results_ce_et_target, axes, "b", 200, 200)
#
# for i, ax in enumerate(axes[-1, :]):
#     ax.set_xlabel(labels[i])
#
# for j, ax in enumerate(axes[1:, 0], 1):
#     ax.set_ylabel(labels[j])
#
# ranges = [
#     (2.15, 2.3),
#     (0.0023, 0.0045),
#     (1.5845e-2 + 6e-7, 1.5845e-2 + 9.5e-7),
#     (-3.5, -2.6),
# ]
# for i in range(ndim):
#     for j in range(i, ndim):
#         axes[j, i].set_xlim(ranges[i])
#
# for i in range(1, ndim):
#     for j in range(0, i):
#         axes[i, j].set_ylim(ranges[i])
#
# fig.tight_layout()
