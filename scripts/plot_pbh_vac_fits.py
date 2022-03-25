import click

# from typing import Callable, Dict, Tuple

import jax.numpy as jnp
from matplotlib import ticker
import matplotlib.pyplot as plt

from pydd.binary import MSUN, YR, MONTH, WEEK, DAY, HOUR

# Contour levels
ALIGO_LEVELS_1YR = {
    "M_chirp_err": jnp.linspace(0, 0.002, 11),
    "snr_loss_frac": jnp.linspace(0, 40, 9),
    "dL": jnp.linspace(0, 300, 11),
    "t_in_band": jnp.geomspace(1e-2, 1e3, 5),
}
CE_LEVELS_1YR = {
    "M_chirp_err": jnp.linspace(0, 0.005, 11),
    "snr_loss_frac": jnp.linspace(0, 80, 9),
    "dL": jnp.linspace(0, 20e3, 11),
    "t_in_band": jnp.geomspace(1e-2, 1e3, 5),
}
ET_LEVELS_1YR = {
    "M_chirp_err": jnp.linspace(0, 0.005, 11),
    "snr_loss_frac": jnp.linspace(0, 80, 9),
    "dL": jnp.linspace(0, 7e3, 9),
    "t_in_band": jnp.geomspace(1e0, 1e3, 5),
}


def plot(m_1_g, m_2_g, results, detector, t_obs, levels, fig_path):
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))

    ax = axes[0, 0]
    cs = ax.contourf(
        m_1_g / MSUN,
        m_2_g / MSUN,
        results["N_v_0_in_band"] - results["N_in_band"],
        locator=ticker.LogLocator(),
        cmap="GnBu",
    )
    plt.colorbar(cs, ax=ax, label=r"$\log_{10} (N_\mathrm{V} - N_\mathrm{D})$")

    ax = axes[0, 1]
    cs = ax.contourf(
        m_1_g / MSUN,
        m_2_g / MSUN,
        results["N_v_in_band"] - results["N_in_band"],
        locator=ticker.LogLocator(),
        cmap="YlOrBr_r",
    )
    plt.colorbar(cs, ax=ax, label=r"$\log_{10} (N_\hat{\mathrm{V}} - N_\mathrm{D})$")
    ax.contour(m_1_g / MSUN, m_2_g / MSUN, results["dN"], [1], colors=["r"])

    ax = axes[1, 0]
    cs = ax.contourf(
        m_1_g / MSUN,
        m_2_g / MSUN,
        (results["M_chirp_MSUN_v"] - results["M_chirp_MSUN"])
        / results["M_chirp_MSUN"]
        * 100,
        levels=levels["M_chirp_err"],
        cmap="YlOrRd",
    )
    plt.colorbar(cs, ax=ax, label=r"% error in $\mathcal{M}$")

    ax = axes[1, 1]
    cs = ax.contourf(
        m_1_g / MSUN,
        m_2_g / MSUN,
        results["snr_loss_frac"] * 100,
        levels=levels["snr_loss_frac"],
        cmap="OrRd",
    )
    plt.colorbar(cs, ax=ax, label="% SNR loss")

    for ax in axes.flatten():
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$m_1$ [M$_\odot$]")
        ax.set_ylabel(r"$m_2$ [M$_\odot$]")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if detector == "et":
        title = "Einstein Telescope, "
    elif detector == "ce":
        title = "Cosmic Explorer, "
    elif detector == "aLIGO":
        title = "aLIGO, "
    else:
        raise ValueError("invalid detector")

    # TODO: move to run and simplify
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

    fig.suptitle(title)
    plt.savefig(fig_path)


@click.command()
@click.option("-d", "--detector", type=str, help="detector name")
@click.option("-t", "--t_obs", type=float, help="observing time")
@click.option(
    "-u",
    "--t-obs-units",
    type=str,
    help="units of t_obs ('SECOND', 'HOUR', 'DAY', 'WEEK', 'MONTH' or 'YR')",
)
@click.option(
    "-s", "--suffix", default="-test", help="suffix for saving plots and results"
)
def run(
    detector: str,
    t_obs: float,
    t_obs_units: str,
    suffix: str,
):
    print(
        f"making discoverability plot for {detector} with t_obs = {t_obs} {t_obs_units}"
    )

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

    base_path = f"vac-fit-{detector}-{timestr}{suffix}"
    results_path = f"vacuum_fits/{base_path}.npz"
    fig_path = f"figures/{base_path}.pdf"
    results = {k: jnp.array(v) for k, v in dict(jnp.load(results_path)).items()}
    m_1_g = results.pop("m_1_mg")[0]
    m_2_g = results.pop("m_2_mg")[:, 0]

    levels = eval(f"{detector.upper()}_LEVELS_{t_obs:g}{t_obs_units}")

    plot(m_1_g, m_2_g, results, detector, t_obs, levels, fig_path)
    print(f"saved plot to {fig_path}")
