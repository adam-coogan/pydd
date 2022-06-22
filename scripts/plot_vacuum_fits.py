import os

import click
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import numpy as np

from pydd.binary import MSUN, PC
from utils import RHO_6_ASTRO, RHO_6_PBH, rho_6_to_rho6T


"""
Plots various measures of the discoverability of dark dresses.

Requires `vacuum_fits/vacuum_fits.npz`, which is produced by `calc_vacuum_fits.py`.

Produces `../figures/discoverability-lisa.pdf`.
"""


def load_bayes_factors():
    rho_6T_bfs = jnp.array([0.0008, 0.001, 0.002, 0.003, 0.004, 0.01])
    gamma_s_bfs = jnp.array([2.25, 2.3, 2.4, 2.5])
    bfs = np.zeros([len(rho_6T_bfs), len(gamma_s_bfs)])

    for i, rho_6T in enumerate(rho_6T_bfs):
        for j, gamma_s in enumerate(gamma_s_bfs):
            path = os.path.join(
                "ns", f"rho_6T={rho_6T:g}_gamma_s={gamma_s:g}-bayes.npz"
            )
            bfs[i, j] = jnp.load(path)["bayes_fact"]

    bfs = jnp.array(bfs)

    return rho_6T_bfs, gamma_s_bfs, bfs


@click.command()
@click.option("--path", default="vacuum_fits.npz")
@click.option("--rho_s/--no-rho_s", default=False, help="plot rho_s contours")
def run(path, rho_s):
    # Load
    f = jnp.load(os.path.join("vacuum_fits", path))
    rho_6Ts = rho_6_to_rho6T(f["rho_6s"])

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4))

    ax = axes[0, 0]
    log_levels = np.linspace(-0.5, 4.5, 11)
    cs = ax.contourf(
        rho_6Ts,
        f["gamma_ss"],
        f["dN_naives"].T,
        levels=10 ** log_levels,
        cmap="GnBu",
        norm=LogNorm(),
    )
    plt.colorbar(
        cs,
        ax=ax,
        fraction=0.055,
        pad=0.04,
        label=r"$N_\mathrm{V} - N_\mathrm{D}$",
        ticks=10 ** log_levels[1::2],
    ).ax.set_yticklabels([r"$10^{%d}$" % e for e in log_levels[1::2].astype(int)])

    ax = axes[0, 1]
    levels = jnp.array([-5, -4, -3, -2, -1, 1])
    cs = ax.contourf(
        rho_6Ts,
        f["gamma_ss"],
        f["dNs"].T,
        levels=levels,
        cmap="YlOrBr_r",
        norm=Normalize(-5.5, 1),
    )
    plt.colorbar(
        cs,
        ax=ax,
        fraction=0.055,
        pad=0.04,
        label=r"$N_{\hat{\mathrm{V}}} - N_\mathrm{D}$",
    )  # .ax.set_yticklabels(levels.astype(int))
    ax.fill_between([2e-3, 9e-3], 2.42, 2.49, color="w")
    ax.text(2e-1, 2.48, r"$< -5$", ha="center", va="center", backgroundcolor="w")

    ax = axes[1, 0]
    # Clean up
    dM_chirps = np.zeros(f["M_chirp_MSUN_bests"].shape)
    dM_chirps = np.array(f["M_chirp_MSUN_bests"] - f["M_chirp_MSUN"])
    dM_chirps[dM_chirps > 1e-2] = np.nan
    dM_chirps[np.isnan(dM_chirps)] = np.nan
    for i, rho_6T in enumerate(rho_6Ts):
        if rho_6T > 5e-2:
            dM_chirps[i] = np.nan
    M_chirp_errs = np.array(f["M_chirp_MSUN_best_errs"])
    M_chirp_errs[M_chirp_errs > 10 ** (-5.2)] = np.nan
    cs = ax.contourf(
        rho_6Ts,
        f["gamma_ss"],
        (dM_chirps / M_chirp_errs).T,
        levels=jnp.linspace(0, 100, 6, dtype=int),
        cmap="YlOrRd",
        norm=Normalize(0, 120),
    )
    plt.colorbar(
        cs,
        ax=ax,
        fraction=0.055,
        pad=0.04,
        label=r"$(\mathcal{M}_{\hat{\mathrm{V}}} - \mathcal{M}) / \Delta\mathcal{M}_{\hat{\mathrm{V}}}$",
    )
    ax.text(2e-1, 2.48, r"$> 100$", ha="center", va="center", backgroundcolor="w")

    ax = axes[1, 1]
    cs = ax.contourf(
        rho_6Ts,
        f["gamma_ss"],
        jnp.clip((f["snrs"] - jnp.sqrt(f["matches"])) / f["snrs"], a_min=0).T * 100,
        levels=jnp.linspace(0, 10, 6),
        cmap="OrRd",
        norm=Normalize(0, 11),
    )
    plt.colorbar(
        cs,
        ax=ax,
        fraction=0.055,
        pad=0.04,
        label=r"% SNR loss with $\hat{\mathrm{V}}$",
    )
    ax.text(2e-1, 2.48, r"$> 10\%$", ha="center", va="center", backgroundcolor="w")

    rho_6T_bfs, gamma_s_bfs, bfs = load_bayes_factors()

    for ax in axes.flatten():  # [1, :]:
        ax.set_xlabel(r"$\rho_6$ [$10^{16}$ $\mathrm{M}_\odot \, \mathrm{pc}^{-3}$]")
    for ax in axes.flatten():  # [:, 0]:
        ax.set_ylabel(r"$\gamma_s$")

    for ax in axes.flatten():
        ax.set_xscale("log")
        ax.set_xlim(rho_6_to_rho6T(f["rho_6s"][0]), 0.7)
        # Benchmarks
        ax.scatter(
            [rho_6_to_rho6T(RHO_6_ASTRO)],
            [7 / 3],
            marker="*",
            color="black",
            s=100,
        )
        ax.scatter(
            [rho_6_to_rho6T(RHO_6_PBH)], [9 / 4], marker=".", color="black", s=200
        )

        # Bayes factors
        cs = ax.contour(
            rho_6T_bfs,
            gamma_s_bfs,
            jnp.log10(bfs).T,
            levels=[-1, 2, 5],
            colors="k",
            linestyles=["-"],
            linewidths=[1],
        )
        ax.clabel(
            cs,
            inline=True,
            fontsize=8,
            fmt=lambda f: r"$\mathrm{BF} = {%g}$" % 10 ** f,
            manual=[(2e-3, 2.28), (2.5e-3, 2.36), (3e-3, 2.46)],
        )

        # rho_s contours
        if rho_s:
            cs = ax.contour(
                rho_6Ts,
                f["gamma_ss"],
                jnp.log10(f["rho_ss"].T / (MSUN / PC ** 3)),
                levels=[-32, -16, -8, -4, -2, 0, 2, 4],
                colors=["k"],
                linestyles=["--"],
                linewidths=[0.5],
            )
            ax.clabel(cs, inline=True, fontsize=8, fmt=r"$10^{%i}$")

    fig.tight_layout()
    fig.savefig(f"../figures/discoverability-lisa.pdf")


if __name__ == "__main__":
    run()
