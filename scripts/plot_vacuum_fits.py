import os

import click
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pydd.binary import MSUN, PC
from utils import RHO_6_ASTRO, RHO_6_PBH, rho_6_to_rho6T


def load_bayes_factors():
    rho_6T_bfs = jnp.array([0.001, 0.002, 0.004, 0.01])
    gamma_s_bfs = jnp.array([2.25, 2.3, 2.4, 2.5])
    bfs = np.zeros([len(rho_6T_bfs), len(gamma_s_bfs)])

    for i, rho_6T in enumerate(rho_6T_bfs):
        for j, gamma_s in enumerate(gamma_s_bfs):
            path = os.path.join(
                "ns", f"rho_6T={rho_6T:g}_gamma_s={gamma_s:g}_test-bayes.npz"
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
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.5))

    ax = axes[0, 0]
    levels = jnp.geomspace(1, 31625, 10, dtype=int)
    cs = ax.contourf(
        rho_6Ts,
        f["gamma_ss"],
        jnp.log10(f["dN_naives"]).T,
        cmap="viridis",
        levels=jnp.log10(levels),
    )
    plt.colorbar(
        cs,
        ax=ax,
        fraction=0.055,
        pad=0.04,
        label=r"$N_\mathrm{no~DM} - N_\mathrm{D}$",
        ticks=jnp.log10(levels)[::2],
    ).ax.set_yticklabels(levels[::2])

    ax = axes[0, 1]
    levels = jnp.array([-5, -4, -3, -2, -1, 0.1])
    cs = ax.contourf(rho_6Ts, f["gamma_ss"], f["dNs"].T, cmap="viridis", levels=levels)
    plt.colorbar(
        cs, ax=ax, fraction=0.055, pad=0.04, label=r"$N_\mathrm{V} - N_\mathrm{D}$"
    ).ax.set_yticklabels(levels.astype(int))
    ax.fill_between([2e-3, 9e-3], 2.42, 2.49, color="w")
    ax.text(2e-1, 2.48, r"$< -5$", ha="center", va="center", backgroundcolor="w")

    ax = axes[1, 0]
    # Clean up
    dM_chirps = np.zeros(f["M_chirp_MSUN_bests"].shape)
    dM_chirps = np.array(f["M_chirp_MSUN_bests"] - f["M_chirp_MSUN"])
    # dM_chirps[jnp.where((dM_chirps > 1e-2) | jnp.isnan(dM_chirps))] = np.nan
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
        levels=jnp.linspace(0, 100, 6),
        cmap="viridis",
    )
    plt.colorbar(
        cs,
        ax=ax,
        fraction=0.055,
        pad=0.04,
        label=r"$(\mathcal{M}_V - \mathcal{M}) / \Delta\mathcal{M}_V$",
    )
    ax.text(2e-1, 2.48, r"$> 100$", ha="center", va="center", backgroundcolor="w")

    ax = axes[1, 1]
    cs = ax.contourf(
        rho_6Ts,
        f["gamma_ss"],
        jnp.clip((f["snrs"] - jnp.sqrt(f["matches"])) / f["snrs"], a_min=0).T * 100,
        levels=jnp.linspace(0, 10, 11),
        cmap="viridis",
    )
    plt.colorbar(
        cs,
        ax=ax,
        fraction=0.055,
        pad=0.04,
        label="% SNR loss with V",
    )
    ax.text(2e-1, 2.48, r"$> 10\%$", ha="center", va="center", backgroundcolor="w")

    rho_6T_bfs, gamma_s_bfs, bfs = load_bayes_factors()

    for ax in axes.flatten():
        ax.set_xscale("log")
        ax.set_xlabel(r"$\rho_6$ [$10^{16}$ $\mathrm{M}_\odot \, \mathrm{pc}^{-3}$]")
        ax.set_ylabel(r"$\gamma_s$")
        ax.set_xlim(rho_6_to_rho6T(f["rho_6s"][0]), 0.7)
        # Benchmarks
        ax.scatter(
            [RHO_6_ASTRO],
            [7 / 3],
            marker="*",
            color="black",
            s=100,
        )
        ax.scatter([RHO_6_PBH], [9 / 4], marker=".", color="black", s=200)

        # Bayes factors
        cs = ax.contour(
            rho_6T_bfs,
            gamma_s_bfs,
            jnp.log10(bfs).T,
            levels=[0, 1, 2],
            colors="red",
            linestyles=["-"],
            linewidths=[1],
        )
        ax.clabel(
            cs,
            inline=True,
            fontsize=8,
            fmt=lambda f: r"$\operatorname{BF} = %g$" % 10 ** f,
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
    fig.savefig(f"figures/{os.path.splitext(path)[0]}.pdf")


if __name__ == "__main__":
    run()
