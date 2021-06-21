from math import pi
import os

import click
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from tqdm.auto import tqdm

from pydd.analysis import calculate_SNR, calculate_match_unnormd_fft
from pydd.binary import MSUN, PC, Phi_to_c, VacuumBinary, get_M_chirp, get_rho_s
from utils import (
    get_loglikelihood_v,
    rho_6T_to_rho6,
    rho_6_to_rho6T,
    setup_astro,
    setup_pbh,
    setup_system,
)


@click.command()
@click.option("--path", default="vacuum_fits.npz")
@click.option("--rho_s/--no-rho_s", default=False)
def run(path, rho_s):
    # Load
    f = jnp.load(path)
    rho_6Ts = rho_6_to_rho6T(f["RHO_6S"])

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.5))

    ax = axes[0, 0]
    levels = jnp.geomspace(1, 31625, 10, dtype=int)
    cs = ax.contourf(
        rho_6Ts,
        f["GAMMA_SS"],
        jnp.log10(f["dN_naives"]).T,
        cmap="plasma",
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
    cs = ax.contourf(rho_6Ts, f["GAMMA_SS"], f["dNs"].T, levels=levels)
    plt.colorbar(
        cs, ax=ax, fraction=0.055, pad=0.04, label=r"$N_\mathrm{V} - N_\mathrm{D}$"
    ).ax.set_yticklabels(levels.astype(int))
    ax.fill_between([2e-3, 9e-3], 2.42, 2.49, color="w")
    ax.text(2e-1, 2.48, r"$< -5$", ha="center", va="center", backgroundcolor="w")

    ax = axes[1, 0]
    # Clean up
    dM_chirps = np.zeros(f["M_chirp_MSUN_bests"].shape)
    dM_chirps = np.array(f["M_chirp_MSUN_bests"] - f["M_CHIRP_MSUN"])
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
        f["GAMMA_SS"],
        (dM_chirps / M_chirp_errs).T,
        levels=jnp.linspace(0, 100, 6),
        cmap="cividis",
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
        f["GAMMA_SS"],
        jnp.clip((f["snrs"] - jnp.sqrt(f["matches"])) / f["snrs"], a_min=0).T * 100,
        levels=jnp.linspace(0, 10, 11),
        cmap="magma",
    )
    plt.colorbar(
        cs,
        ax=ax,
        fraction=0.055,
        pad=0.04,
        label="% SNR loss with V",
    )
    ax.text(2e-1, 2.48, r"$> 10\%$", ha="center", va="center", backgroundcolor="w")

    for ax in axes.flatten():
        ax.set_xscale("log")
        ax.set_xlabel(r"$\rho_6$ [$10^{16}$ $\mathrm{M}_\odot \, \mathrm{pc}^{-3}$]")
        ax.set_ylabel(r"$\gamma_s$")
        ax.set_xlim(rho_6_to_rho6T(f["RHO_6S"][0]), 0.7)
        # Benchmarks
        ax.scatter(
            [rho_6_to_rho6T(setup_astro()[1])], [7 / 3], marker="*", color="r", s=100
        )
        ax.scatter(
            [rho_6_to_rho6T(setup_pbh()[1])], [9 / 4], marker=".", color="r", s=200
        )
        # rho_s contours
        if rho_s:
            cs = ax.contour(
                rho_6Ts,
                f["GAMMA_SS"],
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
