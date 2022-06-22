from jax import jit
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from pydd.analysis import calculate_SNR
from pydd.binary import MSUN, PC, VacuumBinary, WEEK, get_M_chirp, get_f_range
from pydd.noise import S_n_aLIGO, S_n_ce, S_n_et, f_range_aLIGO, f_range_ce, f_range_et

"""
Plots SNRs for GR-in-vacuum binaries as a function of chirp mass and luminosity
distance.

Produces `../figures/snrs-aligo-ce-et-week.pdf`.
"""

T_OBS = 1 * WEEK
SNR_THRESH = 12.0


def get_snrs(M_chirps, dLs, S_n, f_range_n):
    get_snr = jit(lambda vb, fs: calculate_SNR(vb, fs, S_n))

    snrs = np.zeros([len(M_chirps), len(dLs)])
    f_ls = np.zeros([len(M_chirps), len(dLs)])
    for i, M_chirp in enumerate(tqdm(M_chirps)):
        for j, dL in enumerate(dLs):
            vb = VacuumBinary(
                M_chirp, jnp.array(0.0), jnp.array(0.0), dL, f_range_n[-1]
            )
            f_l = get_f_range(vb, T_OBS)[0]
            f_ls[i, j] = f_l
            fs = jnp.linspace(f_l, vb.f_c, 5_000)
            snrs[i, j] = get_snr(vb, fs)

    return snrs, f_ls


if __name__ == "__main__":
    M_chirps = jnp.geomspace(1e-4 * MSUN, 1 * MSUN, 40)
    dLs = jnp.geomspace(1e6 * PC, 1e9 * PC, 35)

    snrs_aLIGO = get_snrs(M_chirps, dLs, S_n_aLIGO, f_range_aLIGO)[0]
    print("Computed aLIGO SNRs")
    snrs_et = get_snrs(M_chirps, dLs, S_n_et, f_range_et)[0]
    print("Computed et SNRs")
    snrs_ce = get_snrs(M_chirps, dLs, S_n_ce, f_range_ce)[0]
    print("Computed ce SNRs")

    fig, axs = plt.subplots(1, 3, figsize=(10, 3.5))

    # Distances at which systems have SNR of 12 over last year of inspiral
    MPC = 1e6 * PC
    dL_12s = [6.5 * MPC, 102 * MPC, 302 * MPC]
    # Contour label positions
    manuals = [
        [(3e-4, 3e2), (1e-3, 7e1), (6e-3, 3e1), (7e-2, 4e0), (6e-1, 1.5e0)],
        [(2e-4, 6e2), (4e-4, 1.5e2), (7e-3, 2e1), (6e-2, 1e1), (2e-1, 2e0), (1e0, 1e0)],
        [(3e-4, 3e2), (2e-3, 1.5e2), (3e-2, 4e1), (1e-1, 7e0), (5e-1, 2e0), (1e0, 1e0)],
    ]

    for i, (ax, snrs, dL_12, manual) in enumerate(
        zip(axs, [snrs_aLIGO, snrs_et, snrs_ce], dL_12s, manuals)
    ):
        ax.set_xscale("log")
        ax.set_yscale("log")
        cs = ax.contour(
            M_chirps / MSUN,
            dLs / MPC,
            jnp.log10(snrs.T),
            levels=[-2, -1, 0, 2, 3, 4, 5],
            colors=["k"],
        )
        ax.clabel(
            cs,
            inline=True,
            fontsize=10,
            fmt=r"$10^{%i}$",
            manual=manual,
        )

        cs = ax.contour(
            M_chirps / MSUN,
            dLs / MPC,
            snrs.T,
            levels=[SNR_THRESH],
            colors=["r"],
        )
        ax.clabel(cs, inline=True, fontsize=10, fmt=r"%g", manual=[(2e-3, 3e1)])

        ax.axvline(get_M_chirp(1, 1e-3), color="r", linestyle="-")
        ax.axhline(dL_12 / MPC, color=f"C{i}", linestyle="-")  # aLIGO

        ax.set_xlabel(r"$\mathcal{M}$ [M$_\odot$]")

    axs[0].set_ylabel(r"$d_L$ [Mpc]")
    axs[0].set_title("aLIGO")
    axs[1].set_title("Einstein Telescope")
    axs[2].set_title("Cosmic Explorer")

    fig.tight_layout()
    figpath = "../figures/snrs-aligo-ce-et-week.pdf"
    fig.savefig(figpath)
    print(f"Saved figure to {figpath}")
