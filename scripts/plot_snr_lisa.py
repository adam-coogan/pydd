import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from tqdm.auto import tqdm

import jax.numpy as jnp

from pydd.analysis import calculate_SNR
from pydd.binary import *
from pydd.noise import S_n_LISA, f_range_LISA

"""
Plots SNRs for GR-in-vacuum binaries as a function of chirp mass and luminosity
distance.

Produces `../figures/snrs-lisa.pdf`.
"""

if __name__ == "__main__":
    t_obs_lisa = 5 * YR
    f_c = 1e2  # get_f_isco(1e3 * MSUN)
    M_chirp_min = 10 * MSUN  # get_M_chirp(1e3 * MSUN, 1 * MSUN)
    M_chirp_max = 2000 * MSUN  # get_M_chirp(1e5 * MSUN, 1e2 * MSUN)
    dL_min = 1e6 * PC
    dL_max = 10e9 * PC

    M_chirps = jnp.geomspace(M_chirp_min, M_chirp_max, 40)
    dLs = jnp.geomspace(dL_min, dL_max, 35)

    snrs = np.zeros([len(M_chirps), len(dLs)])
    f_ls = np.zeros([len(M_chirps), len(dLs)])

    for i, M_chirp in enumerate(tqdm(M_chirps)):
        for j, dL in enumerate(dLs):
            dd_v = VacuumBinary(M_chirp, 0.0, 0.0, dL, f_c)
            f_l = root_scalar(
                lambda f: t_to_c(f, dd_v) - t_obs_lisa,
                bracket=(1e-3, 1e-1),
                rtol=1e-15,
                xtol=1e-100,
            ).root
            f_ls[i, j] = f_l
            fs = jnp.linspace(f_l, f_c, 3000)
            snrs[i, j] = calculate_SNR(dd_v, fs, S_n_LISA)

    plt.figure(figsize=(4, 3.5))

    plt.axvline(get_M_chirp(1e3, 1.4), color="r", linestyle="--")
    plt.axhline(76, color="r", linestyle="--")
    plt.xscale("log")
    plt.yscale("log")
    cs = plt.contour(
        M_chirps / MSUN,
        dLs / (1e6 * PC),
        jnp.log10(snrs.T),
        levels=jnp.linspace(-2, 6, 9).round(),
        alpha=0.8,
    )
    plt.clabel(cs, inline=True, fontsize=10, fmt=r"$10^{%i}$")

    cs = plt.contour(
        M_chirps / MSUN,
        dLs / (1e6 * PC),
        snrs.T,
        levels=[15],
        colors=["r"],
    )
    plt.clabel(cs, inline=True, fontsize=10, fmt=r"%g")

    plt.xlabel(r"$\mathcal{M}$ [M$_\odot$]")
    plt.ylabel(r"$d_L$ [Mpc]")
    plt.tight_layout()

    plt.savefig("../figures/snrs-lisa.pdf")
