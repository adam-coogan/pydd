from math import log10
import os
import pickle

import dynesty
from dynesty import plotting as dyplot
import matplotlib.pyplot as plt
import numpy as np

from pydd.binary import MSUN, get_M_chirp
from utils import (
    GAMMA_S_ASTRO,
    GAMMA_S_PBH,
    M_1_BM,
    M_2_BM,
    RHO_6_ASTRO,
    RHO_6_PBH,
    rho_6_to_rho6T,
)

"""
Plot nested sampling posteriors for the astrophysical and PBH benchmarks.

Requires `ns/rho_6T=0.5448_gamma_s=2.33333.pkl` and
`ns/rho_6T=0.5345_gamma_s=2.25.pkl`, which are produced by
`job_measurability.sh`.

Produces `../figures/ns-astro-lisa.pdf` and `../figures/ns-pbh-lisa.pdf`
"""

labels = (
    r"$\gamma_s$",
    r"$\rho_6$ [$10^{16}$ $\mathrm{M}_\odot \, \mathrm{pc}^{-3}$]",
    r"$\mathcal{M}$ [M$_\odot$]",
    r"$\log_{10} q$",
)
quantiles_2d = [1 - np.exp(-(x ** 2) / 2) for x in [1, 2, 3]]  # what's published
# quantiles_2d = [0.6827, 0.9545, 0.9973]  # what the paper caption says
smooth = 0.01


def get_base_path(rho_6, gamma_s):
    return f"rho_6T={rho_6_to_rho6T(rho_6):g}_gamma_s={gamma_s:g}"


def plot_astro():
    base_path = get_base_path(RHO_6_ASTRO, GAMMA_S_ASTRO)
    with open(os.path.join("ns", f"{base_path}.pkl"), "rb") as infile:
        results = pickle.load(infile)

    cfig, axes = dyplot.cornerplot(
        results,
        labels=labels,
        quantiles_2d=quantiles_2d,
        smooth=0.015,
        show_titles=True,
        title_kwargs={"fontsize": 12},
        span=[(2.25, 2.42), (0.42, 0.72), (19.386, 19.394), (-3.3, -2.5)],
        max_n_ticks=6,
        truths=(
            GAMMA_S_ASTRO,
            rho_6_to_rho6T(RHO_6_ASTRO),
            get_M_chirp(M_1_BM, M_2_BM) / MSUN,
            log10(M_2_BM / M_1_BM),
        ),
    )
    # Deal with chirp mass manually
    axes[2, 0].set_yticklabels([f"{t:.4f}" for t in axes[2, 0].get_yticks()])
    Mc_quantiles = dynesty.utils.quantile(results.samples[:, 2], [0.025, 0.5, 0.975])
    axes[2, 2].set_title(
        labels[2]
        + r" = ${%.3f}_{-%.3f}^{+%.3f}$"
        % (
            Mc_quantiles[1],
            Mc_quantiles[1] - Mc_quantiles[0],
            Mc_quantiles[2] - Mc_quantiles[1],
        ),
        fontsize=12,
    )
    cfig.tight_layout(pad=0.2)
    cfig.savefig(os.path.join("..", "figures", f"ns-astro-lisa.pdf"))


def plot_pbh():
    base_path = get_base_path(RHO_6_PBH, GAMMA_S_PBH)
    with open(os.path.join("ns", f"{base_path}.pkl"), "rb") as infile:
        results = pickle.load(infile)

    cfig, axes = dyplot.cornerplot(
        results,
        labels=labels,
        quantiles_2d=quantiles_2d,
        smooth=0.015,
        show_titles=True,
        title_kwargs={"fontsize": 12},
        span=[(2.25, 2.325), (0.4, 0.75), (19.3875, 19.3895), (-3.0, -2.5)],
        max_n_ticks=6,
        truths=(
            GAMMA_S_PBH,
            rho_6_to_rho6T(RHO_6_PBH),
            get_M_chirp(M_1_BM, M_2_BM) / MSUN,
            log10(M_2_BM / M_1_BM),
        ),
    )
    # Deal with chirp mass manually
    axes[2, 0].set_yticklabels([f"{t:.4f}" for t in axes[2, 0].get_yticks()])
    Mc_quantiles = dynesty.utils.quantile(results.samples[:, 2], [0.025, 0.5, 0.975])
    axes[2, 2].set_title(
        labels[2]
        + r" = ${%.3f}_{-%.3f}^{+%.3f}$"
        % (
            Mc_quantiles[1],
            Mc_quantiles[1] - Mc_quantiles[0],
            Mc_quantiles[2] - Mc_quantiles[1],
        ),
        fontsize=12,
    )
    cfig.tight_layout(pad=0.2)
    cfig.savefig(os.path.join("..", "figures", f"ns-pbh-lisa.pdf"))


if __name__ == "__main__":
    plot_astro()
    plot_pbh()
