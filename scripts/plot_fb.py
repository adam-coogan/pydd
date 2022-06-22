from collections import defaultdict
from contextlib import contextmanager

from jax.config import config
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import beta, betainc as reg_betainc

from pydd.binary import *

config.update("jax_disable_jit", True)


"""
Plots f_b values along with analytic and empirical estimates for calibration
systems.

Requires `fb_fits/fb_fits.npz`, which is produced by `fit_fb.py`.

Produces `../figures/f_b-scaling.pdf`.
"""


@contextmanager
def autoscale_off(ax=None):
    ax = ax or plt.gca()
    lims = [ax.get_xlim(), ax.get_ylim()]
    yield
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])


class RunFit(NamedTuple):
    """
    Container for f_b fit results
    """

    m_1: float  # MSUN
    m_2: float  # MSUN
    rho_s: float  # MSUN / PC**3
    rho6T: float  # 1e16 * MSUN / PC**3
    gamma_s: float
    f_b: float  # Hz


def betainc(a, b, x):
    return beta(a, b) * reg_betainc(a, b, x)


def get_f_b_ana(m_1, m_2, gamma_s):
    """
    Analytic f_b scaling relation
    """
    h = betainc(gamma_s - 1 / 2, 3 / 2, 1.0) - betainc(gamma_s - 1 / 2, 3 / 2, 1 / 2)
    g = (2 ** (3 - gamma_s) + gamma_s - 4) / ((3 - gamma_s) * (2 - gamma_s) * h)
    return m_2 ** (3 / 5) / m_1 ** (8 / 5) * (jnp.log(1 + m_1 / m_2) / g) ** (3 / 5)


if __name__ == "__main__":
    # Load results
    results = dict(jnp.load("fb_fits/fb_fits.npz"))
    vals = jnp.stack(list(results.values()), axis=-1)
    fits = []
    for v in vals:
        fits.append(RunFit(*v))

    # Split into calibration and test runs
    results_val = defaultdict(lambda: [])
    results_cal = defaultdict(lambda: [])
    m_1_cal = jnp.array([1e3, 3e3, 1e4])
    m_2_cal = jnp.array([1.0, 3.0, 10.0])
    gamma_s_cal = jnp.array([2.25, 2.3333, 2.5])

    for val in vals:
        if val[0] in m_1_cal and val[1] in m_2_cal and val[4] in gamma_s_cal:
            for i, k in enumerate(results.keys()):
                results_cal[k].append(val[i])
        else:
            for i, k in enumerate(results.keys()):
                results_val[k].append(val[i])

    results_cal = {k: jnp.array(v) for k, v in results_cal.items()}
    results_val = {k: jnp.array(v) for k, v in results_val.items()}

    # Positions for calibration runs
    x_coords_cal = np.zeros((len(results_cal["m_1"]),))
    for m_1 in m_1_cal:
        for i, m_2 in enumerate(m_2_cal):
            for j, gamma_s in enumerate(gamma_s_cal):
                idxs = jnp.where(
                    (results_cal["m_1"] == m_1)
                    & (results_cal["m_2"] == m_2)
                    & (results_cal["gamma_s"] == gamma_s)
                )
                x_coords_cal[idxs] = i + (j - 1) / 4

    fig, axes = plt.subplots(1, 1, figsize=(6, 3.5), dpi=200)

    ax = axes
    # Must format before autoscale_off
    ax.set_yscale("log")
    ax.set_ylabel(r"$f_b$ [Hz]")
    ax.set_ylim(8.5e-3, 2.9)
    ax.set_xticks(jnp.arange(len(m_2_cal)))
    ax.set_xticklabels([r"$m_2 = %g\, \mathrm{M}_\odot$" % m_2 for m_2 in m_2_cal])
    # ax.set_title("Calibration")

    sc = ax.scatter(
        x_coords_cal,
        results_cal["f_b"],
        c=results_cal["gamma_s"],
        vmin=2.25,
        vmax=2.5,
        label=r"$\mathtt{HaloFeedback}$",
    )
    plt.colorbar(sc, label=r"$\gamma_\mathrm{sp}$", ax=ax)

    ax.scatter(
        x_coords_cal + 0.05,
        get_f_b(
            results_cal["m_1"] * MSUN, results_cal["m_2"] * MSUN, results_cal["gamma_s"]
        ),
        c="r",
        s=10,
        vmin=2.25,
        vmax=2.5,
        label="Empirical",
    )
    ax.scatter(
        x_coords_cal + 0.1,
        get_f_b_ana(
            results_cal["m_1"] * MSUN, results_cal["m_2"] * MSUN, results_cal["gamma_s"]
        )
        * 3.75e34,
        c="c",
        marker="x",
        s=20,
        vmin=2.25,
        vmax=2.5,
        label="Analytic",
    )
    ax.legend(loc="lower right", fontsize=8.5)

    with autoscale_off(ax):
        ax.fill_between([-0.5, 2.5], 0, 0.5, color="k", alpha=0.03)
        ax.fill_between([-0.5, 2.5], 0, 0.09, color="k", alpha=0.03)
        ax.plot([-0.5, 2.5], 2 * [0.5], ":k", linewidth=0.5)
        ax.plot([-0.5, 2.5], 2 * [0.09], ":k", linewidth=0.5)
        ax.fill_betweenx([0, 5], 0.5, 2.5, color="k", alpha=0.03)
        ax.fill_betweenx([0, 5], 1.5, 2.5, color="k", alpha=0.03)

    ax.text(
        0.05,
        6e-2,
        r"$m_1 = 10^4\, \mathrm{M}_\odot$",
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.text(
        0.05,
        3.5e-1,
        r"$m_1 = 3 \times 10^3\, \mathrm{M}_\odot$",
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.text(
        0.05,
        2e0,
        r"$m_1 = 10^3\, \mathrm{M}_\odot$",
        ha="center",
        va="center",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig("../figures/f_b-scaling.pdf")
