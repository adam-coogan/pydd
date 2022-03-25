import jax.numpy as jnp
import matplotlib.pyplot as plt

from pydd.binary import (
    GAMMA_S_PBH,
    MSUN,
    PC,
    MONTH, WEEK, DAY,
    YR,
    get_f_b,
    amp,
)
from pydd.noise import S_n_aLIGO, S_n_ce, S_n_et, S_n_LISA, f_range_LISA, f_range_ce
from pydd.utils import get_target_pbh_dynamicdress

plt.style.use("../plot_style.mplstyle")
PLOT_KWARGS = dict(color="k")
SCATTER_KWARGS = dict(c="k", s=20, zorder=10)


def main():
    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax_top = ax.twiny()

    fs_n = jnp.geomspace(1e-4, 1e4, 1000)
    ax.plot(fs_n, jnp.sqrt(fs_n * S_n_aLIGO(fs_n)), label="aLIGO")
    ax.plot(fs_n, jnp.sqrt(fs_n * S_n_et(fs_n)), label="ET")
    ax.plot(fs_n, jnp.sqrt(fs_n * S_n_ce(fs_n)), label="CE")
    ax.plot(fs_n, jnp.sqrt(fs_n * S_n_LISA(fs_n)), label="LISA")

    # IMRI
    m_1 = 1e3 * MSUN
    m_2 = 1.4 * MSUN
    t_obs = 5 * YR
    dd, f_range_d = get_target_pbh_dynamicdress(
        m_1, m_2, t_obs, 15.0, S_n_LISA, f_range_LISA
    )
    fs = jnp.geomspace(*f_range_d, 500)
    f_b = get_f_b(m_1, m_2, GAMMA_S_PBH)
    ax.plot(fs, 2 * fs * amp(fs, dd), **PLOT_KWARGS)
    ax.scatter(f_b, 2 * f_b * amp(f_b, dd), **SCATTER_KWARGS)
    ax.text(
        1e-2,
        7e-20,
        r"$(m_1,\, m_2,\, d_L) =$"
        "\n"
        r"$(10^3\, \mathrm{M}_\odot,\, 1.4\, \mathrm{M}_\odot,\, %i\, \mathrm{Mpc})$" % (dd.dL / (1e6 * PC)),
        fontsize=8,
        ha="center",
    )

    # Light PBH IMRI
    m_1 = 1 * MSUN
    m_2 = 1e-3 * MSUN
    t_obs = 1 * YR
    dd, f_range_d = get_target_pbh_dynamicdress(
        m_1, m_2, t_obs, 15.0, S_n_ce, f_range_ce
    )
    fs = jnp.geomspace(*f_range_d, 500)
    f_b = get_f_b(m_1, m_2, GAMMA_S_PBH)
    ax.plot(fs, 2 * fs * amp(fs, dd), **PLOT_KWARGS)
    ax.scatter(f_b, 2 * f_b * amp(f_b, dd), **SCATTER_KWARGS)
    # Indicate other observing durations
    f_lows = []
    for t_obs in [1 * YR, 1 * MONTH, 1 * WEEK, 1 * DAY]:
        f_low = get_target_pbh_dynamicdress(
            m_1, m_2, t_obs, 15.0, S_n_ce, f_range_ce
        )[1][0]
        ax.axvline(f_low, color="k", linewidth=0.5)
        f_lows.append(f_low)
    ax.text(
        2e-1,
        5e-24,
        r"$(m_1,\, m_2,\, d_L) =$"
        "\n"
        r"$(1\, \mathrm{M}_\odot,\, 10^{-3}\, \mathrm{M}_\odot,\, %i\, \mathrm{Mpc})$" % (dd.dL / (1e6 * PC)),
        fontsize=8,
        ha="center",
    )

    ax.set_xlabel(r"$f$ [Hz]")
    ax.set_ylabel("Characteristic strain")
    ax.set_xlim(1e-4, 1e4)
    ax.set_ylim(5e-25, 1e-17)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False)

    ax_top.set_xlim(1e-4, 1e4)
    ax_top.set_ylim(5e-25, 1e-17)
    ax_top.set_xscale("log")
    ax_top.set_yscale("log")
    ax_top.set_xticks(f_lows)
    ax_top.set_xticklabels(["Y", "M", "W", "D"], fontsize=9)

    plt.tight_layout()
    plt.savefig("figures/trajectories-and-noise.pdf")


if __name__ == "__main__":
    main()
