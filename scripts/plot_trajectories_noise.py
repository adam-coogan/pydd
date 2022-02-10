import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pydd.binary import *
from pydd.analysis import *
from pydd.noise import (
    S_n_aLIGO,
    S_n_ce,
    S_n_et,
    S_n_LISA,
)

rho_6_pbh = jnp.array(0.5345 * 1e16 * (MSUN / PC ** 3))
gamma_s_pbh = jnp.array(9 / 4)
fs = jnp.geomspace(1e-4, 1e4, 1000)

if __name__ == "__main__":
    plt.figure(figsize=(6, 4))
    plt.plot(fs, jnp.sqrt(fs * S_n_aLIGO(fs)), label="aLIGO")
    plt.plot(fs, jnp.sqrt(fs * S_n_et(fs)), label="ET")
    plt.plot(fs, jnp.sqrt(fs * S_n_ce(fs)), label="CE")
    plt.plot(fs, jnp.sqrt(fs * S_n_LISA(fs)), label="LISA")

    # IMRI
    m_1 = jnp.array(1e3 * MSUN)
    m_2 = jnp.array(1.4 * MSUN)
    dd = DynamicDress(
        gamma_s_pbh,
        rho_6_pbh,
        get_M_chirp(m_1, m_2),
        m_2 / m_1,
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(76e6 * PC),  # gives SNR of 15 at LISA for t_obs = 5 yr
        get_f_isco(m_1),
    )
    t_obs = 5 * YR
    idxs = (fs < get_f_isco(m_1)) & (t_to_c(fs, dd) < t_obs)
    plt.plot(fs[idxs], (2 * fs * amp(fs, dd))[idxs], "k")
    f_b = get_f_b(m_1, m_2, gamma_s_pbh)
    plt.scatter(f_b, 2 * f_b * amp(f_b, dd), c="k", s=20)
    plt.text(
        1e-2,
        7e-20,
        r"$(m_1,\, m_2,\, d_L) =$"
        "\n"
        r"$(10^3\, \mathrm{M}_\odot,\, 1.4\, \mathrm{M}_\odot,\, 76\, \mathrm{Mpc})$",
        fontsize=8,
        ha="center",
    )

    # Light PBH IMRI
    m_1 = jnp.array(1 * MSUN)
    m_2 = jnp.array(1e-3 * MSUN)
    dd = DynamicDress(
        gamma_s_pbh,
        rho_6_pbh,
        get_M_chirp(m_1, m_2),
        m_2 / m_1,
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(302e6 * PC),  # gives SNR of 12 at CE for t_obs = 1 yr
        get_f_isco(m_1),
    )
    t_obs = 1 * YR
    idxs = (fs < get_f_isco(m_1)) & (t_to_c(fs, dd) < t_obs)
    plt.plot(fs[idxs], (2 * fs * amp(fs, dd))[idxs], "k")
    f_b = get_f_b(m_1, m_2, gamma_s_pbh)
    plt.scatter(f_b, 2 * f_b * amp(f_b, dd), c="k", s=20)
    plt.text(
        2e-1,
        5e-24,
        r"$(m_1,\, m_2,\, d_L) =$"
        "\n"
        r"$(1\, \mathrm{M}_\odot,\, 10^{-3}\, \mathrm{M}_\odot,\, 302\, \mathrm{Mpc})$",
        fontsize=8,
        ha="center",
    )

    plt.ylim(5e-25, 1e-17)
    plt.xlabel(r"$f$ [Hz]")
    plt.ylabel("Characteristic strain")
    plt.loglog()
    plt.legend(frameon=False)
    plt.tight_layout()

    plt.savefig("figures/trajectories-and-noise.pdf")
