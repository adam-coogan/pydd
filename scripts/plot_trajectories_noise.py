import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pydd.analysis import *
from pydd.binary import *


def load_S_n(filename):
    _fs, _sqrt_S_ns = np.loadtxt(filename, unpack=True)
    idxs = jnp.where(~jnp.isnan(_sqrt_S_ns))[0]
    return jax.jit(lambda f: jnp.interp(f, _fs, _sqrt_S_ns ** 2, jnp.inf, jnp.inf)), (
        _fs[idxs[0]],
        _fs[idxs[-1]],
    )


if __name__ == "__main__":
    rho_s_pbh = 1.798e4 * MSUN / PC ** 3
    gamma_s_pbh = 9 / 4

    S_n_aLIGO, f_range_aLIGO = load_S_n("data/aLIGO.dat")
    S_n_ce, f_range_ce = load_S_n("data/ce.dat")
    S_n_et, f_range_et = load_S_n("data/et.dat")

    fs = jnp.geomspace(1e-4, 1e4, 1000)

    plt.figure(figsize=(6, 4))
    plt.plot(fs, jnp.sqrt(fs * S_n_aLIGO(fs)), label="aLIGO")
    plt.plot(fs, jnp.sqrt(fs * S_n_ce(fs)), label="CE")
    plt.plot(fs, jnp.sqrt(fs * S_n_et(fs)), label="ET")
    plt.plot(fs, jnp.sqrt(fs * S_n_LISA(fs)), label="LISA")

    # Light PBH IMRI
    m_1 = 1 * MSUN
    m_2 = 1e-3 * MSUN
    vb = make_vacuum_binary(m_1, m_2)
    idxs = (fs < get_f_isco(m_1)) & (t_to_c(fs, vb) < 1 * YR)
    plt.plot(fs[idxs], (2 * fs * amp(fs, vb))[idxs], "k")
    f_b = get_f_b(m_1, m_2, gamma_s_pbh)
    plt.scatter(f_b, 2 * f_b * amp(f_b, vb), c="k", s=20)
    plt.text(
        1e-2, 3e-23, r"$(m_1,\, m_2) = (1,\, 10^{-3})\, \mathrm{M}_\odot$", fontsize=8.5
    )

    # IMRI
    m_1 = 1e3 * MSUN
    m_2 = 1.4 * MSUN
    vb = make_vacuum_binary(m_1, m_2)
    idxs = (fs < get_f_isco(m_1)) & (t_to_c(fs, vb) < 5 * YR)
    plt.plot(fs[idxs], (2 * fs * amp(fs, vb))[idxs], "k")
    f_b = get_f_b(m_1, m_2, gamma_s_pbh)
    plt.scatter(f_b, 2 * f_b * amp(f_b, vb), c="k", s=20)
    plt.text(
        6e-4, 5e-20, r"$(m_1,\, m_2) = (10^3,\, 1.4)\, \mathrm{M}_\odot$", fontsize=8.5
    )

    plt.ylim(5e-25, 1e-17)
    plt.xlabel(r"$f$ [Hz]")
    plt.ylabel("Characteristic strain")
    plt.loglog()
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig("figures/trajectories-and-noise.pdf")
