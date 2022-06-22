import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

from pydd.binary import *


if __name__ == "__main__":
    t_obs_lisa = 5 * YR

    id_str = "grid_rho=226.0_gamma=2.3333_m_imbh=1000_m_bh=1.0_t_m=5.0"
    _ts, _fs = np.loadtxt(
        f"../../finalRuns/full/{id_str}/output_dynamic_dress_{id_str}.dat",
        unpack=True,
        usecols=(0, 2),
    )
    _fs *= 2  # orb -> gw
    _Ns = cumtrapz(_fs, _ts, initial=0)
    _Phi_d = interp1d(_fs, 2 * pi * (_Ns[-1] - _Ns), bounds_error=False, fill_value=0.0)
    f_d = interp1d(_ts - _ts[-1], _fs, bounds_error=False, fill_value=0.0)

    _ts, _fs = np.loadtxt(
        f"../../finalRuns/full/{id_str}/output_vacuum.dat",
        unpack=True,
        usecols=(0, 2),
    )
    _fs *= 2  # orb -> gw
    _Ns = cumtrapz(_fs, _ts, initial=0)
    _Phi_v = interp1d(_fs, 2 * pi * (_Ns[-1] - _Ns), bounds_error=False, fill_value=0.0)
    f_v = interp1d(_ts - _ts[-1], _fs, bounds_error=False, fill_value=0.0)

    # Get shared frequency grid
    t_min = -f_d.x[0]
    print(f"Waveform starts {t_min / YR} yr before merger")
    f_start, f_end = f_d(-t_obs_lisa), min(_Phi_d.x[-1], _Phi_v.x[-1])
    f_grid = jnp.geomspace(f_start, f_end, 1000)

    # Align waveforms at same final frequency
    Phi_v = interp1d(
        f_grid, _Phi_v(f_grid) - _Phi_v(f_end), bounds_error=False, fill_value=0.0
    )
    Phi_d = interp1d(
        f_grid, _Phi_d(f_grid) - _Phi_d(f_end), bounds_error=False, fill_value=0.0
    )

    fig, ax = plt.subplots()

    vb = make_vacuum_binary(1e3 * MSUN, 1 * MSUN)
    dd = make_dynamic_dress(1e3 * MSUN, 1 * MSUN, 226 * MSUN / PC ** 3, 7 / 3)
    fs = jnp.geomspace(2.5e-2, get_f_isco(1e3 * MSUN), 200)
    rs = (G * (1e3 + 1) * MSUN / (pi ** 2 * fs ** 2)) ** (1 / 3)

    ax.loglog(fs, Phi_v(fs) - Phi_d(fs), label="Numerical")
    ax.loglog(fs, Phi_to_c(fs, vb) - Phi_to_c(fs, dd), "--", label="Approximate")

    ax_top = plt.gca().twiny()
    ax_top.loglog(rs / PC, jnp.full_like(rs, 1e-100))
    ax_top.invert_xaxis()
    ax_top.set_xlabel(r"$r$ [pc]")

    ax.set_xlim(fs[0], 5)
    ax.set_ylim(1e-3, 1e6)
    ax.set_xlabel(r"$f$ [Hz]")
    ax.set_ylabel(r"$\Phi_V - \Phi$ [rad]")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig("../figures/dephasing.pdf")
