import click
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import warnings
from math import pi


# SI units
G = 6.67408e-11  # m^3 s^-2 kg^-1
C = 299792458.0  # m/s
MSUN = 1.98855e30  # kg
PC = 3.08567758149137e16  # m
YR = 365.25 * 24 * 3600  # s
# LISA parameters
# https://www.elisascience.org/files/publications/LISA_L3_20170120.pdf
t = 4.0  # LISA lifetime (sec. 3.4)
dt = 1 / 3.33  # LISA sampling rate [Hz] (sec. 5.4.2)


def get_r_isco(m_1):
    return 6 * G * m_1 / C ** 2


def get_f_isco(m_1):
    return np.sqrt(G * m_1 / get_r_isco(m_1) ** 3) / pi


def load_strains(fname, m_1, m_2):
    """
    Loads strain interpolators.

    Reference
    ---------
    https://arxiv.org/abs/1408.3534, eqs. 21 - 22

    Arguments
    ---------
    - m_1, m_2: black hole masses (kg)

    Returns
    -------
    - hp_t, hc_t: functions to compute the time-domain plus and cross
      strains. Their arguments are:
      - t: time relative to merger [s]
      - d_l: luminosity distance to binary [m]
      - iota: inclination angle [rad]
      - phi_c: phase at coalescence [rad]
    """
    _ts, _fs = np.loadtxt(fname, unpack=True, usecols=(0, 2))
    _fs *= 2  # GW frequency
    _ts = _ts - _ts[-1]  # time to merger

    f_isco = get_f_isco(m_1)
    if _fs[-1] < f_isco:
        warnings.warn("waveform doesn't extend to ISCO")

    _omega_gws = 2 * pi * _fs
    omega_gw = interp1d(_ts, _omega_gws)
    _omega_orbs = 2 * pi * (_fs / 2)
    omega_orb = interp1d(_ts, _omega_orbs)
    _rs = (G * (m_1 + m_2) / (pi * _fs) ** 2) ** (1 / 3)
    r = interp1d(_ts, _rs)

    # Strain functions
    def h0(t):
        return 4 * G * m_2 * omega_orb(t) ** 2 * r(t) ** 2 / C ** 4

    def hp_t(t, d_l, iota, phi_c=0):
        return (
            1
            / d_l
            * h0(t)
            * (1 + np.cos(iota) ** 2)
            / 2
            * np.cos(omega_gw(t) * t + phi_c)
        )

    def hc_t(t, d_l, iota, phi_c=0):
        return 1 / d_l * h0(t) * np.cos(iota) * np.sin(omega_gw(t) * t + phi_c)

    return hp_t, hc_t


# @click.command()
# @click.option("--m_1", type=float, help="IMBH mass")
# @click.option("--m_2", type=float, help="BH mass")
# @click.option(
#     "--rho", type=float, help="initial density normalization rho_s [MSUN / PC**3]"
# )
# @click.option("--gamma", type=float, help="initial spike slope")
# @click.option("--t", default=t, help="time before merger [yr]")
# @click.option("--dt", default=dt, help="time step [s]")
# @click.option("--d_l", default=1e6, help="luminosity distance [PC]")
# @click.option("--iota", default=0.0, help="inclination angle [rad]")
# @click.option("--phi_c", default=0.0, help="phase at coalescence [rad]")
# @click.option("--run_dir", default="/Users/acoogan/Physics/dark_dress/finalRuns/full/")
def save_strain(m_1, m_2, rho, gamma, t, dt, d_l, iota, phi_c, run_dir):
    m_1 *= MSUN
    m_2 *= MSUN
    d_l *= PC
    t *= YR
    ts = np.arange(-t, 0, dt)

    # Don't change this
    id_str = f"M1_{m_1 / MSUN:.1f}_M2_{m_2 / MSUN:.1f}_rho_{rho:g}_gamma_{gamma:.4f}"
    fname = os.path.join(run_dir, id_str, f"output_dynamic_dress_{id_str}.dat")
    try:
        hp_t, hc_t = load_strains(fname, m_1, m_2)
    except OSError:
        print(f"{fname} not found")
        return

    hp_ts = hp_t(ts, d_l, iota, phi_c)
    hc_ts = hc_t(ts, d_l, iota, phi_c)
    np.savetxt(
        f"strain-{id_str}.dat",
        np.stack((ts, hp_ts, hc_ts), axis=1),
        header="Columns: t [s], h_+, h_x",
    )


if __name__ == "__main__":
    # m_1s = [1e3, 3e3, 1e4]
    # m_2s = [1, 3, 10]
    # rhos = [20, 200, 2000]
    # gammas = [2.25, 2.3333, 2.5]
    run_dir = 

    if "rho1" not in fname and "ri" not in fname:
        print(fname)
    save_strain()
