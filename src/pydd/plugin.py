""" Exporting the PyCBC plugin interface for the PyDD waveforms
"""


def pydd_td_strain(**params):
    """Generate the GW polarizations in PyCBC types"""

    from pycbc.types import TimeSeries
    import numpy as np
    from scipy.interpolate import interp1d
    from math import pi
    from pydd.binary import t_to_c, make_vacuum_binary, make_dynamic_dress

    # SI units
    G = 6.67408e-11  # m^3 s^-2 kg^-1
    C = 299792458.0  # m/s
    MSUN = 1.98855e30  # kg
    PC = 3.08567758149137e16  # m

    m_1 = params['mass1'] * MSUN
    m_2 = params['mass2'] * MSUN
    iota = params['inclination']
    d_l = params['distance'] * PC * 1e6
    system = params['system']
    dt = params['delta_t']
    phi_c = params['coa_phase']
    f_lower = params['f_lower']

    if (system == "vacuum"):
        dd = make_vacuum_binary(m_1, m_2)
    else:
        gamma = 9./4.
        rho = 1.396e13*(m_1/MSUN)**(3/4)
        rho *= MSUN/PC**3
        dd = make_dynamic_dress(m_1, m_2, rho, gamma)

    if 'f_final' in params and params['f_final'] > params['f_lower']:
        f_final = params['f_final']
    else:
        f_final = dd.f_c

    _fs = np.geomspace(f_lower, f_final, 10000)
    _ts = t_to_c(_fs, dd)
    _ts *= -1  # time to merger

    t_merge = t_to_c(f_lower, dd)
    t_stop = t_to_c(dd.f_c, dd)
    t_obs = t_merge - t_stop
    ts = np.arange(t_stop - t_obs, t_stop, dt)

    _omega_gws = 2 * pi * _fs
    omega_gw = interp1d(_ts, _omega_gws)

    _omega_orbs = 2 * pi * (_fs / 2)
    omega_orb = interp1d(_ts, _omega_orbs)
    _rs = (G * (m_1 + m_2) / (pi * _fs) ** 2) ** (1 / 3)
    r = interp1d(_ts, _rs)

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

    hp_ts = hp_t(ts, d_l, iota, phi_c)
    hc_ts = hc_t(ts, d_l, iota, phi_c)

    return (TimeSeries(hp_ts, delta_t=dt, epoch=float(-t_merge)),
           TimeSeries(hp_ts, delta_t=dt, epoch=float(-t_merge)))
