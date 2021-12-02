import jax.numpy as jnp
from scipy.optimize import root_scalar

from pydd.analysis import get_match_pads, loglikelihood_fft
from pydd.binary import (
    DynamicDress,
    MSUN,
    PC,
    VacuumBinary,
    YR,
    get_f_isco,
    get_m_1,
    make_dynamic_dress,
    t_to_c,
)

"""
Useful definitions shared among scripts.
"""

t_obs_lisa = 5 * YR
f_l = 1e-3  # Hz. Ideally would go down to 2e-5.
f_h = 1.0  # Hz. See fig. 1 of LISA Science Requirements.
n_f = 100000


def rho_6_to_rho6T(rho_6):
    return rho_6 / 1e16 / (MSUN / PC ** 3)


def rho_6T_to_rho6(rho_6T):
    return rho_6T * 1e16 * (MSUN / PC ** 3)


def get_loglikelihood_fn(dd_s, f_l=f_l, f_h=f_h, n_f=n_f):
    """
    x: parameter point
    dd_s: signal system
    """
    fs = jnp.linspace(f_l, f_h, n_f)
    pad_low, pad_high = get_match_pads(fs)

    def _ll(x):
        # Unpack parameters into dark dress ones
        gamma_s, rho_6T, M_chirp_MSUN, log10_q = x
        M_chirp = M_chirp_MSUN * MSUN
        q = 10 ** log10_q
        rho_6 = rho_6T_to_rho6(rho_6T)
        f_c = get_f_isco(get_m_1(M_chirp, q))
        dd_h = DynamicDress(
            gamma_s, rho_6, M_chirp, q, dd_s.Phi_c, dd_s.tT_c, dd_s.dL, f_c
        )
        return loglikelihood_fft(dd_h, dd_s, fs, pad_low, pad_high)

    return _ll


def get_ptform(
    u, gamma_s_range, rho_6T_range, log10_q_range, dM_chirp_MSUN_range, dd_s
):
    gamma_s = (gamma_s_range[1] - gamma_s_range[0]) * u[0] + gamma_s_range[0]
    rho_6T = (rho_6T_range[1] - rho_6T_range[0]) * u[1] + rho_6T_range[0]
    dM_chirp_MSUN = (dM_chirp_MSUN_range[1] - dM_chirp_MSUN_range[0]) * u[
        2
    ] + dM_chirp_MSUN_range[0]
    M_chirp_MSUN = dM_chirp_MSUN + dd_s.M_chirp / MSUN
    log10_q = (log10_q_range[1] - log10_q_range[0]) * u[3] + log10_q_range[0]
    return jnp.array([gamma_s, rho_6T, M_chirp_MSUN, log10_q])


def get_loglikelihood_fn_v(dd_s, f_l=f_l, f_h=f_h, n_f=n_f):
    """
    x: parameter point
    dd_s: signal system
    """
    # Unpack parameters into dark dress ones
    fs = jnp.linspace(f_l, f_h, n_f)
    pad_low, pad_high = get_match_pads(fs)

    def _ll(x):
        dd_h = VacuumBinary(x[0] * MSUN, dd_s.Phi_c, dd_s.tT_c, dd_s.dL, dd_s.f_c)
        return loglikelihood_fft(dd_h, dd_s, fs, pad_low, pad_high)

    return _ll


def get_ptform_v(u, M_chirp_MSUN_range):
    low, high = M_chirp_MSUN_range
    return jnp.array((high - low) * u + low)


# Benchmark parameters
M_1_BM = 1e3 * MSUN
M_2_BM = 1.4 * MSUN
DL_BM = 85e6 * PC  # gives SNR = 15.0
GAMMA_S_ASTRO = 7 / 3
GAMMA_S_PBH = 9 / 4
RHO_6_ASTRO = rho_6T_to_rho6(0.5448)
RHO_6_PBH = rho_6T_to_rho6(0.5345)


def setup_system(gamma_s, rho_6):
    """
    Sets up a dark dress with given spike parameters using benchmark masses and
    distance, and computes frequency at `t_obs_lisa` before coalescence.
    """
    m_1 = jnp.array(M_1_BM)
    m_2 = jnp.array(M_2_BM)

    dd_s = make_dynamic_dress(m_1, m_2, rho_6, gamma_s, dL=jnp.array(DL_BM))

    f_l = root_scalar(
        lambda f: t_to_c(f, dd_s) - t_obs_lisa,
        bracket=(1e-3, 1e-1),
        rtol=1e-15,
        xtol=1e-100,
    ).root

    return dd_s, f_l
