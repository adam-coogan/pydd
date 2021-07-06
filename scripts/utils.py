import jax.numpy as jnp
import numpy as np
from scipy.optimize import root_scalar

from pydd.analysis import loglikelihood_fft
from pydd.binary import (
    DynamicDress,
    MSUN,
    PC,
    VacuumBinary,
    YR,
    get_c_f,
    get_f_isco,
    get_m_1,
    get_m_2,
    get_rho_s,
    make_dynamic_dress,
    t_to_c,
)

t_obs_lisa = 5 * YR
labels = (
    r"$\gamma_s$",
    r"$\rho_6$ [$10^{16}$ $\mathrm{M}_\odot \, \mathrm{pc}^{-3}$]",
    r"$\mathcal{M}$ [M$_\odot$]",
    r"$\log_{10} q$",
)
quantiles = [1 - 0.95, 0.95]
quantiles_2d = [1 - np.exp(-(x ** 2) / 2) for x in [1, 2, 3]]


def rho_6_to_rho6T(rho_6):
    return rho_6 / 1e16 / (MSUN / PC ** 3)


def rho_6T_to_rho6(rho_6T):
    return rho_6T * 1e16 * (MSUN / PC ** 3)


def get_loglikelihood(x, dd_s, f_l):
    """
    x: parameter point
    dd_s: signal system
    """
    # Unpack parameters into dark dress ones
    gamma_s, rho_6T, M_chirp_MSUN, log10_q = x
    M_chirp = M_chirp_MSUN * MSUN
    q = 10 ** log10_q
    m_1 = get_m_1(M_chirp, q)
    m_2 = get_m_2(M_chirp, q)
    rho_6 = rho_6T_to_rho6(rho_6T)
    rho_s = get_rho_s(rho_6, m_1, gamma_s)
    c_f = get_c_f(m_1, m_2, rho_s, gamma_s)
    f_c = get_f_isco(m_1)

    dd_h = DynamicDress(
        gamma_s, c_f, M_chirp, q, dd_s.Phi_c, dd_s.tT_c, dd_s.dL_iota, f_c
    )

    f_h = jnp.maximum(dd_s.f_c, dd_h.f_c)
    return loglikelihood_fft(dd_h, dd_s, f_l, f_h, 100000, 3000)


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


def get_loglikelihood_v(x, dd_s, f_l):
    """
    x: parameter point
    dd_s: signal system
    """
    # Unpack parameters into dark dress ones
    dd_h = VacuumBinary(x[0] * MSUN, dd_s.Phi_c, dd_s.tT_c, dd_s.dL_iota, dd_s.f_c)
    return loglikelihood_fft(dd_h, dd_s, f_l, dd_s.f_c, 100000, 3000)


def get_ptform_v(u, M_chirp_MSUN_range):
    low, high = M_chirp_MSUN_range
    return jnp.array((high - low) * u + low)


def setup_astro():
    m_1 = jnp.array(1e3 * MSUN)
    m_2 = jnp.array(1 * MSUN)
    rho_s = jnp.array(226 * MSUN / PC ** 3)
    gamma_s = jnp.array(7 / 3)

    rho_6 = root_scalar(
        lambda rho: get_rho_s(rho, m_1, gamma_s) - rho_s,
        bracket=(1e-5, 1e-1),
        rtol=1e-15,
        xtol=1e-100,
    ).root

    dd_s = make_dynamic_dress(m_1, m_2, rho_s, gamma_s)
    dd_v = VacuumBinary(dd_s.M_chirp, dd_s.Phi_c, dd_s.tT_c, dd_s.dL_iota, dd_s.f_c)

    f_l = root_scalar(
        lambda f: t_to_c(f, dd_s) - t_obs_lisa,
        bracket=(1e-3, 1e-1),
        rtol=1e-15,
        xtol=1e-100,
    ).root

    return dd_s, rho_6, f_l, dd_v


def setup_pbh():
    m_1 = jnp.array(1e3 * MSUN)
    m_2 = jnp.array(1 * MSUN)
    gamma_s = jnp.array(9 / 4)
    rho_6 = 5345040429615936.0 * MSUN / PC ** 3
    rho_s = get_rho_s(rho_6, m_1, gamma_s)

    dd_s = make_dynamic_dress(m_1, m_2, rho_s, gamma_s)
    dd_v = VacuumBinary(dd_s.M_chirp, dd_s.Phi_c, dd_s.tT_c, dd_s.dL_iota, dd_s.f_c)

    f_l = root_scalar(
        lambda f: t_to_c(f, dd_s) - t_obs_lisa,
        bracket=(1e-3, 1e-1),
        rtol=1e-15,
        xtol=1e-100,
    ).root

    return dd_s, rho_6, f_l, dd_v


def setup_system(gamma_s, rho_6, m_1=1e3 * MSUN, m_2=1 * MSUN, dL=100e6 * PC):
    m_1 = jnp.array(m_1)
    m_2 = jnp.array(m_2)
    rho_s = get_rho_s(rho_6, m_1, gamma_s)

    dd_s = make_dynamic_dress(m_1, m_2, rho_s, gamma_s, dL=dL)

    f_l = root_scalar(
        lambda f: t_to_c(f, dd_s) - t_obs_lisa,
        bracket=(1e-3, 1e-1),
        rtol=1e-15,
        xtol=1e-100,
    ).root

    return dd_s, f_l
