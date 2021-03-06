from math import pi
from typing import NamedTuple, Union

import jax
from jax import jit
import jax.numpy as jnp
from jax.scipy.special import betainc
from jaxinterp2d import interp2d
from scipy.optimize import minimize_scalar
from scipy.special import hyp2f1


"""
Functions for computing waveforms and various parameters for different types of
binaries.

While we use different notation in our paper, the `_to_c` notation means that
the function returns zero when the input frequency equals the binary's
coalescence frequency. This makes it much easier to compute dephasings and the
like since phases don't need to be aligned manually.

Uses SI units.
"""

G = 6.67408e-11  # m^3 s^-2 kg^-1
C = 299792458.0  # m/s
MSUN = 1.98855e30  # kg
PC = 3.08567758149137e16  # m
YR = 365.25 * 24 * 3600  # s


class VacuumBinary(NamedTuple):
    """
    GR-in-vacuum binary.
    """

    M_chirp: jnp.ndarray
    Phi_c: jnp.ndarray
    tT_c: jnp.ndarray
    dL: jnp.ndarray
    f_c: jnp.ndarray


class StaticDress(NamedTuple):
    """
    A dark dress with a non-evolving DM halo.
    """

    gamma_s: jnp.ndarray
    c_f: jnp.ndarray
    M_chirp: jnp.ndarray
    Phi_c: jnp.ndarray
    tT_c: jnp.ndarray
    dL: jnp.ndarray
    f_c: jnp.ndarray


class DynamicDress(NamedTuple):
    """
    A dark dress with an evolving DM halo.
    """

    gamma_s: jnp.ndarray
    rho_6: jnp.ndarray
    M_chirp: jnp.ndarray
    q: jnp.ndarray
    Phi_c: jnp.ndarray
    tT_c: jnp.ndarray
    dL: jnp.ndarray
    f_c: jnp.ndarray


Binary = Union[VacuumBinary, StaticDress, DynamicDress]


# @jit
def get_M_chirp(m_1, m_2):
    return (m_1 * m_2) ** (3 / 5) / (m_1 + m_2) ** (1 / 5)


# @jit
def get_m_1(M_chirp, q):
    return (1 + q) ** (1 / 5) / q ** (3 / 5) * M_chirp


# @jit
def get_m_2(M_chirp, q):
    return (1 + q) ** (1 / 5) * q ** (2 / 5) * M_chirp


# @jit
def get_r_isco(m_1):
    return 6 * G * m_1 / C ** 2


# @jit
def get_f_isco(m_1, m_2):
    return jnp.sqrt(G * (m_1 + m_2) / get_r_isco(m_1) ** 3) / pi
    #return jnp.sqrt(G * m_1 / get_r_isco(m_1) ** 3) / pi


# @jit
def get_r_s(m_1, rho_s, gamma_s):
    return ((3 - gamma_s) * 0.2 ** (3 - gamma_s) * m_1 / (2 * pi * rho_s)) ** (1 / 3)


# @jit
def get_rho_s(rho_6, m_1, gamma_s):
    a = 0.2
    r_6 = 1e-6 * PC
    m_tilde = ((3 - gamma_s) * a ** (3 - gamma_s)) * m_1 / (2 * jnp.pi)
    return (rho_6 * r_6 ** gamma_s / (m_tilde ** (gamma_s / 3))) ** (
        1 / (1 - gamma_s / 3)
    )


# @jit
def get_rho_6(rho_s, m_1, gamma_s):
    a = 0.2
    r_s = ((3 - gamma_s) * a ** (3 - gamma_s) * m_1 / (2 * pi * rho_s)) ** (1 / 3)
    r_6 = 1e-6 * PC
    return rho_s * (r_6 / r_s) ** -gamma_s


# @jit
def get_xi(gamma_s):
    # Could use that I_x(a, b) = 1 - I_{1-x}(b, a)
    return 1 - betainc(gamma_s - 1 / 2, 3 / 2, 1 / 2)


# @jit
def get_c_f(m_1, m_2, rho_s, gamma_s):
    Lambda = jnp.sqrt(m_1 / m_2)
    M = m_1 + m_2
    c_gw = 64 * G ** 3 * M * m_1 * m_2 / (5 * C ** 5)
    c_df = (
        8
        * pi
        * jnp.sqrt(G)
        * (m_2 / m_1)
        * jnp.log(Lambda)
        * (rho_s * get_r_s(m_1, rho_s, gamma_s) ** gamma_s / jnp.sqrt(M))
        * get_xi(gamma_s)
    )
    return c_df / c_gw * (G * M / pi ** 2) ** ((11 - 2 * gamma_s) / 6)


# @jit
def get_f_eq(gamma_s, c_f):
    return c_f ** (3 / (11 - 2 * gamma_s))


# @jit
def get_a_v(M_chirp):
    return 1 / 16 * (C ** 3 / (pi * G * M_chirp)) ** (5 / 3)


# @jit
def PhiT(f, params: Binary):
    return 2 * pi * f * t_to_c(f, params) - Phi_to_c(f, params)


# @jit
def Psi(f, params: Binary):
    return 2 * pi * f * params.tT_c - params.Phi_c - pi / 4 - PhiT(f, params)


# @jit
def h_0(f, params: Binary):
    return jnp.where(
        f <= params.f_c,
        1
        / 2
        * 4
        * pi ** (2 / 3)
        * (G * params.M_chirp) ** (5 / 3)
        * f ** (2 / 3)
        / C ** 4
        * jnp.sqrt(2 * pi / abs(d2Phi_dt2(f, params))),
        0.0,
    )


# @jit
def amp(f, params: Binary):
    """
    Amplitude averaged over inclination angle.
    """
    return jnp.sqrt(4 / 5) * h_0(f, params) / params.dL


# @jit
def Phi_to_c(f, params: Binary):
    return _Phi_to_c_indef(f, params) - _Phi_to_c_indef(params.f_c, params)


# @jit
def t_to_c(f, params: Binary):
    return _t_to_c_indef(f, params) - _t_to_c_indef(params.f_c, params)


# @jit
def _Phi_to_c_indef(f, params: Binary):
    if isinstance(params, VacuumBinary):
        return _Phi_to_c_indef_v(f, params)
    elif isinstance(params, StaticDress):
        return _Phi_to_c_indef_s(f, params)
    elif isinstance(params, DynamicDress):
        return _Phi_to_c_indef_d(f, params)
    else:
        raise ValueError("unrecognized type")


# @jit
def _t_to_c_indef(f, params: Binary):
    if isinstance(params, VacuumBinary):
        return _t_to_c_indef_v(f, params)
    elif isinstance(params, StaticDress):
        return _t_to_c_indef_s(f, params)
    elif isinstance(params, DynamicDress):
        return _t_to_c_indef_d(f, params)
    else:
        raise ValueError("'params' type is not supported")


# @jit
def d2Phi_dt2(f, params: Binary):
    if isinstance(params, VacuumBinary):
        return d2Phi_dt2_v(f, params)
    elif isinstance(params, StaticDress):
        return d2Phi_dt2_s(f, params)
    elif isinstance(params, DynamicDress):
        return d2Phi_dt2_d(f, params)
    else:
        raise ValueError("'params' type is not supported")


# Vacuum binary
# @jit
def _Phi_to_c_indef_v(f, params: VacuumBinary):
    return get_a_v(params.M_chirp) / f ** (5 / 3)


# @jit
def _t_to_c_indef_v(f, params: VacuumBinary):
    return 5 * get_a_v(params.M_chirp) / (16 * pi * f ** (8 / 3))


# @jit
def d2Phi_dt2_v(f, params: VacuumBinary):
    return 12 * pi ** 2 * f ** (11 / 3) / (5 * get_a_v(params.M_chirp))


# @jit
def make_vacuum_binary(
    m_1,
    m_2,
    Phi_c=jnp.array(0.0),
    t_c=None,
    dL=jnp.array(1e8 * PC),
) -> VacuumBinary:
    M_chirp = get_M_chirp(m_1, m_2)
    tT_c = jnp.array(0.0) if t_c is None else t_c + dL / C
    f_c = get_f_isco(m_1, m_2)
    return VacuumBinary(M_chirp, Phi_c, tT_c, dL, f_c)


# Interpolator for special version of hypergeometric function
def hypgeom_scipy(b, z):
    # print(f"b: {b}, |z| min: {jnp.abs(z).min()}, |z| max: {jnp.abs(z).max()}")
    return hyp2f1(1, b, 1 + b, z)


def get_hypgeom_interps(n_bs=5000, n_zs=4950):
    bs = jnp.linspace(0.5, 1.99, n_bs)
    log10_abs_zs = jnp.linspace(-8, 6, n_zs)
    zs = -(10 ** log10_abs_zs)
    b_mg, z_mg = jnp.meshgrid(bs, zs, indexing="ij")

    vals_pos = jnp.array(hypgeom_scipy(b_mg, z_mg))
    vals_neg = jnp.log10(1 - hypgeom_scipy(-b_mg[::-1, :], z_mg))

    interp_pos = lambda b, z: interp2d(
        b, jnp.log10(-z), bs, log10_abs_zs, vals_pos, jnp.nan
    )
    interp_neg = lambda b, z: 1 - 10 ** interp2d(
        b, jnp.log10(-z), -bs[::-1], log10_abs_zs, vals_neg, jnp.nan
    )
    return interp_pos, interp_neg


interp_pos, interp_neg = get_hypgeom_interps()


def restricted_hypgeom(b, z: jnp.ndarray) -> jnp.ndarray:
    # Assumes b is a scalar
    return jax.lax.cond(
        b > 0, lambda z: interp_pos(b, z), lambda z: interp_neg(b, z), z
    )


@jit
def hypgeom_jax(b, z: jnp.ndarray) -> jnp.ndarray:
    # print(
    #     f"b: {b}, "
    #     f"log10(|z|) min: {jnp.log10(jnp.abs(z)).min()}, "
    #     f"log10(|z|) max: {jnp.log10(jnp.abs(z)).max()}"
    # )
    return jax.lax.cond(
        b == 1, lambda z: jnp.log(1 - z) / (-z), lambda z: restricted_hypgeom(b, z), z
    )


# hypgeom = hypgeom_scipy
hypgeom = hypgeom_jax


# Static
# @jit
def get_th_s(gamma_s):
    return 5 / (11 - 2 * gamma_s)


# @jit
def _Phi_to_c_indef_s(f, params: StaticDress):
    x = f / get_f_eq(params.gamma_s, params.c_f)
    th = get_th_s(params.gamma_s)
    return get_a_v(params.M_chirp) / f ** (5 / 3) * hypgeom(th, -(x ** (-5 / (3 * th))))


# @jit
def _t_to_c_indef_s(f, params: StaticDress):
    th = get_th_s(params.gamma_s)
    return (
        5
        * get_a_v(params.M_chirp)
        / (16 * pi * f ** (8 / 3))
        * hypgeom(th, -params.c_f * f ** ((2 * params.gamma_s - 11) / 3))
    )


# @jit
def d2Phi_dt2_s(f, params: StaticDress):
    return (
        12
        * pi ** 2
        * (f ** (11 / 3) + params.c_f * f ** (2 * params.gamma_s / 3))
        / (5 * get_a_v(params.M_chirp))
    )


# @jit
def make_static_dress(
    m_1,
    m_2,
    rho_6,
    gamma_s,
    Phi_c=jnp.array(0.0),
    t_c=None,
    dL=jnp.array(1e8 * PC),
) -> StaticDress:
    rho_s = get_rho_s(rho_6, m_1, gamma_s)
    c_f = get_c_f(m_1, m_2, rho_s, gamma_s)
    M_chirp = get_M_chirp(m_1, m_2)
    tT_c = jnp.array(0.0) if t_c is None else t_c + dL / C
    f_c = get_f_isco(m_1, m_2)
    return StaticDress(gamma_s, c_f, M_chirp, Phi_c, tT_c, dL, f_c)


# Dynamic
# @jit
def get_f_b(m_1, m_2, gamma_s):
    """
    Gets the break frequency for a dynamic dress. This scaling relation was
    derived from fitting `HaloFeedback` runs.
    """
    beta = 0.8162599280541165
    alpha_1 = 1.441237217113085
    alpha_2 = 0.4511442198433961
    rho = -0.49709119294335674
    gamma_r = 1.4395688575650551

    return (
        beta
        * (m_1 / (1e3 * MSUN)) ** (-alpha_1)
        * (m_2 / MSUN) ** alpha_2
        * (1 + rho * jnp.log(gamma_s / gamma_r))
    )


# @jit
def get_f_b_d(params: DynamicDress):
    """
    Gets the break frequency for a dynamic dress using our scaling relation
    derived from fitting `HaloFeedback` runs.
    """
    m_1 = get_m_1(params.M_chirp, params.q)
    m_2 = get_m_2(params.M_chirp, params.q)
    return get_f_b(m_1, m_2, params.gamma_s)


# @jit
def get_th_d():
    GAMMA_E = 5 / 2
    return 5 / (2 * GAMMA_E)


# @jit
def get_lam(gamma_s):
    GAMMA_E = 5 / 2
    return (11 - 2 * (gamma_s + GAMMA_E)) / 3


# @jit
def get_eta(params: DynamicDress):
    GAMMA_E = 5 / 2
    m_1 = get_m_1(params.M_chirp, params.q)
    m_2 = get_m_2(params.M_chirp, params.q)
    rho_s = get_rho_s(params.rho_6, m_1, params.gamma_s)
    c_f = get_c_f(m_1, m_2, rho_s, params.gamma_s)
    f_eq = get_f_eq(params.gamma_s, c_f)
    f_t = get_f_b_d(params)
    return (
        (5 + 2 * GAMMA_E)
        / (2 * (8 - params.gamma_s))
        * (f_eq / f_t) ** ((11 - 2 * params.gamma_s) / 3)
    )


# @jit
def _Phi_to_c_indef_d(f, params: DynamicDress):
    f_t = get_f_b_d(params)
    x = f / f_t
    th = get_th_d()
    return (
        get_a_v(params.M_chirp)
        / f ** (5 / 3)
        * (
            1
            - get_eta(params)
            * x ** (-get_lam(params.gamma_s))
            * (1 - hypgeom(th, -(x ** (-5 / (3 * th)))))
        )
    )


# @jit
def _t_to_c_indef_d(f, params: DynamicDress):
    f_t = get_f_b_d(params)
    x = f / f_t
    lam = get_lam(params.gamma_s)
    th = get_th_d()
    eta = get_eta(params)
    coeff = (
        get_a_v(params.M_chirp)
        * x ** (-lam)
        / (16 * pi * (1 + lam) * (8 + 3 * lam) * f ** (8 / 3))
    )
    term_1 = 5 * (1 + lam) * (8 + 3 * lam) * x ** lam
    term_2 = 8 * lam * (8 + 3 * lam) * eta * hypgeom(th, -(x ** (-5 / (3 * th))))
    term_3 = (
        -40
        * (1 + lam)
        * eta
        * hypgeom(
            -1 / 5 * th * (8 + 3 * lam),
            -(x ** (5 / (3 * th))),
        )
    )
    term_4 = (
        -8
        * lam
        * eta
        * (
            3
            + 3 * lam
            + 5
            * hypgeom(
                1 / 5 * th * (8 + 3 * lam),
                -(x ** (-5 / (3 * th))),
            )
        )
    )
    return coeff * (term_1 + term_2 + term_3 + term_4)


# @jit
def d2Phi_dt2_d(f, params: DynamicDress):
    f_t = get_f_b_d(params)
    x = f / f_t
    lam = get_lam(params.gamma_s)
    th = get_th_d()
    eta = get_eta(params)
    return (
        12
        * pi ** 2
        * f ** (11 / 3)
        * x ** lam
        * (1 + x ** (5 / (3 * th)))
        / (
            get_a_v(params.M_chirp)
            * (
                5 * x ** lam
                - 5 * eta
                - 3 * eta * lam
                + x ** (5 / (3 * th)) * (5 * x ** lam - 3 * eta * lam)
                + 3
                * (1 + x ** (5 / (3 * th)))
                * eta
                * lam
                * hypgeom(th, -(x ** (-5 / (3 * th))))
            )
        )
    )


# @jit
def make_dynamic_dress(
    m_1,
    m_2,
    rho_6,
    gamma_s,
    Phi_c=jnp.array(0.0),
    t_c=None,
    dL=jnp.array(1e8 * PC),
) -> DynamicDress:
    M_chirp = get_M_chirp(m_1, m_2)
    tT_c = jnp.array(0.0) if t_c is None else t_c + dL / C
    f_c = get_f_isco(m_1, m_2)
    return DynamicDress(gamma_s, rho_6, M_chirp, m_2 / m_1, Phi_c, tT_c, dL, f_c)


def convert(params: Binary, NewType) -> Binary:
    """
    Change binary's type by dropping attributes.
    """
    if isinstance(params, DynamicDress) and NewType is StaticDress:
        m_1 = get_m_1(params.M_chirp, params.q)
        m_2 = get_m_2(params.M_chirp, params.q)
        rho_s = get_rho_s(params.rho_6, m_1, params.gamma_s)
        c_f = get_c_f(m_1, m_2, rho_s, params.gamma_s)
        return StaticDress(
            params.gamma_s,
            c_f,
            params.M_chirp,
            params.Phi_c,
            params.tT_c,
            params.dL,
            params.f_c,
        )
    elif (
        isinstance(params, StaticDress) or isinstance(params, DynamicDress)
    ) and NewType is VacuumBinary:
        return VacuumBinary(**{f: getattr(params, f) for f in VacuumBinary._fields})
    else:
        raise ValueError("invalid conversion")


def get_f_range(dd, t_obs):
    """
    Finds the frequency range [f(-(t_obs + tT_c)), f(-tT_c)].
    """
    # Find frequency t_obs + tT_c before merger
    bracket = (dd.f_c * 0.001, dd.f_c * 1.1)
    fn = lambda f_l: (jax.jit(t_to_c)(f_l, dd) - (t_obs + dd.tT_c)) ** 2
    res = minimize_scalar(fn, bounds=bracket)
    assert res.success
    f_l = res.x

    # Find frequency tT_c before merger
    fn = lambda f_h: (jax.jit(t_to_c)(f_h, dd) - dd.tT_c) ** 2
    res = minimize_scalar(fn, bracket=bracket)
    assert res.success
    f_h = res.x

    return (f_l, f_h)
