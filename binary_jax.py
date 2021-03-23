from math import pi
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import betainc
from jaxinterp2d import interp2d
from scipy.special import hyp2f1


G = 6.67408e-11  # m^3 s^-2 kg^-1
C = 299792458.0  # m/s
MSUN = 1.98855e30  # kg
PC = 3.08567758149137e16  # m
YR = 365.25 * 24 * 3600  # s


class VacuumBinary(NamedTuple):
    M_chirp: jnp.ndarray
    Phi_c: jnp.ndarray
    tT_c: jnp.ndarray
    dL_iota: jnp.ndarray


class StaticDress(NamedTuple):
    gamma_s: NamedTuple
    c_f: NamedTuple
    M_chirp: NamedTuple
    Phi_c: NamedTuple
    tT_c: NamedTuple
    dL_iota: NamedTuple


class DynamicDress(NamedTuple):
    gamma_s: NamedTuple
    c_f: NamedTuple
    M_chirp: NamedTuple
    q: NamedTuple
    Phi_c: NamedTuple
    tT_c: NamedTuple
    dL_iota: NamedTuple


def get_M_chirp(m_1, m_2):
    return (m_1 * m_2) ** (3 / 5) / (m_1 + m_2) ** (1 / 5)


def get_m_1(M_chirp, q):
    return (1 + q) ** (1 / 5) / q ** (3 / 5) * M_chirp


def get_m_2(M_chirp, q):
    return (1 + q) ** (1 / 5) * q ** (2 / 5) * M_chirp


def get_r_isco(m_1):
    return 6 * G * m_1 / C ** 2


def get_f_isco(m_1):
    return jnp.sqrt(G * m_1 / get_r_isco(m_1) ** 3) / pi


def get_r_s(m_1, rho_s, gamma_s):
    return ((3 - gamma_s) * 0.2 ** (3 - gamma_s) * m_1 / (2 * pi * rho_s)) ** (1 / 3)


def get_xi(gamma_s):
    # Could use that I_x(a, b) = 1 - I_{1-x}(b, a)
    return 1 - betainc(gamma_s - 1 / 2, 3 / 2, 1 / 2)


def get_dL_iota(dL, iota):
    return jnp.log((1 + jnp.cos(iota) ** 2) / (2 * dL))


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


def get_f_eq(gamma_s, c_f):
    return c_f ** (3 / (11 - 2 * gamma_s))


def get_psi_v(M_chirp):
    return 1 / 16 * (C ** 3 / (pi * G * M_chirp)) ** (5 / 3)


def PhiT(f, f_c, params, kind: str):
    return 2 * pi * f * t_to_c(f, f_c, params, kind) - Phi_to_c(f, f_c, params, kind)


def Psi(f, f_c, params, kind: str):
    return 2 * pi * f * params.tT_c - params.Phi_c - pi / 4 - PhiT(f, f_c, params, kind)


def h_0(f, params, kind: str):
    return (
        1
        / 2
        * 4
        * pi ** (2 / 3)
        * (G * params.M_chirp) ** (5 / 3)
        * f ** (2 / 3)
        / C ** 4
        * jnp.sqrt(2 * pi / abs(d2Phi_dt2(f, params, kind)))
    )


def amp_plus(f, params, kind: str):
    return h_0(f, params, kind) * jnp.exp(params.dL_iota)


def Phi_to_c(f, f_c, params, kind: str):
    return _Phi_to_c_indef(f, params, kind) - _Phi_to_c_indef(f_c, params, kind)


def t_to_c(f, f_c, params, kind: str):
    return _t_to_c_indef(f, params, kind) - _t_to_c_indef(f_c, params, kind)


def _Phi_to_c_indef(f, params, kind):
    if kind == "v":
        return _Phi_to_c_indef_v(f, params)
    elif kind == "s":
        return _Phi_to_c_indef_s(f, params)
    elif kind == "d":
        return _Phi_to_c_indef_d(f, params)
    else:
        raise ValueError("invalid 'kind'")


def _t_to_c_indef(f, params, kind):
    if kind == "v":
        return _t_to_c_indef_v(f, params)
    elif kind == "s":
        return _t_to_c_indef_s(f, params)
    elif kind == "d":
        return _t_to_c_indef_d(f, params)
    else:
        raise ValueError("invalid 'kind'")


def d2Phi_dt2(f, params, kind):
    if kind == "v":
        return d2Phi_dt2_v(f, params)
    elif kind == "s":
        return d2Phi_dt2_s(f, params)
    elif kind == "d":
        return d2Phi_dt2_d(f, params)
    else:
        raise ValueError("invalid 'kind'")


# Vacuum binary
def _Phi_to_c_indef_v(f, params):
    return get_psi_v(params.M_chirp) / f ** (5 / 3)


def _t_to_c_indef_v(f, params):
    return 5 * get_psi_v(params.M_chirp) / (16 * pi * f ** (8 / 3))


def d2Phi_dt2_v(f, params):
    return 12 * pi ** 2 * f ** (11 / 3) / (5 * get_psi_v(params.M_chirp))


def make_vacuum_binary(
    m_1,
    m_2,
    Phi_c=0.0,
    t_c=None,
    dL=1e8 * PC,
    iota=0,
) -> VacuumBinary:
    M_chirp = get_M_chirp(m_1, m_2)
    tT_c = 0.0 if t_c is None else t_c + dL / C
    dL_iota = get_dL_iota(dL, iota)
    return VacuumBinary(M_chirp, Phi_c, tT_c, dL_iota)


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

    log10_abs_zs = jnp.log10(-zs)
    interp_pos = lambda b, z: interp2d(b, jnp.log10(-z), bs, log10_abs_zs, vals_pos)
    interp_neg = lambda b, z: 1 - 10 ** interp2d(
        b, jnp.log10(-z), -bs[::-1], log10_abs_zs, vals_neg
    )
    return interp_pos, interp_neg


_interp_pos, _interp_neg = get_hypgeom_interps()


def _restricted_hypgeom(b, z: jnp.ndarray) -> jnp.ndarray:
    # Assumes b is a scalar
    return jax.lax.cond(
        b > 0, lambda z: _interp_pos(b, z), lambda z: _interp_neg(b, z), z
    )


def hypgeom_jax(b, z: jnp.ndarray) -> jnp.ndarray:
    # print(
    #     f"b: {b}, "
    #     f"log10(|z|) min: {jnp.log10(jnp.abs(z)).min()}, "
    #     f"log10(|z|) max: {jnp.log10(jnp.abs(z)).max()}"
    # )
    return jax.lax.cond(
        b == 1, lambda z: jnp.log(1 - z) / (-z), lambda z: _restricted_hypgeom(b, z), z
    )


# hypgeom = hypgeom_scipy
hypgeom = hypgeom_jax


# Static
def get_th_s(gamma_s):
    return 5 / (11 - 2 * gamma_s)


def _Phi_to_c_indef_s(f, params):
    x = f / get_f_eq(params.gamma_s, params.c_f)
    th = get_th_s(params.gamma_s)
    return (
        get_psi_v(params.M_chirp) / f ** (5 / 3) * hypgeom(th, -(x ** (-5 / (3 * th))))
    )


def _t_to_c_indef_s(f, params):
    th = get_th_s(params.gamma_s)
    return (
        5
        * get_psi_v(params.M_chirp)
        / (16 * pi * f ** (8 / 3))
        * hypgeom(th, -params.c_f * f ** ((2 * params.gamma_s - 11) / 3))
    )


def d2Phi_dt2_s(f, params):
    return (
        12
        * pi ** 2
        * (f ** (11 / 3) + params.c_f * f ** (2 * params.gamma_s / 3))
        / (5 * get_psi_v(params.M_chirp))
    )


def make_static_dress(
    m_1,
    m_2,
    rho_s,
    gamma_s,
    Phi_c=0.0,
    t_c=None,
    dL=1e8 * PC,
    iota=0,
):
    c_f = get_c_f(m_1, m_2, rho_s, gamma_s)
    M_chirp = get_M_chirp(m_1, m_2)
    tT_c = 0.0 if t_c is None else t_c + dL / C
    dL_iota = get_dL_iota(dL, iota)
    return StaticDress(gamma_s, c_f, M_chirp, Phi_c, tT_c, dL_iota)


# Dynamic
GAMMA_E = 5 / 2


def get_f_b(params):
    m_1 = (1 + params.q) ** (1 / 5) / params.q ** (3 / 5) * params.M_chirp
    m_2 = (1 + params.q) ** (1 / 5) * params.q ** (2 / 5) * params.M_chirp

    beta = 0.8162599280541165
    alpha_1 = 1.441237217113085
    alpha_2 = 0.4511442198433961
    rho = -0.49709119294335674
    gamma_r = 1.4395688575650551

    return (
        beta
        * (m_1 / (1e3 * MSUN)) ** (-alpha_1)
        * (m_2 / MSUN) ** alpha_2
        * (1 + rho * jnp.log(params.gamma_s / gamma_r))
    )


def get_th_d():
    return 5 / (2 * GAMMA_E)


def get_lam(params):
    return (11 - 2 * (params.gamma_s + GAMMA_E)) / 3


def get_eta(params):
    f_eq = get_f_eq(params.gamma_s, params.c_f)
    f_t = get_f_b(params)
    return (
        (5 + 2 * GAMMA_E)
        / (2 * (8 - params.gamma_s))
        * (f_eq / f_t) ** ((11 - 2 * params.gamma_s) / 3)
    )


def _Phi_to_c_indef_d(f, params):
    f_t = get_f_b(params)
    x = f / f_t
    th = get_th_d()
    return (
        get_psi_v(params.M_chirp)
        / f ** (5 / 3)
        * (
            1
            - get_eta(params)
            * x ** (-get_lam(params))
            * (1 - hypgeom(th, -(x ** (-5 / (3 * th)))))
        )
    )


def _t_to_c_indef_d(f, params):
    f_t = get_f_b(params)
    x = f / f_t
    lam = get_lam(params)
    th = get_th_d()
    eta = get_eta(params)
    coeff = (
        get_psi_v(params.M_chirp)
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


def d2Phi_dt2_d(f, params):
    f_t = get_f_b(params)
    x = f / f_t
    lam = get_lam(params)
    th = get_th_d()
    eta = get_eta(params)
    return (
        12
        * pi ** 2
        * f ** (11 / 3)
        * x ** lam
        * (1 + x ** (5 / (3 * th)))
        / (
            get_psi_v(params.M_chirp)
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


# OLD
def make_dynamic_dress(
    m_1,
    m_2,
    rho_s,
    gamma_s,
    Phi_c=jnp.array(0.0),
    t_c=None,
    dL=jnp.array(1e8 * PC),
    iota=jnp.array(0.0),
):
    c_f = get_c_f(m_1, m_2, rho_s, gamma_s)
    M_chirp = get_M_chirp(m_1, m_2)
    tT_c = jnp.array(0.0) if t_c is None else t_c + dL / C
    dL_iota = get_dL_iota(dL, iota)
    return DynamicDress(gamma_s, c_f, M_chirp, m_2 / m_1, Phi_c, tT_c, dL_iota)
