from math import pi
from typing import NamedTuple, Union

import jax
from jax import jit
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
    f_c: jnp.ndarray


class StaticDress(NamedTuple):
    gamma_s: jnp.ndarray
    c_f: jnp.ndarray
    M_chirp: jnp.ndarray
    Phi_c: jnp.ndarray
    tT_c: jnp.ndarray
    dL_iota: jnp.ndarray
    f_c: jnp.ndarray


class DynamicDress(NamedTuple):
    gamma_s: jnp.ndarray
    c_f: jnp.ndarray
    M_chirp: jnp.ndarray
    q: jnp.ndarray
    Phi_c: jnp.ndarray
    tT_c: jnp.ndarray
    dL_iota: jnp.ndarray
    f_c: jnp.ndarray


class HypGeomDress(NamedTuple):
    lam: jnp.ndarray
    eta: jnp.ndarray
    M_chirp: jnp.ndarray
    f_b: jnp.ndarray
    Phi_c: jnp.ndarray
    tT_c: jnp.ndarray
    dL_iota: jnp.ndarray
    f_c: jnp.ndarray


Binary = Union[VacuumBinary, StaticDress, DynamicDress, HypGeomDress]


@jit
def get_M_chirp(m_1, m_2):
    return (m_1 * m_2) ** (3 / 5) / (m_1 + m_2) ** (1 / 5)


@jit
def get_m_1(M_chirp, q):
    return (1 + q) ** (1 / 5) / q ** (3 / 5) * M_chirp


@jit
def get_m_2(M_chirp, q):
    return (1 + q) ** (1 / 5) * q ** (2 / 5) * M_chirp


@jit
def get_r_isco(m_1):
    return 6 * G * m_1 / C ** 2


@jit
def get_f_isco(m_1):
    return jnp.sqrt(G * m_1 / get_r_isco(m_1) ** 3) / pi


@jit
def get_r_s(m_1, rho_s, gamma_s):
    return ((3 - gamma_s) * 0.2 ** (3 - gamma_s) * m_1 / (2 * pi * rho_s)) ** (1 / 3)


@jit
def get_rho_s(rho_6, m_1, gamma_s):
    r_6 = 1e-6 * PC
    m_tilde = ((3 - gamma_s) * (0.2) ** (3 - gamma_s)) * m_1 / (2 * jnp.pi)
    return (rho_6 * r_6 ** gamma_s / (m_tilde ** (gamma_s / 3))) ** (
        1 / (1 - gamma_s / 3)
    )


@jit
def get_xi(gamma_s):
    # Could use that I_x(a, b) = 1 - I_{1-x}(b, a)
    return 1 - betainc(gamma_s - 1 / 2, 3 / 2, 1 / 2)


@jit
def get_dL_iota(dL, iota):
    return jnp.log((1 + jnp.cos(iota) ** 2) / (2 * dL))


@jit
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


@jit
def get_f_eq(gamma_s, c_f):
    return c_f ** (3 / (11 - 2 * gamma_s))


@jit
def get_psi_v(M_chirp):
    return 1 / 16 * (C ** 3 / (pi * G * M_chirp)) ** (5 / 3)


@jit
def PhiT(f, params: Binary):
    return 2 * pi * f * t_to_c(f, params) - Phi_to_c(f, params)


@jit
def Psi(f, params: Binary):
    return 2 * pi * f * params.tT_c - params.Phi_c - pi / 4 - PhiT(f, params)


@jit
def h_0(f, params: Binary):
    return (
        1
        / 2
        * 4
        * pi ** (2 / 3)
        * (G * params.M_chirp) ** (5 / 3)
        * f ** (2 / 3)
        / C ** 4
        * jnp.sqrt(2 * pi / abs(d2Phi_dt2(f, params)))
    )


@jit
def amp_plus(f, params: Binary):
    return h_0(f, params) * jnp.exp(params.dL_iota)


@jit
def Phi_to_c(f, params: Binary):
    return _Phi_to_c_indef(f, params) - _Phi_to_c_indef(params.f_c, params)


@jit
def t_to_c(f, params: Binary):
    return _t_to_c_indef(f, params) - _t_to_c_indef(params.f_c, params)


@jit
def _Phi_to_c_indef(f, params: Binary):
    if isinstance(params, VacuumBinary):
        return _Phi_to_c_indef_v(f, params)
    elif isinstance(params, StaticDress):
        return _Phi_to_c_indef_s(f, params)
    elif isinstance(params, DynamicDress):
        return _Phi_to_c_indef_d(f, params)
    elif isinstance(params, HypGeomDress):
        return _Phi_to_c_indef_h(f, params)
    else:
        raise ValueError("unrecognized type")


@jit
def _t_to_c_indef(f, params: Binary):
    if isinstance(params, VacuumBinary):
        return _t_to_c_indef_v(f, params)
    elif isinstance(params, StaticDress):
        return _t_to_c_indef_s(f, params)
    elif isinstance(params, DynamicDress):
        return _t_to_c_indef_d(f, params)
    elif isinstance(params, HypGeomDress):
        return _t_to_c_indef_h(f, params)
    else:
        raise ValueError("'params' type is not supported")


@jit
def d2Phi_dt2(f, params: Binary):
    if isinstance(params, VacuumBinary):
        return d2Phi_dt2_v(f, params)
    elif isinstance(params, StaticDress):
        return d2Phi_dt2_s(f, params)
    elif isinstance(params, DynamicDress):
        return d2Phi_dt2_d(f, params)
    elif isinstance(params, HypGeomDress):
        return d2Phi_dt2_h(f, params)
    else:
        raise ValueError("'params' type is not supported")


# Vacuum binary
@jit
def _Phi_to_c_indef_v(f, params: VacuumBinary):
    return get_psi_v(params.M_chirp) / f ** (5 / 3)


@jit
def _t_to_c_indef_v(f, params: VacuumBinary):
    return 5 * get_psi_v(params.M_chirp) / (16 * pi * f ** (8 / 3))


@jit
def d2Phi_dt2_v(f, params: VacuumBinary):
    return 12 * pi ** 2 * f ** (11 / 3) / (5 * get_psi_v(params.M_chirp))


@jit
def make_vacuum_binary(
    m_1,
    m_2,
    Phi_c=jnp.array(0.0),
    t_c=None,
    dL=jnp.array(1e8 * PC),
    iota=jnp.array(0.0),
) -> VacuumBinary:
    M_chirp = get_M_chirp(m_1, m_2)
    tT_c = jnp.array(0.0) if t_c is None else t_c + dL / C
    dL_iota = get_dL_iota(dL, iota)
    f_c = get_f_isco(m_1)
    return VacuumBinary(M_chirp, Phi_c, tT_c, dL_iota, f_c)


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
@jit
def get_th_s(gamma_s):
    return 5 / (11 - 2 * gamma_s)


@jit
def _Phi_to_c_indef_s(f, params: StaticDress):
    x = f / get_f_eq(params.gamma_s, params.c_f)
    th = get_th_s(params.gamma_s)
    return (
        get_psi_v(params.M_chirp) / f ** (5 / 3) * hypgeom(th, -(x ** (-5 / (3 * th))))
    )


@jit
def _t_to_c_indef_s(f, params: StaticDress):
    th = get_th_s(params.gamma_s)
    return (
        5
        * get_psi_v(params.M_chirp)
        / (16 * pi * f ** (8 / 3))
        * hypgeom(th, -params.c_f * f ** ((2 * params.gamma_s - 11) / 3))
    )


@jit
def d2Phi_dt2_s(f, params: StaticDress):
    return (
        12
        * pi ** 2
        * (f ** (11 / 3) + params.c_f * f ** (2 * params.gamma_s / 3))
        / (5 * get_psi_v(params.M_chirp))
    )


@jit
def make_static_dress(
    m_1,
    m_2,
    rho_s,
    gamma_s,
    Phi_c=jnp.array(0.0),
    t_c=None,
    dL=jnp.array(1e8 * PC),
    iota=jnp.array(0.0),
) -> StaticDress:
    c_f = get_c_f(m_1, m_2, rho_s, gamma_s)
    M_chirp = get_M_chirp(m_1, m_2)
    tT_c = jnp.array(0.0) if t_c is None else t_c + dL / C
    dL_iota = get_dL_iota(dL, iota)
    f_c = get_f_isco(m_1)
    return StaticDress(gamma_s, c_f, M_chirp, Phi_c, tT_c, dL_iota, f_c)


# Dynamic
@jit
def get_f_b(m_1, m_2, gamma_s):
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


@jit
def get_f_b_d(params: DynamicDress):
    m_1 = get_m_1(params.M_chirp, params.q)
    m_2 = get_m_2(params.M_chirp, params.q)
    return get_f_b(m_1, m_2, params.gamma_s)


@jit
def get_th_d():
    GAMMA_E = 5 / 2
    return 5 / (2 * GAMMA_E)


@jit
def get_lam(gamma_s):
    GAMMA_E = 5 / 2
    return (11 - 2 * (gamma_s + GAMMA_E)) / 3


@jit
def get_eta(params: DynamicDress):
    GAMMA_E = 5 / 2
    f_eq = get_f_eq(params.gamma_s, params.c_f)
    f_t = get_f_b_d(params)
    return (
        (5 + 2 * GAMMA_E)
        / (2 * (8 - params.gamma_s))
        * (f_eq / f_t) ** ((11 - 2 * params.gamma_s) / 3)
    )


@jit
def _Phi_to_c_indef_d(f, params: DynamicDress):
    f_t = get_f_b_d(params)
    x = f / f_t
    th = get_th_d()
    return (
        get_psi_v(params.M_chirp)
        / f ** (5 / 3)
        * (
            1
            - get_eta(params)
            * x ** (-get_lam(params.gamma_s))
            * (1 - hypgeom(th, -(x ** (-5 / (3 * th)))))
        )
    )


@jit
def _t_to_c_indef_d(f, params: DynamicDress):
    f_t = get_f_b_d(params)
    x = f / f_t
    lam = get_lam(params.gamma_s)
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


@jit
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


@jit
def make_dynamic_dress(
    m_1,
    m_2,
    rho_s,
    gamma_s,
    Phi_c=jnp.array(0.0),
    t_c=None,
    dL=jnp.array(1e8 * PC),
    iota=jnp.array(0.0),
) -> DynamicDress:
    c_f = get_c_f(m_1, m_2, rho_s, gamma_s)
    M_chirp = get_M_chirp(m_1, m_2)
    tT_c = jnp.array(0.0) if t_c is None else t_c + dL / C
    dL_iota = get_dL_iota(dL, iota)
    f_c = get_f_isco(m_1)
    return DynamicDress(gamma_s, c_f, M_chirp, m_2 / m_1, Phi_c, tT_c, dL_iota, f_c)


def convert(params: Binary, NewType) -> Binary:
    """
    Change binary's type by dropping attributes.
    """
    if isinstance(params, StaticDress) and NewType is DynamicDress:
        raise ValueError("cannot convert a StaticDress to a DynamicDress")
    elif isinstance(params, VacuumBinary) and NewType in [StaticDress, DynamicDress]:
        raise ValueError(
            "cannot convert a StaticDress to a StaticDress or DynamicDress"
        )
    elif isinstance(params, HypGeomDress) or NewType is HypGeomDress:
        raise ValueError("conversion to/from HypGeomDress not supported yet")

    return NewType(**{f: getattr(params, f) for f in NewType._fields})


# General hypergeometric
@jit
def _Phi_to_c_indef_h(f, params: HypGeomDress):
    x = f / params.f_b
    th = get_th_d()
    return (
        get_psi_v(params.M_chirp)
        / f ** (5 / 3)
        * (
            1
            - params.eta
            * x ** (-params.lam)
            * (1 - hypgeom(th, -(x ** (-5 / (3 * th)))))
        )
    )


@jit
def _t_to_c_indef_h(f, params: HypGeomDress):
    x = f / params.f_b
    th = get_th_d()
    coeff = (
        get_psi_v(params.M_chirp)
        * x ** (-params.lam)
        / (16 * pi * (1 + params.lam) * (8 + 3 * params.lam) * f ** (8 / 3))
    )
    term_1 = 5 * (1 + params.lam) * (8 + 3 * params.lam) * x ** params.lam
    term_2 = (
        8
        * params.lam
        * (8 + 3 * params.lam)
        * params.eta
        * hypgeom(th, -(x ** (-5 / (3 * th))))
    )
    term_3 = (
        -40
        * (1 + params.lam)
        * params.eta
        * hypgeom(
            -1 / 5 * th * (8 + 3 * params.lam),
            -(x ** (5 / (3 * th))),
        )
    )
    term_4 = (
        -8
        * params.lam
        * params.eta
        * (
            3
            + 3 * params.lam
            + 5
            * hypgeom(
                1 / 5 * th * (8 + 3 * params.lam),
                -(x ** (-5 / (3 * th))),
            )
        )
    )
    return coeff * (term_1 + term_2 + term_3 + term_4)


@jit
def d2Phi_dt2_h(f, params: HypGeomDress):
    x = f / params.f_b
    th = get_th_d()
    return (
        12
        * pi ** 2
        * f ** (11 / 3)
        * x ** params.lam
        * (1 + x ** (5 / (3 * th)))
        / (
            get_psi_v(params.M_chirp)
            * (
                5 * x ** params.lam
                - 5 * params.eta
                - 3 * params.eta * params.lam
                + x ** (5 / (3 * th))
                * (5 * x ** params.lam - 3 * params.eta * params.lam)
                + 3
                * (1 + x ** (5 / (3 * th)))
                * params.eta
                * params.lam
                * hypgeom(th, -(x ** (-5 / (3 * th))))
            )
        )
    )


@jit
def make_hypgeom_dress(
    m_1,
    m_2,
    rho_s,
    gamma_s,
    Phi_c=jnp.array(0.0),
    t_c=None,
    dL=jnp.array(1e8 * PC),
    iota=jnp.array(0.0),
) -> HypGeomDress:
    GAMMA_E = jnp.array(5 / 2)
    lam = (11 - 2 * (gamma_s + GAMMA_E)) / 3
    c_f = get_c_f(m_1, m_2, rho_s, gamma_s)
    f_eq = get_f_eq(gamma_s, c_f)
    f_b = get_f_b(m_1, m_2, gamma_s)
    eta = (
        (5 + 2 * GAMMA_E)
        / (2 * (8 - gamma_s))
        * (f_eq / f_b) ** ((11 - 2 * gamma_s) / 3)
    )
    M_chirp = get_M_chirp(m_1, m_2)

    tT_c = jnp.array(0.0) if t_c is None else t_c + dL / C
    dL_iota = get_dL_iota(dL, iota)
    f_c = get_f_isco(m_1)

    return HypGeomDress(lam, eta, M_chirp, f_b, Phi_c, tT_c, dL_iota, f_c)
