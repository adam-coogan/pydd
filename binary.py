from abc import ABC, abstractmethod
from math import pi
from typing import Optional

import jax.numpy as jnp
from jax.scipy.special import betainc
import numpy as np
from scipy.special import hyp2f1


G = 6.67408e-11  # m^3 s^-2 kg^-1
C = 299792458.0  # m/s
MSUN = 1.98855e30  # kg
PC = 3.08567758149137e16  # m
YR = 365.25 * 24 * 3600  # s


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


class Binary(ABC):
    def __init__(self, M_chirp, Phi_c, tT_c, dL_iota):
        self.M_chirp = M_chirp
        self.Phi_c = Phi_c
        self.tT_c = tT_c
        self.dL_iota = dL_iota
        self.psi_v = 1 / 16 * (C ** 3 / (pi * G * self.M_chirp)) ** (5 / 3)

    def PhiT(self, f, f_c):
        return 2 * pi * f * self.t_to_c(f, f_c) - self.Phi_to_c(f, f_c)

    def Psi(self, f, f_c):
        return 2 * pi * f * self.tT_c - self.Phi_c - pi / 4 - self.PhiT(f, f_c)

    def h_0(self, f):
        return (
            1
            / 2
            * 4
            * pi ** (2 / 3)
            * (G * self.M_chirp) ** (5 / 3)
            * f ** (2 / 3)
            / C ** 4
            * jnp.sqrt(2 * pi / abs(self.d2Phi_dt2(f)))
        )

    def amp_plus(self, f):
        return self.h_0(f) * jnp.exp(self.dL_iota)

    def Phi_to_c(self, f, f_c):
        return self._Phi_to_c_indef(f) - self._Phi_to_c_indef(f_c)

    def t_to_c(self, f, f_c):
        return self._t_to_c_indef(f) - self._t_to_c_indef(f_c)

    @abstractmethod
    def _Phi_to_c_indef(self, f):
        ...

    @abstractmethod
    def _t_to_c_indef(self, f):
        ...

    @abstractmethod
    def d2Phi_dt2(self, f):
        ...


class VacuumBinary(Binary):
    def _Phi_to_c_indef(self, f):
        return self.psi_v / f ** (5 / 3)

    def _t_to_c_indef(self, f):
        return 5 * self.psi_v / (16 * pi * f ** (8 / 3))

    def d2Phi_dt2(self, f):
        return 12 * pi ** 2 * f ** (11 / 3) / (5 * self.psi_v)

    @classmethod
    def make(
        cls,
        m_1,
        m_2,
        Phi_c=0.0,
        t_c: Optional[float] = None,
        dL=1e8 * PC,
        iota=0,
    ):
        M_chirp = get_M_chirp(m_1, m_2)
        tT_c = 0.0 if t_c is None else t_c + dL / C
        dL_iota = get_dL_iota(dL, iota)
        return cls(M_chirp, Phi_c, tT_c, dL_iota)


class StaticDress(Binary):
    def __init__(
        self,
        gamma_s,
        c_f,
        M_chirp,
        Phi_c,
        tT_c,
        dL_iota,
    ):
        super().__init__(M_chirp, Phi_c, tT_c, dL_iota)
        self.gamma_s = gamma_s
        self.c_f = c_f
        self.f_eq = get_f_eq(self.gamma_s, self.c_f)
        self.th = 5 / (11 - 2 * self.gamma_s)

    def _Phi_to_c_indef(self, f):
        x = f / self.f_eq
        return (
            self.psi_v
            / f ** (5 / 3)
            * hyp2f1(1, self.th, 1 + self.th, -(x ** (-5 / (3 * self.th))))
        )

    def _t_to_c_indef(self, f):
        return (
            5
            * self.psi_v
            / (16 * pi * f ** (8 / 3))
            * hyp2f1(
                1, self.th, 1 + self.th, -self.c_f * f ** ((2 * self.gamma_s - 11) / 3)
            )
        )

    def d2Phi_dt2(self, f):
        return (
            12
            * pi ** 2
            * (f ** (11 / 3) + self.c_f * f ** (2 * self.gamma_s / 3))
            / (5 * self.psi_v)
        )

    @classmethod
    def make(
        cls,
        m_1,
        m_2,
        rho_s,
        gamma_s,
        Phi_c=0.0,
        t_c: Optional[float] = None,
        dL=1e8 * PC,
        iota=0,
    ):
        c_f = get_c_f(m_1, m_2, rho_s, gamma_s)
        M_chirp = get_M_chirp(m_1, m_2)
        tT_c = 0.0 if t_c is None else t_c + dL / C
        dL_iota = get_dL_iota(dL, iota)
        return cls(gamma_s, c_f, M_chirp, Phi_c, tT_c, dL_iota)


class DynamicDress(Binary):
    def __init__(
        self,
        gamma_s,
        c_f,
        M_chirp,
        q,
        Phi_c,
        tT_c,
        dL_iota,
    ):
        super().__init__(M_chirp, Phi_c, tT_c, dL_iota)
        self.gamma_s = gamma_s
        self.c_f = c_f
        self.q = q
        self.f_eq = get_f_eq(self.gamma_s, self.c_f)

        self.m_1 = (1 + self.q) ** (1 / 5) / self.q ** (3 / 5) * self.M_chirp
        self.m_2 = (1 + self.q) ** (1 / 5) * self.q ** (2 / 5) * self.M_chirp

        # Hypergeometric waveform parameters
        self.f_t = get_f_b(self.m_1, self.m_2, self.gamma_s)
        self.gamma_e = 5 / 2
        self.th = 5 / (2 * self.gamma_e)
        self.lam = (11 - 2 * (self.gamma_s + self.gamma_e)) / 3
        self.eta = (
            (5 + 2 * self.gamma_e)
            / (2 * (8 - self.gamma_s))
            * (self.f_eq / self.f_t) ** ((11 - 2 * self.gamma_s) / 3)
        )

    def _Phi_to_c_indef(self, f):
        x = f / self.f_t
        return (
            self.psi_v
            / f ** (5 / 3)
            * (
                1
                - self.eta
                * x ** (-self.lam)
                * (1 - hyp2f1(1, self.th, 1 + self.th, -(x ** (-5 / (3 * self.th)))))
            )
        )

    def _t_to_c_indef(self, f):
        x = f / self.f_t
        coeff = (
            self.psi_v
            * x ** (-self.lam)
            / (16 * pi * (1 + self.lam) * (8 + 3 * self.lam) * f ** (8 / 3))
        )
        term_1 = 5 * (1 + self.lam) * (8 + 3 * self.lam) * x ** self.lam
        term_2 = (
            8
            * self.lam
            * (8 + 3 * self.lam)
            * self.eta
            * hyp2f1(1, self.th, 1 + self.th, -(x ** (-5 / (3 * self.th))))
        )
        term_3 = (
            -40
            * (1 + self.lam)
            * self.eta
            * hyp2f1(
                1,
                -1 / 5 * self.th * (8 + 3 * self.lam),
                1 - 1 / 5 * self.th * (8 + 3 * self.th),
                -(x ** (5 / (3 * self.th))),
            )
        )
        term_4 = (
            -8
            * self.lam
            * self.eta
            * (
                3
                + 3 * self.lam
                + 5
                * hyp2f1(
                    1,
                    1 / 5 * self.th * (8 + 3 * self.lam),
                    1 + 1 / 5 * self.th * (8 + 3 * self.lam),
                    -(x ** (-5 / (3 * self.th))),
                )
            )
        )
        return coeff * (term_1 + term_2 + term_3 + term_4)

    def d2Phi_dt2(self, f):
        x = f / self.f_t
        return (
            12
            * pi ** 2
            * f ** (11 / 3)
            * x ** self.lam
            * (1 + x ** (5 / (3 * self.th)))
            / (
                self.psi_v
                * (
                    5 * x ** self.lam
                    - 5 * self.eta
                    - 3 * self.eta * self.lam
                    + x ** (5 / (3 * self.th))
                    * (5 * x ** self.lam - 3 * self.eta * self.lam)
                    + 3
                    * (1 + x ** (5 / (3 * self.th)))
                    * self.eta
                    * self.lam
                    * hyp2f1(1, self.th, 1 + self.th, -(x ** (-5 / (3 * self.th))))
                )
            )
        )

    @classmethod
    def make(
        cls,
        m_1,
        m_2,
        rho_s,
        gamma_s,
        Phi_c=0.0,
        t_c: Optional[float] = None,
        dL=1e8 * PC,
        iota=0,
    ):
        c_f = get_c_f(m_1, m_2, rho_s, gamma_s)
        M_chirp = get_M_chirp(m_1, m_2)
        tT_c = 0.0 if t_c is None else t_c + dL / C
        dL_iota = get_dL_iota(dL, iota)
        return cls(gamma_s, c_f, M_chirp, m_2 / m_1, Phi_c, tT_c, dL_iota)
