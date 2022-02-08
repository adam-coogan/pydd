from abc import ABC, abstractmethod
from math import pi
from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.special import betainc
from scipy.special import hyp2f1

Array = NDArray[np.float64]
ArrOrFloat = Union[float, Array]


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


def hypgeom(b: float, z: ArrOrFloat) -> ArrOrFloat:
    if b == 1:
        return np.log(1 - z) / (-z)
    else:
        return hyp2f1(1, b, 1 + b, z)


def get_M_chirp(m_1: float, m_2: float) -> float:
    return (m_1 * m_2) ** (3 / 5) / (m_1 + m_2) ** (1 / 5)


def get_m_1(M_chirp: float, q: float) -> float:
    return (1 + q) ** (1 / 5) / q ** (3 / 5) * M_chirp


def get_m_2(M_chirp: float, q: float) -> float:
    return (1 + q) ** (1 / 5) * q ** (2 / 5) * M_chirp


def get_r_isco(m_1: float) -> float:
    return 6 * G * m_1 / C ** 2


def get_f_isco(m_1: float) -> float:
    return np.sqrt(G * m_1 / get_r_isco(m_1) ** 3) / pi


def get_r_s(m_1: float, rho_s: float, gamma_s: float) -> float:
    return ((3 - gamma_s) * 0.2 ** (3 - gamma_s) * m_1 / (2 * pi * rho_s)) ** (1 / 3)


def get_rho_s(rho_6: float, m_1: float, gamma_s: float) -> float:
    a = 0.2
    r_6 = 1e-6 * PC
    m_tilde = ((3 - gamma_s) * a ** (3 - gamma_s)) * m_1 / (2 * pi)
    return (rho_6 * r_6 ** gamma_s / (m_tilde ** (gamma_s / 3))) ** (
        1 / (1 - gamma_s / 3)
    )


def get_rho_6(rho_s: float, m_1: float, gamma_s: float) -> float:
    a = 0.2
    r_s = ((3 - gamma_s) * a ** (3 - gamma_s) * m_1 / (2 * pi * rho_s)) ** (1 / 3)
    r_6 = 1e-6 * PC
    return rho_s * (r_6 / r_s) ** -gamma_s


def get_xi(gamma_s: float) -> float:
    # Could use that I_x(a, b) = 1 - I_{1-x}(b, a)
    return 1 - betainc(gamma_s - 1 / 2, 3 / 2, 1 / 2)


def get_c_f(m_1: float, m_2: float, rho_s: float, gamma_s: float) -> float:
    Lambda = np.sqrt(m_1 / m_2)
    M = m_1 + m_2
    c_gw = 64 * G ** 3 * M * m_1 * m_2 / (5 * C ** 5)
    c_df = (
        8
        * pi
        * np.sqrt(G)
        * (m_2 / m_1)
        * np.log(Lambda)
        * (rho_s * get_r_s(m_1, rho_s, gamma_s) ** gamma_s / np.sqrt(M))
        * get_xi(gamma_s)
    )
    return c_df / c_gw * (G * M / pi ** 2) ** ((11 - 2 * gamma_s) / 6)


def get_f_eq(gamma_s: float, c_f: float) -> float:
    return c_f ** (3 / (11 - 2 * gamma_s))


def get_f_b(M_chirp: float, q: float, gamma_s: float) -> float:
    """
    Gets the break frequency for a dynamic dress. This scaling relation was
    derived from fitting `HaloFeedback` runs.
    """
    beta = 0.8162599280541165
    alpha_1 = 1.441237217113085
    alpha_2 = 0.4511442198433961
    rho = -0.49709119294335674
    gamma_r = 1.4395688575650551

    m_1 = get_m_1(M_chirp, q)
    m_2 = get_m_2(M_chirp, q)

    return (
        beta
        * (m_1 / (1e3 * MSUN)) ** (-alpha_1)
        * (m_2 / MSUN) ** alpha_2
        * (1 + rho * np.log(gamma_s / gamma_r))
    )


def get_lam(gamma_s: float, gamma_e: float) -> float:
    return (11 - 2 * (gamma_s + gamma_e)) / 3


def get_eta(
    M_chirp: float,
    q: float,
    rho_6: float,
    gamma_s: float,
    gamma_e: float,
) -> float:
    m_1 = get_m_1(M_chirp, q)
    m_2 = get_m_2(M_chirp, q)
    rho_s = get_rho_s(rho_6, m_1, gamma_s)
    c_f = get_c_f(m_1, m_2, rho_s, gamma_s)
    f_eq = get_f_eq(gamma_s, c_f)
    f_t = get_f_b(M_chirp, q, gamma_s)
    return (
        (5 + 2 * gamma_e)
        / (2 * (8 - gamma_s))
        * (f_eq / f_t) ** ((11 - 2 * gamma_s) / 3)
    )


class Binary(ABC):
    _M_chirp: float
    Phi_c: float
    tT_c: float
    dL: float
    f_c: float
    a_v: float

    def __init__(
        self, M_chirp: float, Phi_c: float, tT_c: float, dL: float, f_c: float
    ):
        self.M_chirp = M_chirp
        self.Phi_c = Phi_c
        self.tT_c = tT_c
        self.dL = dL
        self.f_c = f_c

    @classmethod
    def get_a_v(cls, M_chirp):
        return 1 / 16 * (C ** 3 / (pi * G * M_chirp)) ** (5 / 3)

    def _update_internals(self):
        self.a_v = self.get_a_v(self.M_chirp)

    @property
    def M_chirp(self):
        return self._M_chirp

    @M_chirp.setter
    def M_chirp(self, val):
        self._M_chirp = val
        self._update_internals()

    @abstractmethod
    def _Phi_to_c_indef(self, f: ArrOrFloat) -> ArrOrFloat:
        ...

    @abstractmethod
    def _t_to_c_indef(self, f: ArrOrFloat) -> ArrOrFloat:
        ...

    @abstractmethod
    def d2Phi_dt2(self, f: ArrOrFloat) -> ArrOrFloat:
        ...

    def PhiT(self, f: ArrOrFloat) -> ArrOrFloat:
        return 2 * pi * f * self.t_to_c(f) - self.Phi_to_c(f)

    def Psi(self, f: ArrOrFloat) -> ArrOrFloat:
        return 2 * pi * f * self.tT_c - self.Phi_c - pi / 4 - self.PhiT(f)

    def h_0(self, f: ArrOrFloat):
        return np.where(
            f <= self.f_c,
            1
            / 2
            * 4
            * pi ** (2 / 3)
            * (G * self.M_chirp) ** (5 / 3)
            * f ** (2 / 3)
            / C ** 4
            * np.sqrt(2 * pi / np.abs(self.d2Phi_dt2(f))),
            0.0,
        )

    def amp(self, f: ArrOrFloat) -> ArrOrFloat:
        """
        Amplitude averaged over inclination angle.
        """
        return np.sqrt(4 / 5) * self.h_0(f) / self.dL

    def Phi_to_c(self, f: ArrOrFloat) -> ArrOrFloat:
        return self._Phi_to_c_indef(f) - self._Phi_to_c_indef(self.f_c)

    def t_to_c(self, f: ArrOrFloat) -> ArrOrFloat:
        return self._t_to_c_indef(f) - self._t_to_c_indef(self.f_c)

    def get_f_range(self, t_obs: float) -> Tuple[float, float]:
        """
        Finds the frequency range [f(-(t_obs + tT_c)), f(-tT_c)].
        """
        # Find frequency t_obs + tT_c before merger
        bracket = (self.f_c * 0.001, self.f_c * 1.1)
        fn = lambda f_l: (self.t_to_c(f_l) - (t_obs + self.tT_c)) ** 2
        res = minimize_scalar(fn, bounds=bracket)
        assert res.success
        f_l = res.x

        # Find frequency tT_c before merger
        fn = lambda f_h: (self.t_to_c(f_h) - self.tT_c) ** 2
        res = minimize_scalar(fn, bracket=bracket)
        assert res.success
        f_h = res.x

        return (f_l, f_h)


class VacuumBinary(Binary):
    """
    GR-in-vacuum binary.
    """

    def _Phi_to_c_indef(self, f: ArrOrFloat) -> ArrOrFloat:
        return self.a_v / f ** (5 / 3)

    def _t_to_c_indef(self, f: ArrOrFloat) -> ArrOrFloat:
        return 5 * self.a_v / (16 * pi * f ** (8 / 3))

    def d2Phi_dt2(self, f: ArrOrFloat) -> ArrOrFloat:
        return 12 * pi ** 2 * f ** (11 / 3) / (5 * self.a_v)


class DynamicDress(Binary):
    """
    A dark dress with an evolving DM halo.
    """

    _q: float
    _gamma_s: float
    _rho_6: float
    gamma_e: float = 5 / 2
    th: float = 1.0  # assuming gamma_e = 5/2
    eta: float
    lam: float
    f_b: float

    def __init__(
        self,
        gamma_s: float,
        rho_6: float,
        M_chirp: float,
        q: float,
        Phi_c: float,
        tT_c: float,
        dL: float,
        f_c: float,
    ):
        self._gamma_s = gamma_s
        self._rho_6 = rho_6
        self._M_chirp = M_chirp
        self._q = q
        self.Phi_c = Phi_c
        self.tT_c = tT_c
        self.dL = dL
        self.f_c = f_c
        self._update_internals()

    @Binary.M_chirp.setter
    def M_chirp(self, val):
        self._M_chirp = val
        self._update_internals()

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, val):
        self._q = val
        self._update_internals()

    @property
    def gamma_s(self):
        return self._gamma_s

    @gamma_s.setter
    def gamma_s(self, val):
        self._gamma_s = val
        self._update_internals()

    @property
    def rho_6(self):
        return self._rho_6

    @rho_6.setter
    def rho_6(self, val):
        self._rho_6 = val
        self._update_internals()

    def _update_internals(self):
        super()._update_internals()
        self.m_1 = get_m_1(self.M_chirp, self.q)
        self.m_2 = get_m_2(self.M_chirp, self.q)
        self.rho_s = get_rho_s(self.rho_6, self.m_1, self.gamma_s)
        self.eta = get_eta(self.M_chirp, self.q, self.rho_6, self.gamma_s, self.gamma_e)
        self.lam = get_lam(self.gamma_s, self.gamma_e)
        self.f_b = get_f_b(self.M_chirp, self.q, self.gamma_s)

    def _Phi_to_c_indef(self, f: ArrOrFloat) -> ArrOrFloat:
        x = f / self.f_b
        return (
            self.a_v
            / f ** (5 / 3)
            * (
                1
                - self.eta
                * x ** (-self.lam)
                * (1 - hypgeom(self.th, -(x ** (-5 / (3 * self.th)))))
            )
        )

    def _t_to_c_indef(self, f: ArrOrFloat) -> ArrOrFloat:
        x = f / self.f_b
        coeff = (
            self.a_v
            * x ** (-self.lam)
            / (16 * pi * (1 + self.lam) * (8 + 3 * self.lam) * f ** (8 / 3))
        )
        term_1 = 5 * (1 + self.lam) * (8 + 3 * self.lam) * x ** self.lam
        term_2 = (
            8
            * self.lam
            * (8 + 3 * self.lam)
            * self.eta
            * hypgeom(self.th, -(x ** (-5 / (3 * self.th))))
        )
        term_3 = (
            -40
            * (1 + self.lam)
            * self.eta
            * hypgeom(
                -1 / 5 * self.th * (8 + 3 * self.lam),
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
                * hypgeom(
                    1 / 5 * self.th * (8 + 3 * self.lam),
                    -(x ** (-5 / (3 * self.th))),
                )
            )
        )
        return coeff * (term_1 + term_2 + term_3 + term_4)

    def d2Phi_dt2(self, f: ArrOrFloat) -> ArrOrFloat:
        x = f / self.f_b
        return (
            12
            * pi ** 2
            * f ** (11 / 3)
            * x ** self.lam
            * (1 + x ** (5 / (3 * self.th)))
            / (
                self.a_v
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
                    * hypgeom(self.th, -(x ** (-5 / (3 * self.th))))
                )
            )
        )
