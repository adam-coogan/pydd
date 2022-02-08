import time

import numpy as np
from scipy.optimize import root_scalar

from pydd.analysis_np import calculate_SNR_cut, loglikelihood_cut
from pydd.binary_np import *
from pydd.noise_np import *

Array = NDArray[np.float64]
ArrOrFloat = Union[float, Array]

# Setup
RHO_S_PBH = 1.798e4 * MSUN / PC ** 3
GAMMA_S_PBH = 9 / 4
S_n, F_RANGE_NOISE = S_n_aLIGO, f_range_aLIGO
FS = np.linspace(*F_RANGE_NOISE, 100_000)
M_1 = 1 * MSUN
M_2 = 1e-3 * MSUN
M_CHIRP = get_M_chirp(M_1, M_2)
Q = M_2 / M_1
RHO_S = RHO_S_PBH
GAMMA_S = GAMMA_S_PBH
RHO_6 = get_rho_6(RHO_S, M_1, GAMMA_S)
TT_C = 0.0
F_C = get_f_isco(M_1)
PHI_C = 0.0
_DD_D = DynamicDress(
    GAMMA_S,
    RHO_6,
    get_M_chirp(M_1, M_2),
    M_2 / M_1,
    PHI_C,
    tT_c=TT_C,
    dL=100e6 * PC,
    f_c=get_f_isco(M_1),
)

# Get f_range
T_OBS = 1 * YR
F_RANGE_D = _DD_D.get_f_range(T_OBS)
# Get dL
SNR_THRESH = 12.0
_fn = lambda dL: calculate_SNR_cut(
    DynamicDress(
        _DD_D.gamma_s,
        _DD_D.rho_6,
        _DD_D.M_chirp,
        _DD_D.q,
        _DD_D.Phi_c,
        _DD_D.tT_c,
        dL,
        _DD_D.f_c,
    ),
    F_RANGE_D,
    FS,
    S_n,
)
res = root_scalar(
    lambda dL: (_fn(dL) - SNR_THRESH), bracket=(0.1e6 * PC, 100000e6 * PC)
)
assert res.converged
DL = res.root

# Signal system
DD_D = DynamicDress(
    _DD_D.gamma_s,
    _DD_D.rho_6,
    _DD_D.M_chirp,
    _DD_D.q,
    _DD_D.Phi_c,
    _DD_D.tT_c,
    DL,
    _DD_D.f_c,
)


def ll_cut_fn(dd_h, f_range_h):
    return loglikelihood_cut(dd_h, DD_D, f_range_h, F_RANGE_D, FS, S_n)


def sample_Mc_tTc() -> Array:
    return np.array(
        [
            M_CHIRP / MSUN * (1 + 6e-8 * (np.random.rand(1) - 0.5)),
            0.0004 * (np.random.rand(1) - 0.5),
        ]
    )


def unpack(x: Array) -> DynamicDress:
    M_chirp_MSUN, tT_c = x
    M_chirp = M_chirp_MSUN * MSUN
    m_1 = get_m_1(M_chirp, DD_D.q)
    rho_6 = get_rho_6(RHO_S, m_1, DD_D.gamma_s)
    f_c = get_f_isco(m_1)
    return DynamicDress(
        DD_D.gamma_s, rho_6, M_chirp, DD_D.q, DD_D.Phi_c, tT_c, DD_D.dL, f_c
    )


def ll_nomax_fn(x: Array) -> Array:
    """
    Log-likelihood without tT_c maximization.
    """
    dd_h = unpack(x)
    f_range_h = dd_h.get_f_range(T_OBS)
    return ll_cut_fn(dd_h, f_range_h)


print("Without tT_c maximization")
print("compiling and evaluating:", ll_nomax_fn(sample_Mc_tTc()))

n_loops = 10
t_start = time.time()
for _ in range(n_loops):
    ll_nomax_fn(sample_Mc_tTc())
print((time.time() - t_start) / n_loops)


def sample_Mc() -> Array:
    return np.array([M_CHIRP / MSUN * (1 + 6e-8 * (np.random.rand(1) - 0.5))])


def ll_fn(x: Array) -> Array:
    """
    Log-likelihood with tT_c maximization.
    """
    M_chirp = x[0] * MSUN
    m_1 = get_m_1(M_chirp, Q)
    rho_6 = get_rho_6(RHO_S, m_1, GAMMA_S)
    f_c = get_f_isco(m_1)

    def _nll(tT_c):
        """
        Negative log-likelihood.
        """
        dd_h = DynamicDress(GAMMA_S, rho_6, M_chirp, Q, PHI_C, tT_c, DL, f_c)
        f_range_h = dd_h.get_f_range(T_OBS)
        return -ll_cut_fn(dd_h, f_range_h)

    bracket = (-1e-3, 1e-3)
    res = minimize_scalar(_nll, bracket=bracket)
    assert res.success, f"tT_c maximization failed: {res}"
    return res.fun


print("With tT_c maximization")
print("compiling and evaluating:", ll_fn(sample_Mc_tTc()))

n_loops = 10
t_start = time.time()
for _ in range(n_loops):
    ll_fn(sample_Mc())
print((time.time() - t_start) / n_loops)
