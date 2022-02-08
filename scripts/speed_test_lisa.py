from functools import partial
import os
import time
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root_scalar

from pydd.analysis import (
    calculate_SNR_cut,
    get_match_pads,
    loglikelihood,
    loglikelihood_cut,
    loglikelihood_fft,
)
from pydd.binary import *
from pydd.noise import *  # type: ignore

from utils import get_ptform, rho_6T_to_rho6, rho_6_to_rho6T

Array = jnp.ndarray

ns = jnp.load(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "ns",
        "rho_6T=0.5345_gamma_s=2.25.pkl",
    ),
    allow_pickle=True,
)
ps = jnp.exp(ns.logwt - ns.logwt.max())
ps = ps / jnp.sum(ps)


def resample_posterior(key, n=1) -> Array:
    return random.choice(key, a=ns.samples, shape=(n,), p=ps)


def replace_tT_c(dd, tT_c) -> DynamicDress:
    return DynamicDress(
        dd.gamma_s, dd.rho_6, dd.M_chirp, dd.q, dd.Phi_c, tT_c, dd.dL, dd.f_c
    )


# Setup
RHO_S_PBH = jnp.array(1.798e4 * MSUN / PC ** 3)
GAMMA_S_PBH = jnp.array(9 / 4)
S_n, F_RANGE_NOISE = S_n_LISA, f_range_LISA
M_1 = jnp.array(1e3 * MSUN)
M_2 = jnp.array(1.4 * MSUN)
M_CHIRP = get_M_chirp(M_1, M_2)
Q = M_2 / M_1
RHO_S = RHO_S_PBH
GAMMA_S = GAMMA_S_PBH
RHO_6 = get_rho_6(RHO_S, M_1, GAMMA_S)
TT_C = jnp.array(0.0)
F_C = get_f_isco(M_1)
PHI_C = jnp.array(0.0)
_DD_D = DynamicDress(
    GAMMA_S,
    RHO_6,
    get_M_chirp(M_1, M_2),
    M_2 / M_1,
    PHI_C,
    tT_c=TT_C,
    dL=jnp.array(100e6 * PC),
    f_c=get_f_isco(M_1),
)

# Get f_range
T_OBS = 5 * YR
F_RANGE_D = get_f_range(_DD_D, T_OBS)
FS = jnp.linspace(*F_RANGE_D, 100_000)
PAD_LOW, PAD_HIGH = get_match_pads(FS)
# Get dL
SNR_THRESH = 12.0
_fn = jax.jit(
    lambda dL: calculate_SNR_cut(
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


def sample_Mc() -> Array:
    return jnp.array(
        [
            # M_CHIRP / MSUN * (1 + 6e-8 * (np.random.rand() - 0.5)),
            M_CHIRP
            / MSUN
            * (1 + 1.25e-7 * 2 * (np.random.rand() - 0.5)),
        ]
    )


def unpack_Mc(x: Array) -> DynamicDress:
    M_chirp_MSUN = x[0]
    M_chirp = M_chirp_MSUN * MSUN
    m_1 = get_m_1(M_chirp, DD_D.q)
    rho_6 = get_rho_6(RHO_S, m_1, DD_D.gamma_s)
    f_c = get_f_isco(m_1)
    return DynamicDress(
        DD_D.gamma_s, rho_6, M_chirp, DD_D.q, DD_D.Phi_c, DD_D.tT_c, DD_D.dL, f_c
    )


# ptform = lambda u: get_ptform(
#     u,
#     gamma_s_range=[2.25, 2.3],
#     rho_6T_range=[0.48e16 * MSUN / PC ** 3, 0.7e16 * MSUN / PC ** 3],
#     log10_q_range=[-3.0, -2.5],
#     dM_chirp_MSUN_range=[-4e-4, 4e-4],
#     dd_s=DD_D,
# )

ptform = lambda u: get_ptform(
    u,
    gamma_s_range=[2.2501, 2.28],
    rho_6T_range=[0.52, 0.56],
    log10_q_range=[-2.7, -2.7],
    dM_chirp_MSUN_range=[-2e-4, 2e-4],
    dd_s=DD_D,
)


def sample_x() -> Array:
    return ptform(np.random.rand(4))


def unpack_x(x) -> DynamicDress:
    gamma_s, rho_6T, M_chirp_MSUN, log10_q = x
    M_chirp = M_chirp_MSUN * MSUN
    q = 10 ** log10_q
    rho_6 = rho_6T_to_rho6(rho_6T)
    f_c = get_f_isco(get_m_1(M_chirp, q))
    return DynamicDress(gamma_s, rho_6, M_chirp, q, DD_D.Phi_c, DD_D.tT_c, DD_D.dL, f_c)


@jax.jit
def get_ll_cut_fn(dd_h, f_range_h):
    return loglikelihood_cut(dd_h, DD_D, f_range_h, F_RANGE_D, FS, S_n)


def get_ll_max(
    x: Array,
    unpack: Callable[[Array], DynamicDress],
    bracket: Tuple[Array, Array] = (jnp.array(-1e5), jnp.array(1e5)),
) -> Tuple[Array, Array]:
    """
    Log-likelihood with tT_c maximization.
    """
    dd = unpack(x)

    def _nll(tT_c):
        """
        Negative log-likelihood.
        """
        dd_h = replace_tT_c(dd, tT_c)
        # f_range_h = get_f_range(dd_h, T_OBS, bracket=F_RANGE_D)
        f_range_h = F_RANGE_D
        return -get_ll_cut_fn(dd_h, f_range_h)

    res = minimize_scalar(_nll, bracket=bracket)
    if not res.success:
        raise RuntimeError(f"tT_c maximization failed: {res}")
    return -res.fun, res.x


# @partial(jax.jit, static_argnames="unpack")
def get_ll_fft(
    x: Array, unpack: Callable[[Array], DynamicDress]
) -> Tuple[Array, Array]:
    dd_h = unpack(x)
    return loglikelihood_fft(dd_h, DD_D, FS, PAD_LOW, PAD_HIGH, S_n)


print("With tT_c maximization")
np.random.seed(0)
key = random.PRNGKey(636)
for _ in range(50):
    try:
        key, subkey = random.split(key)
        # x = sample_x()
        x = resample_posterior(subkey).flatten()
        ll_max, tT_c_max = get_ll_max(x, unpack_x)
        ll_fft, tT_c_fft = get_ll_fft(x, unpack_x)
        print(
            f"LL max: {ll_max:.3f}, "
            f"LL FFT: {ll_fft:.3f}, "
            f"LL % difference: {(ll_fft - ll_max) / ll_max * 100:.3f}, "
            f"pr % difference: {(jnp.exp(ll_fft) - jnp.exp(ll_max)) / jnp.exp(ll_max) * 100:.3f}"
        )
        # print(
        #     f"tT_c max: {tT_c_max:.5f}, "
        #     f"tT_c FFT: {tT_c_fft:.5f}, "
        #     f"% difference: {(tT_c_fft - tT_c_max) / tT_c_max * 100:.5f}"
        # )
    except RuntimeError as e:
        print(e)
    print()

key, subkey = random.split(key)
x = resample_posterior(subkey).flatten()
ll_max, tT_c_max = get_ll_max(x, unpack_x)
dd = unpack_x(x)
tT_cs = jnp.linspace(tT_c_max * 0.9, tT_c_max * 1.1, 100)
lls = []
for tT_c in tT_cs:
    lls.append(
        jax.jit(loglikelihood, static_argnames="S_n")(
            replace_tT_c(dd, tT_c), DD_D, FS, S_n
        )
    )
lls = jnp.array(lls)

plt.plot(tT_cs, lls)
plt.savefig("testll.png")
