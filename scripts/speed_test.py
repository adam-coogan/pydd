import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import root_scalar
import time

from pydd.analysis import loglikelihood_cut, calculate_SNR_cut
from pydd.binary import *  # type: ignore
from pydd.noise import *  # type: ignore

rho_s_pbh = jnp.array(1.798e4 * MSUN / PC ** 3)
gamma_s_pbh = jnp.array(9 / 4)

S_n, f_range = S_n_aLIGO, f_range_aLIGO
fs = jnp.linspace(*f_range, 100_000)

m_1 = jnp.array(1 * MSUN)
m_2 = jnp.array(1e-3 * MSUN)
M_chirp = get_M_chirp(m_1, m_2)
q = m_2 / m_1
rho_s = rho_s_pbh
gamma_s = gamma_s_pbh
rho_6 = get_rho_6(rho_s, m_1, gamma_s)
tT_c = jnp.array(0.0)
f_c = get_f_isco(m_1)
_dd_d = DynamicDress(
    gamma_s,
    rho_6,
    get_M_chirp(m_1, m_2),
    m_2 / m_1,
    jnp.array(0.0),
    tT_c=tT_c,
    dL=jnp.array(100e6 * PC),
    f_c=get_f_isco(m_1),
)

# Get f_range
t_obs = 1 * YR
f_range_d = get_f_range(_dd_d, t_obs)
# Get dL
snr_thresh = 12.0
_fn = jax.jit(
    lambda dL: calculate_SNR_cut(
        DynamicDress(
            _dd_d.gamma_s,
            _dd_d.rho_6,
            _dd_d.M_chirp,
            _dd_d.q,
            _dd_d.Phi_c,
            _dd_d.tT_c,
            dL,
            _dd_d.f_c,
        ),
        f_range_d,
        fs,
        S_n,
    )
)
res = root_scalar(
    lambda dL: (_fn(dL) - snr_thresh), bracket=(0.1e6 * PC, 100000e6 * PC)
)
assert res.converged
dL = res.root

dd_d = DynamicDress(
    _dd_d.gamma_s,
    _dd_d.rho_6,
    _dd_d.M_chirp,
    _dd_d.q,
    _dd_d.Phi_c,
    _dd_d.tT_c,
    dL,
    _dd_d.f_c,
)


@jax.jit
def ll_cut_fn(dd_h, f_range_h):
    return loglikelihood_cut(dd_h, dd_d, f_range_h, f_range_d, fs, S_n)


@jax.jit
def unpack(x):
    M_chirp_MSUN, tT_c = x
    M_chirp = M_chirp_MSUN * MSUN
    m_1 = get_m_1(M_chirp, dd_d.q)
    rho_6 = get_rho_6(rho_s, m_1, dd_d.gamma_s)
    f_c = get_f_isco(m_1)
    return DynamicDress(
        dd_d.gamma_s, rho_6, M_chirp, dd_d.q, dd_d.Phi_c, tT_c, dd_d.dL, f_c
    )


def ll_fn(x):
    # Make template
    dd_h = unpack(x)

    f_range_h = get_f_range(dd_h, t_obs)

    return ll_cut_fn(dd_h, f_range_h)


def sample_x():
    return jnp.array(
        [
            M_chirp / MSUN * (1 + 6e-8 * (np.random.rand(1) - 0.5)),
            0.0004 * (np.random.rand(1) - 0.5),
        ]
    )


print("compiling and evaluating:", ll_fn(sample_x()))

n_loops = 10
t_start = time.time()
for _ in range(n_loops):
    ll_fn(sample_x()).block_until_ready()
print((time.time() - t_start) / n_loops)
