import jax.numpy as jnp
from scipy.optimize import minimize_scalar

from pydd.analysis import loglikelihood, loglikelihood_fft
from pydd.binary import DynamicDress, MSUN, PC, YR, make_dynamic_dress, t_to_c

"""
Check that maximizing over tT_c with an FFT gives the same results as doing it
by hand.
"""


def test():
    # Test system
    m_1 = jnp.array(1e3 * MSUN)
    m_2 = jnp.array(1 * MSUN)
    rho_s = jnp.array(1e-3 * MSUN / PC ** 3)
    gamma_s = jnp.array(2.2)
    dd_s = make_dynamic_dress(m_1, m_2, rho_s, gamma_s)

    t_obs_lisa = 5 * YR
    f_l = minimize_scalar(
        lambda f: (t_to_c(f, dd_s) - t_obs_lisa) ** 2, bracket=(1e-4, 1e-1)
    ).x
    f_c = dd_s.f_c

    # Compute likelihood for different tT_c values
    tT_cs = jnp.linspace(-300, 0, 50)
    logLs = []
    dd_alt = None
    for tT_c in tT_cs:
        dd_alt = DynamicDress(
            jnp.array(2.2),
            jnp.array(0.5e-6),
            dd_s.M_chirp,
            dd_s.q,
            dd_s.Phi_c,
            jnp.array(tT_c),
            dd_s.dL,
            dd_s.f_c,
        )
        logLs.append(loglikelihood(dd_alt, dd_s, f_l, f_c, 20000, 3000))
    logLs = jnp.array(logLs)

    # Maximize over tT_c with FFT
    logL_fft = loglikelihood_fft(dd_alt, dd_s, f_l, f_c, 100000, 3000)

    assert jnp.allclose(logL_fft, logLs.max(), atol=0.0, rtol=0.005)


if __name__ == "__main__":
    test()
