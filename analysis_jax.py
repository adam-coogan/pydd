from typing import NamedTuple

from jax import jit
import jax.numpy as jnp

from binary_jax import DynamicDress, Psi, amp_plus, get_f_isco, get_m_1
from noise import S_n_LISA


def simps(f, a, b, N=128, log=True):
    """
    Stolen from: https://jax-cosmo.readthedocs.io/en/latest/_modules/jax_cosmo/scipy/integrate.html

    Approximate the integral of f(x) from a to b by Simpson's rule.

    Simpson's rule approximates the integral \int_a^b f(x) dx by the sum:
    (dx/3) \sum_{k=1}^{N/2} (f(x_{2i-2} + 4f(x_{2i-1}) + f(x_{2i}))
    where x_i = a + i*dx and dx = (b - a)/N.

    Parameters
    ----------
    f : function
        Vectorized function of a single variable
    a , b : numbers
        Interval of integration [a,b]
    N : (even) integer
        Number of subintervals of [a,b]

    Returns
    -------
    float
        Approximation of the integral of f(x) from a to b using
        Simpson's rule with N subintervals of equal length.

    Examples
    --------

        >>> simps(lambda x : 3 * x ** 2, 0, 1, 10)
        1.0

    Notes:
    ------
    Stolen from: https://www.math.ubc.ca/~pwalls/math-python/integration/simpsons-rule/
    """
    # if N % 2 == 1:
    #     raise ValueError("N must be an even integer.")
    if not log:
        dx = (b - a) / N
        x = jnp.linspace(a, b, N + 1)
        y = f(x)
        S = dx / 3 * jnp.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2], axis=0)
        return S
    else:

        def x_times_f(log_x):
            x = jnp.exp(log_x)
            return x * f(x)

        return simps(x_times_f, jnp.log(a), jnp.log(b), N, False)


def calculate_SNR(params: NamedTuple, f_c, kind: str, f_l, f_h, n):
    fh = jnp.minimum(f_h, f_c)
    # fs = jnp.geomspace(f_l, fh, n)
    modh_integrand = lambda f: 4 * amp_plus(f, params, kind) ** 2 / S_n_LISA(f)
    return jnp.sqrt(simps(modh_integrand, f_l, fh, n, True))


def calculate_match_unnormd(
    params_h: NamedTuple,
    f_c_h,
    kind_h: str,
    params_d: NamedTuple,
    f_c_d,
    kind_d: str,
    f_l,
    f_h,
    n,
):
    fh = jnp.minimum(f_h, jnp.minimum(f_c_h, f_c_d))
    # fs = jnp.geomspace(f_l, fh, n)

    amp_prod = lambda f: amp_plus(f, params_h, kind_h) * amp_plus(f, params_d, kind_d)
    dPsi = lambda f: Psi(f, f_c_h, params_h, kind_h) - Psi(f, f_c_d, params_d, kind_d)
    integrand_re = lambda f: amp_prod(f) * jnp.cos(dPsi(f)) / S_n_LISA(f)
    integrand_im = lambda f: amp_prod(f) * jnp.sin(dPsi(f)) / S_n_LISA(f)
    re = 4 * simps(integrand_re, f_l, fh, n, log=True)
    im = 4 * simps(integrand_im, f_l, fh, n, log=True)
    return jnp.sqrt(re ** 2 + im ** 2)


def loglikelihood(
    params_h: NamedTuple,
    f_c_h,
    kind_h: str,
    params_d: NamedTuple,
    f_c_d,
    kind_d: str,
    f_l,
    f_h,
    n=3000,
    n_same=3000,
):
    # Waveform magnitude
    ip_hh = calculate_SNR(params_h, f_c_h, kind_h, f_l, f_h, n_same) ** 2
    # Inner product of waveforms, maximizing over Phi_c by taking the absolute value
    ip_hd = calculate_match_unnormd(
        params_h, f_c_h, kind_h, params_d, f_c_d, kind_d, f_l, f_h, n
    )
    # Maximize over distance
    return 1 / 2 * ip_hd ** 2 / ip_hh
    # r = ip_hd / ip_hh
    # return r * ip_hd - 1 / 2 * r ** 2 * ip_hh


def test_SNR_loglikelihood():
    dd_s = DynamicDress(
        jnp.array(2.3333333333333335),
        jnp.array(0.00018806659428775589),
        jnp.array(3.151009407916561e31),
        jnp.array(0.001),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(-56.3888135025341),
    )
    f_c = get_f_isco(get_m_1(dd_s.M_chirp, dd_s.q))
    f_l = 0.022621092492458004  # Hz

    dd_h = DynamicDress(
        jnp.array(2.2622788817665738),
        jnp.array(4.921717647731646e-5),
        jnp.array(3.151580260164573e31),
        jnp.array(0.0005415094825728555),
        jnp.array(dd_s.Phi_c),
        jnp.array(-227.74995698),
        jnp.array(dd_s.dL_iota),
    )
    f_c_h = get_f_isco(get_m_1(dd_h.M_chirp, dd_h.q))

    loglikelihood_jit = jit(loglikelihood, static_argnums=(2, 5, 8, 9))

    print("SNR(s) = ", calculate_SNR(dd_s, f_c, "d", f_l, f_c, 3000))
    print("SNR(h) = ", calculate_SNR(dd_h, f_c_h, "d", f_l, f_c_h, 3000))
    print(
        "log L(s|s) = ",
        loglikelihood_jit(dd_s, f_c, "d", dd_s, f_c, "d", f_l, f_c, 3000, 3000),
    )
    print(
        "log L(h|h) = ",
        loglikelihood_jit(dd_h, f_c_h, "d", dd_h, f_c_h, "d", f_l, f_c_h, 3000, 3000),
    )
    print(
        "log L(h|s) = ",
        loglikelihood_jit(dd_h, f_c_h, "d", dd_s, f_c, "d", f_l, f_c, 3000, 3000),
    )


if __name__ == "__main__":
    from jax.config import config

    config.update("jax_enable_x64", True)

    test_SNR_loglikelihood()
