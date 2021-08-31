from functools import partial

from jax import jit
import jax.numpy as jnp

from .binary import Binary, Psi, amp
from .noise import S_n_LISA


"""
SNR, likelihood and match functions.
"""


@partial(jit, static_argnums=(0, 3, 4))
def simps(f, a, b, N, log):
    """
    Stolen from: https://jax-cosmo.readthedocs.io/en/latest/_modules/jax_cosmo/scipy/integrate.html

    Approximate the integral of f(x) from a to b by Simpson's rule.

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


@partial(jit, static_argnums=(3, 4))
def calculate_SNR(params: Binary, f_l, f_h, n=3000, S_n=S_n_LISA):
    f_h = jnp.minimum(f_h, params.f_c)
    modh_integrand = lambda f: 4 * amp(f, params) ** 2 / S_n(f)
    return jnp.sqrt(simps(modh_integrand, f_l, f_h, n, True))


@partial(jit, static_argnums=(4, 5))
def calculate_match_unnormd(
    params_h: Binary, params_d: Binary, f_l, f_h, n, S_n=S_n_LISA
):
    """
    Inner product of waveforms, maximized over Phi_c by taking absolute value.
    """
    f_h = jnp.minimum(f_h, jnp.minimum(params_h.f_c, params_d.f_c))

    amp_prod = lambda f: amp(f, params_h) * amp(f, params_d)
    dPsi = lambda f: Psi(f, params_h) - Psi(f, params_d)
    integrand_re = lambda f: amp_prod(f) * jnp.cos(dPsi(f)) / S_n(f)
    integrand_im = lambda f: amp_prod(f) * jnp.sin(dPsi(f)) / S_n(f)
    re = 4 * simps(integrand_re, f_l, f_h, n, True)
    im = 4 * simps(integrand_im, f_l, f_h, n, True)
    return jnp.sqrt(re ** 2 + im ** 2)


@partial(jit, static_argnums=(4, 5, 6))
def loglikelihood(
    params_h: Binary, params_d: Binary, f_l, f_h, n=3000, n_same=3000, S_n=S_n_LISA
):
    """
    Log-likelihood for a signal from a binary params_d modeled using params_h,
    maximized over the distance to the binary and Phi_c.
    """
    # Waveform magnitude
    ip_hh = calculate_SNR(params_h, f_l, f_h, n_same, S_n) ** 2
    # Inner product of waveforms, maximized over Phi_c by taking absolute value
    ip_hd = calculate_match_unnormd(params_h, params_d, f_l, f_h, n, S_n)
    # Maximize over distance
    return 1 / 2 * ip_hd ** 2 / ip_hh


@partial(jit, static_argnums=(4, 5))
def calculate_match_unnormd_fft(params_h, params_d, f_l, f_h, n, S_n=S_n_LISA):
    """
    Inner product of waveforms, maximized over Phi_c by taking absolute value
    and t_c using the fast Fourier transform.
    """
    f_h = jnp.minimum(f_h, jnp.minimum(params_h.f_c, params_d.f_c))
    fs = jnp.linspace(f_l, f_h, n)
    df = fs[1] - fs[0]

    wf_hs = amp(fs, params_h) * jnp.exp(1j * Psi(fs, params_h))
    wf_ds = amp(fs, params_d) * jnp.exp(1j * Psi(fs, params_d))
    overlap_tc = jnp.fft.fft(4 * wf_hs * wf_ds.conj() * df / S_n(fs))
    return jnp.abs(overlap_tc).max()


@partial(jit, static_argnums=(4, 5, 6))
def loglikelihood_fft(
    params_h: Binary, params_d: Binary, f_l, f_h, n, n_same, S_n=S_n_LISA
):
    """
    Log-likelihood for a signal from a binary params_d modeled using params_h,
    maximized over the distance to the binary, Phi_c and t_c (i.e., all
    extrinsic parameters).
    """
    # Waveform magnitude
    ip_hh = calculate_SNR(params_h, f_l, f_h, n_same, S_n) ** 2
    # Inner product of waveforms, maximized over Phi_c by taking absolute value
    ip_hd = calculate_match_unnormd_fft(params_h, params_d, f_l, f_h, n, S_n)
    # Maximize over distance
    return 1 / 2 * ip_hd ** 2 / ip_hh
