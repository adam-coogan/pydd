from functools import partial

from jax import jit
import jax.numpy as jnp

from .binary import Binary, Psi, amp
from .noise import S_n_LISA


"""
SNR, likelihood and match functions.
"""


# @partial(jit, static_argnums=(0, 3, 4))
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


# @partial(jit, static_argnames="S_n")
def calculate_SNR(params: Binary, fs, S_n=S_n_LISA):
    integrand = amp(fs, params) ** 2 / S_n(fs)
    return jnp.sqrt(4 * jnp.trapz(integrand, fs))


# @partial(jit, static_argnames="S_n")
def calculate_match_unnormd(params_h: Binary, params_d: Binary, fs, S_n=S_n_LISA):
    """
    Inner product of waveforms, maximized over Phi_c by taking absolute value.
    """
    wf_h = amp(fs, params_h) * jnp.exp(1j * Psi(fs, params_h))
    wf_d = amp(fs, params_d) * jnp.exp(1j * Psi(fs, params_d))
    return jnp.abs(4 * jnp.trapz(wf_h.conj() * wf_d / S_n(fs), fs))


# @partial(jit, static_argnames="S_n")
def loglikelihood(params_h: Binary, params_d: Binary, fs, S_n=S_n_LISA):
    """
    Log-likelihood for a signal from a binary params_d modeled using params_h,
    maximized over the distance to the binary and Phi_c.
    """
    # Waveform magnitude
    ip_hh = calculate_SNR(params_h, fs, S_n) ** 2
    # Inner product of waveforms, maximized over Phi_c by taking absolute value
    ip_hd = calculate_match_unnormd(params_h, params_d, fs, S_n)
    # Maximize over distance
    return 1 / 2 * ip_hd ** 2 / ip_hh


def get_match_pads(fs):
    """
    Returns padding arrays required for accurate match calculation.
    Padding `fs` with the returned arrays (almost) doubles its size and extends
    it down to 0.
    """
    df = fs[1] - fs[0]
    N = 2 * jnp.array(fs[-1] / df - 1).astype(int)
    pad_low = jnp.zeros(jnp.array(fs[0] / df).astype(int))
    pad_high = jnp.zeros(N - jnp.array(fs[-1] / df).astype(int))
    return pad_low, pad_high


# @partial(jit, static_argnames="S_n")
def calculate_match_unnormd_fft(
    params_h: Binary, params_d: Binary, fs, pad_low, pad_high, S_n=S_n_LISA
):
    """
    Inner product of waveforms, maximized over Phi_c by taking absolute value
    and t_c using the fast Fourier transform.
    """
    df = fs[1] - fs[0]
    wf_h = amp(fs, params_h) * jnp.exp(1j * Psi(fs, params_h))
    wf_d = amp(fs, params_d) * jnp.exp(1j * Psi(fs, params_d))
    Sns = S_n(fs)

    # Use IFFT trick to maximize over t_c. Ref: Maggiore's book, eq. 7.171.
    integrand = 4 * wf_h.conj() * wf_d / Sns * df
    integrand_padded = jnp.concatenate((pad_low, integrand, pad_high))
    # print(low_padding, high_padding, len(fs), N)
    return jnp.abs(len(integrand_padded) * jnp.fft.ifft(integrand_padded)).max()


# @partial(jit, static_argnames="S_n")
def loglikelihood_fft(
    params_h: Binary, params_d: Binary, fs, pad_low, pad_high, S_n=S_n_LISA
):
    """
    Log-likelihood for a signal from a binary params_d modeled using params_h,
    maximized over the distance to the binary, Phi_c and t_c (i.e., all
    extrinsic parameters).
    """
    # Waveform magnitude
    ip_hh = calculate_SNR(params_h, fs, S_n) ** 2
    # Inner product of waveforms, maximized over Phi_c by taking absolute value
    ip_hd = calculate_match_unnormd_fft(params_h, params_d, fs, pad_low, pad_high, S_n)
    # Maximize over distance
    return 1 / 2 * ip_hd ** 2 / ip_hh
