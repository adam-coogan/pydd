import numpy as np

from .binary_np import Binary
from .noise_np import S_n_LISA


"""
SNR, likelihood and match functions.
"""


def calculate_SNR(b: Binary, fs, S_n=S_n_LISA):
    integrand = b.amp(fs) ** 2 / S_n(fs)
    return np.sqrt(4 * np.trapz(integrand, fs))


def calculate_match_unnormd(b_h: Binary, b_d: Binary, fs, S_n=S_n_LISA):
    """
    Inner product of waveforms, maximized over Phi_c by taking absolute value.
    """
    wf_h = b_h.amp(fs) * np.exp(1j * b_h.Psi(fs))
    wf_d = b_d.amp(fs) * np.exp(1j * b_d.Psi(fs))
    return np.abs(4 * np.trapz(wf_h.conj() * wf_d / S_n(fs), fs))


def loglikelihood(b_h: Binary, b_d: Binary, fs, S_n=S_n_LISA):
    """
    Log-likelihood for a signal from a binary params_d modeled using params_h,
    maximized over the distance to the binary and Phi_c.

    Applies frequency cut to the waveform.
    """
    # Waveform magnitude
    ip_hh = calculate_SNR(b_h, fs, S_n) ** 2
    # Inner product of waveforms, maximized over Phi_c by taking absolute value
    ip_hd = calculate_match_unnormd(b_h, b_d, fs, S_n)
    # Maximize over distance
    return 1 / 2 * ip_hd ** 2 / ip_hh


def calculate_SNR_cut(b: Binary, f_range, fs, S_n=S_n_LISA):
    integrand = np.where(
        (fs >= f_range[0]) & (fs <= f_range[1]), b.amp(fs) ** 2 / S_n(fs), 0.0
    )
    return np.sqrt(4 * np.trapz(integrand, fs))


def calculate_match_unnormd_cut(
    b_h: Binary, b_d: Binary, f_range_h, f_range_d, fs, S_n=S_n_LISA
):
    """
    Inner product of waveforms, maximized over Phi_c by taking absolute value.

    Applies frequency cuts to both waveforms.
    """
    wf_h = np.where(
        (fs >= f_range_h[0]) & (fs <= f_range_h[1]),
        b_h.amp(fs) * np.exp(1j * b_h.Psi(fs)),
        0.0,
    )
    wf_d = np.where(
        (fs >= f_range_d[0]) & (fs <= f_range_d[1]),
        b_d.amp(fs) * np.exp(1j * b_d.Psi(fs)),
        0.0,
    )
    return np.abs(4 * np.trapz(wf_h.conj() * wf_d / S_n(fs), fs))


def loglikelihood_cut(
    b_h: Binary, b_d: Binary, f_range_h, f_range_d, fs, S_n=S_n_LISA
):
    """
    Log-likelihood for a signal from a binary params_d modeled using params_h,
    maximized over the distance to the binary and Phi_c.

    Applies frequency cuts to both waveforms.
    """
    # Waveform magnitude
    ip_hh = calculate_SNR_cut(b_h, f_range_h, fs, S_n) ** 2
    # Inner product of waveforms, maximized over Phi_c by taking absolute value
    ip_hd = calculate_match_unnormd_cut(
        b_h, b_d, f_range_h, f_range_d, fs, S_n
    )
    # Maximize over distance
    return 1 / 2 * ip_hd ** 2 / ip_hh


def get_match_pads(fs):
    """
    Returns padding arrays required for accurate match calculation.
    Padding `fs` with the returned arrays (almost) doubles its size and extends
    it down to 0.
    """
    df = fs[1] - fs[0]
    N = 2 * np.array(fs[-1] / df - 1).astype(int)
    pad_low = np.zeros(np.array(fs[0] / df).astype(int))
    pad_high = np.zeros(N - np.array(fs[-1] / df).astype(int))
    return pad_low, pad_high


def calculate_match_unnormd_fft(
    b_h: Binary, b_d: Binary, fs, pad_low, pad_high, S_n=S_n_LISA
):
    """
    Inner product of waveforms, maximized over Phi_c by taking absolute value
    and t_c using the fast Fourier transform.
    """
    df = fs[1] - fs[0]
    wf_h = b_h.amp(fs) * np.exp(1j * b_h.Psi(fs))
    wf_d = b_d.amp(fs) * np.exp(1j * b_d.Psi(fs))
    Sns = S_n(fs)

    # Use IFFT trick to maximize over t_c. Ref: Maggiore's book, eq. 7.171.
    integrand = 4 * wf_h.conj() * wf_d / Sns * df
    integrand_padded = np.concatenate((pad_low, integrand, pad_high))
    # print(low_padding, high_padding, len(fs), N)
    return np.abs(len(integrand_padded) * np.fft.ifft(integrand_padded)).max()


def loglikelihood_fft(
    b_h: Binary, b_d: Binary, fs, pad_low, pad_high, S_n=S_n_LISA
):
    """
    Log-likelihood for a signal from a binary params_d modeled using params_h,
    maximized over the distance to the binary, Phi_c and t_c (i.e., all
    extrinsic parameters).
    """
    # Waveform magnitude
    ip_hh = calculate_SNR(b_h, fs, S_n) ** 2
    # Inner product of waveforms, maximized over Phi_c by taking absolute value
    ip_hd = calculate_match_unnormd_fft(b_h, b_d, fs, pad_low, pad_high, S_n)
    # Maximize over distance
    return 1 / 2 * ip_hd ** 2 / ip_hh
