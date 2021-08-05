import jax.numpy as jnp

from pydd.analysis import calculate_SNR, loglikelihood
from pydd.binary import DynamicDress, get_f_isco, get_m_1

"""
Make sure the SNR and likelihood calculations run.
"""


def test_SNR_loglikelihood(verbose=False):
    f_c_s = get_f_isco(get_m_1(jnp.array(3.151009407916561e31), jnp.array(0.001)))
    f_c_h = get_f_isco(
        get_m_1(jnp.array(3.151580260164573e31), jnp.array(0.0005415094825728555))
    )
    f_h = jnp.maximum(f_c_s, f_c_h)
    dd_s = DynamicDress(
        jnp.array(2.3333333333333335),
        jnp.array(0.00018806659428775589),
        jnp.array(3.151009407916561e31),
        jnp.array(0.001),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(-56.3888135025341),
        f_c_s,
    )
    f_l = 0.022621092492458004  # Hz

    dd_h = DynamicDress(
        jnp.array(2.2622788817665738),
        jnp.array(4.921717647731646e-5),
        jnp.array(3.151580260164573e31),
        jnp.array(0.0005415094825728555),
        jnp.array(dd_s.Phi_c),
        jnp.array(-227.74995698),
        jnp.array(dd_s.dL),
        f_c_h,
    )

    snr_s = calculate_SNR(dd_s, f_l, f_h, 3000).block_until_ready()
    snr_h = calculate_SNR(dd_h, f_l, f_h, 3000).block_until_ready()
    ll_ss = loglikelihood(dd_s, dd_s, f_l, f_h, 3000, 3000).block_until_ready()
    ll_hh = loglikelihood(dd_h, dd_h, f_l, f_h, 3000, 3000).block_until_ready()
    ll_hs = loglikelihood(dd_h, dd_s, f_l, f_h, 3000, 3000).block_until_ready()

    if verbose:
        print("SNR(s) = ", snr_s)
        print("SNR(h) = ", snr_h)
        print("log L(s|s) = ", ll_ss)
        print("log L(h|h) = ", ll_hh)
        print("log L(h|s) = ", ll_hs)


if __name__ == "__main__":
    test_SNR_loglikelihood(True)
