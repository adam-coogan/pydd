import jax.numpy as jnp

from analysis_jax import calculate_SNR, loglikelihood
from binary_jax import DynamicDress, get_f_isco, get_m_1


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

    print("SNR(s) = ", calculate_SNR(dd_s, f_c, "d", f_l, f_c, 3000))
    print("SNR(h) = ", calculate_SNR(dd_h, f_c_h, "d", f_l, f_c_h, 3000))
    print(
        "log L(s|s) = ",
        loglikelihood(dd_s, f_c, "d", dd_s, f_c, "d", f_l, f_c),
    )
    print(
        "log L(h|h) = ",
        loglikelihood(dd_h, f_c_h, "d", dd_h, f_c_h, "d", f_l, f_c_h),
    )
    print(
        "log L(h|s) = ",
        loglikelihood(dd_h, f_c_h, "d", dd_s, f_c, "d", f_l, f_c),
    )
