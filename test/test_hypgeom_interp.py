import time

import jax
import jax.numpy as jnp
import numpy as np

from pydd.binary import hypgeom_jax, hypgeom_scipy

"""
Make sure the interpolated hypergeometric function agrees to 0.01% with
scipy.

Assumes gamma_s in [2, 3] and gamma_e = 5/2.
"""


def test_hypgeom():
    n = 100000
    rtol = 1e-4
    bs = np.random.rand(n) * (2 - 1.6) + 1.6
    zs = jnp.array(-(10 ** (np.random.rand(n) * (7 + 8) - 8)))

    t_start = time.time()
    vals_scipy = jnp.array(hypgeom_scipy(bs, zs))
    t_end = time.time()
    print(f"scipy timing: {t_end - t_start}")

    jax.vmap(hypgeom_jax, in_axes=(0, 0))(bs[:5], zs[:5]).block_until_ready()
    t_start = time.time()
    vals_jax = jax.vmap(hypgeom_jax, in_axes=(0, 0))(bs, zs).block_until_ready()
    t_end = time.time()
    print(f"jax timing: {t_end - t_start}")

    return jnp.allclose(vals_scipy, vals_jax, rtol=rtol, atol=0)


# def test_hypgeom_interps():
#     n_samples = 1000
#     key = random.PRNGKey(1234)
#
#     # b > 0
#     key, b_key, z_key = random.split(key, 3)
#     bs = random.uniform(b_key, (n_samples,), minval=0.5, maxval=1.99)
#     log10_abs_zs = random.uniform(z_key, (n_samples,), minval=-4.0, maxval=4.0)
#     v_scipy = hypgeom_scipy(bs, -(10 ** log10_abs_zs))
#     v_jax = jax.lax.map(lambda args: hypgeom_jax(*args), (bs, -(10 ** log10_abs_zs)))
#     assert jnp.allclose(v_scipy, v_jax, rtol=1e-5)
#
#     # b < 0
#     # The b values here are restricted, but cover the range relevant for dark
#     # dress calculations
#     key, b_key, z_key = random.split(key, 3)
#     bs = random.uniform(b_key, (1000,), minval=-1.95, maxval=-1.1)
#     log10_abs_zs = random.uniform(z_key, (1000,), minval=-4.0, maxval=4.0)
#     v_scipy = hypgeom_scipy(bs, -(10 ** log10_abs_zs))
#     v_jax = jax.lax.map(lambda args: hypgeom_jax(*args), (bs, -(10 ** log10_abs_zs)))
#     assert jnp.allclose(v_scipy, v_jax, rtol=1e-4)

if __name__ == "__main__":
    test_hypgeom()
