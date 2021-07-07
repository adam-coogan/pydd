import jax
from jax import random
import jax.numpy as jnp

from pydd.binary import hypgeom_jax, hypgeom_scipy


def test_hypgeom_interps():
    """
    Make sure the interpolated hypergeometric function agrees to 0.001-0.01%
    with scipy.
    """
    n_samples = 1000
    key = random.PRNGKey(1234)

    # b > 0
    key, b_key, z_key = random.split(key, 3)
    bs = random.uniform(b_key, (n_samples,), minval=0.5, maxval=1.99)
    log10_abs_zs = random.uniform(z_key, (n_samples,), minval=-4.0, maxval=4.0)
    v_scipy = hypgeom_scipy(bs, -(10 ** log10_abs_zs))
    v_jax = jax.lax.map(lambda args: hypgeom_jax(*args), (bs, -(10 ** log10_abs_zs)))
    assert jnp.allclose(v_scipy, v_jax, rtol=1e-5)

    # b < 0
    # The b values here are restricted, but cover the range relevant for dark
    # dress calculations
    key, b_key, z_key = random.split(key, 3)
    bs = random.uniform(b_key, (1000,), minval=-1.95, maxval=-1.1)
    log10_abs_zs = random.uniform(z_key, (1000,), minval=-4.0, maxval=4.0)
    v_scipy = hypgeom_scipy(bs, -(10 ** log10_abs_zs))
    v_jax = jax.lax.map(lambda args: hypgeom_jax(*args), (bs, -(10 ** log10_abs_zs)))
    assert jnp.allclose(v_scipy, v_jax, rtol=1e-4)
