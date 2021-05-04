from jax import jit
import jax.numpy as jnp
import numpy.random as np_random

from pydd.binary import get_hypgeom_interps, hypgeom_scipy


"""
TODO: revise. Wrong parameter ranges.
"""


def test_hypgeom_interps(n_samples=10000, far_frac=0.005):
    interp_pos, interp_neg = get_hypgeom_interps(5000, 4950)
    interp_pos_jit, interp_neg_jit = jit(interp_pos), jit(interp_neg)

    bs = jnp.array(np_random.uniform(1.55, 1.99, size=n_samples))
    log10_abs_zs = jnp.array(np_random.uniform(-4, 4, size=n_samples))
    zs = -(10 ** log10_abs_zs)

    vals_pos = interp_pos_jit(bs, zs)
    vals_neg = interp_neg_jit(-bs, zs)

    assert (
        ~jnp.isclose(hypgeom_scipy(bs, zs), vals_pos, atol=0)
    ).sum() < far_frac * n_samples
    assert (
        ~jnp.isclose(hypgeom_scipy(-bs, zs), vals_neg, atol=0)
    ).sum() < far_frac * n_samples
