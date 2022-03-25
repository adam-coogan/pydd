from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from scipy.optimize import root_scalar

from .analysis import calculate_SNR
from .binary import (
    DynamicDress,
    GAMMA_S_PBH,
    PC,
    get_M_chirp,
    get_f_isco,
    get_f_range,
    get_rho_6_pbh,
)


Array = jnp.ndarray


def get_target_dynamicdress(
    m_1: float,
    m_2: float,
    gamma_s: float,
    rho_6: float,
    t_obs: float,
    snr_thresh: float,
    S_n: Callable[[Array], Array],
    f_range_n: Tuple[float, float],
) -> Tuple[DynamicDress, Tuple[float, float]]:
    """
    Creates a dark dress with correct SNR for given detector and observing time.

    Returns:
        The dark dress and frequency range corresponding to an observing time
        of ``t_obs`` before coalescence.
    """
    m_1 = jnp.array(m_1)
    m_2 = jnp.array(m_2)
    tT_C = jnp.array(0.0)
    Phi_c = jnp.array(0.0)
    _dd_d = DynamicDress(
        jnp.array(gamma_s),
        jnp.array(rho_6),
        get_M_chirp(m_1, m_2),
        jnp.array(m_2 / m_1),
        Phi_c,
        tT_c=tT_C,
        dL=jnp.array(100e6 * PC),  # initial guess
        f_c=get_f_isco(m_1),
    )

    # Frequency range and grids
    f_range_d = get_f_range(_dd_d, t_obs)
    fs = jnp.linspace(
        max(f_range_d[0], f_range_n[0]), min(f_range_d[1], f_range_n[1]), 10_000
    )

    # Get dL
    _fn = jax.jit(
        lambda dL: calculate_SNR(
            DynamicDress(
                _dd_d.gamma_s,
                _dd_d.rho_6,
                _dd_d.M_chirp,
                _dd_d.q,
                _dd_d.Phi_c,
                _dd_d.tT_c,
                dL,
                _dd_d.f_c,
            ),
            fs,
            S_n,
        )
    )
    res = root_scalar(
        lambda dL: (_fn(dL) - snr_thresh), bracket=(0.1e6 * PC, 100000e6 * PC)
    )
    assert res.converged
    dL = res.root

    # Signal system
    dd_d = DynamicDress(
        _dd_d.gamma_s,
        _dd_d.rho_6,
        _dd_d.M_chirp,
        _dd_d.q,
        _dd_d.Phi_c,
        _dd_d.tT_c,
        dL,
        _dd_d.f_c,
    )

    return dd_d, f_range_d


def get_target_pbh_dynamicdress(
    m_1: float,
    m_2: float,
    t_obs: float,
    snr_thresh: float,
    S_n: Callable[[Array], Array],
    f_range_n: Tuple[float, float],
) -> Tuple[DynamicDress, Tuple[float, float]]:
    return get_target_dynamicdress(
        m_1, m_2, GAMMA_S_PBH, get_rho_6_pbh(m_1), t_obs, snr_thresh, S_n, f_range_n
    )
