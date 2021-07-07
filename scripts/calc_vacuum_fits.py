from math import pi
import os

import click
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize_scalar
from tqdm.auto import tqdm

from pydd.analysis import calculate_SNR, calculate_match_unnormd_fft
from pydd.binary import (
    DynamicDress,
    MSUN,
    Phi_to_c,
    VacuumBinary,
    convert,
    get_M_chirp,
    get_rho_s,
)
from utils import (
    M_1_BM,
    M_2_BM,
    get_loglikelihood_v,
    rho_6T_to_rho6,
    rho_6_to_rho6T,
    setup_system,
)


def fit_v(dd_s: DynamicDress, f_l) -> VacuumBinary:
    """
    Find best-fit vacuum system.
    """
    fun = lambda x: -get_loglikelihood_v([x], dd_s, f_l)
    bracket = (dd_s.M_chirp / MSUN, dd_s.M_chirp / MSUN + 1e-1)
    res = minimize_scalar(fun, bracket, tol=1e-15)

    assert res.success

    return VacuumBinary(
        res.x * MSUN,
        dd_s.Phi_c,
        dd_s.tT_c,
        dd_s.dL_iota,
        dd_s.f_c,
    )


def get_M_chirp_err(dd_v: VacuumBinary, dd_s: DynamicDress, f_l) -> jnp.ndarray:
    """
    Returns an estimate of the error on the best-fit vacuum system's chirp
    mass.
    """
    M_chirp_MSUN = dd_v.M_chirp / MSUN
    M_chirp_MSUN_grid = jnp.linspace(M_chirp_MSUN - 2e-5, M_chirp_MSUN + 2e-5, 500)
    loglikelihoods = jax.lax.map(
        lambda x: get_loglikelihood_v([x], dd_s, f_l), M_chirp_MSUN_grid
    )
    norm = jnp.trapz(jnp.exp(loglikelihoods), M_chirp_MSUN_grid)
    return jnp.sqrt(
        jnp.trapz(
            (M_chirp_MSUN_grid - M_chirp_MSUN) ** 2 * jnp.exp(loglikelihoods),
            M_chirp_MSUN_grid,
        )
        / norm
    )


@click.command()
@click.option("--n_rho", default=4)
@click.option("--n_gamma", default=3)
@click.option(
    "--rho_6t_min", default=1e-4, help="min value of rho_6 / (10^16 MSUN / PC^3)"
)
@click.option(
    "--rho_6t_max", default=1.0, help="max value of rho_6 / (10^16 MSUN / PC^3)"
)
@click.option("--gamma_s_min", default=2.25, help="min value of gamma_s")
@click.option("--gamma_s_max", default=2.5, help="max value of gamma_s")
@click.option("--suffix", default="_test", help="suffix for output file")
def run(n_rho, n_gamma, rho_6t_min, rho_6t_max, gamma_s_min, gamma_s_max, suffix):
    """
    For dark dresses with fixed BH masses and various rho_6 and gamma_s values,
    computes the naive dephasing, best-fit vacuum system and its dephasing,
    chirp mass bias and SNR loss.
    """
    path = os.path.join("vacuum_fits", f"vacuum_fits{suffix}.npz")
    rho_6s = jnp.geomspace(
        rho_6T_to_rho6(rho_6t_min), rho_6T_to_rho6(rho_6t_max), n_rho
    )
    gamma_ss = jnp.linspace(gamma_s_min, gamma_s_max, n_gamma)

    results = {
        k: np.full([n_rho, n_gamma], np.nan)
        for k in [
            "snrs",
            "rho_ss",
            "dN_naives",
            "matches",
            "dNs",
            "M_chirp_MSUN_bests",
            "M_chirp_MSUN_best_errs",
        ]
    }

    for i, rho_6 in enumerate(tqdm(rho_6s)):
        for j, gamma_s in enumerate(gamma_ss):
            dd_s, f_l = setup_system(gamma_s, rho_6)

            results["snrs"][i, j] = calculate_SNR(dd_s, f_l, dd_s.f_c, 3000)
            results["rho_ss"][i, j] = get_rho_s(rho_6, M_1_BM, gamma_s)

            # Dephasing relative to system with no DM
            dd_v = convert(dd_s, VacuumBinary)
            results["dN_naives"][i, j] = (Phi_to_c(f_l, dd_v) - Phi_to_c(f_l, dd_s)) / (
                2 * pi
            )

            # Don't waste effort on systems that are very hard to fit
            if rho_6_to_rho6T(rho_6) < 5e-2:
                dd_v_best = fit_v(dd_s, f_l)
                results["M_chirp_MSUN_bests"][i, j] = dd_v_best.M_chirp / MSUN
                results["matches"][i, j] = calculate_match_unnormd_fft(
                    dd_v_best, dd_s, f_l, dd_s.f_c, 100_000
                )
                results["dNs"][i, j] = (
                    Phi_to_c(f_l, dd_v_best) - Phi_to_c(f_l, dd_s)
                ) / (2 * pi)
                results["M_chirp_MSUN_best_errs"][i, j] = get_M_chirp_err(
                    dd_v_best, dd_s, f_l
                )

    results = {k: jnp.array(v) for k, v in results.items()}
    print(results)

    jnp.savez(
        path,
        m_1=M_1_BM,
        m_2=M_2_BM,
        M_chirp_MSUN=get_M_chirp(M_1_BM, M_2_BM) / MSUN,
        rho_6s=rho_6s,
        gamma_ss=gamma_ss,
        **results,
    )
    print(f"Results saved to {path}")


if __name__ == "__main__":
    run()
