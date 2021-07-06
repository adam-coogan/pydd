from math import pi
import os

import click
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize_scalar
from tqdm.auto import tqdm

from pydd.analysis import calculate_SNR, calculate_match_unnormd_fft
from pydd.binary import MSUN, Phi_to_c, VacuumBinary, get_M_chirp, get_rho_s
from utils import (
    get_loglikelihood_v,
    rho_6T_to_rho6,
    rho_6_to_rho6T,
    setup_system,
)


def loop_fun(rho_6, gamma_ss, m_1, M_chirp_MSUN, bracket):
    """
    Loops over gamma_s values with other parameters fixed and computes the
    naive dephasing, then finds the best-fit vacuum system and computes the
    dephasing, SNR loss and chirp mass bias.
    """
    n_gamma = len(gamma_ss)
    snrs = np.full([n_gamma], np.nan)
    rho_ss = np.full([n_gamma], np.nan)
    dN_naives = np.full([n_gamma], np.nan)
    matches = np.full([n_gamma], np.nan)
    dNs = np.full([n_gamma], np.nan)
    M_chirp_MSUN_bests = np.full([n_gamma], np.nan)
    M_chirp_MSUN_best_errs = np.full([n_gamma], np.nan)

    for i, gamma_s in enumerate(gamma_ss):
        dd_s, f_l = setup_system(gamma_s, rho_6)

        snrs[i] = calculate_SNR(dd_s, f_l, dd_s.f_c, 3000)

        rho_ss[i] = get_rho_s(rho_6, m_1, gamma_s)

        # Dephasing relative to system with no DM
        dd_v = VacuumBinary(
            jnp.array(M_chirp_MSUN * MSUN),
            dd_s.Phi_c,
            dd_s.tT_c,
            dd_s.dL_iota,
            dd_s.f_c,
        )
        dN_naives[i] = (Phi_to_c(f_l, dd_v) - Phi_to_c(f_l, dd_s)) / (2 * pi)

        if rho_6_to_rho6T(rho_6) < 5e-2:
            # Find best-fit vacuum system
            fun = lambda x: -get_loglikelihood_v([x], dd_s, f_l)
            res = minimize_scalar(fun, bracket, tol=1e-15)
            assert res.success
            M_chirp_MSUN_bests[i] = res.x
            dd_v_best = VacuumBinary(
                M_chirp_MSUN_bests[i] * MSUN,
                dd_s.Phi_c,
                dd_s.tT_c,
                dd_s.dL_iota,
                dd_s.f_c,
            )

            # <V | D> match
            matches[i] = calculate_match_unnormd_fft(
                dd_v_best, dd_s, f_l, dd_s.f_c, 100_000
            )

            # True dephasing
            dNs[i] = (Phi_to_c(f_l, dd_v_best) - Phi_to_c(f_l, dd_s)) / (2 * pi)

            # Compute M_chirp error bars
            M_chirp_MSUN_grid = jnp.linspace(
                M_chirp_MSUN_bests[i] - 2e-5, M_chirp_MSUN_bests[i] + 2e-5, 500
            )
            loglikelihoods = jax.lax.map(
                lambda x: get_loglikelihood_v([x], dd_s, f_l), M_chirp_MSUN_grid
            )
            norm = jnp.trapz(jnp.exp(loglikelihoods), M_chirp_MSUN_grid)
            M_chirp_MSUN_best_errs[i] = jnp.sqrt(
                jnp.trapz(
                    (M_chirp_MSUN_grid - M_chirp_MSUN_bests[i]) ** 2
                    * jnp.exp(loglikelihoods),
                    M_chirp_MSUN_grid,
                )
                / norm
            )

    return snrs, rho_ss, dN_naives, matches, dNs, M_chirp_MSUN_best_errs


@click.command()
@click.option("--m1", default=1e3, help="m1 / MSUN")
@click.option("--m2", default=1.4, help="m2 / MSUN")
@click.option("--path", default="vacuum_fits.npz")
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
def run(m1, m2, path, n_rho, n_gamma, rho_6t_min, rho_6t_max, gamma_s_min, gamma_s_max):
    """
    For dark dresses with fixed BH masses and various rho_6 and gamma_s values,
    computes the naive dephasing, best-fit vacuum system and its dephasing,
    chirp mass bias and SNR loss.
    """
    m_1 = m1 * MSUN
    m_2 = m2 * MSUN
    M_chirp_MSUN = get_M_chirp(m_1, m_2) / MSUN
    rho_6s = jnp.geomspace(
        rho_6T_to_rho6(rho_6t_min), rho_6T_to_rho6(rho_6t_max), n_rho
    )
    gamma_ss = jnp.linspace(gamma_s_min, gamma_s_max, n_gamma)
    bracket = (M_chirp_MSUN, M_chirp_MSUN + 1e-1)

    M_chirp_MSUN_bests = np.full([n_rho, n_gamma], np.nan)
    M_chirp_MSUN_best_errs = np.full([n_rho, n_gamma], np.nan)
    matches = np.full([n_rho, n_gamma], np.nan)
    snrs = np.full([n_rho, n_gamma], np.nan)
    dNs = np.full([n_rho, n_gamma], np.nan)
    dN_naives = np.full([n_rho, n_gamma], np.nan)
    rho_ss = np.full([n_rho, n_gamma], np.nan)

    results = map(
        lambda rho_6: loop_fun(rho_6, gamma_ss, m_1, M_chirp_MSUN, bracket),
        tqdm(rho_6s),
    )

    for i, result_i in enumerate(results):
        (
            snrs[i],
            rho_ss[i],
            dN_naives[i],
            matches[i],
            dNs[i],
            M_chirp_MSUN_best_errs[i],
        ) = result_i

    M_chirp_MSUN_bests = jnp.array(M_chirp_MSUN_bests)
    M_chirp_MSUN_best_errs = jnp.array(M_chirp_MSUN_best_errs)
    matches = jnp.array(matches)
    snrs = jnp.array(snrs)
    dNs = jnp.array(dNs)
    dN_naives = jnp.array(dN_naives)
    rho_ss = jnp.array(rho_ss)

    jnp.savez(
        os.path.join("vacuum_fits", path),
        M_1=m_1,
        M_2=m_2,
        M_chirp_MSUN=M_chirp_MSUN,
        rho_6s=rho_6s,
        gamma_ss=gamma_ss,
        M_chirp_MSUN_bests=M_chirp_MSUN_bests,
        M_chirp_MSUN_best_errs=M_chirp_MSUN_best_errs,
        matches=matches,
        snrs=snrs,
        dNs=dNs,
        dN_naives=dN_naives,
        rho_ss=rho_ss,
    )


if __name__ == "__main__":
    run()
