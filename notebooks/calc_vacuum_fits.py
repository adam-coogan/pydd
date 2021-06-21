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


def loop_fun(rho_6, GAMMA_SS, M_1, M_CHIRP_MSUN, bracket):
    snrs = np.full([len(GAMMA_SS)], np.nan)
    rho_ss = np.full([len(GAMMA_SS)], np.nan)
    dN_naives = np.full([len(GAMMA_SS)], np.nan)
    matches = np.full([len(GAMMA_SS)], np.nan)
    dNs = np.full([len(GAMMA_SS)], np.nan)
    M_chirp_MSUN_bests = np.full([len(GAMMA_SS)], np.nan)
    M_chirp_MSUN_best_errs = np.full([len(GAMMA_SS)], np.nan)

    for i, gamma_s in enumerate(GAMMA_SS):
        dd_s, f_l = setup_system(gamma_s, rho_6)

        snrs[i] = calculate_SNR(dd_s, f_l, dd_s.f_c, 3000)

        rho_ss[i] = get_rho_s(rho_6, M_1, gamma_s)

        # Dephasing relative to system with no DM
        dd_v = VacuumBinary(
            jnp.array(M_CHIRP_MSUN * MSUN),
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
@click.option("--path", default="vacuum_fits.npz")
@click.option("--n_rho", default=4)
@click.option("--n_gamma", default=3)
def run(path, n_rho, n_gamma):
    M_1 = 1e3 * MSUN
    M_2 = 1 * MSUN
    M_CHIRP_MSUN = get_M_chirp(M_1, M_2) / MSUN
    RHO_6S = jnp.geomspace(rho_6T_to_rho6(1e-4), rho_6T_to_rho6(1.0), n_rho)
    GAMMA_SS = jnp.linspace(2.25, 2.5, n_gamma)
    bracket = (M_CHIRP_MSUN, M_CHIRP_MSUN + 1e-1)

    M_chirp_MSUN_bests = np.full([len(RHO_6S), len(GAMMA_SS)], np.nan)
    M_chirp_MSUN_best_errs = np.full([len(RHO_6S), len(GAMMA_SS)], np.nan)
    matches = np.full([len(RHO_6S), len(GAMMA_SS)], np.nan)
    snrs = np.full([len(RHO_6S), len(GAMMA_SS)], np.nan)
    dNs = np.full([len(RHO_6S), len(GAMMA_SS)], np.nan)
    dN_naives = np.full([len(RHO_6S), len(GAMMA_SS)], np.nan)
    rho_ss = np.full([len(RHO_6S), len(GAMMA_SS)], np.nan)

    results = map(
        lambda rho_6: loop_fun(rho_6, GAMMA_SS, M_1, M_CHIRP_MSUN, bracket),
        tqdm(RHO_6S),
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
        M_1=M_1,
        M_2=M_2,
        M_CHIRP_MSUN=M_CHIRP_MSUN,
        RHO_6S=RHO_6S,
        GAMMA_SS=GAMMA_SS,
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
