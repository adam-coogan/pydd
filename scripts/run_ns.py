import pickle

import click
import dynesty
from dynesty import plotting as dyplot
import jax.numpy as jnp
from scipy.optimize import root_scalar

from pydd.binary import MSUN, PC, get_rho_s, Binary
from utils import (
    get_loglikelihood,
    get_loglikelihood_v,
    get_ptform,
    get_ptform_v,
    labels,
    quantiles,
    quantiles_2d,
    rho_6T_to_rho6,
    rho_6_to_rho6T,
    setup_system,
)


def run_ns_v(
    dd_s: Binary, f_l, base_path: str, dM_chirp_v_min, dM_chirp_v_max
) -> dynesty.results.Results:
    """
    Runs nested sampling for a vacuum system given a signal system dd_s.
    """
    # Setup
    M_chirp_MSUN_range_v = (
        dd_s.M_chirp / MSUN + dM_chirp_v_min,
        dd_s.M_chirp / MSUN + dM_chirp_v_max,
    )
    ptform_v = lambda u: get_ptform_v(u, M_chirp_MSUN_range_v)
    loglikelihood_v = lambda x: get_loglikelihood_v(x, dd_s, f_l)

    # Run
    sampler_v = dynesty.NestedSampler(
        loglikelihood_v, ptform_v, 1, nlive=100, bound="multi"
    )
    sampler_v.run_nested()
    results_v = sampler_v.results
    print("Vacuum results:\n", results_v.summary())

    # Save
    with open(f"ns/{base_path}-v.pkl", "wb") as output:
        pickle.dump(results_v, output, pickle.HIGHEST_PROTOCOL)

    # Plot
    cfig = dyplot.cornerplot(
        results_v, labels=[r"$\mathcal{M}$ [M$_\odot$]"], quantiles=[1 - 0.95, 0.95]
    )[0]
    cfig.savefig(f"figures/{base_path}-v.pdf")
    cfig.savefig(f"figures/{base_path}-v.png")

    return results_v


def run_ns(
    dd_s: Binary,
    f_l,
    base_path: str,
    rho_6T_min,
    rho_6T_max,
    dM_chirp_abs,
    gamma_s,
    rho_6T,
) -> dynesty.results.Results:
    """
    Runs nested sampling for a dark dress system given a signal system dd_s.
    """
    # Setup
    gamma_s_range = [2.25, 2.5]
    rho_6T_range = [rho_6T_min, rho_6T_max]
    log10_q_range = [-3.5, -2.5]
    dM_chirp_MSUN_range = [-dM_chirp_abs, dM_chirp_abs]
    ptform = lambda u: get_ptform(
        u, gamma_s_range, rho_6T_range, log10_q_range, dM_chirp_MSUN_range, dd_s
    )
    loglikelihood = lambda x: get_loglikelihood(x, dd_s, f_l)

    # Run
    sampler = dynesty.NestedSampler(loglikelihood, ptform, 4, nlive=500)
    sampler.run_nested()
    results = sampler.results
    print("Dark dress results:\n", results.summary())

    # Save
    with open(f"ns/{base_path}.pkl", "wb") as output:
        pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)

    # Plot
    cfig = dyplot.cornerplot(
        results,
        labels=labels,
        quantiles=quantiles,
        quantiles_2d=quantiles_2d,
        smooth=0.015,
        truths=(
            gamma_s,
            rho_6T,
            dd_s.M_chirp / MSUN,
            jnp.log10(dd_s.q),
        ),
        span=[1, 1, 1, 1],
    )[0]
    cfig.tight_layout()
    cfig.savefig(f"figures/{base_path}.pdf")
    cfig.savefig(f"figures/{base_path}.png")

    return results


@click.command()
@click.option("--m1", default=1e3, help="m1 / MSUN")
@click.option("--m2", default=1.4, help="m2 / MSUN")
@click.option("--dL", default=85e6, help="dL / PC")
@click.option("--rho_6t", default=0.01, help="rho_6 / (10^16 MSUN / PC^3)")
@click.option("--gamma_s", default=7 / 3)
@click.option("--rho_6t_min", default=0.0)
@click.option("--rho_6t_max", default=0.035)
@click.option("--dm_chirp_abs", default=2e-3)
@click.option("--dm_chirp_v_min", default=0.0)
@click.option("--dm_chirp_v_max", default=2e-3)
@click.option(
    "--suffix", default="_test", help="suffix for generated figures and bayes factor"
)
def run(
    m1,
    m2,
    dL,
    rho_6t,
    gamma_s,
    rho_6t_min,
    rho_6t_max,
    dm_chirp_abs,
    dm_chirp_v_min,
    dm_chirp_v_max,
    suffix,
):
    """
    Runs vacuum and dark dress nested sampling, saves results and plots, and
    computes and saves the Bayes factor.
    """
    base_path = f"rho_6T={rho_6t:g}_gamma_s={gamma_s:g}{suffix}"
    print("Base path:", base_path)

    rho_6 = rho_6T_to_rho6(rho_6t)
    dd_s, f_l = setup_system(gamma_s, rho_6, m1, m2, dL)

    # Run nested sampling
    results_v = run_ns_v(dd_s, f_l, base_path, dm_chirp_v_min, dm_chirp_v_max)
    results = run_ns(
        dd_s, f_l, base_path, rho_6t_min, rho_6t_max, dm_chirp_abs, gamma_s, rho_6t
    )

    # Correct for rho_6T prior, extending up to the most extreme rho_6T value we've considered
    rho_6T_upper = rho_6_to_rho6T(
        root_scalar(
            lambda rho: get_rho_s(rho, 1e5 * MSUN, 2.5) - 1000 * MSUN / PC ** 3,
            bracket=(1e-3, 1e2),
            rtol=1e-15,
            xtol=1e-100,
        ).root
    )
    rho_6T_fact = (rho_6t_max - rho_6t_min) / rho_6T_upper
    # Correct for different chirp mass priors
    dM_fact = 2 * dm_chirp_abs / (dm_chirp_v_max - dm_chirp_v_min)

    # Compute uncorrected and corrected Bayes factors
    bayes_fact_ns = jnp.exp(results.logz[-1]) / jnp.exp(results_v.logz[-1])
    bayes_fact = dM_fact * rho_6T_fact * bayes_fact_ns
    print(f"The Bayes factor is {bayes_fact}")
    jnp.savez(
        f"ns/{base_path}-bayes.npz", bayes_fact_ns=bayes_fact_ns, bayes_fact=bayes_fact
    )


if __name__ == "__main__":
    run()
