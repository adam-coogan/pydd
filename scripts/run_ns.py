import pickle

import click
import dynesty
from dynesty import plotting as dyplot
from dynesty.results import Results
import jax.numpy as jnp
from scipy.optimize import root_scalar

from plot_measurability import labels, quantiles_2d, smooth
from pydd.binary import DynamicDress, MSUN, PC, get_rho_s
from utils import (
    get_loglikelihood_fn,
    get_loglikelihood_fn_v,
    get_ptform,
    get_ptform_v,
    rho_6T_to_rho6,
    rho_6_to_rho6T,
    setup_system,
)


"""
Runs nested sampler for a system
"""


def run_ns_v(
    dd_s: DynamicDress, base_path: str, dM_chirp_v_min, dM_chirp_v_max
) -> Results:
    """
    Runs nested sampling for a vacuum system given a signal system dd_s. Saves
    results and posterior plots.
    """
    # Setup
    M_chirp_MSUN_range_v = (
        dd_s.M_chirp / MSUN + dM_chirp_v_min,
        dd_s.M_chirp / MSUN + dM_chirp_v_max,
    )
    ptform_v = lambda u: get_ptform_v(u, M_chirp_MSUN_range_v)
    loglikelihood_v = get_loglikelihood_fn_v(dd_s)

    # Run
    sampler_v = dynesty.NestedSampler(
        loglikelihood_v, ptform_v, 1, nlive=100, bound="multi"
    )
    sampler_v.run_nested()
    results_v = sampler_v.results

    # Save
    with open(f"ns/{base_path}-v.pkl", "wb") as output:
        pickle.dump(results_v, output, pickle.HIGHEST_PROTOCOL)

    # Plot
    cfig = dyplot.cornerplot(results_v, labels=[r"$\mathcal{M}$ [M$_\odot$]"])[0]
    cfig.savefig(f"figures/{base_path}-v.pdf")
    cfig.savefig(f"figures/{base_path}-v.png")

    return results_v


def run_ns(
    dd_s: DynamicDress,
    base_path: str,
    rho_6T_min,
    rho_6T_max,
    dM_chirp_abs,
    gamma_s,
    rho_6T,
) -> Results:
    """
    Runs nested sampling for a dark dress system given a signal system dd_s.
    Saves results and posterior plots.
    """
    # Setup
    gamma_s_range = [2.25, 2.5]
    rho_6T_range = [rho_6T_min, rho_6T_max]
    log10_q_range = [-3.5, -2.5]
    dM_chirp_MSUN_range = [-dM_chirp_abs, dM_chirp_abs]
    ptform = lambda u: get_ptform(
        u, gamma_s_range, rho_6T_range, log10_q_range, dM_chirp_MSUN_range, dd_s
    )
    loglikelihood = get_loglikelihood_fn(dd_s)

    # Run
    sampler = dynesty.NestedSampler(loglikelihood, ptform, 4, nlive=500)
    sampler.run_nested()
    results = sampler.results

    # Save
    with open(f"ns/{base_path}.pkl", "wb") as output:
        pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)

    # Plot
    cfig = dyplot.cornerplot(
        results,
        labels=labels,
        quantiles_2d=quantiles_2d,
        smooth=smooth,
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
@click.option("--rho_6t", default=0.01, help="rho_6 / (10^16 MSUN / PC^3)")
@click.option("--gamma_s", default=7 / 3)
@click.option("--rho_6t_min", default=0.0)
@click.option("--rho_6t_max", default=0.035)
@click.option(
    "--dm_chirp_abs",
    default=2e-3,
    help="range above and below true chirp mass to use as prior",
)
@click.option(
    "--calc-bf/--no-calc-bf",
    default=False,
    help="calculate Bayes factor by running nested sampling for vacuum system",
)
@click.option(
    "--dm_chirp_v_min", default=0.0, help="lower bound for vacuum chirp mass prior"
)
@click.option(
    "--dm_chirp_v_max", default=2e-3, help="upper bound for vacuum chirp mass prior"
)
@click.option(
    "--suffix", default="_test", help="suffix for generated figures and bayes factor"
)
def run(
    rho_6t,
    gamma_s,
    rho_6t_min,
    rho_6t_max,
    dm_chirp_abs,
    calc_bf,
    dm_chirp_v_min,
    dm_chirp_v_max,
    suffix,
):
    """
    Runs vacuum and dark dress nested sampling, saves results and plots, and
    computes and saves the Bayes factor.
    """
    base_path = f"rho_6T={rho_6t:g}_gamma_s={gamma_s:g}{suffix}"
    print("Base filename for plots, results and Bayes factor:", base_path)
    rho_6 = rho_6T_to_rho6(rho_6t)
    dd_s = setup_system(gamma_s, rho_6)[0]

    # Run nested sampling
    results = run_ns(
        dd_s, base_path, rho_6t_min, rho_6t_max, dm_chirp_abs, gamma_s, rho_6t
    )
    print("Dark dress results:\n", results.summary())

    if calc_bf:
        results_v = run_ns_v(dd_s, base_path, dm_chirp_v_min, dm_chirp_v_max)
        print("Vacuum results:\n", results_v.summary())

        # Correction for rho_6T prior, extending up to the most extreme rho_6T value
        # we've done simulations for
        rho_6T_upper = rho_6_to_rho6T(
            root_scalar(
                lambda rho: get_rho_s(rho, 1e5 * MSUN, 2.5) - 1000 * MSUN / PC ** 3,
                bracket=(1e-3, 1e2),
                rtol=1e-15,
                xtol=1e-100,
            ).root
        )
        rho_6T_fact = (rho_6t_max - rho_6t_min) / rho_6T_upper

        # Correction for different chirp mass priors
        dM_fact = 2 * dm_chirp_abs / (dm_chirp_v_max - dm_chirp_v_min)

        # Compute uncorrected and corrected Bayes factors
        bayes_fact_ns = jnp.exp(results.logz[-1]) / jnp.exp(results_v.logz[-1])
        bayes_fact = dM_fact * rho_6T_fact * bayes_fact_ns
        jnp.savez(
            f"ns/{base_path}-bayes.npz",
            bayes_fact_ns=bayes_fact_ns,
            bayes_fact=bayes_fact,
        )
        print(f"The Bayes factor is {bayes_fact}")


if __name__ == "__main__":
    run()
