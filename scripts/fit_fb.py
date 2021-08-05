from collections import defaultdict
from math import pi
import os
from typing import NamedTuple
import warnings

from jax.config import config
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, root_scalar
from tqdm.auto import tqdm

import pydd.binary as bi
from pydd.binary import MSUN, PC, YR, get_f_isco, get_rho_s

config.update("jax_disable_jit", True)


"""
Fits `f_b` to `HaloFeedback` runs with nonlinear least squares. If you are not
Adam Coogan or Bradley Kavanagh, you will have to adapt this in a few places
marked with 'CHANGE' to work with your runs.

Note: the fitting function changes `get_f_b_d` in a hacky way, so don't try to
use the dependent `pydd` functions after running it!

Produces `fb_fits/fb_fits.npz`.
"""


run_dir = "/nfs/scratch/kavanagh-runs/"  # <--- CHANGE
t_obs_lisa = 5 * YR


class RunParams(NamedTuple):
    """
    Container for HaloFeedback runs
    """

    m_1: float  # MSUN
    m_2: float  # MSUN
    rho_n: float  # rho_s [MSUN / PC**3] or rho6 [1e16 * MSUN / PC**3]
    gamma_s: float
    rho_kind: str  # 'rho_s' or 'rho_6'
    directory: str
    id_str: str


class RunFit(NamedTuple):
    """
    Container for f_b fit results
    """

    m_1: float  # MSUN
    m_2: float  # MSUN
    rho_s: float  # MSUN / PC**3
    rho6T: float  # 1e16 * MSUN / PC**3
    gamma_s: float
    f_b: float  # Hz
    id_str: str


def load_run(run, kind):
    """
    Returns
    - Phi(f)
    - f(t)
    """
    fname_root = os.path.join(run.directory, run.id_str)
    # Maybe need to CHANGE file names
    if kind == "d":
        fname = os.path.join(fname_root, "output_dynamic_dress_" + run.id_str + ".dat")
    elif kind == "s":
        fname = os.path.join(fname_root, "output_static_dress_" + run.id_str + ".dat")
    elif kind == "v":
        fname = os.path.join(fname_root, "output_vacuum.dat")
    print(f"Loading ID {run.id_str}")

    t, f = np.loadtxt(fname, unpack=True, usecols=(0, 2))  # type: ignore
    f *= 2  # Double it for the GWs

    # Truncate at the isco frequency:
    f_isco = get_f_isco(run.m_1 * MSUN)
    if f[-1] < f_isco:
        warnings.warn("waveform doesn't extend to ISCO")

    N = cumtrapz(f, t, initial=0)
    return (
        interp1d(f, 2 * pi * (N[-1] - N), bounds_error=False, fill_value=0.0),
        interp1d(t - t[-1], f, bounds_error=False, fill_value=0.0),
    )


def fit_f_b(m_1, m_2, rho_s, gamma_s, fs, Phi_v, Phi_d):
    def dPhi(f, f_b):
        # HACK: replace function
        bi.get_f_b_d = lambda *args, **kwargs: jnp.array(f_b)
        dd = bi.make_dynamic_dress(
            m_1 * MSUN, m_2 * MSUN, rho_s * MSUN / PC ** 3, gamma_s
        )
        dd = dd._replace(f_c=Phi_d.x[-1])
        vb = bi.convert(dd, bi.VacuumBinary)
        return bi.Phi_to_c(f, vb) - bi.Phi_to_c(f, dd)

    dPhis = Phi_v(fs) - Phi_d(fs)
    f_b = curve_fit(dPhi, fs, dPhis, p0=[0.2], bounds=[0, np.inf])[0][0]
    return f_b, lambda f: dPhi(f, f_b)


def get_rho6T(m_1, rho_s, gamma_s):
    """
    Arguments
    - m_1: IMBH mass [kg]
    - rho_s: Eda et al normalization [kg / m**3]
    - gamma_s: slope
    """
    return root_scalar(
        lambda rho: get_rho_s(rho, m_1, gamma_s) - rho_s,
        bracket=(1e-5, 1e-1),
        rtol=1e-15,
        xtol=1e-100,
    ).root


def fit_run(run, fig_path=None):
    """
    Returns
    - f_b: fit break frequency [Hz]
    """
    # Load waveforms
    if run.rho_kind == "rho6":
        rho6T = run.rho_n
        rho_s = bi.get_rho_s(
            rho6T * 1e16 * MSUN / PC ** 3, run.m_1 * MSUN, run.gamma_s
        ) / (MSUN / PC ** 3)
    elif run.rho_kind == "rho":  # rho_s
        rho_s = run.rho_n
        rho6T = get_rho6T(run.m_1 * MSUN, rho_s * MSUN / PC ** 3, run.gamma_s) / (
            1e16 * MSUN / PC ** 3
        )
    else:
        raise ValueError("invalid 'rho_kind' for run")

    _Phi_v, _ = load_run(run, "v")
    _Phi_d, f_d = load_run(run, "d")

    # Get shared frequency grid
    t_min = -f_d.x[0]
    print(f"Waveform starts {t_min / YR} yr before merger")
    f_start, f_end = f_d(-t_obs_lisa), min(_Phi_d.x[-1], _Phi_v.x[-1])
    f_grid = jnp.geomspace(f_start, f_end, 1000)

    # Align waveforms at same final frequency
    Phi_v = interp1d(f_grid, _Phi_v(f_grid) - _Phi_v(f_end))
    Phi_d = interp1d(f_grid, _Phi_d(f_grid) - _Phi_d(f_end))

    # Fit
    f_b, dPhi = fit_f_b(run.m_1, run.m_2, rho_s, run.gamma_s, f_grid, Phi_v, Phi_d)

    # Plot
    plt.figure(dpi=120)
    f_grid_full = jnp.geomspace(f_d.y[0], f_end, 1000)  # type: ignore
    dPhi_full = (
        _Phi_v(f_grid_full) - _Phi_v(f_end) - (_Phi_d(f_grid_full) - _Phi_d(f_end))
    )
    plt.loglog(f_grid_full, dPhi_full, linewidth=1, label="HF")
    plt.loglog(f_grid, dPhi(f_grid), "--", linewidth=1, label="Ana")
    plt.axvline(f_start, color="k")
    plt.axvline(f_b, color="r")
    plt.ylim(1e-5, 1e8)
    plt.xlabel(r"$f$ [Hz]")
    plt.ylabel(r"$\Phi_\mathrm{V} - \Phi_\mathrm{D}$ [rad]")
    plt.legend()
    # Formatting
    title = r"$m_1 = %.1f$, $m_2 = %.1f$, " % (run.m_1, run.m_2)
    title += r"$\rho_s = %g$, " % rho_s
    title += r"$\rho_6 = %.4f$, " % rho6T
    title += r"$\gamma_s = %.4f$," % run.gamma_s
    title += "\n"
    title += r"$f_b = %.3f$" % f_b
    plt.title(title)
    plt.tight_layout()

    if fig_path is not None:
        plt.savefig(fig_path)
        print(f"Saved figure to {fig_path}")

    plt.close()

    return RunFit(run.m_1, run.m_2, rho_s, rho6T, run.gamma_s, f_b, run.id_str)


def main():
    # Get all the runs
    runs = []
    for dn in os.listdir(run_dir):
        if "ri" not in dn:
            params = [float(p) for p in dn.split("_")[1::2]]
            if "_rho6_" in dn:
                rho_kind = "rho6"
            elif "_rho_" in dn:
                rho_kind = "rho"
            else:
                raise ValueError("invalid 'rho_kind' for run")

            runs.append(RunParams(*params, rho_kind, run_dir, dn))  # type: ignore

    # Fit all the runs
    all_fits = []
    for run in tqdm(runs):
        try:
            all_fits.append(fit_run(run, f"fit_plots/dPhi-{run.id_str}.png"))
            print("\n")
        except Exception as e:
            print(f"Run {run} failed:", e, "\n")
            all_fits.append(None)

    # Remove runs with break frequencies lower than the frequency at 5 years before
    # merger
    bad_ids = [  # <--- CHANGE
        "M1_26564.3_M2_1.2_rho6_1.0308_gamma_2.2996",
        "M1_49668.8_M2_4.0_rho6_4.2337_gamma_2.2513",
        "M1_81181.1_M2_6.6_rho6_0.0020_gamma_2.3666",
    ]

    results = defaultdict(lambda: [])

    for run, fit in zip(runs, all_fits):
        if "_rho1_" not in run.id_str and fit.id_str not in bad_ids:
            for k, v in fit._asdict().items():
                if k != "id_str":
                    results[k].append(v)

    results = dict(results)

    jnp.savez("fb_fits/fb_fits.npz", **results)
    print("Done!")


if __name__ == "__main__":
    main()
