import time
from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar

from pydd.analysis_np import calculate_SNR_cut, loglikelihood_cut
from pydd.binary_np import *
from pydd.noise_np import *

import cProfile
import pstats
from functools import wraps

Array = NDArray[np.float64]
ArrOrFloat = Union[float, Array]


def profile(
    output_file=None, sort_by="cumulative", lines_to_print=None, strip_dirs=False
):
    """A time profiler decorator.
    Inspired by and modified the profile decorator of Giampaolo Rodola:
    http://code.activestate.com/recipes/577817-profile-decorator/
    Args:
        output_file: str or None. Default is None
            Path of the output file. If only name of the file is given, it's
            saved in the current directory.
            If it's None, the name of the decorated function is used.
        sort_by: str or SortKey enum or tuple/list of str/SortKey enum
            Sorting criteria for the Stats object.
            For a list of valid string and SortKey refer to:
            https://docs.python.org/3/library/profile.html#pstats.Stats.sort_stats
        lines_to_print: int or None
            Number of lines to print. Default (None) is for all the lines.
            This is useful in reducing the size of the printout, especially
            that sorting by 'cumulative', the time consuming operations
            are printed toward the top of the file.
        strip_dirs: bool
            Whether to remove the leading path info from file names.
            This is also useful in reducing the size of the printout
    Returns:
        Profile of the decorated function
    """

    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _output_file = output_file or func.__name__ + ".prof"
            print(_output_file)
            pr = cProfile.Profile()
            pr.enable()
            retval = func(*args, **kwargs)
            pr.disable()
            pr.dump_stats(_output_file)

            with open(_output_file, "w") as f:
                ps = pstats.Stats(pr, stream=f)
                if strip_dirs:
                    ps.strip_dirs()
                if isinstance(sort_by, (tuple, list)):
                    ps.sort_stats(*sort_by)
                else:
                    ps.sort_stats(sort_by)
                ps.print_stats(lines_to_print)
            return retval

        return wrapper

    return inner


# Setup
S_n, F_RANGE_NOISE = S_n_aLIGO, f_range_aLIGO
FS = np.linspace(*F_RANGE_NOISE, 100_000)
M_1 = 1 * MSUN
M_2 = 1e-3 * MSUN
M_CHIRP = get_M_chirp(M_1, M_2)
TT_C = 0.0
F_C = get_f_isco(M_1)
PHI_C = 0.0
# _VB = VacuumBinary(M_CHIRP, PHI_C, TT_C, jnp.array(100e6 * PC), F_C)

# Get f_range
T_OBS = 1 * YR
# F_RANGE_V = get_f_range(_VB, T_OBS)
# # Get dL
# SNR_THRESH = 12.0
# _fn = jax.jit(
#     lambda dL: calculate_SNR_cut(
#         VacuumBinary(_VB.M_chirp, _VB.Phi_c, _VB.tT_c, dL, _VB.f_c),
#         F_RANGE_V,
#         FS,
#         S_n,
#     )
# )
# res = root_scalar(
#     lambda dL: (_fn(dL) - SNR_THRESH), bracket=(0.1e6 * PC, 100000e6 * PC)
# )
# assert res.converged
# DL = res.root

# Signal system
DL = 2.01861755959241e23
F_RANGE_V = (3.10436677, 4397.00982335)
VB = VacuumBinary(M_CHIRP, PHI_C, TT_C, DL, F_C)


def ll_cut_fn(vb_h, f_range_h):
    return loglikelihood_cut(vb_h, VB, f_range_h, F_RANGE_V, FS, S_n)


def sample_Mc_tTc() -> Array:
    return np.array(
        [
            M_CHIRP / MSUN * (1 + 6e-8 * (np.random.rand(1) - 0.5)),
            0.0004 * (np.random.rand(1) - 0.5),
        ]
    )


def unpack(x: Array) -> VacuumBinary:
    M_chirp_MSUN, tT_c = x
    M_chirp = M_chirp_MSUN * MSUN
    return VacuumBinary(M_chirp, VB.Phi_c, tT_c, VB.dL, VB.f_c)


def replace_tT_c(vb, tT_c) -> VacuumBinary:
    return VacuumBinary(vb.M_chirp, vb.Phi_c, tT_c, vb.dL, vb.f_c)


def ll_nomax_fn(x: Array) -> Array:
    """
    Log-likelihood without tT_c maximization.
    """
    vb_h = unpack(x)
    f_range_h = vb_h.get_f_range(T_OBS)
    return ll_cut_fn(vb_h, f_range_h)


print("Without tT_c maximization")
print("compiling and evaluating:", ll_nomax_fn(sample_Mc_tTc()))


# @profile()
def benchmark_nomax():
    n_loops = 10
    t_start = time.time()
    for _ in range(n_loops):
        ll_nomax_fn(sample_Mc_tTc())
    print((time.time() - t_start) / n_loops)


benchmark_nomax()


def sample_Mc() -> Array:
    return np.array([
        # M_CHIRP / MSUN * (1 + 6e-8 * (np.random.rand(1) - 0.5)),
        M_CHIRP / MSUN * (1 + 1.25e-7 * 2 * (np.random.rand() - 0.5)),
    ])


def ll_fn(x: Array) -> Array:
    """
    Log-likelihood with tT_c maximization.
    """
    vb = unpack(x)

    def _nll(tT_c):
        """
        Negative log-likelihood.
        """
        vb_h = replace_tT_c(vb, tT_c)
        f_range_h = vb_h.get_f_range(T_OBS)
        return -ll_cut_fn(vb_h, f_range_h)

    bracket = (-1e-3, 1e-3)
    res = minimize_scalar(_nll, bracket=bracket)
    assert res.success, f"tT_c maximization failed: {res}"
    return res.fun


print("With tT_c maximization")
print("compiling and evaluating:", ll_fn(sample_Mc_tTc()))

# @profile
def benchmark_max():
    n_loops = 10
    t_start = time.time()
    for _ in range(n_loops):
        ll_fn(sample_Mc_tTc())
    print((time.time() - t_start) / n_loops)


benchmark_max()
