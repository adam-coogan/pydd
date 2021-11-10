from math import log10
import os
import pickle

import dynesty
from dynesty import plotting as dyplot
import matplotlib.pyplot as plt
import numpy as np

from pydd.binary import MSUN, get_M_chirp
from utils import (
    GAMMA_S_ASTRO,
    GAMMA_S_PBH,
    M_1_BM,
    M_2_BM,
    RHO_6_ASTRO,
    RHO_6_PBH,
    rho_6_to_rho6T,
)




def get_base_path(rho_6, gamma_s):
    return f"rho_6T={rho_6_to_rho6T(rho_6):g}_gamma_s={gamma_s:g}"


def get_bounds(x, P, threshold = 0.68):
    p_tot = -1e30
    P_cut = np.max(P)
    while p_tot < threshold:
        P_cut -= 1e-2
        mask = P > P_cut
        p_tot = np.trapz(P[mask], x[mask])
    print(P_cut)
    return np.min(x[mask]), np.max(x[mask])

def plot_compare():
    base_path = get_base_path(RHO_6_ASTRO, GAMMA_S_ASTRO)
    with open(os.path.join("ns", f"{base_path}.pkl"), "rb") as infile:
        results_astro = pickle.load(infile)

    base_path = get_base_path(RHO_6_PBH, GAMMA_S_PBH)
    with open(os.path.join("ns", f"{base_path}.pkl"), "rb") as infile:
        results_pbh = pickle.load(infile)

    samps_astro = results_astro.samples
    lw_astro = results_astro.logwt
    
    samps_pbh = results_pbh.samples
    lw_pbh = results_pbh.logwt
    
    Nbins = 50
    
    plt.figure(figsize=(6,5))
    
    P_astro, bins_astro, _ = plt.hist(samps_astro[:,0], weights=np.exp(lw_astro), density=True, bins=Nbins, histtype='step',color='C0', linewidth=2.0)
    P_pbh, bins_pbh, _ = plt.hist(samps_pbh[:,0], weights=np.exp(lw_pbh), density=True, bins=Nbins, histtype='step', color='C1', linewidth=2.0)
    
    x_astro = bins_astro[:-1] + np.diff(bins_astro)/2.0
    x_pbh = bins_pbh[:-1] + np.diff(bins_pbh)/2.0

    min_astro, max_astro = get_bounds(x_astro, P_astro)
    print(min_astro, max_astro)
    
    min_pbh, max_pbh = get_bounds(x_pbh, P_pbh)
    print(min_pbh, max_pbh)
    
    plt.axvline(7.0/3.0, linestyle='--', color='C0')
    plt.axvline(9.0/4.0, linestyle='--', color='C1')
    
    yvals = np.linspace(0, 30)
    plt.fill_betweenx(yvals, 2.20, 2.25, color='grey', zorder=0)
    plt.fill_betweenx(yvals, 2.50, 2.55, color='grey', zorder=0)
    
    plt.scatter([7 / 3], [1], marker="*", color="black", s=200, zorder=10)
    plt.scatter([9 / 4], [1], marker=".", color="black", s=250, zorder=10)
    
    plt.annotate("",xy=(7/3, 1), xytext=(min_astro, 1), arrowprops=dict(arrowstyle='<-', shrinkA=0, shrinkB=0, color='C0'))
    plt.annotate("",xy=(7/3, 1), xytext=(max_astro, 1), arrowprops=dict(arrowstyle='<-', shrinkA=0, shrinkB=0, color='C0'))
    
    plt.annotate("",xy=(9/4, 1), xytext=(min_pbh, 1), arrowprops=dict(arrowstyle='<-', shrinkA=0, shrinkB=0, color='C1'))
    plt.annotate("",xy=(9/4, 1), xytext=(max_pbh, 1), arrowprops=dict(arrowstyle='<-', shrinkA=0, shrinkB=0, color='C1'))
    
    
    #plt.arrow(9/4, 1, -(9/4 - min_pbh), 0, color='black')
    #plt.arrow(9/4, 1, max_pbh - 9/4, 0, color='black')
    
    plt.xlim(2.20, 2.55)
    plt.ylim(0, np.max(yvals))
    
    plt.xticks(np.arange(2.20, 2.60, 0.05))
    
    plt.xlabel(r'$\gamma_\mathrm{sp}$')
    plt.ylabel(r'Posterior $P(\gamma_\mathrm{sp})$')
    
    plt.savefig("figures/P_gamma_sp.pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_compare()
