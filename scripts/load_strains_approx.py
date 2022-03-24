import click
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq
import warnings
from math import pi

from pydd.binary import *

# SI units
G = 6.67408e-11  # m^3 s^-2 kg^-1
C = 299792458.0  # m/s
MSUN = 1.98855e30  # kg
PC = 3.08567758149137e16  # m
HOUR = 3600 #s
DAY = 24*HOUR #s
YR = 365.25 * DAY  # s

#Low frequency cut-off
f_low = 2.0

def load_strains(dd, m_1, m_2):

    #f_l, _ = get_f_range(dd, t_max)
    f_l, f_h = f_low, dd.f_c
    
    _fs = np.geomspace(f_l, f_h, 10000)
    _ts = t_to_c(_fs, dd)
    _ts *= -1  # time to merger

    #plt.figure()
    #plt.semilogy(_ts/YR, _fs)
    #plt.show()
    
    _omega_gws = 2 * pi * _fs
    omega_gw = interp1d(_ts, _omega_gws)
    _omega_orbs = 2 * pi * (_fs / 2)
    omega_orb = interp1d(_ts, _omega_orbs)
    _rs = (G * (m_1 + m_2) / (pi * _fs) ** 2) ** (1 / 3)
    r = interp1d(_ts, _rs)

    # Strain functions
    def h0(t):
        return 4 * G * m_2 * omega_orb(t) ** 2 * r(t) ** 2 / C ** 4

    def hp_t(t, d_l, iota, phi_c=0):
        return (
            1
            / d_l
            * h0(t)
            * (1 + np.cos(iota) ** 2)
            / 2
            * np.cos(omega_gw(t) * t + phi_c)
        )

    def hc_t(t, d_l, iota, phi_c=0):
        return 1 / d_l * h0(t) * np.cos(iota) * np.sin(omega_gw(t) * t + phi_c)

    return hp_t, hc_t
    
    

# @click.command()
# @click.option("--m_1", type=float, help="IMBH mass")
# @click.option("--m_2", type=float, help="BH mass")
# @click.option(
#     "--rho", type=float, help="initial density normalization rho_s [MSUN / PC**3]"
# )
# @click.option("--gamma", type=float, help="initial spike slope")
# @click.option("--t", default=t, help="time before merger [yr]")
# @click.option("--dt", default=dt, help="time step [s]")
# @click.option("--d_l", default=1e6, help="luminosity distance [PC]")
# @click.option("--iota", default=0.0, help="inclination angle [rad]")
# @click.option("--phi_c", default=0.0, help="phase at coalescence [rad]")
# @click.option("--run_dir", default="/Users/acoogan/Physics/dark_dress/finalRuns/full/")
def save_strain(m_1, m_2, d_l, iota, phi_c, dt, t_obs, t_end=0, system= "vacuum", rho = 0.0, gamma = 0.0, out_dir = ""):
    m_1 *= MSUN
    m_2 *= MSUN
    d_l *= PC

    #Checks:
    assert t_end <= 0, "t_end must be <= 0 (i.e. -time to merger)"
    assert t_obs > 0, "t_obs (duration of waveform) must be positive"
    
    Nstep = int(np.floor(t_obs/dt)+1)
    t0 = Nstep*dt
    ts = np.linspace(-t0 + t_end, t_end, Nstep)
    
    print("    > Loading strain interpolators...")
    if (system == "vacuum"):
        id_str = f"M1_{m_1 / MSUN:.4f}_M2_{m_2 / MSUN:.4f}_vacuum"
        dd = make_vacuum_binary(m_1, m_2)
        
    else:
        if (system == "PBH"):
            id_str = f"M1_{m_1 / MSUN:.4f}_M2_{m_2 / MSUN:.4f}_PBH"
            gamma = 9./4.
            rho = 1.396e13*(m_1/MSUN)**(3/4)
        else:
            id_str = f"M1_{m_1 / MSUN:.4f}_M2_{m_2 / MSUN:.4f}_rho6_{rho/1e16:.4f}_gamma_{gamma:.4f}"
            
        rho *= MSUN/PC**3
        dd = make_dynamic_dress(m_1, m_2, rho, gamma)
    
    hp_t, hc_t = load_strains(dd, m_1, m_2)


    print("    > Calculating plus polarisation...")
    hp_ts = hp_t(ts, d_l, iota, phi_c)
    
    print("    > Calculating cross polarisation...")
    hc_ts = hc_t(ts, d_l, iota, phi_c)
    
    fname_out = f"strain-{id_str}.txt.gz"
    print("    > Outputting strain file <" +  fname_out  + ">...")
    np.savetxt(
        os.path.join(out_dir, fname_out),
        np.stack((ts, hp_ts, hc_ts), axis=1),
        header="Columns: t [s], h_+, h_x")


def plot_strains(m_1, m_2, strain_dir = ""):
    m_1 *= MSUN
    m_2 *= MSUN

    id_str_vac = f"M1_{m_1 / MSUN:.4f}_M2_{m_2 / MSUN:.4f}_vacuum"        
    id_str = f"M1_{m_1 / MSUN:.4f}_M2_{m_2 / MSUN:.4f}_PBH"
    
    fname_vac = os.path.join(strain_dir, f"strain-{id_str_vac}.txt.gz")
    fname = os.path.join(strain_dir, f"strain-{id_str}.txt.gz")
    
    print("> Plotting strains...")
    try:
        ts_vac, hp_ts_vac, hc_ts_vac = np.loadtxt(fname_vac, unpack=True)
    except OSError:
        print(f"{fname_vac} not found")
        return
    try:
        ts, hp_ts, hc_ts = np.loadtxt(fname, unpack=True)
    except OSError:
        print(f"{fname} not found")
        return
    
    fig, axes = plt.subplots(figsize=(12, 4), nrows=1, ncols=3)
    
    for ax in axes:
        
        ax.plot(ts_vac, hp_ts_vac, label="Vacuum")
        ax.plot(ts, hp_ts, label="With DM")
        ax.set_xlabel(r"$t$ [s]")
        ax.set_ylabel(r"$h_+$")
        
        ax.legend(loc='best')
    
    #Zoom in on final part of the waveform:
    t_obs = ts[-1] - ts[0]
    frac = 0.00001 #Fraction of the full waveform to visualise at either end.
    
    axes[0].set_xlim(ts[0], ts[0] + t_obs*frac)
    axes[0].set_title("Zoomed (start)")
    
    axes[1].set_title("Full range")
    
    axes[2].set_xlim(ts[-1] - t_obs*frac, ts[-1])
    axes[2].set_title("Zoomed (end)")
    
    plt.tight_layout()
    
    plt.show()

    


if __name__ == "__main__":
    
    out_dir = "/Users/bradkav/Dropbox/Projects/PBH_EMRI_ET/data/waveforms/"
    
    #These values specify which 'trajectory' file to load:
    #rho     = 0.0014e16  # [MSun/pc^3]
    #gamma   = 9/4     # Slope of density profile
    
    
    #EDIT WAVEFORM PARAMETERS BELOW:
    #------------------------------------------------
    m_1     = 1.0     # [MSun]
    m_2     = 0.001   # [MSun]
    t_obs   = 1*DAY   # Duration of the waveform [seconds]
    t_end   = -0.5*YR   # End time of the waveform, measured with respect to the merger (t_end <= 0) [seconds]
    f_samp  = 20 # Sampling frequency [Hz]
    d_l     = 100   # Luminosity distance [pc]
    iota    = 0.0   # Inclination angle [rad]
    phi_c   = 0.0   # Phase at coalescence [rad]

    #------------------------------------------------

    dt      = 1/f_samp # Time between samples [seconds]
    
    print("> Calculating vacuum waveform...")
    save_strain(m_1 = m_1, 
                m_2 = m_2, 
                d_l = d_l, 
                iota = iota,  
                dt = dt,
                t_obs = t_obs,
                t_end = t_end,
                phi_c = phi_c,
                system = "vacuum",
                out_dir = out_dir)
                
    print("> Calculating dephased PBH waveform...")
    save_strain(m_1 = m_1, 
                m_2 = m_2, 
                d_l = d_l, 
                iota = iota,  
                phi_c = phi_c,
                dt = dt,
                t_obs = t_obs,
                t_end = t_end,
                system = "PBH", 
                out_dir = out_dir)
    #Set system = "PBH" and the code will choose the correct parameters for the DM spike
    
    plot_strains(m_1, m_2, strain_dir=out_dir)
