#!/bin/bash
#SBATCH --job-name=ns
#SBATCH --time=05:00:00
#SBATCH --tasks-per-node 2

NPROC=`nproc --all`
echo $NPROC

# SLURM script for performing nested sampling for astrophysical and PBH
# benchmark dark dresses.

# Astro
python run_ns.py --rho_6t 0.5448 --gamma_s 2.33333333 --rho_6t_max 1.2 --dm_chirp_abs 8e-3 &

# PBH
python run_ns.py --rho_6t 0.5345 --gamma_s 2.25 --rho_6t_max 1.2 &

wait
