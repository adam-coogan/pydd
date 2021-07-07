#!/bin/bash
#SBATCH --job-name=ns
#SBATCH --time=08:00:00
#SBATCH --tasks-per-node 2

NPROC=`nproc --all`
echo $NPROC

# Astro
python run_ns.py --rho_6t 0.5448 --gamma_s 2.33333333 --rho_6t_max 1.2 &

# PBH
python run_ns.py --rho_6t 0.5345 --gamma_s 2.25 --rho_6t_max 1.2 &

wait
