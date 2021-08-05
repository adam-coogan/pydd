#!/bin/bash
#SBATCH --job-name=bf
#SBATCH --time=08:00:00
#SBATCH --tasks-per-node 16

# SLURM job script for performing dark dress and vaccum nested sampling runs
# and computing Bayes factors. Takes ~3 hr to run on the SURFsara Lisa cluster.

NPROC=`nproc --all`
echo $NPROC

# python run_ns.py --rho_6t 0.002 --gamma_s 2.25 --rho_6t_max 0.008 --calc-bf &
# python run_ns.py --rho_6t 0.002 --gamma_s 2.30 --rho_6t_max 0.008 --calc-bf &
# python run_ns.py --rho_6t 0.002 --gamma_s 2.40 --rho_6t_max 0.012 --calc-bf &
# python run_ns.py --rho_6t 0.002 --gamma_s 2.50 --rho_6t_max 0.014 --calc-bf &
#
# python run_ns.py --rho_6t 0.004 --gamma_s 2.25 --rho_6t_max 0.022 --calc-bf &
# python run_ns.py --rho_6t 0.004 --gamma_s 2.30 --rho_6t_max 0.026 --calc-bf &
# python run_ns.py --rho_6t 0.004 --gamma_s 2.40 --rho_6t_max 0.026 --calc-bf &
# python run_ns.py --rho_6t 0.004 --gamma_s 2.50 --rho_6t_max 0.030 --calc-bf &
#
# python run_ns.py --rho_6t 0.01 --gamma_s 2.25 --rho_6t_max 0.035 --calc-bf &
# python run_ns.py --rho_6t 0.01 --gamma_s 2.30 --rho_6t_max 0.04 --calc-bf &
# python run_ns.py --rho_6t 0.01 --gamma_s 2.40 --rho_6t_max 0.05 --calc-bf &
# python run_ns.py --rho_6t 0.01 --gamma_s 2.50 --rho_6t_max 0.06 --calc-bf &

python run_ns.py --rho_6t 0.003 --gamma_s 2.25 --rho_6t_max 0.035 --calc-bf &
python run_ns.py --rho_6t 0.003 --gamma_s 2.30 --rho_6t_max 0.04 --calc-bf &
python run_ns.py --rho_6t 0.003 --gamma_s 2.40 --rho_6t_max 0.05 --calc-bf &
python run_ns.py --rho_6t 0.003 --gamma_s 2.50 --rho_6t_max 0.06 --calc-bf &

python run_ns.py --rho_6t 0.001 --gamma_s 2.25 --rho_6t_max 0.035 --calc-bf &
python run_ns.py --rho_6t 0.001 --gamma_s 2.30 --rho_6t_max 0.04 --calc-bf &
python run_ns.py --rho_6t 0.001 --gamma_s 2.40 --rho_6t_max 0.05 --calc-bf &
python run_ns.py --rho_6t 0.001 --gamma_s 2.50 --rho_6t_max 0.06 --calc-bf &

python run_ns.py --rho_6t 0.0008 --gamma_s 2.25 --rho_6t_max 0.035 --calc-bf &
python run_ns.py --rho_6t 0.0008 --gamma_s 2.30 --rho_6t_max 0.04 --calc-bf &
python run_ns.py --rho_6t 0.0008 --gamma_s 2.40 --rho_6t_max 0.05 --calc-bf &
python run_ns.py --rho_6t 0.0008 --gamma_s 2.50 --rho_6t_max 0.06 --calc-bf &

wait
