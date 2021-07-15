#!/bin/bash
#SBATCH --job-name=bayes
#SBATCH --time=08:00:00
#SBATCH --tasks-per-node 16

NPROC=`nproc --all`
echo $NPROC

python run_ns.py --rho_6t 0.001 --gamma_s 2.25 --rho_6t_max 0.006 &
python run_ns.py --rho_6t 0.001 --gamma_s 2.30 --rho_6t_max 0.008 &
python run_ns.py --rho_6t 0.001 --gamma_s 2.40 --rho_6t_max 0.008 &
python run_ns.py --rho_6t 0.001 --gamma_s 2.50 --rho_6t_max 0.009 &

python run_ns.py --rho_6t 0.002 --gamma_s 2.25 --rho_6t_max 0.008 &
python run_ns.py --rho_6t 0.002 --gamma_s 2.30 --rho_6t_max 0.008 &
python run_ns.py --rho_6t 0.002 --gamma_s 2.40 --rho_6t_max 0.012 &
python run_ns.py --rho_6t 0.002 --gamma_s 2.50 --rho_6t_max 0.014 &

python run_ns.py --rho_6t 0.004 --gamma_s 2.25 --rho_6t_max 0.022 &
python run_ns.py --rho_6t 0.004 --gamma_s 2.30 --rho_6t_max 0.026 &
python run_ns.py --rho_6t 0.004 --gamma_s 2.40 --rho_6t_max 0.026 &
python run_ns.py --rho_6t 0.004 --gamma_s 2.50 --rho_6t_max 0.030 &

# python run_ns.py --rho_6t 0.01 --gamma_s 2.25 &
# python run_ns.py --rho_6t 0.01 --gamma_s 2.30 &
# python run_ns.py --rho_6t 0.01 --gamma_s 2.40 &
python run_ns.py --rho_6t 0.01 --gamma_s 2.50 --rho_6t_max 0.06 &

wait
