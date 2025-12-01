#!/bin/sh

srun  -A dp004 -p cosma8-shm -t 24:00:00 -N 1 --mem 800GB ./run_python.sh
#sbatch -A dp004 -n 1 -N 1 -p cosma8-shm --mem 800g -t 24:00:00 run_python.sh
