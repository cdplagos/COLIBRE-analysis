#!/bin/sh


snap=$1
z=$2
snip=$3
ivol=${SLURM_ARRAY_TASK_ID}

python sfr_maps_analysis_snipshot_broken_per_subv.py $snap $z $ivol $snip
