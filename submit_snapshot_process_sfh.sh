#!/bin/bash -l

# 36 snap_files = ['0127', '0119', '0114', '0102', '0092', '0076', '0064', '0056', '0048', '0040', '0032', '0026', '0018']
# 37 zstarget = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]


ztarget="0.0"
snap="0127"
volumes="`eval echo {0..639}`"
#volumes="`eval echo {140..141}`"
#volumes="208 211 214 217 222 228 254 262 285 286 287"

for ivol in $volumes; do
        sbatch -A dp004 -p cosma8 -t 02:00:00 -N 1 --mem 40GB run_python_sfh.sh $snap $ztarget $ivol $snip
done

