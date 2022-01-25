#!/bin/sh

# sbatch -C gpu -A s931 --partition long -t 10080 < q.sh
# squeue -u `whoami`
module load daint-gpu
module load cray-python
d='/scratch/snx3000/eceva/FCD/FCD_Controls/FCD_Control_FCX_1.3_NeuN-Cy3 (after reclearing)'

OMP_NUM_THREADS=1 srun python3 -u poc/viewer/b.py "$d"/*.raw
