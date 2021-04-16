#!/bin/sh
#$ -cwd
#$ -l f_node=2
#$ -l h_rt=00:10:00

. /etc/profile.d/modules.sh
module load cuda openmpi

mpiexec -n 56 -npernode 28 ./diffusion 100
