#!/bin/sh
#$ -cwd
#$ -l q_core=2
#$ -l h_rt=0:10:00

. /etc/profile.d/modules.sh
module load cuda openmpi

mpiexec -n 8 -npernode 4 ./hello
