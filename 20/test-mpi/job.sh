#!/bin/sh
#$ -cwd
#$ -l q_core=1
#$ -l h_rt=0:10:00

. /etc/profile.d/modules.sh
module load cuda openmpi

mpiexec -n 2 ./test
