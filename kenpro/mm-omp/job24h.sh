#!/bin/sh
#$ -cwd
#$ -l q_core=1
#$ -l h_rt=24:00:00

export OMP_NUM_THREADS=4
./mm 1000 1000 1000
