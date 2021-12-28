#!/bin/sh
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=00:10:00

export OMP_NUM_THREADS=28
./mm 1000 1000 1000
