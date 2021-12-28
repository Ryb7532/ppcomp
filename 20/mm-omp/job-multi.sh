#!/bin/sh
#$ -cwd
#$ -l q_node=1
#$ -l h_rt=00:10:00

export OMP_NUM_THREADS=1
echo Number of threads is $OMP_NUM_THREADS !
./mm 1000 1000 1000
export OMP_NUM_THREADS=2
echo Number of threads is $OMP_NUM_THREADS !
./mm 1000 1000 1000
export OMP_NUM_THREADS=4
echo Number of threads is $OMP_NUM_THREADS !
./mm 1000 1000 1000
export OMP_NUM_THREADS=7
echo Number of threads is $OMP_NUM_THREADS !
./mm 1000 1000 1000
