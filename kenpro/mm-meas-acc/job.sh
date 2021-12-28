#!/bin/sh
#$ -cwd
#$ -l q_node=1
#$ -l h_rt=00:10:00

echo --- compute 1000 1000 1000 ---
./mm 1000 1000 1000

echo --- compute 2000 2000 2000 ---
./mm 2000 2000 2000
