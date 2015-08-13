#!/bin/sh

# Torque directives
#PBS -N theanoCode
#PBS -W group_list=yetistats
#PBS -l walltime=10:00:00,mem=15gb,other=gpu
#PBS -M ds3293@columbia.edu
#PBS -m abe
#PBS -V

#set output and error directories
#PBS -o localhost:/u/7/j/jsm2183/output
#PBS -e localhost:/u/7/j/jsm2183/error

module load anaconda/2.7.8
module load cuda/6.5

python Main.py > pyoutfile

#End of script
