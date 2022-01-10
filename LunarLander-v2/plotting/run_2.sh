#!/bin/bash
#SBATCH --account=rrg-lelis
#SBATCH --mem=4000M   
#SBATCH --cpus-per-task=16
#SBATCH --time=0:10:00

module load python/3.6
python 2_compare.py
