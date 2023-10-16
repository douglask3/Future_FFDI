#!/bin/bash -l
#SBATCH --qos=long
#SBATCH --mem=20000
#SBATCH --ntasks=2
#SBATCH --output=/scratch/hadhy/ffdi_fire_paper_%j_%N.out
#SBATCH --time=4320 #360

conda activate scitools

python ffdi_variables_analysis.py
