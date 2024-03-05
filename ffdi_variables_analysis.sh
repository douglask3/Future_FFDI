#!/bin/bash -l
#SBATCH --qos=long # high
#SBATCH --mem=50000
#SBATCH --ntasks=10
#SBATCH --output=/scratch/hadhy/ffdi_fire_paper_%j_%N.out
#SBATCH --time=4320  #4320 #360

conda activate impacts_toolbox

python ffdi_variables_analysis.py
