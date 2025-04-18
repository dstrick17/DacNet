#!/bin/bash -l

# Set SCC project
#$ -P dl4ds


module load miniconda
conda activate dl4ds # activate your conda environment

python auc_chexnet.py ####change for whatever script you want to run

# To submit the job to SCC, run the following command
# qsub -pe omp 4 -P dl4ds -l gpus=1 -o output.txt -e error.txt run.sh -- Copy paste this command in your terminal!

# qstat -u dstrick  ---- to check status of job