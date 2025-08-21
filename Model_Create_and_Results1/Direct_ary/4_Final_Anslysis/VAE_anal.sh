#!/usr/bin/bash

#PBS -l select=1:ncpus=64:mpiprocs=1:ompthreads=64
#PBS -q workq
#PBS -j oe

source ~/.bashrc
cd /home/ianlee/opt_ian/Model_Create_and_Results1/Direct_ary/4_Final_Anslysis_Like_EDBO
conda activate leveler

echo "=========================================================="
echo "Starting on : $(date)"
echo "Running on node : $(hostname)"
echo "Current directory : $(pwd)"
echo "Current job ID : $PBS_JOBID"
echo "=========================================================="


python -u VAE_anal.py | tee /home/ianlee/opt_ian/Model_Create_and_Results1/Direct_ary/4_Final_Anslysis_Like_EDBO/log_VAE_anal.txt

conda deactivate

echo "=========================================================="
echo "Enging on : $(date)"
echo "=========================================================="