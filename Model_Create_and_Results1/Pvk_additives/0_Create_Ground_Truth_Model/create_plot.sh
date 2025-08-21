#!/usr/bin/bash
#PBS -l select=1:ncpus=28:ngpus=0
#PBS -l place=vscatter:shared

source ~/.bashrc
cd /home/ianlee/opt_ian/Model_Create_and_Results1/Pvk_additives/0_Create_Ground_Truth_Model
conda activate leveler2

echo "=========================================================="
echo "Starting on : $(date)"
echo "Running on node : $(hostname)"
echo "Current directory : $(pwd)"
echo "Current job ID : $PBS_JOBID"
echo "=========================================================="

python -u create_plot.py | tee /home/ianlee/opt_ian/Model_Create_and_Results1/Pvk_additives/0_Create_Ground_Truth_Model/log_create_plot.txt
conda deactivate

echo "=========================================================="
echo "Enging on : $(date)"
echo "=========================================================="