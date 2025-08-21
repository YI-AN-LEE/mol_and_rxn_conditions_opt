#!/usr/bin/bash
#PBS -l select=1:ncpus=32
#PBS -q workq

source ~/.bashrc
cd /home/ianlee/opt_ian/Model_Create_and_Results1/Direct_ary/0_Create_Ground_Truth_Model
conda activate leveler2

echo "=========================================================="
echo "Starting on : $(date)"
echo "Running on node : $(hostname)"
echo "Current directory : $(pwd)"
echo "Current job ID : $PBS_JOBID"
echo "=========================================================="

python -u regression_model_analysis.py --seed SEED --drop_rule 1 --model_type rfr --feature mordred | tee /home/ianlee/opt_ian/Model_Create_and_Results1/Direct_ary/0_Create_Ground_Truth_Model/Regression_Model_Analysis/log_aryl_SEED.txt

echo "=========================================================="
echo "Enging on : $(date)"
echo "=========================================================="