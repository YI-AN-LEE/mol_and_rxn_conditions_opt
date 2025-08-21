#!/usr/bin/bash
#PBS -l select=1:ncpus=8:ngpus=0
#PBS -q gpu
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

# python -u ABC_pvk.py --seed SEED --radius 2.5 --center_index CENTER_INDEX --freezed_position 2 \
#:vnode=g1-eno1[NODE_ID]
#seed=$(shuf -i 1-1073741824 -n 1)

python -u create_c0_df.py | tee /home/ianlee/opt_ian/Model_Create_and_Results1/Pvk_additives/0_Create_Ground_Truth_Model/log_create_c0.txt
conda deactivate

echo "=========================================================="
echo "Enging on : $(date)"
echo "=========================================================="