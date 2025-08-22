conda activate #your environment

python -u PSO_ary.py --seed SEED --radius 0.25 \
    -ini_csv_path Model_Create_and_Results1/Direct_ary/3_Make_New_Data_Predictor/PSO/cycle0.csv \
    --xgb_model_path Model_Create_and_Results1/Direct_ary/3_Make_New_Data_Predictor/PSO/cycle0 \
    --vocab VAE_model/smi_vocab-2.txt \
    --latent_size 32 \
    --model VAE_model/model.epoch-39 \
 --generation 1 \
 --pso_epoch 100 \
 --pop_size 10 | tee /home/ianlee/opt_ian/Environments/Direct_Arylation/Jobs/PSO/log_aryl_SEED.txt
 
conda deactivate
