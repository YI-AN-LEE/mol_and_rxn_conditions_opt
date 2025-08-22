conda activate #your environment

python -u random_ary.py --seed SEED --max_iterations 1 --xi 0.01 \
 --ini_csv_path Model_Create_and_Results1/Direct_ary/3_Make_New_Data_Predictor/Random/cycle0.csv \
 --xgb_model_path Model_Create_and_Results1/Direct_ary/3_Make_New_Data_Predictor/Random/cycle0 \
 --vocab VAE_model/smi_vocab-2.txt \
 --latent_size 32 \
 --model VAE_model/model.epoch-39 | tee Scripts/Random_Seed.txt
conda deactivate
