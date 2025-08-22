conda activate #your environment
python -u ABC_ary.py --seed SEED --radius 0.25  \
  conda activate #your environment

python -u random_ary.py --seed SEED \
        --ini_csv_path Model_Create_and_Results1/Direct_ary/3_Make_New_Data_Predictor/Random/cycle0.csv \
        --xgb_model_path Model_Create_and_Results1/Direct_ary/3_Make_New_Data_Predictor/Random/cycle0 \
        --vocab VAE_model/smi_vocab-2.txt \
        --latent_size 32 \
        --model VAE_model/model.epoch-39 \
        --max_iterations 100 \
        --pop_size 10 \
        --latent_size 32 \
        --max_trials 50 | tee Scripts/ABC_Seed.txt
conda deactivate