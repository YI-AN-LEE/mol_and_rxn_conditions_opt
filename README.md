# Continous Optimization of Molecular Structures and Process Conditions via Latent Space Surrogates

<img width="6691" height="3584" alt="demo VAE +proc" src="https://github.com/user-attachments/assets/038c5d59-2ad1-4709-a7cf-9fad51f339a8" />

## Optimization

### Run
Run the python script named objective_algorithm, the objective names are shown as prefix as Direct-Arylation Yield: ary, Perovskite Crystal Size: pvk, for example the direct-arylation yield objective with ABC alogrithm is named ary_ABC.py

### Result Processing and Analysis
Open Model_Create_and_Results1

First preprocess the optimization output file (.o) from the preprocess notebook, then perform result analysis

### Create New Predictor
After each run we need to improve the current model from new data, run the create new predictor notebook

We trained 10 XGBoost models ensemble, the models could be store in the method directory (ABC, PSO, Random), cycle0 (batch 0) models and data are provided inside

## Environments

### Predictor and Decoder
Predictor and decoder functions are stored in Environments/objective_name/libs

### Optimization Algorithm
Algorithm script is in the Algortihm/ 

### Molecular VAE
VAE for optimizing (GPU-based) is in the fast-jtnn file, which is improved from the original python 2 version by CBIIT team (https://github.com/CBIIT/JTVAE), thanks for thier contribution!

To train the VAE please visit CBIIT github page which they offer head-to-tail tutorials. Here we offer the model trained for this work and the vocabulary set (the two files are essential for using VAE in this work) in the VAE_model file

To use the VAE in jupyter via CPU it is needed to use the VAE architecture in VAE_model/cpu/fast-jtnn
