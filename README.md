# Continous Optimization of Molecular Structures and Process Conditions via Latent Space Surrogates

## Optimization

### run
Run the python script named objective_algorithm, the objective names are shown as prefix as Direct-Arylation Yield: ary, Perovskite Crystal Size: pvk, for example the direct-arylation yield objective with ABC alogrithm is named ary_ABC.py

### Result Processing and Analysis
Open Model_Create_and_Results1

First preprocess the optimization output file (.o) from the preprocess notebook, then perform result analysis

### Create New Predictor
After each run we need to improve the current model from new data, run the create new predictor notebook

## Environments

### Predictor and Decoder
Predictor and decoder functions are stored in Environments/objective_name/libs

### Optimization Algorithm
Algorithm script is in the Algortihm/ 

### Molecular VAE
VAE structure is in the fast-jtnn file, which is improved from the original python 2 version by CBIIT team (https://github.com/CBIIT/JTVAE), thanks for thier contribution!
To train the VAE please visit CBIIT github page which they offer head-to-tail tutorials. Here we offer the model trained for this work and the vocabulary set (the two files are essential for using VAE in this work) in the VAE_model file
*important* in the file the fast-jtnn is designed for GPU, if run on CPU all .to('cuda') lines must change to .to('cpu'), note that jupyter files in analysis runs VAE in cpu, when running the code it is recommend to copy a CPU version of fast-jtnn to your local directory and run the CPU file with it
