import os
import re
import datetime
from mordred import Calculator, descriptors
from rdkit import Chem
import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
import sys
import torch
sys.path.append('/home/ianlee/JTVAE/JTVAE/CPU-P3')
from fast_jtnn import *

vocab_path = '/home/ianlee/JTVAE/Ian_train/Vocabulary/smi_vocab-2.txt'
model_path = '/home/ianlee/JTVAE/Ian_train/Train/MODEL-TRAIN-3/model.epoch-39'

# Load vocabulary
vocab = [x.strip("\r\n ") for x in open(vocab_path)]
vocab = Vocab(vocab)

# Initial Step for VAE
vae_model = JTNNVAE(vocab, hidden_size=450, latent_size=32, depthT=3, depthG=20)
vae_model.load_state_dict(torch.load(model_path, map_location='cpu'))
vae_model.cpu()
vae_model.eval()

train_set = '/home/ianlee/JTVAE/Ian_train/Raw-Data/processed_smi.txt'
with open(train_set, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 去除每行的換行符
lines = [line.strip() for line in lines]
latent_Random = vae_model.encode_latent_mean(lines).detach().cpu().numpy()

#latent_Random = np.array(latent_Random)

for i in range(latent_Random.shape[1]):
    col = latent_ABC[:, i]
    print(f"Dim {i}: min={col.min():.2e}, max={col.max():.2e}, mean={col.mean():.2e}, std={col.std():.2e}")
