import torch
from torch.utils.data import Dataset, DataLoader
from fast_jtnn.mol_tree import MolTree
from fast_jtnn.jtnn_enc import JTNNEncoder
from fast_jtnn.mpn import MPN
from fast_jtnn.jtmpn import JTMPN
import pickle
import os, random
import numpy as np

from torch.utils.data.distributed import DistributedSampler

class PairTreeFolder(object):

    def __init__(self, data_folder, vocab, batch_size, num_workers=4, shuffle=True, y_assm=True, replicate=None):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.y_assm = y_assm
        self.shuffle = shuffle

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, 'rb') as f:
                data = pickle.load(f)

            if self.shuffle: 
                random.shuffle(data) #shuffle data before batch

            batches = [data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = PairTreeDataset(batches, self.vocab, self.y_assm)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x:x[0])

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader

class MolTreeFolder(object):

    def __init__(self, data_folder, vocab, batch_size, num_workers=4, shuffle=True, mult_gpus=False, assm=True, replicate=None):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        #print(self.data_files)
#JI
        random.shuffle(self.data_files)

        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm

        self.mult_gpus = mult_gpus

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, 'rb') as f:
                data = pickle.load(f)
            if self.shuffle: 
                random.shuffle(data) #shuffle data before batch

            batches = [data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)]
            
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = MolTreeDataset(batches, self.vocab, self.assm)
            
            if self.mult_gpus == True:
                #For DDP, sampler must be set to DistributedSampler to ensure that each rank gets a different mini batch of molecules
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x:x[0], sampler=DistributedSampler(dataset))
            else:
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x:x[0])
            
            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader

class PairTreeDataset(Dataset):

    def __init__(self, data, vocab, y_assm):
        self.data = data
        self.vocab = vocab
        self.y_assm = y_assm

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        batch0, batch1 = zip(*self.data[idx]) #batch0 is the tree part, 1 is the mpn part
        return tensorize(batch0, self.vocab, assm=False), tensorize(batch1, self.vocab, assm=self.y_assm)

#dataset for property co-train (use the traditional method)
class PropDataset(Dataset):

    def __init__(self, data, vocab, process_cond_file, prop_file, assm=True):
        self.prop_data = np.loadtxt(prop_file)
        self.process_cond_data = np.loadtxt(process_cond_file, delimiter=',')
        with open(data) as f:
            self.data = [line.strip("\r\n ").split()[0] for line in f] #處理SMILES檔案
        self.vocab = vocab
        self.assm = assm

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        smiles = self.data[idx]
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        mol_tree.assemble()         
        #need to store mol_tree to a whole batch to let tensorize reads it
        #mol_tree object is not iterable --> check original function for the encoder
        return mol_tree, self.process_cond_data[idx], self.prop_data[idx]

class MolTreeDataset(Dataset):

    def __init__(self, data, vocab, assm=True):
        self.data = data
        self.vocab = vocab
        self.assm = assm

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return tensorize(self.data[idx], self.vocab, assm=self.assm)

def tensorize(tree_batch, vocab, assm=True):
    set_batch_nodeID(tree_batch, vocab)
    smiles_batch = [tree.smiles for tree in tree_batch]
    jtenc_holder,mess_dict = JTNNEncoder.tensorize(tree_batch)
    jtenc_holder = jtenc_holder
    mpn_holder = MPN.tensorize(smiles_batch)

    if assm is False:
        return tree_batch, jtenc_holder, mpn_holder

    cands = []
    batch_idx = []
    for i,mol_tree in enumerate(tree_batch):
        for node in mol_tree.nodes: #卡在這裡
            #Leaf node's attachment is determined by neighboring node's attachment
            if node.is_leaf or len(node.cands) == 1: continue
            cands.extend( [(cand, mol_tree.nodes, node) for cand in node.cands] )
            batch_idx.extend([i] * len(node.cands))

    jtmpn_holder = JTMPN.tensorize(cands, mess_dict)
    batch_idx = torch.LongTensor(batch_idx)

    return tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder,batch_idx)

def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1
