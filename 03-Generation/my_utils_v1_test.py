#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:05:51 2024

@author: ancona
"""

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import random
from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence


def positional_encoding(max_string_lenght, input_size):
    """Help of positional_encoding."""
    P = np.zeros((max_string_lenght, input_size))
    for i in range(max_string_lenght):
        for j in range(0, input_size, 2): # j is even
            # print(j)
            P[i,j] = np.sin(i / np.power(10000, (2*j)/np.float32(input_size)))
        for j in range(1, input_size, 2): # j is odd
            # print(j)
            P[i,j] = np.cos(i / np.power(10000, (2*j)/np.float32(input_size)))
            
    return P
        
def get_tgt_mask(size) -> torch.tensor:
    # Generates a squeare matrix where the each row allows one word more to be seen
    mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
    mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
    
    # EX for size=5:
    # [[0., -inf, -inf, -inf, -inf],
    #  [0.,   0., -inf, -inf, -inf],
    #  [0.,   0.,   0., -inf, -inf],
    #  [0.,   0.,   0.,   0., -inf],
    #  [0.,   0.,   0.,   0.,   0.]]
    
    return mask

def softmax(x):
    """ Help of softmax. """
    return np.exp(x)/np.sum(np.exp(x))

def datasample(alphabet, yd):
    """ Help of datasample. """
    
    aaa, output_size = yd.shape
    Fy = np.zeros(output_size)
    Fy[0] = yd[0,0]
    for i in range(1,output_size):
        Fy[i] = np.sum(yd[0,0:i+1]) # comulative probability array, where Fy[i] is the probability of picking any character up to index i

    y = random.uniform(0,1) # picks a random probability value
    x = 0
    while Fy[x] < y: x+= 1 # find the first charachter exceeding y
    
    # return the corresponding character based on x
    if x in alphabet:
        out_char = alphabet[x]
    else:
        out_char = '°'
    
    return out_char # alphabet[x]

def hv_matrix2string(hv_matrix, alphabet):
    
    SMILES_out = ''
    smiles_lenght, output_size = hv_matrix.shape
    y = np.zeros((1,output_size))
    # out_tmp = out.detach().numpy()
    #trg_tmp = trg.detach().numpy()
    for k in range(smiles_lenght):
        y[0,:] = hv_matrix[k,:]
        # new_char_out = datasample(alphabet, softmax(y))
        # y1 = softmax(y)
        if np.argmax(y[0]) in alphabet:
            new_char_out = alphabet[np.argmax(y[0])]
        else:
            new_char_out = '°'
        SMILES_out += new_char_out

    return SMILES_out

def read_alphabet(file_name):
    
    SMILES_char2number = {} # character:number pair; this is a dictionary
    number2SMILES_char = {}
    number2hotvec = {} # number:one-hot vector pair; this is a dictionary

    f = open(file_name,"r",encoding='utf-8') # open file for reading, "w"

    SMILES_alphabet_size = int(f.readline()) # read from file
    SMILES_alphabet_symbol = f.readline()
    print("\n" in SMILES_alphabet_symbol)
    SMILES_alphabet_symbol = SMILES_alphabet_symbol[0:SMILES_alphabet_size]
    print("\n" in SMILES_alphabet_symbol)
    for i in range(SMILES_alphabet_size):
        item_tmp = f.readline().split()
        SMILES_char2number[item_tmp[0]] = int(item_tmp[1])
        number2SMILES_char[int(item_tmp[1])] = item_tmp[0]

    N = int(f.readline()) # N is the size of the alphabet !!!
    for h in range(N):
        k = int(f.readline())
        hv = f.readline().split()
        tens_hv = torch.zeros(1, N)
        for i,j in enumerate(hv): tens_hv[0,i] = int(j)
        number2hotvec[k] = tens_hv

    f.close() # close file
    
    return N, SMILES_char2number, number2SMILES_char, number2hotvec

def old_datasample(alphabet, yd, output_size):
    """ Help of datasample. """
    
    Fy = np.zeros(output_size)
    Fy[0] = yd[0,0]
    for i in range(1,output_size):
        Fy[i] = np.sum(yd[0,0:i+1])

    y = random.uniform(0,1)
    x = 0
    while Fy[x] < y: x+= 1
    
    return alphabet[x]


class LoadDatasetFeatsPocket_only(Dataset):
    def __init__(self, pocket_ids, x_file_name_pocket):
        self.pocket_ids = pocket_ids
        pocket_weights = torch.load(x_file_name_pocket)

        self.pocket_embeds = []

        for i in tqdm(range(len(pocket_ids))):
            pocket_id = pocket_ids[i]

            if pocket_id in pocket_weights:
                self.pocket_embeds.append(pocket_weights[pocket_id])
            
            else:
                missing = []
                if pocket_id not in pocket_weights:
                    missing.append(f"pocket: {pocket_id}")
                print(f"Skipping index {i} due to missing: {', '.join(missing)}")
           
        self.pockets = pad_sequence(self.pocket_embeds, batch_first=True, padding_value=0)

        self.number_of_examples = len(self.pockets)
        print(f"Final dataset size: {self.number_of_examples}")

    def __getitem__(self, index):
        return self.pockets[index], self.pocket_ids[index]
    
    def __len__(self):
        return self.number_of_examples

class LoadDatasetFeatsProtein_only(Dataset):
    def __init__(self, protein_ids, x_file_name):
        self.protein_ids = protein_ids

        prot_weights = torch.load(x_file_name)
        
        self.prot_embeds = []

        for i in tqdm(range(len(protein_ids))):
            prot_id = protein_ids[i]

            if prot_id in prot_weights:
                self.prot_embeds.append(prot_weights[prot_id])
            
            else:
                missing = []
                if prot_id not in prot_weights:
                    missing.append(f"protein: {prot_id}")
                print(f"Skipping index {i} due to missing: {', '.join(missing)}")
        
        self.X = pad_sequence(self.prot_embeds, batch_first=True, padding_value=0)

        self.number_of_examples = len(self.X)
        print(f"Final dataset size: {self.number_of_examples}")

    def __getitem__(self, index):
        return self.X[index], self.protein_ids[index]
    
    def __len__(self):
        return self.number_of_examples

class LoadDatasetFeatsProtPocket(Dataset):
    def __init__(self, protein_ids, pocket_ids, x_file_name, x_file_name_pocket):
        self.protein_ids = protein_ids
        self.pocket_ids = pocket_ids

        prot_weights = torch.load(x_file_name)
        pocket_weights = torch.load(x_file_name_pocket)

        # Filter and align
        self.prot_embeds = []
        self.pocket_embeds= []

        for i in tqdm(range(len(protein_ids))):
            prot_id = protein_ids[i]
            pocket_id = pocket_ids[i]

            if prot_id in prot_weights and pocket_id in pocket_weights:
                self.prot_embeds.append(prot_weights[prot_id])
                self.pocket_embeds.append(pocket_weights[pocket_id])
            
            else:
                missing = []
                if prot_id not in prot_weights:
                    missing.append(f"protein: {prot_id}")
                if pocket_id not in pocket_weights:
                    missing.append(f"pocket: {pocket_id}")
                print(f"Skipping index {i} due to missing: {', '.join(missing)}")
        
        # Pad sequences (if variable length); else use torch.stack
        self.X = pad_sequence(self.prot_embeds, batch_first=True, padding_value=0)
        self.pockets = pad_sequence(self.pocket_embeds, batch_first=True, padding_value=0)

        self.number_of_examples = len(self.X)
        print(f"Final dataset size: {self.number_of_examples}")

    def __getitem__(self, index):
        return (self.X[index], self.pockets[index]), self.protein_ids[index]
       
    def __len__(self):
        return self.number_of_examples
