#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 2025

@author: claudia
"""

import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import random

from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

def get_valid_indices(protein_ids, pocket_ids, x_file_name, x_file_name_pocket, weights, y_file_name):
        """
        Returns a list of (protein_id, pocket_id, index_in_Y_all) tuples corresponding to valid entries.
        """
        Y_all = torch.from_numpy(np.load(y_file_name))

        alpha, beta = weights
        valid_indices = []

        prot_weights = {}
        pocket_weights = {}

        if alpha != 0:
            prot_weights = torch.load(x_file_name)
            prot_weights = {key.rstrip(): value for key, value in prot_weights.items()}

        if beta != 0:
            pocket_weights = torch.load(x_file_name_pocket)

        for i in range(len(protein_ids)):
            prot_id = protein_ids[i]
            pocket_id = pocket_ids[i]

            prot_embed = prot_weights.get(prot_id) if alpha != 0 else None
            pocket_embed = pocket_weights.get(pocket_id) if beta != 0 else None

            if (alpha != 0 and prot_embed is None) or (beta != 0 and pocket_embed is None):
                continue  # Skip invalid

            valid_indices.append((prot_id, pocket_id, i))  # i = index in Y_all

        return valid_indices


class LoadDatasetFeatsPocket_only(Dataset):
    def __init__(self, pocket_ids, x_file_name_pocket, y_file_name):
        self.pocket_ids = pocket_ids

        self.Y_all = torch.from_numpy(np.load(y_file_name))

        pocket_weights = torch.load(x_file_name_pocket)

        self.pocket_embeds = []
        self.Y = []

        for i in tqdm(range(len(pocket_ids))):
            pocket_id = pocket_ids[i]

            if pocket_id in pocket_weights:
                self.pocket_embeds.append(pocket_weights[pocket_id])
                self.Y.append(self.Y_all[i])
            
            else:
                missing = []
                if pocket_id not in pocket_weights:
                    missing.append(f"pocket: {pocket_id}")
                print(f"Skipping index {i} due to missing: {', '.join(missing)}")
           
        self.pockets = pad_sequence(self.pocket_embeds, batch_first=True, padding_value=0)
        self.Y = torch.stack(self.Y)

        self.number_of_examples = len(self.Y)
        print(f"Final dataset size: {self.number_of_examples}")

    def __getitem__(self, index):
        return self.pockets[index], self.Y[index]
    
    def __len__(self):
        return self.number_of_examples

class LoadDatasetFeatsProtein_only(Dataset):
    def __init__(self, protein_ids, x_file_name, y_file_name):
        self.protein_ids = protein_ids

        self.Y_all = torch.from_numpy(np.load(y_file_name))

        prot_weights = torch.load(x_file_name)
        
        self.prot_embeds = []
        self.Y = []

        for i in tqdm(range(len(protein_ids))):
            prot_id = protein_ids[i]

            if prot_id in prot_weights:
                self.prot_embeds.append(prot_weights[prot_id])
                self.Y.append(self.Y_all[i])
            
            else:
                missing = []
                if prot_id not in prot_weights:
                    missing.append(f"protein: {prot_id}")
                print(f"Skipping index {i} due to missing: {', '.join(missing)}")
        
        self.X = pad_sequence(self.prot_embeds, batch_first=True, padding_value=0)
        self.Y = torch.stack(self.Y)

        self.number_of_examples = len(self.Y)
        print(f"Final dataset size: {self.number_of_examples}")

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return self.number_of_examples

class LoadDatasetFeatsProtPocket(Dataset):
    def __init__(self, protein_ids, pocket_ids, x_file_name, x_file_name_pocket, y_file_name):
        self.protein_ids = protein_ids
        self.pocket_ids = pocket_ids

        self.Y_all = torch.from_numpy(np.load(y_file_name))

        prot_weights = torch.load(x_file_name)
        pocket_weights = torch.load(x_file_name_pocket)

        # Filter and align
        self.prot_embeds = []
        self.pocket_embeds= []
        self.Y = []

        for i in tqdm(range(len(protein_ids))):
            prot_id = protein_ids[i]
            pocket_id = pocket_ids[i]

            if prot_id in prot_weights and pocket_id in pocket_weights:
                self.prot_embeds.append(prot_weights[prot_id])
                self.pocket_embeds.append(pocket_weights[pocket_id])
                self.Y.append(self.Y_all[i])
            
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
        self.Y = torch.stack(self.Y)        

        self.number_of_examples = len(self.Y)
        print(f"Final dataset size: {self.number_of_examples}")

    def __getitem__(self, index):
        return (self.X[index], self.pockets[index]), self.Y[index]
       
    def __len__(self):
        return self.number_of_examples