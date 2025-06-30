#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:32:42 2024

@author: ancona
"""

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import json


# True: all the proteins and SMILES have the same lenght.
maximize_num_padding_char = True

max_protein_lenght_allowed = 2002 # max number of character in protein sequence

# g = open('synthetic-examples.txt', 'r') # I read the examples dataset;
# g = open('examples-train-ds1-new.txt', 'r') # I read the examples dataset;
# g = open('ts100ns_final.csv', 'r') # I read the examples dataset;
g = open('plinder_pairs_without_pocket_smiles.csv', 'r') # I read the examples dataset;
g = open('cleaned_plinder_pairs.csv', 'r') # I read the examples dataset;
g = open('plinder_pairs_without_pocket_smiles_train.csv', 'r')
g = open('/gpfs/projects/nost02/Prot2Drug/plinder/pocket/plinder_pocket_data_only1_ligands.csv', 'r')

# g = open('example_dist_rep_protein_le500char_ge1000ligands.csv', 'r') # I read the examples dataset;
# g = open('unique-examples-train-new.txt', 'r') # I read the examples dataset;
# protein_string, SMILES_string = g.readlines()

#example = g.readlines() # normal
example = g.readlines()[1:]# to skip the first line (HEADER)

g.close()

dist_rep_string = []

SMILES_string = []
SMILES_lenght = []
SMILES_string_final = []

for i in range(len(example)):
    #pntr = example[i].find(",") # pntr points to the comma, # ;
    #dist_rep_string.append(example[i][0:pntr]) # I extracted the protein from the example
    #str_tmp = example[i][pntr+1:]
    # l'he canviat jo
    str_tmp = example[i].split(',')[-1] # last column is the one that has the correct format smiles (clàudia)
    #print(str_tmp)
    if '\n' in str_tmp:
        SMILES_string.append(str_tmp[:-1])
    else:
        SMILES_string.append(str_tmp) # I extracted the SMILES
    
    #print(SMILES_string[i])
    str1 = SMILES_string[i].replace("Cl", "D")
    str2 = str1.replace("Br", "E")
#    str3 = "$" + str2 + "~" # OLD VERSION
    str3 = "è" + str2 + "§"
    SMILES_string_final.append(str3)
    SMILES_lenght.append(len(SMILES_string_final[i]))
    
min_SMILES_lenght = min(SMILES_lenght)
max_SMILES_lenght = max(SMILES_lenght)
number_of_SMILES = len(SMILES_string_final)
print(f'Number of SMILES = {number_of_SMILES} lenght = [{min_SMILES_lenght}, {max_SMILES_lenght}]')
counts, bins = np.histogram(SMILES_lenght)
plt.stairs(counts, bins)
plt.savefig('num_smiles_plinder_cleaned_train.png')
plt.show()

# Work on SMILES ********************************************************************

# Padding the SMILES
for i in range(number_of_SMILES):
    while len(SMILES_string_final[i]) < max_SMILES_lenght :
        SMILES_string_final[i] += "£"


# Read the alphabets
SMILES_char2number = {} # character:number pair; this is a dictionary
number2SMILES_char = {}
number2hotvec = {} # number:one-hot vector pair; this is a dictionary

f = open("/gpfs/projects/nost02/Prot2Drug/2025-01-13-Prot2Drug-Journal-of-Chemical-Information-and-Modeling-data-and-code/04-Training/dataset/unique_SMILES_alphabet_plinder.txt","r") # open file for reading, "w"

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

print(SMILES_char2number)
#print(number2hotvec)
# Represent SMILES with one-hot vectors.
SMILES_hv = np.zeros((number_of_SMILES, max_SMILES_lenght, N), dtype='float16')
for i in range(number_of_SMILES):
    for j,c in enumerate(SMILES_string_final[i]):
        #print(c)
        #print(SMILES_string_final[i])
        SMILES_hv[i,j,:] = number2hotvec[SMILES_char2number[c]]

print(len(SMILES_hv))
# np.save('SMILES_hv', SMILES_hv[0:10000, :, :])
np.save('SMILES_plinder_pocket_cleaned', SMILES_hv)

'''
# save the distributed representation of the proteins.
NO NECESSARI PER MI
number_of_example = len(dist_rep_string)
number_of_components = len(dist_rep_string[0].split(','))
protein_dist_rep = np.zeros((number_of_example, number_of_components), dtype='float32')
for i in range(number_of_example):
    string_of_numbers_list = dist_rep_string[i].split(',')
    array_of_float_number = [float(number) for number in string_of_numbers_list]
    protein_dist_rep[i,:] = array_of_float_number

np.save('ChEMBL_example_dr_protein', protein_dist_rep)
'''
