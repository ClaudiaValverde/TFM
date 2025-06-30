#!/usr/bin/env python3

import os
import json
import time
import math
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from rdkit import Chem

from my_utils_v1_test import positional_encoding, get_tgt_mask,read_alphabet,LoadDatasetFeatsPocket_only, LoadDatasetFeatsProtein_only, LoadDatasetFeatsProtPocket

#softmax,
#datasample,
#hv_matrix2string,



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # file paths
    p.add_argument('--protein_embeds', type=str, required=True,
                   help='File or directory with protein embeddings (*.pt or directory of *.pt).')
    p.add_argument('--pocket_embeds', type=str, default=None,
                   help='File with pocket embeddings (optional).')
    p.add_argument('--split_csv', type=str, required=True,
                   help='CSV listing the (protein_id, pocket_id, ligand_id …) rows to generate on.')
    p.add_argument('--alphabet', type=str, default='unique_SMILES_alphabet.txt')
    p.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint (.pth).')
    p.add_argument('--out', type=str, default='generated_smiles.tsv', help='Output TSV file.')
    p.add_argument('--dataset', type=str, default='indomain', help='indomain or outdomain')
    p.add_argument('--weights', nargs=2, type=float, default=(1, 2))
    p.add_argument('--protein_id', type=str, default='protid', help='Specific protein index to generate SMILES for (optional).')
    p.add_argument('--protein_index', type=int, default=None,
               help='Index (after filtering) of the protein to generate SMILES for (optional).')

    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--nhead', type=int, default=16)
    p.add_argument('--n_dec_layers', type=int, default=4)
    p.add_argument('--operation', type=str, choices=['concat', 'weighted_add'], default='weighted_add')

    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--max_len', type=int, default=120)
    p.add_argument('--num_samples', type=int, default=1000, help='SMILES to sample per protein.')
    p.add_argument('--topk', type=int, default=50, help='Top‑k sampling.')

    # DDP configs:
    p.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
    p.add_argument('--ntasks', default=1, type=int, help='number of tasks')
    p.add_argument('--world_rank', default=-1, type=int, help='node rank for distributed training')
    # URL specifying how to initialize the process group.
    p.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
    # The NVIDIA Collective Communication Library (NCCL) implements multi-GPU and multi-node communication primitives optimized for NVIDIA GPUs and Networking
    p.add_argument('--dist_backend', default='nccl', type=str,  help='distributed backend')
    
    p.add_argument('--local_rank', type=int, default=-1)

    return p.parse_args()

class MyTransformer(nn.Module):

    def __init__(
        self,
        input_size_prot: int,  # embedding dim of protein features
        input_size_pocket: int,
        output_size: int,  # SMILES alphabet size
        d_model: int = 512,
        nhead: int = 8,
        num_dec_layers: int = 4,
        dropout: float = 0.1,
        operation: str = 'concat',  # 'concat' or 'weighted_add'
    ) -> None:
        super().__init__()

        self.operation = operation.lower()

        self.norm_memory = nn.LayerNorm(d_model)

        self.trg_projection = nn.Linear(output_size, d_model, bias=False)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_dec_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, output_size, bias=False)

    def forward(
        self,
        src_prot: torch.Tensor,  # [B, src_len, input_size_prot]
        src_pocket: torch.Tensor,  # [B, input_size_pocket]
        trg: torch.Tensor,  # [B, trg_len, output_size]
        P_trg: torch.Tensor,  # [B, trg_len, d_model]
        trg_mask: torch.Tensor,  #mask
        weights: tuple = (1, 2) # (alpha, beta) for weighted addition
    ) -> torch.Tensor:

        # Make sure pocket has same [B, 1, feat_dim] shape
        # Make sure protein has same [B, feat_dim] shape

        if src_prot is not None:
            if src_prot.dim() == 2:
                src_prot = src_prot.unsqueeze(1)  # [B, 1, feat_dim]

        if src_pocket is not None:
            if src_pocket.dim() == 2:
                src_pocket = src_pocket.unsqueeze(1)  # [B, 1, feat_dim]

        
        #if torch.isnan(src_prot).any() or torch.isnan(src_pocket).any():
        #    print("NaN in inputs!")

        
        if self.operation == 'concat':
            # concatenate along feature dimension
            memory = torch.cat([src_prot, src_pocket], dim=-1)  # ARA-->  [B, src_len, 2*d_model] # [B, 1, 2*d_model] | ABANS--> [B, src_len, input_size_prot + input_size_pocket] # [B, 1, 1280+4352]

        
        elif self.operation == 'weighted_add':
            alpha, beta = weights
            
            memory = alpha * src_prot + beta * src_pocket  # [B, src_len, d_model] # [B, 1, d_model]
            #print('mem size', memory.size())
        
        memory = self.norm_memory(memory)
        
        # Project target one‑hot i add positional encodings
        trg_emb = self.trg_projection(trg) + P_trg              # [B, trg_len, d_model]
        trg_emb = self.dropout(trg_emb)

        # Decoding
        out = self.transformer_decoder(trg_emb, memory, tgt_mask=trg_mask)
        logits = self.fc_out(out)      # [B, trg_len, output_size]
        if torch.isnan(logits).any():
            print("NaN in output!")
        return logits  # [B, trg_len, output_size] # Mirem si aixó té mes sentit


class ProtPocketDataset(Dataset):

    def __init__(self, prot_embeds, pocket_embeds, prot_ids):
        self.prot_embeds = prot_embeds
        self.pocket_embeds = pocket_embeds
        self.prot_ids = prot_ids

    def __len__(self):
        return len(self.prot_ids)

    def __getitem__(self, idx):
        return (
            self.prot_embeds[idx] if self.pocket_embeds is not None else torch.empty(0),
            self.pocket_embeds[idx] if self.pocket_embeds is not None else torch.empty(0),
            self.prot_ids[idx]
        )



def load_embeddings(path: str, keys) -> dict:
    """Return a dict {key: tensor}. Bulk load desde dicct"""
    p = Path(path)
    if p.is_dir():
        result = {}
        for k in tqdm(keys, desc='Loading embeddings'):
            f = p / f'{k}.pt'
            if not f.exists():
                raise FileNotFoundError(f'Missing embedding {f}')
            result[k] = torch.load(f)['representations'][33].float()
        return result
    else:
        obj = torch.load(p) if p.suffix == '.pt' else np.load(p, allow_pickle=True).item()
        return {k: torch.tensor(v).float() for k, v in obj.items()}


# -----------------------------------------------------------------------------
#  Generation helpers
# -----------------------------------------------------------------------------


def topk_sample(logits: torch.Tensor, k: int) -> int:
    """TopK sampling."""
    probs = torch.softmax(logits, dim=-1)
    topk_probs, topk_idxs = torch.topk(probs, k)
    idx = torch.multinomial(topk_probs, 1)
    return topk_idxs[idx].item()


def generate_smiles(model, prot, pocket, weights, alphabet, pos_enc, max_len=120, k=50, device='cuda'):
    """Funcio sampling desde prot-pock."""
    model.eval()
    char2num, num2char, onehot = alphabet
    BOS = 'è'  # tokens els que siguin
    EOS = '§'   # end of sentence token

    trg = onehot[char2num[BOS]].unsqueeze(0).to(device)  # [1,1,C] # l'alexis n'havia posat 1 d'extra... --> unsqueeze(0)
    generated = []

    for t in range(max_len):
        pos_trg = pos_enc[: trg.size(1), :].unsqueeze(0).to(device)
        mask = get_tgt_mask(trg.size(1)).to(device)
        with torch.no_grad():
            '''
            print('pos_trg', pos_trg.shape)
            print('mask', mask.shape)
            print('trg', trg.shape)
            print('pocket', pocket.shape)
            print('protein', prot.shape)
            '''
            logits = model(prot, pocket, trg, pos_trg, mask, weights)  # [1,L,C]
        next_idx = topk_sample(logits[0, -1], k)
        next_char = num2char[next_idx]
        if next_char == EOS:
            break
        generated.append(next_char)
        trg = torch.cat([trg, onehot[next_idx].unsqueeze(0).to(device)], dim=1) #.unsqueeze(0)
    smiles = ''.join(generated)
    smiles = smiles.replace('D', 'Cl').replace('E', 'Br')  # tret, malmenent, es queda
    return smiles


def main():
    args = parse_args()

    if args.local_rank == -1:
        args.local_rank = 0

    if int(os.environ.get('WORLD_SIZE', 1)) > 1 or args.world_size > 1:
        dist.init_process_group('nccl')
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f'cuda:{args.local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    N, char2num, num2char, onehot = read_alphabet(args.alphabet)
    #alphabet = (char2num, num2char, onehot.to(device))
    alphabet = (char2num, num2char, onehot)#.to(device)

    pos_enc = torch.from_numpy(positional_encoding(args.max_len + 1, args.d_model)).float()

    df = pd.read_csv(args.split_csv)

    if args.dataset == 'indomain': 
        df = df[df['split'] != 'test']
        df = df[df['Label'] != 0]

    if args.dataset == 'outdomain':
        df = df[df['split'] == 'test']
        df = df[df['Label'] != 0]
    
    else:
        df = df
    
    df.reset_index(drop=True, inplace=True)

    # choose specific protein to generate
    if args.protein_index is not None:
        if args.protein_index < 0 or args.protein_index >= len(df):
            raise IndexError(f"protein_index {args.protein_index} is out of range for filtered dataset (0 to {len(df)-1})")
        
        df = df.iloc[[args.protein_index]]  # Keep it as a DataFrame for downstream code
    
    protein_ids, pocket_ids = df["pdb_id"].str.lower().tolist(), df["pocket_key"].tolist()
    print('protein ids', protein_ids)
    print('pocket ids', pocket_ids)

    # comment it if I'm not reading from TensorDTI last layer!!!
    #protein_ids = (df["pdb_id"].str.lower() +'+'+ df["pocket_key"] +'+'+ df["ligand_id"]).tolist()
    #print('triplet_ID', protein_ids)
    
    weights = tuple(map(float, args.weights))
    if weights == (0.0, 0.0):
        raise ValueError("Both weights are zero, at least one must be non‑zero.")

    alpha, beta = weights
    print('operation, weights', args.operation, weights)
    if alpha == 0.0 :
        print('Pocket only dataset...')
        dataset = LoadDatasetFeatsPocket_only(
            pocket_ids = pocket_ids,
            x_file_name_pocket=args.pocket_embeds,
        )
    if beta == 0.0:
        print('Protein only dataset...')
        dataset = LoadDatasetFeatsProtein_only(
            protein_ids = protein_ids,
            x_file_name=args.protein_embeds,
        )
    if alpha != 0.0 and beta != 0.0:
        print('ProteinPocket dataset...')
        dataset = LoadDatasetFeatsProtPocket(
                protein_ids=protein_ids,
                pocket_ids=pocket_ids,
                x_file_name=args.protein_embeds,
                x_file_name_pocket=args.pocket_embeds,
            )

    input_size_prot = dataset.X.shape[-1] if alpha != 0.0 else None
    input_size_pocket = dataset.pockets.shape[-1] if beta != 0.0 else None
    
    if dist.is_initialized():
        model = DDP(model, device_ids=[args.local_rank])

    
    if args.world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=args.world_size, rank=args.world_rank, shuffle=True, drop_last=True
        )
    else:
        sampler = torch.utils.data.RandomSampler(dataset)
    

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    model = MyTransformer(
        input_size_prot=input_size_prot,
        input_size_pocket=input_size_pocket,
        output_size=N,
        d_model=args.d_model,
        nhead=args.nhead,
        num_dec_layers=args.n_dec_layers,
        operation=args.operation,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])

    
    if dist.is_initialized():
        model = DDP(model, device_ids=[args.local_rank])
    
    #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
   
    if dist.is_initialized() and dist.get_rank() == 0:
        print('Generant SMILES…')
    elif not dist.is_initialized():
        print('Generant SMILES…')

    results = []
    print('len loader', len(loader))
    for src, id_batch in loader:
        print('id_batch', id_batch)
        if alpha != 0 and beta != 0:
            prot_batch, pocket_batch = src
            prot_batch = prot_batch.to(device)
            pocket_batch = pocket_batch.to(device)
        
        if alpha == 0:
            pocket_batch = src
            pocket_batch = pocket_batch.to(device)
            prot_batch = torch.empty(1, 1, device=device)
        
        if beta == 0:
            prot_batch = src
            prot_batch = prot_batch.to(device)
            pocket_batch = torch.empty(1, 1, device=device)

        for i in range(prot_batch.size(0)):
            prot = prot_batch[i].unsqueeze(0)
            pocket = pocket_batch[i].unsqueeze(0)
            print('prot', prot.shape)
            print(f"Generating for protein {id_batch[i]}")
            for _ in range(args.num_samples):
                smiles = generate_smiles(model, prot, pocket, weights, alphabet, pos_enc, args.max_len, args.topk, device)
                mol = Chem.MolFromSmiles(smiles)
                valid = int(mol is not None)
                print(smiles if mol is not None else "")
                results.append((id_batch[i], smiles, valid))

    # gather results if distributed
    if dist.is_initialized():
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, results)
        if dist.get_rank() == 0:
            results = [r for sub in gathered for r in sub]
    # --------------------- write ---------------------
    if not dist.is_initialized() or dist.get_rank() == 0:
        if args.protein_index is not None:
            selected_pdb = df.iloc[0]['pdb_id']
            selected_pocket = pocket_ids[0]
            filename = f"{Path(args.out).stem}_{selected_pdb}.tsv"
            output_path = Path(filename)
            output_path.parent.mkdir(parents=True, exist_ok=True)  # Create subfolders if they don't exist

        with open(filename, 'w') as f:
            f.write('protein_id\tpocket_id\tvalid\tsmiles\n')
            for pid, smi, val in results:
                f.write(f'{pid}\t{selected_pocket}\t{val}\t{smi}\n')
        print(f'Wrote {filename} with {len(results)} rows.')


if __name__ == '__main__':
    main()