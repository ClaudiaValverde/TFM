# https://github.com/comet-ml/comet-examples/blob/master/pytorch/comet-pytorch-ddp-cifar10.py
# https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904   
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
# https://discuss.pytorch.org/t/checkpointing-ddp-module-instead-of-ddp-itself/115714
# Prot2Drug: https://doi.org/10.5281/zenodo.14637195

import os
#import builtins
import time
import argparse
import torch
import pickle
import numpy as np 
import pandas as pd
import random
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
#from utils import load_data
import logging
from typing import Tuple, Optional

from utils_protpocket2smiles import LoadDatasetFeatsPocket_only, LoadDatasetFeatsProtein_only, LoadDatasetFeatsProtPocket, get_valid_indices
from my_utils_v1_f16 import positional_encoding, get_tgt_mask


def setup_ddp(args: argparse.Namespace):
  
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = int(os.environ.get('SLURM_JOB_NUM_NODES', 1)) * ngpus_per_node
    distributed = args.world_size > 1

    if distributed:
        if args.local_rank == -1:

            args.world_rank = int(os.environ.get('SLURM_PROCID', 0))
            args.local_rank = args.world_rank % ngpus_per_node
        else:
            args.world_rank = args.local_rank

        torch.cuda.set_device(args.local_rank)

        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.world_rank,
        )
        
        device = torch.device('cuda', args.local_rank)
    else:
        args.world_rank = 0
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    return device, args.world_size


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--p_csv', default='ChEMBL_example_ids_protein_ligand_training_set1.csv')
    p.add_argument('--p_file_name', default='protein_hv.npy')
    p.add_argument('--pocket_file_name', default='pocket_hv.npy')
    p.add_argument('--s_file_name', default='SMILES_hv.npy')
    p.add_argument('--a_file_name', default='unique_alphabet.txt')

    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--operation', choices=['concat', 'weighted_add'], default='concat')
    p.add_argument('--weights', nargs=2, type=float, default=(1, 2))
    p.add_argument('--save_ckpt', default='run')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--r_file_name', default='checkpoint.pth')

    # DDP configs:
    p.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
    p.add_argument('--ntasks', default=1, type=int, help='number of tasks')
    p.add_argument('--world_rank', default=-1, type=int, help='node rank for distributed training')
    # URL specifying how to initialize the process group.
    p.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
    # The NVIDIA Collective Communication Library (NCCL) implements multi-GPU and multi-node communication primitives optimized for NVIDIA GPUs and Networking
    p.add_argument('--dist_backend', default='nccl', type=str,  help='distributed backend')
    p.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training')

    args = p.parse_args()


    return args


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


def build_scheduler(optimiser: optim.Optimizer, warmup: int, total: int):
    def lr_lambda(epoch):
        if epoch < warmup:
            return float(epoch + 1) / max(1, warmup)
        prog = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1 + np.cos(np.pi * prog))
    return optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    P_trg_template: torch.Tensor,
    output_size: int,
    weights: Tuple[float, float],
):
    model.train()
    total_loss, n_batches = 0.0, 0

    for i, (src, trg_batch_full) in enumerate(loader):

        trg_batch_full = trg_batch_full.to(device, non_blocking=True).float()

        alpha, beta = weights
        
        if alpha == 0.0:
            src_batch, src_pocket = None, src.to(device, non_blocking=True).float()
            src_batch = torch.zeros_like(src_pocket).to(device, non_blocking=True).float()

        elif beta == 0.0:
            src_batch, src_pocket = src.to(device, non_blocking=True).float(), None
            src_pocket = torch.zeros_like(src_batch).to(device, non_blocking=True).float()

        else:
            src_batch, src_pocket = src 

            src_batch = src_batch.to(device, non_blocking=True).float()
            src_pocket = src_pocket.to(device, non_blocking=True).float()

        trg_in = trg_batch_full[:, :-1, :]  #tf input (no <eos>)
        trg_out_onehot = trg_batch_full[:, 1:, :]  # prediction targets

        B, L, _ = trg_in.shape

        P_trg = P_trg_template[: L, :].unsqueeze(0).repeat(B, 1, 1).to(device)
        trg_mask = get_tgt_mask(L).to(device)
        
        optimizer.zero_grad(set_to_none=True)
        logits = model(src_batch, src_pocket, trg_in, P_trg, trg_mask, weights)

        logits = logits.reshape(-1, output_size)
        tgt_indices = trg_out_onehot.argmax(dim=-1).reshape(-1)  # [B*L]

        loss = criterion(logits, tgt_indices)
        print('loss', loss)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


def run(args: argparse.Namespace) -> None:
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = int(os.environ.get("SLURM_JOB_NUM_NODES", 1)) * ngpus_per_node
    args.distributed = args.world_size > 1

    if args.distributed:
        if args.local_rank != -1:  
            args.world_rank = args.local_rank
        else:  
            args.world_rank = int(os.environ.get("SLURM_PROCID", 0))
            args.local_rank = args.world_rank % ngpus_per_node

        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.world_rank,
        )
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    else:
        print("Only DistributedDataParallel and DataParallel are supported. Sigle GPU on single node is not supported.")

    if args.world_rank == 0:
        os.makedirs(f"checkpoints/{args.save_ckpt}", exist_ok=True)

    ####
    # Loading Data
    ####

    csv_df = pd.read_csv(args.p_csv, sep=",")  # Prot2Drug with ; Plinder with ,
    print(csv_df)
    csv_df = csv_df[(csv_df['Label'] != 0) & (csv_df["split"] != "test")]
    protein_ids, pocket_ids = csv_df["pdb_id"].str.lower().tolist(), csv_df["pocket_key"].tolist()

    # comment it if I'm not reading from TensorDTI last layer!!!
    protein_ids = (csv_df["pdb_id"].str.lower() +'+'+ csv_df["pocket_key"] +'+'+ csv_df["ligand_id"]).tolist()

    weights = tuple(map(float, args.weights))
    if weights == (0.0, 0.0):
        raise ValueError("Both weights are zero, at least one must be non‑zero.")

    alpha, beta = weights
    print('operation, weights', args.operation, weights)
    if alpha == 0.0 :
        print('Pocket only dataset...')
        dataset = LoadDatasetFeatsPocket_only(
            pocket_ids = pocket_ids,
            x_file_name_pocket=args.pocket_file_name,
            y_file_name=args.s_file_name,
        )
    if beta == 0.0:
        print('Protein only dataset...')
        dataset = LoadDatasetFeatsProtein_only(
            protein_ids = protein_ids,
            x_file_name=args.p_file_name,
            y_file_name=args.s_file_name,
        )
    if alpha != 0.0 and beta != 0.0:
        print('ProteinPocket dataset...')
        dataset = LoadDatasetFeatsProtPocket(
                protein_ids=protein_ids,
                pocket_ids=pocket_ids,
                x_file_name=args.p_file_name,
                x_file_name_pocket=args.pocket_file_name,
                y_file_name=args.s_file_name,
            )

    input_size_prot = dataset.X.shape[-1] if alpha != 0.0 else None
    input_size_pocket = dataset.pockets.shape[-1] if beta != 0.0 else None
    output_size = dataset.Y.shape[-1]
    max_smiles_len = dataset.Y.shape[1]

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
        output_size=output_size,
        d_model=args.d_model,
        nhead=16,
        num_dec_layers=4,
        operation=args.operation,
    ).to(device)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    # Trainable parameters only
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    scheduler = build_scheduler(optimizer, warmup=5, total=args.epochs)
    criterion = nn.CrossEntropyLoss()

    P_trg_template = torch.from_numpy(positional_encoding(max_smiles_len, args.d_model)).float()

    print('-----Training...-----')
    start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        #sampler.set_epoch(epoch)
        if args.world_size > 1 and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)  # type: ignore

        epoch_loss = train_one_epoch(
            model,
            loader,
            criterion,
            optimizer,
            device,
            P_trg_template,
            output_size,
            weights
        )
        scheduler.step()

        if args.world_rank == 0:
            print(f"Epoch {epoch:03d} | loss {epoch_loss:.4f}")
            #writer.add_scalar("train/loss", epoch_loss, epoch)

            ckpt = {
                "epoch": epoch,
                "model_state": model.module.state_dict(),
                "optim_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }
            ckpt_path = f"checkpoints/{args.save_ckpt}/checkpoint_ep{epoch:03d}_loss{epoch_loss}.pth"
            torch.save(ckpt, ckpt_path)
            print(f"Saved {ckpt_path}")
        '''
        if args.world_size > 1:
            dist.barrier(device_ids=[torch.cuda.current_device()])
        '''


if __name__ == "__main__":
    #main()
    args = parse_args()
    run(args)
