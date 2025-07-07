import os
import faiss
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# === Convert SMILES to Morgan Fingerprints (Batch Processing) ===
def smiles_to_fp(smiles_list):
    fps_np = []
    fps_rdkit = []
    valid_idx = []
    valid_smiles = []

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            arr = np.zeros((2048,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps_np.append(arr)
            fps_rdkit.append(fp)
            valid_idx.append(i)
            valid_smiles.append(smiles)

    return np.array(fps_np, dtype=np.uint8), fps_rdkit, valid_idx, valid_smiles

def compute_tanimoto_all_vs_all(source_name, target_name, fingerprints, smiles_dict):
    print(f"\n🔍 Computing All-vs-All Tanimoto Similarity ({source_name} vs {target_name})...")

    source_rdkit = fingerprints[source_name]["rdkit"]
    target_rdkit = fingerprints[target_name]["rdkit"]

    if len(source_rdkit) == 0 or len(target_rdkit) == 0:
        print(f"⚠️ Skipping {source_name} vs {target_name} (No valid fingerprints)")
        return None

    tanimoto_scores = []
    matching_pairs = []

    same_split = source_name == target_name

    for i, fp1 in tqdm(enumerate(source_rdkit), total=len(source_rdkit)):
        sims = DataStructs.BulkTanimotoSimilarity(fp1, target_rdkit)

        for j, sim in enumerate(sims):
            if same_split and i == j:
                continue  # skip self-comparison
            tanimoto_scores.append(sim)

            if sim == 1.0:
                smiles_1 = smiles_dict[source_name][i]
                smiles_2 = smiles_dict[target_name][j]
                matching_pairs.append((smiles_1, smiles_2))

    return np.array(tanimoto_scores), matching_pairs



# === FAISS Search for Cross-Split Similarity ===
def compute_tanimoto_across_splits(source_name, target_name, fingerprints, smiles_dict):
    print(f"\n🔍 Computing Tanimoto Similarity ({source_name} vs {target_name})...")

    source_fp = fingerprints[source_name]["np"]
    target_fp = fingerprints[target_name]["np"]
    source_rdkit = fingerprints[source_name]["rdkit"]
    target_rdkit = fingerprints[target_name]["rdkit"]


    if len(source_fp) == 0 or len(target_fp) == 0:
        print(f"⚠️ Skipping {source_name} vs {target_name} (No valid fingerprints)")
        return None
    
    # Debug: Check shape
    print(f"target_fp type: {type(target_fp)}, shape: {getattr(target_fp, 'shape', 'N/A')}")

    # Ensure target_fp is a NumPy array
    target_fp = np.array(target_fp, dtype=np.float32)

    # Ensure 2D format
    if len(target_fp.shape) == 1:
        target_fp = target_fp.reshape(-1, 2048)  # Adjust based on expected fingerprint size

    # Create FAISS index for target split
    faiss_index = faiss.IndexFlatL2(target_fp.shape[1])#.shape[1]
    faiss_index.add(target_fp)

    print("✅ FAISS index created successfully!")

    batch_size = 100000  # Set batch size to avoid OOM issues
    num_batches = int(np.ceil(len(source_fp) / batch_size))
    tanimoto_scores = []
    matching_pairs = []  # Store pairs where similarity is 1


    D = faiss_index.d  # Expected fingerprint size (1024)
    batch_size = (batch_size // D) * D  # Round batch_size down to nearest multiple of 1024

    for batch_start in tqdm(range(0, len(source_fp), batch_size)):
        #print('len source_fp', len(source_fp))
        batch_end = min(batch_start + batch_size, len(source_fp))
        batch_data = source_fp[batch_start:batch_end]

        # Debug print
        '''
        print(f"Adjusted batch_size: {batch_size}")
        print(f"FAISS index dimension: {D}")
        print(f"batch_data total elements: {batch_data.size}")
        '''
        
        # Convert to NumPy and ensure float32
        batch_data = np.array(batch_data, dtype=np.float32)

        # Ensure batch_data is 2D
        if len(batch_data.shape) == 1:
            batch_data = batch_data.reshape(-1, D)  # Ensure (1, D) shape

        #print(f"After reshape - batch_data shape: {batch_data.shape}")

        # Search nearest neighbor in target split
        distances, indices = faiss_index.search(batch_data, 2)

        # to save in the df        
        non_exact_source_indices = []  # Start assuming all are non-exact

        # Compute Tanimoto Similarity
        for i in range(len(batch_data)):
            neighbor_idx = indices[i][1] # getting 2nd position bc 1st its itself
            global_idx = batch_start + i

            if neighbor_idx < len(target_fp):
                sim = DataStructs.TanimotoSimilarity(source_rdkit[batch_start + i], target_rdkit[neighbor_idx])

                tanimoto_scores.append(sim)

                # ✅ If Tanimoto similarity is exactly 1, store the pair - REPASSAR
                if sim == 1.0:
                    smiles_1 = smiles_dict[source_name][batch_start+i] # get the corresponding SMILES
                    smiles_2 = smiles_dict[target_name][neighbor_idx]
                    #print('neighbors:', smiles_dict[source_name][indices[i][0]], smiles_dict[source_name][indices[i][1]])
                    matching_pairs.append((smiles_1, smiles_2))
                else:
                    non_exact_source_indices.append(global_idx)  # <- track index where sim != 1
    
            else:
                tanimoto_scores.append(0.0)  # Default similarity for invalid neighbor

    return np.array(tanimoto_scores), matching_pairs, non_exact_source_indices

# === Load Data ===
output_dir = '/home/cvalverde/Prot2Drug/2025-01-13-Prot2Drug-Journal-of-Chemical-Information-and-Modeling-data-and-code/05-New-Molecule-Generation/training-set1/results'

splits = {
    
    "train_virus": pd.read_csv('plinder_pocket_data_only1_ligands.csv', index_col=0, header='infer', sep=','),
    #"train_novirus": pd.read_csv('plinder_no_virus.csv', index_col=0, header='infer', sep=','),

    # indomain
    #"TensorDTIpocket_virus": pd.read_csv('./generated_smiles/generate_SMILES_trainedyesvirus_TensorDTI_protpocket_concat_ep6_loss0.16_all.tsv', header='infer', sep=','),
    #"ESM2pocket_virus": pd.read_csv('./generated_smiles/generate_SMILES_trainedyesvirus_protpocket_ESM2_concat_ep7_loss0.17_all.tsv', header='infer', sep=','),
    #"SaProtpocket_virus": pd.read_csv('./generated_smiles/generate_SMILES_trainedyesvirus_protpocket_SaProt_concat_ep7_loss0.15_all.tsv', header='infer', sep=','),

    #"ESM2_virus": pd.read_csv('./generated_smiles/generate_SMILES_trainedyesvirus_protpocket_ESM2_w_1_0_ep8_loss0.18_all.tsv', header='infer', sep=','),
    #"ESM2_novirus": pd.read_csv('./generated_smiles/generate_SMILES_trainednovirus_protpocket_ESM2_w_1_0_ep10_loss018_all.csv', header='infer', sep=','),

    # to infere
    #"TensorDTIpocket_virus": pd.read_csv('./results/TensorDTIpocket_virus_toinfere.csv', header='infer', sep=','),
    #"SaProtpocket_virus": pd.read_csv('./results/SaProtpocket_virus_toinfere.csv', header='infer', sep=','),
    "ESM2pocket_virus": pd.read_csv('./results/ESM2pocket_virus_toinfere.csv', header='infer', sep=','), 
    #"ESM2_virus": pd.read_csv('./results/ESM2_virus_toinfere.csv', header='infer', sep=','),
    #"ESM2_novirus": pd.read_csv('./results/ESM2_novirus_toinfere.csv', header='infer', sep=','),

    # Outdomain
    #"ESM2out_virus": pd.read_csv('./generated_smiles/generate_SMILES_outdomain_yesvirus_ESM2_w_1_0_ep8_loss0.18_all.tsv', header='infer', sep=','),
    #"ESM2pocket_out": pd.read_csv('./generated_smiles/generate_SMILES_outdomain_yesvirus_ESM2prot_pickpocket_concat_ep6_loss0.16_all.tsv', header='infer', sep=','),
    #"SaProtpocket_out": pd.read_csv('./generated_smiles/generate_SMILES_outdomain_yesvirus_SaProt_pocket_concat_ep7_loss0.15_all.tsv', header='infer', sep=','),
    #"TensorDTIpocket_out": pd.read_csv('./generated_smiles/generate_SMILES_outdomain_yesvirus_tensorDTIprot_tensorDTIpocket_concat_ep6_loss0.16_all.tsv', header='infer', sep=','),
    
    #"ESM2out_virus": pd.read_csv('./results/ESM2out_virus_toinfere.csv', header='infer', sep=','),
    #"ESM2pocket_out": pd.read_csv('./results/ESM2pocket_out_toinfere.csv', header='infer', sep=','),
    #"SaProtpocket_out": pd.read_csv('./results/SaProtpocket_out_toinfere.csv', header='infer', sep=','),
    #"TensorDTIpocket_out": pd.read_csv('./results/TensorDTIpocket_out_toinfere.csv', header='infer', sep=','),

    #RETRO
    
    #"ESM2_RET": pd.read_csv('./generated_smiles/generate_SMILES_RETRO_yesvirus_ESM2prot_pickpocket_concat_ep6_loss0.16_all_ret.csv', header='infer', sep=','),
    #"ESM2_CDK2": pd.read_csv('./generated_smiles/generate_SMILES_RETRO_yesvirus_ESM2prot_pickpocket_concat_ep6_loss0.16_all_cdk2.csv', header='infer', sep=','),
    #"SaProt_RET": pd.read_csv('./generated_smiles/generate_SMILES_RETRO_yesvirus_SaProt_prot_pickpocket_concat_ep7_loss0.15_all_ret.csv', header='infer', sep=','), 
    #"SaProt_CDK2": pd.read_csv('./generated_smiles/generate_SMILES_RETRO_yesvirus_SaProt_prot_pickpocket_concat_ep7_loss0.15_all_cdk2.csv', header='infer', sep=','),
    #"TensorDTI_RET": pd.read_csv('./generated_smiles/generate_SMILES_RETRO_yesvirus_tesndorDTI_prot_pickpocket_concat_ep6_loss0.16_all_ret.csv', header='infer', sep=','),
    #"TensorDTI_CDK2": pd.read_csv('./generated_smiles/generate_SMILES_RETRO_yesvirus_tesndorDTI_prot_pickpocket_concat_ep6_loss0.16_all_cdk2.csv', header='infer', sep=','),
    
    # RETRO SELF
    #"ESM2_RET": pd.read_csv('./results/ESM2_RET_toinfere.csv', header='infer', sep=','),
    #"ESM2_CDK2": pd.read_csv('./results/ESM2_CDK2_toinfere.csv', header='infer', sep=','),
    #"SaProt_RET": pd.read_csv('./results/SaProt_RET_toinfere.csv', header='infer', sep=','), 
    #"SaProt_CDK2": pd.read_csv('./results/SaProt_CDK2_toinfere.csv', header='infer', sep=','),
    #"TensorDTI_RET": pd.read_csv('./results/TensorDTI_CDK2_toinfere.csv', header='infer', sep=','),
    #"TensorDTI_CDK2": pd.read_csv('./results/TensorDTI_RET_toinfere.csv', header='infer', sep=','),

    #"ESM2_RET_2ivs": pd.read_csv('./results/ESM2_RET_toinfere_2ivs.csv', header='infer', sep=','),
    #"ESM2_RET_7ju5": pd.read_csv('./results/ESM2_RET_toinfere_7ju5.csv', header='infer', sep=','),
    #"ESM2_CDK2_5cu3": pd.read_csv('./results/ESM2_CDK2_toinfere_5cu3.csv', header='infer', sep=','),
    #"ESM2_CDK2_3fwq": pd.read_csv('./results/ESM2_CDK2_toinfere_3fwq.csv', header='infer', sep=','),
    #"SaProt_RET_7ju5": pd.read_csv('./results/SaProt_RET_toinfere_7ju5.csv', header='infer', sep=','), 
    #"SaProt_RET_2ivs": pd.read_csv('./results/SaProt_RET_toinfere_2ivs.csv', header='infer', sep=','), 
    #"SaProt_CDK2_5cu3": pd.read_csv('./results/SaProt_CDK2_toinfere_5cu3.csv', header='infer', sep=','),
    #"SaProt_CDK2_3fwq": pd.read_csv('./results/SaProt_CDK2_toinfere_3fwq.csv', header='infer', sep=','),
    #"TensorDTI_RET_7ju5": pd.read_csv('./results/TensorDTI_RET_toinfere_7ju5.csv', header='infer', sep=','),
    #"TensorDTI_RET_2ivs": pd.read_csv('./results/TensorDTI_RET_toinfere_2ivs.csv', header='infer', sep=','),
    #"TensorDTI_CDK2_5cu3": pd.read_csv('./results/TensorDTI_CDK2_toinfere_5cu3.csv', header='infer', sep=','),
    #"TensorDTI_CDK2_3fwq": pd.read_csv('./results/TensorDTI_CDK2_toinfere_3fwq.csv', header='infer', sep=','),

    # K - experiments only ESM2 all virus
    #"ESM2_k1": pd.read_csv('./generated_smiles/generate_SMILES_ESM2_pickpocket_concat_ep4_loss0.20_k1_all.tsv', header='infer', sep=','),
    #"ESM2_k5": pd.read_csv('./generated_smiles/generate_SMILES_ESM2_pickpocket_concat_ep4_loss0.20_k5_all.tsv', header='infer', sep=','),
    #"ESM2_k7": pd.read_csv('./generated_smiles/generate_SMILES_ESM2_pickpocket_concat_ep4_loss0.20_k7_all.tsv', header='infer', sep=','),
    #"ESM2_k10": pd.read_csv('./generated_smiles/generate_SMILES_ESM2_pickpocket_concat_ep4_loss0.20_k10_all.tsv', header='infer', sep=','),

    "ESM2_k5": pd.read_csv('./results/ESM2_k5_toinfere.csv', header='infer', sep=','), 
    "ESM2_k1": pd.read_csv('./results/ESM2_k1_toinfere.csv', header='infer', sep=','),
    "ESM2_k7": pd.read_csv('./results/ESM2_k7_toinfere.csv', header='infer', sep=','),
    "ESM2_k10": pd.read_csv('./results/ESM2_k10_toinfere.csv', header='infer', sep=','),

}

# 🔹 Compute fingerprints for each split
fingerprints = {}
smiles_dict = {}  # Store SMILES for each split

for split_name, df in splits.items():
    print(f"\n🔍 Processing {split_name} split...")
    
    # Determine correct SMILES column
    smiles_col = "cleaned_smiles" if "train" in split_name.lower() else "smiles"

    if "valid" in df.columns:
        df = df.query("valid != 0")
        # Remove rows where 'smiles' contains the '£' character
        df = df[~df["smiles"].str.contains("£", na=False)]
        print(f'{split_name} df len {len(df)}')
    
    if "split" in df.columns:
        print(f'{split_name} df len {len(df)}')
        df = df[df['Label'] != 0]
        df = df[df['split'] != 'test']
        print(f'{split_name} df len {len(df)}')

    # Convert SMILES to fingerprints
    subject_fps_np, subject_fps_rdkit, subject_valid_idx, subject_smiles = smiles_to_fp(df[smiles_col].dropna().astype(str).tolist()) # select the SMILES column
    
    # Filter DataFrame and all lists based on valid_idx
    valid_idx = np.array(subject_valid_idx)
    valid_idx = valid_idx[valid_idx < len(subject_fps_np)]

    df = df.iloc[valid_idx].reset_index(drop=True)

    # Apply same filtering to all outputs
    subject_fps_np = np.array(subject_fps_np)[valid_idx]
    subject_fps_rdkit = [subject_fps_rdkit[i] for i in valid_idx]
    subject_smiles = [subject_smiles[i] for i in valid_idx]

    #fingerprints[split_name] = pair_fingerprints
    fingerprints[split_name] = {
        "np": subject_fps_np.astype(np.float32),
        "rdkit": subject_fps_rdkit
    }
    smiles_dict[split_name] = subject_smiles  # Save SMILES for this split
    splits[split_name] = df

    print(f"✅ {split_name} split: {len(subject_fps_rdkit)} valid pairs.")


# Compute and Save Cross-Split Similarities
comparisons = [
    
    #("tensorDTIpocketw12_novirus", "tensorDTIpocketw12_novirus"),
    #("tensorDTIpocketw12_novirus", "tensorDTIpocketw12_virus"),
    #("tensorDTIpocketw12_virus", "tensorDTIpocketw12_virus"),
    
    #("ESM2_virus", "ESM2_virus"), # generated vs train

    #("SaProtpocket_virus", "SaProtpocket_virus"),

    #("TensorDTIpocket_virus", "train_virus"),
    #("SaProtpocket_virus", "train_virus"),
    #("ESM2pocket_virus", "train_virus"),
    #("ESM2_virus", "train_virus"),
    #("ESM2_novirus", "train_virus"),

    # out-domain
    #("ESM2out_virus", "ESM2out_virus"),
    #("ESM2pocket_virus", "ESM2pocket_virus"),
    #("SaProtpocket_virus", "SaProtpocket_virus"),
    #("TensorDTIpocket_virus", "TensorDTIpocket_virus"),
    #("ESM2_virus", "ESM2pocket_virus"),
    #("ESM2pocket_virus", "SaProtpocket_virus"),
    #("ESM2pocket_virus", "TensorDTIpocket_virus"),
    #("SaProtpocket_virus", "TensorDTIpocket_virus"),

    # RETRO
    #("TensorDTI_RET", "train_virus"),
    #("TensorDTI_CDK2", "train_virus"),
    #("SaProt_CDK2", "train_virus"),
    #("SaProt_RET", "train_virus"),
    #("ESM2_RET", "train_virus"),
    #("ESM2_CDK2", "train_virus"),
    
    #("TensorDTI_RET", "TensorDTI_RET"),
    #("TensorDTI_CDK2", "TensorDTI_CDK2"),
    #("SaProt_CDK2", "SaProt_CDK2"),
    #("SaProt_RET", "SaProt_RET"),
    #("ESM2_RET", "ESM2_RET"),
    #("ESM2_CDK2", "ESM2_CDK2"),

    #("TensorDTI_RET_2ivs", "TensorDTI_RET_7ju5"),
    #("TensorDTI_CDK2_3fwq", "TensorDTI_CDK2_5cu3"),
    #("SaProt_CDK2_3fwq", "SaProt_CDK2_5cu3"),
    #("SaProt_RET_2ivs", "SaProt_RET_7ju5"),
    #("ESM2_RET_2ivs", "ESM2_RET_7ju5"),
    #("ESM2_CDK2_3fwq", "ESM2_CDK2_5cu3"),

    # K-experiments ESM2,
    #("ESM2_k1", "train_virus"),
    #("ESM2_k5", "train_virus"),
    #("ESM2_k7", "train_virus"),
    #("ESM2_k10", "train_virus"),

    ("ESM2_k1", "ESM2_k1"),
    ("ESM2_k5", "ESM2_k5"),
    ("ESM2_k7", "ESM2_k7"),
    ("ESM2_k10", "ESM2_k10"),
    ("ESM2_k5", "ESM2pocket_virus"), # different ckpt
    #("ESM2_k1", "ESM2_k5"),
    #("ESM2_k1", "ESM2_k7"),
    #("ESM2_k1", "ESM2_k1"),
]

for source, target in comparisons:
    #tanimoto_scores, matching_pairs, non_exact_source_indices = compute_tanimoto_across_splits(source, target, fingerprints, smiles_dict)
    tanimoto_scores, matching_pairs = compute_tanimoto_all_vs_all(source, target, fingerprints, smiles_dict)

    if tanimoto_scores is not None:
        
        # Compute % of sequences below 50% similarity threshold
        below_threshold = np.sum(tanimoto_scores < 0.5)
        total = len(tanimoto_scores)
        percentage_below_threshold = (below_threshold / total) * 100
        print(f"📉 Percentage of sequences with Tanimoto similarity < 0.5 for {source} vs {target}: {percentage_below_threshold:.2f}% ({below_threshold}/{total})")

        # Compute % of sequences between 0.5 and 1.0 (excluding 1.0)
        between_05_1 = np.sum((tanimoto_scores >= 0.5) & (tanimoto_scores < 1.0))
        percentage_between_05_1 = (between_05_1 / total) * 100
        print(f"📊 Percentage of sequences with 0.5 ≤ Tanimoto similarity < 1.0 for {source} vs {target}: {percentage_between_05_1:.2f}% ({between_05_1}/{total})")

        # ✅ Percentage of sequences with Tanimoto == 1.0
        exact_matches = np.sum(tanimoto_scores == 1.0)
        percentage_exact_matches = (exact_matches / total) * 100
        print(f"📉 Percentage of sequences with Tanimoto similarity == 1.0 for {source} vs {target}: {percentage_exact_matches:.2f}% ({exact_matches}/{total})")
        
        # Save results
        pd.DataFrame({"tanimoto_score": tanimoto_scores}).to_csv(
            os.path.join(output_dir, f"tanimoto_{source}_vs_{target}.csv"), index=False
        )
        # Filter out scores equal to 1 before plotting
        filtered_scores = [score for score in tanimoto_scores if score < 1]

        # Plot similarity distribution
        plt.figure(figsize=(8, 5))
        sns.histplot(filtered_scores, bins=50, kde=True)
        plt.xlabel("Tanimoto Similarity")
        plt.ylabel("Frequency")
        plt.title(f"Nearest Neighbor Tanimoto Similarity ({source} vs {target})")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"tanimoto_{source}_vs_{target}.png"))
        plt.show()
    
    # Print matching pairs at the end
    if matching_pairs:
        print("\n🔍 **Identical Fingerprint Pairs (Tanimoto = 1):**")
        for smiles1, smiles2 in matching_pairs:
            print(f"{smiles1} <-> {smiles2} (Tanimoto = 1.0)")    
    '''
    print(non_exact_source_indices)
    for split_name, df in splits.items():
        
        if split_name == source:
            print(f"\n🔍 Saving {split_name} for inference...")

            if "valid" in df.columns:
                df = df.query("valid != 0").reset_index(drop=True)
            
            if non_exact_source_indices:
                df_subset = df.iloc[non_exact_source_indices]
                df_subset.to_csv(
                    os.path.join(output_dir, f"{split_name}_toinfere.csv"),
                    index=False
                )
    '''
print("\n✅ **Validation Complete. Tanimoto Similarity Across Splits Done.** 🚀")
print(f"📊 Results saved in {output_dir}")