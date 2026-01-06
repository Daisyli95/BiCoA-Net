#!/usr/bin/env python3
"""
Prediction Script for Protein-Ligand Binding (pkoff values)
Uses the trained model.pt model to predict case study data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
import warnings
import esm
from transformers import AutoTokenizer, AutoModel
import math

# RDKit for molecular descriptors
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    print("Warning: RDKit not available. Install with: conda install -c conda-forge rdkit")
    RDKIT_AVAILABLE = False

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ======================== Molecular Descriptors ========================

def compute_molecular_descriptors(smiles):
    """Compute RDKit molecular descriptors"""
    if not RDKIT_AVAILABLE:
        return np.zeros(200)
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(200)
        
        descriptors = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.NumSaturatedRings(mol),
            Descriptors.NumHeteroatoms(mol),
            Descriptors.RingCount(mol),
            Descriptors.BertzCT(mol),
            Descriptors.Chi0v(mol),
            Descriptors.Chi1v(mol),
            Descriptors.Kappa1(mol),
            Descriptors.Kappa2(mol),
            Descriptors.Kappa3(mol),
        ]
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=182)
        fp_array = np.array(fp)
        
        full_desc = np.concatenate([descriptors, fp_array])
        return full_desc
        
    except Exception as e:
        print(f"  Error computing descriptors for SMILES: {e}")
        return np.zeros(200)

# ======================== Embedding Functions ========================

class MolFormerEmbedder:
    def __init__(self, device='cuda'):
        self.device = device
        print("Loading MolFormer...")
        self.tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
        self.model = AutoModel.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k", use_safetensors=True)
        self.model = self.model.to(device).eval()
        print("✓ MolFormer loaded")
    
    def embed_smiles(self, smiles, max_length=512):
        try:
            with torch.no_grad():
                inputs = self.tokenizer(smiles, return_tensors="pt", padding=True, 
                                       truncation=True, max_length=max_length)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                return outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze(), True
        except:
            return np.zeros(768), False

class ESM2Embedder:
    def __init__(self, model_name="esm2_t33_650M_UR50D", device='cuda'):
        self.device = device
        print(f"Loading ESM-2: {model_name}...")
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model = self.model.to(device).eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        print("✓ ESM-2 loaded")
        
    def embed_protein(self, sequence, max_length=1024):
        try:
            with torch.no_grad():
                if len(sequence) > max_length:
                    sequence = sequence[:max_length]
                data = [("protein", sequence)]
                _, _, batch_tokens = self.batch_converter(data)
                batch_tokens = batch_tokens.to(self.device)
                results = self.model(batch_tokens, repr_layers=[33])
                return results["representations"][33].mean(dim=1).cpu().numpy().squeeze(), True
        except:
            return np.zeros(1280), False

def normalize_embeddings(embeddings):
    """Normalize embeddings (subtract mean, divide by std)"""
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    mean = embeddings.mean(axis=0, keepdims=True)
    std = embeddings.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (embeddings - mean) / std

# ======================== Model Architecture (Must match training) ========================

class MultiHeadCoAttentionWithTemp(nn.Module):
    """Multi-head co-attention with temperature scaling"""
    def __init__(self, d_model, n_heads=8, dropout=0.1, temperature=1.0):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.temperature = temperature
        
        self.q_lig = nn.Linear(d_model, d_model)
        self.k_lig = nn.Linear(d_model, d_model)
        self.v_lig = nn.Linear(d_model, d_model)
        
        self.q_prot = nn.Linear(d_model, d_model)
        self.k_prot = nn.Linear(d_model, d_model)
        self.v_prot = nn.Linear(d_model, d_model)
        
        self.out_lig = nn.Linear(d_model, d_model)
        self.out_prot = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout * 0.5)
        self.scale = math.sqrt(self.d_k) * self.temperature
    
    def forward(self, lig, prot):
        batch_size = lig.size(0)
        
        def reshape(x):
            return x.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Ligand attends to protein
        q_l = reshape(self.q_lig(lig.unsqueeze(1)))
        k_p = reshape(self.k_prot(prot.unsqueeze(1)))
        v_p = reshape(self.v_prot(prot.unsqueeze(1)))
        
        attn_l = torch.matmul(q_l, k_p.transpose(-2, -1)) / self.scale
        attn_l = F.softmax(attn_l, dim=-1)
        attn_l = self.attn_dropout(attn_l)
        
        out_l = torch.matmul(attn_l, v_p)
        out_l = out_l.transpose(1, 2).contiguous().view(batch_size, 1, self.d_model)
        out_l = self.out_lig(out_l).squeeze(1)
        
        # Protein attends to ligand
        q_p = reshape(self.q_prot(prot.unsqueeze(1)))
        k_l = reshape(self.k_lig(lig.unsqueeze(1)))
        v_l = reshape(self.v_lig(lig.unsqueeze(1)))
        
        attn_p = torch.matmul(q_p, k_l.transpose(-2, -1)) / self.scale
        attn_p = F.softmax(attn_p, dim=-1)
        attn_p = self.attn_dropout(attn_p)
        
        out_p = torch.matmul(attn_p, v_l)
        out_p = out_p.transpose(1, 2).contiguous().view(batch_size, 1, self.d_model)
        out_p = self.out_prot(out_p).squeeze(1)
        
        return out_l, out_p


class RefinedCoAttentionBlock(nn.Module):
    """Refined co-attention block with residual connections"""
    def __init__(self, d_model, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        
        self.attention = MultiHeadCoAttentionWithTemp(d_model, n_heads, dropout)
        
        self.norm1_lig = nn.LayerNorm(d_model)
        self.norm1_prot = nn.LayerNorm(d_model)
        self.norm2_lig = nn.LayerNorm(d_model)
        self.norm2_prot = nn.LayerNorm(d_model)
        
        self.ff_lig = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.ff_prot = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.alpha_lig = nn.Parameter(torch.ones(1))
        self.alpha_prot = nn.Parameter(torch.ones(1))
    
    def forward(self, lig, prot):
        lig_norm = self.norm1_lig(lig)
        prot_norm = self.norm1_prot(prot)
        
        lig_attn, prot_attn = self.attention(lig_norm, prot_norm)
        
        lig = lig + self.alpha_lig * lig_attn
        prot = prot + self.alpha_prot * prot_attn
        
        lig = lig + self.ff_lig(self.norm2_lig(lig))
        prot = prot + self.ff_prot(self.norm2_prot(prot))
        
        return lig, prot


class ImprovedBilinearFusion(nn.Module):
    """Improved bilinear fusion"""
    def __init__(self, lig_dim, prot_dim, out_dim, dropout=0.1):
        super().__init__()
        
        self.bilinear = nn.Bilinear(lig_dim, prot_dim, out_dim)
        
        self.lig_proj = nn.Sequential(
            nn.Linear(lig_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.prot_proj = nn.Sequential(
            nn.Linear(prot_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        
        self.mult_proj = nn.Linear(lig_dim, out_dim)
        
        self.gate = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, lig, prot):
        bilinear_out = self.bilinear(lig, prot)
        
        linear_lig = self.lig_proj(lig)
        linear_prot = self.prot_proj(prot)
        
        mult = self.mult_proj(lig * prot)
        
        gate_input = torch.cat([bilinear_out, mult], dim=-1)
        gate_weights = self.gate(gate_input)
        
        fused = gate_weights * bilinear_out + (1 - gate_weights) * mult
        fused = fused + linear_lig + linear_prot
        
        fused = self.norm(fused)
        fused = F.gelu(fused)
        fused = self.dropout(fused)
        
        return fused


class RefinedProteinLigandModel(nn.Module):
    """Refined Protein-Ligand Model"""
    def __init__(self, smiles_dim, protein_dim, descriptor_dim, 
                 d_model=768, n_blocks=4, n_heads=12, d_ff=3072, dropout=0.2):
        super().__init__()
        
        self.smiles_proj = nn.Sequential(
            nn.Linear(smiles_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.descriptor_proj = nn.Sequential(
            nn.Linear(descriptor_dim, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        self.co_attention_blocks = nn.ModuleList([
            RefinedCoAttentionBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_blocks)
        ])
        
        self.fusion = ImprovedBilinearFusion(d_model, d_model, d_model, dropout)
        
        self.descriptor_fusion = nn.Sequential(
            nn.Linear(d_model + d_model // 4, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
    def forward(self, smiles_feat, protein_feat, mol_desc):
        lig = self.smiles_proj(smiles_feat)
        prot = self.protein_proj(protein_feat)
        desc = self.descriptor_proj(mol_desc)
        
        for block in self.co_attention_blocks:
            lig, prot = block(lig, prot)
        
        fused = self.fusion(lig, prot)
        
        fused_with_desc = torch.cat([fused, desc], dim=-1)
        final_feat = self.descriptor_fusion(fused_with_desc)
        
        output = self.output_head(final_feat).squeeze(-1)
        
        return output

# ======================== Prediction Function ========================

def predict_casestudy(checkpoint_path, data_files, output_dir='./predictions', device='cuda'):
    """
    Predict pkoff values for case study data
    
    Args:
        checkpoint_path: Path to the trained model checkpoint (.pt file)
        data_files: List of paths to case study Excel files
        output_dir: Directory to save predictions
        device: 'cuda' or 'cpu'
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if CUDA is available
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    print(f"\n{'='*70}")
    print("PROTEIN-LIGAND BINDING PREDICTION (pkoff)")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Number of files: {len(data_files)}")
    print(f"{'='*70}\n")
    
    # Load checkpoint
    print("Loading model checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    normalization = checkpoint['normalization']
    
    print("\nModel Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nNormalization Parameters:")
    print(f"  Target mean: {normalization['mean']:.4f}")
    print(f"  Target std: {normalization['std']:.4f}")
    
    # Initialize model
    model = RefinedProteinLigandModel(
        smiles_dim=config['smiles_dim'],
        protein_dim=config['protein_dim'],
        descriptor_dim=config['descriptor_dim'],
        d_model=config['d_model'],
        n_blocks=config['n_blocks'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        dropout=config['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded successfully\n")
    
    # Initialize embedders
    molformer = MolFormerEmbedder(device=device)
    esm2 = ESM2Embedder(device=device)
    
    # Process each file
    all_predictions = []
    
    for file_path in data_files:
        file_name = os.path.basename(file_path)
        print(f"\n{'='*70}")
        print(f"Processing: {file_name}")
        print(f"{'='*70}")
        
        # Load data
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} samples")
        
        # Compute embeddings
        print("\nComputing SMILES embeddings...")
        smiles_embeddings = []
        for smiles in tqdm(df['smiles'].tolist(), desc="SMILES"):
            emb, _ = molformer.embed_smiles(smiles)
            smiles_embeddings.append(emb)
        smiles_embeddings = normalize_embeddings(np.array(smiles_embeddings))
        
        print("\nComputing protein embeddings...")
        protein_embeddings = []
        for fasta in tqdm(df['FASTA'].tolist(), desc="Proteins"):
            emb, _ = esm2.embed_protein(fasta)
            protein_embeddings.append(emb)
        protein_embeddings = normalize_embeddings(np.array(protein_embeddings))
        
        print("\nComputing molecular descriptors...")
        mol_descriptors = []
        for smiles in tqdm(df['smiles'].tolist(), desc="Descriptors"):
            desc = compute_molecular_descriptors(smiles)
            mol_descriptors.append(desc)
        mol_descriptors = np.array(mol_descriptors)
        
        # Normalize descriptors using training statistics
        desc_mean = normalization['desc_mean']
        desc_std = normalization['desc_std']
        mol_descriptors = (mol_descriptors - desc_mean) / desc_std
        
        # Make predictions
        print("\nMaking predictions...")
        predictions = []
        
        with torch.no_grad():
            for i in tqdm(range(len(df)), desc="Predicting"):
                smiles_feat = torch.FloatTensor(smiles_embeddings[i:i+1]).to(device)
                protein_feat = torch.FloatTensor(protein_embeddings[i:i+1]).to(device)
                desc_feat = torch.FloatTensor(mol_descriptors[i:i+1]).to(device)
                
                # Predict (normalized)
                pred_normalized = model(smiles_feat, protein_feat, desc_feat)
                
                # Denormalize to original scale
                pred_original = pred_normalized.cpu().item() * normalization['std'] + normalization['mean']
                predictions.append(pred_original)
        
        # Add predictions to dataframe
        df['predicted_pkoff'] = predictions
        
        # Calculate errors if true pkoff values exist
        if 'pkoff' in df.columns and not df['pkoff'].isna().all():
            df['prediction_error'] = df['predicted_pkoff'] - df['pkoff']
            df['absolute_error'] = np.abs(df['prediction_error'])
            
            # Calculate metrics
            valid_mask = ~df['pkoff'].isna()
            if valid_mask.sum() > 0:
                true_vals = df.loc[valid_mask, 'pkoff'].values
                pred_vals = df.loc[valid_mask, 'predicted_pkoff'].values
                
                mse = np.mean((true_vals - pred_vals) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(true_vals - pred_vals))
                
                from scipy.stats import pearsonr, spearmanr
                pearson_r, _ = pearsonr(true_vals, pred_vals)
                spearman_r, _ = spearmanr(true_vals, pred_vals)
                
                print(f"\n{'='*70}")
                print("PREDICTION METRICS")
                print(f"{'='*70}")
                print(f"  Samples with true values: {valid_mask.sum()}")
                print(f"  RMSE: {rmse:.4f}")
                print(f"  MAE: {mae:.4f}")
                print(f"  Pearson r: {pearson_r:.4f}")
                print(f"  Spearman r: {spearman_r:.4f}")
                print(f"{'='*70}")
        
        # Save predictions
        output_file = os.path.join(output_dir, f'predictions_{os.path.splitext(file_name)[0]}.xlsx')
        print(f"\n✓ Predictions saved to: {output_file}")
        
        # Also save as CSV for easier viewing
        csv_file = os.path.join(output_dir, f'predictions_{os.path.splitext(file_name)[0]}.csv')
        df.to_csv(csv_file, index=False, float_format='%.6f')
        print(f"✓ CSV saved to: {csv_file}")
        
        all_predictions.append(df)
    
    print(f"\n{'='*70}")
    print("PREDICTION COMPLETE")
    print(f"{'='*70}")
    print(f"Total files processed: {len(data_files)}")
    print(f"All predictions saved to: {output_dir}")
    print(f"{'='*70}\n")
    
    return all_predictions


def main():
    parser = argparse.ArgumentParser(description='Predict pkoff values using trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--data_dir', type=str, default='.',
                        help='Directory containing case study Excel files')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                        help='Directory to save predictions')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--files', type=str, nargs='+', default=None,
                        help='Specific files to process (optional)')
    
    args = parser.parse_args()
    
    # Find case study files
    if args.files is not None:
        data_files = [os.path.join(args.data_dir, f) for f in args.files]
    else:
        # Auto-detect case study files
        data_files = []
        for file in os.listdir(args.data_dir):
            if file.startswith('casestudy-') and file.endswith('.xlsx'):
                data_files.append(os.path.join(args.data_dir, file))
    
    if not data_files:
        print("Error: No case study files found!")
        print(f"Looked in: {args.data_dir}")
        return
    
    # Run predictions
    predict_casestudy(
        checkpoint_path=args.checkpoint,
        data_files=data_files,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == '__main__':
    main()
