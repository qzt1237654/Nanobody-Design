"""
Real VHHCorpus-2M Dataset for Germline-Absorbing Diffusion

Data source: /gpfs/work/bio/zhengtaoqi24/germline/VHHCorpus-2M_top1_pairs_clean.tsv
Total samples: 1,577,197
Sequence length: 96-98 amino acids

VOCABULARY: 20-state biological diffusion (NO PAD TOKEN)
  VOCAB_SIZE = 20 (ONLY canonical amino acids)
  Valid token range: [0, 19]
  Padding uses placeholder ID 0 (with attention_mask to distinguish)

Columns used:
  - mature_v_region: mature VHH sequence
  - germline_v_region: aligned germline sequence

Data properties (pre-verified):
  - mature/germline lengths are 100% equal
  - no gaps, no missing values, no invalid amino acids
  - NO identity filtering (all samples used)
"""

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import pandas as pd
import numpy as np
from pathlib import Path


# Standard amino acid vocabulary (20 canonical amino acids)
# CRITICAL: This is the COMPLETE vocabulary for diffusion
# NO PAD, MASK, or special tokens
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

AA_TO_ID = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
ID_TO_AA = {idx: aa for idx, aa in enumerate(AMINO_ACIDS)}

VOCAB_SIZE = 20  # EXACTLY 20 states for biological diffusion

# Padding placeholder: use ID 0 (Alanine as placeholder)
# IMPORTANT: This 0 at padded positions does NOT represent real Alanine
# The attention_mask distinguishes real vs padded positions
PADDING_PLACEHOLDER_ID = 0


def tokenize_sequence(seq, max_length=None):
    """
    Tokenize amino acid sequence to indices.
    
    Args:
        seq: amino acid sequence string (only canonical 20 AA)
        max_length: if provided, pad to this length
    
    Returns:
        tensor of token indices [L] or [max_length]
        
    Raises:
        ValueError: if sequence contains invalid amino acids
    """
    tokens = []
    for aa in seq:
        if aa not in AA_TO_ID:
            raise ValueError(
                f"Invalid amino acid '{aa}' in sequence. "
                f"Only canonical 20 amino acids allowed: {AMINO_ACIDS}"
            )
        tokens.append(AA_TO_ID[aa])
    
    if max_length is not None:
        if len(tokens) > max_length:
            raise ValueError(
                f"Sequence length {len(tokens)} exceeds max_length {max_length}"
            )
        # Pad with PADDING_PLACEHOLDER_ID
        if len(tokens) < max_length:
            tokens = tokens + [PADDING_PLACEHOLDER_ID] * (max_length - len(tokens))
    
    tensor = torch.tensor(tokens, dtype=torch.long)
    
    # Verify token range
    assert tensor.min() >= 0 and tensor.max() <= 19, \
        f"Token range check failed: min={tensor.min()}, max={tensor.max()}"
    
    return tensor


def decode_sequence(tokens, attention_mask):
    """
    Decode token indices to amino acid sequence.
    
    Args:
        tokens: tensor of token indices [L]
        attention_mask: mask indicating valid positions [L]
    
    Returns:
        amino acid sequence string (only valid positions)
    """
    # Only decode valid positions
    valid_tokens = tokens[attention_mask.bool()]
    
    # Verify range
    assert valid_tokens.min() >= 0 and valid_tokens.max() <= 19, \
        f"Invalid token range in decode: min={valid_tokens.min()}, max={valid_tokens.max()}"
    
    aa_list = [ID_TO_AA[idx.item()] for idx in valid_tokens]
    return ''.join(aa_list)


def create_attention_mask(tokens, original_length, max_length):
    """
    Create attention mask: 1 for valid tokens, 0 for padding.
    
    Args:
        tokens: tensor of token indices [max_length]
        original_length: original sequence length before padding
        max_length: padded length
    
    Returns:
        attention mask tensor [max_length]
    """
    mask = torch.zeros(max_length, dtype=torch.long)
    mask[:original_length] = 1
    return mask


class VHHCorpusDataset(Dataset):
    """
    Dataset for VHHCorpus-2M germline-absorbing diffusion.
    
    Loads pre-aligned mature and germline VHH sequences from TSV file.
    Uses 20-state biological vocabulary (no PAD token).
    """
    
    def __init__(self, tsv_path, max_length=128, split='train', train_ratio=0.95, seed=42):
        """
        Args:
            tsv_path: path to clean TSV file
            max_length: maximum sequence length (for padding)
            split: 'train' or 'valid'
            train_ratio: fraction of data for training
            seed: random seed for train/valid split
        """
        self.max_length = max_length
        self.split = split
        
        print(f"Loading VHHCorpus data from: {tsv_path}")
        
        # Load TSV - only keep necessary columns to save memory
        df = pd.read_csv(tsv_path, sep='\t', usecols=['mature_v_region', 'germline_v_region'])
        
        print(f"Total samples loaded: {len(df)}")
        
        # Verify required columns
        assert 'mature_v_region' in df.columns, "Missing mature_v_region column"
        assert 'germline_v_region' in df.columns, "Missing germline_v_region column"
        
        # Create train/valid split using indices (don't copy data)
        np.random.seed(seed)
        n_total = len(df)
        n_train = int(n_total * train_ratio)
        
        indices = np.random.permutation(n_total)
        
        if split == 'train':
            self.indices = indices[:n_train]
            print(f"Training samples: {len(self.indices)}")
        else:
            self.indices = indices[n_train:]
            print(f"Validation samples: {len(self.indices)}")
        
        # Store reference to DataFrame (shared across train/valid)
        self.df = df
        
        # Verify a few samples
        print(f"\nVerifying sample data...")
        sample_idx = self.indices[0]
        mature_sample = self.df.iloc[sample_idx]['mature_v_region']
        germline_sample = self.df.iloc[sample_idx]['germline_v_region']
        
        print(f"  Sample mature length: {len(mature_sample)}")
        print(f"  Sample germline length: {len(germline_sample)}")
        
        if len(mature_sample) != len(germline_sample):
            raise ValueError(
                f"Mature/germline length mismatch at index {sample_idx}: "
                f"{len(mature_sample)} != {len(germline_sample)}"
            )
        
        if len(mature_sample) > max_length:
            raise ValueError(
                f"Sequence length {len(mature_sample)} exceeds max_length {max_length}"
            )
        
        print(f"  First 20 chars - mature: {mature_sample[:20]}")
        print(f"  First 20 chars - germline: {germline_sample[:20]}")
        
        print(f"\nDataset ready: {len(self.indices)} samples")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get actual dataframe index
        df_idx = self.indices[idx]
        
        mature_seq = self.df.iloc[df_idx]['mature_v_region']
        germline_seq = self.df.iloc[df_idx]['germline_v_region']
        
        # Verify equal length
        if len(mature_seq) != len(germline_seq):
            raise ValueError(
                f"Length mismatch at index {df_idx}: "
                f"mature={len(mature_seq)}, germline={len(germline_seq)}"
            )
        
        original_length = len(mature_seq)
        
        # Tokenize
        mature_tokens = tokenize_sequence(mature_seq, max_length=self.max_length)
        germline_tokens = tokenize_sequence(germline_seq, max_length=self.max_length)
        
        # Create attention mask
        attention_mask = create_attention_mask(mature_tokens, original_length, self.max_length)
        
        return {
            'mature': mature_tokens,
            'germline': germline_tokens,
            'attention_mask': attention_mask
        }


def get_vhh_dataloaders(tsv_path, batch_size, max_length=128, 
                        train_ratio=0.95, num_workers=4, distributed=False, seed=42):
    """
    Create train and validation data loaders for VHHCorpus.
    
    Args:
        tsv_path: path to clean TSV file
        batch_size: batch size per GPU
        max_length: maximum sequence length
        train_ratio: fraction for training
        num_workers: number of data loading workers
        distributed: use DistributedSampler
        seed: random seed for split
    
    Returns:
        train_loader, valid_loader
    """
    # Create datasets (they share the DataFrame)
    train_dataset = VHHCorpusDataset(
        tsv_path,
        max_length=max_length,
        split='train',
        train_ratio=train_ratio,
        seed=seed
    )
    
    valid_dataset = VHHCorpusDataset(
        tsv_path,
        max_length=max_length,
        split='valid',
        train_ratio=train_ratio,
        seed=seed
    )
    
    # Create samplers
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
    else:
        train_sampler = None
        valid_sampler = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # For stable batch statistics
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, valid_loader


# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("VHHCorpus Dataset - 20-State Vocabulary")
    print("="*80)
    
    print(f"\nVocabulary:")
    print(f"  Size: {VOCAB_SIZE}")
    print(f"  Amino acids: {AMINO_ACIDS}")
    print(f"  Valid token range: [0, {VOCAB_SIZE-1}]")
    print(f"  Padding placeholder ID: {PADDING_PLACEHOLDER_ID}")
    print(f"  PAD token in vocabulary: NO")
    
    # Test tokenization
    print(f"\nTesting tokenization...")
    test_seq = "ACDEFGHIKLMNPQRSTVWY"
    tokens = tokenize_sequence(test_seq, max_length=30)
    print(f"  Sequence: {test_seq}")
    print(f"  Tokens: {tokens[:20].tolist()}")
    print(f"  Token range: [{tokens.min()}, {tokens.max()}]")
    
    mask = create_attention_mask(tokens, len(test_seq), 30)
    decoded = decode_sequence(tokens, mask)
    print(f"  Decoded: {decoded}")
    print(f"  Match: {decoded == test_seq}")
    
    print("\nDataset module ready for production use!")
