"""
Analyze recovery metrics for VHH germline-absorbing diffusion.

Computes two types of mutation recovery:
1. Per-sequence recovery: Each sequence evaluated independently, then averaged
2. Per-germline union recovery: For each germline, predictions are correct if 
   they match ANY mature sequence with that germline at that position

Generates a Markdown report showing top 10 germlines by recovery with mutation details.
"""

import argparse
import os
from collections import defaultdict

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

import graph_lib
import noise_lib
import utils
from data_vhh_real import (
    tokenize_sequence,
    decode_sequence,
    AA_TO_ID,
    ID_TO_AA,
    AMINO_ACIDS,
)
from model import SEDD
from model.ema import ExponentialMovingAverage


def load_model_and_config(checkpoint_path, device, config_path=None):
    """Load model and config from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Find config file
    if config_path is None:
        work_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        config_path = os.path.join(work_dir, ".hydra", "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    print(f"Loading config: {config_path}")
    
    # Load config
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(config_path)
    
    # Initialize model
    model = SEDD(cfg).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load EMA parameters
    ema = ExponentialMovingAverage(
        model.parameters(),
        decay=cfg.training.ema,
    )
    
    if "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"])
    
    # Apply EMA weights to model
    ema.copy_to(model.parameters())
    
    model.eval()
    
    print(f"Model loaded. Step: {checkpoint.get('step', 'unknown')}")
    
    return model, cfg, checkpoint.get("step", 0)


def load_full_dataset(tsv_path, max_length):
    """Load all data and group by germline."""
    print(f"Loading dataset: {tsv_path}")
    
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        usecols=["mature_v_region", "germline_v_region"],
    )
    
    print(f"Total sequences: {len(df)}")
    
    # Group by germline
    germline_groups = defaultdict(list)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Grouping by germline"):
        mature_seq = row["mature_v_region"]
        germline_seq = row["germline_v_region"]
        
        # Tokenize
        mature_tokens = tokenize_sequence(mature_seq, max_length=max_length)
        germline_tokens = tokenize_sequence(germline_seq, max_length=max_length)
        
        original_length = len(mature_seq)
        attention_mask = torch.zeros(max_length, dtype=torch.long)
        attention_mask[:original_length] = 1
        
        germline_groups[germline_seq].append({
            "mature": mature_tokens,
            "germline": germline_tokens,
            "attention_mask": attention_mask,
            "mature_seq": mature_seq,
            "germline_seq": germline_seq,
        })
    
    print(f"Unique germlines: {len(germline_groups)}")
    
    return germline_groups


def compute_recovery_for_germline(
    model,
    graph,
    noise,
    germline_data,
    device,
    n_samples=100,
):
    """
    Compute both types of recovery for one germline group.
    
    Args:
        model: trained model
        graph: germline absorbing graph
        noise: noise schedule
        germline_data: list of dicts with mature/germline/mask
        device: torch device
        n_samples: max sequences to evaluate per germline
        
    Returns:
        dict with recovery metrics and details
    """
    # Sample at most n_samples
    if len(germline_data) > n_samples:
        indices = np.random.choice(len(germline_data), n_samples, replace=False)
        germline_data = [germline_data[i] for i in indices]
    
    n_seqs = len(germline_data)
    
    # Prepare batch
    mature_batch = torch.stack([d["mature"] for d in germline_data]).to(device)
    germline_batch = torch.stack([d["germline"] for d in germline_data]).to(device)
    mask_batch = torch.stack([d["attention_mask"] for d in germline_data]).to(device)
    
    with torch.no_grad():
        # Sample random timesteps
        t = torch.rand(n_seqs, device=device) * 0.999 + 0.001
        sigma, _ = noise(t)
        
        # Forward corruption
        perturbed = graph.sample_transition(
            mature_batch,
            sigma[:, None],
            germline=germline_batch,
        )
        
        perturbed = torch.where(
            mask_batch.bool(),
            perturbed,
            mature_batch,
        )
        
        # Model prediction
        log_score = model(
            perturbed,
            sigma,
            germline=germline_batch,
            attention_mask=mask_batch,
        )
        
        pred = log_score.argmax(dim=-1)
    
    # Compute per-sequence recovery
    per_seq_recoveries = []
    
    for i in range(n_seqs):
        valid_mask = mask_batch[i].bool()
        mutation_mask = (
            valid_mask
            & (mature_batch[i] != germline_batch[i])
            & (perturbed[i] == germline_batch[i])
        )
        
        if mutation_mask.sum().item() > 0:
            mutation_correct = (pred[i] == mature_batch[i]) & mutation_mask
            recovery = mutation_correct.sum().item() / mutation_mask.sum().item()
            per_seq_recoveries.append(recovery)
    
    avg_per_seq_recovery = (
        np.mean(per_seq_recoveries) if per_seq_recoveries else 0.0
    )
    
    # Compute union recovery
    # Build union set: for each position, collect all possible amino acids
    germline_ref = germline_batch[0]  # All same
    mask_ref = mask_batch[0]
    seq_len = mask_ref.sum().item()
    
    # Union set for each position
    position_unions = [set() for _ in range(seq_len)]
    
    for i in range(n_seqs):
        for pos in range(seq_len):
            position_unions[pos].add(mature_batch[i, pos].item())
    
    # Check predictions against union
    union_correct_total = 0
    union_mutation_total = 0
    
    for i in range(n_seqs):
        for pos in range(seq_len):
            valid = mask_batch[i, pos].item() == 1
            is_mutation = mature_batch[i, pos].item() != germline_batch[i, pos].item()
            is_absorbed = perturbed[i, pos].item() == germline_batch[i, pos].item()
            
            if valid and is_mutation and is_absorbed:
                pred_aa = pred[i, pos].item()
                
                if pred_aa in position_unions[pos]:
                    union_correct_total += 1
                
                union_mutation_total += 1
    
    union_recovery = (
        union_correct_total / union_mutation_total if union_mutation_total > 0 else 0.0
    )
    
    return {
        "n_sequences": n_seqs,
        "per_seq_recovery": avg_per_seq_recovery,
        "union_recovery": union_recovery,
        "germline_data": germline_data,
    }


def generate_markdown_report(results, output_path, checkpoint_step):
    """Generate markdown report with top germlines."""
    # Sort by union recovery
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]["union_recovery"],
        reverse=True,
    )
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# VHH Recovery Analysis Report\n\n")
        f.write(f"**Checkpoint Step:** {checkpoint_step}\n\n")
        f.write(f"**Total Germlines Analyzed:** {len(results)}\n\n")
        f.write("---\n\n")
        
        f.write("## Top 10 Germlines by Union Recovery\n\n")
        
        for rank, (germline_seq, result) in enumerate(sorted_results[:10], 1):
            f.write(f"### Rank {rank}\n\n")
            f.write(f"**Per-Sequence Recovery:** {result['per_seq_recovery']:.4f}\n\n")
            f.write(f"**Union Recovery:** {result['union_recovery']:.4f}\n\n")
            f.write(f"**Number of Sequences:** {result['n_sequences']}\n\n")
            
            f.write(f"**Germline Sequence:**\n")
            f.write(f"```\n{germline_seq}\n```\n\n")
            
            # Show up to 5 example mature sequences with mutation positions
            f.write(f"**Example Mature Sequences (showing up to 5):**\n\n")
            
            for idx, data in enumerate(result["germline_data"][:5], 1):
                mature_seq = data["mature_seq"]
                
                # Find mutation positions
                mutation_positions = []
                for pos, (m, g) in enumerate(zip(mature_seq, germline_seq)):
                    if m != g:
                        mutation_positions.append(f"{pos}:{g}->{m}")
                
                f.write(f"{idx}. `{mature_seq}`\n")
                f.write(f"   - Mutations: {', '.join(mutation_positions) if mutation_positions else 'None'}\n\n")
            
            f.write("---\n\n")
    
    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze VHH recovery metrics")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output markdown file path (default: auto-generated in checkpoint dir)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: auto-detect from checkpoint location)",
    )
    parser.add_argument(
        "--tsv_path",
        type=str,
        default=None,
        help="Path to VHHCorpus TSV file (default: use path from config)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Max sequences to evaluate per germline",
    )
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    model, cfg, step = load_model_and_config(args.checkpoint, device, args.config)
    
    # Initialize graph and noise
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    
    # Use provided tsv_path or fall back to config
    tsv_path = args.tsv_path if args.tsv_path else cfg.data.tsv_path
    
    # Load dataset
    germline_groups = load_full_dataset(
        tsv_path,
        cfg.data.max_length,
    )
    
    # Compute recovery for each germline
    print(f"\nComputing recovery for {len(germline_groups)} germlines...")
    
    results = {}
    
    for germline_seq, germline_data in tqdm(germline_groups.items(), desc="Analyzing germlines"):
        result = compute_recovery_for_germline(
            model,
            graph,
            noise,
            germline_data,
            device,
            n_samples=args.n_samples,
        )
        results[germline_seq] = result
    
    # Generate report
    if args.output is None:
        work_dir = os.path.dirname(os.path.dirname(args.checkpoint))
        output_path = os.path.join(work_dir, f"recovery_analysis_step_{step}.md")
    else:
        output_path = args.output
    
    generate_markdown_report(results, output_path, step)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
