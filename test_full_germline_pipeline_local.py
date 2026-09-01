"""
Local Full Pipeline Test: VHH Germline-Absorbing Diffusion

Tests complete pipeline with mock data: Dataset → forward → loss → backward → sampling
"""

import torch
import tempfile
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("LOCAL FULL PIPELINE TEST")
print("="*80)

from data_vhh_real import get_vhh_dataloaders, VOCAB_SIZE, AMINO_ACIDS, decode_sequence
from model import SEDD
import graph_lib_germline
import sampling
from noise_lib import LogLinearNoise
from omegaconf import OmegaConf

# Create mock TSV
print("\n[1/12] Creating mock data...")
n_samples = 200
mock_data = []
np.random.seed(42)

for i in range(n_samples):
    length = 96 + (i % 3)
    germline_seq = ''.join([AMINO_ACIDS[np.random.randint(0, 20)] for _ in range(length)])
    mature_list = list(germline_seq)
    n_mutations = int(length * 0.15)
    mutation_pos = np.random.choice(length, n_mutations, replace=False)
    for pos in mutation_pos:
        mature_list[pos] = AMINO_ACIDS[np.random.randint(0, 20)]
    mock_data.append({'mature_v_region': ''.join(mature_list), 'germline_v_region': germline_seq})

df = pd.DataFrame(mock_data)
temp_tsv = tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False)
df.to_csv(temp_tsv.name, sep='\t', index=False)
temp_tsv.close()
print(f"  Mock TSV: {len(df)} samples, length 96-98")

# Load data
print("\n[2/12] Loading data...")
train_loader, valid_loader = get_vhh_dataloaders(
    tsv_path=temp_tsv.name, batch_size=32, max_length=128,
    train_ratio=0.90, num_workers=0, distributed=False, seed=42
)
batch_data = next(iter(train_loader))
mature = batch_data['mature']
germline = batch_data['germline']
attention_mask = batch_data['attention_mask']
print(f"  Batch: {mature.shape}, token range [{mature.min()}, {mature.max()}]")
assert mature.max() <= 19, "Token range check failed!"

# Create graph
print("\n[3/12] Creating graph...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
graph = graph_lib_germline.GermlineAbsorbing(dim=20)
print(f"  Graph dim: {graph.dim}, device: {device}")
assert graph.dim == 20

# Create model
print("\n[4/12] Creating model...")
config = OmegaConf.create({
    'tokens': 20,
    'graph': {'type': 'germline_absorb'},
    'model': {
        'length': 128,
        'hidden_size': 256,
        'n_heads': 4,
        'n_blocks': 2,
        'cond_dim': 256,
        'dropout': 0.0,
        'scale_by_sigma': False
    },
    'sampling': {
        'predictor': 'analytic',
        'steps': 8,
        'noise_removal': True
    }
})
model = SEDD(config).to(device)
print(f"  Params: {sum(p.numel() for p in model.parameters()):,}, vocab: {model.vocab_embed.embedding.shape[0]}")
assert model.vocab_embed.embedding.shape[0] == 20

# Forward corruption
print("\n[5/12] Forward corruption...")
mature_d = mature.to(device)
germline_d = germline.to(device)
mask_d = attention_mask.to(device)

# sigma for graph operations: [B, 1]
sigma_graph = torch.ones(mature_d.shape[0], 1, device=device) * 0.5
perturbed = graph.sample_transition(mature_d, sigma_graph, germline=germline_d)
perturbed = torch.where(mask_d.bool(), perturbed, mature_d)
absorbed = (perturbed == germline_d)
print(f"  Absorbed: {absorbed.sum()}, sigma: {sigma_graph[0,0].item()}")

# Model forward
print("\n[6/12] Model forward...")
model.train()

# sigma for model: [B] (squeeze from [B,1])
sigma_model = sigma_graph.squeeze(-1)

# Disable autocast to avoid dtype issues in test
with torch.amp.autocast('cuda', enabled=False):
    log_score = model(perturbed, sigma_model, germline=germline_d, attention_mask=mask_d)

print(f"  Score: {log_score.shape}, range [{log_score.min().item():.3f}, {log_score.max().item():.3f}]")
assert log_score.shape[-1] == 20

# Score entropy
print("\n[7/12] Score entropy...")
# Ensure log_score is float32 for graph operations
loss_pos = graph.score_entropy(log_score.float(), sigma_graph, perturbed, mature_d, germline=germline_d)
loss_pos = loss_pos * mask_d.float()
pad_loss = loss_pos[mask_d == 0]
assert torch.all(pad_loss == 0), "PAD should have zero loss"
loss = loss_pos.sum(dim=-1).mean()
print(f"  Loss: {loss.item():.6f}, PAD loss: 0 [OK]")
assert torch.isfinite(loss) and loss.item() >= 0

# Backward
print("\n[8/12] Backward...")
loss.backward()
grad_norm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
gp_grad = sum(p.grad.norm().item()**2 for n,p in model.named_parameters() 
              if p.grad is not None and 'germline_proj' in n)**0.5
print(f"  Grad norm: {grad_norm:.6f}, germline_proj: {gp_grad:.6f}")
# Note: germline_proj is zero-initialized, gradients may be very small
assert grad_norm > 0, "Total gradients should be non-zero"

# Optimizer
print("\n[9/12] Optimizer...")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
optimizer.step()
optimizer.zero_grad()
print("  Step completed [OK]")

# Validation
print("\n[10/12] Validation...")
model.eval()
valid_data = next(iter(valid_loader))
vm, vg, va = valid_data['mature'].to(device), valid_data['germline'].to(device), valid_data['attention_mask'].to(device)

# sigma for validation: [batch, 1] for graph, [batch] for model
sigma_graph_valid = sigma_graph[:vm.shape[0]]
sigma_model_valid = sigma_graph_valid.squeeze(-1)

with torch.no_grad():
    vp = graph.sample_transition(vm, sigma_graph_valid, germline=vg)
    vp = torch.where(va.bool(), vp, vm)
    vs = model(vp, sigma_model_valid, germline=vg, attention_mask=va)
    # Ensure float32 for graph operations
    vl = graph.score_entropy(vs.float(), sigma_graph_valid, vp, vm, germline=vg)
    vl = (vl * va.float()).sum(dim=-1).mean()
print(f"  Valid loss: {vl.item():.6f}")
assert torch.isfinite(vl)

# Real reverse diffusion sampling
print("\n[11/12] Real reverse diffusion sampling...")

sample_germline = vg[:1].clone()
sample_mask = va[:1].clone()

noise = LogLinearNoise().to(device)

sampling_fn = sampling.get_sampling_fn(
    config=config,
    graph=graph,
    noise=noise,
    batch_dims=tuple(sample_germline.shape),
    eps=1e-3,
    device=device,
    germline=sample_germline,
    attention_mask=sample_mask,
)

with torch.no_grad():
    x_samp = sampling_fn(model)

print(
    f"  Sample range: "
    f"[{x_samp.min().item()}, {x_samp.max().item()}]"
)

assert x_samp.min().item() >= 0
assert x_samp.max().item() <= 19, "Sample tokens invalid!"

# Padding positions must stay frozen
assert torch.equal(
    x_samp[sample_mask == 0],
    sample_germline[sample_mask == 0]
), "Padding positions changed during sampling!"

decoded = decode_sequence(
    x_samp[0],
    sample_mask[0]
)

print(f"  Decoded length: {len(decoded)}")
print(f"  Decoded (first 50): {decoded[:50]}")
print("  Real reverse sampling completed [OK]")

# Decode verify
print("\n[12/12] Decode verification...")
from data_vhh_real import tokenize_sequence
orig_dec = decode_sequence(mature[0], attention_mask[0])
reenc = tokenize_sequence(orig_dec, max_length=128)
vlen = attention_mask[0].sum().item()
match = torch.equal(mature[0, :vlen], reenc[:vlen])
print(f"  Decode/encode match: {match}")
assert match, "Decode should be exact!"

# Cleanup
import os
os.unlink(temp_tsv.name)

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\n[SUCCESS] All 12 steps passed!")
print(f"\nMetrics:")
print(f"  Train loss: {loss.item():.6f}")
print(f"  Valid loss: {vl.item():.6f}")
print(f"  Grad norm: {grad_norm:.6f}")
print(f"  Vocab size: 20")
print(f"  Token range: [0, 19]")
print(f"  PAD in vocab: NO")
print("\n" + "="*80)
print("LOCAL FULL PIPELINE: VERIFIED [OK]")
print("="*80)
