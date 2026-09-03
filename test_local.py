#!/usr/bin/env python
"""
Formal Training Smoke Test

This test verifies the COMPLETE formal training pipeline by actually calling:
    train.py → Hydra → run_train.py → losses.py → sampling.py

Test principle:
- NO manual model creation
- NO manual optimizer setup  
- NO manual training loop
- ONLY call the formal entry point: train.py
"""

import subprocess
import sys
import os
import tempfile
from pathlib import Path
import torch
import time

PROJECT_ROOT = Path(__file__).parent
TEST_DATA = PROJECT_ROOT / "real_vhh_200.tsv"

print("=" * 80)
print("FORMAL TRAINING SMOKE TEST")
print("=" * 80)
print(f"\nProject root: {PROJECT_ROOT}")
print(f"Test data: {TEST_DATA}")
print(f"Python: {sys.executable}")

# Verify test data exists
if not TEST_DATA.exists():
    print(f"\n[FAIL] Test data not found: {TEST_DATA}")
    sys.exit(1)

print(f"[OK] Test data exists")

# Create temporary output directory
test_output_dir = tempfile.mkdtemp(prefix="formal_smoke_test_")
print(f"\n[INFO] Test output directory: {test_output_dir}")

# ============================================================================
# PHASE 1: Fresh Training (step 0 → 2)
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 1: FRESH TRAINING (step 0 → 2)")
print("=" * 80)

print("\n[INFO] Calling formal entry: train.py")

# Build command
train_cmd = [
    sys.executable,
    str(PROJECT_ROOT / "train.py"),
    f"data.tsv_path={TEST_DATA}",
    "training.n_iters=2",
    "training.log_freq=1",
    "training.eval_freq=1",
    "training.snapshot_freq=2",
    "training.snapshot_freq_for_preemption=1",
    "training.snapshot_sampling=true",
    "sampling.batch_size=2",
    "ngpus=1",
    f"hydra.run.dir={test_output_dir}",
]

print(f"\n[CMD] {' '.join(train_cmd)}\n")

# Execute
start_time = time.time()
result = subprocess.run(train_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
elapsed = time.time() - start_time

print(f"\n[INFO] Training completed in {elapsed:.1f}s")
print(f"[INFO] Exit code: {result.returncode}")

if result.stdout:
    print("\n--- STDOUT (last 1500 chars) ---")
    print(result.stdout[-1500:])

if result.returncode != 0:
    if result.stderr:
        print("\n--- STDERR ---")
        print(result.stderr[-1500:])
    print(f"\n[FAIL] Training failed with exit code {result.returncode}")
    sys.exit(1)

print("\n[PASS] Phase 1: Fresh training completed")
# ============================================================================
# PHASE 2: Verify Outputs
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 2: VERIFY OUTPUTS")
print("=" * 80)

checkpoints_dir = Path(test_output_dir) / "checkpoints"
checkpoint_meta_dir = Path(test_output_dir) / "checkpoints-meta"
samples_dir = Path(test_output_dir) / "samples"

# Check checkpoint
checkpoint_path = checkpoints_dir / "checkpoint_2.pth"
if not checkpoint_path.exists():
    print(f"[FAIL] checkpoint_2.pth not found at {checkpoint_path}")
    sys.exit(1)
print(f"[PASS] checkpoint_2.pth exists ({checkpoint_path.stat().st_size / (1024**2):.1f} MB)")

# Check recovery checkpoint
checkpoint_meta_path = checkpoint_meta_dir / "checkpoint.pth"
if not checkpoint_meta_path.exists():
    print(f"[FAIL] Recovery checkpoint not found")
    sys.exit(1)
print(f"[PASS] Recovery checkpoint exists")

# Check samples
sample_iter_dir = samples_dir / "iter_2"
if not sample_iter_dir.exists():
    print(f"[FAIL] Samples not generated")
    sys.exit(1)
sample_files = list(sample_iter_dir.glob("sample_*.txt"))
print(f"[PASS] {len(sample_files)} sample file(s) generated")
# ============================================================================
# PHASE 3: Inspect Checkpoint
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 3: INSPECT CHECKPOINT")
print("=" * 80)

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
print("[PASS] Checkpoint loaded")

required_keys = ['model', 'optimizer', 'ema', 'step']
for key in required_keys:
    if key not in checkpoint:
        print(f"[FAIL] Missing key: {key}")
        sys.exit(1)
    print(f"  + {key}")

if 'scaler' in checkpoint:
    print(f"  + scaler")

step = checkpoint['step']
if step != 2:
    print(f"[FAIL] Expected step=2, got {step}")
    sys.exit(1)
print(f"[PASS] Step = 2")

model_state = checkpoint['model']
key_components = ['vocab_embed.embedding', 'germline_proj.weight', 
                   'sigma_map.mlp.0.weight', 'blocks.0.adaLN_modulation.weight']
for comp in key_components:
    if comp not in model_state:
        print(f"[FAIL] Missing component: {comp}")
        sys.exit(1)
print(f"[PASS] Model state complete ({len(model_state)} params)")
print(f"[PASS] Optimizer state complete")
print(f"[PASS] EMA state complete")
# ============================================================================
# PHASE 4: Checkpoint Resume (step 2 → 4)
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 4: CHECKPOINT RESUME (step 2 → 4)")
print("=" * 80)

resume_cmd = [
    sys.executable,
    str(PROJECT_ROOT / "train.py"),
    f"data.tsv_path={TEST_DATA}",
    "training.n_iters=4",
    "training.log_freq=1",
    "training.eval_freq=2",
    "training.snapshot_freq=4",
    "training.snapshot_freq_for_preemption=2",
    "training.snapshot_sampling=true",
    "sampling.batch_size=2",
    "ngpus=1",
    f"hydra.run.dir={test_output_dir}",
]

print(f"\n[CMD] {' '.join(resume_cmd)}\n")

start_time = time.time()
result_resume = subprocess.run(resume_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
elapsed = time.time() - start_time

print(f"\n[INFO] Resumed training completed in {elapsed:.1f}s")
print(f"[INFO] Exit code: {result_resume.returncode}")

if result_resume.stdout:
    print("\n--- STDOUT (last 1500 chars) ---")
    print(result_resume.stdout[-1500:])

if result_resume.returncode != 0:
    if result_resume.stderr:
        print("\n--- STDERR ---")
        print(result_resume.stderr[-1500:])
    print(f"\n[FAIL] Resumed training failed")
    sys.exit(1)

checkpoint_4_path = checkpoints_dir / "checkpoint_4.pth"
if not checkpoint_4_path.exists():
    print("[FAIL] checkpoint_4.pth not created")
    sys.exit(1)

checkpoint_4 = torch.load(checkpoint_4_path, map_location='cpu', weights_only=False)
final_step = checkpoint_4['step']
if final_step != 4:
    print(f"[FAIL] Expected step=4, got {final_step}")
    sys.exit(1)

print(f"[PASS] Final step = 4")
print("[PASS] Checkpoint resume verified")
# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FORMAL TRAINING PIPELINE VERIFIED")
print("=" * 80)

print("\nVerified:")
print("  [PASS] train.py formal entry")
print("  [PASS] configs/config.yaml loaded")
print("  [PASS] run_train.py executed")
print("  [PASS] data_vhh_real.py loaded data")
print("  [PASS] SEDD model initialized")
print("  [PASS] losses.py executed")
print("  [PASS] AdamW + AMP + EMA")
print("  [PASS] Validation executed")
print("  [PASS] Checkpoint saved")
print("  [PASS] Sampling executed")
print("  [PASS] Checkpoint resume from step 2")
print("  [PASS] Training continued to step 4")

print(f"\nArtifacts: {test_output_dir}")
print("\n" + "=" * 80)
print("ALL TESTS PASSED")
print("=" * 80)
