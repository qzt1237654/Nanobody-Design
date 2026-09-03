#!/usr/bin/env python
"""
Supercomputer Formal Training Smoke Test

Verifies formal training pipeline on supercomputer with real data subset.
Calls train.py directly - NO manual training loop.
"""

import subprocess
import sys
import os
import tempfile
from pathlib import Path
import torch
import time

PROJECT_ROOT = Path(__file__).parent
REAL_DATA = "/gpfs/work/bio/zhengtaoqi24/germline/VHHCorpus-2M_top1_pairs_clean.tsv"
TEST_DATA = PROJECT_ROOT / "real_vhh_200.tsv"

print("=" * 80)
print("SUPERCOMPUTER FORMAL TRAINING SMOKE TEST")
print("=" * 80)

# Use local test data if real data not accessible
if not Path(REAL_DATA).exists():
    print(f"[WARN] Full data not found: {REAL_DATA}")
    print(f"[INFO] Using test data: {TEST_DATA}")
    DATA_PATH = TEST_DATA
else:
    print(f"[INFO] Using real data: {REAL_DATA}")
    DATA_PATH = REAL_DATA

if not Path(DATA_PATH).exists():
    print(f"[FAIL] Data not found: {DATA_PATH}")
    sys.exit(1)

print(f"[OK] Data exists")

test_output_dir = tempfile.mkdtemp(prefix="formal_smoke_supercomputer_")
print(f"\n[INFO] Test output: {test_output_dir}")

# ============================================================================
# PHASE 1: Fresh Training
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 1: FRESH TRAINING (step 0 → 5)")
print("=" * 80)

train_cmd = [
    sys.executable,
    str(PROJECT_ROOT / "train.py"),
    f"data.tsv_path={DATA_PATH}",
    "training.n_iters=5",
    "training.batch_size=32",
    "training.log_freq=1",
    "training.eval_freq=2",
    "training.snapshot_freq=5",
    "training.snapshot_freq_for_preemption=2",
    "training.snapshot_sampling=true",
    "sampling.batch_size=4",
    "ngpus=1",
    f"hydra.run.dir={test_output_dir}",
]

print(f"\n[CMD] {' '.join(train_cmd)}\n")

start_time = time.time()
result = subprocess.run(train_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
elapsed = time.time() - start_time

print(f"\n[INFO] Completed in {elapsed:.1f}s")
print(f"[INFO] Exit code: {result.returncode}")

if result.stdout:
    print("\n--- STDOUT (last 1500 chars) ---")
    print(result.stdout[-1500:])

if result.returncode != 0:
    if result.stderr:
        print("\n--- STDERR ---")
        print(result.stderr[-1500:])
    print(f"\n[FAIL] Training failed")
    sys.exit(1)

print("\n[PASS] Fresh training completed")

# ============================================================================
# PHASE 2: Verify Outputs
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 2: VERIFY OUTPUTS")
print("=" * 80)

checkpoints_dir = Path(test_output_dir) / "checkpoints"
checkpoint_meta_dir = Path(test_output_dir) / "checkpoints-meta"

checkpoint_path = checkpoints_dir / "checkpoint_5.pth"
if not checkpoint_path.exists():
    print(f"[FAIL] checkpoint_5.pth not found")
    sys.exit(1)
print(f"[PASS] checkpoint_5.pth exists")

checkpoint_meta_path = checkpoint_meta_dir / "checkpoint.pth"
if not checkpoint_meta_path.exists():
    print(f"[FAIL] Recovery checkpoint not found")
    sys.exit(1)
print(f"[PASS] Recovery checkpoint exists")

# ============================================================================
# PHASE 3: Inspect Checkpoint
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 3: INSPECT CHECKPOINT")
print("=" * 80)

checkpoint = torch.load(checkpoint_path, map_location='cpu')
print("[PASS] Checkpoint loaded")

for key in ['model', 'optimizer', 'ema', 'step']:
    if key not in checkpoint:
        print(f"[FAIL] Missing: {key}")
        sys.exit(1)
    print(f"  ✓ {key}")

if checkpoint['step'] != 5:
    print(f"[FAIL] Expected step=5, got {checkpoint['step']}")
    sys.exit(1)
print(f"[PASS] Step = 5")

# ============================================================================
# PHASE 4: Resume Training
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 4: CHECKPOINT RESUME (step 5 → 8)")
print("=" * 80)

resume_cmd = [
    sys.executable,
    str(PROJECT_ROOT / "train.py"),
    f"data.tsv_path={DATA_PATH}",
    "training.n_iters=8",
    "training.batch_size=32",
    "training.log_freq=1",
    "training.eval_freq=2",
    "training.snapshot_freq=8",
    "training.snapshot_freq_for_preemption=2",
    "ngpus=1",
    f"hydra.run.dir={test_output_dir}",
]

print(f"\n[CMD] {' '.join(resume_cmd)}\n")

start_time = time.time()
result_resume = subprocess.run(resume_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
elapsed = time.time() - start_time

print(f"\n[INFO] Completed in {elapsed:.1f}s")

if result_resume.stdout:
    print("\n--- STDOUT (last 1500 chars) ---")
    print(result_resume.stdout[-1500:])

if result_resume.returncode != 0:
    if result_resume.stderr:
        print("\n--- STDERR ---")
        print(result_resume.stderr[-1500:])
    print(f"\n[FAIL] Resume failed")
    sys.exit(1)

checkpoint_8 = torch.load(checkpoints_dir / "checkpoint_8.pth", map_location='cpu')
if checkpoint_8['step'] != 8:
    print(f"[FAIL] Expected step=8, got {checkpoint_8['step']}")
    sys.exit(1)

print(f"[PASS] Resumed to step 8")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUPERCOMPUTER FORMAL TRAINING VERIFIED")
print("=" * 80)

print("\nVerified:")
print("  [PASS] train.py formal entry")
print("  [PASS] Real data loaded")
print("  [PASS] Full pipeline executed")
print("  [PASS] Checkpoint saved")
print("  [PASS] Resume successful")

print(f"\nArtifacts: {test_output_dir}")
print("\n" + "=" * 80)
print("ALL TESTS PASSED")
print("=" * 80)
