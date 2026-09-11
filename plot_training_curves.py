"""
Plot training curves from log file.

Generates smoothed training curves for:
- overall_recovery
- mutation_recovery_per_seq
- mutation_recovery_union

Saves PNG files to figures/ directory.
"""

import argparse
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


def parse_log_file(log_path):
    """
    Parse log file and extract training metrics.
    
    Returns:
        dict with keys: 'step', 'loss', 'overall_recovery', 
        'mutation_recovery_per_seq', 'mutation_recovery_union'
    """
    metrics = defaultdict(list)
    
    # Pattern for training log lines
    pattern = re.compile(
        r'step: (\d+), '
        r'lr: ([\d.e+-]+), '
        r'loss: ([\d.e+-]+), '
        r'overall_recovery: ([\d.]+), '
        r'mutation_recovery_per_seq: ([\d.]+), '
        r'mutation_recovery_union: ([\d.]+)'
    )
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(3))
                overall_rec = float(match.group(4))
                mutation_per_seq = float(match.group(5))
                mutation_union = float(match.group(6))
                
                metrics['step'].append(step)
                metrics['loss'].append(loss)
                metrics['overall_recovery'].append(overall_rec)
                metrics['mutation_recovery_per_seq'].append(mutation_per_seq)
                metrics['mutation_recovery_union'].append(mutation_union)
    
    return metrics


def smooth_curve(data, sigma=2):
    """Apply Gaussian smoothing to curve."""
    if len(data) < 3:
        return data
    return gaussian_filter1d(data, sigma=sigma)


def plot_curve(steps, values, ylabel, title, output_path, smooth_sigma=2):
    """Plot a single smoothed training curve."""
    plt.figure(figsize=(10, 6))
    
    # Plot original data with low alpha
    plt.plot(steps, values, alpha=0.3, color='blue', linewidth=0.5)
    
    # Plot smoothed curve
    smoothed = smooth_curve(values, sigma=smooth_sigma)
    plt.plot(steps, smoothed, color='blue', linewidth=2, label='Smoothed')
    
    plt.xlabel('Step', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Auto-adjust y-axis range for better visualization
    y_min, y_max = min(values), max(values)
    y_range = y_max - y_min
    plt.ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from log file")
    parser.add_argument(
        '--log',
        type=str,
        required=True,
        help='Path to log file (e.g., checkpoints/2026.09.03/014847/logs/log.txt)',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='figures',
        help='Output directory for figures (default: figures)',
    )
    parser.add_argument(
        '--smooth',
        type=float,
        default=2.0,
        help='Gaussian smoothing sigma (default: 2.0)',
    )
    
    args = parser.parse_args()
    
    # Parse log file
    print(f"Parsing log file: {args.log}")
    metrics = parse_log_file(args.log)
    
    if len(metrics['step']) == 0:
        print("No training data found in log file!")
        return
    
    print(f"Found {len(metrics['step'])} data points")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    steps = metrics['step']
    
    # Plot overall recovery
    plot_curve(
        steps,
        metrics['overall_recovery'],
        ylabel='Overall Recovery',
        title='Overall Recovery vs Training Steps',
        output_path=os.path.join(args.output_dir, 'overall_recovery.png'),
        smooth_sigma=args.smooth,
    )
    
    # Plot mutation recovery per-seq
    plot_curve(
        steps,
        metrics['mutation_recovery_per_seq'],
        ylabel='Mutation Recovery (Per-Sequence)',
        title='Mutation Recovery (Per-Sequence) vs Training Steps',
        output_path=os.path.join(args.output_dir, 'mutation_recovery_per_seq.png'),
        smooth_sigma=args.smooth,
    )
    
    # Plot mutation recovery union
    plot_curve(
        steps,
        metrics['mutation_recovery_union'],
        ylabel='Mutation Recovery (Union)',
        title='Mutation Recovery (Union) vs Training Steps',
        output_path=os.path.join(args.output_dir, 'mutation_recovery_union.png'),
        smooth_sigma=args.smooth,
    )
    
    print(f"\nAll figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
