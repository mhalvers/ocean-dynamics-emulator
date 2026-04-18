#!/usr/bin/env python3
"""Update SSH RMSE comparison bar plot with latest experiment results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def load_benchmark_metrics(run_id: str) -> dict:
    """Load benchmark metrics for a run."""
    path = Path(f"experiments/runs/{run_id}/benchmark_metrics.json")
    return json.loads(path.read_text())

def main():
    # Define experiments to plot (in order) - using available runs only
    experiments = [
        ("PCA-LSTM\nBaseline", "20260417T170400Z_ssh-u-v_pca32_h128_e100", "#fdcb6e"),
        ("ConvLSTM\nBaseline", "20260417T171548Z_ssh-u-v_pca32_h64_e100", "#6c5ce7"),
        ("ConvLSTM\nResidual", "20260417T211206Z_ssh-u-v_pca32_h64_e100", "#0984e3"),
    ]
    
    labels = []
    rmses = []
    skills = []
    colors = []
    
    for label, run_id, color in experiments:
        try:
            metrics = load_benchmark_metrics(run_id)
            ssh_metrics = metrics["summary"]["variables"]["ssh"]
            rmse = ssh_metrics["overall_rmse"]
            skill = ssh_metrics["skill_vs_persistence"]
            
            labels.append(label)
            rmses.append(rmse)
            skills.append(skill)
            colors.append(color)
            print(f"✓ {label:20s} RMSE={rmse:.4f} Skill={skill:+.4f}")
        except FileNotFoundError:
            print(f"✗ {label:20s} (run not found)")
            continue
    
    # Create figure with RMSE bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(labels))
    bars = ax.bar(x, rmses, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    
    # Add value labels on bars
    for i, (rmse, bar) in enumerate(zip(rmses, bars)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rmse:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Styling
    ax.set_xlabel("Model Architecture", fontsize=12, fontweight='bold')
    ax.set_ylabel("SSH RMSE (lower is better)", fontsize=12, fontweight='bold')
    ax.set_title("SSH Forecast Error: PCA-LSTM vs ConvLSTM Comparison\n(HYCOM 14-day input, 7-day forecast, 5-window benchmark)", 
                 fontsize=13, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(rmses) * 1.15)
    
    # Add baseline reference line
    baseline_rmse = rmses[0]
    ax.axhline(baseline_rmse, color='red', linestyle='--', linewidth=2, alpha=0.5, label=f'Baseline: {baseline_rmse:.4f}')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path("experiments/plots/ssh_rmse_comparison.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to {plot_path}")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Best RMSE:   {min(rmses):.4f}")
    print(f"  Worst RMSE:  {max(rmses):.4f}")
    print(f"  Range:       {max(rmses) - min(rmses):.4f}")
    print(f"  Improvement: {(rmses[0] - min(rmses)) / rmses[0] * 100:.1f}%")

if __name__ == "__main__":
    main()
