#!/usr/bin/env python3
"""
Compare multiple FIRE2 label propagation runs.

Loads metrics from multiple run_fire2_problabels.py outputs and creates
a comparison table/plot showing how different parameters affect performance.

Usage:
    python compare_fire2_runs.py ../output/fire2_mock/problabels_* 
"""

import sys
import json
from pathlib import Path
import pandas as pd

def load_run_metrics(run_dir):
    """Load metrics from a single run directory."""
    metrics_path = Path(run_dir) / 'classification_metrics_hard.json'
    prob_path = Path(run_dir) / 'probabilistic_metrics.json'
    prop_path = Path(run_dir) / 'label_propagation_results.pkl'
    
    if not metrics_path.exists():
        return None
    
    with open(metrics_path, 'r') as f:
        hard_metrics = json.load(f)
    
    with open(prob_path, 'r') as f:
        prob_metrics = json.load(f)
    
    # Try to load propagation params
    params = {}
    if prop_path.exists():
        import pickle
        with open(prop_path, 'rb') as f:
            prop_data = pickle.load(f)
            params = prop_data.get('label_config', {})
    
    return {
        'run_name': Path(run_dir).name,
        'accuracy': hard_metrics['classification_report']['accuracy'],
        'purity': hard_metrics['overall_purity'],
        'completeness': hard_metrics['overall_completeness'],
        'f1_weighted': hard_metrics['classification_report']['weighted avg']['f1-score'],
        'mean_confidence': prob_metrics['overall_mean_confidence'],
        'top1_accuracy': prob_metrics['top_k_accuracy'].get('top_1', 0),
        'k': params.get('n_neighbors', '?'),
        'n_min': params.get('minimum_labeled_neighbors', '?'),
        'threshold': params.get('prob_threshold', '?'),
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_fire2_runs.py <run_dir1> [run_dir2] ...")
        sys.exit(1)
    
    # Load all runs
    results = []
    for run_dir in sys.argv[1:]:
        metrics = load_run_metrics(run_dir)
        if metrics:
            results.append(metrics)
    
    if not results:
        print("No valid runs found!")
        sys.exit(1)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by accuracy
    df = df.sort_values('accuracy', ascending=False)
    
    # Print comparison table
    print("\n" + "=" * 100)
    print("FIRE2 LABEL PROPAGATION COMPARISON")
    print("=" * 100)
    print()
    print(df.to_string(index=False))
    print()
    print("=" * 100)
    print(f"\nBest accuracy: {df.iloc[0]['run_name']} ({df.iloc[0]['accuracy']:.1%})")
    print(f"  Parameters: k={df.iloc[0]['k']}, n_min={df.iloc[0]['n_min']}, threshold={df.iloc[0]['threshold']}")
    
    # Save to CSV
    output_path = Path('../output/fire2_runs_comparison.csv')
    df.to_csv(output_path, index=False)
    print(f"\nComparison saved to: {output_path}")

if __name__ == '__main__':
    main()
