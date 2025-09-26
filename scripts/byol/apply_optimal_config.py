#!/usr/bin/env python3
"""
Apply Optimal Configuration

Updates the main BYOL configuration with the optimal parameters found
through hyperparameter optimization.
"""

import yaml
import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime

def apply_optimal_config():
    """Apply the optimal configuration from optimization results"""

    print("Applying Optimal BYOL Configuration")
    print("=" * 40)

    # Load optimization results
    log_path = Path("../../local_data/byol_results/hyperparameter_optimization/experiment_log.csv")
    df = pd.read_csv(log_path)

    # Find best configuration
    best_idx = df['objective_score'].idxmin()
    best_result = df.iloc[best_idx]

    print(f"Best experiment: {best_result['experiment_id']}")
    print(f"Objective score: {best_result['objective_score']:.4f}")
    print(f"Improvement: {((0.6366 - best_result['objective_score']) / 0.6366 * 100):.1f}%")
    print()

    # Load current configuration
    config_path = Path("byol_config.yaml")
    backup_path = Path(f"byol_config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")

    # Create backup
    shutil.copy(config_path, backup_path)
    print(f"Backup created: {backup_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Apply optimal parameters
    config['model']['projection_size'] = int(best_result['projection_size'])
    config['model']['projection_hidden_size'] = int(best_result['projection_hidden_size'])
    config['training']['learning_rate'] = float(best_result['learning_rate'])
    config['training']['num_epochs'] = int(best_result['num_epochs'])
    config['training']['batch_size'] = int(best_result['batch_size'])
    config['analysis']['n_neighbors'] = int(best_result['n_neighbors'])
    config['analysis']['min_dist'] = float(best_result['min_dist'])
    config['analysis']['pca_components'] = int(best_result['pca_components'])

    # Save updated configuration
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print("\nOptimal configuration applied to byol_config.yaml:")
    print(f"  projection_size: {int(best_result['projection_size'])}")
    print(f"  projection_hidden_size: {int(best_result['projection_hidden_size'])}")
    print(f"  learning_rate: {float(best_result['learning_rate'])}")
    print(f"  num_epochs: {int(best_result['num_epochs'])}")
    print(f"  batch_size: {int(best_result['batch_size'])}")
    print(f"  n_neighbors: {int(best_result['n_neighbors'])}")
    print(f"  min_dist: {float(best_result['min_dist'])}")
    print(f"  pca_components: {int(best_result['pca_components'])}")

    print(f"\nExpected performance:")
    print(f"  Merger distance: {best_result['merger_norm_dist']:.4f}")
    print(f"  Fragmentation distance: {best_result['fragmentation_norm_dist']:.4f}")
    print(f"  Objective score: {best_result['objective_score']:.4f}")

if __name__ == "__main__":
    apply_optimal_config()