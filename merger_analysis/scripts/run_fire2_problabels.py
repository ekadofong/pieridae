#!/usr/bin/env python3
"""
FIRE2 PCA & Probability Label Re-computation (Semi-Supervised K-NN)

Recomputes PCA and label propagation from pre-trained BYOL embeddings without retraining.
Useful for experimenting with PCA dimensionality and KNN parameters quickly.

This script:
1. Loads pre-computed BYOL embeddings from run_fire2_mock.py output
2. **Recomputes PCA** with configurable dimensionality (n_components, variance threshold)
3. Allows modification of label propagation parameters (K, thresholds, etc.)
4. Recomputes probability labels using K-NN propagation
5. Generates metrics and visualizations

Usage
-----
# Recompute with different PCA dimensionality
python run_fire2_problabels.py --pca-components 5
python run_fire2_problabels.py --pca-components 20
python run_fire2_problabels.py --variance-threshold 0.99

# Experiment with both PCA and KNN parameters
python run_fire2_problabels.py --pca-components 10 --n-neighbors 100

# Use pre-computed PCA, only change KNN (faster)
python run_fire2_problabels.py --use-precomputed-pca --n-neighbors 100

# Change training fraction (creates new train/test split)
python run_fire2_problabels.py --train-fraction 0.1

# Adjust confidence threshold for pseudo-labels
python run_fire2_problabels.py --prob-threshold 0.95 --n-min-auto 20

# Custom input/output directories
python run_fire2_problabels.py --input-dir ../output/fire2_mock_experiment \
    --output-dir ../output/pca_experiment_5comp --pca-components 5

Relationship to run_fire2_mock.py
----------------------------------
Similar to run_analysis.py vs merger_classification.ipynb:
- run_fire2_mock.py: Full pipeline (images → BYOL training → PCA → propagation)
- run_fire2_problabels.py: Fast iteration (BYOL embeddings → PCA → propagation)

This allows rapid experimentation with PCA and KNN without expensive BYOL retraining.
Speed: ~0.1sec for PCA + ~0.01sec for KNN vs ~10min for full BYOL training.
"""

import sys
import argparse
import logging
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, Any

import yaml
import numpy as np

# Add pieridae to path
sys.path.insert(0, str(Path(__file__).parents[2]))

# Import required functions from run_fire2_mock and byol modules
from run_fire2_mock import (
    create_sparse_labels,
    propagate_labels_knn,
    compute_probabilistic_metrics,
    create_visualizations,
    create_probability_analysis_figure,
    setup_logging
)
from pieridae.starbursts.byol import compute_classification_metrics


def load_precomputed_data(input_dir: Path, logger: logging.Logger, load_pca: bool = True):
    """
    Load pre-computed embeddings and metadata from run_fire2_mock.py output.

    Parameters
    ----------
    input_dir : Path
        Directory containing output from run_fire2_mock.py
    logger : logging.Logger
        Logger instance
    load_pca : bool, default=True
        If True, load pre-computed PCA results. If False, only load original embeddings.

    Returns
    -------
    data : dict
        Dictionary containing:
        - images: Original images (for QA only, not used in propagation)
        - img_names: Image identifiers
        - true_labels: Ground truth labels (0-indexed)
        - class_names: Mapping from class ID to galaxy tag
        - embeddings_original: BYOL embeddings before PCA (always loaded)
        - embeddings_pca: PCA-reduced embeddings (only if load_pca=True)
        - scaler: Fitted StandardScaler (only if load_pca=True)
        - pca: Fitted PCA model (only if load_pca=True)
    """
    logger.info(f"Loading pre-computed data from: {input_dir}")

    # Load mock images data (metadata)
    data_path = input_dir / 'fire2_mock_data.pkl'
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"Please run run_fire2_mock.py first to generate embeddings."
        )

    with open(data_path, 'rb') as f:
        mock_data = pickle.load(f)

    logger.info(f"  Loaded {len(mock_data['images'])} images")
    logger.info(f"  Classes: {mock_data['class_names']}")

    # Load dimensionality reduction results
    dim_path = input_dir / 'dimensionality_reduction_results.pkl'
    if not dim_path.exists():
        raise FileNotFoundError(
            f"Dimensionality reduction results not found: {dim_path}\n"
            f"Please run run_fire2_mock.py first to generate embeddings."
        )

    with open(dim_path, 'rb') as f:
        dim_data = pickle.load(f)

    # Always load original embeddings
    logger.info(f"  BYOL embeddings shape: {dim_data['embeddings_original'].shape}")

    # Combine into single dict
    data = {
        'images': mock_data['images'],
        'img_names': mock_data['img_names'],
        'true_labels': mock_data['true_labels'],
        'class_names': mock_data['class_names'],
        'embeddings_original': dim_data['embeddings_original'],
    }

    # Optionally load PCA results
    if load_pca:
        data['embeddings_pca'] = dim_data['embeddings_pca']
        data['scaler'] = dim_data['scaler']
        data['pca'] = dim_data['pca']

        logger.info(f"  Pre-computed PCA shape: {dim_data['embeddings_pca'].shape}")
        explained_var = dim_data['pca'].explained_variance_ratio_.sum() * 100
        logger.info(f"  Pre-computed PCA explained variance: {explained_var:.1f}%")

    return data


def recompute_pca(
    embeddings_original: np.ndarray,
    n_components: int = None,
    variance_threshold: float = 0.95,
    config: dict = None,
    logger: logging.Logger = None
) -> Tuple[np.ndarray, Any, Any]:
    """
    Recompute PCA on original BYOL embeddings with custom parameters.

    Parameters
    ----------
    embeddings_original : np.ndarray
        Original BYOL embeddings before PCA, shape (N, D)
    n_components : int, optional
        Number of PCA components. If None, auto-determined from variance_threshold
    variance_threshold : float, default=0.95
        Explained variance threshold for automatic component selection
    config : dict, optional
        Configuration dictionary (used for other EmbeddingAnalyzer settings)
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    embeddings_pca : np.ndarray
        PCA-transformed embeddings, shape (N, n_components)
    scaler : StandardScaler
        Fitted scaler
    pca : PCA
        Fitted PCA model
    """
    from pieridae.starbursts.byol import EmbeddingAnalyzer

    if logger:
        logger.info(f"\nRecomputing PCA from BYOL embeddings...")
        if n_components is not None:
            logger.info(f"  Target components: {n_components} (fixed)")
        else:
            logger.info(f"  Target variance: {variance_threshold*100:.1f}% (auto components)")

    # Create temporary config with overrides
    temp_config = config.copy() if config else {'analysis': {}}
    if 'analysis' not in temp_config:
        temp_config['analysis'] = {}

    # Override PCA settings
    if n_components is not None:
        temp_config['analysis']['pca_components'] = n_components
    else:
        temp_config['analysis']['pca_components'] = None  # Force auto-selection

    temp_config['analysis']['explained_variance_threshold'] = variance_threshold

    # Use EmbeddingAnalyzer to compute PCA
    analyzer = EmbeddingAnalyzer(temp_config, logger)
    embeddings_pca = analyzer.compute_pca(embeddings_original, n_components=n_components)

    if logger:
        logger.info(f"  Final PCA components: {embeddings_pca.shape[1]}")
        explained_var = analyzer.pca.explained_variance_ratio_.sum() * 100
        logger.info(f"  Explained variance: {explained_var:.1f}%")

    return embeddings_pca, analyzer.scaler, analyzer.pca


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Recompute FIRE2 probability labels from pre-trained embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script loads pre-computed BYOL embeddings and recomputes PCA and label
propagation with configurable parameters. Much faster than retraining BYOL.

Examples:
  # Recompute with different PCA dimensionality
  python run_fire2_problabels.py --pca-components 5
  python run_fire2_problabels.py --pca-components 20
  python run_fire2_problabels.py --variance-threshold 0.99

  # Experiment with both PCA and KNN
  python run_fire2_problabels.py --pca-components 10 --n-neighbors 100

  # Use precomputed PCA, only change KNN (faster)
  python run_fire2_problabels.py --use-precomputed-pca --n-neighbors 100

  # New train/test split
  python run_fire2_problabels.py --train-fraction 0.2 --random-seed 123
        """
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        default='../output/fire2_mock',
        help='Directory with pre-computed embeddings (default: ../output/fire2_mock)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (default: input-dir/problabels_recompute)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='../config.yaml',
        help='Path to configuration YAML file'
    )

    # Train/test split parameters
    parser.add_argument(
        '--train-fraction',
        type=float,
        help='Fraction of labels for training (default: use existing split if available)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        help='Random seed for train/test split (default: use existing or 42)'
    )
    parser.add_argument(
        '--no-stratified-split',
        action='store_true',
        help='Disable stratified sampling'
    )
    parser.add_argument(
        '--use-existing-split',
        action='store_true',
        help='Reuse existing train/test split from input directory'
    )

    # PCA parameters
    parser.add_argument(
        '--pca-components',
        type=int,
        help='Number of PCA components (default: auto from variance threshold)'
    )
    parser.add_argument(
        '--variance-threshold',
        type=float,
        help='Explained variance threshold for auto PCA selection (default: 0.95)'
    )
    parser.add_argument(
        '--use-precomputed-pca',
        action='store_true',
        help='Skip PCA recomputation, use existing (faster, only changes KNN params)'
    )

    # Label propagation parameters
    parser.add_argument(
        '--n-neighbors',
        type=int,
        help='Number of nearest neighbors (default: from config or 50)'
    )
    parser.add_argument(
        '--n-min',
        type=int,
        help='Minimum labeled neighbors for prediction (default: from config or 5)'
    )
    parser.add_argument(
        '--n-min-auto',
        type=int,
        help='Minimum neighbors for auto-labeling (default: from config or 15)'
    )
    parser.add_argument(
        '--prob-threshold',
        type=float,
        help='Confidence threshold for pseudo-labels (default: from config or 0.9)'
    )

    args = parser.parse_args()

    # Setup
    logger = setup_logging()
    config = yaml.safe_load(open(args.config, 'r'))

    input_dir = Path(args.input_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir / 'problabels_recompute'

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("FIRE2 PROBABILITY LABEL RECOMPUTATION")
    logger.info("=" * 70)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    try:
        # Load pre-computed data (with or without PCA depending on args)
        data = load_precomputed_data(input_dir, logger, load_pca=args.use_precomputed_pca)

        true_labels = data['true_labels']
        embeddings_original = data['embeddings_original']
        class_names = data['class_names']
        img_names = data['img_names']

        # Recompute PCA or use pre-computed
        if args.use_precomputed_pca:
            logger.info("\nUsing pre-computed PCA embeddings...")
            embeddings_pca = data['embeddings_pca']
            scaler = data['scaler']
            pca_model = data['pca']
        else:
            # Recompute PCA with potentially different parameters
            n_components = args.pca_components
            variance_threshold = args.variance_threshold if args.variance_threshold else 0.95

            embeddings_pca, scaler, pca_model = recompute_pca(
                embeddings_original,
                n_components=n_components,
                variance_threshold=variance_threshold,
                config=config,
                logger=logger
            )

            # Save recomputed PCA for reference
            pca_results_path = output_dir / 'recomputed_pca.pkl'
            with open(pca_results_path, 'wb') as f:
                pickle.dump({
                    'embeddings_pca': embeddings_pca,
                    'scaler': scaler,
                    'pca': pca_model,
                    'n_components_used': embeddings_pca.shape[1],
                    'n_components_requested': n_components,
                    'variance_threshold': variance_threshold,
                    'explained_variance': pca_model.explained_variance_ratio_.sum()
                }, f)
            logger.info(f"  Recomputed PCA saved to: {pca_results_path}")

        # Handle train/test split
        existing_split_path = input_dir / 'train_test_split.pkl'

        if args.use_existing_split and existing_split_path.exists():
            logger.info("\nUsing existing train/test split...")
            with open(existing_split_path, 'rb') as f:
                split_data = pickle.load(f)
            train_mask = split_data['train_mask']
            test_mask = split_data['test_mask']
            train_fraction = split_data.get('train_fraction', 0.3)
            random_seed = split_data.get('random_seed', 42)

            logger.info(f"  Train samples: {train_mask.sum()}")
            logger.info(f"  Test samples: {test_mask.sum()}")
            logger.info(f"  Original train fraction: {train_fraction}")
            logger.info(f"  Original random seed: {random_seed}")

            # Create sparse labels from existing split
            sparse_labels = np.zeros_like(true_labels)
            sparse_labels[train_mask] = true_labels[train_mask] + 1  # 1-indexed

        else:
            # Create new train/test split
            train_fraction = args.train_fraction if args.train_fraction else 0.3
            random_seed = args.random_seed if args.random_seed else 42
            stratified = not args.no_stratified_split

            logger.info(f"\nCreating new train/test split...")
            logger.info(f"  Train fraction: {train_fraction}")
            logger.info(f"  Random seed: {random_seed}")
            logger.info(f"  Stratified: {stratified}")

            sparse_labels, train_mask, test_mask = create_sparse_labels(
                true_labels,
                sample_fraction=train_fraction,
                random_seed=random_seed,
                stratified=stratified,
                logger=logger
            )

        # Override config with command-line arguments
        label_config = config.get('labels', {})
        if args.n_neighbors:
            label_config['n_neighbors'] = args.n_neighbors
        if args.n_min:
            label_config['minimum_labeled_neighbors'] = args.n_min
        if args.n_min_auto:
            label_config['minimum_labeled_neighbors_for_autoprop'] = args.n_min_auto
        if args.prob_threshold:
            label_config['prob_threshold'] = args.prob_threshold

        config['labels'] = label_config

        logger.info(f"\nLabel propagation parameters:")
        logger.info(f"  n_neighbors: {label_config.get('n_neighbors', 50)}")
        logger.info(f"  n_min: {label_config.get('minimum_labeled_neighbors', 5)}")
        logger.info(f"  n_min_auto: {label_config.get('minimum_labeled_neighbors_for_autoprop', 15)}")
        logger.info(f"  prob_threshold: {label_config.get('prob_threshold', 0.9)}")

        # Propagate labels using iterative K-NN
        n_classes = len(class_names)
        predicted_labels, n_labels, prob_labels, stats = propagate_labels_knn(
            embeddings_pca,
            sparse_labels,
            n_classes=n_classes,
            config=config,
            logger=logger
        )

        # Track pseudo-labels in test set
        pseudo_in_test = (predicted_labels[test_mask] > 0) & (sparse_labels[test_mask] == 0)
        logger.info(
            f"\nPseudo-labels in test set: {pseudo_in_test.sum()}/{test_mask.sum()} "
            f"({pseudo_in_test.sum()/test_mask.sum()*100:.1f}%)"
        )

        # Save results
        results = {
            'train_mask': train_mask,
            'test_mask': test_mask,
            'train_fraction': train_fraction,
            'random_seed': random_seed,
            'propagation_stats': stats,
            'predicted_labels': predicted_labels,
            'n_labels': n_labels,
            'label_config': label_config
        }

        results_path = output_dir / 'label_propagation_results.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"Results saved to: {results_path}")

        # Save probability labels
        prob_labels_path = output_dir / 'probability_labels.npy'
        np.save(prob_labels_path, prob_labels)
        logger.info(f"Probability labels saved to: {prob_labels_path}")

        # Compute metrics on TEST SET ONLY
        logger.info("\n" + "=" * 70)
        logger.info("COMPUTING METRICS ON TEST SET")
        logger.info("=" * 70)

        # Hard classification metrics (argmax of probabilities)
        logger.info("\n1. Hard Classification Metrics (argmax)...")
        final_predictions = np.argmax(prob_labels, axis=1)
        metrics_hard = compute_classification_metrics(
            true_labels[test_mask],
            final_predictions[test_mask],
            class_names
        )

        # Probabilistic metrics
        logger.info("\n2. Probabilistic Metrics...")
        metrics_prob = compute_probabilistic_metrics(
            true_labels[test_mask],
            prob_labels[test_mask],
            class_names,
            logger=logger
        )

        # Save metrics
        metrics_hard_path = output_dir / 'classification_metrics_hard.json'
        with open(metrics_hard_path, 'w') as f:
            json.dump(metrics_hard, f, indent=2)
        logger.info(f"Hard classification metrics saved to: {metrics_hard_path}")

        metrics_prob_path = output_dir / 'probabilistic_metrics.json'
        with open(metrics_prob_path, 'w') as f:
            json.dump(metrics_prob, f, indent=2)
        logger.info(f"Probabilistic metrics saved to: {metrics_prob_path}")

        # Create visualizations
        logger.info("\nCreating visualizations...")
        create_visualizations(
            output_dir,
            embeddings_pca,
            true_labels,
            final_predictions,
            class_names,
            metrics_hard,
            train_mask=train_mask,
            test_mask=test_mask,
            prob_metrics=metrics_prob,
            logger=logger
        )

        # Create detailed probability analysis figure
        create_probability_analysis_figure(
            output_dir,
            true_labels,
            prob_labels,
            class_names,
            test_mask=test_mask,
            logger=logger
        )

        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("RECOMPUTATION RESULTS (TEST SET)")
        logger.info("=" * 70)
        logger.info(f"\nData split:")
        logger.info(f"  Training set: {train_mask.sum()} samples ({train_mask.sum()/len(true_labels)*100:.1f}%)")
        logger.info(f"  Test set: {test_mask.sum()} samples ({test_mask.sum()/len(true_labels)*100:.1f}%)")
        logger.info(f"\nIterative label propagation:")
        logger.info(f"  Initial labels: {stats['n_human']}")
        logger.info(f"  Pseudo-labels added: {stats['n_added_iteration']}")
        logger.info(f"  Total labels after iteration: {stats['n_total_after_iteration']}")
        logger.info(f"  Pseudo-labels in test set: {pseudo_in_test.sum()} ({pseudo_in_test.sum()/test_mask.sum()*100:.1f}% of test)")
        logger.info(f"\nHard Classification (argmax):")
        logger.info(f"  Overall Purity:       {metrics_hard['overall_purity']:.4f}")
        logger.info(f"  Overall Completeness: {metrics_hard['overall_completeness']:.4f}")
        logger.info(f"  Precision (weighted): {metrics_hard['classification_report']['weighted avg']['precision']:.4f}")
        logger.info(f"  Recall (weighted):    {metrics_hard['classification_report']['weighted avg']['recall']:.4f}")
        logger.info(f"  F1-Score (weighted):  {metrics_hard['classification_report']['weighted avg']['f1-score']:.4f}")
        logger.info(f"\nProbabilistic Evaluation:")
        logger.info(f"  Mean Confidence: {metrics_prob['overall_mean_confidence']:.4f}")
        logger.info(f"  Mean Entropy:    {metrics_prob['overall_mean_entropy']:.4f}")
        logger.info(f"  Top-1 Accuracy:  {metrics_prob['top_k_accuracy'].get('top_1', 0):.4f}")
        logger.info(f"  Top-2 Accuracy:  {metrics_prob['top_k_accuracy'].get('top_2', 0):.4f}")
        logger.info(f"\nPer-class Mean Confidence:")
        for class_name in class_names.values():
            if class_name in metrics_prob['mean_confidences']:
                conf = metrics_prob['mean_confidences'][class_name]
                ent = metrics_prob['entropy_per_class'][class_name]
                logger.info(f"  {class_name:20s}: {conf:.4f} (entropy: {ent:.4f})")
        logger.info("=" * 70)

        print("\n✅ Probability label recomputation completed successfully!")
        print(f"   Parameters: k={label_config.get('n_neighbors', 50)}, "
              f"n_min={label_config.get('minimum_labeled_neighbors', 5)}, "
              f"threshold={label_config.get('prob_threshold', 0.9)}")
        print(f"   Accuracy: {metrics_hard['classification_report']['accuracy']:.1%}")
        print(f"   Mean Confidence: {metrics_prob['overall_mean_confidence']:.4f}")
        print(f"   Results saved to: {output_dir}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n❌ Error: {e}")
        print("\nMake sure to run run_fire2_mock.py first to generate embeddings:")
        print(f"  cd {Path(__file__).parent}")
        print(f"  python run_fire2_mock.py --tags m11h_res7100 m11d_res7100 m11e_res7100 --n-per-galaxy 1000")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
