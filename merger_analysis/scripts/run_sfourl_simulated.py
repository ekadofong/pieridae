#!/usr/bin/env python3
"""
BYOL Merger Analysis - Simulated Galaxies

Refactored version using pieridae.starbursts.byol module with simulated Sersic galaxies.
Uses the same framework as run_analysis.py but with simulated galaxy images.

Usage
-----
# Full pipeline
python run_sfourl_simulated.py --mode full

# Training only
python run_sfourl_simulated.py --mode train --epochs 500

# Analysis only (requires trained model)
python run_sfourl_simulated.py --mode analyze

# Custom number per class
python run_sfourl_simulated.py --n-per-class 500 --mode full
"""

import sys
import argparse
import logging
import json
import pickle
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np

# Add pieridae to path (go up 2 levels: scripts/ -> merger_analysis/ -> pieridae/)
sys.path.insert(0, str(Path(__file__).parents[2]))

from pieridae.starbursts.byol import (
    BYOLModelManager,
    EmbeddingAnalyzer,
    FrozenClassifier,
    SimulatedGalaxyGenerator,
    compute_classification_metrics
)

# Plotting
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib not available, plotting disabled")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Convert paths to Path objects
    config['data']['input_path'] = Path(config['data']['input_path'])
    config['data']['output_path'] = Path(config['data']['output_path'])

    return config


def save_effective_config(config: dict, output_path: Path, logger: logging.Logger = None) -> None:
    """
    Save the effective configuration to output directory.

    Converts Path objects back to strings for YAML serialization.

    Parameters
    ----------
    config : dict
        Configuration dictionary with all overrides applied
    output_path : Path
        Output directory where config will be saved
    logger : logging.Logger, optional
        Logger instance
    """
    # Create a copy for serialization
    config_to_save = {}

    for key, value in config.items():
        if isinstance(value, dict):
            config_to_save[key] = {}
            for subkey, subvalue in value.items():
                # Convert Path objects to strings
                if isinstance(subvalue, Path):
                    config_to_save[key][subkey] = str(subvalue)
                else:
                    config_to_save[key][subkey] = subvalue
        elif isinstance(value, Path):
            config_to_save[key] = str(value)
        else:
            config_to_save[key] = value

    # Save to output directory
    config_file = output_path / 'effective_config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config_to_save, f, default_flow_style=False, sort_keys=False)

    if logger:
        logger.info(f"Effective configuration saved to: {config_file}")


def setup_logging(output_path: Path, level: str = 'INFO') -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger('simulated_analysis')
    logger.setLevel(getattr(logging, level))

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_file = output_path / f'analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def create_visualizations(
    output_path: Path,
    embeddings_pca: np.ndarray,
    embeddings_umap: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_names: dict,
    metrics: dict,
    logger: logging.Logger = None
) -> None:
    """Create evaluation visualizations"""
    if not PLOTTING_AVAILABLE:
        if logger:
            logger.warning("Matplotlib not available, skipping visualizations")
        return

    if logger:
        logger.info("Creating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    colors = ['tab:blue', 'tab:orange', 'tab:green']

    # 1. PCA with ground truth labels
    for class_id in range(3):
        mask = true_labels == class_id
        axes[0, 0].scatter(
            embeddings_pca[mask, 0],
            embeddings_pca[mask, 1],
            c=colors[class_id],
            label=class_names[class_id],
            alpha=0.6,
            s=20
        )
    axes[0, 0].set_xlabel('PC1')
    axes[0, 0].set_ylabel('PC2')
    axes[0, 0].set_title('PCA: Ground Truth Labels')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. PCA with predicted labels
    for class_id in range(3):
        mask = predicted_labels == class_id
        axes[0, 1].scatter(
            embeddings_pca[mask, 0],
            embeddings_pca[mask, 1],
            c=colors[class_id],
            label=class_names[class_id],
            alpha=0.6,
            s=20
        )
    axes[0, 1].set_xlabel('PC1')
    axes[0, 1].set_ylabel('PC2')
    axes[0, 1].set_title('PCA: Predicted Labels')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Confusion matrix
    conf_matrix = np.array(metrics['confusion_matrix'])
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[class_names[i] for i in range(3)],
        yticklabels=[class_names[i] for i in range(3)],
        ax=axes[1, 0],
        cbar_kws={'label': 'Count'}
    )
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('True')
    axes[1, 0].set_title('Confusion Matrix')

    # 4. Metrics summary
    axes[1, 1].axis('off')

    metrics_text = f"""
Classification Metrics Summary

Overall Purity: {metrics['overall_purity']:.3f}
Overall Completeness: {metrics['overall_completeness']:.3f}

Per-Class Purity:
  Disk: {metrics['cluster_purities']['disk']:.3f}
  Double Nuclei: {metrics['cluster_purities']['double_nuclei']:.3f}
  Dipole: {metrics['cluster_purities']['dipole']:.3f}

Per-Class Completeness:
  Disk: {metrics['class_completeness']['disk']:.3f}
  Double Nuclei: {metrics['class_completeness']['double_nuclei']:.3f}
  Dipole: {metrics['class_completeness']['dipole']:.3f}

Precision (weighted avg): {metrics['classification_report']['weighted avg']['precision']:.3f}
Recall (weighted avg): {metrics['classification_report']['weighted avg']['recall']:.3f}
F1-Score (weighted avg): {metrics['classification_report']['weighted avg']['f1-score']:.3f}
    """

    axes[1, 1].text(
        0.1, 0.5, metrics_text,
        fontsize=11,
        family='monospace',
        verticalalignment='center'
    )

    plt.tight_layout()
    plt.savefig(output_path / 'evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info("Visualizations saved")


def run_training(
    config: dict,
    images: np.ndarray,
    img_names: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    logger: logging.Logger
) -> None:
    """Run BYOL training"""
    logger.info("=" * 60)
    logger.info("TRAINING MODE")
    logger.info("=" * 60)

    training_labels = labels.copy()
    ndiscard = int(len(training_labels)*0.9)
    print(f'Discarding {ndiscard} labels to construct a 10% training set')
    training_labels[np.random.choice(np.arange(0, labels.size), replace=False, size=ndiscard)] = 0    
    #print(training_labels)
    #print(labels)
    logger.info(f'trimmed training label set: {int((labels>0).sum())} -> {int((training_labels>0).sum())}')

    model_manager = BYOLModelManager(config, output_path, logger)
    model_manager.train_model(
        images,
        training_labels,
        resume=config['training'].get('resume', False),
        patience_limit=config['training'].get('patience_limit', 20)
    )

    logger.info("Training complete")


def run_analysis(
    config: dict,
    images: np.ndarray,
    img_names: np.ndarray,
    true_labels: np.ndarray,
    class_names: dict,
    output_path: Path,
    logger: logging.Logger
) -> None:
    """Run full analysis pipeline"""
    logger.info("=" * 60)
    logger.info("ANALYSIS MODE")
    logger.info("=" * 60)

    # Extract embeddings
    model_manager = BYOLModelManager(config, output_path, logger)
    embeddings = model_manager.extract_embeddings(images)

    # PCA and UMAP
    analyzer = EmbeddingAnalyzer(config, logger)
    embeddings_pca = analyzer.compute_pca(embeddings)
    embeddings_umap = analyzer.compute_umap(embeddings_pca)

    # Save dimensionality reduction results
    results_path = output_path / 'dimensionality_reduction_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump({
            'embeddings_original': embeddings,
            'embeddings_pca': embeddings_pca,
            'embeddings_umap': embeddings_umap,
            'img_names': img_names,
            'true_labels': true_labels,
            'scaler': analyzer.scaler,
            'pca': analyzer.pca,
            'umap': analyzer.umap_reducer
        }, f)

    logger.info(f"Dimensionality reduction results saved to: {results_path}")

    # Use FrozenClassifier for predictions
    logger.info("Loading trained classifier for predictions...")
    classifier = FrozenClassifier(
        model_path=output_path / 'model_checkpoint.pt',
        config=config,
        logger=logger
    )

    # Get probabilistic predictions
    logger.info("Running classification...")
    iterative_labels, n_labels_iter, prob_labels_iter, stats = \
        classifier.iterative_propagation(embeddings, true_labels)
    predicted_labels = np.argmax(prob_labels_iter, axis=1)
    


    # Compute metrics
    logger.info("Computing classification metrics...")
    metrics = compute_classification_metrics(
        true_labels,
        predicted_labels,
        class_names
    )

    # Save metrics
    metrics_path = output_path / 'classification_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved to: {metrics_path}")

    # Save predictions
    predictions_path = output_path / 'predictions.pkl'
    with open(predictions_path, 'wb') as f:
        pickle.dump({
            'prob_labels': prob_labels_iter,
            'predicted_labels': predicted_labels,
            'true_labels': true_labels,
            'img_names': img_names
        }, f)

    logger.info(f"Predictions saved to: {predictions_path}")

    # Create visualizations
    create_visualizations(
        output_path,
        embeddings_pca,
        embeddings_umap,
        true_labels,
        predicted_labels,
        class_names,
        metrics,
        logger
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("CLASSIFICATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Overall Purity:       {metrics['overall_purity']:.4f}")
    logger.info(f"Overall Completeness: {metrics['overall_completeness']:.4f}")
    logger.info("\nPer-class metrics:")
    for class_name in ['disk', 'double_nuclei', 'dipole']:
        logger.info(
            f"  {class_name:20s} - "
            f"Purity: {metrics['cluster_purities'][class_name]:.4f}, "
            f"Completeness: {metrics['class_completeness'][class_name]:.4f}"
        )
    logger.info("=" * 60)

    logger.info("Analysis complete")


def run_full_pipeline(
    config: dict,
    images: np.ndarray,
    img_names: np.ndarray,
    labels: np.ndarray,
    class_names: dict,
    output_path: Path,
    logger: logging.Logger
) -> None:
    """Run complete pipeline: training + analysis"""
    logger.info("=" * 60)
    logger.info("FULL PIPELINE MODE")
    logger.info("=" * 60)

    # Training
    run_training(config, images, img_names, labels, output_path, logger)

    # Analysis
    run_analysis(config, images, img_names, labels, class_names, output_path, logger)

    logger.info("Full pipeline complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='BYOL Analysis for Simulated Galaxies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script:
1. Generates simulated galaxies (disk, double nuclei, dipole)
2. Trains BYOL model with semi-supervised classification head
3. Extracts embeddings and applies PCA/UMAP
4. Uses FrozenClassifier for predictions
5. Evaluates classification performance

Examples:
  # Run full pipeline
  python run_sfourl_simulated.py --mode full

  # Train only with custom epochs
  python run_sfourl_simulated.py --mode train --epochs 500

  # Analyze with existing model
  python run_sfourl_simulated.py --mode analyze

  # Custom number per class
  python run_sfourl_simulated.py --n-per-class 500 --mode full
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='../config.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'analyze', 'full'],
        default='full',
        help='Analysis mode: train, analyze, or full pipeline'
    )
    parser.add_argument(
        '--n-per-class',
        type=int,
        default=300,
        help='Number of galaxies per class (default: 300)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        help='Override output path from config'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='Override number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Override training batch size'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from checkpoint'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Configuration loaded from: {args.config}")

    # Override config with command line arguments
    if args.output_path:
        config['data']['output_path'] = Path(args.output_path).parent
    else:
        config['data']['output_path'] = Path(config['data']['output_path']).parent / 'simulated_sfourl'

    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.resume:
        config['training']['resume'] = args.resume

    # Create output directory
    output_path = config['data']['output_path']
    output_path.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(
        output_path,
        config.get('logging', {}).get('level', 'INFO')
    )

    # Save effective configuration with all command-line overrides
    save_effective_config(config, output_path, logger)

    logger.info(f"Starting simulated galaxy BYOL analysis in {args.mode} mode")
    logger.info(f"Output path: {config['data']['output_path']}")
    logger.info(f"Galaxies per class: {args.n_per_class}")

    try:
        # Generate simulated galaxies
        logger.info("Generating simulated galaxies...")
        generator = SimulatedGalaxyGenerator(
            image_size=config['model']['image_size'],
            random_seed=args.random_seed,
            logger=logger
        )

        images, img_names, true_labels = generator.generate_dataset(
            n_per_class=args.n_per_class
        )

        class_names = generator.class_names

        # Convert labels to 1-indexed (0 = unlabeled, 1/2/3 = classes)
        # True labels are already 0-indexed from generator, add 1
        labels_1indexed = true_labels + 1

        # Save simulated data
        sim_data_path = output_path / 'simulated_galaxies.pkl'
        with open(sim_data_path, 'wb') as f:
            pickle.dump({
                'images': images,
                'img_names': img_names,
                'true_labels': true_labels,
                'class_names': class_names,
                'n_per_class': args.n_per_class,
                'random_seed': args.random_seed
            }, f)

        logger.info(f"Simulated data saved to: {sim_data_path}")

        # Run requested mode
        if args.mode == 'train':
            run_training(config, images, img_names, labels_1indexed, output_path, logger)
        elif args.mode == 'analyze':
            run_analysis(config, images, img_names, labels_1indexed, class_names, output_path, logger)
        elif args.mode == 'full':
            run_full_pipeline(config, images, img_names, labels_1indexed, class_names, output_path, logger)

        logger.info("=" * 60)
        logger.info("SUCCESS")
        logger.info("=" * 60)
        print("\n✅ Simulated analysis completed successfully!")

    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
