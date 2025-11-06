#!/usr/bin/env python3
"""
BYOL Merger Analysis - Main Script

Refactored version using pieridae.starbursts.byol module.
Performs BYOL training, embedding extraction, PCA reduction, and visualization.

Usage
-----
# Full pipeline
python run_analysis.py --mode full

# Training only
python run_analysis.py --mode train --epochs 500

# Analysis only (requires trained model)
python run_analysis.py --mode analyze

# Custom config
python run_analysis.py --config custom_config.yaml --mode full
"""

import os
import sys
import argparse
import logging
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
    load_merian_images
)

# Plotting
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
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
    logger = logging.getLogger('byol_analysis')
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
    labels: np.ndarray = None,
    logger: logging.Logger = None
) -> None:
    """Create PCA and UMAP visualizations"""
    if not PLOTTING_AVAILABLE:
        if logger:
            logger.warning("Matplotlib not available, skipping visualizations")
        return

    if logger:
        logger.info("Creating visualizations...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # PCA plot
    if labels is not None:
        # Color by labels
        unique_labels = np.unique(labels[labels > 0])
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for idx, label in enumerate(unique_labels):
            mask = labels == label
            axes[0].scatter(
                embeddings_pca[mask, 0],
                embeddings_pca[mask, 1],
                c=[colors[idx]],
                label=f'Class {label}',
                alpha=0.6,
                s=20
            )

        # Plot unlabeled
        mask = labels == 0
        if mask.any():
            axes[0].scatter(
                embeddings_pca[mask, 0],
                embeddings_pca[mask, 1],
                c='lightgray',
                label='Unlabeled',
                alpha=0.3,
                s=10
            )

        axes[0].legend()
    else:
        axes[0].scatter(
            embeddings_pca[:, 0],
            embeddings_pca[:, 1],
            alpha=0.5,
            s=10
        )

    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].set_title('PCA Embeddings')
    axes[0].grid(True, alpha=0.3)

    # UMAP plot
    if labels is not None:
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            axes[1].scatter(
                embeddings_umap[mask, 0],
                embeddings_umap[mask, 1],
                c=[colors[idx]],
                label=f'Class {label}',
                alpha=0.6,
                s=20
            )

        mask = labels == 0
        if mask.any():
            axes[1].scatter(
                embeddings_umap[mask, 0],
                embeddings_umap[mask, 1],
                c='lightgray',
                label='Unlabeled',
                alpha=0.3,
                s=1,
                zorder=0
            )

        axes[1].legend()
    else:
        axes[1].scatter(
            embeddings_umap[:, 0],
            embeddings_umap[:, 1],
            alpha=0.5,
            s=10
        )

    axes[1].set_xlabel('UMAP1')
    axes[1].set_ylabel('UMAP2')
    axes[1].set_title('UMAP Embeddings')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'embeddings_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info("Visualizations saved")


def load_labels(config: dict, img_names: np.ndarray, logger: logging.Logger = None) -> np.ndarray:
    """Load classification labels if available"""
    import pandas as pd

    labels = None
    label_file = Path(config.get('labels', {}).get('classifications_file', ''))

    if label_file.exists() and img_names is not None:
        try:
            mergers = pd.read_csv(label_file, index_col=0)
            labels = mergers.reindex(img_names)
            labels = labels.replace(np.nan, 0).values.flatten().astype(int)

            if logger:
                logger.info(f"Loaded classification labels: {len(labels)} objects")

                # Print label distribution
                unique, counts = np.unique(labels, return_counts=True)
                label_meanings = config.get('labels', {}).get('label_mapping', {})

                logger.info("Label distribution:")
                for label_val, count in zip(unique, counts):
                    meaning = label_meanings.get(label_val, f"unknown_{label_val}")
                    logger.info(f"   {label_val} ({meaning}): {count} objects")

        except Exception as e:
            if logger:
                logger.warning(f"Could not load labels: {e}")
            labels = None
    else:
        if logger:
            logger.info(f"Label file not found: {label_file}")
        labels = None

    return labels


def run_training(
    config: dict,
    images: np.ndarray,
    output_path: Path,
    logger: logging.Logger
) -> None:
    """Run BYOL training"""
    logger.info("=" * 60)
    logger.info("TRAINING MODE")
    logger.info("=" * 60)

    model_manager = BYOLModelManager(config, output_path, logger)
    model_manager.train_model(images, resume=config['training'].get('resume', False))

    logger.info("Training complete")


def run_analysis(
    config: dict,
    images: np.ndarray,
    img_names: np.ndarray,
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

    # Save results
    results_path = output_path / 'dimensionality_reduction_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump({
            'embeddings_original': embeddings,
            'embeddings_pca': embeddings_pca,
            'embeddings_umap': embeddings_umap,
            'img_names': img_names,
            'scaler': analyzer.scaler,
            'pca': analyzer.pca,
            'umap': analyzer.umap_reducer
        }, f)

    logger.info(f"Results saved to: {results_path}")

    # Load labels and create visualizations
    labels = load_labels(config, img_names, logger)
    create_visualizations(output_path, embeddings_pca, embeddings_umap, labels, logger)

    logger.info("Analysis complete")


def run_full_pipeline(
    config: dict,
    images: np.ndarray,
    img_names: np.ndarray,
    output_path: Path,
    logger: logging.Logger
) -> None:
    """Run complete pipeline: training + analysis"""
    logger.info("=" * 60)
    logger.info("FULL PIPELINE MODE")
    logger.info("=" * 60)

    # Training
    run_training(config, images, output_path, logger)

    # Analysis
    run_analysis(config, images, img_names, output_path, logger)

    logger.info("Full pipeline complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='BYOL Analysis for Galaxy Merger Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with default config
  python run_analysis.py --mode full

  # Train model only
  python run_analysis.py --mode train --epochs 500

  # Analyze with existing model
  python run_analysis.py --mode analyze

  # Use custom config
  python run_analysis.py --config my_config.yaml --mode full
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
        '--data-path',
        type=str,
        help='Override input data path from config'
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
    if args.data_path:
        config['data']['input_path'] = Path(args.data_path)
    if args.output_path:
        config['data']['output_path'] = Path(args.output_path)
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

    logger.info(f"Starting BYOL analysis in {args.mode} mode")
    logger.info(f"Input path: {config['data']['input_path']}")
    logger.info(f"Output path: {config['data']['output_path']}")

    try:
        # Load images
        logger.info("Loading images...")
        images, img_names = load_merian_images(
            config['data']['input_path'],
            logger,
        )

        # Run requested mode
        if args.mode == 'train':
            run_training(config, images, output_path, logger)
        elif args.mode == 'analyze':
            run_analysis(config, images, img_names, output_path, logger)
        elif args.mode == 'full':
            run_full_pipeline(config, images, img_names, output_path, logger)

        logger.info("=" * 60)
        logger.info("SUCCESS")
        logger.info("=" * 60)
        print("\n Analysis completed successfully!")

    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        print(f"\n Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
