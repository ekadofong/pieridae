#!/usr/bin/env python3
"""
BYOL Merger Analysis - Broadband-Only Version

This is a modified version of run_analysis.py that uses ONLY broadband images
(g-band and i-band) WITHOUT the high-frequency (HF) decomposition.

Key Difference from run_analysis.py:
- Image channels: [g-band, i-band, i-band] instead of [g-band, i-band, hf_i-band]
- The third channel is a duplicate of i-band rather than starlet HF decomposition
- All other analysis steps (BYOL training, PCA, UMAP, etc.) are identical

This allows direct comparison of classification performance with and without
high-frequency structural features.

Usage
-----
# Full pipeline
python run_analysis_bbonly.py --mode full

# Training only
python run_analysis_bbonly.py --mode train --epochs 500

# Analysis only (requires trained model)
python run_analysis_bbonly.py --mode analyze

# Custom config
python run_analysis_bbonly.py --config custom_config.yaml --mode full
"""

import os
import sys
import argparse
import logging
import pickle
import glob
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Union

import yaml
import numpy as np
from tqdm import tqdm

# Add pieridae to path (go up 2 levels: scripts/ -> merger_analysis/ -> pieridae/)
sys.path.insert(0, str(Path(__file__).parents[2]))

from pieridae.starbursts.byol import (
    BYOLModelManager,
    EmbeddingAnalyzer
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


def load_merian_images_bbonly(
    data_path: Union[str, Path],
    logger: Optional[logging.Logger] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Merian galaxy images from pickle files - BROADBAND ONLY VERSION.

    Loads g-band and i-band images. The third channel is i-band (duplicate)
    instead of high-frequency decomposition.

    Parameters
    ----------
    data_path : str or Path
        Path to directory containing M*/\*_i_results.pkl files
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    images : np.ndarray
        Loaded images with shape (N, 3, H, W)
        Channel order: [g-band, i-band, i-band]
    img_names : np.ndarray
        Object identifiers
    """
    if logger is None:
        logger = logging.getLogger('load_merian_images_bbonly')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        # Only add handler if none exist to avoid duplicates
        if not logger.handlers:
            logger.addHandler(handler)

    data_path = Path(data_path)
    pattern = f"{data_path}/M*/*i_results.pkl"
    filenames = glob.glob(pattern)

    if not filenames:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")

    logger.info(f"Found {len(filenames)} image files")
    logger.info("MODE: Broadband-only (g, i, i) - NO high-frequency features")

    # First pass: count valid images and get shape
    logger.info("Counting valid images...")
    valid_files = []
    img_shape = None

    for fname in tqdm(filenames, desc="Validating files"):
        g_file = fname.replace('_i_', '_g_')
        i_file = fname

        if os.path.exists(g_file) and os.path.exists(i_file):
            if img_shape is None:
                with open(i_file, 'rb') as f:
                    xf = pickle.load(f)
                    img_shape = xf['image'].shape
            valid_files.append(fname)

    n_images = len(valid_files)
    logger.info(f"Found {n_images} valid image sets")

    if n_images == 0:
        raise ValueError("No valid image files found")

    # Pre-allocate arrays
    images = np.zeros((n_images, 3, img_shape[0], img_shape[1]), dtype=np.float32)
    img_names = []

    # Second pass: load images
    idx = 0
    for fname in tqdm(valid_files, desc="Loading images"):
        img = []
        for band in 'gi':
            current_filename = fname.replace('_i_', f'_{band}_')

            try:
                with open(current_filename, 'rb') as f:
                    xf = pickle.load(f)
                    img.append(xf['image'])
                    if band == 'i':
                        # KEY DIFFERENCE: Use i-band again instead of hf_image
                        img.append(xf['image'])  # i-band duplicate (broadband only)
                    del xf
            except FileNotFoundError:
                logger.warning(f"File not found: {current_filename}")
                continue

        if len(img) == 3:
            images[idx] = np.array(img)
            img_names.append(Path(fname).parent.name)
            idx += 1
            del img

    # Trim if some failed
    if idx < n_images:
        images = images[:idx]
        logger.warning(f"Loaded {idx} images (expected {n_images})")

    img_names = np.array(img_names)
    logger.info(f"Loaded {len(images)} images with shape: {images.shape}")
    logger.info(f"Image channels: [g-band, i-band, i-band (duplicate)]")

    return images, img_names


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Convert paths to Path objects
    config['data']['input_path'] = Path(config['data']['input_path'])
    config['data']['output_path'] = Path(config['data']['output_path'])

    return config


def setup_logging(output_path: Path, level: str = 'INFO') -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger('byol_analysis_bbonly')
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
    log_file = output_path / f'analysis_bbonly_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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
    axes[0].set_title('PCA Embeddings (Broadband Only)')
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
                s=10
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
    axes[1].set_title('UMAP Embeddings (Broadband Only)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'embeddings_visualization_bbonly.png', dpi=300, bbox_inches='tight')
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
    logger.info("TRAINING MODE (Broadband Only)")
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
    logger.info("ANALYSIS MODE (Broadband Only)")
    logger.info("=" * 60)

    # Extract embeddings
    model_manager = BYOLModelManager(config, output_path, logger)
    embeddings = model_manager.extract_embeddings(images)

    # PCA and UMAP
    analyzer = EmbeddingAnalyzer(config, logger)
    embeddings_pca = analyzer.compute_pca(embeddings)
    embeddings_umap = analyzer.compute_umap(embeddings_pca)

    # Save results
    results_path = output_path / 'dimensionality_reduction_results_bbonly.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump({
            'embeddings_original': embeddings,
            'embeddings_pca': embeddings_pca,
            'embeddings_umap': embeddings_umap,
            'img_names': img_names,
            'scaler': analyzer.scaler,
            'pca': analyzer.pca,
            'umap': analyzer.umap_reducer,
            'mode': 'broadband_only',
            'channels': 'g, i, i (no HF)'
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
    logger.info("FULL PIPELINE MODE (Broadband Only)")
    logger.info("=" * 60)

    # Training
    run_training(config, images, output_path, logger)

    # Analysis
    run_analysis(config, images, img_names, output_path, logger)

    logger.info("Full pipeline complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='BYOL Analysis for Galaxy Merger Classification - Broadband Only',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This is the BROADBAND-ONLY version that uses [g, i, i] instead of [g, i, hf_i].

Examples:
  # Run full pipeline with default config
  python run_analysis_bbonly.py --mode full

  # Train model only
  python run_analysis_bbonly.py --mode train --epochs 500

  # Analyze with existing model
  python run_analysis_bbonly.py --mode analyze

  # Use custom config
  python run_analysis_bbonly.py --config my_config.yaml --mode full
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
    print("MODE: Broadband-only analysis (g, i, i) - NO high-frequency features")

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

    logger.info(f"Starting BYOL analysis in {args.mode} mode (BROADBAND ONLY)")
    logger.info(f"Input path: {config['data']['input_path']}")
    logger.info(f"Output path: {config['data']['output_path']}")
    logger.info("Image channels: [g-band, i-band, i-band] - NO HF decomposition")

    try:
        # Load images - using broadband-only version
        logger.info("Loading images (broadband only)...")
        images, img_names = load_merian_images_bbonly(
            config['data']['input_path'],
            logger
        )

        # Run requested mode
        if args.mode == 'train':
            run_training(config, images, output_path, logger)
        elif args.mode == 'analyze':
            run_analysis(config, images, img_names, output_path, logger)
        elif args.mode == 'full':
            run_full_pipeline(config, images, img_names, output_path, logger)

        logger.info("=" * 60)
        logger.info("SUCCESS (Broadband Only)")
        logger.info("=" * 60)
        print("\n✅ Broadband-only analysis completed successfully!")
        print("   Image channels: [g-band, i-band, i-band] (no HF)")

    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
