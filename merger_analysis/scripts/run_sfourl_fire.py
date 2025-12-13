#!/usr/bin/env python3
"""
BYOL Merger Analysis - FIRE2 Mock Images

Refactored version using pieridae.starbursts.byol module with FIRE2 simulation images.
Uses the same framework as run_analysis.py but with FIRE2 mock galaxy images.

Usage
-----
# Full pipeline
python run_sfourl_fire.py --mode full --tags m11h_res7100 m11d_res7100 m11e_res7100

# Training only
python run_sfourl_fire.py --mode train --epochs 500

# Analysis only (requires trained model)
python run_sfourl_fire.py --mode analyze

# Custom config
python run_sfourl_fire.py --config custom_config.yaml --mode full
"""

import os
import sys
import argparse
import logging
import pickle
import glob
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

import yaml
import numpy as np
from scipy import ndimage
from tqdm import tqdm

# Add pieridae to path (go up 2 levels: scripts/ -> merger_analysis/ -> pieridae/)
sys.path.insert(0, str(Path(__file__).parents[2]))

from pieridae.starbursts.byol import (
    BYOLModelManager,
    EmbeddingAnalyzer,
    FrozenClassifier,
    compute_classification_metrics
)

# Starlet transform for high-frequency images
try:
    from ekfstats import imstats
    import sep
    STARLET_AVAILABLE = True
except ImportError:
    STARLET_AVAILABLE = False
    print("Warning: ekfstats not available, using simplified HF image generation")

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
    logger = logging.getLogger('fire2_analysis')
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


def load_fire2_mock_images(
    mock_images_dir: Path,
    galaxy_tags: List[str],
    n_per_galaxy: int = None,
    logger: logging.Logger = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Load FIRE2 mock images from generated datasets.

    Parameters
    ----------
    mock_images_dir : Path
        Base directory containing mock_images/{tag}/ subdirectories
    galaxy_tags : List[str]
        List of galaxy tags to load (e.g., ['m11b_res2100', 'm11d_res7100'])
    n_per_galaxy : int, optional
        Number of images to load per galaxy. If None, loads all available.
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    images : np.ndarray
        Array of images (N, H, W)
    img_names : np.ndarray
        Array of image names
    true_labels : np.ndarray
        Array of labels (1, 2, 3, ... for each galaxy tag, 1-indexed)
    class_names : dict
        Mapping from label ID to galaxy tag
    """
    if logger:
        logger.info(f"Loading FIRE2 mock images from: {mock_images_dir}")
        logger.info(f"Galaxy tags: {galaxy_tags}")

    images = []
    img_names = []
    true_labels = []
    class_names = {}

    # Use 1-indexed labels (0 will be unlabeled)
    for class_id, tag in enumerate(galaxy_tags, start=1):
        class_names[class_id] = tag
        galaxy_dir = mock_images_dir / tag

        if not galaxy_dir.exists():
            raise FileNotFoundError(
                f"Galaxy directory not found: {galaxy_dir}\n"
                f"Please generate images first using:\n"
                f"  python generate_fire2_images.py --tag {tag} --n-images {n_per_galaxy or 100}"
            )

        # Load metadata
        metadata_path = galaxy_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            image_metadata_list = metadata.get('images', [])
            n_available = len(image_metadata_list)
        else:
            # Fallback: count .npy files
            image_files = sorted(galaxy_dir.glob('*.npy'))
            n_available = len(image_files)
            image_metadata_list = [{'filename': f.stem} for f in image_files]

        if n_per_galaxy and n_available < n_per_galaxy:
            raise IOError(f"Requested {n_per_galaxy} mock images but only {n_available} are available!")

        n_to_load = n_per_galaxy if n_per_galaxy is not None else n_available

        if logger:
            logger.info(f"Loading {n_to_load} images from {tag}...")

        # Load images
        loaded_count = 0
        for img_meta in image_metadata_list[:n_to_load]:
            filename = img_meta.get('filename', f"image_{loaded_count:04d}")
            img_path = galaxy_dir / f"{filename}.npy"

            if not img_path.exists():
                if logger:
                    logger.warning(f"Image not found: {img_path}, skipping...")
                continue

            img = np.load(img_path)
            images.append(img)
            img_names.append(f"{tag}_{filename}")
            true_labels.append(class_id)
            loaded_count += 1

        if logger:
            logger.info(f"  Loaded {loaded_count} images from {tag}")

    # Convert to arrays
    images = np.array(images)
    img_names = np.array(img_names)
    true_labels = np.array(true_labels)

    if logger:
        logger.info(f"Loaded {len(images)} total images")
        logger.info(f"Image shape: {images.shape}")
        logger.info(f"Class distribution: {np.bincount(true_labels)}")
        logger.info(f"Class names: {class_names}")

    return images, img_names, true_labels, class_names


def create_hf_image(i_band: np.ndarray, logger: logging.Logger = None) -> np.ndarray:
    """
    Create high-frequency residual image using starlet decomposition.

    Parameters
    ----------
    i_band : np.ndarray
        Input i-band image (2D array)
    logger : logging.Logger
        Logger instance

    Returns
    -------
    hf_image : np.ndarray
        High-frequency residual image
    """
    if STARLET_AVAILABLE:
        try:
            # Apply starlet wavelet transform
            wt = imstats.starlet_transform(i_band, gen2=True)

            segmap_l = []
            im_recon = []

            for ix in range(len(wt)):
                # Estimate noise from corners
                err_samples = [
                    np.std(abs(wt[ix])[:25, -25:]),
                    np.std(abs(wt[ix])[-25:, -25:]),
                    np.std(abs(wt[ix])[:25, :25]),
                    np.std(abs(wt[ix])[-25:, :25])
                ]

                # Extract features in this wavelet scale
                _, segmap = sep.extract(
                    abs(wt[ix]),
                    10.,
                    err=np.median(err_samples),
                    segmentation_map=True,
                    deblend_cont=1.
                )

                # Keep only central source features
                sidx = segmap[segmap.shape[0]//2, segmap.shape[0]//2]
                segmap_l.append(segmap)
                im_recon.append(np.where(segmap == sidx, wt[ix], 0.))

            # Reconstruct image and create high-frequency residual
            im_recon = imstats.inverse_starlet_transform(im_recon, gen2=True)
            hf_image = i_band - im_recon
            hf_image = hf_image - ndimage.median_filter(hf_image, size=20)

            return hf_image

        except Exception as e:
            if logger:
                logger.warning(f"Starlet transform failed: {e}, using fallback method")
            return create_hf_image_fallback(i_band)
    else:
        return create_hf_image_fallback(i_band)


def create_hf_image_fallback(i_band: np.ndarray) -> np.ndarray:
    """
    Fallback method for creating high-frequency image.
    Uses simple high-pass filtering if starlet transform not available.
    """
    from scipy.ndimage import gaussian_filter

    # Apply Gaussian smoothing and subtract to get high-frequency
    smoothed = gaussian_filter(i_band, sigma=5.0)
    hf_image = i_band - smoothed

    # Subtract median filter
    hf_image = hf_image - ndimage.median_filter(hf_image, size=20)

    return hf_image


def preprocess_images_for_byol(
    images: np.ndarray,
    target_channels: int = 3,
    logger: logging.Logger = None
) -> np.ndarray:
    """
    Preprocess FIRE2 images for BYOL.

    FIRE2 images are 2D histograms (H, W). Convert to (N, C, H, W)
    format where C=3 for BYOL with [g, i, hf] channels.

    Parameters
    ----------
    images : np.ndarray
        Input images, shape (N, H, W)
    target_channels : int
        Target number of channels (default: 3)
    logger : logging.Logger
        Logger instance

    Returns
    -------
    processed_images : np.ndarray
        Processed images, shape (N, 3, H, W) with [g, i, hf] channels
    """
    if logger:
        logger.info(f"Preprocessing images: {images.shape}")
        logger.info("Generating high-frequency images using starlet decomposition...")

    if images.ndim != 3:
        raise ValueError(f"Expected 3D input (N, H, W), got shape: {images.shape}")

    n_images = images.shape[0]

    # Create 3-channel output
    images_3ch = np.zeros((n_images, 3, images.shape[1], images.shape[2]), dtype=np.float32)

    # Process each image to create [g, i, hf] channels
    for i in tqdm(range(n_images), desc="Creating HF images"):
        i_band = images[i]

        # Channel 0: g-band (simulate as 0.9 * i-band)
        g_band = i_band * 0.9

        # Channel 1: i-band (original)
        # (keep as is)

        # Channel 2: High-frequency residual
        hf_band = create_hf_image(i_band, logger=logger)

        images_3ch[i, 0] = g_band
        images_3ch[i, 1] = i_band
        images_3ch[i, 2] = hf_band

    if logger:
        logger.info(f"Processed images shape: {images_3ch.shape}")
        logger.info(f"  Channel 0 (g): simulated g-band")
        logger.info(f"  Channel 1 (i): stellar density histogram")
        logger.info(f"  Channel 2 (hf): high-frequency residual")

    return images_3ch


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
        # Color by labels (1-indexed, 0 is unlabeled)
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

    model_manager = BYOLModelManager(config, output_path, logger)
    model_manager.train_model(
        images,
        labels,
        resume=config['training'].get('resume', False),
        patience_limit=config['training'].get('patience_limit', 20)
    )

    logger.info("Training complete")


def run_analysis(
    config: dict,
    images: np.ndarray,
    img_names: np.ndarray,
    labels: np.ndarray,
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

    # Use FrozenClassifier for predictions
    logger.info("Loading trained classifier for predictions...")
    classifier = FrozenClassifier(
        model_path=output_path / 'model_checkpoint.pt',
        config=config,
        logger=logger
    )

    # Get probabilistic predictions
    prob_labels = classifier.predict(images)
    predicted_labels = np.argmax(prob_labels, axis=1)

    # Save predictions
    predictions_path = output_path / 'predictions.pkl'
    with open(predictions_path, 'wb') as f:
        pickle.dump({
            'prob_labels': prob_labels,
            'predicted_labels': predicted_labels,
            'true_labels': labels,
            'img_names': img_names
        }, f)

    logger.info(f"Predictions saved to: {predictions_path}")

    # Create visualizations
    create_visualizations(output_path, embeddings_pca, embeddings_umap, labels, logger)

    logger.info("Analysis complete")


def run_full_pipeline(
    config: dict,
    images: np.ndarray,
    img_names: np.ndarray,
    labels: np.ndarray,
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
    run_analysis(config, images, img_names, labels, output_path, logger)

    logger.info("Full pipeline complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='BYOL Analysis for FIRE2 Mock Images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with default config
  python run_sfourl_fire.py --mode full --tags m11h_res7100 m11d_res7100

  # Train model only
  python run_sfourl_fire.py --mode train --epochs 500

  # Analyze with existing model
  python run_sfourl_fire.py --mode analyze

  # Use custom config
  python run_sfourl_fire.py --config my_config.yaml --mode full
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
        '--tags',
        type=str,
        nargs='+',
        default=['m11h_res7100', 'm11d_res7100', 'm11e_res7100'],
        help='Galaxy tags to use'
    )
    parser.add_argument(
        '--n-per-galaxy',
        type=int,
        default=3000,
        help='Number of images per galaxy'
    )
    parser.add_argument(
        '--mock-images-dir',
        type=str,
        default='../local_data/mock_images',
        help='Directory containing mock images'
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
        config['data']['output_path'] = Path(args.output_path)
    else:
        config['data']['output_path'] = Path(config['data']['output_path']) / 'fire2_sfourl'

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

    logger.info(f"Starting FIRE2 BYOL analysis in {args.mode} mode")
    logger.info(f"Output path: {config['data']['output_path']}")
    logger.info(f"Galaxy tags: {args.tags}")
    logger.info(f"Images per galaxy: {args.n_per_galaxy}")

    try:
        # Load FIRE2 mock images
        logger.info("Loading FIRE2 mock images...")
        mock_images_dir = Path(args.mock_images_dir)

        images, img_names, true_labels, class_names = load_fire2_mock_images(
            mock_images_dir,
            args.tags,
            args.n_per_galaxy,
            logger
        )

        # Preprocess images for BYOL (create 3-channel [g, i, hf] images)
        logger.info("Preprocessing images for BYOL...")
        images_processed = preprocess_images_for_byol(images, target_channels=3, logger=logger)

        # Run requested mode
        if args.mode == 'train':
            run_training(config, images_processed, img_names, true_labels, output_path, logger)
        elif args.mode == 'analyze':
            run_analysis(config, images_processed, img_names, true_labels, output_path, logger)
        elif args.mode == 'full':
            run_full_pipeline(config, images_processed, img_names, true_labels, output_path, logger)

        logger.info("=" * 60)
        logger.info("SUCCESS")
        logger.info("=" * 60)
        print("\n✅ FIRE2 analysis completed successfully!")

    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
