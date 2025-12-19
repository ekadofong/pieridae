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
    logger: logging.Logger = None,
    use_metadata: bool = False,
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
        if metadata_path.exists() and use_metadata:
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
            #logger.info(f'Loading: {image_metadata_list[:n_to_load]}')
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

    # Get unique class IDs (1-indexed, 0 is unlabeled)
    unique_labels = np.unique(true_labels[true_labels > 0])
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    # 1. PCA with ground truth labels
    for idx, label in enumerate(unique_labels):
        mask = true_labels == label
        axes[0, 0].scatter(
            embeddings_pca[mask, 0],
            embeddings_pca[mask, 1],
            c=[colors[idx]],
            label=class_names.get(label, f'Class {label}'),
            alpha=0.6,
            s=20
        )
    axes[0, 0].set_xlabel('PC1')
    axes[0, 0].set_ylabel('PC2')
    axes[0, 0].set_title('PCA: Ground Truth Labels')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. PCA with predicted labels
    for idx, label in enumerate(unique_labels):
        mask = predicted_labels == label
        axes[0, 1].scatter(
            embeddings_pca[mask, 0],
            embeddings_pca[mask, 1],
            c=[colors[idx]],
            label=class_names.get(label, f'Class {label}'),
            alpha=0.6,
            s=20
        )
    axes[0, 1].set_xlabel('PC1')
    axes[0, 1].set_ylabel('PC2')
    axes[0, 1].set_title('PCA: Predicted Labels')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Confusion matrix
    try:
        import seaborn as sns
        conf_matrix = np.array(metrics['confusion_matrix'])
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[class_names.get(i, f'Class {i}') for i in unique_labels],
            yticklabels=[class_names.get(i, f'Class {i}') for i in unique_labels],
            ax=axes[1, 0],
            cbar_kws={'label': 'Count'}
        )
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('True')
        axes[1, 0].set_title('Confusion Matrix')
    except ImportError:
        axes[1, 0].text(0.5, 0.5, 'Seaborn not available\nfor confusion matrix',
                        ha='center', va='center')
        axes[1, 0].set_title('Confusion Matrix')

    # 4. Metrics summary
    axes[1, 1].axis('off')

    # Build metrics text from class_names
    class_purities_text = "\n".join([
        f"  {class_names.get(label, f'Class {label}')}: {metrics['cluster_purities'].get(class_names.get(label, f'class_{label}'), 0.0):.3f}"
        for label in unique_labels
    ])

    class_completeness_text = "\n".join([
        f"  {class_names.get(label, f'Class {label}')}: {metrics['class_completeness'].get(class_names.get(label, f'class_{label}'), 0.0):.3f}"
        for label in unique_labels
    ])

    metrics_text = f"""
Classification Metrics Summary

Overall Purity: {metrics['overall_purity']:.3f}
Overall Completeness: {metrics['overall_completeness']:.3f}

Per-Class Purity:
{class_purities_text}

Per-Class Completeness:
{class_completeness_text}

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
    ndiscard = int(len(training_labels)*0.5)
    print(f'Discarding {ndiscard} labels to construct a 10% training set')
    training_labels[np.random.choice(np.arange(0, labels.size), replace=False, size=ndiscard)] = 0    
    # Save training labels
    results_path = output_path / 'training_data.pkl'
    with open(results_path, 'wb') as f: 
        pickle.dump({
            'training_labels': training_labels,
            'f_discard':0.5,
            'training_images':images[training_labels>0]
        }, f)
    n_classes = np.unique(labels).size
    model_manager = BYOLModelManager(config, output_path, logger, n_classes=n_classes)
    model_manager.train_model(
        images,
        training_labels,
        resume=config['training'].get('resume', False),
        patience_limit=config['training'].get('patience_limit', 20)
    )

    logger.info("Training complete")


def create_random_mock_grid(
    images: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_names: dict,
    output_path: Path,
    logger: logging.Logger = None
) -> None:
    """
    Create a grid showing correctly classified and misclassified examples.
    Left column: correctly classified case (broadband + HF)
    Right column: misclassified case (broadband + HF), or removed if no misclassifications.

    Parameters
    ----------
    images : np.ndarray
        Processed images array, shape (N, C, H, W)
    true_labels : np.ndarray
        Array of true labels for each image
    predicted_labels : np.ndarray
        Array of predicted labels for each image
    class_names : dict
        Mapping from label ID to galaxy tag
    output_path : Path
        Directory to save the figure
    logger : logging.Logger
        Logger instance
    """
    if not PLOTTING_AVAILABLE:
        if logger:
            logger.warning("Matplotlib not available, skipping mock image grid")
        return

    if logger:
        logger.info("Creating classification examples grid (correct vs misclassified)...")

    # Get unique class IDs (sorted)
    unique_labels = sorted([k for k in class_names.keys()])
    n_galaxies = len(unique_labels)

    if n_galaxies > 4:
        if logger:
            logger.warning(f"Expected up to 4 galaxies, found {n_galaxies}. Using first 4...")
        unique_labels = unique_labels[:4]
        n_galaxies = 4

    # Determine if we have any misclassifications
    has_misclassifications = np.any(true_labels != predicted_labels)

    if has_misclassifications:
        # 4 columns: correct BB, correct HF, misclassified BB, misclassified HF
        n_cols = 4
        _, axes = plt.subplots(4, n_cols, figsize=(12, 12))
        show_misclassified = True
    else:
        # 2 columns: correct BB, correct HF only
        n_cols = 2
        _, axes = plt.subplots(4, n_cols, figsize=(6, 12))
        show_misclassified = False
        if logger:
            logger.info("No misclassifications found - showing only correctly classified examples")

    # Ensure axes is 2D
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    for row_idx in range(4):
        if row_idx < n_galaxies:
            label_id = unique_labels[row_idx]
            galaxy_name = class_names[label_id]

            # Get correctly classified images for this galaxy
            correct_mask = (true_labels == label_id) & (predicted_labels == label_id)
            correct_images_idx = np.where(correct_mask)[0]

            # Get misclassified images for this galaxy (if any)
            if show_misclassified:
                misclass_mask = (true_labels == label_id) & (predicted_labels != label_id)
                misclass_images_idx = np.where(misclass_mask)[0]

            # Select correct example
            if len(correct_images_idx) > 0:
                correct_idx = np.random.choice(correct_images_idx)
                correct_img = images[correct_idx]

                # Broadband (i-band, channel 1)
                i_band = correct_img[1, :, :]
                vmin_bb, vmax_bb = np.percentile(i_band, [1, 99])

                ax_bb = axes[row_idx, 0]
                ax_bb.imshow(i_band, origin='lower', cmap='gray', vmin=vmin_bb, vmax=vmax_bb)
                ax_bb.set_xticks([])
                ax_bb.set_yticks([])

                # High-frequency (HF, channel 2)
                hf_band = correct_img[2, :, :]
                vmin_hf, vmax_hf = np.percentile(hf_band, [1, 99])

                ax_hf = axes[row_idx, 1]
                ax_hf.imshow(hf_band, origin='lower', cmap='gray', vmin=vmin_hf, vmax=vmax_hf)
                ax_hf.set_xticks([])
                ax_hf.set_yticks([])

                # Add column labels to top row
                if row_idx == 0:
                    ax_bb.set_title('Correct\nBroadband', fontsize=10, fontweight='bold', color='green')
                    ax_hf.set_title('Correct\nHF', fontsize=10, fontweight='bold', color='green')

                # Add row label to first column
                ax_bb.set_ylabel(galaxy_name, fontsize=11, fontweight='bold')
            else:
                # No correct examples
                axes[row_idx, 0].axis('off')
                axes[row_idx, 1].axis('off')
                if row_idx == 0:
                    axes[row_idx, 0].text(0.5, 0.5, 'No correct\nclassifications',
                                          ha='center', va='center', fontsize=9)

            # Select misclassified example (if applicable)
            if show_misclassified:
                if len(misclass_images_idx) > 0:
                    misclass_idx = np.random.choice(misclass_images_idx)
                    misclass_img = images[misclass_idx]
                    predicted_class = predicted_labels[misclass_idx]
                    predicted_name = class_names.get(predicted_class, f'Class {predicted_class}')

                    # Broadband (i-band, channel 1)
                    i_band = misclass_img[1, :, :]
                    vmin_bb, vmax_bb = np.percentile(i_band, [1, 99])

                    ax_bb = axes[row_idx, 2]
                    ax_bb.imshow(i_band, origin='lower', cmap='gray', vmin=vmin_bb, vmax=vmax_bb)
                    ax_bb.set_xticks([])
                    ax_bb.set_yticks([])

                    # High-frequency (HF, channel 2)
                    hf_band = misclass_img[2, :, :]
                    vmin_hf, vmax_hf = np.percentile(hf_band, [1, 99])

                    ax_hf = axes[row_idx, 3]
                    ax_hf.imshow(hf_band, origin='lower', cmap='gray', vmin=vmin_hf, vmax=vmax_hf)
                    ax_hf.set_xticks([])
                    ax_hf.set_yticks([])

                    # Add column labels to top row
                    if row_idx == 0:
                        ax_bb.set_title('Misclassified\nBroadband', fontsize=10, fontweight='bold', color='red')
                        ax_hf.set_title('Misclassified\nHF', fontsize=10, fontweight='bold', color='red')

                    # Add predicted class label
                    ax_bb.text(0.02, 0.98, f'→{predicted_name}',
                              transform=ax_bb.transAxes, fontsize=8,
                              va='top', ha='left', color='red', fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                else:
                    # No misclassifications for this galaxy
                    axes[row_idx, 2].axis('off')
                    axes[row_idx, 3].axis('off')
                    axes[row_idx, 2].text(0.5, 0.5, 'No\nmisclassifications',
                                          ha='center', va='center', fontsize=9, color='green')
        else:
            # Empty row if fewer than 4 galaxies
            for col_idx in range(n_cols):
                axes[row_idx, col_idx].axis('off')

    if show_misclassified:
        title = 'Classification Examples: Correctly Classified vs Misclassified'
    else:
        title = 'Classification Examples: All Correctly Classified'

    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save figure
    output_file = output_path / 'classification_examples_grid.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"Classification examples grid saved to: {output_file}")


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
    #logger.info(f'[classes: {np.unique(true_labels)}]')

    # Extract embeddings
    n_classes = np.unique(true_labels).size 
    model_manager = BYOLModelManager(config, output_path, logger, n_classes=n_classes)
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
    logger.info(f"Loading trained classifier for predictions...")

    classifier = FrozenClassifier(
        model_path=output_path / 'model_checkpoint.pt',
        config=config,
        logger=logger,
        n_classes=n_classes     
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

    # Create classification examples grid (correct vs misclassified)
    create_random_mock_grid(
        images,
        true_labels,
        predicted_labels,
        class_names,
        output_path,
        logger=logger
    )

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
    for label_id in sorted([k for k in class_names.keys()]):
        class_name = class_names[label_id]
        logger.info(
            f"  {class_name:20s} - "
            f"Purity: {metrics['cluster_purities'].get(class_name, 0.0):.4f}, "
            f"Completeness: {metrics['class_completeness'].get(class_name, 0.0):.4f}"
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
        config['data']['output_path'] = Path(args.output_path).parent
    else:
        config['data']['output_path'] = Path(config['data']['output_path']).parent / 'fire2_sfourl'

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
            run_analysis(config, images_processed, img_names, true_labels, class_names, output_path, logger)
        elif args.mode == 'full':
            run_full_pipeline(config, images_processed, img_names, true_labels, class_names, output_path, logger)

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
