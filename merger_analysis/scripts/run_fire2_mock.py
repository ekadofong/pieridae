#!/usr/bin/env python3
"""
BYOL Analysis with FIRE2 Mock Images (Semi-Supervised K-NN Propagation)

Uses randomly projected FIRE2 simulation images to evaluate semi-supervised
label propagation performance. Treats a small fraction (default 10%) of ground
truth labels as "human-classified" visible labels and propagates them using K-NN
in PCA space. Ground truth labels are based on galaxy initial conditions
(e.g., m11b, m11d, m11h).

Evaluation includes:
1. Hard classification metrics (confusion matrix from argmax of probabilities)
2. Probabilistic metrics (P(predicted | true) distributions and confidence)

Usage
-----
# Run with default config (10% training labels)
python run_fire2_mock.py

# Specify galaxy tags and training fraction
python run_fire2_mock.py --tags m11b_res2100 m11d_res7100 --train-fraction 0.2

# Custom number of images per galaxy
python run_fire2_mock.py --n-per-galaxy 100

# Custom epochs and batch size
python run_fire2_mock.py --epochs 300 --batch-size 512

# Disable stratified sampling
python run_fire2_mock.py --no-stratified-split

# Custom output path
python run_fire2_mock.py --output-path ./my_fire2_results
"""

import sys
import argparse
import logging
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import yaml
import numpy as np
from scipy import ndimage

# Try to import starlet transform
try:
    from ekfstats import imstats
    import sep
    STARLET_AVAILABLE = True
except ImportError:
    STARLET_AVAILABLE = False
    print("Warning: ekfstats not available, using simplified HF image generation")

# Add pieridae to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from pieridae.starbursts.byol import (
    BYOLModelManager,
    EmbeddingAnalyzer,
    LabelPropagation,
    compute_classification_metrics
)

# Plotting
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.cluster import KMeans
    from sklearn.metrics import confusion_matrix
    from scipy.optimize import linear_sum_assignment
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging() -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger('fire2_mock_analysis')
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def load_fire2_mock_images(
    mock_images_dir: Path,
    galaxy_tags: List[str],
    n_per_galaxy: int = None,
    logger: logging.Logger = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Load FIRE2 mock images from generated datasets.

    This function loads all images for each galaxy tag across all snapshots.
    Images are grouped by tag (not by snapshot), so purity and completeness
    metrics will be calculated over all snapshots for a given tag.

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
        Array of image names (format: {tag}_{snapshot_id}_{index:04d})
    true_labels : np.ndarray
        Array of labels (0, 1, 2, ... for each galaxy tag)
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

    for class_id, tag in enumerate(galaxy_tags):
        class_names[class_id] = tag
        galaxy_dir = mock_images_dir / tag

        if not galaxy_dir.exists():
            raise FileNotFoundError(
                f"Galaxy directory not found: {galaxy_dir}\n"
                f"Please generate images first using:\n"
                f"  python generate_fire2_images.py --tag {tag} --n-images {n_per_galaxy or 100}"
            )

        # Load metadata to get list of images
        metadata_path = galaxy_dir / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Get all images from metadata (may include multiple snapshots)
        image_metadata_list = metadata.get('images', [])
        n_available = len(image_metadata_list)
        if n_available < n_per_galaxy:
            raise IOError (f"Requested {n_per_galaxy} mock images but only {n_available} are available!")
        n_to_load = n_per_galaxy if n_per_galaxy is not None else n_available

        if n_to_load > n_available:
            if logger:
                logger.warning(
                    f"Requested {n_to_load} images for {tag}, but only {n_available} available. "
                    f"Using {n_available} images."
                )
            n_to_load = n_available

        if logger:
            logger.info(f"Loading {n_to_load} images from {tag}...")

        # Load images using metadata filenames (supports new naming scheme)
        loaded_count = 0
        for img_meta in image_metadata_list[:n_to_load]:
            # Get filename from metadata (format: {tag}_{snapshot_id}_{index:04d})
            filename = img_meta.get('filename', '')
            if not filename:
                # Fallback for old metadata format
                img_idx = img_meta.get('index', loaded_count)
                filename = f"image_{img_idx:04d}"

            img_path = galaxy_dir / f"{filename}.npy"
            if not img_path.exists():
                if logger:
                    logger.warning(f"Image not found: {img_path}, skipping...")
                continue

            img = np.load(img_path)
            images.append(img)
            img_names.append(filename)
            true_labels.append(class_id)
            loaded_count += 1

        if logger:
            # Get snapshot info for logging
            snapshots_used = set()
            for img_meta in image_metadata_list[:n_to_load]:
                snap_id = img_meta.get('snapshot_id', metadata.get('snapshot_id', 'unknown'))
                snapshots_used.add(snap_id)
            logger.info(f"  Loaded {loaded_count} images from {len(snapshots_used)} snapshot(s): {sorted(snapshots_used)}")

    # Convert to arrays
    images = np.array(images)
    img_names = np.array(img_names)
    true_labels = np.array(true_labels)

    if logger:
        logger.info(f"Loaded {len(images)} total images")
        logger.info(f"Image shape: {images.shape}")
        logger.info(f"Class distribution: {np.bincount(true_labels)}")
        logger.info(f"Class names: {class_names}")
        logger.info("NOTE: Labels are by galaxy tag, so purity/completeness are calculated across all snapshots per tag")

    return images, img_names, true_labels, class_names


def create_hf_image(i_band: np.ndarray, logger: logging.Logger = None) -> np.ndarray:
    """
    Create high-frequency residual image using starlet decomposition.

    Based on byol_simulated.py's create_hf_image() method.

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

    Parameters
    ----------
    i_band : np.ndarray
        Input i-band image

    Returns
    -------
    hf_image : np.ndarray
        High-frequency residual image
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

    FIRE2 images are 2D histograms (H, W). We need to convert them to
    (N, C, H, W) format where C=3 for BYOL with [g, i, hf] channels.

    Since FIRE2 only has stellar density, we use:
    - Channel 0 (g): Original image * 0.9 (simulate g-band)
    - Channel 1 (i): Original image (i-band)
    - Channel 2 (hf): High-frequency residual from starlet decomposition

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
    from tqdm import tqdm
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


def create_sparse_labels(
    true_labels: np.ndarray,
    sample_fraction: float = 0.1,
    random_seed: int = 42,
    stratified: bool = True,
    logger: logging.Logger = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sparse labels by sampling a fraction of ground truth as "visible" training labels.

    This simulates a semi-supervised scenario where only a small fraction of data
    has human-provided labels.

    IMPORTANT: Returns 1-indexed labels to distinguish labeled class 0 from unlabeled.
    - 0 = unlabeled
    - 1, 2, 3, ... = labeled classes (true_labels + 1)

    Parameters
    ----------
    true_labels : np.ndarray
        Complete ground truth labels (0-indexed: 0, 1, 2, ...)
    sample_fraction : float
        Fraction of labels to keep as "visible" (default: 0.1 = 10%)
    random_seed : int
        Random seed for reproducibility
    stratified : bool
        If True, sample proportionally from each class to maintain class balance
    logger : logging.Logger
        Logger instance

    Returns
    -------
    sparse_labels : np.ndarray
        1-indexed labels with most set to 0 (unlabeled), only sample_fraction retained
        Values are true_labels + 1 for labeled samples, 0 for unlabeled
    train_mask : np.ndarray
        Boolean mask indicating which samples are in training set (visible)
    test_mask : np.ndarray
        Boolean mask indicating which samples are in test set (held out)
    """
    if logger:
        logger.info(f"\nCreating sparse label set ({sample_fraction*100:.1f}% visible)...")

    rng = np.random.RandomState(random_seed)
    n_samples = len(true_labels)
    train_mask = np.zeros(n_samples, dtype=bool)

    if stratified:
        # Sample proportionally from each class
        unique_labels = np.unique(true_labels)
        for label in unique_labels:
            label_indices = np.where(true_labels == label)[0]
            n_to_sample = max(1, int(len(label_indices) * sample_fraction))

            sampled_indices = rng.choice(
                label_indices,
                size=n_to_sample,
                replace=False
            )
            train_mask[sampled_indices] = True

            if logger:
                logger.info(
                    f"  Class {label}: {n_to_sample}/{len(label_indices)} samples "
                    f"({n_to_sample/len(label_indices)*100:.1f}%) used for training"
                )
    else:
        # Random sampling without stratification
        n_train = int(n_samples * sample_fraction)
        train_indices = rng.choice(n_samples, size=n_train, replace=False)
        train_mask[train_indices] = True

        if logger:
            logger.info(f"  Random sampling: {n_train}/{n_samples} samples for training")

    test_mask = ~train_mask

    # Create sparse labels (0 = unlabeled)
    # IMPORTANT: Use 1-indexed labels so class 0 doesn't look like unlabeled
    sparse_labels = np.zeros_like(true_labels)
    sparse_labels[train_mask] = true_labels[train_mask] + 1  # Make 1-indexed

    if logger:
        logger.info(f"\nLabel split summary:")
        logger.info(f"  Training samples: {train_mask.sum()} ({train_mask.sum()/n_samples*100:.1f}%)")
        logger.info(f"  Test samples: {test_mask.sum()} ({test_mask.sum()/n_samples*100:.1f}%)")
        logger.info(f"  Unlabeled in sparse set: {(sparse_labels == 0).sum()}")
        logger.info(f"  Using 1-indexed labels: 0=unlabeled, 1-{len(unique_labels)}=classes")

    return sparse_labels, train_mask, test_mask


def propagate_labels_knn(
    pca_embeddings: np.ndarray,
    sparse_labels: np.ndarray,
    n_classes: int,
    config: dict,
    logger: logging.Logger = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Propagate labels using iterative K-NN in PCA space.

    Uses two-stage iterative propagation (matching merger_classification.ipynb):
    1. Initial label estimation from sparse labels
    2. Add high-confidence predictions as pseudo-labels
    3. Re-estimate with expanded label set

    IMPORTANT: Handles 1-indexed sparse_labels internally and converts back to 0-indexed.

    Parameters
    ----------
    pca_embeddings : np.ndarray
        PCA-reduced embeddings
    sparse_labels : np.ndarray
        Sparse labels (1-indexed: 0 = unlabeled, 1,2,3,... = labeled classes)
    n_classes : int
        Number of classes (0-indexed count, e.g., 3 for classes 0,1,2)
    config : dict
        Configuration dictionary with label propagation parameters
    logger : logging.Logger
        Logger instance

    Returns
    -------
    predicted_labels : np.ndarray
        0-indexed labels including both original and pseudo-labels from iteration
    n_labels : np.ndarray
        Count of labeled neighbors (from second iteration)
    prob_labels : np.ndarray
        Probability distributions over 0-indexed classes (N x n_classes, from second iteration)
    stats : dict
        Statistics about propagation process
    """
    if logger:
        logger.info(f"\nPropagating labels using iterative K-NN...")
        logger.info(f"  Labeled samples: {(sparse_labels > 0).sum()}")
        logger.info(f"  Unlabeled samples: {(sparse_labels == 0).sum()}")

    # Get parameters from config
    label_config = config.get('labels', {})
    n_neighbors = label_config.get('n_neighbors', 50)
    n_min = label_config.get('minimum_labeled_neighbors', 5)
    n_min_auto = label_config.get('minimum_labeled_neighbors_for_autoprop', 15)
    prob_threshold = label_config.get('prob_threshold', 0.9)

    # Initialize LabelPropagation
    propagator = LabelPropagation(
        n_neighbors=n_neighbors,
        n_min=n_min,
        n_min_auto=n_min_auto,
        prob_threshold=prob_threshold,
        logger=logger
    )

    # Run iterative propagation (two-stage process) with 1-indexed labels
    predicted_labels_1indexed, n_labels, prob_labels_1indexed, stats = \
        propagator.iterative_propagation(pca_embeddings, sparse_labels, handle_fragmentation_separately=False)

    # Convert back to 0-indexed for evaluation
    # predicted_labels: 0 for unlabeled, >0 for labeled -> convert >0 to 0-indexed
    predicted_labels = np.where(
        predicted_labels_1indexed > 0,
        predicted_labels_1indexed - 1,  # Convert 1,2,3 -> 0,1,2
        0  # Keep unlabeled as 0
    )

    # prob_labels: Remove the unlabeled class (index 0) and shift to 0-indexed
    prob_labels = prob_labels_1indexed[:, 1:]  # Drop class 0 (unlabeled), keep classes 1,2,3 as 0,1,2

    if logger:
        logger.info(f"\nIterative propagation statistics:")
        logger.info(f"  Initial training labels: {stats['n_human']}")
        logger.info(f"  Pseudo-labels added (high confidence): {stats['n_added_iteration']}")
        logger.info(f"  Total labels after iteration: {stats['n_total_after_iteration']}")
        logger.info(f"  Objects with auto-labels: {stats['n_final_auto']}")
        logger.info(f"  Predicted label distribution (0-indexed): {np.bincount(predicted_labels)}")

    return predicted_labels, n_labels, prob_labels, stats


def compute_probabilistic_metrics(
    true_labels: np.ndarray,
    prob_labels: np.ndarray,
    class_names: Dict[int, str],
    logger: logging.Logger = None
) -> Dict:
    """
    Compute probabilistic evaluation metrics.

    Analyzes the probability distributions P(predicted | true) for each class,
    providing richer evaluation than hard classification alone.

    Parameters
    ----------
    true_labels : np.ndarray
        Ground truth labels (test set only)
    prob_labels : np.ndarray
        Probability distributions over classes (N x n_classes)
    class_names : dict
        Mapping from class IDs to names
    logger : logging.Logger
        Logger instance

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - prob_distributions: P(predicted | true) for each true class
        - mean_confidences: Mean probability assigned to correct class
        - entropy_per_class: Prediction uncertainty per true class
        - top_k_accuracy: Accuracy when considering top-k predictions
    """
    if logger:
        logger.info("\nComputing probabilistic metrics...")

    n_classes = prob_labels.shape[1]
    unique_labels = np.unique(true_labels)

    # Initialize results
    prob_distributions = {}
    mean_confidences = {}
    entropy_per_class = {}

    for true_class in unique_labels:
        # Process all classes (true_labels are 0-indexed: 0, 1, 2, ...)
        class_mask = true_labels == true_class
        class_probs = prob_labels[class_mask]

        # Mean probability distribution for this true class
        mean_prob_dist = np.mean(class_probs, axis=0)
        prob_distributions[class_names[true_class]] = mean_prob_dist.tolist()

        # Mean confidence (probability assigned to correct class)
        correct_class_probs = class_probs[:, true_class]
        mean_conf = np.mean(correct_class_probs)
        mean_confidences[class_names[true_class]] = float(mean_conf)

        # Entropy (uncertainty) per class
        # H = -sum(p * log(p))
        eps = 1e-10  # Avoid log(0)
        entropy = -np.sum(
            class_probs * np.log(class_probs + eps), axis=1
        )
        mean_entropy = np.mean(entropy)
        entropy_per_class[class_names[true_class]] = float(mean_entropy)

        if logger:
            logger.info(
                f"  {class_names[true_class]:20s}: "
                f"Mean confidence = {mean_conf:.4f}, "
                f"Mean entropy = {mean_entropy:.4f}"
            )

    # Top-k accuracy
    top_k_accuracy = {}
    for k in [1, 2, min(3, n_classes)]:
        if k > n_classes:
            continue
        # Get top-k predicted classes
        top_k_preds = np.argsort(prob_labels, axis=1)[:, -k:]

        # Check if true label is in top-k
        # All samples in test set are labeled (true_labels are 0-indexed)
        correct = np.array([
            true_labels[i] in top_k_preds[i]
            for i in range(len(true_labels))
        ])

        accuracy = np.mean(correct) if len(correct) > 0 else 0.0
        top_k_accuracy[f'top_{k}'] = float(accuracy)

        if logger:
            logger.info(f"  Top-{k} accuracy: {accuracy:.4f}")

    # Overall statistics
    overall_mean_confidence = np.mean(list(mean_confidences.values()))
    overall_mean_entropy = np.mean(list(entropy_per_class.values()))

    metrics = {
        'prob_distributions': prob_distributions,
        'mean_confidences': mean_confidences,
        'entropy_per_class': entropy_per_class,
        'top_k_accuracy': top_k_accuracy,
        'overall_mean_confidence': float(overall_mean_confidence),
        'overall_mean_entropy': float(overall_mean_entropy),
        'n_classes': n_classes,
        'class_names': class_names
    }

    if logger:
        logger.info(f"\nOverall statistics:")
        logger.info(f"  Mean confidence across classes: {overall_mean_confidence:.4f}")
        logger.info(f"  Mean entropy across classes: {overall_mean_entropy:.4f}")

    return metrics


def create_visualizations(
    output_path: Path,
    embeddings_pca: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_names: dict,
    metrics: dict,
    train_mask: np.ndarray = None,
    test_mask: np.ndarray = None,
    prob_metrics: dict = None,
    logger: logging.Logger = None
) -> None:
    """
    Create evaluation visualizations.

    Parameters
    ----------
    output_path : Path
        Directory to save figures
    embeddings_pca : np.ndarray
        PCA embeddings
    true_labels : np.ndarray
        Ground truth labels
    predicted_labels : np.ndarray
        Predicted labels
    class_names : dict
        Mapping from class IDs to names
    metrics : dict
        Hard classification metrics
    train_mask : np.ndarray, optional
        Boolean mask for training samples
    test_mask : np.ndarray, optional
        Boolean mask for test samples
    prob_metrics : dict, optional
        Probabilistic metrics
    logger : logging.Logger
        Logger instance
    """
    if not PLOTTING_AVAILABLE:
        if logger:
            logger.warning("Matplotlib not available, skipping visualizations")
        return

    if logger:
        logger.info("Creating visualizations...")

    n_classes = len(class_names)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Use colormap for potentially many classes
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    # 1. PCA with ground truth labels
    for class_id in range(n_classes):
        mask = true_labels == class_id
        axes[0, 0].scatter(
            embeddings_pca[mask, 0],
            embeddings_pca[mask, 1],
            c=[colors[class_id]],
            label=class_names[class_id],
            alpha=0.6,
            s=20
        )
    axes[0, 0].set_xlabel('PC1')
    axes[0, 0].set_ylabel('PC2')
    axes[0, 0].set_title('PCA: Ground Truth Labels (Galaxy Tags)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. PCA with predicted labels
    print(f'PCA embeddings shape: {embeddings_pca.shape}')
    for class_id in range(n_classes):
        mask = predicted_labels == class_id
        axes[0, 1].scatter(
            embeddings_pca[mask, 0],
            embeddings_pca[mask, 1],
            c=[colors[class_id]],
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
        xticklabels=[class_names[i] for i in range(n_classes)],
        yticklabels=[class_names[i] for i in range(n_classes)],
        ax=axes[1, 0],
        cbar_kws={'label': 'Count'}
    )
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('True')
    axes[1, 0].set_title('Confusion Matrix')

    # Rotate labels for readability
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
    axes[1, 0].set_yticklabels(axes[1, 0].get_yticklabels(), rotation=0)

    # 4. Metrics summary
    axes[1, 1].axis('off')

    metrics_text = f"""
Classification Metrics Summary (Test Set)

Hard Classification:
  Overall Purity: {metrics['overall_purity']:.3f}
  Overall Completeness: {metrics['overall_completeness']:.3f}
  Precision (weighted): {metrics['classification_report']['weighted avg']['precision']:.3f}
  Recall (weighted): {metrics['classification_report']['weighted avg']['recall']:.3f}
  F1-Score (weighted): {metrics['classification_report']['weighted avg']['f1-score']:.3f}
"""

    if prob_metrics is not None:
        metrics_text += f"""
Probabilistic Metrics:
  Mean Confidence: {prob_metrics['overall_mean_confidence']:.3f}
  Mean Entropy: {prob_metrics['overall_mean_entropy']:.3f}
  Top-1 Accuracy: {prob_metrics['top_k_accuracy'].get('top_1', 0):.3f}
  Top-2 Accuracy: {prob_metrics['top_k_accuracy'].get('top_2', 0):.3f}

Per-Class Mean Confidence:
"""
        for class_name in class_names.values():
            if class_name in prob_metrics['mean_confidences']:
                conf = prob_metrics['mean_confidences'][class_name]
                ent = prob_metrics['entropy_per_class'][class_name]
                metrics_text += f"  {class_name}: {conf:.3f} (entropy: {ent:.3f})\n"
    else:
        metrics_text += "\nPer-Class Metrics:\n"
        for class_name in class_names.values():
            pur = metrics['cluster_purities'][class_name]
            comp = metrics['class_completeness'][class_name]
            metrics_text += f"  {class_name}: Purity={pur:.3f}, Completeness={comp:.3f}\n"

    axes[1, 1].text(
        0.1, 0.5, metrics_text,
        fontsize=9,
        family='monospace',
        verticalalignment='center'
    )

    plt.tight_layout()
    plt.savefig(output_path / 'evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info("Visualizations saved")


def create_probability_analysis_figure(
    output_path: Path,
    true_labels: np.ndarray,
    prob_labels: np.ndarray,
    class_names: Dict[int, str],
    test_mask: np.ndarray = None,
    logger: logging.Logger = None
) -> None:
    """
    Create detailed probability analysis figure.

    Shows probability distributions P(predicted | true) for each true class
    and confidence statistics.

    Parameters
    ----------
    output_path : Path
        Directory to save figure
    true_labels : np.ndarray
        Ground truth labels
    prob_labels : np.ndarray
        Probability distributions (N x n_classes)
    class_names : dict
        Mapping from class IDs to names
    test_mask : np.ndarray, optional
        If provided, only analyze test samples
    logger : logging.Logger
        Logger instance
    """
    if not PLOTTING_AVAILABLE:
        if logger:
            logger.warning("Matplotlib not available, skipping probability analysis figure")
        return

    if logger:
        logger.info("Creating probability analysis figure...")

    # Apply test mask if provided
    if test_mask is not None:
        true_labels = true_labels[test_mask]
        prob_labels = prob_labels[test_mask]

    n_classes = prob_labels.shape[1]
    unique_labels = [i for i in range(n_classes) if i > 0]  # Skip class 0 (unlabeled)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Probability heatmap: P(predicted | true)
    prob_matrix = np.zeros((n_classes, n_classes))
    for true_class in unique_labels:
        class_mask = true_labels == true_class
        if class_mask.sum() > 0:
            prob_matrix[true_class, :] = np.mean(prob_labels[class_mask], axis=0)

    # Only show non-zero classes
    prob_matrix_plot = prob_matrix[unique_labels][:, unique_labels]
    class_names_list = [class_names[i] for i in unique_labels]

    sns.heatmap(
        prob_matrix_plot,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        xticklabels=class_names_list,
        yticklabels=class_names_list,
        ax=axes[0, 0],
        cbar_kws={'label': 'Mean Probability'}
    )
    axes[0, 0].set_xlabel('Predicted Class')
    axes[0, 0].set_ylabel('True Class')
    axes[0, 0].set_title('P(Predicted | True) - Mean Probabilities')
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45, ha='right')

    # 2. Confidence distribution (violin plot)
    confidence_data = []
    confidence_labels = []
    for true_class in unique_labels:
        class_mask = true_labels == true_class
        if class_mask.sum() > 0:
            # Confidence = probability assigned to correct class
            confidences = prob_labels[class_mask, true_class]
            confidence_data.extend(confidences)
            confidence_labels.extend([class_names[true_class]] * len(confidences))

    import pandas as pd
    df_conf = pd.DataFrame({'Class': confidence_labels, 'Confidence': confidence_data})

    # Violin plot
    parts = axes[0, 1].violinplot(
        [df_conf[df_conf['Class'] == cn]['Confidence'].values for cn in class_names_list],
        positions=range(len(class_names_list)),
        showmeans=True,
        showmedians=True
    )
    axes[0, 1].set_xticks(range(len(class_names_list)))
    axes[0, 1].set_xticklabels(class_names_list, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Confidence (P(correct class))')
    axes[0, 1].set_title('Confidence Distribution per True Class')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_ylim(0, 1)

    # 3. Mean confidence and entropy bar chart
    mean_confidences = []
    entropies = []
    for true_class in unique_labels:
        class_mask = true_labels == true_class
        if class_mask.sum() > 0:
            class_probs = prob_labels[class_mask]
            # Mean confidence
            mean_conf = np.mean(class_probs[:, true_class])
            mean_confidences.append(mean_conf)
            # Mean entropy
            eps = 1e-10
            entropy = -np.sum(class_probs * np.log(class_probs + eps), axis=1)
            entropies.append(np.mean(entropy))
        else:
            mean_confidences.append(0)
            entropies.append(0)

    x = np.arange(len(class_names_list))
    width = 0.35

    ax3_twin = axes[1, 0].twinx()
    bars1 = axes[1, 0].bar(x - width/2, mean_confidences, width, label='Mean Confidence', color='steelblue')
    bars2 = ax3_twin.bar(x + width/2, entropies, width, label='Mean Entropy', color='coral')

    axes[1, 0].set_xlabel('True Class')
    axes[1, 0].set_ylabel('Mean Confidence', color='steelblue')
    ax3_twin.set_ylabel('Mean Entropy', color='coral')
    axes[1, 0].set_title('Confidence and Uncertainty per Class')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(class_names_list, rotation=45, ha='right')
    axes[1, 0].tick_params(axis='y', labelcolor='steelblue')
    ax3_twin.tick_params(axis='y', labelcolor='coral')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Add legend
    lines1, labels1 = axes[1, 0].get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    axes[1, 0].legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # 4. Cumulative probability plot
    axes[1, 1].axis('off')
    summary_text = "Probability Analysis Summary\n\n"
    summary_text += f"Dataset: {len(true_labels)} test samples\n"
    summary_text += f"Classes: {len(unique_labels)}\n\n"
    summary_text += "Mean Confidence by Class:\n"
    for i, true_class in enumerate(unique_labels):
        summary_text += f"  {class_names[true_class]:15s}: {mean_confidences[i]:.4f}\n"

    summary_text += "\nMean Entropy by Class:\n"
    for i, true_class in enumerate(unique_labels):
        summary_text += f"  {class_names[true_class]:15s}: {entropies[i]:.4f}\n"

    axes[1, 1].text(
        0.1, 0.5, summary_text,
        fontsize=10,
        family='monospace',
        verticalalignment='center'
    )

    plt.tight_layout()
    plt.savefig(output_path / 'probability_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"Probability analysis figure saved to: {output_path / 'probability_analysis.png'}")


def create_sample_galaxy_figure(
    output_path: Path,
    images: np.ndarray,
    true_labels: np.ndarray,
    class_names: dict,
    n_samples: int = 5,
    logger: logging.Logger = None
) -> None:
    """
    Create QA figure showing sample FIRE2 mock images for each galaxy.

    Parameters
    ----------
    output_path : Path
        Output directory for saving figure
    images : np.ndarray
        Array of images (N, 3, H, W) - 3-channel images [g, i, hf]
    true_labels : np.ndarray
        Ground truth class labels (galaxy tags)
    class_names : dict
        Mapping from class ID to galaxy tag
    n_samples : int
        Number of samples to show per galaxy (default: 5)
    logger : logging.Logger
        Logger instance
    """
    if not PLOTTING_AVAILABLE:
        if logger:
            logger.warning("Matplotlib not available, skipping sample galaxy figure")
        return

    if logger:
        logger.info(f"Creating sample FIRE2 image figure ({n_samples} per galaxy)...")

    # Set up random state for consistent sampling
    rng = np.random.RandomState(42)

    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, n_samples, figsize=(n_samples * 2.5, n_classes * 2.5))

    # Handle case of single class
    if n_classes == 1:
        axes = axes.reshape(1, -1)

    for class_id in range(n_classes):
        # Get indices for this class
        class_indices = np.where(true_labels == class_id)[0]

        # Sample n_samples from this class
        if len(class_indices) >= n_samples:
            sample_indices = rng.choice(class_indices, size=n_samples, replace=False)
        else:
            sample_indices = class_indices

        for i, idx in enumerate(sample_indices):
            img = images[idx]  # Shape: (3, H, W) with [g, i, hf]

            # Use i-band (channel 1) for display
            i_band = img[1]

            # Normalize for display using percentiles
            vmin, vmax = np.percentile(i_band, [1, 99])
            img_norm = np.clip((i_band - vmin) / (vmax - vmin + 1e-8), 0, 1)

            axes[class_id, i].imshow(img_norm, origin='lower', cmap='viridis')
            axes[class_id, i].set_title(f"{class_names[class_id]}\n#{i+1}", fontsize=9)
            axes[class_id, i].axis('off')

    plt.suptitle('Sample FIRE2 Mock Images (i-band) by Galaxy Tag', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path / 'sample_fire2_galaxies.png', dpi=300, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"Sample galaxy figure saved to: {output_path / 'sample_fire2_galaxies.png'}")


def create_qa_figure(
    output_path: Path,
    images: np.ndarray,
    true_labels: np.ndarray,
    class_names: dict,
    logger: logging.Logger = None
) -> None:
    """
    Create QA figure showing broadband and high-frequency images.

    Based on byol_simulated.py's _create_qa_figure() method.
    Shows i-band (channel 1) and HF (channel 2) for one example per galaxy.

    Parameters
    ----------
    output_path : Path
        Output directory for saving figure
    images : np.ndarray
        Array of images (N, 3, H, W) where channels are [g, i, hf]
    true_labels : np.ndarray
        Ground truth class labels (galaxy tags)
    class_names : dict
        Mapping from class ID to galaxy tag
    logger : logging.Logger
        Logger instance
    """
    if not PLOTTING_AVAILABLE:
        if logger:
            logger.warning("Matplotlib not available, skipping QA figure")
        return

    if logger:
        logger.info("Creating QA figure with broadband and HF images...")

    # Set up random state for consistent sampling
    rng = np.random.RandomState(42)

    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, 2, figsize=(10, n_classes * 3.5))

    # Handle case of single class
    if n_classes == 1:
        axes = axes.reshape(1, -1)

    for class_id in range(n_classes):
        # Get indices for this class
        class_indices = np.where(true_labels == class_id)[0]

        # Select one random example from this class
        idx = rng.choice(class_indices)
        img = images[idx]  # Shape: (3, H, W) with [g, i, hf]

        # Extract i-band and HF images
        i_band = img[1]  # i-band is second channel
        hf_image = img[2]  # HF is third channel

        # Normalize for display using percentile clipping
        i_norm = (i_band - np.percentile(i_band, 1)) / (np.percentile(i_band, 99) - np.percentile(i_band, 1))
        i_norm = np.clip(i_norm, 0, 1)

        hf_norm = (hf_image - np.percentile(hf_image, 1)) / (np.percentile(hf_image, 99) - np.percentile(hf_image, 1))
        hf_norm = np.clip(hf_norm, 0, 1)

        # Plot i-band
        axes[class_id, 0].imshow(i_norm, origin='lower', cmap='gray')
        axes[class_id, 0].set_title(f"{class_names[class_id]} - i-band", fontsize=10)
        axes[class_id, 0].axis('off')

        # Plot HF image
        axes[class_id, 1].imshow(hf_norm, origin='lower', cmap='gray')
        axes[class_id, 1].set_title(f"{class_names[class_id]} - HF", fontsize=10)
        axes[class_id, 1].axis('off')

    plt.suptitle('QA Figure: Broadband and High-Frequency Images by Galaxy Tag', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / 'qa_figure.png', dpi=300, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"QA figure saved to: {output_path / 'qa_figure.png'}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='BYOL Analysis with FIRE2 Mock Images (Semi-Supervised K-NN)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script evaluates semi-supervised label propagation using FIRE2 mock images:

1. Loads FIRE2 mock images from multiple galaxy simulations
2. Splits ground truth labels into train (default 10%%) and test (90%%) sets
3. Trains BYOL model on projected simulation images
4. Extracts embeddings and applies PCA dimensionality reduction
5. Propagates labels from train to test using K-NN in PCA space
6. Evaluates both hard classification (argmax) and probabilistic metrics

Output includes:
- Hard classification metrics (confusion matrix, purity, completeness)
- Probabilistic metrics (P(predicted | true), confidence, entropy)
- Visualizations showing train/test split and probability distributions

Examples:
  # Use 3 galaxies with 50 images each (10%% training labels)
  python run_fire2_mock.py --tags m11b_res2100 m11d_res7100 m11h_res7100 --n-per-galaxy 50

  # Use 20%% of labels for training
  python run_fire2_mock.py --train-fraction 0.2

  # Disable stratified sampling
  python run_fire2_mock.py --no-stratified-split
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='../config.yaml',
        help='Path to BYOL configuration YAML file'
    )
    parser.add_argument(
        '--tags',
        type=str,
        nargs='+',
        default=['m11h_res7100', 'm11d_res7100', 'm11e_res7100'],
        help='Galaxy tags to use (default: m11b_res2100 m11d_res7100)'
    )
    parser.add_argument(
        '--n-per-galaxy',
        type=int,
        default=10000,
        help='Number of images per galaxy (default: 50)'
    )
    parser.add_argument(
        '--mock-images-dir',
        type=str,
        default='../local_data/mock_images',
        help='Directory containing mock images (default: ../local_data/mock_images)'
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
        '--train-fraction',
        type=float,
        default=0.3,
        help='Fraction of labels to use as "visible" training labels (default: 0.1 = 10%%)'
    )
    parser.add_argument(
        '--no-stratified-split',
        action='store_true',
        help='Disable stratified sampling (default: stratified sampling is enabled)'
    )

    args = parser.parse_args()

    # Set stratified_split attribute based on flag
    args.stratified_split = not args.no_stratified_split

    # Setup
    logger = setup_logging()
    config = load_config(args.config)

    logger.info("=" * 70)
    logger.info("FIRE2 MOCK IMAGE ANALYSIS")
    logger.info("=" * 70)

    # Override config
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = Path(config['data']['output_path']) / 'fire2_mock'

    output_path.mkdir(parents=True, exist_ok=True)

    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size

    logger.info(f"Output path: {output_path}")
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

        # Save loaded data
        data_path = output_path / 'fire2_mock_data.pkl'
        with open(data_path, 'wb') as f:
            pickle.dump({
                'images': images,
                'img_names': img_names,
                'true_labels': true_labels,
                'class_names': class_names,
                'galaxy_tags': args.tags,
                'n_per_galaxy': args.n_per_galaxy,
                'random_seed': args.random_seed
            }, f)

        logger.info(f"Data saved to: {data_path}")

        # Preprocess images for BYOL (create 3-channel [g, i, hf] images)
        logger.info("Preprocessing images for BYOL...")
        images_processed = preprocess_images_for_byol(images, target_channels=3, logger=logger)

        # Create sample galaxy QA figure after preprocessing
        create_sample_galaxy_figure(
            output_path,
            images_processed,  # Use 3-channel processed images
            true_labels,
            class_names,
            n_samples=5,
            logger=logger
        )

        # Create QA figure showing broadband and HF images
        create_qa_figure(
            output_path,
            images_processed,  # Use 3-channel processed images
            true_labels,
            class_names,
            logger=logger
        )

        # Train BYOL model
        logger.info("Training BYOL model...")
        model_manager = BYOLModelManager(config, output_path, logger)
        model_manager.train_model(images_processed, resume=False)

        # Extract embeddings
        logger.info("Extracting embeddings...")
        embeddings = model_manager.extract_embeddings(images_processed)

        # PCA
        logger.info("Computing PCA...")
        analyzer = EmbeddingAnalyzer(config, logger)
        embeddings_pca = analyzer.compute_pca(embeddings)

        # Save dimensionality reduction results
        results_path = output_path / 'dimensionality_reduction_results.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump({
                'embeddings_original': embeddings,
                'embeddings_pca': embeddings_pca,
                'img_names': img_names,
                'true_labels': true_labels,
                'scaler': analyzer.scaler,
                'pca': analyzer.pca
            }, f)

        # Create sparse labels (10% visible for semi-supervised learning)
        train_fraction = getattr(args, 'train_fraction', 0.1)
        stratified = getattr(args, 'stratified_split', True)
        sparse_labels, train_mask, test_mask = create_sparse_labels(
            true_labels,
            sample_fraction=train_fraction,
            random_seed=args.random_seed,
            stratified=stratified,
            logger=logger
        )

        # Propagate labels using iterative K-NN (on PCA embeddings)
        n_classes = len(args.tags)
        predicted_labels, n_labels, prob_labels, stats = propagate_labels_knn(
            embeddings_pca,  # Use PCA embeddings
            sparse_labels,
            n_classes=n_classes,
            config=config,
            logger=logger
        )

        # Track pseudo-labels in test set
        pseudo_in_test = (predicted_labels[test_mask] > 0) & (sparse_labels[test_mask] == 0)
        logger.info(f"\nPseudo-labels in test set: {pseudo_in_test.sum()}/{test_mask.sum()} ({pseudo_in_test.sum()/test_mask.sum()*100:.1f}%)")

        # Save train/test split with propagation stats
        split_path = output_path / 'train_test_split.pkl'
        with open(split_path, 'wb') as f:
            pickle.dump({
                'train_mask': train_mask,
                'test_mask': test_mask,
                'train_fraction': train_fraction,
                'stratified': stratified,
                'random_seed': args.random_seed,
                'propagation_stats': stats
            }, f)
        logger.info(f"Train/test split (with propagation stats) saved to: {split_path}")

        # Save probability labels
        prob_labels_path = output_path / 'probability_labels.npy'
        np.save(prob_labels_path, prob_labels)
        logger.info(f"Probability labels saved to: {prob_labels_path}")

        # Compute metrics on TEST SET ONLY
        logger.info("\n" + "=" * 70)
        logger.info("COMPUTING METRICS ON TEST SET")
        logger.info("=" * 70)

        # Hard classification metrics (argmax of probabilities)
        logger.info("\n1. Hard Classification Metrics (argmax)...")
        # Use argmax of prob_labels for final predictions, not iterative predicted_labels
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

        # Save both metric sets
        metrics_hard_path = output_path / 'classification_metrics_hard.json'
        with open(metrics_hard_path, 'w') as f:
            json.dump(metrics_hard, f, indent=2)
        logger.info(f"Hard classification metrics saved to: {metrics_hard_path}")

        metrics_prob_path = output_path / 'probabilistic_metrics.json'
        with open(metrics_prob_path, 'w') as f:
            json.dump(metrics_prob, f, indent=2)
        logger.info(f"Probabilistic metrics saved to: {metrics_prob_path}")

        # Create visualizations
        logger.info("\nCreating visualizations...")
        create_visualizations(
            output_path,
            embeddings_pca,
            true_labels,
            final_predictions,  # Use final predictions from argmax
            class_names,
            metrics_hard,
            train_mask=train_mask,
            test_mask=test_mask,
            prob_metrics=metrics_prob,
            logger=logger
        )

        # Create detailed probability analysis figure
        create_probability_analysis_figure(
            output_path,
            true_labels,
            prob_labels,
            class_names,
            test_mask=test_mask,
            logger=logger
        )

        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("SEMI-SUPERVISED CLASSIFICATION RESULTS (TEST SET)")
        logger.info("=" * 70)
        logger.info(f"\nData split:")
        logger.info(f"  Training set: {train_mask.sum()} samples ({train_mask.sum()/len(true_labels)*100:.1f}%)")
        logger.info(f"  Test set: {test_mask.sum()} samples ({test_mask.sum()/len(true_labels)*100:.1f}%)")
        logger.info(f"\nIterative label propagation:")
        logger.info(f"  Initial labels: {stats['n_human']}")
        logger.info(f"  Pseudo-labels added (high confidence): {stats['n_added_iteration']}")
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

        print("\n✅ FIRE2 mock image analysis completed successfully!")
        print(f"   Train/Test Split: {train_mask.sum()}/{test_mask.sum()} samples ({pseudo_in_test.sum()} pseudo-labels in test)")
        print(f"   Hard Classification - Purity: {metrics_hard['overall_purity']:.4f}, Completeness: {metrics_hard['overall_completeness']:.4f}")
        print(f"   Probabilistic - Mean Confidence: {metrics_prob['overall_mean_confidence']:.4f}")
        print(f"   Results saved to: {output_path}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n❌ Error: {e}")
        print("\nMake sure to generate FIRE2 images first using:")
        for tag in args.tags:
            print(f"  python generate_fire2_images.py --tag {tag} --n-images {args.n_per_galaxy}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
