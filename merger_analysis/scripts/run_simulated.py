#!/usr/bin/env python3
"""
BYOL Simulated Galaxy Analysis

Refactored version using pieridae.starbursts.byol module.
Tests BYOL classification performance on simulated Sersic galaxies.

Usage
-----
# Run with defaults (300 galaxies per class)
python run_simulated.py

# Custom number per class
python run_simulated.py --n-per-class 500

# Custom epochs and batch size
python run_simulated.py --epochs 300 --batch-size 512

# Custom output path
python run_simulated.py --output-path ./my_sim_results
"""

import sys
import argparse
import logging
import json
import pickle
from pathlib import Path

import yaml
import numpy as np

# Add pieridae to path (go up 2 levels: scripts/ -> merger_analysis/ -> pieridae/)
sys.path.insert(0, str(Path(__file__).parents[2]))

from pieridae.starbursts.byol import (
    BYOLModelManager,
    EmbeddingAnalyzer,
    SimulatedGalaxyGenerator,
    compute_classification_metrics
)

# Plotting
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.cluster import KMeans
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
    logger = logging.getLogger('simulated_analysis')
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def classify_embeddings(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    n_clusters: int = 3,
    random_seed: int = 42,
    logger: logging.Logger = None
) -> np.ndarray:
    """
    Classify embeddings using KMeans and align with true labels.

    Parameters
    ----------
    embeddings : np.ndarray
        Input embeddings
    true_labels : np.ndarray
        Ground truth labels
    n_clusters : int
        Number of clusters
    random_seed : int
        Random seed
    logger : logging.Logger

    Returns
    -------
    aligned_labels : np.ndarray
        Predicted labels aligned with true labels
    """
    if logger:
        logger.info(f"Clustering embeddings into {n_clusters} clusters...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Align cluster labels with true labels using Hungarian algorithm
    from sklearn.metrics import confusion_matrix
    from scipy.optimize import linear_sum_assignment

    conf_matrix = confusion_matrix(true_labels, cluster_labels)
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)
    cluster_to_true = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}

    aligned_labels = np.array([cluster_to_true[c] for c in cluster_labels])

    if logger:
        logger.info(f"Cluster alignment mapping: {cluster_to_true}")
        logger.info(f"Predicted label distribution: {np.bincount(aligned_labels)}")

    return aligned_labels


def create_visualizations(
    output_path: Path,
    embeddings_pca: np.ndarray,
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


def create_sample_galaxy_figure(
    output_path: Path,
    images: np.ndarray,
    true_labels: np.ndarray,
    class_names: dict,
    n_samples: int = 3,
    logger: logging.Logger = None
) -> None:
    """
    Create QA figure showing sample input images for each classification.

    Based on byol_simulated.py's _create_sample_galaxy_images() method.
    Shows RGB composites of sample galaxies from each class.

    Parameters
    ----------
    output_path : Path
        Output directory for saving figure
    images : np.ndarray
        Array of images (N, C, H, W) where C=3 (g, i, hf channels)
    true_labels : np.ndarray
        Ground truth class labels
    class_names : dict
        Mapping from class ID to class name
    n_samples : int
        Number of samples to show per class (default: 3)
    logger : logging.Logger
        Logger instance
    """
    if not PLOTTING_AVAILABLE:
        if logger:
            logger.warning("Matplotlib not available, skipping sample galaxy figure")
        return

    if logger:
        logger.info(f"Creating sample galaxy figure ({n_samples} per class)...")

    # Set up random state for consistent sampling
    rng = np.random.RandomState(42)

    fig, axes = plt.subplots(3, n_samples, figsize=(12, 10))

    for class_id in range(3):
        # Get indices for this class
        class_indices = np.where(true_labels == class_id)[0]

        # Sample n_samples from this class
        if len(class_indices) >= n_samples:
            sample_indices = rng.choice(class_indices, size=n_samples, replace=False)
        else:
            sample_indices = class_indices

        for i, idx in enumerate(sample_indices):
            img = images[idx]  # Shape: (3, H, W) with [g, i, hf]

            # Create RGB composite from g, i bands (normalize for display)
            g_band = img[0]
            i_band = img[1]
            hf_band = img[2]

            # Normalize each channel to [0, 1]
            g_norm = (g_band - g_band.min()) / (g_band.max() - g_band.min() + 1e-8)
            i_norm = (i_band - i_band.min()) / (i_band.max() - i_band.min() + 1e-8)
            hf_norm = (hf_band - hf_band.min()) / (hf_band.max() - hf_band.min() + 1e-8)

            # Create RGB composite (R=i, G=g, B=g for pseudo-color)
            rgb = np.stack([i_norm, g_norm, g_norm], axis=-1)

            axes[class_id, i].imshow(rgb, origin='lower')
            axes[class_id, i].set_title(f"{class_names[class_id]} {i+1}", fontsize=10)
            axes[class_id, i].axis('off')

    plt.suptitle('Sample Simulated Galaxies by Class', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path / 'sample_galaxies.png', dpi=300, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"Sample galaxy figure saved to: {output_path / 'sample_galaxies.png'}")


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
    Shows i-band (channel 1) and HF (channel 2) for one example per class.

    Parameters
    ----------
    output_path : Path
        Output directory for saving figure
    images : np.ndarray
        Array of images (N, 3, H, W) where channels are [g, i, hf]
    true_labels : np.ndarray
        Ground truth class labels
    class_names : dict
        Mapping from class ID to class name
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

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))

    for class_id in range(3):
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

    plt.suptitle('QA Figure: Broadband and High-Frequency Images by Class', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / 'qa_figure.png', dpi=300, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"QA figure saved to: {output_path / 'qa_figure.png'}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='BYOL Simulated Galaxy Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script:
1. Generates simulated galaxies (disk, double nuclei, dipole)
2. Trains BYOL model
3. Extracts embeddings and applies PCA
4. Clusters embeddings and evaluates classification performance

Output includes:
- Simulated galaxy images
- Classification metrics (purity, completeness, confusion matrix)
- Visualizations
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='../config.yaml',
        help='Path to configuration YAML file'
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

    args = parser.parse_args()

    # Setup
    logger = setup_logging()
    config = load_config(args.config)

    logger.info("=" * 60)
    logger.info("SIMULATED GALAXY ANALYSIS")
    logger.info("=" * 60)

    # Override config
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = Path(config['data']['output_path']) / 'simulated'

    output_path.mkdir(parents=True, exist_ok=True)

    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size

    logger.info(f"Output path: {output_path}")
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

        # Train BYOL model
        logger.info("Training BYOL model...")
        model_manager = BYOLModelManager(config, output_path, logger)
        model_manager.train_model(images, resume=False)

        # Extract embeddings
        logger.info("Extracting embeddings...")
        embeddings = model_manager.extract_embeddings(images)

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

        # Classify using KMeans
        predicted_labels = classify_embeddings(
            embeddings,
            true_labels,
            n_clusters=3,
            random_seed=args.random_seed,
            logger=logger
        )

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

        # Create sample galaxy QA figure (before evaluation results)
        create_sample_galaxy_figure(
            output_path,
            images,
            true_labels,
            class_names,
            n_samples=3,
            logger=logger
        )

        # Create QA figure showing broadband and HF images
        create_qa_figure(
            output_path,
            images,
            true_labels,
            class_names,
            logger=logger
        )

        # Create visualizations
        create_visualizations(
            output_path,
            embeddings_pca,
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

        print("\n✅ Simulated analysis completed successfully!")
        print(f"   Overall Purity: {metrics['overall_purity']:.4f}")
        print(f"   Overall Completeness: {metrics['overall_completeness']:.4f}")
        print(f"   Results saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
