#!/usr/bin/env python3
"""
BYOL Merger Analysis - AION-1 Embeddings with Linear Classifier

Uses AION-1 foundation model for embeddings (frozen) and trains only a
linear classification head. Follows the same framework as run_analysis.py.

Key Differences from BYOL:
- Uses AION-1 for embedding extraction (frozen, not trained)
- Only trains the linear classifier on top of embeddings
- Loads g-band and i-band images (no hf_image)
- Properly formats images as HSC data for AION-1

Usage
-----
# Full pipeline
python run_sfourl_aion1.py --mode full

# Train classifier only (AION embeddings are frozen)
python run_sfourl_aion1.py --mode train --epochs 100

# Analysis only (requires trained classifier)
python run_sfourl_aion1.py --mode analyze

# Custom config
python run_sfourl_aion1.py --config custom_config.yaml --mode full
"""

import os
import sys
import argparse
import logging
import pickle
import glob
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

# AION-1 imports
try:
    from aion import AION
    from aion.codecs import CodecManager
    from aion.modalities import HSCImage
    AION_AVAILABLE = True
except ImportError as e:
    AION_AVAILABLE = False
    print("Error: AION not installed. Please install with: pip install polymathic-aion")
    print(f"Import error: {e}")

# Add pieridae to path (go up 2 levels: scripts/ -> merger_analysis/ -> pieridae/)
sys.path.insert(0, str(Path(__file__).parents[2]))

from pieridae.starbursts.byol import (
    EmbeddingAnalyzer,
    compute_classification_metrics
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
    logger = logging.getLogger('aion1_analysis')
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


def load_merian_images_gi_only(
    data_path: Path,
    max_images: int = None,
    logger: logging.Logger = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Merian galaxy images (g and i bands only) from pickle files.

    Parameters
    ----------
    data_path : Path
        Path to directory containing M*/*_i_results.pkl files
    max_images : int, optional
        Maximum number of images to load (for testing)
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    images : np.ndarray
        Loaded images with shape (N, 2, H, W) - g and i bands only
    img_names : np.ndarray
        Object identifiers
    """
    pattern = f"{data_path}/M*/*i_results.pkl"
    filenames = glob.glob(pattern)

    if not filenames:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")

    if logger:
        logger.info(f"Found {len(filenames)} image files")

    # First pass: count valid images and get shape
    if logger:
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

            if max_images and len(valid_files) >= max_images:
                break

    n_images = len(valid_files)
    if logger:
        logger.info(f"Found {n_images} valid image sets")

    if n_images == 0:
        raise ValueError("No valid image files found")

    # Pre-allocate arrays - only 2 channels (g, i)
    images = np.zeros((n_images, 2, img_shape[0], img_shape[1]), dtype=np.float32)
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
            except Exception as e:
                if logger:
                    logger.warning(f"Error loading {current_filename}: {e}")
                continue

        if len(img) == 2:  # Only if both bands loaded successfully
            images[idx] = np.array(img, dtype=np.float32)
            img_names.append(Path(fname).parent.name)
            idx += 1

    # Trim to actual loaded images
    images = images[:idx]
    img_names = np.array(img_names)

    if logger:
        logger.info(f"Successfully loaded {idx} image sets")

    return images, img_names


def extract_aion1_embeddings(
    images: np.ndarray,
    output_file: Path,
    device: str = 'mps',
    batch_size: int = 32,
    num_encoder_tokens: int = 600,
    logger: logging.Logger = None
) -> np.ndarray:
    """
    Extract embeddings from images using AION-1 model.

    Uses memory-mapped file to avoid loading all embeddings into memory at once.

    Parameters
    ----------
    images : np.ndarray
        Input images with shape (N, 2, H, W) - g and i bands
    output_file : Path
        Path where embeddings will be saved (used for memory mapping)
    device : str
        Computing device ('cuda', 'mps', or 'cpu')
    batch_size : int
        Batch size for encoding
    num_encoder_tokens : int
        Number of encoder tokens to use
    logger : logging.Logger
        Logger instance

    Returns
    -------
    embeddings : np.ndarray
        Extracted embeddings with shape (N, embedding_dim)
    """
    if not AION_AVAILABLE:
        raise ImportError("AION not available. Please install: pip install polymathic-aion")

    if logger:
        logger.info("Loading AION-1 model...")

    # Determine device
    if device == 'cuda' and not torch.cuda.is_available():
        if logger:
            logger.warning("CUDA not available, trying MPS...")
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    elif device == 'mps' and not torch.backends.mps.is_available():
        if logger:
            logger.warning("MPS not available, using CPU...")
        device = 'cpu'

    if logger:
        logger.info(f"Using device: {device}")

    # Load AION-1 model
    try:
        model = AION.from_pretrained('polymathic-ai/aion-base').to(device)
        codec_manager = CodecManager(device=device)
        model.eval()
        if logger:
            logger.info("AION-1 model loaded successfully")
    except Exception as e:
        if logger:
            logger.error(f"Error loading AION-1 model: {e}")
        raise

    # Extract embeddings in batches
    n_images = len(images)

    # Determine embedding dimension from first batch
    if logger:
        logger.info("Determining embedding dimensions from first batch...")

    with torch.no_grad():
        first_batch = images[0:1]
        batch_tensor = torch.from_numpy(first_batch).to(device)
        hsc_image = HSCImage(flux=batch_tensor, bands=['HSC-G', 'HSC-I'])
        tokens = codec_manager.encode(hsc_image)
        test_embedding = model.encode(tokens, num_encoder_tokens=num_encoder_tokens)
        # AION returns (batch, seq_len, embed_dim), apply mean pooling
        test_embedding = test_embedding.mean(dim=1)  # (batch, seq_len, 768) -> (batch, 768)
        embedding_dim = test_embedding.shape[-1]

        if logger:
            logger.info(f"Embedding dimension: {embedding_dim}")

    # Create memory-mapped file for embeddings
    if logger:
        logger.info(f"Creating memory-mapped file at {output_file}")

    embeddings = np.memmap(
        output_file,
        dtype='float32',
        mode='w+',
        shape=(n_images, embedding_dim)
    )

    if logger:
        logger.info(f"Extracting embeddings for {n_images} images...")

    with torch.no_grad():
        for i in tqdm(range(0, n_images, batch_size), desc="Encoding batches"):
            batch = images[i:i+batch_size]
            batch_end = min(i + batch_size, n_images)

            try:
                # Convert to torch tensor
                batch_tensor = torch.from_numpy(batch).to(device)

                # Create HSC modality object
                hsc_image = HSCImage(
                    flux=batch_tensor,
                    bands=['HSC-G', 'HSC-I']  # Specify which bands we have
                )

                # Encode to tokens
                tokens = codec_manager.encode(hsc_image)

                # Extract embeddings
                batch_embeddings = model.encode(tokens, num_encoder_tokens=num_encoder_tokens)

                # Apply mean pooling: (batch, seq_len, 768) -> (batch, 768)
                batch_embeddings = batch_embeddings.mean(dim=1)

                # Write directly to memory-mapped file
                embeddings[i:batch_end] = batch_embeddings.cpu().numpy()

            except Exception as e:
                if logger:
                    logger.error(f"Error encoding batch {i}: {e}")
                # Fallback: create zero embeddings for this batch
                if logger:
                    logger.warning(f"Creating zero embeddings for batch {i}")
                embeddings[i:batch_end] = 0.0

    # Flush to disk
    embeddings.flush()

    if logger:
        logger.info(f"Embeddings shape: {embeddings.shape}")
        logger.info(f"Embeddings saved to {output_file}")

    return embeddings


class AION1Classifier:
    """
    Linear classifier trained on frozen AION-1 embeddings.

    This class trains only a linear classification head on top of
    pre-extracted AION-1 embeddings, similar to the BYOL semi-supervised
    approach but with frozen AION-1 features.

    Parameters
    ----------
    embedding_dim : int
        Dimension of AION-1 embeddings
    n_classes : int, default=5
        Number of classification classes
    device : torch.device, optional
        Device to run training/inference on
    logger : logging.Logger, optional
        Logger instance
    """

    def __init__(
        self,
        embedding_dim: int,
        n_classes: int = 5,
        device: torch.device = None,
        logger: logging.Logger = None
    ):
        self.embedding_dim = embedding_dim
        self.n_classes = n_classes
        self.logger = logger or self._setup_default_logger()

        # Setup device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        self.logger.info(f"AION1Classifier using device: {self.device}")

        # Create linear classifier
        self.classifier = nn.Linear(embedding_dim, n_classes).to(self.device)
        self.logger.info(f"Created linear classifier: {embedding_dim} -> {n_classes}")

    def _setup_default_logger(self) -> logging.Logger:
        """Setup a default logger if none provided"""
        logger = logging.getLogger('aion1_classifier')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            logger.addHandler(handler)
        return logger

    def train(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        num_epochs: int = 100,
        learning_rate: float = 1e-3,
        batch_size: int = 512,
        patience_limit: int = 20
    ) -> None:
        """
        Train linear classifier on AION-1 embeddings.

        Parameters
        ----------
        embeddings : np.ndarray
            AION-1 embeddings (N, embedding_dim)
        labels : np.ndarray
            1-indexed labels (0 = unlabeled, 1-5 = classes)
        num_epochs : int
            Number of training epochs
        learning_rate : float
            Learning rate
        batch_size : int
            Batch size for training
        patience_limit : int
            Early stopping patience
        """
        self.logger.info("Training linear classifier on AION-1 embeddings...")

        # Get labeled samples
        labeled_indices = np.where(labels > 0)[0]
        n_labeled = len(labeled_indices)

        if n_labeled == 0:
            raise ValueError("No labeled samples found for training!")

        self.logger.info(f"Found {n_labeled} labeled samples ({100*n_labeled/len(labels):.2f}% of dataset)")

        # Prepare training data
        train_embeddings = embeddings[labeled_indices]
        train_labels = labels[labeled_indices] - 1  # Convert to 0-indexed

        # Optimizer
        optimizer = Adam(self.classifier.parameters(), lr=learning_rate)

        # Training loop
        best_loss = float('inf')
        stop = 0

        for epoch in tqdm(range(num_epochs), desc="Training classifier"):
            # Shuffle training data
            perm = np.random.permutation(n_labeled)
            epoch_losses = []

            # Mini-batch training
            for i in range(0, n_labeled, batch_size):
                batch_idx = perm[i:min(i+batch_size, n_labeled)]

                # Get batch
                batch_emb = torch.tensor(
                    train_embeddings[batch_idx],
                    dtype=torch.float32
                ).to(self.device)

                batch_lbl = torch.tensor(
                    train_labels[batch_idx],
                    dtype=torch.long
                ).to(self.device)

                # Forward pass
                logits = self.classifier(batch_emb)
                loss = nn.functional.cross_entropy(logits, batch_lbl)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            # Epoch statistics
            mean_loss = np.mean(epoch_losses)

            if mean_loss > best_loss:
                stop += 1
            else:
                best_loss = mean_loss
                stop = 0

            if epoch % 10 == 0 or stop:
                self.logger.info(
                    f"Epoch {epoch}, Loss: {mean_loss:.4f}, "
                    f"Patience left: {patience_limit-stop}/{patience_limit}"
                )

            # Early stopping
            if stop >= patience_limit:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        self.logger.info("Classifier training complete")

    def predict(
        self,
        embeddings: np.ndarray,
        batch_size: int = 512
    ) -> np.ndarray:
        """
        Generate probabilistic predictions.

        Parameters
        ----------
        embeddings : np.ndarray
            AION-1 embeddings (N, embedding_dim)
        batch_size : int
            Batch size for inference

        Returns
        -------
        probabilities : np.ndarray
            Class probabilities (N, n_classes)
        """
        self.classifier.eval()
        n_samples = len(embeddings)
        all_probs = []

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch = torch.tensor(
                    embeddings[i:min(i+batch_size, n_samples)],
                    dtype=torch.float32
                ).to(self.device)

                logits = self.classifier(batch)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)

    def save(self, path: Path) -> None:
        """Save classifier state"""
        torch.save({
            'classifier_state_dict': self.classifier.state_dict(),
            'embedding_dim': self.embedding_dim,
            'n_classes': self.n_classes
        }, path)
        self.logger.info(f"Classifier saved to: {path}")

    def load(self, path: Path) -> None:
        """Load classifier state"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.logger.info(f"Classifier loaded from: {path}")


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
    axes[0].set_title('PCA Embeddings (AION-1)')
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
    axes[1].set_title('UMAP Embeddings (AION-1)')
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
    output_path: Path,
    logger: logging.Logger
) -> None:
    """Run AION-1 embedding extraction + classifier training"""
    logger.info("=" * 60)
    logger.info("TRAINING MODE (AION-1 Embeddings + Linear Classifier)")
    logger.info("=" * 60)

    # Load labels
    labels = load_labels(config, img_names, logger)

    # Extract AION-1 embeddings (frozen)
    embeddings_file = output_path / 'aion1_embeddings.npy'

    if embeddings_file.exists():
        logger.info(f"Loading existing AION-1 embeddings from {embeddings_file}")
        embeddings = np.load(embeddings_file, mmap_mode='r')
    else:
        logger.info("Extracting AION-1 embeddings...")
        embeddings = extract_aion1_embeddings(
            images,
            output_file=embeddings_file,
            device=config.get('device', 'mps'),
            batch_size=config.get('inference', {}).get('batch_size', 32),
            num_encoder_tokens=config.get('aion', {}).get('num_encoder_tokens', 600),
            logger=logger
        )

    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Train linear classifier
    logger.info("Training linear classifier on AION-1 embeddings...")
    classifier = AION1Classifier(
        embedding_dim=embeddings.shape[1],
        n_classes=5,
        logger=logger
    )

    classifier.train(
        embeddings=embeddings,
        labels=labels,
        num_epochs=config['training'].get('num_epochs', 100),
        learning_rate=config['training'].get('learning_rate', 1e-3),
        batch_size=config['training'].get('batch_size', 512),
        patience_limit=config['training'].get('patience_limit', 20)
    )

    # Save classifier
    classifier_path = output_path / 'aion1_classifier.pt'
    classifier.save(classifier_path)

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

    # Load or extract AION-1 embeddings
    embeddings_file = output_path / 'aion1_embeddings.npy'

    if embeddings_file.exists():
        logger.info(f"Loading existing AION-1 embeddings from {embeddings_file}")
        embeddings = np.load(embeddings_file, mmap_mode='r')
    else:
        logger.info("Extracting AION-1 embeddings...")
        embeddings = extract_aion1_embeddings(
            images,
            output_file=embeddings_file,
            device=config.get('device', 'mps'),
            batch_size=config.get('inference', {}).get('batch_size', 32),
            num_encoder_tokens=config.get('aion', {}).get('num_encoder_tokens', 600),
            logger=logger
        )

    # PCA and UMAP
    logger.info("Computing PCA and UMAP...")
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

    # Load classifier and make predictions
    logger.info("Loading trained classifier...")
    classifier = AION1Classifier(
        embedding_dim=embeddings.shape[1],
        n_classes=5,
        logger=logger
    )

    classifier_path = output_path / 'aion1_classifier.pt'
    classifier.load(classifier_path)

    # Get predictions
    logger.info("Generating predictions...")
    prob_labels = classifier.predict(embeddings)
    predicted_labels = np.argmax(prob_labels, axis=1)

    # Save predictions
    predictions_path = output_path / 'predictions.pkl'
    with open(predictions_path, 'wb') as f:
        pickle.dump({
            'prob_labels': prob_labels,
            'predicted_labels': predicted_labels,
            'img_names': img_names
        }, f)

    logger.info(f"Predictions saved to: {predictions_path}")

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
    """Run complete pipeline: embedding extraction + classifier training + analysis"""
    logger.info("=" * 60)
    logger.info("FULL PIPELINE MODE")
    logger.info("=" * 60)

    # Training (embedding extraction + classifier training)
    run_training(config, images, img_names, output_path, logger)

    # Analysis
    run_analysis(config, images, img_names, output_path, logger)

    logger.info("Full pipeline complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='AION-1 Analysis for Galaxy Merger Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python run_sfourl_aion1.py --mode full

  # Extract embeddings and train classifier
  python run_sfourl_aion1.py --mode train --epochs 100

  # Analyze with existing classifier
  python run_sfourl_aion1.py --mode analyze

  # Use custom config
  python run_sfourl_aion1.py --config my_config.yaml --mode full
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
        '--max-images',
        type=int,
        help='Maximum number of images to load (for testing)'
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
    else:
        config['data']['output_path'] = Path(config['data']['output_path']) / 'aion1_sfourl'

    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size

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

    logger.info(f"Starting AION-1 analysis in {args.mode} mode")
    logger.info(f"Input path: {config['data']['input_path']}")
    logger.info(f"Output path: {config['data']['output_path']}")

    try:
        # Load images (g and i bands only)
        logger.info("Loading images...")
        images, img_names = load_merian_images_gi_only(
            config['data']['input_path'],
            max_images=args.max_images,
            logger=logger
        )

        logger.info(f"Loaded {len(images)} images with shape {images.shape}")

        # Run requested mode
        if args.mode == 'train':
            run_training(config, images, img_names, output_path, logger)
        elif args.mode == 'analyze':
            run_analysis(config, images, img_names, output_path, logger)
        elif args.mode == 'full':
            run_full_pipeline(config, images, img_names, output_path, logger)

        logger.info("=" * 60)
        logger.info("SUCCESS")
        logger.info("=" * 60)
        print("\n✅ AION-1 analysis completed successfully!")

    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
