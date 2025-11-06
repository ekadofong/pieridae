"""
BYOL Analysis Core Module for Merger Classification

This module provides reusable components for BYOL-based galaxy merger classification,
including model management, embedding analysis, and label propagation.

Classes
-------
BYOLModelManager
    Handles BYOL model setup, training, and embedding extraction
EmbeddingAnalyzer
    Performs PCA/UMAP dimensionality reduction and similarity analysis
LabelPropagation
    K-NN based label propagation and object flagging
SimulatedGalaxyGenerator
    Generates simulated galaxy images for testing
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import models, transforms
from tqdm import tqdm
import pickle
import glob

# ML libraries
from byol_pytorch import BYOL
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, classification_report
from scipy.optimize import linear_sum_assignment

# Astropy for simulated galaxies
try:
    from astropy.modeling.functional_models import Sersic2D
    from scipy import ndimage
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

# Starlet transform for high-frequency images
try:
    from ekfstats import imstats
    import sep
    STARLET_AVAILABLE = True
except ImportError:
    STARLET_AVAILABLE = False


class BYOLModelManager:
    """
    Manages BYOL model setup, training, and inference.

    This class handles device configuration (MPS/CUDA/CPU), model initialization,
    training with checkpointing, and embedding extraction from trained models.

    Parameters
    ----------
    config : dict
        Configuration dictionary with model, training, and inference parameters
    output_path : Path
        Directory for saving models and checkpoints
    logger : logging.Logger, optional
        Logger instance. If None, creates a new logger

    Attributes
    ----------
    device : torch.device
        Computing device (MPS, CUDA, or CPU)
    learner : BYOL
        BYOL model instance
    config : dict
        Configuration parameters
    """

    def __init__(
        self,
        config: Dict[str, Any],
        output_path: Path,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.logger = logger or self._setup_default_logger()
        self.device = self._setup_device()
        self.learner = None

        # Adjust batch sizes for device limitations
        self._adjust_batch_sizes()

        self.logger.info(f"BYOLModelManager initialized on device: {self.device}")

    def _setup_default_logger(self) -> logging.Logger:
        """Setup a default logger if none provided"""
        logger = logging.getLogger('byol_model_manager')
        logger.setLevel(logging.INFO)
        # Only add handler if none exist to avoid duplicates
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
        return logger

    def _setup_device(self) -> torch.device:
        """
        Setup computing device with MPS/CUDA/CPU support.

        Returns
        -------
        device : torch.device
            Selected computing device
        """
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            self.logger.info(f"Using Apple Silicon GPU (MPS): {device}")
            torch.mps.set_per_process_memory_fraction(0.8)
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            self.logger.info(f"Using NVIDIA GPU: {device}")
        else:
            device = torch.device('cpu')
            self.logger.info(f"Using CPU: {device}")
            torch.set_num_threads(os.cpu_count())

        return device

    def _adjust_batch_sizes(self) -> None:
        """Adjust batch sizes for MPS device limitations"""
        if self.device.type == 'mps':
            original_train_batch = self.config['training']['batch_size']
            self.config['training']['batch_size'] = min(128, original_train_batch)

            original_inf_batch = self.config['inference']['batch_size']
            self.config['inference']['batch_size'] = min(128, original_inf_batch)

            if original_train_batch != self.config['training']['batch_size']:
                self.logger.info(
                    f"Adjusted training batch size for MPS: "
                    f"{original_train_batch} -> {self.config['training']['batch_size']}"
                )
            if original_inf_batch != self.config['inference']['batch_size']:
                self.logger.info(
                    f"Adjusted inference batch size for MPS: "
                    f"{original_inf_batch} -> {self.config['inference']['batch_size']}"
                )

    def setup_model(self) -> BYOL:
        """
        Initialize BYOL model with data augmentations.

        Returns
        -------
        learner : BYOL
            Initialized BYOL model
        """
        self.logger.info("Setting up BYOL model...")

        # Data augmentations
        transform1 = nn.Sequential(
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        )

        transform2 = nn.Sequential(
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
        )

        # Base model - ResNet18 for compatibility
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # BYOL learner
        self.learner = BYOL(
            resnet,
            image_size=self.config['model']['image_size'],
            hidden_layer='avgpool',
            projection_size=self.config['model']['projection_size'],
            projection_hidden_size=self.config['model']['projection_hidden_size'],
            moving_average_decay=self.config['model']['moving_average_decay'],
            use_momentum=True,
            augment_fn=transform1,
            augment_fn2=transform2
        ).to(self.device)

        self.logger.info(f"BYOL model setup complete on {self.device}")
        return self.learner

    def train_model(
        self,
        images: np.ndarray,
        resume: bool = True,
        patience_limit: int = 20
    ) -> None:
        """
        Train BYOL model with checkpointing.

        Parameters
        ----------
        images : np.ndarray
            Training images with shape (N, C, H, W)
        resume : bool, default=True
            Whether to resume from checkpoint if available
        """
        self.logger.info("Starting BYOL training...")

        if self.learner is None:
            self.setup_model()

        optimizer = Adam(
            self.learner.parameters(),
            lr=float(self.config['training']['learning_rate'])
        )

        # Setup checkpointing
        checkpoint_path = self.output_path / 'model_checkpoint.pt'
        start_epoch = 0

        # Resume from checkpoint if exists
        if checkpoint_path.exists() and resume:
            self.logger.info("Loading checkpoint...")
            try:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location=self.device,
                    weights_only=False
                )
                self.learner.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                self.logger.info(f"Resumed from epoch {start_epoch}")
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint: {e}")
                self.logger.info("Starting training from scratch...")
                start_epoch = 0

        # Training loop
        num_epochs = self.config['training']['num_epochs']
        save_interval = self.config['training']['save_interval']
        batch_size = self.config['training']['batch_size']
        
        prev_loss = []
        stop = 0
        
        for epoch in tqdm(range(start_epoch, num_epochs), desc="Training BYOL"):
            try:
                # Sample random batch
                indices = np.random.permutation(len(images))[:batch_size]
                batch = torch.tensor(
                    images[indices],
                    dtype=torch.float32
                ).to(self.device)

                loss = self.learner(batch)

                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping for MPS stability
                if self.device.type == 'mps':
                    torch.nn.utils.clip_grad_norm_(self.learner.parameters(), max_norm=1.0)

                optimizer.step()
                self.learner.update_moving_average()
                
                if epoch<450:
                    pass
                elif (loss > np.min(prev_loss)):
                    stop += 1 
                else:
                    stop = 0

                    
                if (epoch % 10 == 0) or stop:
                    self.logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}. Mean(Loss[-50:]): {np.mean(prev_loss):.4f}")

                # Save checkpoint
                if ((epoch + 1) % save_interval == 0) or stop:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.learner.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                        'config': self.config,
                        'device': str(self.device)
                    }
                    torch.save(checkpoint, checkpoint_path)
                    self.logger.info(f"Checkpoint saved at epoch {epoch}")
                
                if (stop>patience_limit):
                    self.logger.info('Patience limit exceeded. Enforcing early training stop.')
                    break
                else:
                    prev_loss.append(loss.item())
                    #if len(prev_loss) > 50:
                    #    prev_loss.pop(0)

            except RuntimeError as e:
                if "MPS" in str(e):
                    self.logger.error(f"MPS error at epoch {epoch}: {e}")
                    self.logger.info("Trying to continue with CPU fallback...")
                    self.device = torch.device('cpu')
                    self.learner = self.learner.to(self.device)
                    continue
                else:
                    raise e

        # Save final model
        final_model_path = self.output_path / 'byol_final_model.pt'
        torch.save({
            'model_state_dict': self.learner.state_dict(),
            'config': self.config,
            'training_complete': True,
            'device': str(self.device)
        }, final_model_path)

        self.logger.info("Training completed successfully")

    def load_trained_model(
        self,
        model_path: Optional[Path] = None
    ) -> BYOL:
        """
        Load trained model from checkpoint.

        Parameters
        ----------
        model_path : Path, optional
            Path to model file. If None, looks for standard checkpoint files

        Returns
        -------
        learner : BYOL
            Loaded BYOL model in eval mode
        """
        if model_path is None:
            model_path = self.output_path / 'byol_final_model.pt'
            if not model_path.exists():
                model_path = self.output_path / 'model_checkpoint.pt'

        if not model_path.exists():
            raise FileNotFoundError(f"No trained model found: {model_path}")

        self.logger.info(f"Loading trained model from: {model_path}")

        if self.learner is None:
            self.setup_model()

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if 'model_state_dict' in checkpoint:
            self.learner.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.learner.load_state_dict(checkpoint)

        self.learner.eval()
        self.logger.info("Model loaded successfully")
        return self.learner

    def extract_embeddings(
        self,
        images: np.ndarray,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract embeddings from images using trained model.

        Parameters
        ----------
        images : np.ndarray
            Images with shape (N, C, H, W)
        batch_size : int, optional
            Batch size for inference. If None, uses config value

        Returns
        -------
        embeddings : np.ndarray
            Extracted embeddings with shape (N, embedding_dim)
        """
        self.logger.info(f"Extracting embeddings from {len(images)} images...")

        if self.learner is None:
            self.load_trained_model()

        self.learner.eval()
        all_embeddings = []

        if batch_size is None:
            batch_size = self.config['inference']['batch_size']

        # Adjust for MPS
        if self.device.type == 'mps':
            batch_size = min(64, batch_size)

        num_batches = (len(images) + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Extracting embeddings"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(images))

                batch = torch.tensor(
                    images[start_idx:end_idx],
                    dtype=torch.float32
                ).to(self.device)

                try:
                    _, embeddings = self.learner(batch, return_embedding=True)
                    all_embeddings.append(embeddings.cpu().numpy())

                except RuntimeError as e:
                    if "MPS" in str(e) and self.device.type == 'mps':
                        self.logger.warning(f"MPS error, falling back to CPU for batch {i}")
                        batch = batch.cpu()
                        learner_cpu = self.learner.cpu()
                        _, embeddings = learner_cpu(batch, return_embedding=True)
                        all_embeddings.append(embeddings.cpu().numpy())
                        self.learner = self.learner.to(self.device)
                    else:
                        raise e

        embeddings = np.vstack(all_embeddings)

        # Save embeddings
        embeddings_path = self.output_path / 'embeddings.npy'
        np.save(embeddings_path, embeddings)

        self.logger.info(f"Extracted embeddings shape: {embeddings.shape}")
        return embeddings


class EmbeddingAnalyzer:
    """
    Performs dimensionality reduction and similarity analysis on embeddings.

    This class handles PCA and UMAP computations, with automatic component
    selection based on explained variance.

    Parameters
    ----------
    config : dict
        Configuration dictionary with analysis parameters
    logger : logging.Logger, optional
        Logger instance

    Attributes
    ----------
    scaler : StandardScaler
        Fitted scaler for embeddings
    pca : PCA
        Fitted PCA model
    umap_reducer : umap.UMAP
        Fitted UMAP model
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.logger = logger or self._setup_default_logger()

        self.scaler = None
        self.pca = None
        self.umap_reducer = None

    def _setup_default_logger(self) -> logging.Logger:
        """Setup a default logger if none provided"""
        logger = logging.getLogger('embedding_analyzer')
        logger.setLevel(logging.INFO)
        # Only add handler if none exist to avoid duplicates
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
        return logger

    def compute_pca(
        self,
        embeddings: np.ndarray,
        n_components: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply PCA to embeddings with automatic component selection.

        Parameters
        ----------
        embeddings : np.ndarray
            Input embeddings with shape (N, D)
        n_components : int, optional
            Number of PCA components. If None, determined from config

        Returns
        -------
        embeddings_pca : np.ndarray
            PCA-transformed embeddings with shape (N, n_components)
        """
        self.logger.info("Computing PCA...")

        # Clean data
        embeddings_clean = np.nan_to_num(
            embeddings, nan=0.0, posinf=0.0, neginf=0.0
        )

        # Standardization
        self.scaler = StandardScaler()
        embeddings_scaled = self.scaler.fit_transform(embeddings_clean)

        # Determine number of components
        if n_components is None:
            n_components = self.config['analysis'].get('pca_components')

        if n_components is None:
            self.logger.info("Automatically selecting N(PCA)")
            # Auto-determine from explained variance threshold
            pca_full = PCA()
            pca_full.fit(embeddings_scaled)
            cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
            threshold = self.config['analysis'].get('explained_variance_threshold', 0.95)
            n_components = np.argmax(cumsum_var >= threshold) + 1

        # Limit to valid range
        max_components = min(
            embeddings_scaled.shape[0] - 1,
            embeddings_scaled.shape[1]
        )
        n_components = min(n_components, max_components)

        # Fit PCA
        self.pca = PCA(n_components=n_components, random_state=42)
        embeddings_pca = self.pca.fit_transform(embeddings_scaled)

        explained_var = self.pca.explained_variance_ratio_.sum() * 100
        self.logger.info(f"PCA components: {n_components}")
        self.logger.info(f"Explained variance: {explained_var:.1f}%")

        return embeddings_pca

    def compute_umap(
        self,
        embeddings_pca: np.ndarray,
        n_components: int = 2
    ) -> np.ndarray:
        """
        Apply UMAP to PCA-transformed embeddings.

        Parameters
        ----------
        embeddings_pca : np.ndarray
            PCA-transformed embeddings
        n_components : int, default=2
            Number of UMAP components

        Returns
        -------
        embeddings_umap : np.ndarray
            UMAP-transformed embeddings with shape (N, n_components)
        """
        self.logger.info("Computing UMAP...")

        n_neighbors = min(
            self.config['analysis'].get('n_neighbors', 15),
            len(embeddings_pca) - 1
        )

        self.umap_reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=self.config['analysis'].get('min_dist', 0.1),
            metric=self.config['analysis'].get('metric', 'euclidean'),
            random_state=42,
            verbose=True
        )

        embeddings_umap = self.umap_reducer.fit_transform(embeddings_pca)

        self.logger.info(f"UMAP embedding shape: {embeddings_umap.shape}")
        return embeddings_umap

    def compute_similarity_matrix(
        self,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity matrix.

        Parameters
        ----------
        embeddings : np.ndarray
            Input embeddings

        Returns
        -------
        similarity : np.ndarray
            Pairwise cosine similarity matrix
        """
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(embeddings)


class LabelPropagation:
    """
    K-NN based label propagation for semi-supervised classification.

    Uses nearest neighbors in PCA space to estimate labels for unlabeled objects,
    with iterative refinement and confidence thresholding.

    Parameters
    ----------
    n_neighbors : int, default=50
        Number of nearest neighbors to consider
    n_min : int, default=8
        Minimum number of labeled neighbors required for propagation
    prob_threshold : float, default=0.6
        Confidence threshold for adding auto-labels
    frag_threshold : float, default=0.4
        Special threshold for fragmentation class
    logger : logging.Logger, optional
        Logger instance
    """

    def __init__(
        self,
        n_neighbors: int = 50,
        n_min: int = 8,
        n_min_auto: int = 15,
        prob_threshold: float = 0.6,
        frag_threshold: float = 0.1,
        merger_threshold: float = 0.4,
        logger: Optional[logging.Logger] = None
    ):
        self.n_neighbors = n_neighbors
        self.n_min = n_min
        self.n_min_auto = n_min_auto
        self.prob_threshold = prob_threshold
        self.frag_threshold = frag_threshold
        self.merger_threshold = merger_threshold
        self.logger = logger or self._setup_default_logger()

        self.nbrs = None
        self.distances = None
        self.indices = None

    def _setup_default_logger(self) -> logging.Logger:
        """Setup a default logger if none provided"""
        logger = logging.getLogger('label_propagation')
        logger.setLevel(logging.INFO)
        # Only add handler if none exist to avoid duplicates
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
        return logger

    def fit_neighbors(
        self,
        pca_embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit K-NN model on PCA embeddings.

        Parameters
        ----------
        pca_embeddings : np.ndarray
            PCA-transformed embeddings

        Returns
        -------
        distances : np.ndarray
            Distances to nearest neighbors
        indices : np.ndarray
            Indices of nearest neighbors
        """
        self.logger.info(f"Finding {self.n_neighbors} nearest neighbors...")

        self.nbrs = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm='ball_tree'
        ).fit(pca_embeddings)

        self.distances, self.indices = self.nbrs.kneighbors(pca_embeddings)

        # Set distance to self as NaN
        self.distances[:, 0] = np.nan

        return self.distances, self.indices

    def estimate_labels(
        self,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate probability labels from neighbors.

        Parameters
        ----------
        labels : np.ndarray
            Known labels (0 = unlabeled)

        Returns
        -------
        prob_labels : np.ndarray
            Probability distribution over labels
        n_labels : np.ndarray
            Count of labeled neighbors for each object
        """
        if self.indices is None:
            raise ValueError("Must call fit_neighbors() first")

        neighbor_labels = labels[self.indices]

        # Compute weights based on inverse distance, only for labeled neighbors
        weights = np.where(neighbor_labels > 0, 1. / self.distances, 0.)
        weights /= np.nansum(weights, axis=1).reshape(-1, 1)
        #print(f'W SHAPE: {weights.shape}')

        # Compute probability labels for each class
        n_classes = labels.max() + 1
        prob_labels = np.zeros([len(labels), n_classes])

        for ix in range(n_classes):
            prob_labels[:, ix] = np.nansum(
                np.where(neighbor_labels == ix, weights, 0), axis=1
            )

        # Count number of labeled neighbors
        n_labels = np.sum(neighbor_labels > 0, axis=1)

        # Zero out probability labels for objects with too few labeled neighbors
        prob_labels[n_labels < self.n_min] = 0.

        n_auto = (prob_labels > 0).any(axis=1).sum()
        self.logger.info(f"{n_auto} objects have auto-labels")
        self.logger.info(
            f"{(n_labels < self.n_min).sum()} objects have fewer than "
            f"{self.n_min} labeled neighbors"
        )
        return prob_labels, n_labels

    def iterative_propagation(
        self,
        pca_embeddings: np.ndarray,
        labels: np.ndarray,
        handle_fragmentation_separately: bool = True,
        handle_mergers_separately: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
        """
        Iteratively propagate labels with high-confidence expansion.

        Parameters
        ----------
        pca_embeddings : np.ndarray
            PCA-transformed embeddings
        labels : np.ndarray
            Initial labels (0 = unlabeled)

        Returns
        -------
        iterative_labels : np.ndarray
            Labels after adding auto-labels
        n_labels_iter : np.ndarray
            Number of labeled neighbors after iteration
        prob_labels_iter : np.ndarray
            Probability labels after iteration
        stats : dict
            Statistics about label counts at each stage
        """
        self.logger.info("\nStarting iterative label estimation...")

        # Fit neighbors if not already done
        if self.indices is None:
            self.fit_neighbors(pca_embeddings)

        # Count initial human labels
        n_human = (labels > 0).sum()
        self.logger.info(f"Human labels: {n_human}")

        # First iteration: Get initial probability labels
        prob_labels, n_labels = self.estimate_labels(labels)

        n_initial_auto = (prob_labels > 0).any(axis=1).sum()
        self.logger.info(f"Initial auto-labels: {n_initial_auto} objects have potential probabilistic labels")

        # Add high-confidence labels
        iterative_labels = labels.copy()
        n_labeled_before = (iterative_labels > 0).sum()

        # Add labels where probability > threshold and enough labeled neighbors
        additions = np.where(prob_labels[n_labels >= self.n_min_auto] > self.prob_threshold)
        
        new_labels = np.zeros_like(iterative_labels)
        new_labels[additions[0]] = additions[1]

        # Special case: Add fragmentation labels with lower threshold
        if handle_fragmentation_separately:
            self.logger.info(f"Handling fragmentation as a special case")
            new_labels[(prob_labels[:, 4] > self.frag_threshold) & (n_labels >= self.n_min_auto)] = 4
        if handle_mergers_separately:
            self.logger.info(f"Handling mergers as a special case")
            new_labels[(prob_labels[:, 3] > self.merger_threshold) & (n_labels >= self.n_min_auto)] = 3

        # Only update unlabeled objects
        iterative_labels[iterative_labels == 0] = new_labels[iterative_labels == 0]

        n_added = (iterative_labels > 0).sum() - n_labeled_before
        self.logger.info(f"Added {n_added} auto-labels in first iteration")
        self.logger.info(f"Total labels after iteration: {(iterative_labels > 0).sum()}")

        # Second iteration: Recalculate with expanded labels
        self.logger.info("\nRecalculating with expanded label set...")
        prob_labels_iter, n_labels_iter = self.estimate_labels(iterative_labels)

        n_final_auto = (prob_labels_iter > 0).any(axis=1).sum()
        self.logger.info(f"After second iteration: {n_final_auto} objects have auto-labels")

        stats = {
            'n_human': n_human,
            'n_initial_auto': n_initial_auto,
            'n_added_iteration': n_added,
            'n_total_after_iteration': (iterative_labels > 0).sum(),
            'n_final_auto': n_final_auto
        }

        return iterative_labels, n_labels_iter, prob_labels_iter, stats

    def flag_objects_for_classification(
        self,
        img_names: np.ndarray,
        n_labels: np.ndarray,
        output_file: Union[str, Path],
        n_min: Optional[int] = None
    ) -> np.ndarray:
        """
        Flag objects with insufficient labeled neighbors for manual classification.

        Parameters
        ----------
        img_names : np.ndarray
            Object identifiers
        n_labels : np.ndarray
            Count of labeled neighbors for each object
        output_file : str or Path
            Path to save CSV of flagged objects
        n_min : int, optional
            Minimum labeled neighbors threshold. If None, uses self.n_min

        Returns
        -------
        flagged_objects : np.ndarray
            Array of flagged object identifiers
        """
        import pandas as pd

        if n_min is None:
            n_min = self.n_min

        # Get objects with fewer than n_min labeled neighbors
        flagged_mask = n_labels < n_min
        flagged_objects = img_names[flagged_mask]

        # Create DataFrame
        df = pd.DataFrame({'object_id': flagged_objects})

        # Save to CSV
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        self.logger.info(f"\nSaved {len(flagged_objects)} flagged object IDs to: {output_path}")

        return flagged_objects


class SimulatedGalaxyGenerator:
    """
    Generate simulated galaxy images for testing BYOL classification.

    Creates three classes of galaxies using Sersic profiles:
    - Disk: Single exponential disk
    - Double nuclei: Two offset Sersic profiles (addition)
    - Dipole: Two Sersic profiles with different ellipticities (subtraction)

    Parameters
    ----------
    image_size : int, default=150
        Size of generated images in pixels
    random_seed : int, default=42
        Random seed for reproducibility
    logger : logging.Logger, optional
        Logger instance
    """

    def __init__(
        self,
        image_size: int = 150,
        random_seed: int = 42,
        logger: Optional[logging.Logger] = None
    ):
        if not ASTROPY_AVAILABLE:
            raise ImportError("Astropy required for simulated galaxy generation")

        self.image_size = image_size
        self.rng = np.random.RandomState(random_seed)
        self.logger = logger or self._setup_default_logger()

        self.class_names = {
            0: 'disk',
            1: 'double_nuclei',
            2: 'dipole'
        }

    def _setup_default_logger(self) -> logging.Logger:
        """Setup a default logger if none provided"""
        logger = logging.getLogger('simulated_galaxy_generator')
        logger.setLevel(logging.INFO)
        # Only add handler if none exist to avoid duplicates
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
        return logger

    def generate_disk(self) -> np.ndarray:
        """
        Generate a disk galaxy (single Sersic profile).

        Returns
        -------
        img : np.ndarray
            3-channel image (g, i, hf_i) with shape (3, H, W)
        """
        center = self.image_size / 2.0

        # Random parameters
        r_eff = self.rng.uniform(10, 30)
        ellip = self.rng.uniform(0.8, 0.9)
        theta = self.rng.uniform(0, 2 * np.pi)
        amplitude = self.rng.uniform(0.8, 1.2)

        # Create coordinate grids
        y, x = np.mgrid[0:self.image_size, 0:self.image_size]

        # Create Sersic profile
        sersic = Sersic2D(
            amplitude=amplitude,
            r_eff=r_eff,
            n=1.0,  # Exponential disk
            x_0=center,
            y_0=center,
            ellip=ellip,
            theta=theta
        )

        base_image = sersic(x, y)

        # Add noise
        noise_level = 0.01
        noise = self.rng.normal(0, noise_level, base_image.shape)

        g_band = base_image * 0.9 + noise
        i_band = base_image * 1.0 + noise
        hf_i_band = self._create_hf_image(i_band)

        return np.stack([g_band, i_band, hf_i_band], axis=0)

    def generate_double_nuclei(self) -> np.ndarray:
        """
        Generate a double nuclei galaxy (two added Sersic profiles).

        Returns
        -------
        img : np.ndarray
            3-channel image (g, i, hf_i) with shape (3, H, W)
        """
        center = self.image_size / 2.0

        # Random parameters for both components
        r_eff_1 = self.rng.uniform(10, 20)
        r_eff_2 = self.rng.uniform(10, 20)
        ellip_1 = self.rng.uniform(0.0, 0.3)
        ellip_2 = self.rng.uniform(0.0, 0.3)
        theta_1 = self.rng.uniform(0, 2 * np.pi)
        theta_2 = self.rng.uniform(0, 2 * np.pi)

        # Offset for second component (10-17 pixels)
        offset_distance = self.rng.uniform(10, 17)
        offset_angle = self.rng.uniform(0, 2 * np.pi)
        x_offset = offset_distance * np.cos(offset_angle)
        y_offset = offset_distance * np.sin(offset_angle)

        amplitude_1 = self.rng.uniform(0.8, 1.2)
        amplitude_2 = self.rng.uniform(0.8, 1.2)

        # Create coordinate grids
        y, x = np.mgrid[0:self.image_size, 0:self.image_size]

        # Create two Sersic profiles
        sersic_1 = Sersic2D(
            amplitude=amplitude_1, r_eff=r_eff_1, n=1.0,
            x_0=center, y_0=center, ellip=ellip_1, theta=theta_1
        )

        sersic_2 = Sersic2D(
            amplitude=amplitude_2, r_eff=r_eff_2, n=1.0,
            x_0=center + x_offset, y_0=center + y_offset,
            ellip=ellip_2, theta=theta_2
        )

        base_image = sersic_1(x, y) + sersic_2(x, y)

        # Add noise
        noise_level = 0.01
        noise = self.rng.normal(0, noise_level, base_image.shape)

        g_band = base_image * 0.9 + noise
        i_band = base_image * 1.0 + noise
        hf_i_band = self._create_hf_image(i_band)

        return np.stack([g_band, i_band, hf_i_band], axis=0)

    def generate_dipole(self) -> np.ndarray:
        """
        Generate a dipole galaxy (two subtracted Sersic profiles).

        Returns
        -------
        img : np.ndarray
            3-channel image (g, i, hf_i) with shape (3, H, W)
        """
        center = self.image_size / 2.0

        # Same center, r_eff, and amplitude for both components
        r_eff = self.rng.uniform(10, 30)
        amplitude = self.rng.uniform(0.8, 1.2)

        # Different ellipticities
        ellip_1 = self.rng.uniform(0.3, 0.6)
        ellip_2 = self.rng.uniform(0.6, 0.9)

        # Different position angles (offset by 45-90 degrees)
        theta_1 = self.rng.uniform(0, 2 * np.pi)
        theta_offset = self.rng.uniform(np.pi/4, np.pi/2)
        theta_2 = theta_1 + theta_offset

        # Create coordinate grids
        y, x = np.mgrid[0:self.image_size, 0:self.image_size]

        # Create two Sersic profiles
        sersic_1 = Sersic2D(
            amplitude=amplitude, r_eff=r_eff, n=1.0,
            x_0=center, y_0=center, ellip=ellip_1, theta=theta_1
        )

        sersic_2 = Sersic2D(
            amplitude=amplitude, r_eff=r_eff, n=1.0,
            x_0=center, y_0=center, ellip=ellip_2, theta=theta_2
        )

        # Subtract to create dipole
        base_image = sersic_1(x, y) - sersic_2(x, y)
        base_image = base_image - np.min(base_image) + 0.1

        # Add noise
        noise_level = 0.01
        noise = self.rng.normal(0, noise_level, base_image.shape)

        g_band = base_image * 0.9 + noise
        i_band = base_image * 1.0 + noise
        hf_i_band = self._create_hf_image(i_band)

        return np.stack([g_band, i_band, hf_i_band], axis=0)

    def _create_hf_image(self, i_band: np.ndarray) -> np.ndarray:
        """Create high-frequency residual image using starlet transform or fallback"""
        if STARLET_AVAILABLE:
            try:
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
                        abs(wt[ix]), 10.,
                        err=np.median(err_samples),
                        segmentation_map=True,
                        deblend_cont=1.
                    )

                    # Keep only central source features
                    sidx = segmap[segmap.shape[0]//2, segmap.shape[0]//2]
                    segmap_l.append(segmap)
                    im_recon.append(np.where(segmap == sidx, wt[ix], 0.))

                # Reconstruct and create high-frequency residual
                im_recon = imstats.inverse_starlet_transform(im_recon, gen2=True)
                hf_image = i_band - im_recon
                hf_image = hf_image - ndimage.median_filter(hf_image, size=20)

                return hf_image

            except Exception as e:
                self.logger.warning(f"Starlet transform failed: {e}, using fallback")

        # Fallback: simple high-pass filtering
        from scipy.ndimage import gaussian_filter
        smoothed = gaussian_filter(i_band, sigma=5.0)
        hf_image = i_band - smoothed
        hf_image = hf_image - ndimage.median_filter(hf_image, size=20)
        return hf_image

    def generate_dataset(
        self,
        n_per_class: int = 300
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate complete dataset with all three galaxy classes.

        Parameters
        ----------
        n_per_class : int, default=300
            Number of galaxies to generate per class

        Returns
        -------
        images : np.ndarray
            Generated images with shape (N, 3, H, W)
        img_names : np.ndarray
            Object identifiers
        true_labels : np.ndarray
            Ground truth class labels
        """
        self.logger.info(f"Generating {n_per_class * 3} simulated galaxies...")

        imgs = []
        img_names = []
        true_labels = []

        for class_id in range(3):
            self.logger.info(f"Generating class {class_id} ({self.class_names[class_id]})...")

            for i in tqdm(range(n_per_class), desc=f"Class {class_id}"):
                if class_id == 0:
                    img = self.generate_disk()
                elif class_id == 1:
                    img = self.generate_double_nuclei()
                else:
                    img = self.generate_dipole()

                imgs.append(img)
                img_names.append(f"{self.class_names[class_id]}_{i:04d}")
                true_labels.append(class_id)

        images = np.array(imgs)
        img_names = np.array(img_names)
        true_labels = np.array(true_labels)

        self.logger.info(f"Generated {len(images)} images with shape {images.shape}")
        self.logger.info(f"Class distribution: {np.bincount(true_labels)}")

        return images, img_names, true_labels


def load_merian_images(
    data_path: Union[str, Path],
    logger: Optional[logging.Logger] = None,    
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Merian galaxy images from pickle files.

    Loads g-band, i-band, and high-frequency i-band images from the standard
    pieridae output format.

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
    img_names : np.ndarray
        Object identifiers
    """
    if logger is None:
        logger = logging.getLogger('load_merian_images')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        logger.addHandler(handler)

    data_path = Path(data_path)
    pattern = f"{data_path}/M*/*i_results.pkl"
    filenames = glob.glob(pattern)

    if not filenames:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")

    logger.info(f"Found {len(filenames)} image files")

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
                        img.append(xf['hf_image'])
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

    return images, img_names


def compute_classification_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_names: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """
    Compute classification metrics (purity, completeness, etc.).

    Parameters
    ----------
    true_labels : np.ndarray
        Ground truth labels
    predicted_labels : np.ndarray
        Predicted labels
    class_names : dict, optional
        Mapping from class IDs to names

    Returns
    -------
    metrics : dict
        Dictionary containing classification metrics
    """
    n_samples = len(true_labels)
    n_classes = len(np.unique(true_labels))

    if class_names is None:
        class_names = {i: f'class_{i}' for i in range(n_classes)}

    # Confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Purity (for each predicted cluster, fraction of most common true class)
    cluster_purities = []
    for cluster_id in range(n_classes):
        cluster_mask = predicted_labels == cluster_id
        if np.sum(cluster_mask) > 0:
            true_labels_in_cluster = true_labels[cluster_mask]
            most_common_count = np.max(np.bincount(true_labels_in_cluster))
            purity = most_common_count / np.sum(cluster_mask)
            cluster_purities.append(purity)
        else:
            cluster_purities.append(0.0)

    overall_purity = np.sum([
        np.sum(predicted_labels == i) * cluster_purities[i]
        for i in range(n_classes)
    ]) / n_samples

    # Completeness (for each true class, fraction in most common predicted cluster)
    class_completeness = []
    for class_id in range(n_classes):
        class_mask = true_labels == class_id
        if np.sum(class_mask) > 0:
            pred_labels_in_class = predicted_labels[class_mask]
            most_common_cluster = np.argmax(np.bincount(pred_labels_in_class))
            completeness = np.sum(
                pred_labels_in_class == most_common_cluster
            ) / np.sum(class_mask)
            class_completeness.append(completeness)
        else:
            class_completeness.append(0.0)

    overall_completeness = np.mean(class_completeness)

    # Classification report
    class_report = classification_report(
        true_labels,
        predicted_labels,
        target_names=[class_names[i] for i in range(n_classes)],
        output_dict=True
    )

    metrics = {
        'overall_purity': float(overall_purity),
        'overall_completeness': float(overall_completeness),
        'cluster_purities': {
            class_names[i]: float(p) for i, p in enumerate(cluster_purities)
        },
        'class_completeness': {
            class_names[i]: float(c) for i, c in enumerate(class_completeness)
        },
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report,
        'n_samples': n_samples,
        'n_classes': n_classes,
        'class_names': class_names
    }

    return metrics
