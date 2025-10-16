#!/usr/bin/env python3
"""
DINOv2 Classifier - Galaxy Classification using DINOv2 Embeddings

This script ports the dino_analysis_notebook.ipynb workflow to a production script.
It extracts DINOv2 embeddings and performs k-NN label propagation for galaxy classification.
Optionally supports fine-tuning of the DINOv2 model with a classification head.

Usage:
    # Basic usage (pre-trained model only)
    python dino_classifier.py --config dino_config.yaml
    python dino_classifier.py --mode full
    python dino_classifier.py --mode analyze  # Skip embedding extraction

    # Fine-tuning examples
    python dino_classifier.py --finetune --epochs 20
    python dino_classifier.py --finetune --freeze-backbone --epochs 10
    python dino_classifier.py --config dino_config.yaml --finetune
"""

import os
import sys
import argparse
import logging
import pickle
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
import yaml
from tqdm import tqdm
from PIL import Image

# HuggingFace transformers
from transformers import AutoImageProcessor, AutoModel

# ML libraries
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Plotting
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib import colors
    from scipy import ndimage
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib not available, plotting disabled")


def setup_device():
    """Setup device with MPS support for Apple Silicon"""
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✅ Using Apple Silicon GPU (MPS): {device}")
        torch.mps.set_per_process_memory_fraction(0.8)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Using NVIDIA GPU: {device}")
    else:
        device = torch.device('cpu')
        print(f"Using CPU: {device}")
        torch.set_num_threads(os.cpu_count())
    return device


def prepare_images_for_dinov2(images_batch):
    """Convert numpy arrays to PIL images for DINOv2 processor"""
    pil_images = []
    for img in images_batch:
        # Transpose from (3, H, W) to (H, W, 3)
        img_hwc = np.transpose(img, (1, 2, 0))

        # Normalize to 0-255 range
        if img_hwc.max() <= 1.0:
            img_hwc = (img_hwc * 255).astype(np.uint8)
        else:
            img_hwc = img_hwc.astype(np.float32)
            img_min, img_max = img_hwc.min(), img_hwc.max()
            if img_max > img_min:
                img_hwc = ((img_hwc - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img_hwc = np.zeros_like(img_hwc, dtype=np.uint8)

        pil_img = Image.fromarray(img_hwc)
        pil_images.append(pil_img)

    return pil_images


class DINOv2WithHead(nn.Module):
    """DINOv2 model with classification head for fine-tuning"""

    def __init__(self, backbone, num_classes: int, hidden_dim: int = 256, dropout: float = 0.1, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Get embedding dimension from backbone
        # DINOv2 models have different dimensions: small=384, base=768, large=1024, giant=1536
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.backbone(pixel_values=dummy_input)
            embedding_dim = dummy_output.last_hidden_state.shape[-1]

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, pixel_values, return_embedding=False):
        """Forward pass through model

        Args:
            pixel_values: Input images
            return_embedding: If True, return both logits and embeddings

        Returns:
            If return_embedding=False: logits
            If return_embedding=True: (logits, embeddings)
        """
        # Get embeddings from backbone
        outputs = self.backbone(pixel_values=pixel_values)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token

        # Get logits from classifier
        logits = self.classifier(embeddings)

        if return_embedding:
            return logits, embeddings
        return logits


class DINOv2Classifier:
    """DINOv2-based galaxy classifier with k-NN label propagation"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = setup_device()

        # Setup paths
        self.data_path = Path(config['data']['input_path'])
        self.output_path = Path(config['data']['output_path'])
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = self._setup_logging()

        # Initialize components
        self.processor = None
        self.model = None
        self.is_finetuned = False  # Track if model has classification head
        self.images = None
        self.img_names = None
        self.labels = None
        self.embeddings = None
        self.reduction_results = None

        # Classification results
        self.prob_labels = None
        self.iterative_labels = None
        self.n_labels = None

        self.logger.info(f"Initialized DINOv2 Classifier on device: {self.device}")
        self.logger.info(f"Model: {config['model']['model_name']}")
        self.logger.info(f"Fine-tuning enabled: {config['model'].get('enable_finetuning', False)}")
        self.logger.info(f"Output path: {self.output_path}")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('dino_classifier')
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        log_file = self.output_path / 'dino_classifier.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load image data from pickle files"""
        self.logger.info(f"Loading image data from: {self.data_path}")

        pattern = f"{self.data_path}/M*/*i_results.pkl"
        filenames = glob.glob(pattern)

        if not filenames:
            raise FileNotFoundError(f"No files found matching pattern: {pattern}")

        self.logger.info(f"Found {len(filenames)} image files")

        imgs = []
        img_names = []

        for fname in tqdm(filenames, desc="Loading images"):
            img = []
            for band in 'gi':
                current_filename = fname.replace('_i_', f'_{band}_')

                try:
                    with open(current_filename, 'rb') as f:
                        xf = pickle.load(f)
                        img.append(xf['image'])
                        if band == 'i':
                            img.append(xf['hf_image'])
                except FileNotFoundError:
                    self.logger.warning(f"File not found: {current_filename}")
                    continue

            if len(img) == 3:
                imgs.append(np.array(img))
                img_names.append(Path(fname).parent.name)

        self.images = np.array(imgs)
        self.img_names = np.array(img_names)

        self.logger.info(f"Loaded {len(self.images)} images with shape: {self.images.shape}")
        return self.images, self.img_names

    def setup_model(self) -> Tuple[AutoImageProcessor, AutoModel]:
        """Setup DINOv2 model from HuggingFace"""
        self.logger.info(f"Setting up DINOv2 model: {self.config['model']['model_name']}...")

        self.processor = AutoImageProcessor.from_pretrained(self.config['model']['model_name'])
        backbone = AutoModel.from_pretrained(self.config['model']['model_name'])

        # Check if fine-tuning is enabled
        enable_finetuning = self.config['model'].get('enable_finetuning', False)

        if enable_finetuning:
            self.logger.info("Fine-tuning mode enabled - adding classification head")

            # Get fine-tuning config
            ft_config = self.config['model'].get('finetuning', {})
            num_classes = 6  # 0-5: unclassified, undisturbed, ambiguous, merger, fragmentation, artifact
            hidden_dim = ft_config.get('hidden_dim', 256)
            dropout = ft_config.get('dropout', 0.1)
            freeze_backbone = ft_config.get('freeze_backbone', True)

            # Create model with classification head
            self.model = DINOv2WithHead(
                backbone=backbone,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                dropout=dropout,
                freeze_backbone=freeze_backbone
            )
            self.is_finetuned = True

            self.logger.info(f"Classification head added (freeze_backbone={freeze_backbone})")
        else:
            # Use backbone only (no fine-tuning)
            self.model = backbone
            self.is_finetuned = False
            self.logger.info("Using pre-trained model (no fine-tuning)")

        self.model = self.model.to(self.device)
        self.model.eval()

        self.logger.info(f"DINOv2 model setup complete on {self.device}")
        return self.processor, self.model

    def train_model(self) -> None:
        """Fine-tune the model on labeled data"""
        if not self.is_finetuned:
            raise ValueError("Fine-tuning not enabled. Set enable_finetuning=True in config")

        if self.labels is None:
            raise ValueError("Labels not loaded. Cannot train without labels.")

        self.logger.info("Starting fine-tuning...")

        # Get training config
        ft_config = self.config['model'].get('finetuning', {})
        learning_rate = ft_config.get('learning_rate', 1e-5)
        num_epochs = ft_config.get('num_epochs', 10)
        batch_size = min(32, self.config['inference']['batch_size'])

        # Adjust for device
        if self.device.type == 'mps':
            batch_size = min(16, batch_size)
        elif self.device.type == 'cpu':
            batch_size = min(8, batch_size)

        # Get labeled data only (exclude class 0 - unclassified)
        labeled_mask = self.labels > 0
        labeled_images = self.images[labeled_mask]
        labeled_targets = self.labels[labeled_mask]

        self.logger.info(f"Training on {len(labeled_images)} labeled samples")
        self.logger.info(f"Batch size: {batch_size}, Learning rate: {learning_rate}, Epochs: {num_epochs}")

        # Setup optimizer and loss
        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        self.model.train()
        num_batches = (len(labeled_images) + batch_size - 1) // batch_size

        for epoch in range(num_epochs):
            total_loss = 0
            correct = 0
            total = 0

            # Shuffle data each epoch
            perm = np.random.permutation(len(labeled_images))
            shuffled_images = labeled_images[perm]
            shuffled_targets = labeled_targets[perm]

            pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}")
            for i in pbar:
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(shuffled_images))

                # Prepare batch
                batch_numpy = shuffled_images[start_idx:end_idx]
                batch_labels = torch.tensor(shuffled_targets[start_idx:end_idx], dtype=torch.long).to(self.device)

                pil_images = prepare_images_for_dinov2(batch_numpy)

                try:
                    # Process images
                    inputs = self.processor(images=pil_images, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    # Forward pass
                    logits = self.model(inputs['pixel_values'])

                    # Compute loss
                    loss = criterion(logits, batch_labels)

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Track metrics
                    total_loss += loss.item()
                    _, predicted = torch.max(logits.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()

                    # Update progress bar
                    pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})

                except RuntimeError as e:
                    if "MPS" in str(e) and self.device.type == 'mps':
                        self.logger.warning(f"MPS error during training, falling back to CPU")
                        self.model = self.model.cpu()
                        self.device = torch.device('cpu')
                        # Continue with next batch on CPU
                        continue
                    else:
                        raise e

            # Epoch summary
            avg_loss = total_loss / num_batches
            accuracy = 100 * correct / total
            self.logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

            # Save checkpoint
            if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
                checkpoint_path = self.output_path / f'finetuned_model_epoch{epoch+1}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'accuracy': accuracy,
                    'config': self.config
                }, checkpoint_path)
                self.logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Save final model
        final_model_path = self.output_path / 'finetuned_model_final.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'final_accuracy': accuracy
        }, final_model_path)
        self.logger.info(f"Fine-tuning complete. Final model saved: {final_model_path}")

        # Set back to eval mode
        self.model.eval()

    def extract_embeddings(self) -> np.ndarray:
        """Extract embeddings from DINOv2 model"""
        self.logger.info("Extracting embeddings from images...")

        if self.model is None or self.processor is None:
            self.setup_model()

        self.model.eval()
        all_embeddings = []

        batch_size = self.config['inference']['batch_size']

        # Adjust batch size for device
        if self.device.type == 'mps':
            batch_size = min(32, batch_size)
        elif self.device.type == 'cpu':
            batch_size = min(16, batch_size)

        num_batches = (len(self.images) + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Extracting embeddings"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(self.images))

                batch_numpy = self.images[start_idx:end_idx]
                pil_images = prepare_images_for_dinov2(batch_numpy)

                try:
                    inputs = self.processor(images=pil_images, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    # Handle fine-tuned vs regular model
                    if self.is_finetuned:
                        # Fine-tuned model with classification head
                        _, cls_embeddings = self.model(inputs['pixel_values'], return_embedding=True)
                    else:
                        # Regular DINOv2 backbone
                        outputs = self.model(**inputs)
                        cls_embeddings = outputs.last_hidden_state[:, 0, :]

                    all_embeddings.append(cls_embeddings.cpu().numpy())

                except RuntimeError as e:
                    if "MPS" in str(e) and self.device.type == 'mps':
                        self.logger.warning(f"MPS error, falling back to CPU for batch {i}")
                        inputs = {k: v.cpu() for k, v in inputs.items()}
                        self.model = self.model.cpu()
                        self.device = torch.device('cpu')

                        if self.is_finetuned:
                            _, cls_embeddings = self.model(inputs['pixel_values'], return_embedding=True)
                        else:
                            outputs = self.model(**inputs)
                            cls_embeddings = outputs.last_hidden_state[:, 0, :]

                        all_embeddings.append(cls_embeddings.cpu().numpy())
                    else:
                        raise e

        self.embeddings = np.vstack(all_embeddings)

        # Save embeddings
        embeddings_path = self.output_path / 'embeddings.npy'
        np.save(embeddings_path, self.embeddings)

        self.logger.info(f"Extracted embeddings shape: {self.embeddings.shape}")
        return self.embeddings

    def compute_pca_umap(self) -> Dict[str, Any]:
        """Apply PCA and UMAP to embeddings"""
        self.logger.info("Computing PCA and UMAP...")

        # Clean data
        embeddings_clean = np.nan_to_num(self.embeddings, nan=0.0, posinf=0.0, neginf=0.0)

        # Standardization
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings_clean)

        # PCA
        pca_components = self.config['analysis']['pca_components']
        if pca_components is None:
            pca_full = PCA()
            pca_full.fit(embeddings_scaled)
            cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
            threshold = self.config['analysis']['explained_variance_threshold']
            pca_components = np.argmax(cumsum_var >= threshold) + 1

        max_components = min(embeddings_scaled.shape[0] - 1, embeddings_scaled.shape[1])
        pca_components = min(pca_components, max_components)

        pca = PCA(n_components=pca_components, random_state=42)
        embeddings_pca = pca.fit_transform(embeddings_scaled)

        # UMAP
        n_neighbors = min(self.config['analysis']['n_neighbors'], len(embeddings_pca) - 1)

        umap_reducer = umap.UMAP(
            n_components=self.config['analysis']['umap_components'],
            n_neighbors=n_neighbors,
            min_dist=self.config['analysis']['min_dist'],
            metric=self.config['analysis']['metric'],
            random_state=42,
            verbose=True
        )

        embeddings_umap = umap_reducer.fit_transform(embeddings_pca)

        self.reduction_results = {
            'scaler': scaler,
            'pca': pca,
            'umap': umap_reducer,
            'embeddings_original': embeddings_clean,
            'embeddings_pca': embeddings_pca,
            'embeddings_umap': embeddings_umap
        }

        # Save results
        results_path = self.output_path / 'dimensionality_reduction_results.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(self.reduction_results, f)

        self.logger.info(f"PCA components: {pca_components}")
        self.logger.info(f"Explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")
        self.logger.info(f"UMAP embedding shape: {embeddings_umap.shape}")

        return self.reduction_results

    def load_labels(self) -> Optional[np.ndarray]:
        """Load classification labels"""
        label_file = Path(self.config.get('labels', {}).get('classifications_file',
                                                            './classifications_kadofong_current.csv'))

        if label_file.exists() and self.img_names is not None:
            try:
                mergers = pd.read_csv(label_file, index_col=0)
                self.labels = mergers.reindex(self.img_names)
                self.labels = self.labels.replace(np.nan, 0).values.flatten().astype(int)

                self.logger.info(f"Loaded classification labels: {len(self.labels)} objects")

                # Print label distribution
                unique, counts = np.unique(self.labels, return_counts=True)
                label_meanings = {
                    0: "unclassified", 1: "undisturbed", 2: "ambiguous",
                    3: "merger", 4: "fragmentation", 5: "artifact"
                }

                self.logger.info("Label distribution:")
                for label_val, count in zip(unique, counts):
                    meaning = label_meanings.get(label_val, f"unknown_{label_val}")
                    self.logger.info(f"   {label_val} ({meaning}): {count} objects")

            except Exception as e:
                self.logger.warning(f"Could not load labels: {e}")
                self.labels = None
        else:
            self.logger.warning(f"Label file not found: {label_file}")
            self.labels = None

        return self.labels

    def compute_knn_label_propagation(self, n_neighbors: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Compute k-NN label propagation"""
        self.logger.info("Computing k-NN label propagation...")

        if self.labels is None:
            raise ValueError("Labels not loaded. Cannot perform label propagation.")

        pca_embeddings = self.reduction_results['embeddings_pca']

        # Fit k-NN
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(pca_embeddings)
        distances, indices = nbrs.kneighbors(pca_embeddings)
        distances[:, 0] = np.nan  # Ignore self

        # Get neighbor labels
        neighbor_labels = self.labels[indices]

        # Compute weights
        weights = np.where(neighbor_labels > 0, 1. / distances, 0.)
        weights /= np.nansum(weights, axis=1).reshape(-1, 1)

        # Compute probability distribution
        self.prob_labels = np.zeros([pca_embeddings.shape[0], self.labels.max() + 1])

        for ix in range(self.labels.max() + 1):
            self.prob_labels[:, ix] = np.nansum(np.where(neighbor_labels == ix, weights, 0), axis=1)

        self.n_labels = np.sum(neighbor_labels > 0, axis=1)

        # Get minimum from config
        n_min = self.config.get('labels', {}).get('minimum_labeled_neighbors', 8)
        self.prob_labels[self.n_labels < n_min] = 0.

        n_auto_labeled = (self.prob_labels > 0).any(axis=1).sum()
        self.logger.info(f"{n_auto_labeled} galaxies have auto-labels")

        return self.prob_labels, self.n_labels

    def iterative_label_propagation(self) -> np.ndarray:
        """Perform iterative label propagation"""
        self.logger.info("Performing iterative label propagation...")

        self.iterative_labels = self.labels.copy()
        n_labeled = (self.iterative_labels > 0).sum()

        n_min = self.config.get('labels', {}).get('minimum_labeled_neighbors', 8)

        # Add high-confidence labels
        additions = np.where(self.prob_labels[self.n_labels >= n_min] > 0.6)
        new_labels = np.zeros_like(self.iterative_labels)
        new_labels[additions[0]] = additions[1]

        # Special handling for fragmentation
        new_labels[(self.prob_labels[:, 4] > 0.1) & (self.n_labels >= n_min)] = 4

        self.iterative_labels[self.iterative_labels == 0] = new_labels[self.iterative_labels == 0]
        n_new = (self.iterative_labels > 0).sum() - n_labeled

        self.logger.info(f"{(self.labels > 0).sum()} human labels")
        self.logger.info(f"{n_new} auto-labels added, {(self.iterative_labels > 0).sum()} labels total")

        # Re-compute probabilities
        self.logger.info("Re-computing probabilities with propagated labels...")
        pca_embeddings = self.reduction_results['embeddings_pca']

        nbrs = NearestNeighbors(n_neighbors=50, algorithm='ball_tree').fit(pca_embeddings)
        distances, indices = nbrs.kneighbors(pca_embeddings)
        distances[:, 0] = np.nan

        neighbor_labels = self.iterative_labels[indices]
        weights = np.where(neighbor_labels > 0, 1. / distances, 0.)
        weights /= np.nansum(weights, axis=1).reshape(-1, 1)

        self.prob_labels = np.zeros([pca_embeddings.shape[0], self.labels.max() + 1])
        for ix in range(self.labels.max() + 1):
            self.prob_labels[:, ix] = np.nansum(np.where(neighbor_labels == ix, weights, 0), axis=1)

        self.n_labels = np.sum(neighbor_labels > 0, axis=1)
        self.prob_labels[self.n_labels < n_min] = 0.

        n_final = (self.prob_labels > 0).any(axis=1).sum()
        self.logger.info(f"{n_final} galaxies have auto-labels (after iteration)")

        return self.iterative_labels

    def classify_mergers(self) -> np.ndarray:
        """Classify objects as mergers"""
        self.logger.info("Classifying mergers...")

        is_merger = (
            ((self.prob_labels[:, 2] + self.prob_labels[:, 3]) > self.prob_labels[:, 1]) &
            (self.prob_labels[:, 4] < 0.1)
        )

        self.logger.info(f"Classified {is_merger.sum()} objects as mergers")

        return is_merger

    def create_visualizations(self, is_merger: np.ndarray) -> None:
        """Create classification visualizations"""
        if not PLOTTING_AVAILABLE:
            self.logger.warning("Matplotlib not available, skipping visualizations")
            return

        self.logger.info("Creating visualizations...")

        # Histogram: merger classification vs human labels
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        bins = np.arange(0.5, 5.5, 1)
        ax.hist(self.labels[~is_merger], bins=bins, density=True, lw=3, alpha=0.3,
               label='DINOv2+PCA = not merger')
        ax.hist(self.labels[is_merger], bins=bins, density=True, lw=3, alpha=0.3,
               label='DINOv2+PCA = is merger')

        ax.set_xticklabels(['', 'undisturbed', 'ambiguous', 'merger', 'fragmented'], rotation=35)
        ax.set_xlabel('Human labels')
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_path / 'merger_classification.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info("Visualizations saved")

    def save_results(self, is_merger: np.ndarray) -> None:
        """Save classification results"""
        self.logger.info("Saving results...")

        # Save embeddings
        embeddings_path = self.output_path / 'embeddings.npy'
        np.save(embeddings_path, self.embeddings)

        # Save image info
        image_info = {
            'img_names': self.img_names,
            'labels': self.labels,
            'num_images': len(self.images)
        }
        info_path = self.output_path / 'image_info.pkl'
        with open(info_path, 'wb') as f:
            pickle.dump(image_info, f)

        # Save classification results
        results = {
            'prob_labels': self.prob_labels,
            'iterative_labels': self.iterative_labels,
            'is_merger': is_merger,
            'n_labels': self.n_labels
        }
        results_path = self.output_path / 'classification_results.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)

        # Save CSV
        if self.img_names is not None:
            df = pd.DataFrame({
                'object_id': self.img_names,
                'p_unclassified': self.prob_labels[:, 0],
                'p_undisturbed': self.prob_labels[:, 1],
                'p_ambiguous': self.prob_labels[:, 2],
                'p_merger': self.prob_labels[:, 3],
                'p_fragmentation': self.prob_labels[:, 4],
                'p_artifact': self.prob_labels[:, 5] if self.prob_labels.shape[1] > 5 else 0,
                'iterative_label': self.iterative_labels,
                'is_merger': is_merger,
                'original_label': self.labels if self.labels is not None else 0
            })

            csv_path = self.output_path / 'classification_results.csv'
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Saved CSV to: {csv_path}")

        self.logger.info(f"Results saved to: {self.output_path}")

    def run_full_pipeline(self) -> None:
        """Run the complete classification pipeline"""
        self.logger.info("\n" + "="*60)
        self.logger.info("STARTING DINOV2 CLASSIFICATION PIPELINE")
        self.logger.info("="*60 + "\n")

        # Load data
        self.load_data()

        # Load labels (needed for fine-tuning)
        self.load_labels()

        # Fine-tune model if enabled
        if self.config['model'].get('enable_finetuning', False):
            if self.labels is not None:
                self.train_model()
            else:
                self.logger.warning("Fine-tuning enabled but no labels available. Skipping training.")

        # Extract embeddings
        self.extract_embeddings()

        # Dimensionality reduction
        self.compute_pca_umap()

        if self.labels is not None:
            # k-NN label propagation
            self.compute_knn_label_propagation()

            # Iterative propagation
            self.iterative_label_propagation()

            # Classify mergers
            is_merger = self.classify_mergers()

            # Visualizations
            self.create_visualizations(is_merger)
        else:
            self.logger.warning("No labels available, skipping classification steps")
            is_merger = np.zeros(len(self.embeddings), dtype=bool)

        # Save results
        self.save_results(is_merger)

        self.logger.info("\n" + "="*60)
        self.logger.info("PIPELINE COMPLETE")
        self.logger.info("="*60 + "\n")

    def run_analysis_only(self) -> None:
        """Run analysis on pre-computed embeddings"""
        self.logger.info("\n" + "="*60)
        self.logger.info("STARTING ANALYSIS (USING EXISTING EMBEDDINGS)")
        self.logger.info("="*60 + "\n")

        # Load embeddings
        embeddings_path = self.output_path / 'embeddings.npy'
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

        self.embeddings = np.load(embeddings_path)
        self.logger.info(f"Loaded embeddings shape: {self.embeddings.shape}")

        # Load image info
        info_path = self.output_path / 'image_info.pkl'
        if info_path.exists():
            with open(info_path, 'rb') as f:
                info = pickle.load(f)
                self.img_names = info['img_names']
                self.labels = info.get('labels', None)

        # If no labels in pickle, try CSV
        if self.labels is None:
            self.load_labels()

        # Dimensionality reduction
        self.compute_pca_umap()

        if self.labels is not None:
            # k-NN label propagation
            self.compute_knn_label_propagation()

            # Iterative propagation
            self.iterative_label_propagation()

            # Classify mergers
            is_merger = self.classify_mergers()

            # Visualizations
            self.create_visualizations(is_merger)
        else:
            self.logger.warning("No labels available, skipping classification steps")
            is_merger = np.zeros(len(self.embeddings), dtype=bool)

        # Save results
        self.save_results(is_merger)

        self.logger.info("\n" + "="*60)
        self.logger.info("ANALYSIS COMPLETE")
        self.logger.info("="*60 + "\n")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='DINOv2 Galaxy Classifier')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--mode', type=str, choices=['full', 'analyze'], default='full',
                       help='full: extract embeddings + analyze, analyze: use existing embeddings')
    parser.add_argument('--data-path', type=str, help='Path to input data')
    parser.add_argument('--output-path', type=str, help='Path to output directory')
    parser.add_argument('--model-name', type=str, help='DINOv2 model name')
    parser.add_argument('--finetune', action='store_true', help='Enable fine-tuning')
    parser.add_argument('--freeze-backbone', action='store_true', help='Freeze backbone during fine-tuning')
    parser.add_argument('--epochs', type=int, help='Number of fine-tuning epochs')

    args = parser.parse_args()

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        default_config_path = Path(__file__).parent / 'dino_config.yaml'
        if default_config_path.exists():
            config = load_config(str(default_config_path))
        else:
            raise FileNotFoundError("No config file found. Please provide --config or create dino_config.yaml")

    # Override config with command line arguments
    if args.data_path:
        config['data']['input_path'] = args.data_path
    if args.output_path:
        config['data']['output_path'] = args.output_path
    if args.model_name:
        config['model']['model_name'] = args.model_name
    if args.finetune:
        config['model']['enable_finetuning'] = True
    if args.freeze_backbone:
        if 'finetuning' not in config['model']:
            config['model']['finetuning'] = {}
        config['model']['finetuning']['freeze_backbone'] = True
    if args.epochs:
        if 'finetuning' not in config['model']:
            config['model']['finetuning'] = {}
        config['model']['finetuning']['num_epochs'] = args.epochs

    # Convert paths
    config['data']['input_path'] = Path(config['data']['input_path'])
    config['data']['output_path'] = Path(config['data']['output_path'])

    # Initialize classifier
    classifier = DINOv2Classifier(config)

    # Run pipeline
    try:
        if args.mode == 'full':
            classifier.run_full_pipeline()
        elif args.mode == 'analyze':
            classifier.run_analysis_only()

        print("\n✅ Classification completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during classification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
