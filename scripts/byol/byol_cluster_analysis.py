#!/usr/bin/env python3
"""
BYOL Analysis for Slurm Cluster
Converts notebook analysis to production cluster script for Tiger at Princeton
"""

import os
import sys
import argparse
import logging
import pickle
import glob
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import models, transforms
from tqdm import tqdm
import yaml

# ML libraries
from byol_pytorch import BYOL
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Plotting (optional for cluster)
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib not available, plotting disabled")


class BYOLClusterAnalysis:
    """Main class for BYOL analysis on cluster"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Setup paths first (needed for logging)
        self.data_path = Path(config['data']['input_path'])
        self.output_path = Path(config['data']['output_path'])
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Setup logging after output directory is created
        self.logger = self._setup_logging()

        # Initialize model components
        self.learner = None
        self.images = None
        self.img_names = None
        self.embeddings = None

        self.logger.info(f"Initialized BYOL analysis on device: {self.device}")
        self.logger.info(f"Output path: {self.output_path}")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('byol_analysis')
        logger.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        log_file = self.config['data']['output_path'] / 'byol_analysis.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load image data from pickle files"""
        self.logger.info("Loading image data...")

        pattern = str(self.data_path / "starlet/starbursts_v0/M*/*i_results.pkl")
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

            if len(img) == 3:  # Only add if we have all bands
                imgs.append(np.array(img))
                img_names.append(Path(fname).parent.name)

        self.images = np.array(imgs)
        self.img_names = np.array(img_names)

        self.logger.info(f"Loaded {len(self.images)} images with shape {self.images.shape}")
        return self.images, self.img_names

    def setup_model(self) -> BYOL:
        """Setup BYOL model with transforms"""
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

        # Base model
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

        self.logger.info("BYOL model setup complete")
        return self.learner

    def sample_unlabelled_images(self) -> torch.Tensor:
        """Sample random batch of images for training"""
        indices = np.random.permutation(len(self.images))
        batch_size = min(self.config['training']['batch_size'], len(indices))
        return torch.tensor(
            self.images[indices[:batch_size]],
            dtype=torch.float32
        ).to(self.device)

    def train_model(self) -> None:
        """Train BYOL model"""
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
        if checkpoint_path.exists() and self.config['training']['resume']:
            self.logger.info("Loading checkpoint...")
            try:
                # Try with weights_only=False for our trusted checkpoint
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
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

        for epoch in tqdm(range(start_epoch, num_epochs), desc="Training BYOL"):
            images = self.sample_unlabelled_images()
            loss = self.learner(images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.learner.update_moving_average()

            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")

            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.learner.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'config': self.config
                }
                torch.save(checkpoint, checkpoint_path)
                self.logger.info(f"Checkpoint saved at epoch {epoch}")

        # Save final model
        final_model_path = self.output_path / 'byol_final_model.pt'
        torch.save({
            'model_state_dict': self.learner.state_dict(),
            'config': self.config,
            'training_complete': True
        }, final_model_path)

        self.logger.info("Training completed successfully")

    def extract_embeddings(self) -> np.ndarray:
        """Extract embeddings from trained model"""
        self.logger.info("Extracting embeddings...")

        if self.learner is None:
            self.logger.info("Loading trained model...")
            model_path = self.output_path / 'byol_final_model.pt'
            if not model_path.exists():
                raise FileNotFoundError(f"Trained model not found: {model_path}")

            self.setup_model()
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.learner.load_state_dict(checkpoint['model_state_dict'])

        self.learner.eval()
        all_embeddings = []

        batch_size = self.config['inference']['batch_size']
        num_batches = (len(self.images) + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Extracting embeddings"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(self.images))

                batch = torch.tensor(
                    self.images[start_idx:end_idx],
                    dtype=torch.float32
                ).to(self.device)

                _, embeddings = self.learner(batch, return_embedding=True)
                all_embeddings.append(embeddings.cpu().numpy())

        self.embeddings = np.vstack(all_embeddings)

        # Save embeddings
        embeddings_path = self.output_path / 'embeddings.npy'
        np.save(embeddings_path, self.embeddings)

        self.logger.info(f"Extracted embeddings shape: {self.embeddings.shape}")
        return self.embeddings

    def compute_pca_umap(self) -> Dict[str, Any]:
        """Apply PCA and UMAP to embeddings"""
        self.logger.info("Computing PCA and UMAP...")

        if self.embeddings is None:
            self.logger.info("Loading embeddings...")
            embeddings_path = self.output_path / 'embeddings.npy'
            self.embeddings = np.load(embeddings_path)

        # Clean data
        embeddings_clean = np.nan_to_num(
            self.embeddings, nan=0.0, posinf=0.0, neginf=0.0
        )

        # Standardization
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings_clean)

        # PCA
        pca_components = self.config['analysis']['pca_components']
        if pca_components is None:
            # Auto-determine based on variance threshold
            pca_full = PCA()
            pca_full.fit(embeddings_scaled)
            cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
            threshold = self.config['analysis']['explained_variance_threshold']
            pca_components = np.argmax(cumsum_var >= threshold) + 1

        max_components = min(
            embeddings_scaled.shape[0] - 1,
            embeddings_scaled.shape[1]
        )
        pca_components = min(pca_components, max_components)

        pca = PCA(n_components=pca_components, random_state=42)
        embeddings_pca = pca.fit_transform(embeddings_scaled)

        # UMAP
        n_neighbors = min(
            self.config['analysis']['n_neighbors'],
            len(embeddings_pca) - 1
        )

        umap_reducer = umap.UMAP(
            n_components=self.config['analysis']['umap_components'],
            n_neighbors=n_neighbors,
            min_dist=self.config['analysis']['min_dist'],
            metric=self.config['analysis']['metric'],
            random_state=42,
            verbose=True
        )

        embeddings_umap = umap_reducer.fit_transform(embeddings_pca)

        # Prepare results
        result = {
            'scaler': scaler,
            'pca': pca,
            'umap': umap_reducer,
            'embeddings_original': embeddings_clean,
            'embeddings_pca': embeddings_pca,
            'embeddings_umap': embeddings_umap,
            'pca_info': {
                'n_components': pca_components,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'cumulative_explained_variance': np.cumsum(pca.explained_variance_ratio_),
                'total_explained_variance': np.sum(pca.explained_variance_ratio_)
            }
        }

        # Save results
        results_path = self.output_path / 'dimensionality_reduction_results.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(result, f)

        self.logger.info(f"PCA components: {pca_components}")
        self.logger.info(f"Total explained variance: {result['pca_info']['total_explained_variance']:.3f}")
        self.logger.info(f"UMAP embedding shape: {embeddings_umap.shape}")

        return result

    def compute_similarity_analysis(self) -> Dict[str, Any]:
        """Compute cosine similarity analysis"""
        self.logger.info("Computing similarity analysis...")

        if self.embeddings is None:
            embeddings_path = self.output_path / 'embeddings.npy'
            self.embeddings = np.load(embeddings_path)

        # Compute similarity matrix
        similarity_matrix = cosine_similarity(self.embeddings)

        # Find most similar pairs
        n_pairs = min(100, len(self.embeddings))  # Top 100 pairs
        similarity_pairs = []

        for i in range(len(self.embeddings)):
            # Get most similar (excluding self)
            similarities = similarity_matrix[i]
            most_similar_idx = similarities.argsort()[-2]  # -1 is self
            similarity_score = similarities[most_similar_idx]

            similarity_pairs.append({
                'image_idx': i,
                'image_name': self.img_names[i] if self.img_names is not None else f'img_{i}',
                'similar_idx': most_similar_idx,
                'similar_name': self.img_names[most_similar_idx] if self.img_names is not None else f'img_{most_similar_idx}',
                'similarity_score': similarity_score
            })

        # Sort by similarity score
        similarity_pairs.sort(key=lambda x: x['similarity_score'], reverse=True)

        result = {
            'similarity_matrix': similarity_matrix,
            'top_similar_pairs': similarity_pairs[:n_pairs],
            'mean_similarity': np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]),
            'std_similarity': np.std(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
        }

        # Save results
        similarity_path = self.output_path / 'similarity_analysis.pkl'
        with open(similarity_path, 'wb') as f:
            pickle.dump(result, f)

        self.logger.info(f"Mean similarity: {result['mean_similarity']:.4f}")
        self.logger.info(f"Std similarity: {result['std_similarity']:.4f}")

        return result

    def create_visualizations(self) -> None:
        """Create visualization plots (if matplotlib available)"""
        if not PLOTTING_AVAILABLE:
            self.logger.warning("Matplotlib not available, skipping visualizations")
            return

        self.logger.info("Creating visualizations...")

        # Load results
        results_path = self.output_path / 'dimensionality_reduction_results.pkl'
        if not results_path.exists():
            self.logger.error("Dimensionality reduction results not found")
            return

        with open(results_path, 'rb') as f:
            result = pickle.load(f)

        # Load labels if available
        labels = None
        label_file = Path('./classifications_kadofong_20250925.csv')
        if label_file.exists() and self.img_names is not None:
            try:
                mergers = pd.read_csv(label_file, index_col=0)
                labels = mergers.reindex(self.img_names)
                labels = labels.replace(np.nan, 0).values.flatten()
                self.logger.info("Loaded classification labels")
            except Exception as e:
                self.logger.warning(f"Could not load labels: {e}")

        # Create PCA analysis plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('PCA Analysis of BYOL Embeddings', fontsize=16)

        pca_info = result['pca_info']
        n_top = min(20, len(pca_info['explained_variance_ratio']))

        # Explained variance
        axes[0,0].bar(range(1, n_top + 1), pca_info['explained_variance_ratio'][:n_top])
        axes[0,0].set_xlabel('Principal Component')
        axes[0,0].set_ylabel('Explained Variance Ratio')
        axes[0,0].set_title('Individual Component Variance')
        axes[0,0].grid(True, alpha=0.3)

        # Cumulative variance
        axes[0,1].plot(range(1, len(pca_info['cumulative_explained_variance']) + 1),
                       pca_info['cumulative_explained_variance'], 'bo-')
        axes[0,1].axhline(y=0.95, color='r', linestyle='--', label='95%')
        axes[0,1].set_xlabel('Number of Components')
        axes[0,1].set_ylabel('Cumulative Explained Variance')
        axes[0,1].set_title('Cumulative Variance Explained')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].legend()

        plt.tight_layout()
        plt.savefig(self.output_path / 'pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Create embedding comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Define custom colormap for galaxy classifications
        try:
            from ekfplot import colors as ec
            # Define colors for each classification
            classification_colors = ['lightgrey', 'tab:blue', 'C1', 'r', 'lime', 'magenta']
            cmap = ec.colormap_from_list(classification_colors, 'discrete')
        except ImportError:
            # Fallback if ekfplot not available
            import matplotlib.colors as mcolors
            classification_colors = ['lightgrey', 'tab:blue', 'orange', 'red', 'lime', 'magenta']
            cmap = mcolors.ListedColormap(classification_colors)

        # PCA plot
        axes[0].scatter(result['embeddings_pca'][:, 0],
                       result['embeddings_pca'][:, 1],
                       cmap=cmap,
                       alpha=1., s=10, c='grey' if labels is None else labels)
        axes[0].set_title('PCA (First 2 Components)')
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        axes[0].grid(True, alpha=0.3)

        # UMAP plot
        scatter = axes[1].scatter(result['embeddings_umap'][:, 0],
                                 result['embeddings_umap'][:, 1],
                                 cmap=cmap,
                                 alpha=1., s=10, c='grey' if labels is None else labels)
        axes[1].set_title('PCA + UMAP')
        axes[1].set_xlabel('UMAP 1')
        axes[1].set_ylabel('UMAP 2')
        axes[1].grid(True, alpha=0.3)

        # Add colorbar with classification labels if labels are provided
        if labels is not None:
            import matplotlib.patches as mpatches
            legend_labels = {0:'unlabeled', 1:'undisturbed', 2:'ambiguous',
                           3:'merger', 4:'fragmentation', 5:'artifact'}
            legend_colors = ['lightgrey', 'tab:blue', 'C1', 'r', 'lime', 'magenta']

            # Create legend patches
            patches = []
            for i, (label, color) in enumerate(zip(legend_labels.values(), legend_colors)):
                if i in labels:  # Only show labels that exist in the data
                    patches.append(mpatches.Patch(color=color, label=label))

            if patches:
                axes[1].legend(handles=patches, loc='upper right', fontsize=8)

        plt.tight_layout()
        plt.savefig(self.output_path / 'embeddings_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info("Visualizations saved")

    def compute_labeled_distance_metrics(self) -> Dict[str, Any]:
        """Compute average intra-class distances in UMAP space, normalized by overall average distance"""
        self.logger.info("Computing labeled distance metrics...")

        # Load UMAP results if not already available
        results_path = self.output_path / 'dimensionality_reduction_results.pkl'
        if not results_path.exists():
            self.logger.error("Dimensionality reduction results not found. Run analysis first.")
            return {}

        with open(results_path, 'rb') as f:
            result = pickle.load(f)

        umap_embeddings = result['embeddings_umap']

        # Load labels if available
        labels = None
        label_file = Path('./classifications_kadofong_20250925.csv')
        if label_file.exists() and self.img_names is not None:
            try:
                import pandas as pd
                mergers = pd.read_csv(label_file, index_col=0)
                labels = mergers.reindex(self.img_names)
                labels = labels.replace(np.nan, 0).values.flatten().astype(int)
                self.logger.info("Loaded classification labels for distance metrics")
            except Exception as e:
                self.logger.warning(f"Could not load labels: {e}")
                return {}
        else:
            self.logger.warning("No classification labels found for distance metrics")
            return {}

        # Define label meanings
        label_meanings = {
            0: "unclassified", 1: "undisturbed", 2: "ambiguous",
            3: "merger", 4: "fragmentation", 5: "artifact"
        }

        # Compute pairwise distances in UMAP space
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(umap_embeddings, metric='euclidean'))

        # Compute overall average distance (for normalization)
        # Use upper triangle to avoid double counting and diagonal
        upper_triangle_indices = np.triu_indices(distances.shape[0], k=1)
        overall_avg_distance = np.mean(distances[upper_triangle_indices])

        # Compute intra-class distances for each label (excluding 0 and 1)
        metrics = {
            'overall_avg_distance': overall_avg_distance,
            'intra_class_distances': {},
            'normalized_intra_class_distances': {},
            'class_counts': {},
            'label_meanings': label_meanings
        }

        for label_val in [2, 3, 4, 5]:  # Exclude 0 (unclassified) and 1 (undisturbed)
            if label_val in labels:
                # Get indices for this label
                class_indices = np.where(labels == label_val)[0]

                if len(class_indices) > 1:  # Need at least 2 objects for distance
                    # Extract distances between objects of same class
                    class_distances = distances[np.ix_(class_indices, class_indices)]

                    # Get upper triangle (avoid diagonal and double counting)
                    upper_indices = np.triu_indices(class_distances.shape[0], k=1)
                    intra_class_dists = class_distances[upper_indices]

                    avg_intra_distance = np.mean(intra_class_dists)
                    normalized_distance = avg_intra_distance / overall_avg_distance

                    metrics['intra_class_distances'][label_val] = {
                        'mean': float(avg_intra_distance),
                        'std': float(np.std(intra_class_dists)),
                        'median': float(np.median(intra_class_dists)),
                        'min': float(np.min(intra_class_dists)),
                        'max': float(np.max(intra_class_dists)),
                        'num_pairs': len(intra_class_dists)
                    }

                    metrics['normalized_intra_class_distances'][label_val] = {
                        'mean_normalized': float(normalized_distance),
                        'label': label_meanings[label_val]
                    }

                    metrics['class_counts'][label_val] = len(class_indices)

                    self.logger.info(f"Class {label_val} ({label_meanings[label_val]}): "
                                   f"{len(class_indices)} objects, "
                                   f"avg distance = {avg_intra_distance:.4f}, "
                                   f"normalized = {normalized_distance:.4f}")
                else:
                    self.logger.info(f"Class {label_val} ({label_meanings[label_val]}): "
                                   f"only {len(class_indices)} objects (need ≥2 for distance)")
                    metrics['class_counts'][label_val] = len(class_indices)

        # Compute summary statistics
        if metrics['normalized_intra_class_distances']:
            normalized_values = [v['mean_normalized'] for v in metrics['normalized_intra_class_distances'].values()]
            metrics['summary'] = {
                'mean_normalized_distance': float(np.mean(normalized_values)),
                'std_normalized_distance': float(np.std(normalized_values)),
                'min_normalized_distance': float(np.min(normalized_values)),
                'max_normalized_distance': float(np.max(normalized_values))
            }

        # Save metrics
        metrics_path = self.output_path / 'labeled_distance_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Also save as pickle for easier loading
        metrics_pickle_path = self.output_path / 'labeled_distance_metrics.pkl'
        with open(metrics_pickle_path, 'wb') as f:
            pickle.dump(metrics, f)

        self.logger.info(f"Distance metrics saved to {metrics_path}")

        # Print summary
        if 'summary' in metrics:
            self.logger.info(f"Summary - Mean normalized intra-class distance: "
                           f"{metrics['summary']['mean_normalized_distance']:.4f} ± "
                           f"{metrics['summary']['std_normalized_distance']:.4f}")

        return metrics

    def run_full_pipeline(self) -> None:
        """Run the complete analysis pipeline"""
        self.logger.info("Starting full BYOL analysis pipeline...")

        # Data loading
        self.load_data()

        # Model training
        self.train_model()

        # Embedding extraction
        self.extract_embeddings()

        # Dimensionality reduction
        self.compute_pca_umap()

        # Similarity analysis
        self.compute_similarity_analysis()

        # Visualizations
        self.create_visualizations()

        # Labeled distance metrics
        self.compute_labeled_distance_metrics()

        self.logger.info("Full pipeline completed successfully")

        # Save summary (convert Path objects to strings for JSON serialization)
        def convert_paths_to_strings(obj):
            """Recursively convert Path objects to strings for JSON serialization"""
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {key: convert_paths_to_strings(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths_to_strings(item) for item in obj]
            else:
                return obj

        summary = {
            'completion_time': datetime.now().isoformat(),
            'num_images': len(self.images),
            'embedding_dimension': self.embeddings.shape[1],
            'device_used': str(self.device),
            'config': convert_paths_to_strings(self.config)
        }

        with open(self.output_path / 'analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_default_config() -> Dict[str, Any]:
    """Create default configuration"""
    return {
        'data': {
            'input_path': '../../local_data/pieridae_output',
            'output_path': 'byol_results'
        },
        'model': {
            'image_size': 150,
            'projection_size': 256,
            'projection_hidden_size': 1024,
            'moving_average_decay': 0.99
        },
        'training': {
            'num_epochs': 50,
            'batch_size': 32,
            'learning_rate': 3e-4,
            'save_interval': 10,
            'resume': True
        },
        'inference': {
            'batch_size': 64
        },
        'analysis': {
            'pca_components': 50,
            'explained_variance_threshold': 0.95,
            'umap_components': 2,
            'n_neighbors': 30,
            'min_dist': 0.1,
            'metric': 'euclidean'
        }
    }


def main():
    parser = argparse.ArgumentParser(description='BYOL Analysis for Slurm Cluster')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--mode', type=str, choices=['train', 'analyze', 'full'],
                       default='full', help='Analysis mode')
    parser.add_argument('--data-path', type=str, help='Path to input data')
    parser.add_argument('--output-path', type=str, help='Path to output directory')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Training batch size')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')

    args = parser.parse_args()

    # Load or create config
    if args.config:
        config = load_config(args.config)
    else:
        # Try to load byol_config.yaml from script directory first
        default_config_path = Path(__file__).parent / 'byol_config.yaml'
        if default_config_path.exists():
            config = load_config(str(default_config_path))
        else:
            config = create_default_config()

    # Override config with command line arguments
    if args.data_path:
        config['data']['input_path'] = args.data_path
    if args.output_path:
        config['data']['output_path'] = args.output_path
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.resume:
        config['training']['resume'] = args.resume

    # Convert paths to Path objects
    config['data']['input_path'] = Path(config['data']['input_path'])
    config['data']['output_path'] = Path(config['data']['output_path'])

    # Initialize analysis
    analysis = BYOLClusterAnalysis(config)

    # Run analysis based on mode
    if args.mode == 'full':
        analysis.run_full_pipeline()
    elif args.mode == 'train':
        analysis.load_data()
        analysis.train_model()
    elif args.mode == 'analyze':
        analysis.load_data()
        analysis.extract_embeddings()
        analysis.compute_pca_umap()
        analysis.compute_similarity_analysis()
        analysis.create_visualizations()
        analysis.compute_labeled_distance_metrics()

    print("Analysis completed successfully!")


if __name__ == '__main__':
    main()