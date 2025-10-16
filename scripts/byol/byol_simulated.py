#!/usr/bin/env python3
"""
BYOL Analysis with Simulated Galaxy Data
Uses simulated Sersic2D profiles to test BYOL clustering performance
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import pickle
import sep
from tqdm import tqdm

# Astropy for Sersic profiles
from astropy.modeling.functional_models import Sersic2D

# ML and clustering
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
from scipy.optimize import linear_sum_assignment

# Import base analysis class
sys.path.insert(0, os.path.dirname(__file__))
from byol_cluster_analysis_macos import BYOLMacOSAnalysis, load_config, create_default_config

# For starlet transform - will use ekfstats if available
try:
    from ekfstats import imstats
    STARLET_AVAILABLE = True
except ImportError:
    print("Warning: ekfstats not available, using simplified hf_image generation")
    STARLET_AVAILABLE = False

# For scipy ndimage
from scipy import ndimage

# Plotting
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib not available")


class BYOLSimulatedAnalysis(BYOLMacOSAnalysis):
    """BYOL analysis using simulated Sersic2D galaxy profiles"""

    def __init__(self, config: Dict[str, Any], n_per_class: int = 300, random_seed: int = 42):
        """
        Initialize simulated analysis

        Parameters
        ----------
        config : dict
            Configuration dictionary
        n_per_class : int
            Number of galaxies to generate per class
        random_seed : int
            Random seed for reproducibility
        """
        self.n_per_class = n_per_class
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

        # Store ground truth labels
        self.true_labels = None
        self.predicted_labels = None

        # Class names
        self.class_names = {
            0: 'disk',
            1: 'double_nuclei',
            2: 'dipole'
        }

        super().__init__(config)

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate simulated galaxy data instead of loading from disk"""
        self.logger.info(f"Generating {self.n_per_class * 3} simulated galaxies...")

        image_size = self.config['model']['image_size']
        n_total = self.n_per_class * 3

        # Initialize arrays
        imgs = []
        img_names = []
        true_labels = []

        # Generate each class
        for class_id in range(3):
            self.logger.info(f"Generating class {class_id} ({self.class_names[class_id]})...")

            for i in tqdm(range(self.n_per_class), desc=f"Class {class_id}"):
                if class_id == 0:
                    img = self.generate_disk(image_size)
                elif class_id == 1:
                    img = self.generate_double_nuclei(image_size)
                else:  # class_id == 2
                    img = self.generate_dipole(image_size)

                imgs.append(img)
                img_names.append(f"{self.class_names[class_id]}_{i:04d}")
                true_labels.append(class_id)

        # Convert to arrays
        self.images = np.array(imgs)
        self.img_names = np.array(img_names)
        self.true_labels = np.array(true_labels)

        self.logger.info(f"Generated {len(self.images)} images with shape {self.images.shape}")
        self.logger.info(f"Class distribution: {np.bincount(self.true_labels)}")

        # Save simulated data
        sim_data_path = self.output_path / 'simulated_galaxies.pkl'
        with open(sim_data_path, 'wb') as f:
            pickle.dump({
                'images': self.images,
                'img_names': self.img_names,
                'true_labels': self.true_labels,
                'class_names': self.class_names,
                'n_per_class': self.n_per_class,
                'random_seed': self.random_seed
            }, f)

        return self.images, self.img_names

    def generate_disk(self, image_size: int = 150) -> np.ndarray:
        """
        Generate a disk galaxy (single Sersic profile)

        Parameters
        ----------
        image_size : int
            Size of image in pixels

        Returns
        -------
        img : np.ndarray
            3-channel image (g, i, hf_i)
        """
        center = image_size / 2.0

        # Random parameters
        r_eff = self.rng.uniform(10, 30)
        ellip = self.rng.uniform(0.8, 0.9)
        theta = self.rng.uniform(0, 2 * np.pi)
        amplitude = self.rng.uniform(0.8, 1.2)  # Normalized amplitude

        # Create coordinate grids
        y, x = np.mgrid[0:image_size, 0:image_size]

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

        # Generate base image
        base_image = sersic(x, y)

        # Add noise
        noise_level = 0.01
        noise = self.rng.normal(0, noise_level, base_image.shape)

        # Create g and i bands with slight amplitude differences
        g_band = base_image * 0.9 + noise
        i_band = base_image * 1.0 + noise

        # Create high-frequency image using starlet decomposition
        hf_i_band = self.create_hf_image(i_band)

        # Stack into 3-channel image
        img = np.stack([g_band, i_band, hf_i_band], axis=0)

        return img

    def generate_double_nuclei(self, image_size: int = 150) -> np.ndarray:
        """
        Generate a double nuclei galaxy (two added Sersic profiles)

        Parameters
        ----------
        image_size : int
            Size of image in pixels

        Returns
        -------
        img : np.ndarray
            3-channel image (g, i, hf_i)
        """
        center = image_size / 2.0

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
        y, x = np.mgrid[0:image_size, 0:image_size]

        # Create first Sersic profile (centered)
        sersic_1 = Sersic2D(
            amplitude=amplitude_1,
            r_eff=r_eff_1,
            n=1.0,
            x_0=center,
            y_0=center,
            ellip=ellip_1,
            theta=theta_1
        )

        # Create second Sersic profile (offset)
        sersic_2 = Sersic2D(
            amplitude=amplitude_2,
            r_eff=r_eff_2,
            n=1.0,
            x_0=center + x_offset,
            y_0=center + y_offset,
            ellip=ellip_2,
            theta=theta_2
        )

        # Add the two profiles
        base_image = sersic_1(x, y) + sersic_2(x, y)

        # Add noise
        noise_level = 0.01
        noise = self.rng.normal(0, noise_level, base_image.shape)

        # Create g and i bands
        g_band = base_image * 0.9 + noise
        i_band = base_image * 1.0 + noise

        # Create high-frequency image
        hf_i_band = self.create_hf_image(i_band)

        # Stack into 3-channel image
        img = np.stack([g_band, i_band, hf_i_band], axis=0)

        return img

    def generate_dipole(self, image_size: int = 150) -> np.ndarray:
        """
        Generate a dipole galaxy (two subtracted Sersic profiles)

        Parameters
        ----------
        image_size : int
            Size of image in pixels

        Returns
        -------
        img : np.ndarray
            3-channel image (g, i, hf_i)
        """
        center = image_size / 2.0

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
        y, x = np.mgrid[0:image_size, 0:image_size]

        # Create first Sersic profile
        sersic_1 = Sersic2D(
            amplitude=amplitude,
            r_eff=r_eff,
            n=1.0,
            x_0=center,
            y_0=center,
            ellip=ellip_1,
            theta=theta_1
        )

        # Create second Sersic profile (same amplitude, different ellipticity/theta)
        sersic_2 = Sersic2D(
            amplitude=amplitude,
            r_eff=r_eff,
            n=1.0,
            x_0=center,
            y_0=center,
            ellip=ellip_2,
            theta=theta_2
        )

        # Subtract the two profiles to create dipole
        base_image = sersic_1(x, y) - sersic_2(x, y)

        # Add offset to keep values positive
        base_image = base_image - np.min(base_image) + 0.1

        # Add noise
        noise_level = 0.01
        noise = self.rng.normal(0, noise_level, base_image.shape)

        # Create g and i bands
        g_band = base_image * 0.9 + noise
        i_band = base_image * 1.0 + noise

        # Create high-frequency image
        hf_i_band = self.create_hf_image(i_band)

        # Stack into 3-channel image
        img = np.stack([g_band, i_band, hf_i_band], axis=0)

        return img

    def create_hf_image(self, i_band: np.ndarray) -> np.ndarray:
        """
        Create high-frequency residual image using starlet decomposition
        (analogous to run_starlet.py L112-222)

        Parameters
        ----------
        i_band : np.ndarray
            Input i-band image

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
                #hf_image = np.where(sources > 0, vm, hf_image) #\\ no other sources in the mock images

                return hf_image

            except Exception as e:
                self.logger.warning(f"Starlet transform failed: {e}, using fallback method")
                return self._create_hf_image_fallback(i_band)
        else:
            return self._create_hf_image_fallback(i_band)

    def _create_hf_image_fallback(self, i_band: np.ndarray) -> np.ndarray:
        """
        Fallback method for creating high-frequency image
        Uses simple high-pass filtering if starlet transform not available
        """
        # Apply Gaussian smoothing and subtract to get high-frequency
        from scipy.ndimage import gaussian_filter

        smoothed = gaussian_filter(i_band, sigma=5.0)
        hf_image = i_band - smoothed

        # Subtract median filter
        hf_image = hf_image - ndimage.median_filter(hf_image, size=20)

        return hf_image

    def classify_embeddings(self, n_clusters: int = 3) -> np.ndarray:
        """
        Classify embeddings using KMeans clustering

        Parameters
        ----------
        n_clusters : int
            Number of clusters (should match number of classes)

        Returns
        -------
        predicted_labels : np.ndarray
            Predicted cluster labels
        """
        self.logger.info(f"Clustering embeddings into {n_clusters} clusters...")

        if self.embeddings is None:
            self.logger.info("Loading embeddings...")
            embeddings_path = self.output_path / 'embeddings.npy'
            self.embeddings = np.load(embeddings_path)

        # Apply KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_seed, n_init=10)
        cluster_labels = kmeans.fit_predict(self.embeddings)

        # Align cluster labels with true labels using Hungarian algorithm
        aligned_labels = self.align_cluster_labels(cluster_labels, self.true_labels)

        self.predicted_labels = aligned_labels

        self.logger.info(f"Clustering complete. Predicted label distribution: {np.bincount(aligned_labels)}")

        # Save predicted labels
        labels_path = self.output_path / 'predicted_labels.npy'
        np.save(labels_path, aligned_labels)

        return aligned_labels

    def align_cluster_labels(self, cluster_labels: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
        """
        Align cluster labels with true labels using Hungarian algorithm

        Parameters
        ----------
        cluster_labels : np.ndarray
            Raw cluster assignments from KMeans
        true_labels : np.ndarray
            Ground truth labels

        Returns
        -------
        aligned_labels : np.ndarray
            Cluster labels aligned with true labels
        """
        # Create confusion matrix
        conf_matrix = confusion_matrix(true_labels, cluster_labels)

        # Use Hungarian algorithm to find optimal alignment
        # We want to maximize overlap, so use negative cost
        row_ind, col_ind = linear_sum_assignment(-conf_matrix)

        # Create mapping from cluster ID to true label
        cluster_to_true = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}

        # Apply mapping
        aligned_labels = np.array([cluster_to_true[c] for c in cluster_labels])

        self.logger.info(f"Cluster alignment mapping: {cluster_to_true}")

        return aligned_labels

    def compute_purity_completeness(self) -> Dict[str, Any]:
        """
        Compute purity and completeness metrics

        Returns
        -------
        metrics : dict
            Dictionary containing purity, completeness, and other metrics
        """
        self.logger.info("Computing purity and completeness metrics...")

        if self.predicted_labels is None:
            raise ValueError("Must run classify_embeddings() first")

        n_samples = len(self.true_labels)
        n_classes = len(np.unique(self.true_labels))

        # Compute confusion matrix
        conf_matrix = confusion_matrix(self.true_labels, self.predicted_labels)

        # Compute purity (for each predicted cluster, fraction of most common true class)
        cluster_purities = []
        for cluster_id in range(n_classes):
            cluster_mask = self.predicted_labels == cluster_id
            if np.sum(cluster_mask) > 0:
                true_labels_in_cluster = self.true_labels[cluster_mask]
                most_common_count = np.max(np.bincount(true_labels_in_cluster))
                purity = most_common_count / np.sum(cluster_mask)
                cluster_purities.append(purity)
            else:
                cluster_purities.append(0.0)

        overall_purity = np.sum([np.sum(self.predicted_labels == i) * cluster_purities[i]
                                 for i in range(n_classes)]) / n_samples

        # Compute completeness (for each true class, fraction in most common predicted cluster)
        class_completeness = []
        for class_id in range(n_classes):
            class_mask = self.true_labels == class_id
            if np.sum(class_mask) > 0:
                pred_labels_in_class = self.predicted_labels[class_mask]
                most_common_cluster = np.argmax(np.bincount(pred_labels_in_class))
                completeness = np.sum(pred_labels_in_class == most_common_cluster) / np.sum(class_mask)
                class_completeness.append(completeness)
            else:
                class_completeness.append(0.0)

        overall_completeness = np.mean(class_completeness)

        # Classification report
        class_report = classification_report(
            self.true_labels,
            self.predicted_labels,
            target_names=[self.class_names[i] for i in range(n_classes)],
            output_dict=True
        )

        # Compile metrics
        metrics = {
            'overall_purity': float(overall_purity),
            'overall_completeness': float(overall_completeness),
            'cluster_purities': {self.class_names[i]: float(p) for i, p in enumerate(cluster_purities)},
            'class_completeness': {self.class_names[i]: float(c) for i, c in enumerate(class_completeness)},
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report,
            'n_samples': n_samples,
            'n_classes': n_classes,
            'class_names': self.class_names
        }

        # Save metrics
        metrics_path = self.output_path / 'classification_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Log results
        self.logger.info(f"Overall Purity: {overall_purity:.4f}")
        self.logger.info(f"Overall Completeness: {overall_completeness:.4f}")
        self.logger.info(f"Per-class purity: {cluster_purities}")
        self.logger.info(f"Per-class completeness: {class_completeness}")

        return metrics

    def create_evaluation_visualizations(self) -> None:
        """Create visualizations comparing ground truth and predicted labels"""
        if not PLOTTING_AVAILABLE:
            self.logger.warning("Matplotlib not available, skipping evaluation visualizations")
            return

        self.logger.info("Creating evaluation visualizations...")

        # Load PCA results if needed
        results_path = self.output_path / 'dimensionality_reduction_results.pkl'
        if not results_path.exists():
            self.logger.error("Dimensionality reduction results not found")
            return

        with open(results_path, 'rb') as f:
            result = pickle.load(f)

        pca_embeddings = result['embeddings_pca']

        # Load classification metrics
        metrics_path = self.output_path / 'classification_metrics.json'
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Color maps for classes
        colors = ['tab:blue', 'tab:orange', 'tab:green']

        # 1. PCA with ground truth labels
        for class_id in range(3):
            mask = self.true_labels == class_id
            axes[0, 0].scatter(
                pca_embeddings[mask, 0],
                pca_embeddings[mask, 1],
                c=colors[class_id],
                label=self.class_names[class_id],
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
            mask = self.predicted_labels == class_id
            axes[0, 1].scatter(
                pca_embeddings[mask, 0],
                pca_embeddings[mask, 1],
                c=colors[class_id],
                label=self.class_names[class_id],
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
            xticklabels=[self.class_names[i] for i in range(3)],
            yticklabels=[self.class_names[i] for i in range(3)],
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

        axes[1, 1].text(0.1, 0.5, metrics_text,
                       fontsize=11,
                       family='monospace',
                       verticalalignment='center')

        plt.tight_layout()
        plt.savefig(self.output_path / 'evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info("Evaluation visualizations saved")

        # Create sample galaxy images
        self._create_sample_galaxy_images()

        # Create QA figure with i-band and HF images
        self._create_qa_figure()

    def _create_sample_galaxy_images(self, n_samples: int = 3) -> None:
        """Create visualization showing sample galaxies from each class"""
        if not PLOTTING_AVAILABLE:
            return

        self.logger.info("Creating sample galaxy visualizations...")

        fig, axes = plt.subplots(3, n_samples, figsize=(12, 10))

        for class_id in range(3):
            # Get indices for this class
            class_indices = np.where(self.true_labels == class_id)[0]

            # Sample n_samples from this class
            sample_indices = self.rng.choice(class_indices, size=min(n_samples, len(class_indices)), replace=False)

            for i, idx in enumerate(sample_indices):
                img = self.images[idx]

                # Create RGB composite from g, i bands (normalize for display)
                g_norm = (img[0] - img[0].min()) / (img[0].max() - img[0].min() + 1e-8)
                i_norm = (img[1] - img[1].min()) / (img[1].max() - img[1].min() + 1e-8)
                hf_norm = (img[2] - img[2].min()) / (img[2].max() - img[2].min() + 1e-8)

                # Show composite
                rgb = np.stack([i_norm, g_norm, g_norm], axis=-1)
                axes[class_id, i].imshow(rgb, origin='lower')
                axes[class_id, i].set_title(f"{self.class_names[class_id]} {i+1}")
                axes[class_id, i].axis('off')

        plt.suptitle('Sample Galaxies by Class', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_path / 'sample_galaxies.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info("Sample galaxy visualizations saved")

    def _create_qa_figure(self) -> None:
        """Create QA figure showing i-band and HF images for one example from each class"""
        if not PLOTTING_AVAILABLE:
            return

        self.logger.info("Creating QA figure with i-band and HF images...")

        fig, axes = plt.subplots(3, 2, figsize=(10, 12))

        for class_id in range(3):
            # Get indices for this class
            class_indices = np.where(self.true_labels == class_id)[0]

            # Select one random example from this class
            idx = self.rng.choice(class_indices)
            img = self.images[idx]

            # Extract i-band and HF images (img is shape [3, H, W] with [g, i, hf])
            i_band = img[1]  # i-band is second channel
            hf_image = img[2]  # HF is third channel

            # Normalize for display
            i_norm = (i_band - np.percentile(i_band, 1)) / (np.percentile(i_band, 99) - np.percentile(i_band, 1))
            i_norm = np.clip(i_norm, 0, 1)

            hf_norm = (hf_image - np.percentile(hf_image, 1)) / (np.percentile(hf_image, 99) - np.percentile(hf_image, 1))
            hf_norm = np.clip(hf_norm, 0, 1)

            # Plot i-band
            axes[class_id, 0].imshow(i_norm, origin='lower', cmap='gray')
            axes[class_id, 0].set_title(f"{self.class_names[class_id]} - i-band")
            axes[class_id, 0].axis('off')

            # Plot HF image
            axes[class_id, 1].imshow(hf_norm, origin='lower', cmap='gray')
            axes[class_id, 1].set_title(f"{self.class_names[class_id]} - HF")
            axes[class_id, 1].axis('off')

        plt.suptitle('QA Figure: i-band and High-Frequency Images by Galaxy Type', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_path / 'qa_figure.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info("QA figure saved")

    def create_visualizations(self) -> None:
        """Override parent visualization to skip UMAP and use simulated labels"""
        if not PLOTTING_AVAILABLE:
            self.logger.warning("Matplotlib not available, skipping visualizations")
            return

        self.logger.info("Creating PCA visualizations...")

        # Load results
        results_path = self.output_path / 'dimensionality_reduction_results.pkl'
        if not results_path.exists():
            self.logger.error("Dimensionality reduction results not found")
            return

        with open(results_path, 'rb') as f:
            result = pickle.load(f)

        # Use simulated labels
        labels = self.true_labels

        # Create PCA scatter plot with ground truth labels
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        colors = ['tab:blue', 'tab:orange', 'tab:green']

        for class_id in range(3):
            mask = labels == class_id
            ax.scatter(result['embeddings_pca'][mask, 0],
                      result['embeddings_pca'][mask, 1],
                      c=colors[class_id],
                      label=self.class_names[class_id],
                      alpha=0.6, s=20)
        ax.set_title('PCA: Ground Truth Labels')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_path / 'pca_embeddings.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info("Visualizations saved")

    def run_full_pipeline(self) -> None:
        """Run the complete simulated analysis pipeline"""
        self.logger.info("Starting full BYOL simulated analysis pipeline...")

        # Data generation
        self.load_data()

        # Model training
        self.train_model()

        # Embedding extraction
        self.extract_embeddings()

        # Dimensionality reduction (PCA only, no UMAP)
        self.compute_pca_umap()

        # Similarity analysis
        self.compute_similarity_analysis()

        # Visualizations
        self.create_visualizations()

        # Classification
        self.classify_embeddings()

        # Evaluation metrics
        self.compute_purity_completeness()

        # Evaluation visualizations
        self.create_evaluation_visualizations()

        self.logger.info("Full simulated pipeline completed successfully")


def main():
    """Main function for simulated analysis"""
    import argparse

    parser = argparse.ArgumentParser(description='BYOL Analysis with Simulated Galaxies')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--n-per-class', type=int, default=300,
                       help='Number of galaxies per class')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--output-path', type=str,
                       default='byol_simulated_results',
                       help='Path to output directory')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Training batch size')

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
    config['data']['output_path'] = Path(args.output_path)
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size

    # Initialize simulated analysis
    analysis = BYOLSimulatedAnalysis(
        config,
        n_per_class=args.n_per_class,
        random_seed=args.random_seed
    )

    # Run full pipeline
    try:
        analysis.run_full_pipeline()
        print("✅ Simulated analysis completed successfully!")

        # Print summary
        metrics_path = analysis.output_path / 'classification_metrics.json'
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        print("\n" + "="*60)
        print("CLASSIFICATION RESULTS")
        print("="*60)
        print(f"Overall Purity:       {metrics['overall_purity']:.4f}")
        print(f"Overall Completeness: {metrics['overall_completeness']:.4f}")
        print("\nPer-class metrics:")
        for class_name in ['disk', 'double_nuclei', 'dipole']:
            print(f"  {class_name:20s} - Purity: {metrics['cluster_purities'][class_name]:.4f}, "
                  f"Completeness: {metrics['class_completeness'][class_name]:.4f}")
        print("="*60)

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
