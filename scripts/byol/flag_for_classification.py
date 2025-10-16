#!/usr/bin/env python3
"""
BYOL Analysis Script - Flag objects for classification

This script reproduces the analysis from byol_analysis_notebook.ipynb
up through the "Use PCA to estimate labels" section. It identifies
objects with fewer than 5 labeled neighbors (n_labels < 5) and saves
their IDs to a CSV file for further manual classification.
"""

import os
import sys
import gc
import torch
import numpy as np
import yaml
import glob
import pickle
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torchvision import models, transforms
from torch import nn
from byol_pytorch import BYOL
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


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


def load_config(config_path: str):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Convert paths to Path objects
    config['data']['input_path'] = Path(config['data']['input_path'])
    config['data']['output_path'] = Path(config['data']['output_path'])

    return config


def load_data(data_path: Path):
    """Load image data from pickle files"""
    print(f"🔍 Loading image data from: {data_path}")

    pattern = f"{data_path}/M*/*i_results.pkl"
    filenames = glob.glob(pattern)

    if not filenames:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")

    print(f"📸 Found {len(filenames)} image files")

    # First pass: count valid images and get image shape
    print("🔍 Counting valid images...")
    valid_files = []
    img_shape = None

    for fname in tqdm(filenames, desc="Validating files"):
        # Check if all required files exist
        g_file = fname.replace('_i_', '_g_')
        i_file = fname

        if os.path.exists(g_file) and os.path.exists(i_file):
            if img_shape is None:
                # Get shape from first valid image
                with open(i_file, 'rb') as f:
                    xf = pickle.load(f)
                    img_shape = xf['image'].shape
            valid_files.append(fname)

    n_images = len(valid_files)
    print(f"✅ Found {n_images} valid image sets")

    if n_images == 0:
        raise ValueError("No valid image files found")

    # Pre-allocate arrays (avoids list->array conversion)
    images = np.zeros((n_images, 3, img_shape[0], img_shape[1]), dtype=np.float32)
    img_names = []

    # Second pass: load images directly into pre-allocated array
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
                    del xf  # Free pickle data immediately
            except FileNotFoundError:
                print(f"⚠️  File not found: {current_filename}")
                continue

        if len(img) == 3:  # Only add if we have all bands
            images[idx] = np.array(img)
            img_names.append(Path(fname).parent.name)
            idx += 1
            del img  # Free temporary list

    # Trim array if some files failed to load
    if idx < n_images:
        images = images[:idx]
        print(f"⚠️  Loaded {idx} images (expected {n_images})")

    img_names = np.array(img_names)

    print(f"✅ Loaded {len(images)} images with shape: {images.shape}")
    return images, img_names


def setup_byol_model(config, device):
    """Setup BYOL model for inference"""
    print("🤖 Setting up BYOL model...")

    # Data augmentations (same as training)
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

    # Base model - use ResNet18 for compatibility
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # BYOL learner
    learner = BYOL(
        resnet,
        image_size=config['model']['image_size'],
        hidden_layer='avgpool',
        projection_size=config['model']['projection_size'],
        projection_hidden_size=config['model']['projection_hidden_size'],
        moving_average_decay=config['model']['moving_average_decay'],
        use_momentum=True,
        augment_fn=transform1,
        augment_fn2=transform2
    ).to(device)

    print(f"✅ BYOL model setup complete on {device}")
    return learner


def load_trained_model(learner, output_path: Path, device):
    """Load trained model weights"""
    model_path = output_path / 'byol_final_model.pt'

    if not model_path.exists():
        model_path = output_path / 'model_checkpoint.pt'

    if not model_path.exists():
        raise FileNotFoundError(f"No trained model found in {output_path}")

    print(f"📥 Loading trained model from: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        learner.load_state_dict(checkpoint['model_state_dict'])
    else:
        learner.load_state_dict(checkpoint)

    learner.eval()
    print(f"✅ Model loaded successfully")
    return learner


def extract_embeddings(learner, images, device, batch_size=128):
    """Extract embeddings from images using trained BYOL model"""
    print(f"🧠 Extracting embeddings from {len(images)} images...")

    learner.eval()
    all_embeddings = []

    # Adjust batch size for device capabilities
    if device.type == 'mps':
        batch_size = min(64, batch_size)

    num_batches = (len(images) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Extracting embeddings"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(images))

            batch = torch.tensor(
                images[start_idx:end_idx],
                dtype=torch.float32
            ).to(device)

            try:
                _, embeddings = learner(batch, return_embedding=True)
                all_embeddings.append(embeddings.cpu().numpy())

            except RuntimeError as e:
                if "MPS" in str(e) and device.type == 'mps':
                    print(f"⚠️  MPS error, falling back to CPU for batch {i}")
                    batch = batch.cpu()
                    learner_cpu = learner.cpu()
                    _, embeddings = learner_cpu(batch, return_embedding=True)
                    all_embeddings.append(embeddings.cpu().numpy())
                    learner = learner.to(device)
                else:
                    raise e

    embeddings = np.vstack(all_embeddings)
    del all_embeddings  # Free memory immediately after vstacking
    gc.collect()  # Force garbage collection
    print(f"✅ Extracted embeddings shape: {embeddings.shape}")

    return embeddings


def compute_pca(embeddings, config):
    """Apply PCA to embeddings"""
    print("🔄 Computing PCA...")

    # Clean data
    embeddings_clean = np.nan_to_num(
        embeddings, nan=0.0, posinf=0.0, neginf=0.0
    )

    # Standardization
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_clean)

    # PCA
    pca_components = config['analysis']['pca_components']
    if pca_components is None:
        pca_full = PCA()
        pca_full.fit(embeddings_scaled)
        cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
        threshold = config['analysis']['explained_variance_threshold']
        pca_components = np.argmax(cumsum_var >= threshold) + 1

    max_components = min(
        embeddings_scaled.shape[0] - 1,
        embeddings_scaled.shape[1]
    )
    pca_components = min(pca_components, max_components)

    pca = PCA(n_components=pca_components, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings_scaled)

    print(f"✅ PCA components: {pca_components}")
    print(f"✅ Explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")

    return embeddings_pca, pca


def load_labels(config, img_names):
    """Load classification labels"""
    labels = None
    label_file = Path(config.get('labels', {}).get('classifications_file',
                                                     './classifications_kadofong_20250929.csv'))

    if label_file.exists() and img_names is not None:
        try:
            mergers = pd.read_csv(label_file, index_col=0)
            labels = mergers.reindex(img_names)
            labels = labels.replace(np.nan, 0).values.flatten().astype(int)
            print(f"✅ Loaded classification labels: {len(labels)} objects")

            # Print label distribution
            unique, counts = np.unique(labels, return_counts=True)
            label_meanings = {
                0: "unclassified", 1: "undisturbed", 2: "ambiguous",
                3: "merger", 4: "fragmentation", 5: "artifact"
            }

            print("📊 Label distribution:")
            for label_val, count in zip(unique, counts):
                meaning = label_meanings.get(label_val, f"unknown_{label_val}")
                print(f"   {label_val} ({meaning}): {count} objects")

        except Exception as e:
            print(f"⚠️  Could not load labels: {e}")
            labels = None
    else:
        print(f"⚠️  Label file not found: {label_file}")
        labels = None

    return labels


def estimate_labels_from_neighbors(pca_embeddings, labels, n_neighbors=50, n_min=8):
    """
    Use k-nearest neighbors in PCA space to estimate labels
    Returns n_labels (count of labeled neighbors for each object)
    """
    print(f"🔍 Finding {n_neighbors} nearest neighbors in PCA space...")

    # Fit nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(pca_embeddings)
    distances, indices = nbrs.kneighbors(pca_embeddings)

    # Set distance to self as NaN
    distances[:, 0] = np.nan

    # Get neighbor labels
    neighbor_labels = labels[indices]

    # Compute weights based on inverse distance, only for labeled neighbors
    weights = np.where(neighbor_labels > 0, 1. / distances, 0.)
    weights /= np.nansum(weights, axis=1).reshape(-1, 1)

    # Compute probability labels for each class
    prob_labels = np.zeros([pca_embeddings.shape[0], labels.max() + 1])

    for ix in range(labels.max() + 1):
        prob_labels[:, ix] = np.nansum(
            np.where(neighbor_labels == ix, weights, 0), axis=1
        )

    # Count number of labeled neighbors
    n_labels = np.sum(neighbor_labels > 0, axis=1)

    # Zero out probability labels for objects with too few labeled neighbors
    prob_labels[n_labels < n_min] = 0.

    print(f"✅ {(prob_labels > 0).any(axis=1).sum()} objects have auto-labels")
    print(f"✅ {(n_labels < n_min).sum()} objects have fewer than {n_min} labeled neighbors")

    return n_labels, prob_labels


def iterative_label_estimation(pca_embeddings, labels, n_neighbors=50, n_min=8, prob_threshold=0.6, frag_threshold=0.1):
    """
    Iteratively estimate labels using k-nearest neighbors

    First iteration: Add high-confidence auto-labels
    Second iteration: Recalculate with expanded label set

    Returns:
        iterative_labels: Labels after adding auto-labels
        n_labels_iter: Number of labeled neighbors after iteration
        prob_labels_iter: Probability labels after iteration
        stats: Dictionary with label counts at each stage
    """
    print("\n🔄 Starting iterative label estimation...")

    # Count initial human labels
    n_human = (labels > 0).sum()
    print(f"📊 Human labels: {n_human}")

    # First iteration: Get initial probability labels
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(pca_embeddings)
    distances, indices = nbrs.kneighbors(pca_embeddings)
    distances[:, 0] = np.nan

    neighbor_labels = labels[indices]
    weights = np.where(neighbor_labels > 0, 1. / distances, 0.)
    weights /= np.nansum(weights, axis=1).reshape(-1, 1)

    prob_labels = np.zeros([pca_embeddings.shape[0], labels.max() + 1])
    for ix in range(labels.max() + 1):
        prob_labels[:, ix] = np.nansum(
            np.where(neighbor_labels == ix, weights, 0), axis=1
        )

    n_labels = np.sum(neighbor_labels > 0, axis=1)
    prob_labels[n_labels < n_min] = 0.

    n_initial_auto = (prob_labels > 0).any(axis=1).sum()
    print(f"📊 Initial auto-labels: {n_initial_auto} objects have auto-labels")

    # Add high-confidence labels
    iterative_labels = labels.copy()
    n_labeled_before = (iterative_labels > 0).sum()

    # Add labels where probability > threshold and enough labeled neighbors
    additions = np.where(prob_labels[n_labels >= n_min] > prob_threshold)
    new_labels = np.zeros_like(iterative_labels)
    new_labels[additions[0]] = additions[1]

    # Special case: Add fragmentation labels with lower threshold
    new_labels[(prob_labels[:, 4] > frag_threshold) & (n_labels >= n_min)] = 4

    # Only update unlabeled objects
    iterative_labels[iterative_labels == 0] = new_labels[iterative_labels == 0]

    n_added = (iterative_labels > 0).sum() - n_labeled_before
    print(f"📊 Added {n_added} auto-labels in first iteration")
    print(f"📊 Total labels after iteration: {(iterative_labels > 0).sum()}")

    # Second iteration: Recalculate with expanded labels
    print("\n🔄 Recalculating with expanded label set...")
    neighbor_labels_iter = iterative_labels[indices]
    weights_iter = np.where(neighbor_labels_iter > 0, 1. / distances, 0.)
    weights_iter /= np.nansum(weights_iter, axis=1).reshape(-1, 1)

    prob_labels_iter = np.zeros([pca_embeddings.shape[0], labels.max() + 1])
    for ix in range(labels.max() + 1):
        prob_labels_iter[:, ix] = np.nansum(
            np.where(neighbor_labels_iter == ix, weights_iter, 0), axis=1
        )

    n_labels_iter = np.sum(neighbor_labels_iter > 0, axis=1)
    prob_labels_iter[n_labels_iter < n_min] = 0.

    n_final_auto = (prob_labels_iter > 0).any(axis=1).sum()
    print(f"📊 After second iteration: {n_final_auto} objects have auto-labels")

    stats = {
        'n_human': n_human,
        'n_initial_auto': n_initial_auto,
        'n_added_iteration': n_added,
        'n_total_after_iteration': (iterative_labels > 0).sum(),
        'n_final_auto': n_final_auto
    }

    return iterative_labels, n_labels_iter, prob_labels_iter, stats


def save_flagged_objects(img_names, n_labels, output_file, n_min=8):
    """Save object IDs with n_labels < 5 to CSV"""

    # Get objects with fewer than 5 labeled neighbors
    flagged_mask = n_labels < n_min
    flagged_objects = img_names[flagged_mask]

    # Create DataFrame
    df = pd.DataFrame({'object_id': flagged_objects})

    # Save to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n💾 Saved {len(flagged_objects)} flagged object IDs to: {output_path}")

    return flagged_objects


def main():
    # Load configuration
    config_path = 'byol_config.yaml'
    config = load_config(config_path)
    print(f"📋 Configuration loaded from: {config_path}")

    # Setup device
    device = setup_device()

    # Load images
    images, img_names = load_data(config['data']['input_path'])

    # Setup and load BYOL model
    learner = setup_byol_model(config, device)
    learner = load_trained_model(learner, config['data']['output_path'], device)

    # Extract embeddings
    embeddings = extract_embeddings(
        learner,
        images,
        device,
        batch_size=config['inference']['batch_size']
    )

    # Compute PCA
    pca_embeddings, pca = compute_pca(embeddings, config)

    # Load labels
    labels = load_labels(config, img_names)

    if labels is None:
        raise ValueError("Could not load classification labels")

    # Get minimum labeled neighbors threshold from config
    n_min = config.get('labels', {}).get('minimum_labeled_neighbors', 8)
    print(f"\n📋 Using minimum_labeled_neighbors = {n_min} from config")

    # Run iterative label estimation
    iterative_labels, n_labels_iter, prob_labels_iter, stats = iterative_label_estimation(
        pca_embeddings, labels, n_neighbors=50, n_min=n_min
    )

    # Save flagged objects (using iterative labels)
    output_file = config['data']['output_path'] / 'flagged_for_classification.csv'
    flagged_objects = save_flagged_objects(img_names, n_labels_iter, output_file, n_min=n_min)

    print("\n" + "=" * 60)
    print("🎉 ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"📊 Total objects: {len(img_names)}")
    print(f"🏷️  Human labels: {stats['n_human']}")
    print(f"🏷️  Auto-labels added (iteration 1): {stats['n_added_iteration']}")
    print(f"🏷️  Total labels after iteration: {stats['n_total_after_iteration']}")
    print(f"🏷️  Objects with auto-labels (iteration 2): {stats['n_final_auto']}")
    print(f"🚩 Flagged for classification: {len(flagged_objects)}")
    print(f"💾 Output: {output_file}")


if __name__ == "__main__":
    main()
