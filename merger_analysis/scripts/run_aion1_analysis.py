#!/usr/bin/env python
"""
AION-1 Merger Classification Analysis

This script replicates the merger classification analysis from
merger_classification_exploration-Copy1.ipynb using AION-1 embeddings
instead of BYOL embeddings.

Differences from BYOL notebook:
- Uses AION-1 foundation model for encoding
- Only loads g-band and i-band images (no hf_image)
- Properly formats images as HSC data for AION-1
- Generates same analysis plots and figures

Output: Figures saved to output/aion1_test/
"""

import os
import sys
from pathlib import Path
import pickle
import glob
import logging
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import torch
from tqdm import tqdm

# AION-1 imports
try:
    from aion import AION
    from aion.codecs import CodecManager
    from aion.modalities import HSCImage
except ImportError as e:
    print("Error: AION not installed. Please install with: pip install polymathic-aion")
    print(f"Import error: {e}")
    sys.exit(1)

# Pieridae imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pieridae.starbursts.byol import (
    EmbeddingAnalyzer,
    LabelPropagation,
)

# Plotting utilities
from ekfplot import plot as ek, colors as ec
from matplotlib import colors
from ekfstats import sampling
from ekfphys import calibrations
from pieridae.starbursts import sample

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_merian_images_gi_only(
    data_path: Path,
    max_images: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Merian galaxy images (g and i bands only) from pickle files.

    Parameters
    ----------
    data_path : Path
        Path to directory containing M*/*_i_results.pkl files
    max_images : int, optional
        Maximum number of images to load (for testing)

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

            if max_images and len(valid_files) >= max_images:
                break

    n_images = len(valid_files)
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
                logger.warning(f"Error loading {current_filename}: {e}")
                continue

        if len(img) == 2:  # Only if both bands loaded successfully
            images[idx] = np.array(img, dtype=np.float32)
            img_names.append(Path(fname).parent.name)
            idx += 1

    # Trim to actual loaded images
    images = images[:idx]
    img_names = np.array(img_names)

    logger.info(f"Successfully loaded {idx} image sets")
    return images, img_names


def extract_aion1_embeddings(
    images: np.ndarray,
    output_file: Path,
    device: str = 'mps',
    batch_size: int = 32,
    num_encoder_tokens: int = 600
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

    Returns
    -------
    embeddings : np.ndarray
        Extracted embeddings with shape (N, embedding_dim)
    """
    logger.info("Loading AION-1 model...")

    # Determine device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, trying MPS...")
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    elif device == 'mps' and not torch.backends.mps.is_available():
        logger.warning("MPS not available, using CPU...")
        device = 'cpu'

    logger.info(f"Using device: {device}")

    # Load AION-1 model
    try:
        model = AION.from_pretrained('polymathic-ai/aion-base').to(device)
        codec_manager = CodecManager(device=device)
        model.eval()
        logger.info("AION-1 model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading AION-1 model: {e}")
        raise

    # Extract embeddings in batches
    n_images = len(images)

    # Determine embedding dimension from first batch
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
        logger.info(f"Embedding dimension: {embedding_dim}")

    # Create memory-mapped file for embeddings
    logger.info(f"Creating memory-mapped file at {output_file}")
    embeddings = np.memmap(
        output_file,
        dtype='float32',
        mode='w+',
        shape=(n_images, embedding_dim)
    )

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
                # This handles variable sequence lengths from different images
                batch_embeddings = batch_embeddings.mean(dim=1)

                # Write directly to memory-mapped file
                embeddings[i:batch_end] = batch_embeddings.cpu().numpy()

            except Exception as e:
                logger.error(f"Error encoding batch {i}: {e}")
                # Fallback: create zero embeddings for this batch
                logger.warning(f"Creating zero embeddings for batch {i}")
                embeddings[i:batch_end] = 0.0

    # Flush to disk
    embeddings.flush()
    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info(f"Embeddings saved to {output_file}")

    return embeddings


def load_image_by_name(img_name: str, data_path: Path) -> np.ndarray:
    """
    Load a single galaxy image on-demand (g and i bands only).

    Parameters
    ----------
    img_name : str
        Galaxy ID (e.g., 'M1234567890123456789')
    data_path : Path
        Base data directory path

    Returns
    -------
    image : np.ndarray
        Image array with shape (2, H, W) containing [g-band, i-band]
    """
    i_file = data_path / img_name / f"{img_name}_i_results.pkl"
    g_file = data_path / img_name / f"{img_name}_g_results.pkl"

    img = []
    for band_file in [g_file, i_file]:
        with open(band_file, 'rb') as f:
            xf = pickle.load(f)
            img.append(xf['image'])

    return np.array(img, dtype=np.float32)


def plot_label_distribution(
    labels: np.ndarray,
    is_merger: np.ndarray,
    is_undisturbed: np.ndarray,
    config: Dict[str, Any],
    output_path: Path
):
    """Plot label distribution comparison."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    hist_kwargs = {
        'alpha': 0.3,
        'lw': 3,
        'bins': np.arange(0.5, 5.5),
        'density': True
    }

    ek.hist(labels[is_undisturbed], ax=ax,
            label='auto-classification: not merger', **hist_kwargs)
    ek.hist(labels[is_merger], ax=ax,
            label='auto-classification: merger', **hist_kwargs)

    ax.set_xticks(np.arange(1, 5),
                  ['undisturbed', 'ambiguous', 'merger', 'fragmented'],
                  rotation=35)
    ax.set_xlabel('Manual labels')
    ax.set_ylabel('Density')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path / 'label_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved label distribution plot")


def plot_merger_examples(
    img_names: np.ndarray,
    data_path: Path,
    candidate_indices: np.ndarray,
    n_labels_iter: np.ndarray,
    prob_labels_iter: np.ndarray,
    output_path: Path,
    filename: str,
    n_examples: int = 5,
    n_rows: int = 2
):
    """Plot example merger candidates."""
    n_examples = min(n_examples, len(candidate_indices))

    if n_examples == 0:
        logger.warning(f"No examples to plot for {filename}")
        return

    example_indices = np.random.choice(candidate_indices, n_examples, replace=False)

    fig, axarr = plt.subplots(n_rows, n_examples, figsize=(n_examples*2.5, n_rows*3.5))
    if n_rows == 1:
        axarr = axarr.reshape(1, -1)

    for idx, gix in enumerate(example_indices):
        # Load this specific image on-demand
        img_name = img_names[gix]
        image = load_image_by_name(img_name, data_path)

        # i-band (linear scale)
        ek.imshow(image[1], ax=axarr[0, idx], q=0.01, cmap='Greys')

        # i-band (log scale)
        axarr[1, idx].imshow(
            image[1],
            origin='lower',
            cmap='Greys',
            norm=colors.SymLogNorm(linthresh=0.1)
        )

        # Add statistics
        ek.text(
            0.025, 0.025,
            f"""N_labels = {n_labels_iter[gix]}
Pr[ud] = {prob_labels_iter[gix, 1]:.2f}
Pr[amb] = {prob_labels_iter[gix, 2]:.2f}
Pr[merg] = {prob_labels_iter[gix, 3]:.2f}
Pr[frag] = {prob_labels_iter[gix, 4]:.2f}""",
            ax=axarr[0, idx],
            fontsize=9,
            bordercolor='w',
            color='k',
            borderwidth=3
        )

    for ax in axarr.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    ek.text(0.05, 0.95, 'HSC i-band', ax=axarr[0, 0], fontsize=12,
            bordercolor='k', color='w', borderwidth=6)
    ek.text(0.05, 0.95, 'HSC i-band (LSB)', ax=axarr[1, 0], fontsize=12,
            bordercolor='k', color='w', borderwidth=6)

    plt.tight_layout()
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {filename}")


def plot_sfs_analysis(
    catalog: pd.DataFrame,
    prob_labels_iter: np.ndarray,
    fragmented: np.ndarray,
    output_path: Path
):
    """Plot star formation sequence analysis."""
    fig, axarr = plt.subplots(1, 2, figsize=(10, 4))

    bins = [np.logspace(7.75, 10.5, 15), np.logspace(39, 41.9, 20)]

    # Average merger probability
    im, _ = ek.pcolor_avg2d(
        10.**catalog['logmass_adjusted'],
        catalog['L_Ha'],
        catalog['p_merger'] + catalog['p_ambig'],
        cmap='coolwarm',
        yscale='log',
        xscale='log',
        zscale='log',
        bins=bins,
        ax=axarr[0],
    )

    # 2D histogram
    imx = ek.hist2d(
        10.**catalog['logmass_adjusted'],
        catalog['L_Ha'],
        cmap=ec.ColorBase('k').sequential_cmap(fade=1.),
        yscale='log',
        xscale='log',
        bins=bins,
        ax=axarr[1],
    )

    # Overlay probable mergers
    probable_merger = (catalog['p_ambig'] + catalog['p_merger']) > catalog['p_undisturbed']
    axarr[1].scatter(
        10.**catalog.loc[probable_merger, 'logmass_adjusted'],
        catalog.loc[probable_merger, 'L_Ha'],
        fc=plt.cm.coolwarm(.95),
        ec=plt.cm.coolwarm(0.8),
        s=4**2,
        label=r'Pr[merger] > Pr[undisturbed]'
    )
    axarr[1].legend(loc='lower right', fontsize=10)

    # Plot SFS line
    ms = im._coordinates.data[0, :, 0]
    alpha = -0.13*0.08 + 0.8
    norm = 1.24*0.08 - 1.47
    sfs = 10.**(alpha*(np.log10(ms) - 8.5) + norm)
    ha_sfs = calibrations.SFR2LHa(sfs)

    plt.colorbar(imx[0][-1], ax=axarr[1], label=r'$\langle N_{\rm merger} \rangle$')
    plt.colorbar(im, ax=axarr[0], label=r'$\langle {\rm Pr[merger]}\rangle$')

    for ax in axarr:
        ek.outlined_plot(ms, ha_sfs, ax=ax, lw=1, ls='--')
        ek.text(7e7, 3e39, 'EKF+24b SFS', ax=ax, rotation=37,
                coord_type='absolute', va='bottom', ha='left',
                bordercolor='w', borderwidth=2, fontsize=8)
        ax.set_xlabel(ek.common_labels['mstar'])
        ax.set_ylabel(ek.common_labels['halum'])
        ek.loglog(ax=ax)

    plt.tight_layout()
    plt.savefig(output_path / 'pmerger_sfs.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved SFS analysis plot")


def plot_mass_pdfs(
    catalog: pd.DataFrame,
    prob_labels_iter: np.ndarray,
    fragmented: np.ndarray,
    output_path: Path
):
    """Plot PDFs of L_Ha/L_Ha^MS in mass bins."""
    alpha = -0.13*0.08 + 0.8
    norm = 1.24*0.08 - 1.47
    sfs_at_mass = 10.**(alpha*(catalog['logmass_adjusted'] - 8.5) + norm)
    ha_sfs_at_mass = calibrations.SFR2LHa(sfs_at_mass)

    groups, logmstar_bins = sampling.bin_by_count(catalog['logmass_adjusted'], 4000, dx_min=0.1)

    groupids = np.unique(groups)
    ngroups = groupids.size
    fig, f_axarr = plt.subplots(ngroups, 2, figsize=(8, 1.5*ngroups),
                                gridspec_kw={'width_ratios': (1, 6)})
    axarr = f_axarr[:, 1]

    mbins = np.arange(7.5, 10.8, 0.15)

    # Compute main sequence for plotting
    ms = np.logspace(7.75, 10.5, 50)
    ha_sfs = calibrations.SFR2LHa(10.**(alpha*(np.log10(ms) - 8.5) + norm))

    for idx, gid in enumerate(groupids[::-1]):
        ms_at_mass = np.interp(catalog.loc[groups==gid, 'logmass_adjusted'],
                               np.log10(ms), np.log10(ha_sfs))

        hkwargs = {
            'alpha': 0.3,
            'lw': 3,
            'bins': np.arange(-0.25, 1.5, 0.05),
            'density': True,
            'ax': axarr[idx]
        }

        logmbounds = logmstar_bins[gid-1], logmstar_bins[gid]

        pmerger = catalog['p_merger'] + catalog['p_ambig']
        bkgcolor = ec.ColorBase('grey')

        ek.hist(np.log10(catalog.loc[groups==gid, 'L_Ha']) - ms_at_mass,
                color=bkgcolor.base, **hkwargs)
        ek.hist(np.log10(catalog.loc[groups==gid, 'L_Ha']) - ms_at_mass,
                color='r', label='Pr[merger]-weighted',
                weights=pmerger.loc[groups==gid], **hkwargs)

        dm = np.diff(mbins)[0]
        ek.hist(catalog['logmass_adjusted'], bins=mbins,
                weights=np.full(len(catalog), 1./dm),
                ax=f_axarr[idx, 0], color='lightgrey')

        mmask = ((catalog['logmass_adjusted'] > logmbounds[0]) &
                 (catalog['logmass_adjusted'] <= logmbounds[1]))
        ek.hist(catalog.loc[mmask, 'logmass_adjusted'], color=bkgcolor.base,
                ax=f_axarr[idx, 0], bins=mbins, weights=np.full(mmask.sum(), 1./dm))

    axarr[0].legend(loc='upper right')
    axarr[-1].set_xlabel(r'$\Delta (L_{\rm H\alpha}/L_{\rm H\alpha}^{MS})$')

    for ax in axarr:
        ax.set_facecolor((1, 1, 1, 0))
        ax.grid('lightgrey', ls=':', axis='x', which='both')
        ax.set_ylabel('PDF')

    for ax in axarr[1:]:
        ax.spines[['top']].set_visible(False)
    for ax in axarr[:-1]:
        ax.spines[['bottom']].set_visible(False)
        ax.set_xticklabels([])

    for ax in f_axarr[:, 0]:
        ax.set_yscale('log')
        ax.set_ylabel(r'$\frac{dN}{d(\log_{10} \rm M_\bigstar)}$')

    for ax in f_axarr[:-1, 0]:
        ax.set_xticks([])

    f_axarr[-1, 0].set_xlabel(r'$\log_{10} (\frac{\rm M_\bigstar}{M_\odot})$')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0., wspace=0.2)
    plt.savefig(output_path / 'lha_ms_pdfs.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved mass PDFs plot")


def plot_pmerger_vs_lha(
    catalog: pd.DataFrame,
    prob_labels_iter: np.ndarray,
    output_path: Path
):
    """Plot merger probability vs L_Ha/L_Ha^MS."""
    fig, axarr = plt.subplots(1, 2, figsize=(12, 5))

    loglhabins = np.arange(0., 2.5, 0.25)

    groups, logmstar_bins = sampling.bin_by_count(catalog['logmass_adjusted'], 4000, dx_min=0.1)
    groupids = np.unique(groups)

    # Compute main sequence
    ms = np.logspace(7.75, 10.5, 50)
    alpha = -0.13*0.08 + 0.8
    norm = 1.24*0.08 - 1.47
    ha_sfs = calibrations.SFR2LHa(10.**(alpha*(np.log10(ms) - 8.5) + norm))

    pmerger = catalog['p_merger'] + catalog['p_ambig']

    for idx, gid in enumerate(groupids):
        ngal = (groups == gid).sum()

        nmc = 100
        xs = sampling.midpts(loglhabins)
        yarr = np.zeros([nmc, xs.size])

        for _ in range(nmc):
            selected = catalog.loc[groups==gid].iloc[np.random.randint(0, ngal, ngal)]
            ms_at_mass = np.interp(selected['logmass_adjusted'], np.log10(ms), np.log10(ha_sfs))
            assns = np.digitize(np.log10(selected['L_Ha']) - ms_at_mass, loglhabins)

            vals = pmerger.reindex(selected.index).groupby(assns)
            yarr[_] = np.nan
            yarr[_, vals.mean().index] = np.where(vals.count() > 10, vals.mean().values, np.nan)

        for plot_idx, is_normalized in enumerate([False, True]):
            mask = np.isfinite(yarr).any(axis=0)

            ys = np.nanmedian(yarr, axis=0)[mask]
            ys_low = np.nanquantile(yarr, 0.16, axis=0)[mask]
            ys_high = np.nanquantile(yarr, 0.84, axis=0)[mask]

            if is_normalized:
                ys_low /= ys[0]
                ys_high /= ys[0]
                ys /= ys[0]

            ek.outlined_plot(xs[mask], ys, lw=1, ax=axarr[plot_idx])
            axarr[plot_idx].fill_between(
                xs[mask], ys_low, ys_high,
                label=f'[{logmstar_bins[gid-1]:.2f},{logmstar_bins[gid]:.2f}]',
                alpha=0.3,
            )

    plt.legend()

    for ax in axarr:
        ax.set_xlabel(r'$\log_{10}(L_{\rm H\alpha}/L_{\rm H\alpha}^{\rm SFS})$')
    axarr[0].set_ylabel(r'$\langle \rm Pr[merger] \rangle$')
    axarr[1].set_ylabel(r'$\langle \rm Pr[merger]/Pr[merger|SFS] \rangle$')

    plt.tight_layout()
    plt.savefig(output_path / 'pmerge_v_mass.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved Pmerger vs L_Ha plot")


def main():
    """Main analysis pipeline."""
    logger.info("=" * 80)
    logger.info("AION-1 Merger Classification Analysis")
    logger.info("=" * 80)

    # Setup paths
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    config_path = base_dir / 'config.yaml'

    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update paths
    config['data']['input_path'] = Path(config['data']['input_path'])
    config['data']['output_path'] = base_dir / 'output' / 'aion1_test'
    config['data']['output_path'].mkdir(parents=True, exist_ok=True)

    logger.info(f"Input path: {config['data']['input_path']}")
    logger.info(f"Output path: {config['data']['output_path']}")

    # Load images (g and i bands only)
    logger.info("\n" + "=" * 80)
    logger.info("Loading images...")
    logger.info("=" * 80)

    images, img_names = load_merian_images_gi_only(
        config['data']['input_path'],
        max_images=None  # Set to small number for testing
    )

    logger.info(f"Loaded {len(images)} images with shape {images.shape}")

    # Extract embeddings with AION-1
    logger.info("\n" + "=" * 80)
    logger.info("Extracting AION-1 embeddings...")
    logger.info("=" * 80)

    embeddings_file = config['data']['output_path'] / 'aion1_embeddings.npy'

    if embeddings_file.exists():
        logger.info(f"Loading existing embeddings from {embeddings_file}")
        embeddings = np.load(embeddings_file, mmap_mode='r', allow_pickle=True)
    else:
        embeddings = extract_aion1_embeddings(
            images,
            output_file=embeddings_file,
            device='mps',  # Change to 'mps' or 'cpu' as needed
            batch_size=32,
            num_encoder_tokens=600
        )
        logger.info(f"Embeddings saved to {embeddings_file}")

    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Apply PCA
    logger.info("\n" + "=" * 80)
    logger.info("Applying PCA...")
    logger.info("=" * 80)

    analyzer = EmbeddingAnalyzer(config)
    embeddings_pca = analyzer.compute_pca(embeddings)

    explained_var = analyzer.pca.explained_variance_ratio_.sum() * 100
    logger.info(f"PCA complete: {analyzer.pca.n_components_} components")
    logger.info(f"Explained variance: {explained_var:.1f}%")

    # Load labels
    logger.info("\n" + "=" * 80)
    logger.info("Loading classification labels...")
    logger.info("=" * 80)

    label_file = Path(config.get('labels', {}).get('classifications_file', ''))

    if label_file.exists():
        mergers = pd.read_csv(label_file, index_col=0)
        labels = mergers.reindex(img_names)
        labels = labels.replace(np.nan, 0).values.flatten().astype(int)

        logger.info(f"Loaded {len(labels)} labels")

        # Print distribution
        label_meanings = config.get('labels', {}).get('label_mapping', {})
        unique, counts = np.unique(labels, return_counts=True)

        logger.info("\nLabel distribution:")
        for label_val, count in zip(unique, counts):
            meaning = label_meanings.get(label_val, f"unknown_{label_val}")
            logger.info(f"  {label_val} ({meaning}): {count} objects")
    else:
        logger.warning(f"Label file not found: {label_file}")
        labels = np.zeros(len(img_names), dtype=int)

    # K-NN Label Propagation
    logger.info("\n" + "=" * 80)
    logger.info("Running K-NN label propagation...")
    logger.info("=" * 80)

    n_neighbors = config.get('labels', {}).get('n_neighbors', 15)
    n_min = config.get('labels', {}).get('minimum_labeled_neighbors', 5)
    n_min_auto = config.get('labels', {}).get('minimum_labeled_neighbors_for_autoprop', 15)

    logger.info(f"Using minimum_labeled_neighbors_for_autoprop = {n_min_auto}")
    logger.info(f"Using {n_neighbors} neighbors for weighted voting")
    logger.info(f"Using minimum_labeled_neighbors = {n_min}")

    propagator = LabelPropagation(
        n_neighbors=n_neighbors,
        n_min=n_min,
        n_min_auto=n_min_auto,
        prob_threshold=0.7,
        frag_threshold=config.get('labels', {}).get('frag_threshold', 0.25),
    )

    iterative_labels, n_labels_iter, prob_labels_iter, stats = \
        propagator.iterative_propagation(embeddings_pca, labels)

    logger.info(f"\nLabel propagation complete")
    logger.info(f"  Human labels: {stats['n_human']}")
    logger.info(f"  Auto-labels added: {stats['n_added_iteration']}")
    logger.info(f"  Total labels: {stats['n_final_auto']}")

    # Identify merger candidates
    logger.info("\n" + "=" * 80)
    logger.info("Identifying merger candidates...")
    logger.info("=" * 80)

    fragmented = prob_labels_iter[:, 4] > 0.2
    possible_merger = ((prob_labels_iter[:, 2] + prob_labels_iter[:, 3]) >
                       prob_labels_iter[:, 1])

    is_merger = possible_merger & ~fragmented
    is_undisturbed = ~possible_merger & ~fragmented

    logger.info(f"Fragmented objects: {fragmented.sum()}")
    logger.info(f"Possible mergers: {possible_merger.sum()}")
    logger.info(f"Merger candidates (excluding fragmented): {is_merger.sum()}")

    # Generate plots
    logger.info("\n" + "=" * 80)
    logger.info("Generating plots...")
    logger.info("=" * 80)

    output_path = config['data']['output_path']

    # 1. Label distribution
    if (labels > 0).any():
        plot_label_distribution(labels, is_merger, is_undisturbed, config, output_path)

    # 2-4. Merger example plots
    data_path = config['data']['input_path']
    manual_merger = (labels == 3) | (labels == 2)
    manual_nonmerger = labels == 1

    # Conflicts: auto says not merger, manual says merger
    candidates_1 = np.arange(len(img_names))[possible_merger & ~fragmented & manual_nonmerger]
    plot_merger_examples(img_names, data_path, candidates_1, n_labels_iter,
                        prob_labels_iter, output_path,
                        'merger_examples_conflict1.png', n_examples=5, n_rows=2)

    # Conflicts: auto says merger, manual says not merger
    candidates_2 = np.arange(len(img_names))[~possible_merger & ~fragmented & manual_merger]
    plot_merger_examples(img_names, data_path, candidates_2, n_labels_iter,
                        prob_labels_iter, output_path,
                        'merger_examples_conflict2.png', n_examples=5, n_rows=2)

    # All merger candidates
    candidates_3 = np.arange(len(img_names))[possible_merger & ~fragmented]
    plot_merger_examples(img_names, data_path, candidates_3, n_labels_iter,
                        prob_labels_iter, output_path,
                        'merger_examples_all.png', n_examples=10, n_rows=2)

    # Load catalog for SFS analysis
    logger.info("\n" + "=" * 80)
    logger.info("Loading catalog for SFS analysis...")
    logger.info("=" * 80)

    try:
        catalog_file = '/Users/kadofong/work/projects/merian/local_data/base_catalogs/mdr1_n708maglt26_and_pzgteq0p1.parquet'
        full_catalog, masks = sample.load_sample(catalog_file)
        base_catalog = full_catalog.loc[masks['is_good'][0]]

        datadir = config['data']['input_path']
        for sid in tqdm(base_catalog.index, desc="Loading adjusted masses"):
            filename = f'{datadir}/{sid}/{sid}_i_results.pkl'
            if not os.path.exists(filename):
                continue
            with open(filename, 'rb') as f:
                x = pickle.load(f)
            base_catalog.loc[sid, 'logmass_adjusted'] = x['logmass_adjusted']

        base_catalog.loc[base_catalog['logmass_adjusted'].isna(), 'logmass_adjusted'] = \
            base_catalog.loc[base_catalog['logmass_adjusted'].isna(), 'logmass']

        # Create catalog with merger probabilities
        fragmented_cat = prob_labels_iter[:, 4] > 0.3
        catalog = base_catalog.reindex(img_names[~fragmented_cat])
        catalog['p_merger'] = np.where(
            (prob_labels_iter[~fragmented_cat] == 0).all(axis=1),
            np.nan,
            prob_labels_iter[~fragmented_cat, 3]
        )
        catalog['p_ambig'] = np.where(
            (prob_labels_iter[~fragmented_cat] == 0).all(axis=1),
            np.nan,
            prob_labels_iter[~fragmented_cat, 2]
        )
        catalog['p_undisturbed'] = np.where(
            (prob_labels_iter[~fragmented_cat] == 0).all(axis=1),
            np.nan,
            prob_labels_iter[~fragmented_cat, 1]
        )

        dm = catalog['logmass_adjusted'] - catalog['logmass']
        catalog = catalog.loc[dm < 0.5]

        # Generate SFS plots
        plot_sfs_analysis(catalog, prob_labels_iter, fragmented_cat, output_path)
        plot_mass_pdfs(catalog, prob_labels_iter, fragmented_cat, output_path)
        plot_pmerger_vs_lha(catalog, prob_labels_iter, output_path)

    except Exception as e:
        logger.warning(f"Could not generate SFS analysis plots: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("Analysis complete!")
    logger.info(f"All figures saved to: {output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
