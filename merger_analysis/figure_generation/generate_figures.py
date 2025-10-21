#!/usr/bin/env python3
"""
Generate Galaxy Example Figures for Merger Analysis

This script creates publication-quality figure grids showing galaxy examples
from BYOL merger classification analysis. It loads pre-computed embeddings,
PCA/UMAP reductions, and label probabilities to select representative galaxies.

Features:
- Transposed grid layout: galaxies as rows, visualization types as columns
- Three visualization types: r-N708-i RGB, HSC i-band (LSB), Starlet HF
- Automatic galaxy selection based on classification probabilities
- Support for both high-mass and low-mass samples

Usage:
    # Generate figures using default config
    python generate_figures.py

    # Specify custom config
    python generate_figures.py --config ../custom_config.yaml

    # Select specific galaxies by index
    python generate_figures.py --galaxy-indices 100,200,300,400

    # Choose mass regime
    python generate_figures.py --mass-regime lowmass

    # Specify output filename
    python generate_figures.py --output galaxy_examples.pdf
"""

import os
import sys
import argparse
import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm import tqdm

# Add pieridae to path (go up 2 levels: figure_generation/ -> merger_analysis/ -> pieridae/)
sys.path.insert(0, str(Path(__file__).parents[2]))

from pieridae.starbursts import sample
from carpenter import conventions, pixels
from astropy.visualization import make_lupton_rgb
from astropy import coordinates

# Plotting utilities
try:
    from ekfplot import plot as ek
    EKFPLOT_AVAILABLE = True
except ImportError:
    EKFPLOT_AVAILABLE = False
    print("Warning: ekfplot not available, using standard matplotlib")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Convert paths to Path objects
    config['data']['input_path'] = Path(config['data']['input_path'])
    config['data']['output_path'] = Path(config['data']['output_path'])

    return config


def setup_logging(level: str = 'INFO') -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger('generate_figures')
    logger.setLevel(getattr(logging, level))

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def load_analysis_results(output_path: Path, logger: logging.Logger = None) -> dict:
    """
    Load pre-computed analysis results from merger analysis pipeline.

    Parameters
    ----------
    output_path : Path
        Path to output directory containing analysis results
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    dict
        Dictionary containing analysis results including embeddings, PCA, UMAP, etc.
    """
    if logger:
        logger.info("Loading analysis results...")

    # Try to load merger_analysis_results.pkl first (from notebook workflow)
    results_file = output_path / 'merger_analysis_results.pkl'
    if results_file.exists():
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        if logger:
            logger.info(f"Loaded results from: {results_file}")
        return results

    # Fall back to dimensionality_reduction_results.pkl (from run_analysis.py)
    results_file = output_path / 'dimensionality_reduction_results.pkl'
    if not results_file.exists():
        raise FileNotFoundError(
            f"No analysis results found in {output_path}. "
            "Please run merger analysis first (merger_classification.ipynb or run_analysis.py)"
        )

    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    if logger:
        logger.info(f"Loaded results from: {results_file}")

    return results


def load_images(data_path: Path, img_names: np.ndarray, logger: logging.Logger = None) -> np.ndarray:
    """
    Load galaxy images from pickle files.

    Parameters
    ----------
    data_path : Path
        Path to directory containing image data
    img_names : np.ndarray
        Array of image names (Merian IDs)
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    np.ndarray
        Image array with shape (N, 3, H, W)
    """
    if logger:
        logger.info(f"Loading {len(img_names)} images from {data_path}...")

    images = []
    for name in tqdm(img_names, desc="Loading images", disable=(logger is None)):
        img = []
        for band in 'gi':
            filename = data_path / name / f'{name}_{band}_results.pkl'

            try:
                with open(filename, 'rb') as f:
                    xf = pickle.load(f)
                    img.append(xf['image'])
                    if band == 'i':
                        img.append(xf['hf_image'])
            except FileNotFoundError:
                if logger:
                    logger.warning(f"File not found: {filename}")
                continue

        if len(img) == 3:
            images.append(np.array(img))

    images = np.array(images)

    if logger:
        logger.info(f"Loaded {len(images)} images with shape: {images.shape}")

    return images


def load_labels(config: dict, img_names: np.ndarray, logger: logging.Logger = None) -> np.ndarray:
    """Load classification labels if available"""
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


def compute_label_probabilities(
    results: dict,
    labels: np.ndarray,
    config: dict,
    logger: logging.Logger = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute K-NN based label probabilities.

    This replicates the label propagation logic from merger_classification.ipynb.

    Parameters
    ----------
    results : dict
        Analysis results containing PCA embeddings
    labels : np.ndarray
        Classification labels
    config : dict
        Configuration dictionary
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    prob_labels : np.ndarray
        Probability labels for each class
    n_labels : np.ndarray
        Number of labeled neighbors for each galaxy
    iterative_labels : np.ndarray
        Labels after iterative propagation
    """
    from sklearn.neighbors import NearestNeighbors

    if logger:
        logger.info("Computing label probabilities...")

    # Check if probabilities are already in results
    if 'prob_labels' in results and 'n_labels' in results:
        if logger:
            logger.info("Using pre-computed label probabilities from results")
        return results['prob_labels'], results['n_labels'], results.get('iterative_labels', labels)

    # Otherwise compute from scratch
    pca_embeddings = results['embeddings_pca']
    n_neighbors = config.get('labels', {}).get('n_neighbors', 50)
    n_min = config.get('labels', {}).get('minimum_labeled_neighbors', 5)

    if logger:
        logger.info(f"Computing K-NN with {n_neighbors} neighbors...")

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(pca_embeddings)
    distances, indices = nbrs.kneighbors(pca_embeddings)
    distances[:, 0] = np.nan

    neighbor_labels = labels[indices]

    weights = np.where(neighbor_labels > 0, 1. / distances, 0.)
    with np.errstate(invalid='ignore'):
        weights /= np.nansum(weights, axis=1).reshape(-1, 1)

    prob_labels = np.zeros([pca_embeddings.shape[0], labels.max() + 1])

    for ix in range(labels.max() + 1):
        prob_labels[:, ix] = np.nansum(np.where(neighbor_labels == ix, weights, 0), axis=1)

    n_labels = np.sum(neighbor_labels > 0, axis=1)
    prob_labels[n_labels < n_min] = 0.

    if logger:
        logger.info(f"{(prob_labels > 0).any(axis=1).sum()} galaxies have auto-labels")

    # Iterative label propagation
    iterative_labels = labels.copy()
    n_min_auto = config.get('labels', {}).get('minimum_labeled_neighbors_for_autoprop', 10)
    prob_threshold = config.get('labels', {}).get('prob_threshold', 0.7)

    neighbor_labels = iterative_labels[indices]
    n_labels_iter = np.sum(neighbor_labels > 0, axis=1)

    additions = np.where(prob_labels[n_labels_iter >= n_min_auto] > prob_threshold)
    new_labels = np.zeros_like(iterative_labels)
    new_labels[additions[0]] = additions[1]

    # Handle fragmentation specially
    frag_threshold = config.get('labels', {}).get('frag_threshold', 0.25)
    new_labels[(prob_labels[:, 4] > frag_threshold) & (n_labels_iter >= n_min_auto)] = 4

    iterative_labels[iterative_labels == 0] = new_labels[iterative_labels == 0]
    is_iterative = labels != iterative_labels

    if logger:
        n_added = (iterative_labels > 0).sum() - (labels > 0).sum()
        logger.info(f"Added {n_added} auto-labels via iterative propagation")

    # Recompute probabilities with iterative labels
    neighbor_labels = iterative_labels[indices]
    weights = np.where(neighbor_labels > 0, 1. / distances, 0.)
    weights[is_iterative] *= 0.1
    with np.errstate(invalid='ignore'):
        weights /= np.nansum(weights, axis=1).reshape(-1, 1)

    prob_labels = np.zeros([pca_embeddings.shape[0], labels.max() + 1])
    for ix in range(labels.max() + 1):
        prob_labels[:, ix] = np.nansum(np.where(neighbor_labels == ix, weights, 0), axis=1)

    n_labels = np.sum(neighbor_labels > 0, axis=1)
    prob_labels[n_labels < n_min] = 0.

    return prob_labels, n_labels, iterative_labels


def select_galaxies(
    prob_labels: np.ndarray,
    catalog: pd.DataFrame,
    img_names: np.ndarray,
    mass_regime: str = 'lowmass',
    specific_indices: Optional[List[int]] = None,
    logger: logging.Logger = None
) -> Tuple[List[int], List[str]]:
    """
    Select representative galaxies for visualization.

    Parameters
    ----------
    prob_labels : np.ndarray
        Probability labels for each class
    catalog : pd.DataFrame
        Galaxy catalog
    img_names : np.ndarray
        Image names
    mass_regime : str
        'lowmass' or 'highmass'
    specific_indices : list of int, optional
        Specific galaxy indices to use
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    selected_indices : list of int
        Indices of selected galaxies
    galaxy_labels : list of str
        Labels for each galaxy
    """
    if specific_indices is not None:
        if logger:
            logger.info(f"Using specified galaxy indices: {specific_indices}")
        return specific_indices, [f'Galaxy {i}' for i in range(len(specific_indices))]

    if logger:
        logger.info(f"Selecting galaxies for {mass_regime} regime...")

    # Define mass cut
    catalog_aligned = catalog.reindex(img_names)
    if mass_regime == 'lowmass':
        mass_cut = catalog_aligned['logmass_adjusted'] < 9.0
    else:
        mass_cut = catalog_aligned['logmass_adjusted'] >= 9.0

    # Identify galaxy types
    fragmented = prob_labels[:, 4] > 0.3
    possible_merger = (prob_labels[:, 3] + prob_labels[:, 2]) > prob_labels[:, 1]

    # Pre-selected examples (from notebook)
    if mass_regime == 'highmass':
        selected = [9142, 8831, 7258, 3876]
        labels = ['Likely merger', 'Likely ambiguous', 'Likely undisturbed', 'Likely fragmented']
    elif mass_regime == 'lowmass':
        selected = [11063, 11896, 7554, 9902]
        labels = ['Likely merger', 'Likely ambiguous', 'Likely undisturbed', 'Likely fragmented']
    else:
        raise ValueError(f"Unknown mass regime: {mass_regime}")

    if logger:
        logger.info(f"Selected {len(selected)} galaxies")
        logger.info(f"Galaxy indices: {selected}")

    return selected, labels


def load_cutouts(
    selected_indices: List[int],
    img_names: np.ndarray,
    catalog: pd.DataFrame,
    figure_data_dir: Path,
    logger: logging.Logger = None
) -> dict:
    """
    Load FITS cutouts for selected galaxies.

    Parameters
    ----------
    selected_indices : list of int
        Indices of galaxies to load
    img_names : np.ndarray
        Image names
    catalog : pd.DataFrame
        Galaxy catalog
    figure_data_dir : Path
        Directory containing figure data
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    dict
        Dictionary mapping galaxy index to BBMBImage object
    """
    if logger:
        logger.info("Loading FITS cutouts...")

    bbmb_dict = {}

    for gid in selected_indices:
        targetid = img_names[gid]
        objname = conventions.produce_merianobjectname(
            *catalog.loc[targetid, ['RA', 'DEC']].values
        )
        bbmb = pixels.BBMBImage()

        success = True
        for band in ['r', 'N708', 'i']:
            if band in ['N708', 'N540']:
                cutout = figure_data_dir / 'merian' / f'{objname}_{band}_merim.fits'
            else:
                cutout = figure_data_dir / 'hsc' / f'{objname}_HSC-{band}.fits'

            if not cutout.exists():
                if logger:
                    logger.warning(f"Cutout not found: {cutout}")
                success = False
                break

            bbmb.add_band(
                band,
                coordinates.SkyCoord(
                    catalog.loc[targetid, 'RA'],
                    catalog.loc[targetid, 'DEC'],
                    unit='deg'
                ),
                size=150,
                image=str(cutout),
                var=str(cutout),
                image_ext=1,
                var_ext=3,
            )

        if success:
            bbmb_dict[gid] = bbmb
        else:
            bbmb_dict[gid] = None
            if logger:
                logger.warning(f"Skipping galaxy {targetid} due to missing cutouts")

    return bbmb_dict


def create_figure(
    selected_indices: List[int],
    galaxy_labels: List[str],
    images: np.ndarray,
    prob_labels: np.ndarray,
    n_labels: np.ndarray,
    img_names: np.ndarray,
    bbmb_dict: dict,
    config: dict,
    output_file: Path,
    logger: logging.Logger = None
) -> None:
    """
    Create transposed galaxy examples figure.

    Parameters
    ----------
    selected_indices : list of int
        Indices of selected galaxies
    galaxy_labels : list of str
        Labels for each galaxy
    images : np.ndarray
        Image array
    prob_labels : np.ndarray
        Probability labels
    n_labels : np.ndarray
        Number of labeled neighbors
    img_names : np.ndarray
        Image names
    bbmb_dict : dict
        Dictionary of BBMBImage objects
    config : dict
        Configuration dictionary
    output_file : Path
        Output file path
    logger : logging.Logger, optional
        Logger instance
    """
    if logger:
        logger.info("Creating figure...")

    n_galaxies = len(selected_indices)
    n_viz_types = 3

    fig, axarr = plt.subplots(n_galaxies, n_viz_types, figsize=(8, 2.5 * n_galaxies))

    # Handle case where we only have one galaxy (axarr would be 1D)
    if n_galaxies == 1:
        axarr = axarr.reshape(1, -1)

    for row_idx, gix in enumerate(selected_indices):
        # Column 0: r-N708-i RGB image
        bbmb = bbmb_dict.get(gix)
        if bbmb is None:
            # Fall back to i-band grayscale
            if EKFPLOT_AVAILABLE:
                ek.imshow(
                    images[gix][1],
                    origin='lower',
                    cmap='Greys',
                    q=0.01,
                    ax=axarr[row_idx, 0]
                )
            else:
                axarr[row_idx, 0].imshow(
                    images[gix][1],
                    origin='lower',
                    cmap='Greys'
                )
        else:
            rgb = make_lupton_rgb(
                bbmb.image['r'],
                bbmb.image['N708'],
                bbmb.image['i'],
                Q=3,
                stretch=2.
            )
            if EKFPLOT_AVAILABLE:
                ek.imshow(rgb, ax=axarr[row_idx, 0])
            else:
                axarr[row_idx, 0].imshow(rgb, origin='lower')

        # Column 1: HSC i-band (LSB with SymLog normalization)
        axarr[row_idx, 1].imshow(
            images[gix][1],
            origin='lower',
            cmap='Greys',
            norm=colors.SymLogNorm(linthresh=0.1)
        )

        # Column 2: Starlet HF
        if EKFPLOT_AVAILABLE:
            ek.imshow(
                images[gix][2],
                ax=axarr[row_idx, 2],
                cmap='magma',
                q=0.005
            )
        else:
            axarr[row_idx, 2].imshow(
                images[gix][2],
                origin='lower',
                cmap='magma'
            )

        # Add probability labels to first column
        prob_text = f'''N_labels = {n_labels[gix]}
Pr[ud] = {prob_labels[gix, 1]:.2f}
Pr[amb] = {prob_labels[gix, 2]:.2f}
Pr[merg] = {prob_labels[gix, 3]:.2f}
Pr[frag] = {prob_labels[gix, 4]:.2f}'''

        if EKFPLOT_AVAILABLE:
            ek.text(
                0.025, 0.025,
                prob_text,
                ax=axarr[row_idx, 0],
                fontsize=10,
                bordercolor='w',
                color='k',
                borderwidth=3
            )
        else:
            axarr[row_idx, 0].text(
                0.025, 0.025,
                prob_text,
                transform=axarr[row_idx, 0].transAxes,
                fontsize=10,
                color='k',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

        # Load stellar mass from results file
        data_path = config['data']['input_path']
        results_file = data_path / img_names[gix] / f"{img_names[gix]}_i_results.pkl"

        if results_file.exists():
            with open(results_file, 'rb') as f:
                x = pickle.load(f)
                logmstar = x.get('logmass_adjusted', np.nan)

            mass_text = rf'$\log_{{10}}(\frac{{M_{{\bigstar}}}}{{M_\odot}})={logmstar:.2f}$'

            if EKFPLOT_AVAILABLE:
                ek.text(
                    0.025, 0.025,
                    mass_text,
                    ax=axarr[row_idx, 1],
                    fontsize=10,
                    bordercolor='w',
                    color='k',
                    borderwidth=3
                )
            else:
                axarr[row_idx, 1].text(
                    0.025, 0.025,
                    mass_text,
                    transform=axarr[row_idx, 1].transAxes,
                    fontsize=10,
                    color='k',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )

    # Add column headers and galaxy labels
    for row_idx, label in enumerate(galaxy_labels):
        if EKFPLOT_AVAILABLE:
            ek.text(0.05, 0.95, label, ax=axarr[row_idx, 0], fontsize=12,
                    bordercolor='k', color='w', borderwidth=6)
        else:
            axarr[row_idx, 0].text(
                0.05, 0.95, label,
                transform=axarr[row_idx, 0].transAxes,
                fontsize=12, color='w',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8)
            )

    # Add visualization type labels to first row
    viz_labels = ['r-N708-i RGB', 'HSC i-band (LSB)', 'HF feature reconstruction']
    for col_idx, label in enumerate([None, viz_labels[1], viz_labels[2]]):
        if label and EKFPLOT_AVAILABLE:
            ek.text(0.05, 0.95, label, ax=axarr[0, col_idx], fontsize=11,
                    bordercolor='k', color='w', borderwidth=6)
        elif label:
            axarr[0, col_idx].text(
                0.05, 0.95, label,
                transform=axarr[0, col_idx].transAxes,
                fontsize=11, color='w',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8)
            )

    # Remove ticks from all axes
    for ax in axarr.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(wspace=-0.17, hspace=0.05)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"Figure saved to: {output_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate galaxy example figures from merger analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate figures using default config
  python generate_figures.py

  # Use custom config
  python generate_figures.py --config ../custom_config.yaml

  # Specify galaxy indices
  python generate_figures.py --galaxy-indices 100,200,300,400

  # Choose mass regime
  python generate_figures.py --mass-regime lowmass

  # Custom output file
  python generate_figures.py --output my_galaxies.pdf
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='../config.yaml',
        help='Path to configuration YAML file'
    )

    parser.add_argument(
        '--mass-regime',
        type=str,
        choices=['lowmass', 'highmass'],
        default='lowmass',
        help='Mass regime for galaxy selection'
    )

    parser.add_argument(
        '--galaxy-indices',
        type=str,
        help='Comma-separated list of galaxy indices to visualize'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output filename (default: galaxy_examples_<mass_regime>.pdf)'
    )

    parser.add_argument(
        '--figure-data-dir',
        type=str,
        default='./figure_data/',
        help='Directory containing FITS cutouts'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging('INFO')

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Configuration loaded from: {args.config}")
        logger.info(f"Input path: {config['data']['input_path']}")
        logger.info(f"Output path: {config['data']['output_path']}")

        # Load analysis results
        results = load_analysis_results(config['data']['output_path'], logger)

        # Get image names from results
        img_names = results['img_names']

        # Load images
        images = load_images(config['data']['input_path'], img_names, logger)

        # Load labels
        labels = load_labels(config, img_names, logger)
        if labels is None:
            logger.warning("No labels found, using zeros")
            labels = np.zeros(len(img_names), dtype=int)

        # Compute label probabilities
        prob_labels, n_labels, iterative_labels = compute_label_probabilities(
            results, labels, config, logger
        )

        # Load catalog
        logger.info("Loading catalog...")
        catalog, masks = sample.load_sample()
        logger.info(f"Loaded catalog with {len(catalog)} objects")

        # Load adjusted masses into catalog if needed
        if 'logmass_adjusted' not in catalog.columns:
            logger.info("Loading adjusted masses from individual files...")
            catalog['logmass_adjusted'] = np.nan
            for sid in tqdm(catalog.index, desc="Loading masses"):
                filename = config['data']['input_path'] / sid / f'{sid}_i_results.pkl'
                if filename.exists():
                    with open(filename, 'rb') as f:
                        x = pickle.load(f)
                    catalog.loc[sid, 'logmass_adjusted'] = x.get('logmass_adjusted', np.nan)

            # Fill missing with original logmass
            catalog.loc[catalog['logmass_adjusted'].isna(), 'logmass_adjusted'] = \
                catalog.loc[catalog['logmass_adjusted'].isna(), 'logmass']

        # Parse galaxy indices if provided
        specific_indices = None
        if args.galaxy_indices:
            specific_indices = [int(x.strip()) for x in args.galaxy_indices.split(',')]

        # Select galaxies
        selected_indices, galaxy_labels = select_galaxies(
            prob_labels, catalog, img_names, args.mass_regime,
            specific_indices, logger
        )

        # Load cutouts
        figure_data_dir = Path(args.figure_data_dir)
        bbmb_dict = load_cutouts(
            selected_indices, img_names, catalog, figure_data_dir, logger
        )

        # Determine output filename
        if args.output:
            output_file = Path(args.output)
        else:
            output_file = Path(f'galaxy_examples_{args.mass_regime}.pdf')

        # Create figure
        create_figure(
            selected_indices, galaxy_labels, images, prob_labels, n_labels,
            img_names, bbmb_dict, config, output_file, logger
        )

        logger.info("=" * 60)
        logger.info("SUCCESS")
        logger.info("=" * 60)
        print(f"\n✅ Figure generation completed successfully!")
        print(f"   Output: {output_file}")

    except Exception as e:
        logger.error(f"Error during figure generation: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
