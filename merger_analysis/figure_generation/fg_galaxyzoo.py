#!/usr/bin/env python3
"""
Generate Galaxy Zoo Edge-on Disk Analysis Figure

This script reproduces the Galaxy Zoo edge-on disk (EoD) classification analysis
from the exploratory notebook and creates a publication-quality 3-panel figure.

Features:
- Loads BYOL embeddings, PCA reductions, and label propagations
- Matches with Galaxy Zoo classifications
- Computes EoD fraction and stellar mass distributions
- Shows examples of EoD candidates in a 3x3 grid

Usage:
    # Generate figure using default config
    python fg_galaxyzoo.py

    # Specify custom config
    python fg_galaxyzoo.py --config ../configs/galaxyzoo.yaml

    # Specify output directory
    python fg_galaxyzoo.py --output-dir ../output/figures/
"""

import os
import sys
import argparse
import logging
import pickle
import glob
from pathlib import Path
from typing import Dict, Tuple, Optional

import yaml
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from astropy import table

# Add pieridae to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from pieridae.starbursts.byol import (
    EmbeddingAnalyzer,
    LabelPropagation,
)
from pieridae.starbursts import sample
from ekfplot import plot as ek, colorlists
from ekfstats import sampling
from ekfparse import query


def setup_logging(level: str = 'INFO') -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger('fg_galaxyzoo')
    logger.setLevel(getattr(logging, level))

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['data']['input_path'] = Path(config['data']['input_path'])
    config['data']['output_path'] = Path(config['data']['output_path'])

    return config


def load_image_by_name(img_name: str, data_path: Path) -> np.ndarray:
    """
    Load a single galaxy image on-demand.

    Parameters
    ----------
    img_name : str
        Galaxy ID (e.g., 'M1234567890123456789')
    data_path : Path
        Base data directory path

    Returns
    -------
    image : np.ndarray
        Image array with shape (3, H, W) containing [g-band, i-band, hf_i-band]
    """
    i_file = data_path / img_name / f"{img_name}_i_results.pkl"
    g_file = data_path / img_name / f"{img_name}_g_results.pkl"

    img = []
    for band_file in [g_file, i_file]:
        with open(band_file, 'rb') as f:
            xf = pickle.load(f)
            img.append(xf['image'])
            if band_file == i_file:
                img.append(xf['hf_image'])  # Add HF image for i-band

    return np.array(img, dtype=np.float32)


def load_data(config: dict, logger: logging.Logger) -> Dict:
    """
    Load all data needed for figure generation.

    This function consolidates all data loading operations from the notebook
    into a single place, including:
    - Embeddings and PCA reduction
    - Image names and labels
    - Label propagation results
    - Full catalog with adjusted masses
    - Galaxy Zoo classifications and matching

    Parameters
    ----------
    config : dict
        Configuration dictionary
    logger : logging.Logger
        Logger instance

    Returns
    -------
    data : dict
        Dictionary containing all loaded data
    """
    logger.info("=" * 60)
    logger.info("LOADING DATA")
    logger.info("=" * 60)

    data = {}

    # Get paths
    data_path = config['data']['input_path']
    output_path = config['data']['output_path']

    logger.info(f"Input path: {data_path}")
    logger.info(f"Output path: {output_path}")

    # Load image names
    logger.info("Loading image names...")
    pattern = f"{data_path}/M*/*i_results.pkl"
    filenames = glob.glob(pattern)

    valid_files = []
    for fname in filenames:
        g_file = fname.replace('_i_', '_g_')
        i_file = fname
        if os.path.exists(g_file) and os.path.exists(i_file):
            valid_files.append(fname)

    img_names = np.array([Path(fname).parent.name for fname in valid_files])
    data['img_names'] = img_names
    data['data_path'] = data_path
    logger.info(f"Found {len(img_names)} images")

    # Load embeddings
    logger.info("Loading embeddings...")
    embeddings_file = output_path / 'embeddings.npy'
    if not embeddings_file.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_file}")

    embeddings = np.load(embeddings_file)
    data['embeddings'] = embeddings
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Compute PCA
    logger.info("Computing PCA...")
    analyzer = EmbeddingAnalyzer(config)
    embeddings_pca = analyzer.compute_pca(embeddings)
    data['embeddings_pca'] = embeddings_pca
    explained_var = analyzer.pca.explained_variance_ratio_.sum() * 100
    logger.info(f"PCA complete: {analyzer.pca.n_components_} components, {explained_var:.1f}% variance")

    # Load labels
    logger.info("Loading classification labels...")
    label_file = Path(config.get('labels', {}).get('classifications_file', ''))

    if label_file.exists():
        mergers = pd.read_csv(label_file, index_col=0)
        labels = mergers.reindex(img_names)
        labels = labels.replace(np.nan, 0).values.flatten().astype(int)
        data['labels'] = labels

        label_meanings = config.get('labels', {}).get('label_mapping', {})
        data['label_meanings'] = label_meanings
        unique, counts = np.unique(labels, return_counts=True)

        logger.info("Label distribution:")
        for label_val, count in zip(unique, counts):
            meaning = label_meanings.get(label_val, f"unknown_{label_val}")
            logger.info(f"  {label_val} ({meaning}): {count} objects")
    else:
        logger.warning(f"Label file not found: {label_file}")
        labels = np.zeros(len(img_names), dtype=int)
        data['labels'] = labels
        data['label_meanings'] = {0: 'unclassified'}

    # Run label propagation
    logger.info("Running label propagation...")
    n_neighbors = config.get('labels', {}).get('n_neighbors', 50)
    n_min = config.get('labels', {}).get('minimum_labeled_neighbors', 5)
    n_min_auto = config.get('labels', {}).get('minimum_labeled_neighbors_for_autoprop', 10)

    propagator = LabelPropagation(
        n_neighbors=n_neighbors,
        n_min=n_min,
        n_min_auto=n_min_auto,
        prob_threshold=config.get('labels', {}).get('prob_threshold', 0.9),
        frag_threshold=config.get('labels', {}).get('frag_threshold', 0.25),
    )

    iterative_labels, n_labels_iter, prob_labels_iter, stats = \
        propagator.iterative_propagation(embeddings_pca, labels, handle_fragmentation_separately=False, handle_mergers_separately=False)

    data['iterative_labels'] = iterative_labels
    data['n_labels_iter'] = n_labels_iter
    data['prob_labels_iter'] = prob_labels_iter

    logger.info(f"Human labels: {stats['n_human']}")
    logger.info(f"Auto-labels added: {stats['n_added_iteration']}")
    logger.info(f"Total labels: {stats['n_final_auto']}")

    # Load full catalog
    logger.info("Loading full catalog...")
    full_catalog, masks = sample.load_sample(
        '/Users/kadofong/work/projects/merian/local_data/base_catalogs/mdr1_n708maglt26_and_pzgteq0p1.parquet'
    )
    base_catalog = full_catalog.loc[masks['is_good'][0]]

    # Load adjusted masses
    logger.info("Loading adjusted masses from individual files...")
    datadir = '../../local_data/pieridae_output/starlet/msorabove_v0/'
    for sid in tqdm(base_catalog.index, desc="Loading masses"):
        filename = f'{datadir}/{sid}/{sid}_i_results.pkl'
        if not os.path.exists(filename):
            continue
        with open(filename, 'rb') as f:
            x = pickle.load(f)
        base_catalog.loc[sid, 'logmass_adjusted'] = x['logmass_adjusted']

    base_catalog.loc[base_catalog['logmass_adjusted'].isna(), 'logmass_adjusted'] = \
        base_catalog.loc[base_catalog['logmass_adjusted'].isna(), 'logmass']

    dm = base_catalog['logmass_adjusted'] - base_catalog['logmass']
    base_catalog_trimmed = base_catalog.loc[dm < 0.5]

    # Create catalog subset
    catalog = base_catalog_trimmed.reindex(img_names)
    dm = catalog['logmass_adjusted'] - catalog['logmass']
    catalog = catalog.loc[
        (dm < 0.5) &
        (catalog['logmass_adjusted'] <= 10.5) &
        (catalog['logmass_adjusted'] >= 7.5)
    ]

    # Load Galaxy Zoo classifications
    logger.info("Loading Galaxy Zoo classifications...")
    gz = table.Table.read(
        '../local_data/galaxy_zoo_classifications/dr5/gzdv5.dat',
        readme='../local_data/galaxy_zoo_classifications/dr5/ReadMe',
        format='cds'
    )
    gz_df = gz.to_pandas()

    logger.info("Matching catalogs with Galaxy Zoo...")
    mmatch, gzmatch = query.match_catalogs(catalog, gz_df, coordkeysB=['RAdeg', 'DEdeg'])
    gzmatch = gzmatch.reset_index().set_index(mmatch.index)

    data['catalog'] = catalog
    data['gzmatch'] = gzmatch
    logger.info(f"Catalog size: {len(catalog)} objects")
    logger.info(f"Galaxy Zoo matches: {len(gzmatch)} objects")

    # Compute probabilities for catalog
    catalog['p_eod'] = prob_labels_iter[:, 2]
    catalog['p_not_eod'] = prob_labels_iter[:, 1]

    # Define has_classification
    has_classification = (prob_labels_iter > 0).any(axis=1) & (catalog['logmass_adjusted'].values > 9.)
    data['has_classification'] = has_classification

    logger.info(f"Objects with classification: {has_classification.sum()}")

    logger.info("Data loading complete!")
    logger.info("=" * 60)

    return data


def make_figure_galaxyzoo(
    data: Dict,
    output_dir: Path,
    logger: logging.Logger,
    seed: int = 1209
) -> None:
    """
    Generate Galaxy Zoo EoD analysis figure with 3 panels.

    Creates a figure with:
    - Left panel: Edge-on disk fraction vs stellar mass
    - Middle panel: Stellar mass PDF distributions
    - Right panel: 3x3 grid of example EoD candidate images

    Parameters
    ----------
    data : dict
        Data dictionary from load_data()
    output_dir : Path
        Output directory for figures
    logger : logging.Logger
        Logger instance
    seed : int
        Random seed for image selection
    """
    logger.info("Generating Galaxy Zoo EoD analysis figure")

    catalog = data['catalog']
    gzmatch = data['gzmatch']
    prob_labels_iter = data['prob_labels_iter']
    has_classification = data['has_classification']
    img_names = data['img_names']
    data_path = data['data_path']

    # Set up histogram bins
    histkwargs = {
        'bins': np.linspace(
            *np.quantile(catalog.reindex(gzmatch.index).loc[:, 'rmag'], [0.025, 0.99]),
            10
        )
    }

    # Compute histograms
    logger.info("Computing bootstrap histograms...")
    hasvotes = np.isfinite(catalog.reindex(gzmatch.index).loc[:, 'rmag']) & \
               np.isfinite(gzmatch['DEOyes']) & (gzmatch['NbDEO']>1)
    fullgz_counts, _ = np.histogram(
        catalog.reindex(gzmatch.index).loc[hasvotes, 'rmag'],
        **histkwargs
    )
    mindices = sampling.make_matched_sample(catalog['rmag'], catalog.reindex(gzmatch.index)['rmag']).index
    matched = catalog.reindex(mindices)
    full_counts, _ = np.histogram(matched.loc[:, 'rmag'], **histkwargs)

    pweighted_counts = sampling.bootstrap_histcounts(
        catalog.reindex(gzmatch.index).loc[:, 'rmag'],
        weights=catalog.reindex(gzmatch.index).loc[:, 'p_eod'],
        **histkwargs
    )
    pweighted_fullcounts = sampling.bootstrap_histcounts(
        matched.loc[:, 'rmag'],
        weights=matched.loc[:, 'p_eod'],
        **histkwargs
    )


    gzhc_counts = sampling.bootstrap_histcounts(
        catalog.reindex(gzmatch.index).loc[hasvotes & (gzmatch['DEOyes'] > 0.5), 'rmag'],
        **histkwargs
    )
    
    # Helper function to plot with quantiles
    def qplot(xs, color, ax=None, normalize=False, lw=1, ls='-', type='fill_between', lcolor=None, **kwargs):
        if ax is None:
            ax = plt.subplot(111)

        bmidpts = sampling.midpts(histkwargs['bins'])

        if normalize:
            nrml = lambda ys: ys / np.nanmedian(xs, axis=0).mean()
        else:
            nrml = lambda ys: np.where(np.nanmedian(xs, axis=0) == 0, np.nan, ys)

        if type == 'fill_between':
            ax.plot(
                bmidpts, 
                nrml(np.nanmean(xs, axis=0) ), 
                color=lcolor is None and color or lcolor, 
                lw=lw, 
                ls=ls, 
                **kwargs
            )
            ax.fill_between(
                bmidpts,
                nrml(np.nanquantile(xs, 0.16, axis=0)),
                nrml(np.nanquantile(xs, 0.84, axis=0)),
                alpha=0.3,
                color=color,
            )
        elif type == 'errorbar':
            ek.errorbar(
                bmidpts,
                nrml(np.nanmean(xs, axis=0) ),
                ylow=nrml(np.nanquantile(xs, 0.16, axis=0) ),
                yhigh=nrml(np.nanquantile(xs, 0.84, axis=0)),
                markerfacecolor=color,
                markeredgecolor='w',
                ecolor=color,
                ax=ax,
                **kwargs
            )

    # Create figure with custom layout
    logger.info("Creating figure...")
    fig = plt.figure(figsize=(12,4))
    #gs = GridSpec(1,3, figure=fig, width_ratios=[1, 1., 0.65])
    gs = GridSpec(1, 2, figure=fig, left=0.05, right=0.74, width_ratios=[1, 1])
    gs_right = GridSpec(1, 1, figure=fig, left=0.75, right=0.95)  # Adjust left value to control spacing

    # Left panel: EoD fraction
    ax0 = fig.add_subplot(gs[0])
    qplot(pweighted_counts / fullgz_counts, color=colorlists.slides['bluebird'],lw=2,
          ax=ax0, label='Pr[EoD]-weighted (GZ sample)',)
    qplot(gzhc_counts / fullgz_counts, colorlists.slides['orange'],
          ax=ax0, label='GZ classification', type='errorbar')
    #qplot(pweighted_fullcounts / full_counts, colorlists.slides['red'],
    #      label='Pr[EoD]-weighted (all)', ls='-', ax=ax0, lw=2)

    ax0.legend(fontsize=12)
    ax0.set_ylim(0., 0.5)
    ax0.set_xlabel(r'$m_r$')
    ax0.set_ylabel('Edge-on disk fraction')
    #ax0.set_xlim(9.21, 10.4)

    # \\ MIDDLE PANEL : Spearman rank coefficients
    p_eod = data['catalog'].reindex(data['gzmatch'].index)['p_eod']

    pca_embeddings = pd.DataFrame(data['embeddings_pca'], index=data['catalog'].index)

    full = stats.spearmanr( *sampling.fmasker(data['gzmatch']['DEOyes'], p_eod))

    gzcolumns = data['gzmatch'].select_dtypes(include=np.number).columns
    espr = np.zeros([len(pca_embeddings.columns), len(gzcolumns)])
    for ix in range(len(espr)):
        for ic,col in enumerate(gzcolumns):
            x = stats.spearmanr( *sampling.fmasker(data['gzmatch'][col], pca_embeddings.reindex(data['gzmatch'].index)[ix]))
            espr[ix, ic] = abs(x.statistic)

    random_espr = np.zeros([len(pca_embeddings.columns), len(gzcolumns)])
    for ix in range(len(espr)):
        for ic,col in enumerate(gzcolumns):
            x = stats.spearmanr( *sampling.fmasker(data['gzmatch'][col], 
                                                np.random.choice(
                                                    pca_embeddings.reindex(data['gzmatch'].index)[ix],
                                                    pca_embeddings.reindex(data['gzmatch'].index)[ix].size,
                                                )
                                                )
                            )
            random_espr[ix, ic] = abs(x.statistic)

    maxgkde = stats.gaussian_kde(espr.max(axis=1))
    eodgkde = stats.gaussian_kde(espr[:,np.where(gzcolumns=='DEOyes')[0][0]])
    randomgkde = stats.gaussian_kde(random_espr.max(axis=1))

    ax = fig.add_subplot(gs[1])

    show_comparison = False
    if show_comparison:
        # \\ Wu & Walmsley 2025
        ax.axvspan(
            0.618 - 0.149,
            0.618 + 0.149,
            color=colorlists.slides['orange'],
            alpha=0.4
        )
        ax.axvline(
            0.618,
            color=colorlists.slides['orange'],
            lw=4,
            ls=':'
        )
        
        # \\ Wu & Walmsley 2025
        ax.axvspan(
            0.455 - 0.2,
            0.455 + 0.2,
            color=colorlists.slides['yellow'],
            alpha=0.4
        )
        ax.axvline(
            0.455,
            color=colorlists.slides['yellow'],
            lw=4,
            ls='--',
        )
        
    print(rf'{espr.max(axis=1)[:5].mean():.2f}\\pm{espr.max(axis=1)[:5].std():.2f}')
    ax.axvspan(
        espr.max(axis=1)[:5].mean() - espr.max(axis=1)[:5].std(),
        espr.max(axis=1)[:5].mean() + espr.max(axis=1)[:5].std(),
        color=colorlists.slides['bluebird'],
        alpha=0.4,
        hatch='//'
    )
    ax.axvline(
        espr.max(axis=1)[:5].mean(),
        color=colorlists.slides['bluebird'],
        lw=4
    )

    rs = np.linspace(0.,0.8,100)
    #ek.outlined_plot(rs, randomgkde(rs), ls=':', color=colorlists.slides['grey'], ax=ax)

    ek.outlined_plot(rs, eodgkde(rs), ls='--', color=colorlists.slides['blue'], ax=ax, label='against DEOyes')
    ek.outlined_plot(rs, maxgkde(rs), color=colorlists.slides['bluebird'], ax=ax, label='against all')

    ax.set_xlabel('Max Spearman Correlation: PCA-GZ')
    ax.set_ylabel('Density')
    ax.legend(fontsize=12)

    # Middle panel: logMstar distribution
    ## ax1 = fig.add_subplot(gs[1])
    ## bins = histkwargs['bins']
    ## ek.hist(
    ##     matched.loc[:, 'rmag'],
    ##     bins=bins,
    ##     density=True,
    ##     ax=ax1,
    ##     alpha=0.1,
    ##     lw=3,
    ##     color=colorlists.slides['red'],
    ##     label=r'$\log_{10}(\rm M_\bigstar/M_\odot)>9$'
    ## )
    ## ek.hist(
    ##     catalog.reindex(gzmatch.index).loc[:, 'rmag'],
    ##     bins=bins,
    ##     density=True,
    ##     ax=ax1,
    ##     alpha=0.1,
    ##     lw=3,
    ##     color=colorlists.slides['blue'],
    ##     label=r'Has GZ classification'
    ## )
    ## 
    ## ax1.set_xlabel(ek.common_labels['logmstar'])
    ## ax1.set_ylabel("PDF")
    ## #ax1.set_xlim(9.21, 10.4)
    ## ax1.legend()

    # Right panel: 3x3 grid of EoD candidate images
    logger.info("Loading example images...")
    candidates = np.arange(len(img_names))[has_classification][
        (prob_labels_iter[has_classification, 2] > np.nanquantile(prob_labels_iter[has_classification, 2], 0.9))
    ]

    n_examples = min(12, len(candidates))
    if n_examples > 0:
        np.random.seed(seed)
        example_indices = np.random.choice(candidates, n_examples, replace=False)

        # Create 3x3 grid within the third panel
        gs_inner = gs_right[0].subgridspec(4,3, hspace=0.02, wspace=0.02)

        for idx, gix in enumerate(example_indices):
            row = idx // 3
            col = idx - (idx//3)*3
            
            ax = fig.add_subplot(gs_inner[row, col])
            
            # Load this specific image on-demand
            img_name = img_names[gix]
            image = load_image_by_name(img_name, data_path)

            # i-band only
            ek.imshow(image[1], ax=ax, q=0.01, cmap='Greys')

            ax.set_xticks([])
            ax.set_yticks([])
    else:
        logger.warning("No EoD candidates found for image grid")

    plt.tight_layout()

    # Save figure
    if output_dir is not None:
        output_file = output_dir / 'fg_galaxyzoo.pdf'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {output_file}")
    else:
        plt.show()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate Galaxy Zoo EoD analysis figure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate figure using default config
  python fg_galaxyzoo.py

  # Use custom config
  python fg_galaxyzoo.py --config ../configs/galaxyzoo.yaml

  # Specify output directory
  python fg_galaxyzoo.py --output-dir ../output/figures/
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='../configs/galaxyzoo.yaml',
        help='Path to configuration YAML file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./punchline_figures/',
        help='Output directory for figures'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=19,
        help='Random seed for image selection'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging('INFO')

    try:
        # Set random seed
        np.random.seed(args.seed)

        # Load configuration
        config = load_config(args.config)
        logger.info(f"Configuration loaded from: {args.config}")

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # Load data
        data = load_data(config, logger)

        # Generate figure
        logger.info("=" * 60)
        logger.info("GENERATING FIGURE")
        logger.info("=" * 60)

        make_figure_galaxyzoo(data, output_dir, logger, seed=args.seed)

        logger.info("=" * 60)
        logger.info("SUCCESS")
        logger.info("=" * 60)
        print(f"\n✅ Figure generation completed successfully!")
        print(f"   Output directory: {output_dir}")

    except Exception as e:
        logger.error(f"Error during figure generation: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
