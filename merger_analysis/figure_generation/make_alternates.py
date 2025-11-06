#!/usr/bin/env python3
"""
Generate Alternative Sample Figures for Merger Analysis

This script remakes figures 6, 7, and 9 from the merger classification analysis
using two alternative samples:
- spec-z only sample: galaxies with spectroscopic redshifts 0.04 < z_spec < 0.12
- high-confidence mergers: galaxies where Pr[merger] + Pr[ambig] > Pr[undisturbed]

Each figure is generated as a two-row plot:
- Top row: spec-z sample
- Bottom row: high-confidence sample

Usage:
    # Generate all figures using default config
    python make_alternates.py

    # Specify custom config
    python make_alternates.py --config ../custom_config.yaml

    # Specify output directory
    python make_alternates.py --output-dir ../figures/

    # Generate only specific figures
    python make_alternates.py --figures 6,7,9
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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm import tqdm

# Add pieridae to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from pieridae.starbursts.byol import (
    EmbeddingAnalyzer,
    LabelPropagation,
)
from pieridae.starbursts import sample
from ekfplot import plot as ek, colors as ec, colorlists
from ekfphys import calibrations
from ekfstats import sampling


def setup_logging(level: str = 'INFO') -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger('make_alternates')
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
    config['data']['output_path'] = Path('../output/fiducial/')

    return config


def load_data(config: dict, logger: logging.Logger) -> Dict:
    """
    Load all data needed for figure generation.

    This function consolidates all data loading operations from the notebook
    into a single place, including:
    - Embeddings and PCA reduction
    - Image names and labels
    - Label propagation results
    - Full catalog with adjusted masses
    - H-alpha morphology statistics
    - Satellite classifications

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
        prob_threshold=config.get('labels', {}).get('prob_threshold', 0.7),
        frag_threshold=config.get('labels', {}).get('frag_threshold', 0.25),
    )

    iterative_labels, n_labels_iter, prob_labels_iter, stats = \
        propagator.iterative_propagation(embeddings_pca, labels)

    data['iterative_labels'] = iterative_labels
    data['n_labels_iter'] = n_labels_iter
    data['prob_labels_iter'] = prob_labels_iter

    logger.info(f"Human labels: {stats['n_human']}")
    logger.info(f"Auto-labels added: {stats['n_added_iteration']}")
    logger.info(f"Total labels: {stats['n_final_auto']}")

    # Identify merger candidates
    logger.info("Identifying merger candidates...")
    fragmented = prob_labels_iter[:, 4] > config.get('labels',{}).get('frag_threshold',0.25)
    pmerger = prob_labels_iter[:, 2] + prob_labels_iter[:, 3]
    possible_merger = pmerger > prob_labels_iter[:, 1]

    is_merger = possible_merger & ~fragmented
    is_undisturbed = ~possible_merger & ~fragmented

    data['fragmented'] = fragmented
    data['pmerger'] = pmerger
    data['possible_merger'] = possible_merger
    data['is_merger'] = is_merger
    data['is_undisturbed'] = is_undisturbed

    logger.info(f"Fragmented objects: {fragmented.sum()}")
    logger.info(f"Merger candidates: {is_merger.sum()}")

    # Load full catalog
    logger.info("Loading full catalog...")
    full_catalog, masks = sample.load_sample(
        '/Users/kadofong/work/projects/merian/local_data/base_catalogs/mdr1_n708maglt26_and_pzgteq0p1.parquet'
    )
    base_catalog = full_catalog.loc[masks['is_good'][0]]

    # Load adjusted masses
    logger.info("Loading adjusted masses from individual files...")
    for sid in tqdm(base_catalog.index, desc="Loading masses"):
        filename = f'{data_path}/{sid}/{sid}_i_results.pkl'
        if not os.path.exists(filename):
            continue
        with open(filename, 'rb') as f:
            x = pickle.load(f)
        base_catalog.loc[sid, 'logmass_adjusted'] = x['logmass_adjusted']

    base_catalog.loc[base_catalog['logmass_adjusted'].isna(), 'logmass_adjusted'] = \
        base_catalog.loc[base_catalog['logmass_adjusted'].isna(), 'logmass']

    # Create catalog subset
    fragmented_highthresh = prob_labels_iter[:, 4] > 0.3
    catalog = base_catalog.reindex(img_names[~fragmented_highthresh])
    catalog['p_merger'] = np.where(
        (prob_labels_iter[~fragmented_highthresh] == 0).all(axis=1),
        np.nan,
        prob_labels_iter[~fragmented_highthresh, 3]
    )
    catalog['p_ambig'] = np.where(
        (prob_labels_iter[~fragmented_highthresh] == 0).all(axis=1),
        np.nan,
        prob_labels_iter[~fragmented_highthresh, 2]
    )
    catalog['p_undisturbed'] = np.where(
        (prob_labels_iter[~fragmented_highthresh] == 0).all(axis=1),
        np.nan,
        prob_labels_iter[~fragmented_highthresh, 1]
    )

    dm = catalog['logmass_adjusted'] - catalog['logmass']
    catalog = catalog.loc[
        (dm < 0.5) &
        (catalog['logmass_adjusted'] <= 10.5) &
        (catalog['logmass_adjusted'] >= 7.5)
    ]

    data['catalog'] = catalog
    logger.info(f"Catalog size: {len(catalog)} objects")

    # Load H-alpha morphology
    logger.info("Loading H-alpha morphology statistics...")
    hamorph_file = '../../local_data/abby_morphology_statistics.csv'
    if os.path.exists(hamorph_file):
        hamorph = pd.read_csv(hamorph_file, index_col=0)
        hamorph['halpha_m20'] = np.where(hamorph['halpha_m20'] < -3, np.nan, hamorph['halpha_m20'])
        hamorph['continuum_m20'] = np.where(hamorph['continuum_m20'] < -3, np.nan, hamorph['continuum_m20'])
        data['hamorph'] = hamorph
        logger.info(f"Loaded morphology for {len(hamorph)} objects")
    else:
        logger.warning(f"H-alpha morphology file not found: {hamorph_file}")
        data['hamorph'] = None

    # Load satellite catalog
    logger.info("Loading satellite catalog...")
    cached_file = '../local_data/yue_meriansatellites_catalog/cached_catalog.csv'
    lucas_file = '../local_data/yue_meriansatellites_catalog/Merian_dwarf_hosts.csv'

    if os.path.exists(cached_file) and os.path.exists(lucas_file):
        yue = pd.read_csv(cached_file, index_col=0)
        lucas = pd.read_csv(lucas_file)
        lucas_trimmed = lucas[['objectId_Merian', 'coord_ra_Merian', 'coord_dec_Merian']]
        lucas_trimmed.index = [f'M{oid}' for oid in lucas_trimmed['objectId_Merian']]

        final_satellites = pd.concat([yue, lucas_trimmed])
        final_satellites = final_satellites.loc[~final_satellites.index.duplicated()]

        is_satellite = np.in1d(catalog.index, final_satellites.index)
        data['is_satellite'] = is_satellite
        logger.info(f"Identified {is_satellite.sum()} satellites")
    else:
        logger.warning("Satellite catalog not found")
        data['is_satellite'] = None

    logger.info("Data loading complete!")
    logger.info("=" * 60)

    return data


def make_figure_merger_prob_vs_dsfs_alternates(
    data: Dict,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Figure 6 (alternates): Merger probability vs distance from star-forming sequence.

    Creates a 2-row figure:
    - Top row: spec-z only sample
    - Bottom row: high-confidence mergers only sample

    Parameters
    ----------
    data : dict
        Data dictionary from load_data()
    output_dir : Path
        Output directory for figures
    logger : logging.Logger
        Logger instance
    """
    logger.info("Generating Figure 6 (alternates): Merger probability vs dSFS")

    catalog = data['catalog']

    # Define alternative samples
    specz_sample = catalog.loc[(catalog['z_spec'] > 0.04) & (catalog['z_spec'] < 0.12)]
    hiconf_sample = catalog.loc[(catalog['p_merger'] + catalog['p_ambig']) > catalog['p_undisturbed']]

    logger.info(f"Spec-z sample size: {len(specz_sample)}")
    logger.info(f"High-confidence sample size: {len(hiconf_sample)}")

    # Compute SFS relation
    alpha = -0.13 * 0.08 + 0.8
    norm = 1.24 * 0.08 - 1.47
    sfs_std = 0.22 * 0.08 + 0.38
    sfs = lambda logmstar: alpha * (logmstar - 8.5) + norm

    # Create figure with 2 rows
    fig, axarr = plt.subplots(1, 2, figsize=(12, 5))
    axarr = axarr.reshape(1,-1)
    
    samples = [specz_sample]
    sample_names = ['Spec-z only (0.04 < z < 0.12)', 'High-confidence mergers']

    for row_idx, (sample_catalog, sample_name) in enumerate(zip(samples, sample_names)):
        logger.info(f"Processing {sample_name}...")

        pmerger = sample_catalog['p_merger'] + sample_catalog['p_ambig']

        # Compute baseline merger probability as function of mass
        dsfs = np.log10(calibrations.LHa2SFR(sample_catalog['L_Ha'])) - sfs(sample_catalog['logmass_adjusted'])
        mask = abs(dsfs / sfs_std) < 0.2

        out_baseline = sampling.running_metric(
            sample_catalog.loc[mask, 'logmass_adjusted'],
            pmerger.loc[mask],
            np.nanmean,
            np.linspace(7., 10.25, 12),
            erronmetric=True
        )
        pmerger_baseline_by_mass = lambda logmstar: np.interp(
            logmstar,
            out_baseline[0].flatten(),
            out_baseline[1][:, 0, 2].flatten()
        )

        logmstar_bins = [6] + list(np.arange(8.5, 10., 0.3)) + [12]
        groups = np.digitize(sample_catalog['logmass_adjusted'], logmstar_bins)
        groupids = np.unique(groups)

        for gidx, gid in enumerate(groupids):
            for idx, is_normalized in enumerate([False, True]):
                selected = sample_catalog.loc[groups == gid]
                ms_at_mass = sfs(selected['logmass_adjusted'])
                dsfs = np.log10(calibrations.LHa2SFR(selected['L_Ha'])) - ms_at_mass
                assns, loglhabins = sampling.bin_by_count(dsfs, 10, 0.25)
                xs = sampling.midpts(loglhabins) / sfs_std

                if is_normalized:
                    factor = 1. / pmerger_baseline_by_mass(selected['logmass_adjusted'])
                else:
                    factor = 1.

                _, ys, _ = sampling.running_metric(
                    dsfs,
                    pmerger.reindex(selected.index) * factor,
                    np.nanmean,
                    sampling.midpts(loglhabins),
                    erronmetric=True
                )

                ek.outlined_plot(
                    xs,
                    ys[:, 0, 2],
                    lw=2,
                    ax=axarr[row_idx, idx],
                    color=plt.cm.coolwarm(gidx / len(groupids))
                )
                axarr[row_idx, idx].fill_between(
                    xs,
                    ys[:, 0, 1],
                    ys[:, 0, 3],
                    label=f'[{logmstar_bins[gid-1]:.2f},{logmstar_bins[gid]:.2f}]',
                    alpha=0.3,
                    color=plt.cm.coolwarm(gidx / len(groupids))
                )

        # Add overall trend to normalized panel
        xs = (np.log10(calibrations.LHa2SFR(sample_catalog['L_Ha'])) - sfs(sample_catalog['logmass_adjusted'])) / sfs_std
        ys = pmerger / pmerger_baseline_by_mass(sample_catalog['logmass_adjusted'])
        out = sampling.running_metric(xs, ys, np.nanmean, np.arange(-0.5, 3.5, 0.2), dx=0.4, erronmetric=True)
        axarr[row_idx, 1].fill_between(
            out[0],
            out[1][:, 0, 1],
            out[1][:, 0, 3],
            color='grey',
            alpha=0.4,
        )
        ek.outlined_plot(
            out[0],
            out[1][:, 0, 2],
            ax=axarr[row_idx, 1],
            ls='--',
            lw=2
        )

        # Labels for this row
        for ax in axarr[row_idx]:
            ax.set_xlabel(r'$ \frac{\log_{10}[{\rm SFR}/{\rm SFS(M_\bigstar)}]}{\sigma_{\rm SFS}}$', fontsize=20)
        axarr[row_idx, 0].set_ylabel(r'$\langle \rm Pr[merger] \rangle$')
        axarr[row_idx, 1].set_ylabel(r'$\langle \rm Pr[merger]/Pr[merger|SFS] \rangle$')
        axarr[row_idx, 1].set_ylim(0.5, axarr[row_idx, 1].get_ylim()[-1])
        axarr[row_idx, 1].set_yscale('log')

        # Add sample name as title
        #axarr[row_idx, 0].set_title(sample_name, fontsize=14)

    plt.tight_layout()
    output_file = output_dir / 'fig6_alternates.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved: {output_file}")


def make_figure_hamorph_distributions_alternates(
    data: Dict,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Figure 7 (alternates): H-alpha morphology distributions.

    Creates a 2-row figure:
    - Top row: spec-z only sample
    - Bottom row: high-confidence mergers only sample

    Parameters
    ----------
    data : dict
        Data dictionary from load_data()
    output_dir : Path
        Output directory for figures
    logger : logging.Logger
        Logger instance
    """
    logger.info("Generating Figure 7 (alternates): H-alpha morphology distributions")

    catalog = data['catalog']
    hamorph = data['hamorph']

    if hamorph is None:
        logger.warning("H-alpha morphology data not available, skipping")
        return

    # Define alternative samples
    specz_sample = catalog.loc[(catalog['z_spec'] > 0.04) & (catalog['z_spec'] < 0.12)]
    hiconf_sample = catalog.loc[(catalog['p_merger'] + catalog['p_ambig']) > catalog['p_undisturbed']]

    logger.info(f"Spec-z sample size: {len(specz_sample)}")
    logger.info(f"High-confidence sample size: {len(hiconf_sample)}")

    # Create figure with 2 rows
    fig, axarr = plt.subplots(1, 3, figsize=(15, 4))
    axarr = axarr.reshape(1,-1)

    tags = {'continuum': 'continuum', 'halpha': r'H$\alpha$'}
    labels = ['Asymmetry', r'G', r'$M_{20}$']
    keys = ['asymmetry', 'gini', 'm20']
    prefix = 'halpha'

    samples = [specz_sample, ]
    sample_names = ['Spec-z only (0.04 < z < 0.12)', 'High-confidence mergers']

    for row_idx, (sample_catalog, sample_name) in enumerate(zip(samples, sample_names)):
        logger.info(f"Processing {sample_name}...")

        pmerger = sample_catalog['p_merger'] + sample_catalog['p_ambig']

        for idx, key in enumerate(keys):
            morph_key = f'{prefix}_{key}'
            out = ek.hist(
                hamorph.reindex(sample_catalog.index)[morph_key],
                density=True,
                alpha=0.2,
                lw=2,
                color=ec.ColorBase(colorlists.slides['grey']).base,
                hatch='//',
                label='Unweighted',
                ax=axarr[row_idx, idx],
                binalpha=0.005
            )
            bins = out[1][1]
            ek.hist(
                hamorph.reindex(sample_catalog.index)[morph_key],
                weights=pmerger,
                density=True,
                alpha=0.4,
                lw=2.,
                color=colorlists.slides['red'],
                label='Weighted by Pr[interaction]',
                ax=axarr[row_idx, idx],
                bins=bins
            )
            ek.hist(
                hamorph.reindex(sample_catalog.loc[(pmerger > sample_catalog['p_undisturbed'])].index)[morph_key],
                density=True,
                alpha=0.4,
                lw=2,
                color=colorlists.slides['blue'],
                label='High-confidence mergers',
                ax=axarr[row_idx, idx],
                bins=bins
            )
            if idx == 0:
                ek.text(0.025, 0.975, 'Unweighted', color='grey', ax=axarr[row_idx, idx], fontsize=11)
                ek.text(0.025, 0.9, 'Pr[merger]-weighted', color=colorlists.slides['red'], ax=axarr[row_idx, idx], fontsize=11)
                ek.text(0.025, 0.825, '''High-confidence
mergers''', color=colorlists.slides['blue'], ax=axarr[row_idx, idx], fontsize=11)
            axarr[row_idx, idx].set_xlabel(rf'{labels[idx]}({tags[prefix]})')
            axarr[row_idx, idx].set_ylabel('PDF')

        # Add sample name as title
        #axarr[row_idx, 0].set_title(sample_name, fontsize=14, loc='left')

    plt.tight_layout()
    output_file = output_dir / 'fig7_alternates.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved: {output_file}")


def make_figure_merger_prob_vs_environment_alternates(
    data: Dict,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Figure 9 (alternates): Merger probability vs environment.

    Creates a 2-row figure:
    - Top row: spec-z only sample
    - Bottom row: high-confidence mergers only sample

    Parameters
    ----------
    data : dict
        Data dictionary from load_data()
    output_dir : Path
        Output directory for figures
    logger : logging.Logger
        Logger instance
    """
    logger.info("Generating Figure 9 (alternates): Merger probability vs environment")

    catalog = data['catalog']
    is_satellite = data['is_satellite']

    if is_satellite is None:
        logger.warning("Satellite data not available, skipping")
        return

    # Define alternative samples
    specz_mask = catalog['logmass_adjusted'] > 8.5
    #(catalog['z_spec'] > 0.04) & (catalog['z_spec'] < 0.12)
    specz_sample = catalog.loc[specz_mask]
    is_satellite = is_satellite[specz_mask]
    
    logger.info(f"Spec-z sample size: {len(specz_sample)}")

    # Compute SFS relation
    alpha = -0.13 * 0.08 + 0.8
    norm = 1.24 * 0.08 - 1.47
    sfs_std = 0.22 * 0.08 + 0.38
    sfs = lambda logmstar: alpha * (logmstar - 8.5) + norm

    pmerger = specz_sample['p_merger'] + specz_sample['p_ambig']

    # Compute baseline merger probability
    dsfs = np.log10(calibrations.LHa2SFR(specz_sample['L_Ha'])) - sfs(specz_sample['logmass_adjusted'])
    mask = abs(dsfs / sfs_std) < 0.2

    out_baseline = sampling.running_metric(
        specz_sample.loc[mask, 'logmass_adjusted'],
        pmerger.loc[mask],
        np.nanmean,
        np.linspace(7., 10.25, 12),
        erronmetric=True
    )
    pmerger_baseline_by_mass = lambda logmstar: np.interp(
        logmstar,
        out_baseline[0].flatten(),
        out_baseline[1][:, 0, 2].flatten()
    )

    fig, axarr = plt.subplots(1,2, figsize=(10,5))

    xs = (np.log10(calibrations.LHa2SFR(specz_sample['L_Ha'])) - sfs(specz_sample['logmass_adjusted'])) / sfs_std
    ys = pmerger / pmerger_baseline_by_mass(specz_sample['logmass_adjusted'])

    ax = axarr[1]
    for envkey in [0, 1]:
        if envkey == 0:
            envmask = ~is_satellite
            env_indices = sampling.make_matched_sample(     
                specz_sample.loc[~is_satellite, 'logmass_adjusted'],
                specz_sample.loc[is_satellite, 'logmass_adjusted']
            ).index            
        elif envkey == 1:
            envmask = is_satellite
            env_indices = specz_sample.loc[envmask].index

        out = sampling.running_metric(
            xs[envmask],
            ys[envmask],
            np.nanmean,
            np.arange(-0.5, 3.75, 0.4),
            dx=0.5,
            erronmetric=True
        )
        cc = [colorlists.slides['blue'], colorlists.slides['red']][envkey]
        ax.fill_between(
            out[0],
            out[1][:, 0, 1],
            out[1][:, 0, 3],
            color=cc,
            alpha=0.4,
            hatch=envkey == 1 and '||' or None,
        )
        ek.outlined_plot(
            out[0],
            out[1][:, 0, 2],
            ax=ax,
            color=cc,
            lw=2,
            label=['Field?', 'Satellite'][envkey]
        )

    ek.text(
        0.975,
        0.975,
        r'$0.04<z_{\rm spec}<0.12$',
        ax=axarr[0],
        bordercolor='w',
        borderwidth=1
    )
    ax.set_xlabel(r'$\mathcal{S}\equiv \frac{\log_{10}[{\rm SFR}/{\rm SFS(M_\bigstar)}]}{\sigma_{\rm SFS}}$', fontsize=20)
    ax.set_ylabel(r'Excess mean interaction probability')
    ax.axhline(1., color='lightgrey', ls=':')
    ax.legend()

    # Top panel: SFR offset distributions
    xbins = np.linspace(min(out[0]), max(out[0]), 30)
    cc = [colorlists.slides['blue'], colorlists.slides['red']]
    lbls = ['Field?', 'Satellite']
    for envidx, mask in enumerate([~is_satellite, is_satellite]):
        hcounts = sampling.bootstrap_histcounts(
            xs[mask],
            bins=xbins
        )
        cumulative_hist = np.cumsum(hcounts, axis=1)/np.sum(hcounts,axis=1).reshape(-1,1)
        if envidx == 0:
            nrml = np.quantile(cumulative_hist, 0.5, axis=0)
        
        axarr[0].fill_between(
            sampling.midpts(xbins),
            np.quantile(cumulative_hist, 0.16, axis=0)/nrml,
            np.quantile(cumulative_hist, 0.84, axis=0)/nrml,
            alpha=0.3,
            color = cc[envidx],
            step='mid'
        )
        axarr[0].step(
            sampling.midpts(xbins),
            np.quantile(cumulative_hist, 0.5, axis=0)/nrml,
            lw=2, 
            color=cc[envidx],
            where='mid',
            label=lbls[envidx]
        )
    axarr[0].set_ylabel(r'$N(<\mathcal{S})/N_{\rm tot}$')
    axarr[0].set_xlabel(r'$\mathcal{S}\equiv \frac{\log_{10}[{\rm SFR}/{\rm SFS(M_\bigstar)}]}{\sigma_{\rm SFS}}$', fontsize=20)
    axarr[0].legend()

    for ax in axarr:
        ax.grid(axis='x', color='lightgrey')
    plt.tight_layout()
    if output_dir is not None:
        output_file = output_dir / 'fig9_merger_prob_vs_environment.pdf'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved: {output_file}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate alternative sample figures from merger analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all figures using default config
  python make_alternates.py

  # Use custom config
  python make_alternates.py --config ../custom_config.yaml

  # Specify output directory
  python make_alternates.py --output-dir ../figures/

  # Generate only specific figures (comma-separated)
  python make_alternates.py --figures 6,7,9
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='../config.yaml',
        help='Path to configuration YAML file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./punchline_figures/',
        help='Output directory for figures'
    )

    parser.add_argument(
        '--figures',
        type=str,
        help='Comma-separated list of figure numbers to generate (6,7,9). If not specified, generates all.'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
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

        # Determine which figures to generate
        if args.figures:
            figure_nums = [int(x.strip()) for x in args.figures.split(',')]
        else:
            figure_nums = [6, 7, 9]

        logger.info("=" * 60)
        logger.info(f"GENERATING ALTERNATIVE SAMPLE FIGURES: {figure_nums}")
        logger.info("=" * 60)

        # Generate figures
        figure_functions = {
            6: make_figure_merger_prob_vs_dsfs_alternates,
            7: make_figure_hamorph_distributions_alternates,
            9: make_figure_merger_prob_vs_environment_alternates,
        }

        for fig_num in figure_nums:
            if fig_num in figure_functions:
                figure_functions[fig_num](data, output_dir, logger)
            else:
                logger.warning(f"Unknown figure number: {fig_num}. Valid options: 6, 7, 9")

        logger.info("=" * 60)
        logger.info("SUCCESS")
        logger.info("=" * 60)
        print(f"\n✅ Alternative sample figure generation completed successfully!")
        print(f"   Output directory: {output_dir}")

    except Exception as e:
        logger.error(f"Error during figure generation: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
