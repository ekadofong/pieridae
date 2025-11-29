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

# Import load_data from make_punchlines
from make_punchlines import load_data


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


def load_config(config_path: str, input_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['data']['input_path'] = Path(config['data']['input_path'])
    config['data']['output_path'] = Path(config['data']['output_path'])

    # Override output_path if provided via command line
    if input_path is not None:
        config['data']['output_path'] = Path(input_path)

    return config


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
            np.linspace(8., 11., 12),
            erronmetric=True
        )
        pmerger_baseline_by_mass = lambda logmstar: np.interp(
            logmstar,
            out_baseline[0].flatten(),
            out_baseline[1][:, 0, 2].flatten()
        )

        # Use mass bins matching make_punchlines
        logmstar_bins = list(np.arange(8.25, 10.5, 0.3)) + [11.]
        _, logmstar_bins = sampling.bin_by_count(sample_catalog.loc[sample_catalog['logmass_adjusted']>8.25, 'logmass_adjusted'], 500, 0.25 )
        groups = np.digitize(sample_catalog['logmass_adjusted'], logmstar_bins)
        groupids = np.arange(1, len(logmstar_bins))

        # Use colormap matching make_punchlines
        cmap = ec.colormap_from_list([colorlists.slides['orange'], plt.cm.coolwarm(0.5), colorlists.slides['bluebird']])

        # Set axis limits matching make_punchlines
        axarr[row_idx, 0].set_xlim(-0.75, 3)
        axarr[row_idx, 0].set_ylim(0., 0.45)

        for gidx, gid in enumerate(groupids):
            selected = sample_catalog.loc[groups == gid]

            for idx, is_normalized in enumerate([False, True]):
                ms_at_mass = sfs(selected['logmass_adjusted'])
                dsfs = np.log10(calibrations.LHa2SFR(selected['L_Ha'])) - ms_at_mass
                assns, loglhabins = sampling.bin_by_count(dsfs, 20, 0.25)
                xs = sampling.midpts(loglhabins) / sfs_std

                if is_normalized:
                    factor = 1. / pmerger_baseline_by_mass(selected['logmass_adjusted'])
                    nrml = 1.
                else:
                    factor = 1.
                    nrml = 1.

                _, ys, _ = sampling.running_metric(
                    dsfs,
                    pmerger.reindex(selected.index) * factor,
                    np.nanmean,
                    sampling.midpts(loglhabins),
                    erronmetric=True
                )

                if is_normalized:
                    nrml = np.interp(0., xs, ys[:,0,2])

                ek.outlined_plot(
                    xs,
                    ys[:, 0, 2]/nrml,
                    lw=2,
                    ax=axarr[row_idx, idx],
                    color=cmap(gidx / len(groupids))
                )
                axarr[row_idx, idx].fill_between(
                    xs,
                    ys[:, 0, 1]/nrml,
                    ys[:, 0, 3]/nrml,
                    label=f'[{logmstar_bins[gid-1]:.2f},{logmstar_bins[gid]:.2f}]',
                    alpha=0.3,
                    color=cmap(gidx / len(groupids))
                )

        # Add overall trend to normalized panel
        mask = (sample_catalog['logmass_adjusted'] > logmstar_bins[0]) & (sample_catalog['logmass_adjusted'] < logmstar_bins[-1])
        xs = (np.log10(calibrations.LHa2SFR(sample_catalog['L_Ha'])) - sfs(sample_catalog['logmass_adjusted'])) / sfs_std
        ys = pmerger / pmerger_baseline_by_mass(sample_catalog['logmass_adjusted'])
        assns, loglhabins = sampling.bin_by_count(xs[(xs>-0.5)&(xs<3.5)], 20, 0.25)
        out = sampling.running_metric(xs.loc[mask], ys.loc[mask], np.nanmean, sampling.midpts(loglhabins), dx=0.4, erronmetric=True)
        nrml = np.interp(0., out[0], out[1][:,0,2])
        axarr[row_idx, 1].fill_between(
            out[0],
            out[1][:, 0, 1]/nrml,
            out[1][:, 0, 3]/nrml,
            color='grey',
            alpha=0.4,
        )
        ek.outlined_plot(
            out[0],
            out[1][:, 0, 2]/nrml,
            ax=axarr[row_idx, 1],
            ls='--',
            lw=2
        )

        # Labels for this row - updated to match make_punchlines
        lkwargs = {'ls':':', 'color':'lightgrey', 'zorder':-1}
        axarr[row_idx, 1].axhline(1., **lkwargs)
        axarr[row_idx, 1].axvline(0., **lkwargs)
        for ax in axarr[row_idx]:
            ax.set_xlabel(r'$ \mathcal{S} = \frac{\log_{10}[{\rm SFR}/{\rm SFS(M_\bigstar)}]}{\sigma_{\rm SFS}}$', fontsize=20)
        axarr[row_idx, 0].set_ylabel(r'$\langle \rm Pr[interaction] \rangle$')
        axarr[row_idx, 1].set_ylabel(r'$\mathcal{R}_{\rm int}(\mathcal{S})$')
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
        pmerger_threshold = sample_catalog['p_undisturbed']

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
                color=colorlists.slides['bluebird'],
                label='Weighted by Pr[interaction]',
                ax=axarr[row_idx, idx],
                bins=bins
            )
            ek.hist(
                hamorph.reindex(sample_catalog.loc[(pmerger > pmerger_threshold)].index)[morph_key],
                density=True,
                alpha=0.4,
                lw=2,
                color=colorlists.slides['orange'],
                label='High-confidence mergers',
                ax=axarr[row_idx, idx],
                bins=bins
            )
            if idx == 0:
                ek.text(0.025, 0.975, 'Unweighted', color='grey', ax=axarr[row_idx, idx], fontsize=11)
                ek.text(0.025, 0.9, '''Weighted by
Pr[interaction]''', color=colorlists.slides['bluebird'], ax=axarr[row_idx, idx], fontsize=11)
                ek.text(0.025, 0.75, '''High-confidence
mergers''', color=colorlists.slides['orange'], ax=axarr[row_idx, idx], fontsize=11)
            axarr[row_idx, idx].set_xlabel(rf'{labels[idx]}({tags[prefix]})')
            if idx == 0:
                axarr[row_idx, idx].set_ylabel('PDF')

        # Add sample name as title
        #axarr[row_idx, 0].set_title(sample_name, fontsize=14, loc='left')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15)
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
        '--input-path',
        type=str,
        default=None,
        help='Path to input data directory (overrides config["data"]["output_path"])'
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
        config = load_config(args.config, input_path=args.input_path)
        logger.info(f"Configuration loaded from: {args.config}")
        if args.input_path:
            logger.info(f"Input path overridden to: {args.input_path}")

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # Load data
        data = load_data(config, logger, use_nn_classifier=True, force_pca_load=False)

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
