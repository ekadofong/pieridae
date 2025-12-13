#!/usr/bin/env python3
"""
Generate Alternative Sample Figures for Merger Analysis

This script remakes figures 4, 6, 7, and 9 from the merger classification analysis
using a spectroscopic redshift sample:
- spec-z only sample: galaxies with spectroscopic redshifts 0.04 < z_spec < 0.12

The analysis uses the same improved methodology as make_punchlines.py:
- Floor correction for low-mass galaxy bias
- Uncertainty propagation from multi-run analysis (optional)
- DataFrame-based merger probability extraction

Available figures:
- Figure 4: H-alpha luminosity vs stellar mass with merger probability overlay
- Figure 6: Merger probability vs dSFS (with mass labels and visual guides)
- Figure 7: H-alpha morphology distributions
- Figure 9: Merger probability vs environment

Usage:
    # Generate all figures using default config (single run)
    python make_alternates.py

    # Use multi-run analysis for uncertainty estimates
    python make_alternates.py --use-multirun

    # Specify custom config
    python make_alternates.py --config ../custom_config.yaml

    # Specify output directory
    python make_alternates.py --output-dir ../figures/

    # Generate only specific figures
    python make_alternates.py --figures 4,6,7,9

    # Customize runs for multi-run analysis
    python make_alternates.py --use-multirun --run-names fiducial,run1,run2
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

# Import load_data and load_data_multirun from make_punchlines
from make_punchlines import load_data, load_data_multirun


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


def make_figure_ha_sfs_merger_fraction_alternates(
    data: Dict,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Figure 4 (alternates): H-alpha luminosity vs stellar mass with merger probability overlay.

    Shows the star-forming sequence with color-coded average merger probability
    for the spec-z sample.

    Parameters
    ----------
    data : dict
        Data dictionary from load_data()
    output_dir : Path
        Output directory for figures
    logger : logging.Logger
        Logger instance
    """
    logger.info("Generating Figure 4 (alternates): H-alpha vs M* with merger fraction")

    catalog = data['catalog']

    # Define alternative sample: spec-z only
    specz_sample = catalog.loc[(catalog['z_spec'] > 0.04) & (catalog['z_spec'] < 0.12)]

    logger.info(f"Spec-z sample size: {len(specz_sample)}")

    # Compute SFS relation
    alpha = -0.13 * 0.08 + 0.8
    norm = 1.24 * 0.08 - 1.47
    sfs_lambda = lambda logmstar: alpha * (logmstar - 8.5) + norm

    fig, axarr = plt.subplots(1, 2, figsize=(10, 4), width_ratios=(1.2,1))

    bins = [np.logspace(7.75, 10.5, 20), np.logspace(39, 41.9, 25)]

    cmap = ec.colormap_from_list([
        ec.ColorBase(colorlists.slides['orange']).modulate(-0.3,0.3).base,
        ec.ColorBase(colorlists.slides['orange']).modulate(0.,-0.3).base,
        plt.cm.coolwarm(0.5),
        ec.ColorBase(colorlists.slides['bluebird']).modulate(0.3,-0.3).base,
        ec.ColorBase(colorlists.slides['bluebird']).modulate(-0.1,0.5).base,
    ])

    # Left panel: average merger probability
    im, _ = ek.pcolor_avg2d(
        10.**specz_sample['logmass_adjusted'],
        specz_sample['L_Ha'],
        specz_sample['p_merger'] + specz_sample['p_ambig'],
        cmap=cmap,
        yscale='log',
        xscale='log',
        zscale='log',
        bins=bins,
        ax=axarr[0],
    )

    # Right panel: number count histogram with scatter
    imx = ek.hist2d(
        10.**specz_sample['logmass_adjusted'],
        specz_sample['L_Ha'],
        cmap=ec.ColorBase('k').sequential_cmap(fade=1.),
        yscale='log',
        xscale='log',
        bins=bins,
        ax=axarr[1],
        zorder=0
    )

    thresh = np.nanquantile((specz_sample['p_ambig'] + specz_sample['p_merger']), 0.9)
    logger.info(f'Pr[TF]_90 is {thresh:.3f}')
    probable_merger = (specz_sample['p_ambig'] + specz_sample['p_merger']) > thresh
    ek.density_contour_scatter(
        10.**specz_sample.loc[probable_merger, 'logmass_adjusted'],
        specz_sample.loc[probable_merger, 'L_Ha'],
        ax=axarr[1],
        cmap=cmap,
        quantiles=np.linspace(0., 0.8, 10),
        yscale='log',
        xscale='log',
        scatter_s=4,
    )
    ek.text(
        0.975,
        0.025,
        'All galaxies',
        ax=axarr[1],
        color='grey'
    )
    ek.text(
        0.975,
        0.125,
        r'Pr[TF] > Pr[TF]$_{90}$',
        ax=axarr[1],
        color=cmap(0.1)
    )

    # Plot star-forming sequence
    ms = im._coordinates.data[0, :, 0]
    sfr_sfs = 10.**sfs_lambda(np.log10(ms))
    ha_sfs = calibrations.SFR2LHa(sfr_sfs)

    plt.colorbar(im, ax=axarr[0], label=r'$\langle {\rm Pr[TF]}\rangle$')

    for ax in axarr:
        ek.outlined_plot(
            ms,
            ha_sfs,
            ax=ax,
            lw=1,
            ls='--',
        )

        ek.text(
            7e7,
            3e39,
            'EKF+24b SFS',
            ax=ax,
            rotation=37,
            coord_type='absolute',
            va='bottom',
            ha='left',
            bordercolor='w',
            borderwidth=2,
            fontsize=8,
        )
        ax.set_xlabel(ek.common_labels['mstar'])
        ax.set_ylabel(ek.common_labels['halum'])
        ek.loglog(ax=ax)

    plt.tight_layout()
    output_file = output_dir / 'fig4_alternates.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved: {output_file}")


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

    # Define alternative sample: spec-z only
    specz_sample = catalog.loc[(catalog['z_spec'] > 0.04) & (catalog['z_spec'] < 0.12)]

    logger.info(f"Spec-z sample size: {len(specz_sample)}")

    # Compute SFS relation
    alpha = -0.13 * 0.08 + 0.8
    norm = 1.24 * 0.08 - 1.47
    sfs_std = 0.22 * 0.08 + 0.38
    sfs = lambda logmstar: alpha * (logmstar - 8.5) + norm

    # Extract merger probabilities from prob_labels_iter (matching make_punchlines.py)
    pmerger = pd.Series(data['prob_labels_iter'][:,2] + data['prob_labels_iter'][:,3], index=data['img_names'])
    pmerger = pmerger.reindex(specz_sample.index)

    # Compute floor from low-mass galaxies to correct for bias
    mask_floor = data['catalog'].reindex(data['img_names'])['logmass_adjusted'] < 8.
    floor = np.mean(data['mean_prob_labels'][mask_floor, 2])

    # Get uncertainties from multi-run analysis
    u_pmerger = pd.Series((data['std_prob_labels'][:,2]**2 + data['std_prob_labels'][:,3]**2)**0.5, index=data['img_names'])
    u_pmerger = u_pmerger.reindex(specz_sample.index)

    # Create figure
    fig, axarr = plt.subplots(1, 2, figsize=(12, 5))

    logger.info(f"Processing spec-z sample...")

    # Compute baseline merger probability as function of mass
    dsfs = np.log10(calibrations.LHa2SFR(specz_sample['L_Ha'])) - sfs(specz_sample['logmass_adjusted'])
    mask = abs(dsfs / sfs_std) < 0.2

    out_baseline = sampling.running_metric(
        specz_sample.loc[mask, 'logmass_adjusted'],
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
    _, logmstar_bins = sampling.bin_by_count(specz_sample.loc[specz_sample['logmass_adjusted']>8.25, 'logmass_adjusted'], 500, 0.25 )
    groups = np.digitize(specz_sample['logmass_adjusted'], logmstar_bins)
    groupids = np.arange(1, len(logmstar_bins))

    # Use colormap matching make_punchlines
    cmap = ec.colormap_from_list([colorlists.slides['orange'], plt.cm.coolwarm(0.5), colorlists.slides['bluebird']])

    # Set axis limits matching make_punchlines
    axarr[0].set_xlim(-0.75, 3)
    axarr[0].set_ylim(0., 0.45)

    # Initialize dictionary to store curve data
    curve_data = {'mass_bins': {}}

    for gidx, gid in enumerate(groupids):
        selected = specz_sample.loc[groups == gid]

        for idx, is_normalized in enumerate([False, True]):
            ms_at_mass = sfs(selected['logmass_adjusted'])
            dsfs = np.log10(calibrations.LHa2SFR(selected['L_Ha'])) - ms_at_mass
            assns, loglhabins = sampling.bin_by_count(dsfs, 20, 0.25)
            xs = sampling.midpts(loglhabins) / sfs_std

            if is_normalized:
                factor = 1. / (pmerger_baseline_by_mass(selected['logmass_adjusted']) - floor)
            else:
                factor = 1.
                nrml = 1.

            pm = pmerger.reindex(selected.index) - floor
            u_pm = u_pmerger.reindex(selected.index)

            _, ys, _ = sampling.running_metric(
                dsfs,
                pm * factor,
                np.nanmean,
                sampling.midpts(loglhabins),
                yerr=u_pm * factor,
                erronmetric=True
            )

            if is_normalized:
                nrml = np.interp(0., xs, ys[:,0,2])

            ek.outlined_plot(
                xs,
                ys[:, 0, 2]/nrml,
                lw=2,
                ax=axarr[idx],
                color=cmap(gidx / len(groupids))
            )
            axarr[idx].fill_between(
                xs,
                ys[:, 0, 1]/nrml,
                ys[:, 0, 3]/nrml,
                label=f'[{logmstar_bins[gid-1]:.2f},{logmstar_bins[gid]:.2f}]',
                alpha=0.3,
                color=cmap(gidx / len(groupids))
            )

            # Store curve data
            mass_key = (logmstar_bins[gid-1], logmstar_bins[gid])
            if mass_key not in curve_data['mass_bins']:
                curve_data['mass_bins'][mass_key] = {}

            data_key = 'normalized' if is_normalized else 'unnormalized'
            curve_data['mass_bins'][mass_key][data_key] = [
                xs,
                ys[:, 0, 2]/nrml,  # mean
                ys[:, 0, 1]/nrml,  # lower bound
                ys[:, 0, 3]/nrml   # upper bound
            ]

            # Add mass labels in left panel (matching make_punchlines.py)
            if not is_normalized:
                offset= 0
                slope = (ys[3+offset,0,2]-ys[2+offset,0,2])/(xs[3+offset]-xs[2+offset])

                ek.text(
                    sampling.midpts(xs[2+offset:4+offset]),
                    sampling.midpts(ys[2+offset:4+offset,0,2]),
                    rf'[{logmstar_bins[gid-1]:.1f},{logmstar_bins[gid]:.1f}]',
                    ha='center',
                    va='center',
                    rotation=np.rad2deg(np.arctan(slope * ek.get_subplot_aspectratio(axarr[idx]))),
                    coord_type='absolute',
                    ax=axarr[idx],
                    color=cmap(gidx / len(groupids)),
                    bordercolor='w',
                    borderwidth=3,
                    fontsize=12
                )

                # Add mass axis label on last curve
                if gidx == (len(groupids)-1):
                    offset = -1
                    slope = (ys[3+offset,0,2]-ys[2+offset,0,2])/(xs[3+offset]-xs[2+offset])
                    ek.text(
                        sampling.midpts(xs[2+offset:4+offset]),
                        sampling.midpts(ys[2+offset:4+offset,0,2]),
                        r'$\log_{10}(M_\bigstar/M_\odot)$',
                        ha='center',
                        va='bottom',
                        rotation=np.rad2deg(np.arctan(slope * ek.get_subplot_aspectratio(axarr[idx]))),
                        coord_type='absolute',
                        ax=axarr[idx],
                        color=ec.ColorBase(cmap(0.5)).modulate(-0.2).base,
                        bordercolor='w',
                        borderwidth=3,
                        fontsize=13
                    )

    # Add overall trend to normalized panel
    mask = (specz_sample['logmass_adjusted'] > logmstar_bins[0]) & (specz_sample['logmass_adjusted'] < logmstar_bins[-1])
    xs = (np.log10(calibrations.LHa2SFR(specz_sample['L_Ha'])) - sfs(specz_sample['logmass_adjusted'])) / sfs_std
    ys = (pmerger - floor) / (pmerger_baseline_by_mass(specz_sample['logmass_adjusted']) - floor)
    assns, loglhabins = sampling.bin_by_count(xs[(xs>-0.5)&(xs<3.5)], 20, 0.25)
    out = sampling.running_metric(xs.loc[mask], ys.loc[mask], np.nanmean, sampling.midpts(loglhabins), dx=0.4, erronmetric=True)
    nrml = np.interp(0., out[0], out[1][:,0,2])
    axarr[1].fill_between(
        out[0],
        out[1][:, 0, 1]/nrml,
        out[1][:, 0, 3]/nrml,
        color='grey',
        alpha=0.4,
    )
    ek.outlined_plot(
        out[0],
        out[1][:, 0, 2]/nrml,
        ax=axarr[1],
        ls='--',
        lw=2
    )

    # Store overall trend
    curve_data['overall_trend'] = [out[0], out[1][:, 0, 2]/nrml, out[1][:, 0, 1]/nrml, out[1][:, 0, 3]/nrml]

    # Add visual guides to right panel (matching make_punchlines.py)
    textcolor = ec.ColorBase(cmap(0.5)).modulate(-0.2).base
    ek.arrow(
        -0.4,
        1.1,
        0.,
        10.,
        ax=axarr[1],
        color=textcolor,
    )
    ek.text(
        -0.4,
        1.3,
        'more interactions',
        color=textcolor,
        va='bottom',
        ha='right',
        coord_type='absolute',
        rotation=90.,
        ax=axarr[1],
        fontsize=15,
    )

    ek.arrow(
        0.3,
        0.65,
        2.5,
        0.,
        ax=axarr[1],
        color=textcolor,
    )
    ek.text(
        1.5,
        0.7,
        'more vigorous star formation',
        color=textcolor,
        va='bottom',
        ha='center',
        coord_type='absolute',
        ax=axarr[1],
        fontsize=15
    )

    # Labels - updated to match make_punchlines
    lkwargs = {'ls':':', 'color':'lightgrey', 'zorder':-1}
    axarr[1].axhline(1., **lkwargs)
    axarr[1].axvline(0., **lkwargs)
    for ax in axarr:
        ax.set_xlabel(r'$ \mathcal{S} = \frac{\log_{10}[{\rm SFR}/{\rm SFS(M_\bigstar)}]}{\sigma_{\rm SFS}}$', fontsize=20)
    axarr[0].set_ylabel(r'$\langle \rm Pr[TF] \rangle$')
    axarr[1].set_ylabel(r'$\mathcal{R}_{\rm int}(\mathcal{S})$')
    axarr[1].set_ylim(0.5, 17.)
    axarr[0].set_xlim(-0.7, 3.)
    axarr[1].set_xlim(-0.7, 3.)
    axarr[1].set_yscale('log')

    plt.tight_layout()
    if output_dir:
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

    # Define alternative sample: spec-z only
    specz_sample = catalog.loc[(catalog['z_spec'] > 0.04) & (catalog['z_spec'] < 0.12)]

    logger.info(f"Spec-z sample size: {len(specz_sample)}")

    # Create figure
    fig, axarr = plt.subplots(1, 3, figsize=(12, 4))

    tags = {'continuum': 'continuum', 'halpha': r'H$\alpha$'}
    labels = ['Asymmetry', r'G', r'$M_{20}$']
    keys = ['asymmetry', 'gini', 'm20']
    prefix = 'halpha'

    logger.info(f"Processing spec-z sample...")

    pmerger = specz_sample['p_merger'] + specz_sample['p_ambig']
    pmerger_threshold = specz_sample['p_undisturbed']

    for idx, key in enumerate(keys):
        morph_key = f'{prefix}_{key}'
        out = ek.hist(
            hamorph.reindex(specz_sample.index)[morph_key],
            density=True,
            alpha=0.2,
            lw=2,
            color=ec.ColorBase(colorlists.slides['grey']).base,
            hatch='//',
            label='Unweighted',
            ax=axarr[idx],
            binalpha=0.005
        )
        bins = out[1][1]
        ek.hist(
            hamorph.reindex(specz_sample.index)[morph_key],
            weights=pmerger,
            density=True,
            alpha=0.4,
            lw=2.,
            color=colorlists.slides['bluebird'],
            label='Weighted by Pr[interaction]',
            ax=axarr[idx],
            bins=bins
        )
        ek.hist(
            hamorph.reindex(specz_sample.loc[(pmerger > pmerger_threshold)].index)[morph_key],
            density=True,
            alpha=0.4,
            lw=2,
            color=colorlists.slides['orange'],
            label='High-confidence mergers',
            ax=axarr[idx],
            bins=bins
        )
        if idx == 0:
            ek.text(0.025, 0.975, 'Unweighted', color='grey', ax=axarr[idx], fontsize=11)
            ek.text(0.025, 0.9, '''Weighted by
Pr[interaction]''', color=colorlists.slides['bluebird'], ax=axarr[idx], fontsize=11)
            ek.text(0.025, 0.75, '''High-confidence
mergers''', color=colorlists.slides['orange'], ax=axarr[idx], fontsize=11)
        axarr[idx].set_xlabel(rf'{labels[idx]}({tags[prefix]})')
        if idx == 0:
            axarr[idx].set_ylabel('PDF')

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

    # Define alternative sample: spec-z only
    specz_mask = (catalog['z_spec'] > 0.04) & (catalog['z_spec'] < 0.12)
    specz_sample = catalog.loc[specz_mask]
    is_satellite = is_satellite[specz_mask]

    logger.info(f"Spec-z sample size: {len(specz_sample)}")

    # Compute SFS relation
    alpha = -0.13 * 0.08 + 0.8
    norm = 1.24 * 0.08 - 1.47
    sfs_std = 0.22 * 0.08 + 0.38
    sfs = lambda logmstar: alpha * (logmstar - 8.5) + norm

    # Extract merger probabilities using DataFrame wrapper (matching make_punchlines.py)
    ptf = pd.DataFrame(
        data['prob_labels_iter'][:,2] + data['prob_labels_iter'][:,3],
        index=data['img_names'],
        columns=['ptf']
    )
    pmerger = ptf.reindex(specz_sample.index)['ptf']

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
            xs.reindex(env_indices),
            ys.reindex(env_indices),
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
            label=['Field? (sat mass-matched)', 'Satellite'][envkey]
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
        help='Comma-separated list of figure numbers to generate (4,6,7,9). If not specified, generates all.'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--use-multirun',
        action='store_true',
        help='Use multi-run analysis for uncertainty estimates (requires multiple runs in ../output/)'
    )

    parser.add_argument(
        '--run-names',
        type=str,
        default='fiducial,fiducial_rerun_0,fiducial_rerun_1,fiducial_rerun_2,fiducial_rerun_3,fiducial_rerun_4',
        help='Comma-separated list of run names to use for multi-run analysis'
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
        if args.use_multirun:
            logger.info("Using multi-run analysis for uncertainty estimates")
            run_names = [name.strip() for name in args.run_names.split(',')]
            logger.info(f"Runs: {run_names}")
            data = load_data_multirun(
                Path('../output/'),
                run_names,
                logger
            )
        else:
            logger.info("Using single-run analysis")
            data = load_data(config, logger, use_nn_classifier=True, force_pca_load=False)
            # Add mean/std fields for compatibility (single run = zero uncertainty)
            if 'mean_prob_labels' not in data:
                data['mean_prob_labels'] = data['prob_labels_iter']
            if 'std_prob_labels' not in data:
                data['std_prob_labels'] = np.zeros_like(data['prob_labels_iter'])

        # Determine which figures to generate
        if args.figures:
            figure_nums = [int(x.strip()) for x in args.figures.split(',')]
        else:
            figure_nums = [4, 6, 7, 9]

        logger.info("=" * 60)
        logger.info(f"GENERATING ALTERNATIVE SAMPLE FIGURES: {figure_nums}")
        logger.info("=" * 60)

        # Generate figures
        figure_functions = {
            4: make_figure_ha_sfs_merger_fraction_alternates,
            6: make_figure_merger_prob_vs_dsfs_alternates,
            7: make_figure_hamorph_distributions_alternates,
            9: make_figure_merger_prob_vs_environment_alternates,
        }

        for fig_num in figure_nums:
            if fig_num in figure_functions:
                figure_functions[fig_num](data, output_dir, logger)
            else:
                logger.warning(f"Unknown figure number: {fig_num}. Valid options: 4, 6, 7, 9")

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
