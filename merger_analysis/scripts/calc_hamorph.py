#!/usr/bin/env python3
"""
H-alpha Morphology Analysis Script

Measures morphology in H-alpha and z-band images for Merian galaxies.
Adapted from calc_morph.py to use pieridae catalog structure and local test cutouts.

By default, only processes galaxies that have available cutouts in the specified
image directory. Use --skip-cutout-check to attempt processing all catalog galaxies.

Usage
-----
# Run on galaxies with available cutouts (default behavior)
python calc_hamorph.py --config ../configs/hamorph_config.yaml

# Run as SLURM array job (processes chunk of galaxies with cutouts)
sbatch --array=0-49 run_hamorph.sh  # where run_hamorph.sh calls this script

# Run on ALL catalog galaxies (will fail on missing cutouts)
python calc_hamorph.py --config ../configs/hamorph_config.yaml --skip-cutout-check

# Run with specific H-alpha method
python calc_hamorph.py --config ../configs/hamorph_config.yaml --ha-method zscale

# Verbose output
python calc_hamorph.py --config ../configs/hamorph_config.yaml --verbose
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import yaml
import numpy as np
from astropy.table import Table
import astropy.units as u

# Add pieridae to path (go up 2 levels: scripts/ -> merger_analysis/ -> pieridae/)
sys.path.insert(0, str(Path(__file__).parents[2]))

from carpenter import emission
from carpenter import conventions
from carpenter import pixels
from pieridae.starbursts import sample
from statmorph_joint.statmorph_joint import source_morphology_joint, source_morphology
from statmorph_joint.segmap import get_segmap
import sep


def setup_logging(verbose: bool = True) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger('hamorph_analysis')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_zband_scale_fact(mer_obj: Dict[str, Any], plawbands: str = 'riz') -> float:
    """
    Calculate z-band scaling factor for continuum subtraction.

    Parameters
    ----------
    mer_obj : dict-like
        Catalog row containing photometry columns
    plawbands : str
        Bands to use for powerlaw fit (e.g., 'riz', 'griz')

    Returns
    -------
    z_band_scale : float
        Scaling factor for z-band continuum
    """
    # Map catalog column names to expected format
    n708_flux = mer_obj['N708_gaap1p0Flux_Merian']
    g_flux = mer_obj['g_gaap1p0Flux_aperCorr_Merian']
    r_flux = mer_obj['r_gaap1p0Flux_aperCorr_Merian']
    i_flux = mer_obj['i_gaap1p0Flux_aperCorr_Merian']
    z_flux = mer_obj['z_gaap1p0Flux_aperCorr_Merian']
    n708_fluxerr = mer_obj['N708_gaap1p0FluxErr_Merian']

    # Get redshift - try multiple sources
    if 'Z_1' in mer_obj.index and np.isfinite(mer_obj['Z_1']):
        redshift = mer_obj['Z_1']
    elif 'z_spec' in mer_obj.index and np.isfinite(mer_obj['z_spec']):
        redshift = mer_obj['z_spec']
    elif 'z_ml' in mer_obj.index and np.isfinite(mer_obj['z_ml']):
        redshift = mer_obj['z_ml']
    else:
        redshift = 0.08  # Default to typical Merian redshift

    n708_fluxes, n708_luminosities, n708_eqws, n708_fcont = emission.mbestimate_emission_line(
        n708_flux,      # N708 narrow-band flux
        g_flux,         # g-band flux
        r_flux,         # r-band flux
        i_flux,         # i-band flux
        z_flux,         # z-band flux
        redshift,       # Assumed redshift
        n708_fluxerr,   # N708 flux uncertainty
        0., 0.,         # Additional uncertainties
        band='n708',                             # Specify this is N708 analysis
        apercorr=1.,                            # No aperture correction applied here
        ge_correction=None,       # Galactic extinction correction
        ex_correction=1,  # Internal extinction correction
        u_ex_correction=0.,  # Extinction uncertainty
        ns_correction=None,     # Adjacent line correction
        do_aperturecorrection=False,            # Don't apply aperture correction
        do_gecorrection=False,                  # Don't apply galactic extinction
        do_extinctioncorrection=False,          # Don't apply internal extinction
        do_linecorrection=False,                 # Do apply line corrections
        specflux_unit=u.nJy,                   # Flux units
        ctype='powerlaw',                       # Continuum model type
        plawbands=plawbands,                        # Bands used for powerlaw fit
    )

    z_band_scale = n708_fcont.value / z_flux

    return z_band_scale


def load_images(mer_obj: Dict[str, Any], image_path: str) -> pixels.BBMBImage:
    """
    Load HSC and Merian images for a single galaxy.

    Parameters
    ----------
    mer_obj : dict-like
        Catalog row containing RA, DEC coordinates
    image_path : str
        Base path to cutout directory containing hsc/ and merian/ subdirectories

    Returns
    -------
    bbmb : BBMBImage
        Loaded image object with all bands
    """
    # Get coordinates - use RA/DEC from catalog
    ra = mer_obj['RA']
    dec = mer_obj['DEC']

    # Generate object name
    objectname = conventions.produce_merianobjectname(ra, dec)

    bbmb = pixels.BBMBImage()

    # Add HSC bands
    for b in 'griz':
        bbmb.add_band(
            b,
            [ra, dec],
            100,
            image=f'{image_path}/hsc/{objectname}_HSC-{b}.fits',
            var=f'{image_path}/hsc/{objectname}_HSC-{b}.fits',
            psf=f'{image_path}/hsc/{objectname}_HSC-{b}_psf.fits',
            image_ext=1,
            var_ext=3
        )

    # Add Merian bands
    for b in ['N540', 'N708']:
        bbmb.add_band(
            b,
            [ra, dec],
            100,
            image=f'{image_path}/merian/{objectname}_{b}_merim.fits',
            var=f'{image_path}/merian/{objectname}_{b}_merim.fits',
            psf=f'{image_path}/merian/{objectname}_{b}_merpsf.fits',
            image_ext=1,
            var_ext=3
        )

    return bbmb


def make_zband_scaled_ha_im(
    mer_obj: Dict[str, Any],
    bbmb: pixels.BBMBImage,
    plawbands: str = 'riz'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create z-band scaled H-alpha image.

    Parameters
    ----------
    mer_obj : dict-like
        Catalog row containing photometry
    bbmb : BBMBImage
        Image object with loaded bands
    plawbands : str
        Bands to use for powerlaw fit

    Returns
    -------
    ha_im : ndarray
        H-alpha emission image
    ha_var : ndarray
        H-alpha variance image
    """
    z_band_scale = get_zband_scale_fact(mer_obj, plawbands=plawbands)

    fwhm_a, model_psf = bbmb.measure_psfsizes(save=True)

    match_psf_band = ['z', 'N708'][fwhm_a[np.isin(bbmb.bands, ['z', 'N708'])].argmax()]

    bbmb.reproject(refband=match_psf_band, psf_matched=False)
    _ = bbmb.match_psfs(
        refband=match_psf_band,
        verbose=False,
        w_type='cosine',
        reprojected=True,
        cbell_alpha=0.5
    )

    ha_im, ha_var, _ = bbmb.compute_mbexcess(
        band='N708',
        method='single',
        scaling_factor=z_band_scale,
        scaling_band='z',
        psf_matched=True
    )

    return ha_im, ha_var


def make_plaw_pixbypix_ha_im(
    mer_obj: Dict[str, Any],
    bbmb: pixels.BBMBImage,
    post_smooth: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create pixel-by-pixel powerlaw H-alpha image.

    Parameters
    ----------
    mer_obj : dict-like
        Catalog row
    bbmb : BBMBImage
        Image object with loaded bands
    post_smooth : bool
        Apply post-smoothing to H-alpha image

    Returns
    -------
    ha_im : ndarray
        H-alpha emission image
    ha_var : ndarray
        H-alpha variance image
    """
    fwhm_a, model_psf = bbmb.measure_psfsizes(save=True)

    match_psf_band = ['r', 'i', 'z', 'N708'][
        fwhm_a[np.isin(bbmb.bands, ['r', 'i', 'z', 'N708'])].argmax()
    ]

    bbmb.reproject(refband=match_psf_band, psf_matched=False)
    _ = bbmb.match_psfs(
        refband=match_psf_band,
        verbose=False,
        w_type='cosine',
        reprojected=True,
        cbell_alpha=0.5
    )

    ha_im, ha_var, _ = bbmb.compute_mbexcess(
        band='N708',
        method='2dpowerlaw',
        psf_matched=True,
        post_smooth=post_smooth
    )

    return ha_im, ha_var


def make_ri_avg_ha_im(
    mer_obj: Dict[str, Any],
    bbmb: pixels.BBMBImage
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create r-i average H-alpha image.

    Parameters
    ----------
    mer_obj : dict-like
        Catalog row
    bbmb : BBMBImage
        Image object with loaded bands

    Returns
    -------
    ha_im : ndarray
        H-alpha emission image
    ha_var : ndarray
        H-alpha variance image
    """
    fwhm_a, model_psf = bbmb.measure_psfsizes(save=True)

    match_psf_band = ['r', 'i', 'N708'][
        fwhm_a[np.isin(bbmb.bands, ['r', 'i', 'N708'])].argmax()
    ]

    bbmb.reproject(refband=match_psf_band, psf_matched=False)
    _ = bbmb.match_psfs(
        refband=match_psf_band,
        verbose=False,
        w_type='cosine',
        reprojected=True,
        cbell_alpha=0.5
    )

    ha_im, ha_var, _ = bbmb.compute_mbexcess(
        band='N708',
        method='average',
        scaling_band='ri',
        psf_matched=True
    )

    return ha_im, ha_var


def measure_ha_z_morph(
    mer_obj: Dict[str, Any],
    config: dict,
    logger: logging.Logger,
    plawbands: str = 'riz',
    save: bool = True,
    return_morph: bool = False,
    savedir: Optional[str] = None,
    ha_im_method: str = 'zscale',
    post_smooth: bool = False
) -> Optional[Tuple]:
    """
    Measure H-alpha and z-band morphology for a single galaxy.

    Parameters
    ----------
    mer_obj : dict-like
        Catalog row for galaxy
    config : dict
        Configuration dictionary
    logger : Logger
        Logging object
    plawbands : str
        Bands for powerlaw fit (for zscale method)
    save : bool
        Save morphology results to file
    return_morph : bool
        Return morphology objects
    savedir : str, optional
        Directory to save results
    ha_im_method : str
        H-alpha image generation method ('zscale', 'plaw_pixbypix', 'ri_avg')
    post_smooth : bool
        Apply post-smoothing (plaw_pixbypix only)

    Returns
    -------
    morph_results : tuple or None
        If return_morph=True, returns (morph_z, morph_joint, morph_ha)
    """
    
    # Load images
    image_path = config['cutout_data']['image_path']
    bbmb = load_images(mer_obj, image_path)

    # Generate H-alpha image based on method
    if ha_im_method == 'zscale':
        image_ha, var_ha = make_zband_scaled_ha_im(mer_obj, bbmb, plawbands=plawbands)
    elif ha_im_method == 'plaw_pixbypix':
        image_ha, var_ha = make_plaw_pixbypix_ha_im(mer_obj, bbmb, post_smooth=post_smooth)
    elif ha_im_method == 'ri_avg':
        image_ha, var_ha = make_ri_avg_ha_im(mer_obj, bbmb)
    else:
        raise ValueError(f"Unknown ha_im_method: {ha_im_method}")

    # Get z-band images
    image_z = bbmb.matched_image['z']
    var_z = bbmb.matched_var['z']
    psf = bbmb.matched_psf['N708']

    # Create segmentation map
    segmap, mask = get_segmap(image_z, psf)

    if len(segmap.labels) == 0:
        logger.warning(f"No sources found in object index {mer_obj.name} - SKIPPING")
        return None

    # Compute backgrounds
    ha_bkg = sep.Background(image_ha.astype(image_ha.dtype.newbyteorder('=')))
    z_bkg = sep.Background(image_z.astype(image_z.dtype.newbyteorder('=')))

    # Measure morphology
    morph_ha = source_morphology(
        image_ha - ha_bkg,
        segmap,
        weightmap=var_ha**0.5,
        mask=mask
    )[0]

    morph_joint = source_morphology_joint(
        [image_z - z_bkg, image_ha - ha_bkg],
        segmap,
        weightmaps=[var_z**0.5, var_ha**0.5],
        mask=mask
    )

    # Save results
    if save:
        # Get object ID - try different column names
        if hasattr(mer_obj, 'name'):
            object_id = mer_obj.name
        else:
            object_id = mer_obj.get('objectId_Merian', mer_obj.name)

        save_path = os.path.join(savedir, f'{object_id}_morph.npy')
        np.save(save_path, (*morph_joint, morph_ha))
        logger.info(f"Saved index {mer_obj.name} to {save_path}")

    if return_morph:
        return (*morph_joint, morph_ha)

    return None


def filter_catalog_with_cutouts(
    catalog: Any,
    image_path: str,
    logger: logging.Logger
) -> Any:
    """
    Filter catalog to only include galaxies with available cutouts.

    Parameters
    ----------
    catalog : DataFrame
        Full catalog
    image_path : str
        Path to cutout directory
    logger : Logger
        Logging object

    Returns
    -------
    filtered_catalog : DataFrame
        Catalog containing only galaxies with available cutouts
    """
    import pandas as pd

    logger.info("Checking for available cutouts...")

    has_cutouts = []
    for idx, row in catalog.iterrows():
        objectname = conventions.produce_merianobjectname(row['RA'], row['DEC'])

        # Check if HSC g-band file exists (as proxy for all files)
        hsc_file = f'{image_path}/hsc/{objectname}_HSC-g.fits'
        merian_file = f'{image_path}/merian/{objectname}_N708_merim.fits'

        if os.path.exists(hsc_file) and os.path.exists(merian_file):
            has_cutouts.append(True)
        else:
            has_cutouts.append(False)

    filtered_catalog = catalog[has_cutouts]

    logger.info(f"Found {len(filtered_catalog)} galaxies with available cutouts "
                f"(out of {len(catalog)} total)")

    return filtered_catalog


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Measure H-alpha and z-band morphology for Merian galaxies'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='../configs/hamorph_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--ha-method',
        type=str,
        default=None,
        help='H-alpha method override (zscale, plaw_pixbypix, ri_avg)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--skip-cutout-check',
        action='store_true',
        help='Skip checking for available cutouts (process all catalog entries)'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    verbose = args.verbose or config['processing'].get('verbose', True)
    logger = setup_logging(verbose=verbose)

    logger.info(f"Loading configuration from {args.config}")

    # Load catalog
    catalog_file = config['catalog']['catalog_file']
    mask_name = config['catalog']['mask_name']

    logger.info(f"Loading catalog from {catalog_file}")
    full_catalog, masks = sample.load_sample(catalog_file)
    base_catalog = full_catalog.loc[masks[mask_name][0]]

    logger.info(f"Loaded {len(base_catalog)} galaxies with mask '{mask_name}'")

    # Filter catalog to only include galaxies with available cutouts
    image_path = config['cutout_data']['image_path']
    if not args.skip_cutout_check:
        base_catalog = filter_catalog_with_cutouts(base_catalog, image_path, logger)
    else:
        logger.info("Skipping cutout check (--skip-cutout-check flag set)")

    # Setup output directory
    save_dir = config['output']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Output directory: {save_dir}")

    # Get analysis parameters
    ha_im_method = args.ha_method or config['analysis']['ha_im_method']
    plawbands = config['analysis']['plawbands']
    post_smooth = config['analysis']['post_smooth']
    skip_objects = config['processing'].get('skip_objects', [])

    logger.info(f"H-alpha method: {ha_im_method}")
    logger.info(f"Powerlaw bands: {plawbands}")
    logger.info(f"Post-smooth: {post_smooth}")

    # Check if running as SLURM array job
    slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")

    if slurm_task_id is not None:
        # SLURM array mode - process chunk
        idx = int(slurm_task_id)
        nchunk = config['processing']['n_chunks']
        chunk_ind = np.arange(
            idx * len(base_catalog) // nchunk,
            (idx + 1) * len(base_catalog) // nchunk
        )
        logger.info(f"SLURM array mode: processing indices {chunk_ind[0]} to {chunk_ind[-1]}")
        indices_to_process = chunk_ind
    else:
        # Serial mode - process all galaxies
        logger.info("Serial mode: processing all galaxies")
        indices_to_process = range(len(base_catalog))

    # Process galaxies
    n_success = 0
    n_failed = 0
    n_skipped = 0

    for i in indices_to_process:
        mer_obj = base_catalog.iloc[i]

        # Check if object should be skipped
        object_id = mer_obj.name
        if object_id in skip_objects:
            logger.info(f"Skipping object {object_id} (in skip list)")
            n_skipped += 1
            continue

        try:
            measure_ha_z_morph(
                mer_obj,
                config,
                logger,
                plawbands=plawbands,
                save=config['output']['save_results'],
                savedir=save_dir,
                ha_im_method=ha_im_method,
                post_smooth=post_smooth,
                return_morph=config['output']['return_morph']
            )
            n_success += 1
        except KeyboardInterrupt as e:
            logger.error(f"Failed on index {i} (object {object_id}): {str(e)}")
            n_failed += 1
            continue

    # Summary
    logger.info("=" * 60)
    logger.info(f"Processing complete!")
    logger.info(f"Successful: {n_success}")
    logger.info(f"Failed: {n_failed}")
    logger.info(f"Skipped: {n_skipped}")
    logger.info(f"Total processed: {len(indices_to_process)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
