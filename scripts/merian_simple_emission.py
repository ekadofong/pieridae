#!/usr/bin/env python
"""
Merian Simple Emission Line Visualization

This script creates RGB and emission line images from Merian data without any modeling.
It generates the left two panels (RGB image and emission line image) from the QA figure
in the original merian_mcmc_modeling.py script.

Simplified from merian_mcmc_modeling.py by removing all MCMC modeling components.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy import wcs, coordinates, table
from astropy import units as u
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
from astropy.visualization import make_lupton_rgb
from typing import Tuple, Optional, List

from ekfplot import plot as ek
from ekfplot import colors as ec
from agrias import photometry, utils
from carpenter import emission, conventions, pixels
from photutils.segmentation import detect_sources, deblend_sources, make_2dgaussian_kernel, SourceCatalog
from astropy.convolution import convolve
from ekfphys import calibrations
from ekfstats import fit, sampling


def extract_target_name(filepath):
    """Extract target name from filepath."""
    filename = os.path.basename(filepath)
    if '_HSC-' in filename:
        target_name = filename.split('_HSC-')[0]
    else:
        parts = filename.split('_')
        target_name = parts[0]
    return target_name


def get_available_targets(dirname):
    """Get all unique J* target names from HSC g-band files."""
    pattern = os.path.join(dirname, 'hsc', '*_HSC-g.fits')
    g_files = glob.glob(pattern)
    
    targets = []
    for filepath in g_files:
        target_name = extract_target_name(filepath)
        if target_name.startswith('J'):
            targets.append(target_name)
    
    return sorted(list(set(targets)))


def process_target_simple_emission(target, catalog, dirname, output_dir, emission_corrections, rgb_bands=None):
    """
    Process a single target to generate RGB and emission line images only
    
    This function loads the data and creates the visualization without any modeling
    """
    # Get target information
    targetid = conventions.merianobjectname_to_catalogname(target, catalog)
    if targetid not in catalog.index:
        print(f"Target {targetid} not found in catalog, skipping...")
        return   
    print(f"Processing target for simple emission visualization: {targetid}") 
    
    if rgb_bands is None:
        rgb_bands=[('i','n708','r'),('r','n540','g')]
    
    # Create output directory for this target
    target_output_dir = output_dir
    os.makedirs(target_output_dir, exist_ok=True)
    
    try:
        targetindex = np.where(np.in1d(catalog.index, targetid))[0][0]
        
        # Load images (same as original script)
        bbmb = pixels.BBMBImage()
        for band in ['g', 'n540', 'r', 'n708', 'i', 'z']:
            if band in ['n540', 'n708']:
                cutout = f'{dirname}/merian/{target}_{band.upper()}_merim.fits'
                psf = f'{dirname}/merian/{target}_{band.upper()}_merpsf.fits'
            else:
                cutout = f'{dirname}/hsc/{target}_HSC-{band}.fits'
                psf = f'{dirname}/hsc/{target}_HSC-{band}_psf.fits'
            
            if not os.path.exists(cutout) or not os.path.exists(psf):
                print(f"Missing files for {targetid} band {band}, skipping target...")
                return
            
            bbmb.add_band(
                band,
                coordinates.SkyCoord(catalog.loc[targetid, 'RA'], catalog.loc[targetid, 'DEC'], unit='deg'),
                size=150,
                image=cutout,
                var=cutout,
                psf=psf,
                image_ext=1,
                var_ext=3,
                psf_ext=0
            )
        
        # Compute excess images (same as original script)
        excess_bbmb = pixels.BBMBImage()
        fcs = {'n540': n540_fcont, 'n708': n708_fcont}
        
        for band in ['n540', 'n708']:
            fwhm_a, _ = bbmb.measure_psfsizes()
            mim, mpsf = bbmb.match_psfs(refband=band)
            excess_bbmb.image[band], excess_bbmb.var[band], _ = bbmb.compute_mbexcess(
                band,
                psf_matched=True,
                method='2dpowerlaw',
                #scaling_band='z',
                #scaling_factor=fcs[band][targetindex].value / catalog.loc[targetid, utils.photcols['z']],
            )
            excess_bbmb.bands.append(band)
        
        excess_bbmb.clean_nonexcess_sources()
        
        # Create simple figure with only RGB and emission line images
        fig, axarr = plt.subplots(2, 2, figsize=(10, 10))
        
        # RGB image (left column for both rows)
        try:
            if len(rgb_bands) == 2:
                rgb_img_n708 = make_lupton_rgb(
                    bbmb.matched_image[rgb_bands[0][0]],
                    bbmb.matched_image[rgb_bands[0][1]],
                    bbmb.matched_image[rgb_bands[0][2]],
                    stretch=3.,
                    Q=5
                )
                rgb_img_n540 = make_lupton_rgb(
                    bbmb.matched_image[rgb_bands[1][0]],
                    bbmb.matched_image[rgb_bands[1][1]],
                    bbmb.matched_image[rgb_bands[1][2]],
                    stretch=3.,
                    Q=5
                )
                ek.imshow(rgb_img_n708, ax=axarr[0,0])
                ek.imshow(rgb_img_n540, ax=axarr[1,0])                 
            else:
                rgb_img = make_lupton_rgb(
                    bbmb.matched_image[rgb_bands[0]],
                    bbmb.matched_image[rgb_bands[1]],
                    bbmb.matched_image[rgb_bands[2]],
                    stretch=3.,
                    Q=5
                )                               
                ek.imshow(rgb_img, ax=axarr[0,0])
                ek.imshow(rgb_img, ax=axarr[1,0])
        except:
            axarr[0,0].text(0.5, 0.5, 'RGB Failed', ha='center', va='center', transform=axarr[0,0].transAxes)
            axarr[1,0].text(0.5, 0.5, 'RGB Failed', ha='center', va='center', transform=axarr[1,0].transAxes)
        
        # Emission line images (right column)
        for adx, model_band in enumerate(['n708','n540']):            
            if model_band in excess_bbmb.bands:
                ek.imshow(excess_bbmb.image[model_band], q=0.01, ax=axarr[adx,1])
            else:
                axarr[adx,1].text(0.5, 0.5, f'{model_band} not available', 
                                ha='center', va='center', transform=axarr[adx,1].transAxes)

        # Add labels
        ek.text(0.025, 0.975, r'N708 (H$\alpha$)', ax=axarr[0,0], color='w', size=13)
        ek.text(0.025, 0.975, r'N540 ([OIII]5007)', ax=axarr[1,0], color='w', size=13)        
        axarr[0,0].set_title('RGB')
        axarr[0,1].set_title('N708 Excess')
        axarr[1,0].set_title('RGB')
        axarr[1,1].set_title('N540 Excess')
        
        plt.tight_layout()
        save_file = os.path.join(target_output_dir, f"{targetid}_simple_emission.png")
        print(save_file)
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save emission line arrays
        #for band in ['n708','n540']:
        #    if band in excess_bbmb.bands:
        #        np.save(
        #            os.path.join(target_output_dir, f"{targetid}_{band}_emission_arr.npy"),
        #            excess_bbmb.image[band]
        #        )
        
        print(f"Successfully processed {targetid} - simple emission visualization saved")
        
    except Exception as e:
        print(f"Error processing target {targetid}: {e}")
        import traceback
        traceback.print_exc()


def main(dirname, output_dir):
    """Main function - simplified version without MCMC"""
    print("Loading catalog and computing emission corrections...")
    catalog = pd.read_parquet('../../carpenter/data/MDR1_catalogs/mdr1_n708maglt26_and_pzgteq0p1.parquet')
    catalog = catalog.rename({'coord_ra_Merian':'RA','coord_dec_Merian':"DEC"},axis=1)
    # Estimate AV
    av, u_av = emission.estimate_av(catalog)
    catalog['AV'] = av
    catalog['u_AV'] = u_av
    
    # Compute emission corrections
    emission_corrections = emission.compute_emissioncorrections(catalog, logmstar_key='logmass_gaap1p0')
    ancline_correction, ge_correction, extinction_correction, catalog_apercorr = emission_corrections
    
    # Compute line fluxes and luminosities (needed for continuum scaling)
    print("Computing line fluxes for continuum scaling...")
    global n708_fluxes, n708_luminosities, n708_eqws, n708_fcont
    global n540_fluxes, n540_luminosities, n540_eqws, n540_fcont
    global is_good
    
    n708_fluxes, n708_luminosities, n708_eqws, n708_fcont = photometry.mbestimate_halpha(
        catalog[utils.photcols['N708']].values,
        catalog[utils.photcols['g']].values,
        catalog[utils.photcols['r']].values,
        catalog[utils.photcols['i']].values,
        catalog[utils.photcols['z']].values,
        np.full(len(catalog), 0.08),
        catalog[utils.u_photcols['N708']].values,
        0., 0.,
        band='n708',
        apercorr=1.,
        ge_correction=ge_correction[:, 2],
        ex_correction=extinction_correction[0, :, 2],
        u_ex_correction=0.*extinction_correction[1, :, 2],
        ns_correction=ancline_correction[:],
        do_aperturecorrection=False,
        do_gecorrection=False,
        do_extinctioncorrection=False,
        do_linecorrection=True,
        specflux_unit=u.nJy,
        ctype='powerlaw',
        plawbands='riz',
    )
    
    n540_fluxes, n540_luminosities, n540_eqws, n540_fcont = photometry.mbestimate_halpha(
        catalog[utils.photcols['N540']].values,
        catalog[utils.photcols['g']].values,
        catalog[utils.photcols['r']].values,
        catalog[utils.photcols['i']].values,
        catalog[utils.photcols['z']].values,
        np.full(len(catalog), 0.08),
        catalog[utils.u_photcols['N540']].values,
        0., 0.,
        band='n540',
        apercorr=1.,
        ge_correction=ge_correction[:, 2],
        ex_correction=extinction_correction[0, :, 3],
        u_ex_correction=0.*extinction_correction[1, :, 3],
        ns_correction=ancline_correction[:],
        do_aperturecorrection=False,
        do_gecorrection=False,
        do_extinctioncorrection=False,
        do_linecorrection=True,
        specflux_unit=u.nJy,
        ctype='linear',
        plawbands='gr',
    )
    
    # Add computed quantities to catalog (minimal subset needed)
    catalog['haew'] = n708_eqws[0].value
    catalog['oiiiew'] = np.where(catalog['logmass'] < 9.6, n540_eqws[0].value, np.nan)
    catalog['rmag'] = -2.5*np.log10(catalog['r_cModelFlux_Merian']*1e-9/3631.)
    catalog['L_Ha'] = (n708_luminosities[0].value * extinction_correction[0, :, 2] * 
                      ge_correction[:, 2] * catalog_apercorr)
    catalog['L_OIII'] = (n540_luminosities[0].value * extinction_correction[0, :, 3] * 
                        ge_correction[:, 3] * catalog_apercorr)
    catalog['n540_apercorr'] = catalog['N540_cModelFlux_Merian']/catalog['N540_gaap1p0Flux_Merian']
    catalog['i_apercorr'] = catalog['i_cModelFlux_Merian']/catalog['i_gaap1p0Flux_aperCorr_Merian']
    catalog['lineratio'] = catalog.loc[:, 'L_OIII']/catalog.loc[:, 'L_Ha']
    catalog['pz'] = catalog['pz1']+catalog['pz2']+catalog['pz3']+catalog['pz4']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get available targets
    targets = get_available_targets(dirname)
    print(f"Found {len(targets)} targets to process for simple emission visualization")
    
    # Process each target
    for i, target in enumerate(targets):
        if not target == 'J092918.41+002813.17':
            continue
        print(f"Processing target {i+1}/{len(targets)}: {target}")
        process_target_simple_emission(target, catalog, dirname, output_dir, emission_corrections)
    
    print("Simple emission visualization processing complete!")


if __name__ == "__main__":    
    # Set up directories
    dirname = '../local_data/MDR1_starbursts_specz/'
    output_dir = '../local_data/pieridae_output/MDR1_starbursts_specz_simple/'    
    main(dirname, output_dir)