#!/usr/bin/env python3
"""
Script to process all J* targets found in the MDR1_oiiiexcess directory.
Runs the pixel-level emission line analysis for each target.
"""

import os
import glob
import numpy as np
import pandas as pd
from astropy import coordinates, table
from astropy import units as u
from astropy.io import fits
import matplotlib.pyplot as plt

# Import the required modules from the notebook
from ekfplot import plot as ek
from ekfplot import colors as ec
from agrias import photometry, utils
from carpenter import emission, conventions, pixels

def extract_target_name(filepath):
    """Extract target name from HSC g-band file path."""
    filename = os.path.basename(filepath)
    # Extract J* name from filename like "J144732.50-013836.08_HSC-g.fits"
    target_name = filename.split('_HSC-g.fits')[0]
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

def check_files_exist(target, dirname):
    """Check if all required files exist for a target."""
    required_files = {
        'merian': ['N540', 'N708'],
        'hsc': ['g', 'r', 'i', 'z']
    }
    
    for subdir, bands in required_files.items():
        for band in bands:
            if subdir == 'merian':
                cutout_file = f'{dirname}/{subdir}/{target}_{band.upper()}_merim.fits'
                psf_file = f'{dirname}/{subdir}/{target}_{band.upper()}_merpsf.fits'
            else:
                cutout_file = f'{dirname}/{subdir}/{target}_HSC-{band}.fits'
                psf_file = f'{dirname}/{subdir}/{target}_HSC-{band}_psf.fits'
            
            if not os.path.exists(cutout_file) or not os.path.exists(psf_file):
                return False, f"Missing: {cutout_file} or {psf_file}"
    
    return True, "All files present"

def process_target(target, catalog, emission_corrections, dirname, output_dir=None):
    """Process a single target - equivalent to cells 86-91."""
    
    print(f"Processing target: {target}")
    
    # Convert target name to catalog name and get index (cell 86)
    try:
        targetid = conventions.merianobjectname_to_catalogname(target, catalog)
        targetindex = np.where(np.in1d(catalog.index, targetid))[0][0]
    except (IndexError, ValueError) as e:
        print(f"  Error: Target {target} not found in catalog: {e}")
        return None
    
    # Create BBMBImage and add bands (cell 87)
    try:
        bbmb = pixels.BBMBImage()
        for band in ['g','n540','r','n708','i','z']:
            if band in ['n540','n708']:
                cutout = f'{dirname}/merian/{target}_{band.upper()}_merim.fits'
                psf = f'{dirname}/merian/{target}_{band.upper()}_merpsf.fits'
            else:
                cutout = f'{dirname}/hsc/{target}_HSC-{band}.fits'
                psf = f'{dirname}/hsc/{target}_HSC-{band}_psf.fits'
            
            bbmb.add_band(
                band,
                coordinates.SkyCoord(catalog.loc[targetid,'RA'], catalog.loc[targetid,'DEC'], unit='deg'),
                size=100,
                image=cutout,
                var=cutout,
                psf=psf,
                image_ext=(1),
                var_ext=(3),
                psf_ext=0
            )
    except Exception as e:
        print(f"  Error creating BBMBImage for {target}: {e}")
        return None
    
    # Compute excess maps (cell 89)
    try:
        excess_bbmb = pixels.BBMBImage()
        fcs = {'n540': n540_fcont, 'n708': n708_fcont}
        
        for band in ['n540','n708']:
            fwhm_a, _ = bbmb.measure_psfsizes()
            mim, mpsf = bbmb.match_psfs(refband=band)
            excess_bbmb.image[band], excess_bbmb.var[band] = bbmb.compute_mbexcess(
                band, 
                psf_matched=True,
                method='single',
                scaling_band='z',
                scaling_factor=fcs[band][targetindex].value/catalog.loc[targetid,utils.photcols['z']],
            )
            excess_bbmb.bands.append(band)
    except Exception as e:
        print(f"  Error computing excess maps for {target}: {e}")
        return None
    
    # Extract emission sources and compute fluxes (cells 90-91)
    try:
        excess_bbmb.clean_nonexcess_sources()
        
        ancline_correction, ge_correction, extinction_correction, _ = emission_corrections
        
        emission_bundle = {}
        correction_indices = {'n540':3,'n708':2}
        
        for band in ['n540','n708']:
            emcat, emseg = pixels.sep.extract(excess_bbmb.image[band], 5., var=excess_bbmb.var[band], segmentation_map=True)
            if len(emcat) == 0:
                print(f'  No source detected in {band}')
                emission_bundle[band] = (None, None, np.nan, np.nan)
                continue
                
            ecatindex = emseg[emseg.shape[0]//2, emseg.shape[1]//2] - 1
            conversion = 10.**(-0.4*(27-31.4))
            integrated_flux = emission.excess_to_lineflux(float(table.Table(emcat)[ecatindex]['flux'])*conversion*u.nJy, band=band)
            integrated_flux_corrected = integrated_flux * extinction_correction[0][targetindex,correction_indices[band]]
            integrated_flux_corrected *= ge_correction[targetindex,correction_indices[band]]
            if band == 'n708':
                integrated_flux_corrected *= ancline_correction[targetindex]

            emission_bundle[band] = (emcat, emseg, integrated_flux, integrated_flux_corrected)
    except Exception as e:
        print(f"  Error extracting emission sources for {target}: {e}")
        return None
    
    # Create and save figure (based on cell 92)
    if output_dir:
        try:
            from astropy.visualization import make_lupton_rgb
            
            fig, axarr = plt.subplots(2, 2, figsize=(9, 8))
            
            # Top left: i-N540-g RGB
            rgb_img1 = make_lupton_rgb(
                bbmb.matched_image['i'],
                bbmb.matched_image['n540'],
                bbmb.matched_image['g'],
                stretch=1.,
                Q=3
            )
            ek.imshow(rgb_img1, ax=axarr[0,0])
            axarr[0,0].set_title(f'{target}: i-N540-g RGB')
            
            # Bottom left: i-N708-r RGB
            rgb_img2 = make_lupton_rgb(
                bbmb.matched_image['i'],
                bbmb.matched_image['n708'],
                bbmb.matched_image['r'],
                stretch=1.,
                Q=3
            )
            ek.imshow(rgb_img2, ax=axarr[1,0])
            axarr[1,0].set_title(f'{target}: i-N708-r RGB')
            
            # Top right: N540 excess
            ek.imshow(excess_bbmb.image['n540'], ax=axarr[0,1], cmap='Blues', q=0.)
            axarr[0,1].set_title('N540 Excess')
            
            # Bottom right: N708 excess
            ek.imshow(excess_bbmb.image['n708'], ax=axarr[1,1], cmap='Reds', q=0.)
            axarr[1,1].set_title('N708 Excess')
            
            plt.tight_layout()
            
            # Save figure
            os.makedirs(output_dir, exist_ok=True)
            fig_file = os.path.join(output_dir, f'{targetid}_emission_figure.png')
            plt.savefig(fig_file, dpi=150, bbox_inches='tight')
            plt.close(fig)  # Close to free memory
            
            print(f"  Figure saved to {fig_file}")
            
        except Exception as e:
            print(f"  Error creating figure for {target}: {e}")
    
    # Save results if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save emission bundle results
        results = {
            'target': target,
            'targetid': targetid,
            #'targetindex': targetindex,
            'ra': catalog.loc[targetid,'RA'],
            'dec': catalog.loc[targetid,'DEC'],
        }
        
        for band in ['n540', 'n708']:
            if emission_bundle[band][0] is not None:
                results[f'{band}_flux'] = emission_bundle[band][2].value
                results[f'{band}_flux_corrected'] = emission_bundle[band][3].value
                results[f'{band}_nsources'] = len(emission_bundle[band][0])
            else:
                results[f'{band}_flux'] = np.nan
                results[f'{band}_flux_corrected'] = np.nan
                results[f'{band}_nsources'] = 0
        
        # Save to CSV
        results_df = pd.DataFrame([results])
        csv_file = os.path.join(output_dir, f'{target}_emission_results.csv')
        results_df.to_csv(csv_file, index=False)
        
        print(f"  Results saved to {csv_file}")
    
    print(f"  Successfully processed {target}")
    return emission_bundle

def main(
        dirname = '../local_data/MDR1_excess_sample/', 
        output_dir = '../local_data/pieridae_output/MDR1_excess_sample/'
    ):
    """Main processing function."""
    
    # Load the catalog and compute emission corrections (from earlier cells)
    print("Loading catalog and computing corrections...")
    catalog = pd.read_parquet('../../carpenter/data/MDR1_catalogs/MDR1_merianselect_v2.0.parquet')
    emission_corrections = emission.compute_emissioncorrections(catalog)
    
    # Compute continuum estimates (from cell 79)
    print("Computing continuum estimates...")
    ancline_correction, ge_correction, extinction_correction, _ = emission_corrections
    
    global n708_fcont, n540_fcont  # Make these global so process_target can access them
    
    n708_fluxes, n708_luminosities, n708_eqws, n708_fcont = photometry.mbestimate_halpha(
        catalog[utils.photcols['N708']].values,
        catalog[utils.photcols['g']].values,
        catalog[utils.photcols['r']].values,
        catalog[utils.photcols['i']].values,
        catalog[utils.photcols['z']].values,
        np.full(len(catalog),0.08),
        catalog[utils.u_photcols['N708']].values,
        0.,
        0.,
        band='n708',
        apercorr=1.,
        ge_correction=ge_correction[:,2],
        ex_correction=extinction_correction[0,:,2],
        u_ex_correction = 0.*extinction_correction[1,:,2],
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
        np.full(len(catalog),0.08),
        catalog[utils.u_photcols['N540']].values,
        0.,
        0.,
        band='n540',
        apercorr=1.,
        ge_correction=ge_correction[:,2],
        ex_correction=extinction_correction[0,:,2],
        u_ex_correction = 0.*extinction_correction[1,:,2],
        ns_correction=ancline_correction[:],
        do_aperturecorrection=False,
        do_gecorrection=False,
        do_extinctioncorrection=False,
        do_linecorrection=True,
        specflux_unit=u.nJy,
        ctype='linear',
        plawbands='gr',
    )
    
    # Set up directories
    
    
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all available targets
    print(f"Scanning directory: {dirname}")
    targets = get_available_targets(dirname)
    print(f"Found {len(targets)} unique targets")
    
    # Process each target
    successful = 0
    failed = 0
    
    for i, target in enumerate(targets):
        print(f"\n[{i+1}/{len(targets)}] Processing {target}")
        
        # Check if all required files exist
        files_exist, message = check_files_exist(target, dirname)
        if not files_exist:
            print(f"  Skipping {target}: {message}")
            failed += 1
            continue
        
        # Process the target
        try:
            result = process_target(target, catalog, emission_corrections, dirname, output_dir)
            if result is not None:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  Error processing {target}: {e}")
            failed += 1
    
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {successful + failed}")
    
    emfiles = glob.glob(f'{output_dir}/J*csv')
    dfs = []
    for emf in emfiles:
        dfs.append(pd.read_csv(emf, index_col=0))
        os.remove(emf)
    summary = pd.concat(dfs)
    summary.to_csv(f'{output_dir}/emission_line_photometry.csv')
    
    if successful > 0:
        print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()