#!/usr/bin/env python
# coding: utf-8

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
    # For HSC files: target_HSC-band.fits
    # For Merian files: target_BAND_type.fits
    if '_HSC-' in filename:
        target_name = filename.split('_HSC-')[0]
    else:
        # For Merian files, take everything before the first underscore after 'J'
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




def process_target(target, catalog, dirname, output_dir, emission_corrections):
    """Process a single target and save outputs."""    
    # Get target information
    targetid = conventions.merianobjectname_to_catalogname(target, catalog)
    if targetid not in catalog.index:
        print(f"Target {targetid} not found in catalog, skipping...")
        return   
    print(f"Processing target: {targetid}") 
    
    # Create output directory for this target
    target_output_dir = os.path.join(output_dir, targetid)
    os.makedirs(target_output_dir, exist_ok=True)
    
    try:

            
        targetindex = np.where(np.in1d(catalog.index, targetid))[0][0]
        
        # Load images
        bbmb = pixels.BBMBImage()
        for band in ['g', 'n540', 'r', 'n708', 'i', 'z']:
            if band in ['n540', 'n708']:
                cutout = f'{dirname}/merian/{target}_{band.upper()}_merim.fits'
                psf = f'{dirname}/merian/{target}_{band.upper()}_merpsf.fits'
            else:
                cutout = f'{dirname}/hsc/{target}_HSC-{band}.fits'
                psf = f'{dirname}/hsc/{target}_HSC-{band}_psf.fits'
            
            # Check if files exist
            if not os.path.exists(cutout) or not os.path.exists(psf):
                print(f"Missing files for {targetid} band {band}, skipping target...")
                return
            
            bbmb.add_band(
                band,
                coordinates.SkyCoord(catalog.loc[targetid, 'RA'], catalog.loc[targetid, 'DEC'], unit='deg'),
                size=100,
                image=cutout,
                var=cutout,
                psf=psf,
                image_ext=1,
                var_ext=3,
                psf_ext=0
            )
        
        # Compute excess images
        excess_bbmb = pixels.BBMBImage()
        fcs = {'n540': n540_fcont, 'n708': n708_fcont}
        
        for band in ['n540', 'n708']:
            fwhm_a, _ = bbmb.measure_psfsizes()
            mim, mpsf = bbmb.match_psfs(refband=band)
            excess_bbmb.image[band], excess_bbmb.var[band] = bbmb.compute_mbexcess(
                band,
                psf_matched=True,
                method='single',
                scaling_band='z',
                scaling_factor=fcs[band][targetindex].value / catalog.loc[targetid, utils.photcols['z']],
            )
            excess_bbmb.bands.append(band)
        
        excess_bbmb.clean_nonexcess_sources()
        
        # Process emission lines
        ancline_correction, ge_correction, extinction_correction, catalog_apercorr = emission_corrections
        emission_bundle = {}
        correction_indices = {'n540': 3, 'n708': 2}

        
        for band in ['n540', 'n708']:
            conversion = 10.**(-0.4*(27-31.4))
            
            total_detimage = bbmb.image[band]
            total_segm = detect_sources(
                    total_detimage, 
                    threshold=5.*sampling.sigmaclipped_std(bbmb.image[band], low=4., high=3.),
                    npixels=5,
                    connectivity=8
            )
            #total_segm_deblend = deblend_sources(total_detimage, total_segm,
            #                    npixels=10, nlevels=32, contrast=0.001,
            #                    progress_bar=False)
            if total_segm is None:
                print(f'No sources detected in {targetid} cutout!')
                emission_bundle[band] = (None, None, np.nan*u.erg/u.s/u.cm**2, np.nan, None, None)
                continue                   
            sid = total_segm.data[total_detimage.shape[0]//2, total_detimage.shape[1]//2]
            if sid == 0:
                print(f'No source detected in _detection_ band for {targetid}')
                emission_bundle[band] = (None, None, np.nan*u.erg/u.s/u.cm**2, np.nan, None, None)
                continue   
            
            total_cat = SourceCatalog(excess_bbmb.image[band], total_segm)   
            bb_cat = SourceCatalog(bbmb.matched_image['r'], total_segm)          
            catindex = total_segm.get_index(sid)
            
            total_flux = emission.excess_to_lineflux(total_cat.kron_flux[catindex]  * conversion *u.nJy, band)
            total_flux *= extinction_correction[0][targetindex,correction_indices[band]] * ge_correction[targetindex,correction_indices[band]]
            if band == 'n708':
                total_flux *= ancline_correction[targetindex]   
            total_bbflux = bb_cat.kron_flux[catindex] * conversion
                            
            # Segment image for source detection
            detimage = excess_bbmb.image[band] - np.nanmedian(excess_bbmb.image[band])
            segm = detect_sources(
                detimage,
                threshold=5. * sampling.sigmaclipped_std(excess_bbmb.image[band], low=4., high=3.),
                npixels=5,
                connectivity=8
            )
            
            if segm is None:
                print(f'No source detected in {band} for {targetid}')
                emission_bundle[band] = (None, None, total_flux, np.nan, None, None)
                continue
                
            segm_deblend = deblend_sources(detimage, segm,
                                         npixels=10, nlevels=32, contrast=0.001,
                                         progress_bar=False)
            cat = SourceCatalog(detimage, segm_deblend)
            
            if len(cat) == 0:
                print(f'No source detected in {band} for {targetid}')
                emission_bundle[band] = (None, None, total_flux, np.nan, None, None)
                continue
            
            model_obj, model_pred = fit.fit_multi_moffat_2d(
                excess_bbmb.image[band],
                init_x_0=cat.xcentroid,
                init_y_0=cat.ycentroid,
                psf_fwhm=bbmb.fwhm_to_match
            ) 

            integrated_flux = emission.excess_to_lineflux(cat.kron_flux*conversion*u.nJy, band)
            integrated_flux_corrected = integrated_flux * extinction_correction[0][targetindex, correction_indices[band]]
            integrated_flux_corrected *= ge_correction[targetindex, correction_indices[band]]
            if band == 'n708':
                integrated_flux_corrected *= ancline_correction[targetindex]        
            
            emission_bundle[band] = (cat, segm_deblend, total_flux, integrated_flux_corrected, model_obj, model_pred)
        
        # Create and save figures
        # Figure 1: Model comparison
        fig, axarr = plt.subplots(2, 4, figsize=(14, 8))
        
        # RGB image
        try:
            rgb_img = make_lupton_rgb(
                bbmb.matched_image['i'],
                bbmb.matched_image['r'],
                bbmb.matched_image['g'],
                stretch=1.,
                Q=3
            )
            ek.imshow(rgb_img, ax=axarr[0,0])
            ek.imshow(rgb_img, ax=axarr[1,0])
        except:
            axarr[0,0].text(0.5, 0.5, 'RGB Failed', ha='center', va='center', transform=axarr[0].transAxes)
            axarr[1,0].text(0.5, 0.5, 'RGB Failed', ha='center', va='center', transform=axarr[0].transAxes)
        
        # Choose band for model display (prefer n708, fallback to n540)
        #model_band = 'n708' if emission_bundle.get('n708', [None])[0] is not None else 'n540'
        for adx,model_band in enumerate(['n708','n540']):            
            _, _, _, _ , _, model_pred = emission_bundle[model_band] if len(emission_bundle[model_band]) > 3 else (None, None, None, None, None, None)
            
            
            if model_pred is not None:
                alpha = 1e-4
                lim = np.nanquantile(model_pred,1.-alpha)
                axarr[adx,1].imshow(excess_bbmb.image[model_band], 
                                vmin=-lim, vmax=lim, origin='lower')
                axarr[adx,2].imshow(model_pred, 
                                    vmin=-lim, vmax=lim, origin='lower')
                axarr[adx,3].imshow(excess_bbmb.image[model_band] - model_pred, 
                            vmin=-lim, 
                            vmax=lim, 
                            origin='lower')
            else:
                ek.imshow(excess_bbmb.image[model_band], q=0.01, ax=axarr[adx,1])
            
            if total_segm is not None:
                for ax in axarr[adx]:
                    ek.contour(
                        total_segm,
                        levels=[0,1],
                        ax=ax,
                        linestyles='--',
                        colors='lightgrey'
                    )

        ek.text(0.025, 0.975, r'N708 (H$\alpha$)', ax=axarr[0,0], color='w', size=13)
        ek.text(0.025, 0.975, r'N540 ([OIII]5007)', ax=axarr[1,0], color='w', size=13)        
        axarr[0,0].set_title('RGB')
        axarr[0,1].set_title('Excess')
        axarr[0,2].set_title('Model')
        axarr[0,3].set_title('Residual')
        
        plt.tight_layout()
        plt.savefig(os.path.join(target_output_dir, f"{targetid}_model.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Figure 2: SFS comparison
        fig, axarr = plt.subplots(1, 2, figsize=(14, 5))
        
        for adx, key in enumerate(['L_Ha', 'L_OIII']):
            try:
                im, ax = ek.hist2d(
                    catalog.loc[is_good, 'logmass'],
                    catalog.loc[is_good, key],
                    yscale='log',
                    bins=50,
                    ax=axarr[adx],
                )
                
                for qt in [0.01, 0.05, 0.16]:
                    _ = ek.running_quantile(
                        catalog.loc[is_good, 'logmass'],
                        catalog.loc[is_good, key],
                        bins=np.linspace(7.6,9.6,20),
                        std_format='fill_between',
                        color='r',
                        ax=axarr[adx],
                        alpha=qt,
                        std_alpha=0.09,
                        zorder=1
                    )
                
                # Plot target
                band_map = {'L_Ha': 'n708', 'L_OIII': 'n540'}
                band = band_map[key]
                if emission_bundle.get(band, [None])[3] is not None:
                    region_flux = emission_bundle[band][3]
                    if np.isnan(region_flux).all():
                        region_lum = 0.*u.erg/u.s                      
                    elif len(region_flux) > 0:
                        region_lum = region_flux[0] * 4. * np.pi * cosmo.luminosity_distance(0.08)**2
                        region_lum = region_lum.to(u.erg/u.s)
                    else:
                        raise ValueError                      

                    total_flux = emission_bundle[band][2]
                    total_lum = (total_flux * 4.*np.pi*cosmo.luminosity_distance(0.08)**2).to(u.erg/u.s)                

                    axarr[adx].scatter(
                        catalog.loc[targetid, 'logmass'],
                        total_lum.value,
                        zorder=3,
                        color='lime',
                        s=11**2,
                        label='Total'
                    )
                    axarr[adx].scatter(
                        catalog.loc[targetid, 'logmass'],
                        np.nanmax(region_lum.value),
                        edgecolor='b',
                        facecolor='None',
                        s=121,
                        label='Brightest Region'
                    )
                    axarr[adx].scatter(
                        catalog.loc[targetid, 'logmass'],
                        total_lum.value - np.nanmax(region_lum.value),
                        color='b',
                        s=121,
                        label='Catalog - Brightest Region'
                    )                    

                #axarr[adx].axvline(np.log10(2.7)+9., color='k', ls='--', alpha=0.7, label='LMC M*')
                #axarr[adx].axvline(np.log10(3.1)+8., color='k', ls='--', alpha=0.7, label='SMC M*')
                axarr[adx].set_ylim(3e38, 2e41)
                axarr[adx].set_xlabel('log M* [M☉]')
                axarr[adx].set_ylabel(f'{key} [erg/s]')
                axarr[adx].legend(fontsize=7)
                
            except IOError as e:
                print(f"Error creating SFS plot for {key}: {e}")
                axarr[adx].text(0.5, 0.5, f'Error: {key}', ha='center', va='center', transform=axarr[adx].transAxes)
        
        plt.tight_layout()
        plt.savefig(os.path.join(target_output_dir, f"{targetid}_sfs.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save photometry catalogs
        for band in ['n540', 'n708']:
            if emission_bundle.get(band, [None])[0] is not None:
                cat, _, total_flux, integrated_flux_corrected, model_obj, _ = emission_bundle[band]
                
                # Convert to astropy table and add corrected flux
                phot_table = cat.to_table()
                phot_table['lineflux_corrected'] = integrated_flux_corrected                
                phot_table['band'] = band
                phot_table['target'] = target
                
                
                mod_components = fit.extract_moffat_component_parameters(model_obj, len(cat))
                region_model_fluxes = []
                model_amplitudes = []
                model_pixfluxes = []
                for region in mod_components:
                    amplitude = region['amplitude']
                    gamma = region['gamma']
                    alpha = region['alpha']
                    moffat_integral = amplitude * np.pi*gamma**2/(alpha-1)
                    model_flux = emission.excess_to_lineflux(moffat_integral*conversion*u.nJy, band)
                    model_flux = model_flux * extinction_correction[0][targetindex, correction_indices[band]]
                    model_flux *= ge_correction[targetindex, correction_indices[band]]    
                    region_model_fluxes.append(model_flux)
                    model_amplitudes.append(amplitude)
                    model_pixfluxes.append(moffat_integral)
                                    
                phot_table['model_lineflux'] = 0.
                phot_table['model_lineflux'] = region_model_fluxes
                phot_table['model_integral'] = model_pixfluxes
                phot_table['model_amplitude'] = model_amplitudes
                
                integrated_row = {
                    'label':-1,
                    'lineflux_corrected':total_flux,                    
                    'band':band,
                    'target':target
                }
                phot_table.add_row(integrated_row)
                integrated_row = {
                    'label':-2,
                    'kron_flux':total_bbflux,
                    'band':'r',
                    'target':target
                }                
                phot_table.add_row(integrated_row)
                
                # Save to CSV
                phot_table.write(
                    os.path.join(target_output_dir, f"{targetid}_{band}_region_photometry.csv"),
                    format='csv',
                    overwrite=True
                )
        
        print(f"Successfully processed {targetid}")
        
    except IOError as e:
        print(f"Error processing target {targetid}: {e}")
        import traceback
        traceback.print_exc()


def main():
    # Load catalog and compute emission corrections
    print("Loading catalog and computing emission corrections...")
    catalog = pd.read_parquet('../../carpenter/data/MDR1_catalogs/mdr1_n708maglt26_and_pzgteq0p1.parquet')
    catalog = catalog.set_index('objectId_Merian')
    catalog.index = [f'M{sidx}' for sidx in catalog.index]
    
    # Estimate AV
    av, u_av = emission.estimate_av(catalog)
    catalog['AV'] = av
    catalog['u_AV'] = u_av
    
    # Compute emission corrections
    emission_corrections = emission.compute_emissioncorrections(catalog, logmstar_key='logmass_gaap1p0')
    ancline_correction, ge_correction, extinction_correction, catalog_apercorr = emission_corrections
    
    # Compute line fluxes and luminosities
    print("Computing line fluxes and luminosities...")
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
        ex_correction=extinction_correction[0, :, 2],
        u_ex_correction=0.*extinction_correction[1, :, 2],
        ns_correction=ancline_correction[:],
        do_aperturecorrection=False,
        do_gecorrection=False,
        do_extinctioncorrection=False,
        do_linecorrection=True,
        specflux_unit=u.nJy,
        ctype='linear',
        plawbands='gr',
    )
    
    # Add computed quantities to catalog
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
    
    # Define selection criteria
    is_mcmass = (catalog['logmass']>7.75)&(catalog['logmass']<9.4)&(catalog['i_apercorr']<4.)&(catalog['n540_apercorr']<4.)
    is_emitter = (catalog['haew']>5.)&(catalog['oiiiew']>5.)
    in_band = (catalog['z_spec']>0.05)&(catalog['z_spec']<0.11)
    zphot_select = catalog['pz']>0.26
    is_good = is_mcmass & zphot_select & is_emitter
    
    has_zspec = np.isfinite(catalog['z_spec'])
    in_band[~has_zspec] = np.nan
    
    # Set up directories
    dirname = '../local_data/MDR1_mcmasses/'
    output_dir = '../local_data/pieridae_output/MDR1_mcmasses/'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get available targets
    targets = get_available_targets(dirname)
    print(f"Found {len(targets)} targets to process")
    
    # Process each target
    for i, target in enumerate(targets):
        print(f"Processing target {i+1}/{len(targets)}: {target}")
        process_target(target, catalog, dirname, output_dir, emission_corrections)
    
    print("Processing complete!")


if __name__ == "__main__":
    main()