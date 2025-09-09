#!/usr/bin/env python
"""
Starlet Wavelet Analysis for Merian Data

This script performs starlet wavelet decomposition on Merian imaging data to detect
low surface brightness features and substructure. It generates QA figures and saves
feature maps for each target.

Based on analysis from notebooks/Starlet.ipynb
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import sep
import argparse
import pickle
from scipy import ndimage
from astropy import coordinates
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from astropy import cosmology

from ekfplot import plot as ek
from ekfstats import imstats, fit
from carpenter import pixels, conventions
from pieridae.starbursts import sample

cosmo = cosmology.FlatLambdaCDM(70.,0.3)

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


def process_starlet_analysis(target, catalog, dirname, output_dir):
    """
    Process a single target with starlet wavelet analysis
    
    Parameters:
    -----------
    target : str
        Target name (e.g., 'J095618.67+030835.28')
    catalog : pd.DataFrame
        Merian catalog with target information
    dirname : str
        Input directory containing hsc/ and merian/ subdirectories
    output_dir : str
        Output directory for figures and arrays
    """
    # Get target information
    targetid = conventions.merianobjectname_to_catalogname(target, catalog)
    if targetid not in catalog.index:
        print(f"Target {targetid} not found in catalog, skipping...")
        return
    
    print(f"Processing starlet analysis for: {targetid}")
    
    # Create output directory for this target
    target_output_dir = os.path.join(output_dir, targetid)
    os.makedirs(target_output_dir, exist_ok=True)
    
    try:
        # Load images
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
        
        # Remove contaminating sources from i-band
        _, sources = sep.extract(
            bbmb.image['i'].byteswap().newbyteorder(), 
            3, 
            var=bbmb.var['i'].byteswap().newbyteorder(), 
            segmentation_map=True
        )
        central_source = sources[sources.shape[0]//2, sources.shape[1]//2]
        central_source = sources[sources.shape[0]//2, sources.shape[1]//2]
        csource = np.where(sources==central_source, 1, 0)
        sources = np.where(sources == central_source, 0, sources)
        
        # Starlet wavelet transform
        use_gen2 = True
        wt = imstats.starlet_transform(bbmb.image['i'], gen2=use_gen2)
        
        # Process each wavelet scale
        segmap_l = []
        im_recon = []
        
        for ix in range(len(wt)):
            # Estimate noise from corners
            err_samples = [
                np.std(abs(wt[ix])[:25, -25:]),
                np.std(abs(wt[ix])[-25:, -25:]),
                np.std(abs(wt[ix])[:25, :25]),
                np.std(abs(wt[ix])[-25:, :25])
            ]
            
            # Extract features in this wavelet scale
            _, segmap = sep.extract(
                abs(wt[ix]), 
                10., 
                err=np.median(err_samples), 
                segmentation_map=True, 
                deblend_cont=1.
            )
            
            # Keep only central source features
            sidx = segmap[segmap.shape[0]//2, segmap.shape[0]//2]
            segmap_l.append(segmap)
            im_recon.append(np.where(segmap == sidx, wt[ix], 0.))
        
        # Reconstruct image and create high-frequency residual
        im_recon = imstats.inverse_starlet_transform(im_recon, gen2=use_gen2)
        hf_image = bbmb.image['i'] - im_recon
        hf_image = hf_image - ndimage.median_filter(hf_image, size=20)
        hf_image = np.where(sources > 0, 0, hf_image)
        
        # Estimate noise in high-frequency image
        err_samples = [
            np.std(hf_image[:25, -25:]),
            np.std(hf_image[-25:, -25:]),
            np.std(hf_image[:25, :25]),
            np.std(hf_image[-25:, :25])
        ]
        
        # Detect LSB and HSB features
        _, lsb_features = sep.extract(
            hf_image, 
            1, 
            err=np.median(err_samples), 
            deblend_cont=1.,
            segmentation_map=True
        )
        minarea = np.pi*(bbmb.measure_psfsizes()[0][np.in1d(bbmb.bands, 'i')])**2
        feature_cat, hsb_features = sep.extract(
            hf_image, 
            2, 
            err=np.median(err_samples), 
            minarea=minarea,
            segmentation_map=True
        )
        
        # Combine LSB and HSB features
        features = np.zeros_like(hsb_features)
        for ix in np.unique(lsb_features)[1:]:
            if (hsb_features[lsb_features == ix] > 0).any():
                features[lsb_features == ix] = 1
        
        features = ndimage.label(features)[0]
        
        # \\ remove central object
        cid = features[sources.shape[0]//2, sources.shape[1]//2]
        if cid > 0:
            features = np.where(features!=cid, features, 0)
        features = ndimage.label(features)[0]
        
        # \\ remove background red objects
        findices = np.unique(features)[1:]
        gr = -2.5*np.log10(
            ndimage.sum(bbmb.image['g'], features, findices)/
            ndimage.sum(bbmb.image['r'], features, findices)
        )
        ri = -2.5*np.log10(
            ndimage.sum(bbmb.image['r'], features, findices)/
            ndimage.sum(bbmb.image['i'], features, findices)
        )
        too_red = ri > (0.48*gr+0.2)
        for ft in findices[too_red]:
            features = np.where(features==ft, 0, features)
        features = ndimage.label(features)[0]
        
        ridge_stats = []
        for feat in np.arange(1,np.max(features)+1):
            rout = fit.fit_ridgeline_image(
                hf_image, 
                features==feat, 
                order=1, 
                return_stats=False
            )
            # coefficients, predict_func, fitted_coordinates  = rout
            ridge_stats.append(rout)
        csersic, cim = fit.fit_sersic_2d(np.where(csource, bbmb.image['i'],0.))

        rmag = -2.5*np.log10(csersic.luminosity) + 27.
        rmag_catalog = -2.5*np.log10(catalog.loc[targetid,'r_cModelFlux_Merian']*1e-9/3631.)
        absmag_r =  rmag - cosmo.distmod(0.08).value
        logmstar_adjusted = 0.4*(rmag_catalog - rmag) + catalog.loc[targetid,'logmass']        
        # Generate 4-panel QA figure
        fig, axarr = plt.subplots(1, 4, figsize=(15, 4))
        
        # Panel 0: RGB image
        ek.imshow(
            make_lupton_rgb(bbmb.image['i'], bbmb.image['n708'], bbmb.image['r'], stretch=1, Q=7), 
            axarr[0]
        )
        
        # Panel 1: Original i-band image
        ek.imshow(bbmb.image['i'], q=0.05, ax=axarr[1], cmap='Greys')
        
        # Panel 2: High-frequency image with feature contours
        ek.imshow(hf_image, q=0.01, ax=axarr[2], cmap='viridis')
        
        
        # Panel 3: Sum of segmentation maps
        ek.imshow(hf_image, q=0.01, ax=axarr[3], cmap='Greys')
        ek.contour(features, ax=axarr[3], colors='r')

        xs = np.arange(cim.shape[1])
        ys = np.arctan(csersic.theta.value)*(xs-csersic.x_0.value) + csersic.y_0.value
        ys = np.where((ys>0)&(ys<xs.max()), ys, np.nan)
        #ek.outlined_plot(central_fitted_coordinates['x'], central_predict_func(central_fitted_coordinates['x']), color='b', lw=1)        
        ek.outlined_plot(xs, ys, color='b', ls='--', lw=1, ax=axarr[3])
        ek.contour(cim, colors='b', ax=axarr[3])
        for rout in ridge_stats:
            coefficients, predict_func, fitted_coordinates = rout
            ek.outlined_plot(fitted_coordinates['x'], predict_func(fitted_coordinates['x']), color='r', lw=1, ax=axarr[3])
        
        ek.text(
            0.025,
            0.975,
            rf'''{targetid}
({target})
$\log_{{10}}(\rm M_\star/M_\odot) = {logmstar_adjusted:.2f}$
''',
            ax=axarr[0],
            color='w',
            fontsize=10
        )
        # Remove axis ticks and add titles
        titles = ['RGB (i,N708,r)', 'i-band', 'High-freq + Features', 'Wavelet Segmaps']
        for ax, title in zip(axarr, titles):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(target_output_dir, f"{targetid}_starlet_qa.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved QA figure: {fig_path}")
        
        # Save results dictionary with feature map, coefficients, and Sersic parameters
        results_dict = {
            'feature_map': features,
            'ridge_coefficients': [rout[0] for rout in ridge_stats],
            'sersic_parameters': list(zip(csersic.param_names, csersic.parameters)),
            'absmag_r':absmag_r,
            'logmass_adjusted':logmstar_adjusted,
        }
        results_path = os.path.join(target_output_dir, f"{targetid}_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(results_dict, f)
        print(f"Saved results dictionary: {results_path}")
        
        print(f"Successfully processed {targetid}")
        
    except Exception as e:
        print(f"Error processing target {targetid}: {e}")
        import traceback
        traceback.print_exc()

def singleton (target, dirname, output_dir):
    print("Loading catalog...")
    catalog, masks = sample.load_sample(filename='../../local_data/base_catalogs/mdr1_n708maglt26_and_pzgteq0p1.parquet')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each target
    print(f"Processing: {target}")
    process_starlet_analysis(target, catalog, dirname, output_dir)
    
    print("Starlet analysis complete!")    

def main(dirname, output_dir):
    """
    Main function to process all targets with starlet analysis
    
    Parameters:
    -----------
    dirname : str
        Input directory containing hsc/ and merian/ subdirectories with cutouts
    output_dir : str
        Output directory for QA figures and features arrays
    """
    print("Loading catalog...")
    catalog, masks = sample.load_sample(filename='../../carpenter/data/MDR1_catalogs/mdr1_n708maglt26_and_pzgteq0p1.parquet')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get available targets
    targets = get_available_targets(dirname)
    print(f"Found {len(targets)} targets for starlet analysis")
    
    # Process each target
    for i, target in enumerate(targets):
        print(f"Processing target {i+1}/{len(targets)}: {target}")
        process_starlet_analysis(target, catalog, dirname, output_dir)
    
    print("Starlet analysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Starlet Wavelet Analysis for Merian Data')
    parser.add_argument('-N', '--target', type=str, help='Process a single target (e.g., ABCDEF)')
    args = parser.parse_args()
    
    dirname = '../local_data/starbursts_v0/'
    output_dir = '../local_data/pieridae_output/starlet_starbursts_v0/'
    
    if args.target:
        singleton(args.target, dirname, output_dir)
    else:
        main(dirname, output_dir)