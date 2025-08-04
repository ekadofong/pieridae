#!/usr/bin/env python
"""
Minimal Working Example: Continuum-Subtracted Emission Line Imaging

This script demonstrates how to create continuum-subtracted emission line images
for Merian survey data without fitting models. It focuses on the core functionality
of loading multi-band cutout images, creating continuum-subtracted images, and 
generating quality assurance plots.

Key Features:
- Loads HSC gri + Merian N540/N708 cutout images for a single target
- Creates continuum-subtracted emission line images (no modeling)
- Generates a 3-panel QA figure: RGB composite + N540 + N708 emission images
- Heavily documented to explain each function's purpose and usage

Usage:
    python mwe_continuum_subtraction.py <target_id>
    
Example:
    python mwe_continuum_subtraction.py M12345
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy import coordinates, units as u
from astropy.visualization import make_lupton_rgb

# Import carpenter modules for image processing
from carpenter import conventions, pixels
from ekfplot import plot as ek
from agrias import utils, photometry
from carpenter import emission


def load_catalog_and_corrections():
    """
    Load the galaxy catalog and compute emission line corrections.
    
    This function demonstrates how to:
    1. Load a parquet catalog file containing galaxy properties
    2. Set up the object ID indexing system
    3. Estimate dust extinction (AV values)
    4. Compute various corrections needed for emission line photometry
    
    Returns:
    --------
    catalog : pandas.DataFrame
        Galaxy catalog with photometry and derived properties
    emission_corrections : tuple
        Corrections for emission line analysis including:
        - ancline_correction: Correction for adjacent line contamination  
        - ge_correction: Galactic extinction correction
        - extinction_correction: Internal dust extinction correction
        - catalog_apercorr: Aperture correction factors
    n540_fcont : array
        Continuum flux estimates for N540 band
    n708_fcont : array  
        Continuum flux estimates for N708 band
    """
    print("Loading catalog and computing emission corrections...")
    
    # Load the main galaxy catalog from parquet format
    # This catalog contains multi-band photometry and derived galaxy properties
    catalog = pd.read_parquet('../../carpenter/data/MDR1_catalogs/mdr1_n708maglt26_and_pzgteq0p1.parquet')
    
    # Set up indexing: convert Merian object IDs to standardized format
    # The original objectId_Merian values are converted to 'M<number>' format
    catalog = catalog.set_index('objectId_Merian')
    catalog.index = [f'M{sidx}' for sidx in catalog.index]
    
    # Estimate dust extinction (AV) for each galaxy
    # This uses galaxy colors and templates to estimate internal dust content
    av, u_av = emission.estimate_av(catalog)
    catalog['AV'] = av
    catalog['u_AV'] = u_av
    
    # Compute comprehensive emission line corrections
    # This includes corrections for:
    # - Galactic extinction (Milky Way dust along line of sight)
    # - Internal galaxy extinction (dust within the target galaxy)
    # - Adjacent line contamination (other emission lines in the filter)
    # - Aperture corrections (flux outside the measurement aperture)
    emission_corrections = emission.compute_emissioncorrections(catalog, logmstar_key='logmass_gaap1p0')
    
    # Extract the individual correction components
    ancline_correction, ge_correction, extinction_correction, catalog_apercorr = emission_corrections
    
    # Estimate continuum flux levels for both emission line bands
    # This uses the broadband photometry to predict the continuum level
    # under each narrow-band filter, which is needed for continuum subtraction
    
    # N708 continuum estimation (for H-alpha line)
    n708_fluxes, n708_luminosities, n708_eqws, n708_fcont = photometry.mbestimate_halpha(
        catalog[utils.photcols['N708']].values,  # N708 narrow-band flux
        catalog[utils.photcols['g']].values,     # g-band flux
        catalog[utils.photcols['r']].values,     # r-band flux  
        catalog[utils.photcols['i']].values,     # i-band flux
        catalog[utils.photcols['z']].values,     # z-band flux
        np.full(len(catalog), 0.08),             # Assumed redshift
        catalog[utils.u_photcols['N708']].values, # N708 flux uncertainty
        0., 0.,                                  # Additional uncertainties
        band='n708',                             # Specify this is N708 analysis
        apercorr=1.,                            # No aperture correction applied here
        ge_correction=ge_correction[:, 2],       # Galactic extinction correction
        ex_correction=extinction_correction[0, :, 2],  # Internal extinction correction
        u_ex_correction=0.*extinction_correction[1, :, 2],  # Extinction uncertainty
        ns_correction=ancline_correction[:],     # Adjacent line correction
        do_aperturecorrection=False,            # Don't apply aperture correction
        do_gecorrection=False,                  # Don't apply galactic extinction
        do_extinctioncorrection=False,          # Don't apply internal extinction
        do_linecorrection=True,                 # Do apply line corrections
        specflux_unit=u.nJy,                   # Flux units
        ctype='powerlaw',                       # Continuum model type
        plawbands='riz',                        # Bands used for powerlaw fit
    )
    
    # N540 continuum estimation (for [OIII] line)  
    n540_fluxes, n540_luminosities, n540_eqws, n540_fcont = photometry.mbestimate_halpha(
        catalog[utils.photcols['N540']].values,  # N540 narrow-band flux
        catalog[utils.photcols['g']].values,     # g-band flux
        catalog[utils.photcols['r']].values,     # r-band flux
        catalog[utils.photcols['i']].values,     # i-band flux  
        catalog[utils.photcols['z']].values,     # z-band flux
        np.full(len(catalog), 0.08),             # Assumed redshift
        catalog[utils.u_photcols['N540']].values, # N540 flux uncertainty
        0., 0.,                                  # Additional uncertainties
        band='n540',                             # Specify this is N540 analysis
        apercorr=1.,                            # No aperture correction applied here
        ge_correction=ge_correction[:, 2],       # Galactic extinction correction
        ex_correction=extinction_correction[0, :, 3],  # Internal extinction correction (index 3 for N540)
        u_ex_correction=0.*extinction_correction[1, :, 3],  # Extinction uncertainty
        ns_correction=ancline_correction[:],     # Adjacent line correction
        do_aperturecorrection=False,            # Don't apply aperture correction
        do_gecorrection=False,                  # Don't apply galactic extinction  
        do_extinctioncorrection=False,          # Don't apply internal extinction
        do_linecorrection=True,                 # Do apply line corrections
        specflux_unit=u.nJy,                   # Flux units
        ctype='linear',                         # Continuum model type (linear for bluer bands)
        plawbands='gr',                         # Bands used for linear fit
    )
    
    return catalog, emission_corrections, n540_fcont, n708_fcont


def load_target_images(target_name, catalog, targetid, dirname='../local_data/MDR1_starbursts/'):
    """
    Load multi-band cutout images for a specific target galaxy.
    
    This function demonstrates how to use the carpenter.pixels.BBMBImage class
    to load and organize multi-band imaging data. The BBMBImage class handles:
    - Loading cutout images and their variance maps
    - Loading PSF models for each band
    - Organizing the data by band for easy access
    - Coordinate transformations and image alignment
    
    Parameters:
    -----------
    target_name : str
        The target name in J-coordinate format (e.g., 'J123456.78+901234.5')
    catalog : pandas.DataFrame
        Galaxy catalog containing target coordinates
    targetid : str
        Standardized target ID (e.g., 'M12345')
    dirname : str
        Base directory containing the image data
        
    Returns:
    --------
    bbmb : carpenter.pixels.BBMBImage
        Multi-band image bundle containing all loaded data
    """
    print(f"Loading images for target: {targetid}")
    
    # Initialize the BBMBImage object
    # This is carpenter's main class for handling multi-band imaging data
    bbmb = pixels.BBMBImage()
    
    # Define the bands we need to load
    # HSC bands: g, r, i, z (broadband photometry)
    # Merian bands: n540 ([OIII]), n708 (H-alpha)
    bands = ['g', 'n540', 'r', 'n708', 'i', 'z']
    
    # Get target coordinates from catalog
    target_coord = coordinates.SkyCoord(
        catalog.loc[targetid, 'RA'], 
        catalog.loc[targetid, 'DEC'], 
        unit='deg'
    )
    
    # Load each band's data
    for band in bands:
        print(f"  Loading {band} band...")
        
        # Construct file paths based on band type
        if band in ['n540', 'n708']:
            # Merian narrow-band data
            cutout_path = f'{dirname}/merian/{target_name}_{band.upper()}_merim.fits'
            psf_path = f'{dirname}/merian/{target_name}_{band.upper()}_merpsf.fits'
        else:
            # HSC broadband data  
            cutout_path = f'{dirname}/hsc/{target_name}_HSC-{band}.fits'
            psf_path = f'{dirname}/hsc/{target_name}_HSC-{band}_psf.fits'
        
        # Check that required files exist
        if not os.path.exists(cutout_path):
            raise FileNotFoundError(f"Cutout image not found: {cutout_path}")
        if not os.path.exists(psf_path):
            raise FileNotFoundError(f"PSF file not found: {psf_path}")
        
        # Add this band to the BBMBImage object
        # The add_band method handles:
        # - Loading the FITS files
        # - Extracting the correct HDU extensions
        # - Setting up coordinate systems
        # - Storing variance information
        bbmb.add_band(
            band,                    # Band identifier
            target_coord,            # Target coordinates for centering
            size=100,               # Cutout size in pixels (100x100)
            image=cutout_path,      # Path to science image
            var=cutout_path,        # Path to variance image (same file, different extension)
            psf=psf_path,          # Path to PSF model
            image_ext=1,           # FITS extension containing science image
            var_ext=3,             # FITS extension containing variance map
            psf_ext=0              # FITS extension containing PSF model
        )
    
    print(f"Successfully loaded {len(bands)} bands for {targetid}")
    return bbmb


def create_continuum_subtracted_images(bbmb, catalog, targetid, n540_fcont, n708_fcont):
    """
    Create continuum-subtracted emission line images.
    
    This function demonstrates the core continuum subtraction process used
    in emission line imaging. The method:
    1. Matches PSFs across all bands to ensure consistent resolution
    2. Estimates continuum flux under each narrow-band filter
    3. Scales a reference broad-band to subtract the continuum
    4. Creates "excess" images showing only emission line flux
    
    Parameters:
    -----------
    bbmb : carpenter.pixels.BBMBImage
        Multi-band image bundle with loaded data
    catalog : pandas.DataFrame
        Galaxy catalog for flux scaling
    targetid : str
        Target identifier for catalog lookup
    n540_fcont : array
        Pre-computed continuum flux estimates for N540
    n708_fcont : array  
        Pre-computed continuum flux estimates for N708
        
    Returns:
    --------
    excess_bbmb : carpenter.pixels.BBMBImage
        New image bundle containing only the continuum-subtracted images
    """
    print("Creating continuum-subtracted images...")
    
    # Get the target's index in the catalog arrays
    # This is needed to access the pre-computed continuum flux estimates
    targetindex = np.where(np.in1d(catalog.index, targetid))[0][0]
    
    # Initialize a new BBMBImage to hold the continuum-subtracted results
    excess_bbmb = pixels.BBMBImage()
    
    # Store continuum flux estimates in a dictionary for easy access
    continuum_fluxes = {'n540': n540_fcont, 'n708': n708_fcont}
    
    # Process each emission line band
    for band in ['n540', 'n708']:
        print(f"  Processing {band} continuum subtraction...")
        
        # Step 1: Measure and match PSF sizes across all bands
        # This ensures all images have the same spatial resolution
        # The measure_psfsizes method analyzes the PSF FWHM in each band
        fwhm_dict, _ = bbmb.measure_psfsizes()
        
        # The match_psfs method convolves all images to match the worst seeing
        # Returns PSF-matched images and the convolution kernels used
        matched_images, matched_psfs = bbmb.match_psfs(refband=band)
        
        # Step 2: Compute the continuum-subtracted ("excess") image
        # The compute_mbexcess method performs the continuum subtraction:
        # excess = narrow_band - (scaling_factor * broad_band)
        excess_image, excess_variance = bbmb.compute_mbexcess(
            band,                    # Target narrow-band to subtract from
            psf_matched=True,        # Use PSF-matched images
            method='single',         # Use single broad-band for subtraction
            scaling_band='z',        # Use z-band as continuum reference
            # Scaling factor: ratio of predicted continuum flux to observed z-band flux
            scaling_factor=continuum_fluxes[band][targetindex].value / catalog.loc[targetid, utils.photcols['z']],
        )
        
        # Store the results in the excess image bundle
        excess_bbmb.image[band] = excess_image
        excess_bbmb.var[band] = excess_variance
        excess_bbmb.bands.append(band)
    
    # Clean up any artifacts from the continuum subtraction
    # This removes negative flux regions that are clearly non-physical
    excess_bbmb.clean_nonexcess_sources()
    
    print("Continuum subtraction complete!")
    return excess_bbmb


def create_qa_figure(bbmb, excess_bbmb, targetid, output_dir):
    """
    Create a 3-panel quality assurance figure.
    
    This function demonstrates how to create publication-quality figures
    showing the results of continuum subtraction. The figure layout:
    - Left panel: RGB composite from gri bands
    - Middle panel: N540 emission line image ([OIII])  
    - Right panel: N708 emission line image (H-alpha)
    
    Parameters:
    -----------
    bbmb : carpenter.pixels.BBMBImage
        Original multi-band images (for RGB composite)
    excess_bbmb : carpenter.pixels.BBMBImage  
        Continuum-subtracted emission line images
    targetid : str
        Target identifier for labeling
    output_dir : str
        Directory to save the output figure
    """
    print("Creating QA figure...")
    
    # Create figure with 3 panels arranged horizontally
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: RGB composite image
    # This demonstrates how to create false-color images from multi-band data
    try:
        # The make_lupton_rgb function creates publication-quality RGB images
        # It uses the Lupton et al. algorithm optimized for astronomical data
        rgb_image = make_lupton_rgb(
            bbmb.matched_image['i'],    # Red channel: i-band (reddest)
            bbmb.matched_image['r'],    # Green channel: r-band (middle)  
            bbmb.matched_image['g'],    # Blue channel: g-band (bluest)
            stretch=1.0,               # Stretch parameter (higher = more contrast)
            Q=3                        # Softening parameter (higher = less saturation)
        )
        
        # Display the RGB image using ekfplot's enhanced imshow
        # ekfplot.imshow provides better defaults for astronomical images
        ek.imshow(rgb_image, ax=axes[0])
        axes[0].set_title('gri RGB Composite')
        
    except Exception as e:
        # If RGB creation fails, show an error message
        axes[0].text(0.5, 0.5, f'RGB creation failed:\n{str(e)}', 
                    ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('RGB Composite (Failed)')
    
    # Panel 2: N540 emission line image ([OIII] λ5007)
    # This shows the continuum-subtracted emission from the [OIII] line
    if 'n540' in excess_bbmb.image:
        # Use ekfplot's imshow with automatic scaling
        # The q parameter sets the quantile for the color stretch
        ek.imshow(excess_bbmb.image['n540'], q=0.01, ax=axes[1])
        axes[1].set_title('N540 Emission ([OIII]λ5007)')
    else:
        axes[1].text(0.5, 0.5, 'N540 data not available', 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('N540 Emission (No Data)')
    
    # Panel 3: N708 emission line image (H-alpha)  
    # This shows the continuum-subtracted emission from the H-alpha line
    if 'n708' in excess_bbmb.image:
        # Use consistent scaling with the N540 panel
        ek.imshow(excess_bbmb.image['n708'], q=0.01, ax=axes[2])
        axes[2].set_title('N708 Emission (Hα)')
    else:
        axes[2].text(0.5, 0.5, 'N708 data not available',
                    ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('N708 Emission (No Data)')
    
    # Add target ID as figure title
    fig.suptitle(f'Target: {targetid}', fontsize=16, fontweight='bold')
    
    # Adjust layout to prevent overlapping elements
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f"{targetid}_continuum_subtraction_qa.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"QA figure saved to: {output_path}")


def process_single_target(targetid, dirname='../local_data/MDR1_starbursts/', 
                         output_dir='../local_data/pieridae_output/mwe/'):
    """
    Process a single target through the complete continuum subtraction pipeline.
    
    This is the main processing function that coordinates all the steps:
    1. Load catalog and compute corrections
    2. Convert target ID to coordinate-based filename
    3. Load multi-band images
    4. Create continuum-subtracted images  
    5. Generate QA figure
    
    Parameters:
    -----------
    targetid : str
        Target identifier (e.g., 'M12345')
    dirname : str
        Base directory containing input data
    output_dir : str  
        Directory for output files
    """
    print(f"\n{'='*60}")
    print(f"Processing target: {targetid}")
    print(f"{'='*60}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Step 1: Load catalog and compute emission line corrections
        catalog, emission_corrections, n540_fcont, n708_fcont = load_catalog_and_corrections()
        
        # Step 2: Check if target exists in catalog
        if targetid not in catalog.index:
            raise ValueError(f"Target {targetid} not found in catalog")
        
        # Step 3: Convert target ID back to coordinate-based filename
        # The catalog uses standardized M* IDs, but files use J-coordinate names
        target_name = conventions.merianobjectname_to_catalogname(targetid, catalog)
        
        # Step 4: Load multi-band cutout images
        bbmb = load_target_images(target_name, catalog, targetid, dirname)
        
        # Step 5: Create continuum-subtracted emission line images
        excess_bbmb = create_continuum_subtracted_images(bbmb, catalog, targetid, 
                                                        n540_fcont, n708_fcont)
        
        # Step 6: Generate quality assurance figure
        create_qa_figure(bbmb, excess_bbmb, targetid, output_dir)
        
        print(f"\n✓ Successfully processed {targetid}")
        
    except Exception as e:
        print(f"\n✗ Error processing {targetid}: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """
    Main function to handle command line arguments and run the pipeline.
    
    This demonstrates how to create a simple command-line interface
    for astronomical data processing scripts.
    """
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python mwe_continuum_subtraction.py <target_id>")
        print("Example: python mwe_continuum_subtraction.py M12345")
        print("\nTarget IDs should be in the format 'M' followed by numbers (e.g., M12345)")
        sys.exit(1)
    
    # Get target ID from command line
    targetid = sys.argv[1]
    
    # Validate target ID format
    if not targetid.startswith('M') or not targetid[1:].isdigit():
        print(f"Error: Invalid target ID format '{targetid}'")
        print("Target IDs should be in the format 'M' followed by numbers (e.g., M12345)")
        sys.exit(1)
    
    # Process the target
    process_single_target(targetid)
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()