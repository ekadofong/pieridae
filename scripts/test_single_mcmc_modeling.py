#!/usr/bin/env python
"""
Single Target MCMC Modeling Test Script

This script runs MCMC modeling on a single Merian galaxy, allowing for quick testing
and debugging of the MCMC pipeline. It imports merian_mcmc_modeling.py and processes
only the specified target.

Key Features:
- Command-line interface for specifying Merian ID
- Single galaxy processing for fast iteration
- Comprehensive error handling and logging
- Same output format as full pipeline
- Debugging and QA capabilities

Usage:
    python test_single_mcmc_modeling.py <merian_id>
    
Example:
    python test_single_mcmc_modeling.py M123456
"""

import os
import sys
import numpy as np
import pandas as pd
from astropy import coordinates, units as u
import importlib.util

# Import the MCMC modeling module
import merian_mcmc_modeling as mcmc_mod
from agrias import utils, photometry
from carpenter import emission, conventions


def load_catalog_and_corrections():
    """
    Load catalog and compute corrections - same as main pipeline
    
    Returns:
    --------
    catalog : pandas.DataFrame
        Galaxy catalog with all properties
    emission_corrections : tuple
        All emission line corrections
    n540_fcont, n708_fcont : arrays
        Continuum flux estimates
    """
    print("Loading catalog and computing emission corrections...")
    
    # Load catalog
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
    
    # Compute continuum flux estimates
    print("Computing continuum flux estimates...")
    
    # N708 continuum
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
    
    # N540 continuum
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
    
    return catalog, emission_corrections, n540_fcont, n708_fcont


def find_target_file(targetid, catalog, dirname):
    """
    Find the J-coordinate filename for a given Merian ID
    
    Parameters:
    -----------
    targetid : str
        Merian ID (e.g., 'M123456')
    catalog : pandas.DataFrame
        Galaxy catalog
    dirname : str
        Data directory
    
    Returns:
    --------
    target_name : str or None
        J-coordinate filename if found, None otherwise
    """
    if targetid not in catalog.index:
        return None
    
    # Convert to J-coordinate name
    row = catalog.loc[targetid]
    target_name = conventions.produce_merianobjectname(row.RA, row.DEC)
    
    # Check if files exist
    test_file = f'{dirname}/hsc/{target_name}_HSC-g.fits'
    if os.path.exists(test_file):
        return target_name
    else:
        print(f"Warning: Files not found for {target_name}")
        return None


def process_single_target(targetid, dirname='../local_data/MDR1_mcmasses/', 
                         output_dir='../local_data/pieridae_output/single_mcmc_test/'):
    """
    Process a single target with MCMC modeling
    
    Parameters:
    -----------
    targetid : str
        Merian target ID (e.g., 'M123456')
    dirname : str
        Input data directory
    output_dir : str
        Output directory for results
    """
    print(f"\n{'='*60}")
    print(f"MCMC Modeling Test - Target: {targetid}")
    print(f"{'='*60}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Step 1: Load catalog and corrections
        catalog, emission_corrections, n540_fcont, n708_fcont = load_catalog_and_corrections()
        
        # Step 2: Check if target exists
        if targetid not in catalog.index:
            raise ValueError(f"Target {targetid} not found in catalog")
        
        print(f"Target found in catalog: {targetid}")
        print(f"  RA: {catalog.loc[targetid, 'RA']:.6f}")
        print(f"  DEC: {catalog.loc[targetid, 'DEC']:.6f}")
        print(f"  log(M*): {catalog.loc[targetid, 'logmass']:.2f}")
        
        # Step 3: Find corresponding data files
        target_name = find_target_file(targetid, catalog, dirname)
        if target_name is None:
            raise FileNotFoundError(f"Data files not found for {targetid}")
        
        print(f"Using data files for: {target_name}")
        
        # Step 4: Set up global variables needed by the processing function
        # These would normally be set in main() but we need them here
        mcmc_mod.n540_fcont = n540_fcont
        mcmc_mod.n708_fcont = n708_fcont
        mcmc_mod.is_good = np.ones(len(catalog), dtype=bool)  # For single target, just use True
        
        # Step 5: Run MCMC processing
        print("Starting MCMC processing...")
        mcmc_mod.process_target_mcmc(target_name, catalog, dirname, output_dir, emission_corrections, 
                                     rgb_bands=(['i','n708','r'], ['r','n540','g']))
        
        print(f"\n✓ Successfully processed {targetid} with MCMC modeling")
        
        # Step 6: Report output files
        target_output_dir = os.path.join(output_dir, targetid)
        if os.path.exists(target_output_dir):
            print(f"\nOutput files created in: {target_output_dir}")
            files = os.listdir(target_output_dir)
            for filename in sorted(files):
                filepath = os.path.join(target_output_dir, filename)
                size_mb = os.path.getsize(filepath) / (1024*1024)
                print(f"  {filename} ({size_mb:.2f} MB)")
        
        # Step 7: Quick summary of MCMC results
        print(f"\nMCMC Results Summary:")
        for band in ['n540', 'n708']:
            mcmc_file = os.path.join(target_output_dir, f"{targetid}_{band}_mcmc_results.npz")
            if os.path.exists(mcmc_file):
                try:
                    mcmc_data = np.load(mcmc_file)
                    print(f"  {band} band:")
                    print(f"    MLE sources: {mcmc_data['mle_n_sources']}")
                    print(f"    MLE log prob: {mcmc_data['mle_log_prob']:.2f}")
                    print(f"    Acceptance fraction: {np.mean(mcmc_data['acceptance_fraction']):.3f}")
                    if mcmc_data['mle_n_sources'] > 0:
                        print(f"    Source positions: {list(zip(mcmc_data['mle_x_sources'], mcmc_data['mle_y_sources']))}")
                        print(f"    Source fluxes: {mcmc_data['mle_fluxes']}")
                except Exception as e:
                    print(f"    Error reading MCMC results: {e}")
        
    except KeyboardInterrupt:
        print(f"\n✗ Processing interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n✗ Error processing {targetid}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """
    Main function - handle command line arguments and run MCMC test
    """
    print("Merian MCMC Modeling - Single Target Test")
    print("=" * 50)
    
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python test_single_mcmc_modeling.py <merian_id>")
        print("Example: python test_single_mcmc_modeling.py M123456")
        print("\nMerian IDs should be in the format 'M' followed by numbers")
        print("You can find valid IDs by checking the catalog or running the full pipeline first.")
        sys.exit(1)
    
    # Get target ID from command line
    targetid = sys.argv[1]
    
    # Validate target ID format
    if not targetid.startswith('M') or not targetid[1:].isdigit():
        print(f"Error: Invalid Merian ID format '{targetid}'")
        print("Merian IDs should be in the format 'M' followed by numbers (e.g., M123456)")
        sys.exit(1)
    
    print(f"Target ID: {targetid}")
    print(f"Starting MCMC modeling test...")
    
    # Process the target
    process_single_target(targetid)
    
    print(f"\n{'='*60}")
    print("MCMC modeling test complete!")
    print("Check the output directory for results and QA figures.")
    print(f'[Output directory:] ')
    stem = '/scratch/gpfs/MERIAN/user/kadofong/pixel_excess_local_data'
    print(f'{stem}/pieridae_output/single_mcmc_test/{targetid}/{targetid}_mcmc_model.png')
    print(f"{'='*60}")


if __name__ == "__main__":
    main()