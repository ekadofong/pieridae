#!/usr/bin/env python
"""
Merian MCMC Modeling for Region Photometry

This script performs MCMC modeling of point sources in Merian emission line data,
replacing the traditional Moffat fitting approach with a full Bayesian analysis.
It uses the data format and structure from do_region_photometry.py but replaces
the modeling at ~L180 with MCMC-based inference.

Key Features:
- Uses the same data loading and continuum subtraction as do_region_photometry.py
- Replaces fit.fit_multi_moffat_2d with MCMC modeling using emcee
- PSF modeled as Moffat profile with alpha=2.5 (astropy Moffat2D parameterization)
- MLE estimate used as fiducial model
- Comprehensive QA figures and model persistence
- Compatible output format with existing pipeline
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
from astropy.modeling import models
from typing import Tuple, Optional, List
import emcee
import corner
from scipy import special
from scipy.ndimage import maximum_filter

from ekfplot import plot as ek
from ekfplot import colors as ec
from agrias import photometry, utils
from carpenter import emission, conventions, pixels
from photutils.segmentation import detect_sources, deblend_sources, make_2dgaussian_kernel, SourceCatalog
from astropy.convolution import convolve
from ekfphys import calibrations
from ekfstats import fit, sampling


class MoffatPSF:
    """
    Moffat PSF model using astropy Moffat2D parameterization
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 2.5):
        """
        Initialize Moffat PSF model
        
        Parameters:
        -----------
        gamma : float
            Core width parameter (HWHM)
        alpha : float  
            Power law index (fixed at 2.5)
        """
        self.gamma = gamma
        self.alpha = alpha
        # Normalization for astropy Moffat2D parameterization
        self.normalization = (alpha - 1) / (np.pi * gamma**2)
    
    def __call__(self, x_grid: np.ndarray, y_grid: np.ndarray, 
                 x_center: np.ndarray, y_center: np.ndarray) -> np.ndarray:
        """
        Evaluate PSF at grid positions for given source centers using astropy parameterization
        
        Parameters:
        -----------
        x_grid, y_grid : np.ndarray, shape (H, W)
            Coordinate grids
        x_center, y_center : np.ndarray, shape (N_sources,)
            Source centers
        
        Returns:
        --------
        psf_vals : np.ndarray, shape (N_sources, H, W)
            PSF values for each source
        """
        if len(x_center) == 0:
            return np.zeros((0, x_grid.shape[0], x_grid.shape[1]))
        
        # Broadcast for vectorized computation
        x_center = x_center[:, None, None]  # (N_sources, 1, 1)
        y_center = y_center[:, None, None]  # (N_sources, 1, 1)
        
        # Distance from center
        r_sq = (x_grid[None, :, :] - x_center)**2 + (y_grid[None, :, :] - y_center)**2
        
        # Moffat profile using astropy parameterization
        # f(r) = A * (1 + (r/gamma)^2)^(-alpha)
        psf = self.normalization * np.power(1 + r_sq / (self.gamma**2), -self.alpha)
        
        return psf


class MerianPointSourceModel:
    """
    Forward model for point sources in Merian data
    Adapted for real astronomical data with proper noise handling
    """
    
    def __init__(self, psf: MoffatPSF, image: np.ndarray, variance: np.ndarray):
        """
        Initialize point source model
        
        Parameters:
        -----------
        psf : MoffatPSF
            PSF model to use
        image : np.ndarray
            Observed image data
        variance : np.ndarray
            Variance map for the image
        """
        self.psf = psf
        self.image = image
        self.variance = variance
        self.H, self.W = image.shape
        
        # Create coordinate grids
        y, x = np.meshgrid(np.arange(self.H, dtype=np.float32), 
                          np.arange(self.W, dtype=np.float32), 
                          indexing='ij')
        self.x_grid = x
        self.y_grid = y
        
        # Estimate noise characteristics
        self.noise_std = np.sqrt(np.nanmedian(variance))
    
    def forward(self, x_sources: np.ndarray, y_sources: np.ndarray, 
                fluxes: np.ndarray) -> np.ndarray:
        """Generate model image from point source parameters"""
        if len(x_sources) == 0:
            return np.zeros((self.H, self.W))
        
        psf_vals = self.psf(self.x_grid, self.y_grid, x_sources, y_sources)  # (N, H, W)
        model_image = np.sum(psf_vals * fluxes[:, None, None], axis=0)  # (H, W)
        
        return model_image
    
    def log_likelihood(self, x_sources: np.ndarray, y_sources: np.ndarray, 
                      fluxes: np.ndarray) -> float:
        """
        Compute log likelihood using proper variance weighting
        """
        model_image = self.forward(x_sources, y_sources, fluxes)
        
        # Use variance map for proper weighting
        residual = self.image - model_image
        
        # Only include pixels with finite variance
        mask = np.isfinite(self.variance) & (self.variance > 0)
        if np.sum(mask) == 0:
            return -np.inf
            
        chi_sq = np.sum(residual[mask]**2 / self.variance[mask])
        log_like = -0.5 * chi_sq
        log_like -= 0.5 * np.sum(np.log(2 * np.pi * self.variance[mask]))
        
        return log_like


class MerianMCMCParameterization:
    """
    Handle variable number of sources for Merian data
    """
    
    def __init__(self, max_sources: int, image_shape: Tuple[int, int], psf_fwhm: float):
        """
        Initialize parameterization
        
        Parameters:
        -----------
        max_sources : int
            Maximum number of sources to consider
        image_shape : tuple
            (H, W) shape of the image
        psf_fwhm : float
            PSF FWHM for reasonable flux priors
        """
        self.max_sources = max_sources
        self.H, self.W = image_shape
        self.psf_fwhm = psf_fwhm
        
        # Parameter layout: [n_sources, x1, y1, flux1, x2, y2, flux2, ...]
        self.n_params = 1 + 3 * max_sources
        
        # Setup bounds
        self.bounds = self._setup_bounds()
    
    def _setup_bounds(self):
        """Setup parameter bounds"""
        bounds = []
        
        # Number of sources
        bounds.append((0, self.max_sources))
        
        # Source parameters - allow some buffer around image edges
        buffer = 2.0
        for i in range(self.max_sources):
            bounds.append((-buffer, self.W - 1 + buffer))  # x position
            bounds.append((-buffer, self.H - 1 + buffer))  # y position  
            bounds.append((1e-6, 100.0))                   # flux (broad range for emission lines)
        
        return np.array(bounds)
    
    def params_to_sources(self, params: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """Convert parameter vector to source arrays"""
        n_sources = int(np.round(np.clip(params[0], 0, self.max_sources)))
        
        if n_sources == 0:
            return 0, np.array([]), np.array([]), np.array([])
        
        # Extract active source parameters
        x_sources = params[1:1+3*n_sources:3]
        y_sources = params[2:2+3*n_sources:3] 
        fluxes = params[3:3+3*n_sources:3]
        
        # Ensure positive fluxes
        fluxes = np.abs(fluxes)
        
        return n_sources, x_sources, y_sources, fluxes
    
    def initialize_from_catalog(self, catalog: SourceCatalog, n_walkers: int,
                               use_catalog: bool = True) -> np.ndarray:
        """
        Initialize walker positions from source catalog
        
        Parameters:
        -----------
        catalog : SourceCatalog or None
            Detected sources from photutils
        n_walkers : int
            Number of MCMC walkers
        use_catalog : bool
            Whether to use catalog for initialization
        
        Returns:
        --------
        initial_params : np.ndarray, shape (n_walkers, n_params)
            Initial parameter values for walkers
        """
        initial_params = np.zeros((n_walkers, self.n_params))
        
        if use_catalog and catalog is not None and len(catalog) > 0:
            # Use detected sources for initialization
            n_detected = len(catalog)
            n_init = min(n_detected, self.max_sources)
            
            print(f"Initializing MCMC from {n_detected} detected sources")
            
            # Initialize number of sources around detected number
            initial_params[:, 0] = np.random.normal(n_init, 0.5, n_walkers)
            initial_params[:, 0] = np.clip(initial_params[:, 0], 0, self.max_sources)
            
            # Initialize source parameters with noise around detected sources
            for i in range(n_walkers):
                n_sources_walker = int(np.round(initial_params[i, 0]))
                n_sources_walker = max(0, min(n_sources_walker, self.max_sources))
                
                for j in range(n_sources_walker):
                    if j < n_detected:
                        # Use detected source with some jitter
                        jitter_x = np.random.normal(0, 1.0)
                        jitter_y = np.random.normal(0, 1.0) 
                        jitter_flux = np.exp(np.random.normal(0, 0.3))
                        
                        initial_params[i, 1 + j*3] = catalog.xcentroid[j] + jitter_x
                        initial_params[i, 2 + j*3] = catalog.ycentroid[j] + jitter_y
                        initial_params[i, 3 + j*3] = catalog.kron_flux[j] * jitter_flux
                    else:
                        # Random initialization for extra sources
                        initial_params[i, 1 + j*3] = np.random.uniform(0, self.W - 1)
                        initial_params[i, 2 + j*3] = np.random.uniform(0, self.H - 1)
                        initial_params[i, 3 + j*3] = np.random.gamma(2.0, 0.1)
                
                # Fill unused parameters with random values to maintain diversity
                for j in range(n_sources_walker, self.max_sources):
                    initial_params[i, 1 + j*3] = np.random.uniform(0, self.W - 1)
                    initial_params[i, 2 + j*3] = np.random.uniform(0, self.H - 1)
                    initial_params[i, 3 + j*3] = np.random.gamma(1.0, 0.05)
        else:
            # Random initialization
            print("Using random MCMC initialization")
            self._random_init(initial_params)
        
        # Enforce bounds
        initial_params = self._enforce_bounds(initial_params)
        
        return initial_params
    
    def _random_init(self, initial_params: np.ndarray):
        """Random initialization"""
        n_walkers = initial_params.shape[0]
        
        # Random number of sources (favor lower numbers for emission lines)
        initial_params[:, 0] = np.random.poisson(1.5, n_walkers)
        
        # Random source parameters
        for i in range(n_walkers):
            for j in range(self.max_sources):
                initial_params[i, 1 + j*3] = np.random.uniform(0, self.W - 1)
                initial_params[i, 2 + j*3] = np.random.uniform(0, self.H - 1) 
                initial_params[i, 3 + j*3] = np.random.gamma(2.0, 0.1)
    
    def _enforce_bounds(self, params: np.ndarray) -> np.ndarray:
        """Enforce parameter bounds"""
        params_clipped = params.copy()
        
        for i, (low, high) in enumerate(self.bounds):
            params_clipped[:, i] = np.clip(params_clipped[:, i], low, high)
        
        return params_clipped


class MerianMCMCPosterior:
    """
    Compute log posterior probability for Merian data
    """
    
    def __init__(self, forward_model: MerianPointSourceModel, 
                 parameterization: MerianMCMCParameterization):
        """
        Initialize posterior
        
        Parameters:
        -----------
        forward_model : MerianPointSourceModel
            Forward model for likelihood computation
        parameterization : MerianMCMCParameterization
            Parameter handling
        """
        self.forward_model = forward_model
        self.parameterization = parameterization
        
        # Prior parameters optimized for emission line data
        self.poisson_lambda = 1.5  # Expect fewer sources in emission lines
        self.gamma_alpha = 2.0     # Shape parameter for flux prior
        self.gamma_beta = 5.0      # Rate parameter for flux prior
    
    def log_prior(self, params: np.ndarray) -> float:
        """Compute log prior probability"""
        n_sources, x_sources, y_sources, fluxes = self.parameterization.params_to_sources(params)
        
        # Check bounds
        if not self._within_bounds(params):
            return -np.inf
        
        log_prior = 0.0
        
        # Poisson prior on number of sources
        if n_sources == 0:
            log_prior += -self.poisson_lambda
        else:
            log_prior += (n_sources * np.log(self.poisson_lambda) - 
                         self.poisson_lambda - special.loggamma(n_sources + 1))
        
        if n_sources > 0:
            # Uniform priors on positions (implicitly handled by bounds)
            log_prior += -n_sources * np.log(self.parameterization.H * self.parameterization.W)
            
            # Gamma prior on fluxes (favoring moderate flux values)
            if np.any(fluxes <= 0):
                return -np.inf
            log_prior += np.sum((self.gamma_alpha - 1) * np.log(fluxes) - self.gamma_beta * fluxes)
        
        return log_prior
    
    def log_likelihood(self, params: np.ndarray) -> float:
        """Compute log likelihood"""
        n_sources, x_sources, y_sources, fluxes = self.parameterization.params_to_sources(params)
        return self.forward_model.log_likelihood(x_sources, y_sources, fluxes)
    
    def log_posterior(self, params: np.ndarray) -> float:
        """Compute log posterior probability"""
        log_prior = self.log_prior(params)
        if not np.isfinite(log_prior):
            return -np.inf
        
        log_like = self.log_likelihood(params)
        if not np.isfinite(log_like):
            return -np.inf
        
        return log_prior + log_like
    
    def _within_bounds(self, params: np.ndarray) -> bool:
        """Check if parameters are within bounds"""
        bounds = self.parameterization.bounds
        return np.all((params >= bounds[:, 0]) & (params <= bounds[:, 1]))


def run_merian_mcmc(image: np.ndarray, variance: np.ndarray, catalog: SourceCatalog,
                    psf_fwhm: float, max_sources: int = 5, n_walkers: int = 50, 
                    n_steps: int = 2000, burn_in: int = 500) -> Tuple[emcee.EnsembleSampler, MerianMCMCParameterization, MerianPointSourceModel]:
    """
    Run MCMC sampling for Merian emission line data
    
    Parameters:
    -----------
    image : np.ndarray
        Observed emission line image
    variance : np.ndarray  
        Variance map
    catalog : SourceCatalog
        Detected sources from photutils
    psf_fwhm : float
        PSF FWHM in pixels
    max_sources : int
        Maximum number of sources to fit
    n_walkers : int
        Number of MCMC walkers
    n_steps : int
        Number of production steps
    burn_in : int
        Number of burn-in steps
    
    Returns:
    --------
    sampler : emcee.EnsembleSampler
        MCMC sampler with chains
    parameterization : MerianMCMCParameterization
        Parameter handling object
    forward_model : MerianPointSourceModel
        Forward model object
    """
    print(f"Running MCMC for emission line modeling...")
    print(f"Image shape: {image.shape}")
    print(f"PSF FWHM: {psf_fwhm:.2f} pixels")
    print(f"Max sources: {max_sources}")
    
    # Setup PSF model with gamma derived from FWHM
    # FWHM = 2 * gamma * sqrt(2^(1/alpha) - 1) for Moffat
    # For alpha=2.5: gamma = FWHM / (2 * sqrt(2^(1/2.5) - 1))
    gamma = psf_fwhm / (2.0 * np.sqrt(2**(1/2.5) - 1))
    psf = MoffatPSF(gamma=gamma, alpha=2.5)
    
    # Setup components
    forward_model = MerianPointSourceModel(psf, image, variance)
    parameterization = MerianMCMCParameterization(max_sources, image.shape, psf_fwhm)
    posterior = MerianMCMCPosterior(forward_model, parameterization)
    
    # Initialize walkers
    print(f"Initializing {n_walkers} walkers...")
    initial_positions = parameterization.initialize_from_catalog(catalog, n_walkers)
    
    # Setup sampler
    sampler = emcee.EnsembleSampler(
        n_walkers,
        parameterization.n_params,
        posterior.log_posterior
    )
    
    print(f"Running burn-in: {burn_in} steps...")
    state = sampler.run_mcmc(initial_positions, burn_in, progress=True)
    sampler.reset()
    
    print(f"Running production: {n_steps} steps...")
    sampler.run_mcmc(state, n_steps, progress=True)
    
    print(f"MCMC complete! Acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
    
    return sampler, parameterization, forward_model


def get_mle_model(sampler: emcee.EnsembleSampler, parameterization: MerianMCMCParameterization,
                  forward_model: MerianPointSourceModel) -> Tuple[np.ndarray, dict]:
    """
    Get maximum likelihood estimate model
    
    Parameters:
    -----------
    sampler : emcee.EnsembleSampler
        Completed MCMC sampler
    parameterization : MerianMCMCParameterization
        Parameter handling object  
    forward_model : MerianPointSourceModel
        Forward model object
    
    Returns:
    --------
    mle_model : np.ndarray
        Maximum likelihood model image
    mle_params : dict
        MLE parameters and metadata
    """
    # Get all samples and log probabilities
    samples = sampler.get_chain(flat=True)
    log_probs = sampler.get_log_prob(flat=True)
    
    # Find maximum likelihood sample
    mle_idx = np.argmax(log_probs)
    mle_sample = samples[mle_idx]
    mle_logprob = log_probs[mle_idx]
    
    # Extract source parameters
    n_sources, x_sources, y_sources, fluxes = parameterization.params_to_sources(mle_sample)
    
    # Generate MLE model
    mle_model = forward_model.forward(x_sources, y_sources, fluxes)
    
    # Package results
    mle_params = {
        'n_sources': n_sources,
        'x_sources': x_sources,
        'y_sources': y_sources, 
        'fluxes': fluxes,
        'log_prob': mle_logprob,
        'mle_sample': mle_sample
    }
    
    print(f"MLE model: {n_sources} sources, log_prob={mle_logprob:.2f}")
    if n_sources > 0:
        print(f"Source positions: {list(zip(x_sources, y_sources))}")
        print(f"Source fluxes: {fluxes}")
    
    return mle_model, mle_params


def save_merian_mcmc_results(sampler: emcee.EnsembleSampler, parameterization: MerianMCMCParameterization,
                            mle_params: dict, band: str, target: str, output_dir: str):
    """
    Save MCMC results to disk
    
    Parameters:
    -----------
    sampler : emcee.EnsembleSampler
        Completed MCMC sampler
    parameterization : MerianMCMCParameterization
        Parameter handling object
    mle_params : dict
        MLE parameter dictionary
    band : str
        Band name (e.g., 'n540', 'n708')
    target : str
        Target name
    output_dir : str
        Output directory
    """
    # Save MCMC chain and results
    chain = sampler.get_chain()
    log_prob = sampler.get_log_prob()
    
    output_file = os.path.join(output_dir, f"{target}_{band}_mcmc_results.npz")
    
    np.savez(
        output_file,
        chain=chain,
        log_prob=log_prob,
        acceptance_fraction=sampler.acceptance_fraction,
        max_sources=parameterization.max_sources,
        image_shape=(parameterization.H, parameterization.W),
        mle_n_sources=mle_params['n_sources'],
        mle_x_sources=mle_params['x_sources'],
        mle_y_sources=mle_params['y_sources'], 
        mle_fluxes=mle_params['fluxes'],
        mle_log_prob=mle_params['log_prob'],
        band=band,
        target=target
    )
    
    print(f"MCMC results saved to: {output_file}")


# Keep the same target processing structure as do_region_photometry.py
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


def process_target_mcmc(target, catalog, dirname, output_dir, emission_corrections):
    """
    Process a single target with MCMC modeling instead of traditional fitting
    
    This replaces the modeling section (~L180) from do_region_photometry.py
    """
    # Get target information
    targetid = conventions.merianobjectname_to_catalogname(target, catalog)
    if targetid not in catalog.index:
        print(f"Target {targetid} not found in catalog, skipping...")
        return   
    print(f"Processing target with MCMC: {targetid}") 
    
    # Create output directory for this target
    target_output_dir = os.path.join(output_dir, targetid)
    os.makedirs(target_output_dir, exist_ok=True)
    
    try:
        targetindex = np.where(np.in1d(catalog.index, targetid))[0][0]
        
        # Load images (same as do_region_photometry.py)
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
                size=100,
                image=cutout,
                var=cutout,
                psf=psf,
                image_ext=1,
                var_ext=3,
                psf_ext=0
            )
        
        # Compute excess images (same as do_region_photometry.py)
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
        
        # Process emission lines with MCMC instead of traditional fitting
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
            
            # REPLACE TRADITIONAL FITTING WITH MCMC
            # This replaces: model_obj, model_pred = fit.fit_multi_moffat_2d(...)
            print(f"Running MCMC modeling for {band} band...")
            
            # Run MCMC modeling
            sampler, parameterization, forward_model = run_merian_mcmc(
                excess_bbmb.image[band],
                excess_bbmb.var[band], 
                cat,
                bbmb.fwhm_to_match,
                max_sources=min(len(cat) + 2, 5),  # Reasonable max based on detections
                n_walkers=32,
                n_steps=1000,
                burn_in=200
            )
            
            # Get MLE model (this is the "fiducial" model)
            model_pred, mle_params = get_mle_model(sampler, parameterization, forward_model)
            
            # Save MCMC results
            save_merian_mcmc_results(sampler, parameterization, mle_params, 
                                   band, targetid, target_output_dir)
            
            # Create model object compatible with existing output format
            # Store MLE parameters in a format similar to fit.fit_multi_moffat_2d output
            model_obj = {
                'mle_params': mle_params,
                'sampler': sampler,
                'parameterization': parameterization,
                'forward_model': forward_model,
                'band': band,
                'mcmc_results': True  # Flag to indicate this is MCMC not traditional fit
            }
            
            # Compute integrated flux from catalog as before
            integrated_flux = emission.excess_to_lineflux(cat.kron_flux*conversion*u.nJy, band)
            integrated_flux_corrected = integrated_flux * extinction_correction[0][targetindex, correction_indices[band]]
            integrated_flux_corrected *= ge_correction[targetindex, correction_indices[band]]
            if band == 'n708':
                integrated_flux_corrected *= ancline_correction[targetindex]        
            
            emission_bundle[band] = (cat, segm_deblend, total_flux, integrated_flux_corrected, model_obj, model_pred)
        
        # Create and save figures (same structure as do_region_photometry.py)
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
        
        # Model displays
        for adx, model_band in enumerate(['n708','n540']):            
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

        ek.text(0.025, 0.975, r'N708 (H$\alpha$) - MCMC Model', ax=axarr[0,0], color='w', size=13)
        ek.text(0.025, 0.975, r'N540 ([OIII]5007) - MCMC Model', ax=axarr[1,0], color='w', size=13)        
        axarr[0,0].set_title('RGB')
        axarr[0,1].set_title('Excess')
        axarr[0,2].set_title('MCMC MLE Model')
        axarr[0,3].set_title('Residual')
        
        plt.tight_layout()
        plt.savefig(os.path.join(target_output_dir, f"{targetid}_mcmc_model.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save photometry catalogs with MCMC model fluxes
        for band in ['n540', 'n708']:
            bundle_data = emission_bundle.get(band, [None, None, np.nan*u.erg/u.s/u.cm**2, np.nan, None, None])
            cat, _, total_flux, integrated_flux_corrected, model_obj, _ = bundle_data
            
            if cat is not None and model_obj is not None:
                # Extract MCMC model fluxes
                mle_params = model_obj['mle_params']
                n_sources = mle_params['n_sources']
                x_sources = mle_params['x_sources']
                y_sources = mle_params['y_sources']
                model_fluxes = mle_params['fluxes']
                
                # Create photometry table
                phot_table = cat.to_table()
                phot_table['lineflux_corrected'] = integrated_flux_corrected                
                phot_table['band'] = band
                phot_table['target'] = target
                
                # Add MCMC model results
                mcmc_model_fluxes = []
                mcmc_model_x = []
                mcmc_model_y = []
                
                # Match MCMC sources to detected sources by proximity
                for i in range(len(cat)):
                    cat_x, cat_y = cat.xcentroid[i], cat.ycentroid[i]
                    
                    if n_sources > 0:
                        # Find closest MCMC source
                        distances = np.sqrt((x_sources - cat_x)**2 + (y_sources - cat_y)**2)
                        closest_idx = np.argmin(distances)
                        
                        if distances[closest_idx] < 3.0:  # Within 3 pixels
                            mcmc_flux = emission.excess_to_lineflux(model_fluxes[closest_idx]*conversion*u.nJy, band)
                            mcmc_flux = mcmc_flux * extinction_correction[0][targetindex, correction_indices[band]]
                            mcmc_flux *= ge_correction[targetindex, correction_indices[band]]
                            if band == 'n708':
                                mcmc_flux *= ancline_correction[targetindex]
                                
                            mcmc_model_fluxes.append(mcmc_flux)
                            mcmc_model_x.append(x_sources[closest_idx])
                            mcmc_model_y.append(y_sources[closest_idx])
                        else:
                            mcmc_model_fluxes.append(0.)
                            mcmc_model_x.append(np.nan)
                            mcmc_model_y.append(np.nan)
                    else:
                        mcmc_model_fluxes.append(0.)
                        mcmc_model_x.append(np.nan) 
                        mcmc_model_y.append(np.nan)
                
                phot_table['mcmc_model_lineflux'] = mcmc_model_fluxes
                phot_table['mcmc_model_x'] = mcmc_model_x
                phot_table['mcmc_model_y'] = mcmc_model_y
                phot_table['mcmc_n_sources'] = n_sources
                
                # Add integrated rows
                integrated_row = {
                    'label': -1,
                    'lineflux_corrected': total_flux,                    
                    'band': band,
                    'target': target,
                    'mcmc_n_sources': n_sources
                }
                phot_table.add_row(integrated_row)
                
                integrated_row = {
                    'label': -2,
                    'kron_flux': total_bbflux,
                    'band': 'r',
                    'target': target,
                    'mcmc_n_sources': n_sources
                }                
                phot_table.add_row(integrated_row)
                
                # Save to CSV
                phot_table.write(
                    os.path.join(target_output_dir, f"{targetid}_{band}_mcmc_region_photometry.csv"),
                    format='csv',
                    overwrite=True
                )
            else:
                # No sources detected - save basic integrated photometry
                if not np.isnan(total_flux):
                    from astropy.table import Table
                    
                    phot_table = Table()
                    phot_table['label'] = [-1, -2]
                    phot_table['lineflux_corrected'] = [total_flux, total_flux]
                    phot_table['band'] = [band, band]
                    phot_table['target'] = [target, target]
                    phot_table['mcmc_n_sources'] = [0, 0]
                    
                    phot_table.write(
                        os.path.join(target_output_dir, f"{targetid}_{band}_mcmc_region_photometry.csv"),
                        format='csv',
                        overwrite=True
                    )
        
        print(f"Successfully processed {targetid} with MCMC modeling")
        
    except Exception as e:
        print(f"Error processing target {targetid}: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function - same structure as do_region_photometry.py but with MCMC"""
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
    output_dir = '../local_data/pieridae_output/MDR1_mcmasses_mcmc/'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get available targets
    targets = get_available_targets(dirname)
    print(f"Found {len(targets)} targets to process with MCMC")
    
    # Process each target with MCMC
    for i, target in enumerate(targets):
        print(f"Processing target {i+1}/{len(targets)}: {target}")
        process_target_mcmc(target, catalog, dirname, output_dir, emission_corrections)
    
    print("MCMC processing complete!")


if __name__ == "__main__":
    main()