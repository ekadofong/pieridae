#!/usr/bin/env python3
"""
MCMC Point Source Detection Algorithm using emcee
Provides direct comparison to VI approach with same features:
- Optimized priors (Poisson λ=2.0, Gamma(3.0,2.0))
- Peak detection initialization (optional)
- Comprehensive diagnostics and QA figures
- Model persistence capabilities
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
from scipy.ndimage import maximum_filter
from scipy import special
import emcee
import corner
from ekfstats import sampling

# Reuse device setup from VI version
prefer_cpu = True

if torch.backends.mps.is_available() and not prefer_cpu:
    device = torch.device("mps")
    print("Using M1 GPU (MPS)")
elif torch.cuda.is_available() and not prefer_cpu:
    device = torch.device("cuda")
    print("Using NVIDIA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

class PSF:
    """Moffat PSF model (NumPy implementation for MCMC)"""
    def __init__(self, gamma: float = 2.0, alpha: float = 2.5):
        self.gamma = gamma
        self.alpha = alpha
        self.normalization = np.pi * (gamma**2) / (alpha - 1)
    
    def __call__(self, x_grid: np.ndarray, y_grid: np.ndarray, 
                 x_center: np.ndarray, y_center: np.ndarray) -> np.ndarray:
        """
        Evaluate PSF at grid positions for given source centers
        
        Args:
            x_grid, y_grid: (H, W) coordinate grids
            x_center, y_center: (N_sources,) source centers
        
        Returns:
            psf_vals: (N_sources, H, W) PSF values
        """
        if len(x_center) == 0:
            return np.zeros((0, x_grid.shape[0], x_grid.shape[1]))
        
        # Broadcast for vectorized computation
        x_center = x_center[:, None, None]  # (N_sources, 1, 1)
        y_center = y_center[:, None, None]  # (N_sources, 1, 1)
        
        r_sq = (x_grid[None, :, :] - x_center)**2 + (y_grid[None, :, :] - y_center)**2
        psf = np.power(1 + r_sq / (self.gamma**2), -self.alpha)
        
        return psf / self.normalization

def detect_peaks(image: np.ndarray, psf: PSF, min_distance: int = 3, threshold_rel: float = 10.,
                bkg_std: float = 0.01, ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect peaks and estimate flux (NumPy version)"""
    local_max = maximum_filter(image, size=min_distance) == image
    threshold = threshold_rel * bkg_std
    peaks_mask = local_max & (image > threshold)
    
    y_coords, x_coords = np.where(peaks_mask)
    
    if len(x_coords) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Flux estimation using PSF normalization
    peak_fluxes = image[y_coords, x_coords]
    flux_estimates = peak_fluxes * np.pi * psf.gamma**2 / (psf.alpha - 1.)
    sort_idx = np.argsort(flux_estimates)[::-1]
    
    x_peaks = x_coords[sort_idx].astype(np.float32)
    y_peaks = y_coords[sort_idx].astype(np.float32) 
    flux_peaks = flux_estimates[sort_idx].astype(np.float32)
    
    return x_peaks, y_peaks, flux_peaks

class PointSourceModel:
    """Forward model for point sources (NumPy implementation)"""
    
    def __init__(self, psf: PSF, noise_std: float = 0.01, image_size: Tuple[int, int] = (32, 32)):
        self.psf = psf
        self.noise_std = noise_std
        self.H, self.W = image_size
        
        # Create coordinate grids
        y, x = np.meshgrid(np.arange(self.H, dtype=np.float32), 
                          np.arange(self.W, dtype=np.float32), 
                          indexing='ij')
        self.x_grid = x
        self.y_grid = y
    
    def forward(self, x_sources: np.ndarray, y_sources: np.ndarray, 
                fluxes: np.ndarray) -> np.ndarray:
        """Generate image from point source parameters"""
        if len(x_sources) == 0:
            return np.zeros((self.H, self.W))
        
        psf_vals = self.psf(self.x_grid, self.y_grid, x_sources, y_sources)  # (N, H, W)
        image = np.sum(psf_vals * fluxes[:, None, None], axis=0)  # (H, W)
        
        return image
    
    def log_likelihood(self, observed_image: np.ndarray, 
                      x_sources: np.ndarray, y_sources: np.ndarray, 
                      fluxes: np.ndarray) -> float:
        """Compute log likelihood"""
        model_image = self.forward(x_sources, y_sources, fluxes)
        
        residual = observed_image - model_image
        log_like = -0.5 * np.sum(residual**2) / (self.noise_std**2)
        log_like -= 0.5 * np.log(2 * np.pi * self.noise_std**2) * observed_image.size
        
        if not np.isfinite(log_like).all():
            errmessage = f'lnL is not finite!\n'
            for idx in range(len(x_sources)):
                errmessage = errmessage + f'Source {idx}: {x_sources[idx]:.1f} {y_sources[idx]:.1f} {fluxes[idx]:.2e}\n'
            raise ValueError(errmessage)
        return log_like

class MCMCParameterization:
    """Handle variable number of sources in fixed parameter space"""
    
    def __init__(self, max_sources: int, image_size: Tuple[int, int]):
        self.max_sources = max_sources
        self.H, self.W = image_size
        
        # Parameter layout: [n_sources, x1, y1, log10_flux1, x2, y2, log10_flux2, ...]
        self.n_params = 1 + 3 * max_sources  # n_sources + (x, y, flux) * max_sources
        
        # Parameter bounds
        self.bounds = self._setup_bounds()
    
    def _setup_bounds(self):
        """Setup parameter bounds for sampling (now only for n_sources)"""
        bounds = []
        
        # Number of sources: [0, max_sources]
        bounds.append((0, self.max_sources))
        
        # Source parameters - no hard bounds, will use soft priors
        for i in range(self.max_sources):
            bounds.append((0, self.W))    # x position (unbounded)
            bounds.append((0, self.H))    # y position (unbounded)
            bounds.append((-1, 3.))    # log10_flux (unbounded)
        
        return np.array(bounds)
    
    def params_to_sources(self, params: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """Convert parameter vector to source arrays"""
        n_sources = int(np.round(params[0]))
        n_sources = max(0, min(n_sources, self.max_sources))  # Clamp
        
        if n_sources == 0:
            return 0, np.array([]), np.array([]), np.array([])
        
        # Extract active source parameters
        x_sources = params[1:1+3*n_sources:3]
        y_sources = params[2:2+3*n_sources:3] 
        log10_fluxes = params[3:3+3*n_sources:3]
        
        # Convert log10_flux back to flux
        fluxes = 10**log10_fluxes
        
        return n_sources, x_sources, y_sources, fluxes
    
    def initialize_from_peaks(self, 
                              observed_image: np.ndarray, 
                              psf: PSF,
                              n_walkers: int, 
                              bkg_std: float,
                              find_peaks: bool = True) -> np.ndarray:
        """Initialize walker positions"""
        initial_params = np.zeros((n_walkers, self.n_params))
        
        if find_peaks:
            # Detect peaks
            x_peaks, y_peaks, flux_peaks = detect_peaks(observed_image, psf, bkg_std=bkg_std)
            n_detected = len(x_peaks)
            jitter = 10
            print(flux_peaks)
            if n_detected > 0:
                print(f"Detected {n_detected} peaks for MCMC initialization")
                
                # Initialize around detected number of sources
                n_init = min(n_detected, self.max_sources)
                initial_params[:, 0] = np.clip(np.random.normal(n_init, 0.5, n_walkers), 1., self.max_sources)
                initial_params
                
                # Initialize source parameters with noise around detected peaks
                for i in range(n_walkers):
                    n_sources_walker = int(np.clip(np.round(initial_params[i, 0]), 0, self.max_sources))
                    
                    for j in range(n_sources_walker):
                        if j < n_detected:
                            # Use detected peak with substantial noise for diversity
                            initial_params[i, 1 + j*3] = x_peaks[j] + np.random.normal(0, jitter)  # x
                            initial_params[i, 2 + j*3] = y_peaks[j] + np.random.normal(0, jitter)  # y
                            initial_params[i, 3 + j*3] = np.log10(np.maximum(flux_peaks[j], 1e-6)) + np.random.normal(0, 0.5)  # log10_flux                                                           
                        else:
                            # Random initialization for extra sources
                            initial_params[i, 1 + j*3] = np.random.uniform(jitter, self.W - jitter)  # x
                            initial_params[i, 2 + j*3] = np.random.uniform(jitter, self.H - jitter)  # y  
                            initial_params[i, 3 + j*3] = np.log10(np.mean(flux_peaks)) + np.random.normal(0, 0.5)  # log10_flux

                    # Add noise to unused parameters to ensure diversity
                    for j in range(n_sources_walker, self.max_sources):
                        initial_params[i, 1 + j*3] = np.random.uniform(0, self.W - 1)  # x
                        initial_params[i, 2 + j*3] = np.random.uniform(0, self.H - 1)  # y
                        initial_params[i, 3 + j*3] = np.random.normal(0, 1.0)  # log10_flux
                
                print(f"Initialized MCMC with {n_init} sources from peaks")
            else:
                print("No peaks detected, using random MCMC initialization")
                self._random_init(initial_params)
        else:
            print("Using random MCMC initialization")
            self._random_init(initial_params)
        
        # Ensure bounds are respected
        initial_params = self._enforce_bounds(initial_params)
        
        return initial_params
    
    def _random_init(self, initial_params: np.ndarray):
        """Random initialization when no peaks found"""
        n_walkers = initial_params.shape[0]
        
        # Random number of sources (favor lower numbers)
        initial_params[:, 0] = np.random.poisson(2.0, n_walkers)  # Use optimized prior
        
        # Random source parameters for ALL parameters (not just active ones)
        for i in range(n_walkers):
            for j in range(self.max_sources):
                initial_params[i, 1 + j*3] = np.random.uniform(0, self.W - 1)  # x
                initial_params[i, 2 + j*3] = np.random.uniform(0, self.H - 1)  # y
                initial_params[i, 3 + j*3] = np.random.normal(0, 3.)  # log10_flux
    
    def _enforce_bounds(self, params: np.ndarray) -> np.ndarray:
        """Enforce parameter bounds (only n_sources now)"""
        params_clipped = params.copy()
        
        # Only enforce n_sources bounds
        params_clipped[:, 0] = np.clip(params_clipped[:, 0], 0, self.max_sources)
        
        return params_clipped

class MCMCPosterior:
    """Compute log posterior probability"""
    
    def __init__(self, forward_model: PointSourceModel, parameterization: MCMCParameterization,
                 observed_image: np.ndarray):
        self.forward_model = forward_model
        self.parameterization = parameterization
        self.observed_image = observed_image
        
        # Optimized prior parameters (from VI analysis)
        self.poisson_lambda = 4.0
        self.gamma_alpha = 3.0
        self.gamma_beta = 2.0
    
    def log_prior(self, params: np.ndarray) -> float:
        """Compute log prior probability with soft priors"""        
        n_sources, x_sources, y_sources, fluxes = self.parameterization.params_to_sources(params)
                
        #if not self._within_bounds(params):
        #    return -1000.
        
        log_prior = 0.0
    
        # Poisson prior on number of sources
        if n_sources == 0:
            log_prior += -self.poisson_lambda
        else:
            log_prior += (n_sources * np.log(self.poisson_lambda) - 
                        self.poisson_lambda - special.loggamma(n_sources + 1))
        
        if n_sources > 0:
            # Sigmoid priors on positions - sharp falloff near image boundaries
            def sigmoid_prior(a, min_a, max_a, edge_width=.1):
                """
                Sigmoid-based position prior with sharp boundaries
                
                Returns log prior that is:
                - ~0 for sources well inside image (minimal penalty)  
                - Rapidly decreasing as sources approach edges
                - Strongly negative for sources outside image
                """
                # Sigmoid transitions: penalty increases sharply within edge_width pixels of boundaries
                left_penalty = -np.log(1.0 + np.exp(-(a - min_a) / edge_width))
                right_penalty = -np.log(1.0 + np.exp(-(max_a - a) / edge_width))
                return left_penalty + right_penalty            
            
            # Apply sigmoid priors to positions
            x_prior = np.sum([sigmoid_prior(x, 0., self.parameterization.W, edge_width=2.0) for x in x_sources])
            y_prior = np.sum([sigmoid_prior(y, 0, self.parameterization.H, edge_width=2.0) for y in y_sources])
            log_prior += x_prior + y_prior
            
            # Prior on log10_flux - extract from params directly
            log10_fluxes = params[3:3+3*n_sources:3]
            # Wide normal prior on log10_flux: mean=1 (flux=1), std=2.5 (covers broad astronomical range)
            #log_prior += np.sum(-0.5 * (log10_fluxes)**2)
            #log_prior += -n_sources * np.log(2.5 * np.sqrt(2 * np.pi))
            log_prior += np.sum([sigmoid_prior(lg10flux, -1, 2.5, 0.1) for lg10flux in log10_fluxes])


        return log_prior

    def log_likelihood(self, params: np.ndarray) -> float:
        """Compute log likelihood with safety checks"""        
        n_sources, x_sources, y_sources, fluxes = self.parameterization.params_to_sources(params)
        
        # Check for invalid parameter values
        if n_sources > 0:
            if not (np.all(np.isfinite(x_sources)) and 
                    np.all(np.isfinite(y_sources)) and 
                    np.all(np.isfinite(fluxes)) and
                    np.all(fluxes > 0)):
                return -np.inf
        
        log_like = self.forward_model.log_likelihood(self.observed_image, x_sources, y_sources, fluxes)
        
        # Check result is finite
        if not np.isfinite(log_like):
            return -np.inf
            
        return log_like

    
    def log_posterior(self, params: np.ndarray) -> float:
        """Compute log posterior probability with safety checks"""
        log_prior = self.log_prior(params)
        
        # If prior is invalid, don't compute likelihood
        if not np.isfinite(log_prior):            
            return -np.inf
        
        log_like = self.log_likelihood(params)
        
        # Check final result
        log_post = log_prior + log_like
        if not np.isfinite(log_post):
            return -np.inf
            
        return log_post
    
    def _within_bounds(self, params: np.ndarray) -> bool:
        """Check if parameters are within bounds (only n_sources now)"""
        # Only check n_sources bounds - positions and log10_flux are unbounded with soft priors
        return ((params > self.parameterization.bounds[:,0])&(params < self.parameterization.bounds[:,1])).all()

def run_mcmc_sampling(observed_image: np.ndarray, psf: PSF, max_sources: int = 10, n_walkers: int = 100, 
                     n_steps: int = 5000, find_peaks: bool = True, 
                     noise_std: float = 0.01,
                     burn_in: int = 1000,
                     ) -> Tuple[emcee.EnsembleSampler, MCMCParameterization, PointSourceModel]:
    """Run MCMC sampling for point source detection
    
    Parameters:
    -----------
    observed_image : np.ndarray
        The observed image data
    psf : PSF
        PSF model object to use for forward modeling
    max_sources : int, optional
        Maximum number of sources to fit, by default 10
    n_walkers : int, optional
        Number of MCMC walkers, by default 100
    n_steps : int, optional  
        Number of production steps, by default 5000
    find_peaks : bool, optional
        Whether to initialize from detected peaks, by default True
    noise_std : float, optional
        Noise standard deviation, by default 0.01
    burn_in : int, optional
        Number of burn-in steps, by default 1000
        
    Returns:
    --------
    sampler : emcee.EnsembleSampler
        MCMC sampler with chains
    parameterization : MCMCParameterization  
        Parameter handling object
    forward_model : PointSourceModel
        Forward model object
    """
    
    print("=== MCMC POINT SOURCE DETECTION ===")
    
    # Convert torch tensor to numpy if needed
    if torch.is_tensor(observed_image):
        observed_image = observed_image.cpu().numpy()
    
    image_size = observed_image.shape
    
    # Setup components using provided PSF
    forward_model = PointSourceModel(psf, noise_std=noise_std, image_size=image_size)
    parameterization = MCMCParameterization(max_sources, image_size)
    posterior = MCMCPosterior(forward_model, parameterization, observed_image)
    
    # Initialize walkers
    print(f"Initializing {n_walkers} walkers...")
    initial_positions = parameterization.initialize_from_peaks(observed_image, psf,n_walkers, noise_std, find_peaks)
    
    # Setup sampler
    sampler = emcee.EnsembleSampler(
        n_walkers, 
        parameterization.n_params, 
        posterior.log_posterior,
        args=()
    )
    
    print(f"Running MCMC sampling: {n_steps} steps, {burn_in} burn-in...")
    
    # Run burn-in
    print("Running burn-in...")
    state = sampler.run_mcmc(initial_positions, burn_in, progress=True)
    sampler.reset()
    
    # Production run
    print("Running production chains...")
    sampler.run_mcmc(state, n_steps, progress=True)
    
    print(f"MCMC sampling complete!")
    print(f"Acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
    
    return sampler, parameterization, forward_model

def analyze_convergence(sampler: emcee.EnsembleSampler, parameterization: MCMCParameterization,
                       burn_in_steps: int = 1000) -> dict:
    """Analyze MCMC convergence"""
    print(f"\n=== CONVERGENCE ANALYSIS ===")
    
    # Autocorrelation time
    try:
        tau = sampler.get_autocorr_time()
        print(f"Autocorrelation time: {np.mean(tau):.1f} ± {np.std(tau):.1f}")
    except Exception as e:
        print(f"Could not compute autocorrelation time: {e}")
        tau = np.full(parameterization.n_params, np.nan)
    
    # Effective sample size
    n_samples = sampler.chain.shape[1] * sampler.chain.shape[0]
    eff_samples = n_samples / (2 * np.nanmax(tau)) if not np.isnan(tau).all() else n_samples / 10
    print(f"Effective sample size: ~{eff_samples:.0f}")
    
    # Gelman-Rubin diagnostic (simplified)
    chains = sampler.chain  # (n_walkers, n_steps, n_params)
    n_walkers, n_steps, n_params = chains.shape
    
    if n_walkers >= 4 and n_steps >= 100:
        # Split each chain in half and treat as separate chains
        split_chains = chains[:, n_steps//2:, :].reshape(-1, n_steps//2, n_params)
        
        # Compute R-hat for number of sources parameter
        r_hat_n_sources = gelman_rubin(split_chains[:, :, 0])
        print(f"R-hat (n_sources): {r_hat_n_sources:.3f}")
    else:
        r_hat_n_sources = np.nan
    
    convergence_info = {
        'autocorr_time': tau,
        'eff_sample_size': eff_samples,
        'acceptance_fraction': sampler.acceptance_fraction,
        'r_hat_n_sources': r_hat_n_sources
    }
    
    return convergence_info

def gelman_rubin(chains: np.ndarray) -> float:
    """Compute Gelman-Rubin R-hat statistic"""
    n_chains, n_steps = chains.shape[:2]
    
    # Chain means
    chain_means = np.mean(chains, axis=1)
    
    # Overall mean
    overall_mean = np.mean(chain_means)
    
    # Between-chain variance
    B = n_steps * np.var(chain_means, ddof=1)
    
    # Within-chain variance
    W = np.mean(np.var(chains, axis=1, ddof=1))
    
    # Marginal posterior variance estimate
    var_plus = ((n_steps - 1) * W + B) / n_steps
    
    # R-hat
    r_hat = np.sqrt(var_plus / W)
    
    return r_hat

def evaluate_mcmc_performance(sampler: emcee.EnsembleSampler, parameterization: MCMCParameterization,
                             forward_model: PointSourceModel, observed_image: np.ndarray,
                             true_likelihood: float, n_eval_samples: int = 1000) -> Tuple[List[float], dict]:
    """Evaluate MCMC performance"""
    print(f"\n=== MCMC PERFORMANCE EVALUATION ===")
    
    # Get samples (flattened)
    samples = sampler.get_chain(flat=True)
    n_total_samples = samples.shape[0]
    
    # Randomly select evaluation samples
    eval_indices = np.random.choice(n_total_samples, min(n_eval_samples, n_total_samples), replace=False)
    eval_samples = samples[eval_indices]
    
    performances = []
    best_likelihood = -float('inf')
    best_sample = None
    n_sources_counts = []
    
    for i, params in enumerate(eval_samples):
        n_sources, x_sources, y_sources, fluxes = parameterization.params_to_sources(params)
        likelihood = forward_model.log_likelihood(observed_image, x_sources, y_sources, fluxes)
        
        performance = likelihood / true_likelihood
        performances.append(performance)
        n_sources_counts.append(n_sources)
        
        if likelihood > best_likelihood:
            best_likelihood = likelihood
            best_sample = (n_sources, x_sources, y_sources, fluxes)
    
    mean_perf = np.mean(performances)
    std_perf = np.std(performances)
    best_perf = np.max(performances)
    
    print(f"Mean performance: {mean_perf:.3f} ± {std_perf:.3f}")
    print(f"Best performance: {best_perf:.3f}")
    print(f"Ground truth: {true_likelihood:.1f}")
    print(f"Best achieved: {best_likelihood:.1f}")
    
    # Analyze number of sources
    unique_n, counts = np.unique(n_sources_counts, return_counts=True)
    print(f"\nNumber of sources distribution:")
    for n, count in zip(unique_n, counts):
        print(f"  {n} sources: {count/len(n_sources_counts):.3f} ({count}/{len(n_sources_counts)})")
    
    summary = {
        'mean_performance': mean_perf,
        'std_performance': std_perf,
        'best_performance': best_perf,
        'best_sample': best_sample,
        'n_sources_distribution': dict(zip(unique_n, counts/len(n_sources_counts)))
    }
    
    return performances, summary

def generate_mcmc_qa_figure(sampler: emcee.EnsembleSampler, parameterization: MCMCParameterization,
                           forward_model: PointSourceModel, observed_image: np.ndarray,
                           true_x: np.ndarray = None, true_y: np.ndarray = None, 
                           save_path: str = "mcmc_qa.png") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate QA figure for MCMC results"""
    print(f"\n=== GENERATING MCMC QA FIGURE ===")
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Panel 1: Observed image
    im = axes[0].imshow(observed_image, cmap='viridis')
    if true_x is not None:
        axes[0].scatter(true_x, true_y, c='red', s=50, marker='x', label='True sources')
    axes[0].set_title('Observed Image')
    axes[0].legend()
    
    # Get samples for reconstruction
    samples = sampler.get_chain(flat=True)
    n_eval_samples = min(1000, samples.shape[0])
    eval_indices = np.random.choice(samples.shape[0], n_eval_samples, replace=False)
    eval_samples = samples[eval_indices]
    
    # Find best likelihood sample
    best_likelihood = -float('inf')
    best_recon = None
    best_x, best_y, best_flux = None, None, None
    best_n_sources = 0
    
    for params in eval_samples:
        n_sources, x_sources, y_sources, fluxes = parameterization.params_to_sources(params)
        likelihood = forward_model.log_likelihood(observed_image, x_sources, y_sources, fluxes)
        
        if likelihood > best_likelihood:
            best_likelihood = likelihood
            best_recon = forward_model.forward(x_sources, y_sources, fluxes)
            best_x, best_y, best_flux = x_sources.copy(), y_sources.copy(), fluxes.copy()
            best_n_sources = n_sources
    
    print(f"Best MCMC sample: {best_n_sources} sources, likelihood={best_likelihood:.1f}")
    
    # Panel 2: Maximum likelihood reconstruction
    axes[1].imshow(best_recon, cmap='viridis', vmin=im.get_clim()[0], vmax=im.get_clim()[1])
    if true_x is not None:
        axes[1].scatter(true_x, true_y, c='red', s=50, marker='x', label='True sources')
    #if len(best_x) > 0:
    #    axes[1].scatter(best_x, best_y, c='lime', s=50, marker='+', label='Best MCMC sample', alpha=1.0)
    axes[1].set_title('Maximum Likelihood Reconstruction')
    axes[1].legend()
    
    # Panel 3: Residuals
    residual = observed_image - best_recon
    axes[2].imshow(residual, cmap='viridis', vmin=im.get_clim()[0], vmax=im.get_clim()[1])
    axes[2].set_title('Residual (Observed - Model)')
    
    # Panel 4: Number of sources trace
    n_sources_chain = []
    chain = sampler.get_chain()  # (n_steps, n_walkers, n_params)
    lnprob = sampler.get_log_prob () # (n_steps, n_walkers )
    
    #for step in range(chain.shape[0]):
    #    for walker in range(chain.shape[1]):
    #        n_sources, _, _, _ = parameterization.params_to_sources(chain[step, walker])
    #        n_sources_chain.append(n_sources)
    
    # Plot trace (subsample for visibility)
    #trace_subsample = n_sources_chain[::max(1, len(n_sources_chain)//1000)]
    #axes[3].plot(trace_subsample, alpha=0.7)
    for walker in range(chain.shape[1]):
        axes[3].plot(lnprob[:, walker], alpha=0.1, color='grey')
    
    axes[3].set_ylim(*np.quantile(lnprob, [0.1, 1.]))
    axes[3].set_xlabel('MCMC Step')
    axes[3].set_ylabel('ln(Pr)')
    axes[3].set_title('MCMC Chain Trace (n_sources)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"MCMC QA figure saved to: {save_path}")
    plt.show()
    
    return best_recon, best_x, best_y

def save_mcmc_results(sampler: emcee.EnsembleSampler, parameterization: MCMCParameterization,
                     convergence_info: dict, save_path: str = "mcmc_results.npz"):
    """Save MCMC results"""
    chain = sampler.get_chain()
    log_prob = sampler.get_log_prob()
    
    np.savez(
        save_path,
        chain=chain,
        log_prob=log_prob,
        acceptance_fraction=sampler.acceptance_fraction,
        max_sources=parameterization.max_sources,
        image_size=(parameterization.H, parameterization.W),
        convergence_info=convergence_info
    )
    print(f"MCMC results saved to: {save_path}")

def create_synthetic_data(psf: PSF, n_true_sources: int = 3, image_size: Tuple[int, int] = (64,64), 
                         pad: int = 10, noise_std: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic test data (NumPy version)
    
    Parameters:
    -----------
    psf : PSF
        PSF model to use for generating synthetic data
    n_true_sources : int, optional
        Number of true sources to generate, by default 3
    image_size : Tuple[int, int], optional
        Image dimensions (H, W), by default (100, 100)
    pad : int, optional
        Padding from edges for source placement, by default 10
    noise_std : float, optional
        Noise standard deviation, by default 0.01
        
    Returns:
    --------
    observed_image : np.ndarray
        Synthetic observed image with noise
    true_x : np.ndarray
        True x coordinates of sources
    true_y : np.ndarray  
        True y coordinates of sources
    true_flux : np.ndarray
        True flux values of sources
    """
    H, W = image_size
    
    true_x = np.random.uniform(pad, W - pad, n_true_sources)
    true_y = np.random.uniform(pad, H - pad, n_true_sources)
    true_flux = np.random.uniform(0.5, 3.0, n_true_sources)
    
    model = PointSourceModel(psf, noise_std, image_size)
    
    clean_image = model.forward(true_x, true_y, true_flux)
    noise = np.random.normal(0, noise_std, clean_image.shape)
    observed_image = clean_image + noise
    
    return observed_image, true_x, true_y, true_flux

# Example usage
if __name__ == "__main__":
    # Create PSF object
    alpha = 2.5
    psf_fwhm = 5.9
    gamma = psf_fwhm / (2.0 * np.sqrt(2**(1/2.5) - 1))
    psf = PSF(gamma=gamma, alpha=alpha)
    
    # Create synthetic data
    use_synthetic = False
    if use_synthetic:
        observed_image, true_x, true_y, true_flux = create_synthetic_data(psf, n_true_sources=3)
        
        print("=== MCMC POINT SOURCE DETECTION ===")
        print("True sources:")
        for i in range(len(true_x)):
            print(f"  Source {i}: x={true_x[i]:.2f}, y={true_y[i]:.2f}, flux={true_flux[i]:.3f}")
    else:
        observed_image = np.load('/Users/kadofong/Downloads/pieridae_testing_output/single_mcmc_test/M3495470610002241759/M3495470610002241759_n540_arr.npy')
        true_x = None
        true_y = None

    
    # Ground truth likelihood using the same PSF object
    flat_bkg = sampling.sigmaclipped_std(observed_image)
    print(f'Flat background = {flat_bkg:.2f} cts')
    forward_model = PointSourceModel(psf, noise_std=flat_bkg, image_size=observed_image.shape)
    
    if use_synthetic:
        true_likelihood = forward_model.log_likelihood(observed_image, true_x, true_y, true_flux)
        print(f"Ground truth likelihood: {true_likelihood:.1f}")
    
    # Run MCMC
    sampler, parameterization, model = run_mcmc_sampling(
        observed_image,
        psf,
        max_sources=8,  # Reduced for testing
        n_walkers=64,   # Reduced for testing
        n_steps=2000,    # Reduced for testing
        find_peaks=True,
        burn_in=1,     # Reduced for testing
        noise_std=flat_bkg,
    )
    
    # Analyze convergence
    convergence_info = analyze_convergence(sampler, parameterization)
    
    if use_synthetic:
        # Evaluate performance
        performances, summary = evaluate_mcmc_performance(
            sampler, parameterization, model, observed_image, true_likelihood
        )
        
        print(f"\nFinal MCMC performance: {summary['mean_performance']:.1%} of ground truth")
        
    # Generate QA figure
    best_recon, best_x, best_y = generate_mcmc_qa_figure(
        sampler, parameterization, model, observed_image, true_x, true_y
    )
    
    # Save results
    save_mcmc_results(sampler, parameterization, convergence_info)
    
    if use_synthetic:
        print(f"\n=== FINAL SUMMARY ===")
        print(f"Mean performance: {summary['mean_performance']:.3f}")
        print(f"Best performance: {summary['best_performance']:.3f}")
        print(f"Performance std: {summary['std_performance']:.3f}")        
    print(f"MCMC analysis complete!")