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

def detect_peaks(image: np.ndarray, min_distance: int = 3, threshold_rel: float = 5.,
                bkg_std: float = 0.01, gamma: float = 2.0, alpha: float = 2.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect peaks and estimate flux (NumPy version)"""
    local_max = maximum_filter(image, size=min_distance) == image
    threshold = threshold_rel * bkg_std
    peaks_mask = local_max & (image > threshold)
    
    y_coords, x_coords = np.where(peaks_mask)
    
    if len(x_coords) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Flux estimation using PSF normalization
    peak_fluxes = image[y_coords, x_coords]
    flux_estimates = peak_fluxes * np.pi * gamma**2 / (alpha - 1.)
    
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
        
        return log_like

class MCMCParameterization:
    """Handle variable number of sources in fixed parameter space"""
    
    def __init__(self, max_sources: int, image_size: Tuple[int, int]):
        self.max_sources = max_sources
        self.H, self.W = image_size
        
        # Parameter layout: [n_sources, x1, y1, flux1, x2, y2, flux2, ...]
        self.n_params = 1 + 3 * max_sources  # n_sources + (x, y, flux) * max_sources
        
        # Parameter bounds
        self.bounds = self._setup_bounds()
    
    def _setup_bounds(self):
        """Setup parameter bounds for sampling"""
        bounds = []
        
        # Number of sources: [0, max_sources]
        bounds.append((0, self.max_sources))
        
        # Source parameters
        for i in range(self.max_sources):
            bounds.append((0, self.W - 1))      # x position
            bounds.append((0, self.H - 1))      # y position  
            bounds.append((0.01, 10.0))         # flux (positive)
        
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
        fluxes = params[3:3+3*n_sources:3]
        
        return n_sources, x_sources, y_sources, fluxes
    
    def initialize_from_peaks(self, observed_image: np.ndarray, n_walkers: int, 
                             find_peaks: bool = True) -> np.ndarray:
        """Initialize walker positions"""
        initial_params = np.zeros((n_walkers, self.n_params))
        
        if find_peaks:
            # Detect peaks
            x_peaks, y_peaks, flux_peaks = detect_peaks(observed_image)
            n_detected = len(x_peaks)
            jitter = 5
            
            if n_detected > 0:
                print(f"Detected {n_detected} peaks for MCMC initialization")
                
                # Initialize around detected number of sources
                n_init = min(n_detected, self.max_sources)
                initial_params[:, 0] = np.random.normal(n_init, 0.5, n_walkers)
                
                # Initialize source parameters with noise around detected peaks
                for i in range(n_walkers):
                    n_sources_walker = int(np.clip(np.round(initial_params[i, 0]), 0, self.max_sources))
                    
                    for j in range(n_sources_walker):
                        if j < n_detected:
                            # Use detected peak with substantial noise for diversity
                            initial_params[i, 1 + j*3] = x_peaks[j] + np.random.normal(0, jitter)  # x
                            initial_params[i, 2 + j*3] = y_peaks[j] + np.random.normal(0, jitter)  # y
                            initial_params[i, 3 + j*3] = flux_peaks[j] * np.exp(np.random.normal(0, 0.5))  # flux
                        else:
                            # Random initialization for extra sources
                            initial_params[i, 1 + j*3] = np.random.uniform(0, self.W - 1)  # x
                            initial_params[i, 2 + j*3] = np.random.uniform(0, self.H - 1)  # y  
                            initial_params[i, 3 + j*3] = np.random.gamma(2.0, 0.5)  # flux
                    
                    # Add noise to unused parameters to ensure diversity
                    for j in range(n_sources_walker, self.max_sources):
                        initial_params[i, 1 + j*3] = np.random.uniform(0, self.W - 1)  # x
                        initial_params[i, 2 + j*3] = np.random.uniform(0, self.H - 1)  # y
                        initial_params[i, 3 + j*3] = np.random.gamma(1.0, 0.5)  # flux
                
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
                initial_params[i, 3 + j*3] = np.random.gamma(3.0) / 2.0  # flux (optimized prior)
    
    def _enforce_bounds(self, params: np.ndarray) -> np.ndarray:
        """Enforce parameter bounds"""
        params_clipped = params.copy()
        
        for i, (low, high) in enumerate(self.bounds):
            params_clipped[:, i] = np.clip(params_clipped[:, i], low, high)
        
        return params_clipped

class MCMCPosterior:
    """Compute log posterior probability"""
    
    def __init__(self, forward_model: PointSourceModel, parameterization: MCMCParameterization,
                 observed_image: np.ndarray):
        self.forward_model = forward_model
        self.parameterization = parameterization
        self.observed_image = observed_image
        
        # Optimized prior parameters (from VI analysis)
        self.poisson_lambda = 2.0
        self.gamma_alpha = 3.0
        self.gamma_beta = 2.0
    
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
            # Uniform priors on positions (implicitly satisfied by bounds)
            log_prior += -n_sources * np.log(self.parameterization.H * self.parameterization.W)
            
            # Gamma prior on fluxes
            log_prior += np.sum((self.gamma_alpha - 1) * np.log(fluxes) - self.gamma_beta * fluxes)
        
        return log_prior
    
    def log_likelihood(self, params: np.ndarray) -> float:
        """Compute log likelihood"""
        n_sources, x_sources, y_sources, fluxes = self.parameterization.params_to_sources(params)
        return self.forward_model.log_likelihood(self.observed_image, x_sources, y_sources, fluxes)
    
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

def run_mcmc_sampling(observed_image: np.ndarray, max_sources: int = 10, n_walkers: int = 100, 
                     n_steps: int = 5000, find_peaks: bool = True, 
                     burn_in: int = 1000) -> Tuple[emcee.EnsembleSampler, MCMCParameterization, PointSourceModel]:
    """Run MCMC sampling for point source detection"""
    
    print("=== MCMC POINT SOURCE DETECTION ===")
    
    # Convert torch tensor to numpy if needed
    if torch.is_tensor(observed_image):
        observed_image = observed_image.cpu().numpy()
    
    image_size = observed_image.shape
    
    # Setup components
    psf = PSF(gamma=2.0, alpha=2.5)
    forward_model = PointSourceModel(psf, noise_std=0.01, image_size=image_size)
    parameterization = MCMCParameterization(max_sources, image_size)
    posterior = MCMCPosterior(forward_model, parameterization, observed_image)
    
    # Initialize walkers
    print(f"Initializing {n_walkers} walkers...")
    initial_positions = parameterization.initialize_from_peaks(observed_image, n_walkers, find_peaks)
    
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
                           true_x: np.ndarray, true_y: np.ndarray, 
                           save_path: str = "mcmc_qa.png") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate QA figure for MCMC results"""
    print(f"\n=== GENERATING MCMC QA FIGURE ===")
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Panel 1: Observed image
    im = axes[0].imshow(observed_image, cmap='viridis')
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
    axes[1].scatter(true_x, true_y, c='red', s=50, marker='x', label='True sources')
    if len(best_x) > 0:
        axes[1].scatter(best_x, best_y, c='lime', s=50, marker='+', label='Best MCMC sample', alpha=1.0)
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

def create_synthetic_data(n_true_sources: int = 3, image_size: Tuple[int, int] = (100, 100), 
                         pad: int = 10, noise_std: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic test data (NumPy version)"""
    H, W = image_size
    
    true_x = np.random.uniform(pad, W - pad, n_true_sources)
    true_y = np.random.uniform(pad, H - pad, n_true_sources)
    true_flux = np.random.uniform(0.5, 1.5, n_true_sources)
    
    psf = PSF(gamma=2.0, alpha=2.5)
    model = PointSourceModel(psf, noise_std, image_size)
    
    clean_image = model.forward(true_x, true_y, true_flux)
    noise = np.random.normal(0, noise_std, clean_image.shape)
    observed_image = clean_image + noise
    
    return observed_image, true_x, true_y, true_flux

# Example usage
if __name__ == "__main__":
    # Create synthetic data
    observed_image, true_x, true_y, true_flux = create_synthetic_data(n_true_sources=9)
    
    print("=== MCMC POINT SOURCE DETECTION ===")
    print("True sources:")
    for i in range(len(true_x)):
        print(f"  Source {i}: x={true_x[i]:.2f}, y={true_y[i]:.2f}, flux={true_flux[i]:.3f}")
    
    # Ground truth likelihood
    psf = PSF(gamma=2.0, alpha=2.5)
    forward_model = PointSourceModel(psf, noise_std=0.01, image_size=observed_image.shape)
    true_likelihood = forward_model.log_likelihood(observed_image, true_x, true_y, true_flux)
    print(f"Ground truth likelihood: {true_likelihood:.1f}")
    
    # Run MCMC
    sampler, parameterization, model = run_mcmc_sampling(
        observed_image, 
        max_sources=10,  # Reduced for testing
        n_walkers=64,   # Reduced for testing
        n_steps=3000,    # Reduced for testing
        find_peaks=True,
        burn_in=100     # Reduced for testing
    )
    
    # Analyze convergence
    convergence_info = analyze_convergence(sampler, parameterization)
    
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
    
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Mean performance: {summary['mean_performance']:.3f}")
    print(f"Best performance: {summary['best_performance']:.3f}")
    print(f"Performance std: {summary['std_performance']:.3f}")
    print(f"MCMC analysis complete!")