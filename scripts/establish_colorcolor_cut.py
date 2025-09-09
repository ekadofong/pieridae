#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt

from ekfplot import plot as ek
from ekfstats import sampling, fit
from pieridae.starbursts import sample

def main():
    # Load catalog data
    catalog, masks = sample.load_sample(filename='../../local_data/base_catalogs/mdr1_n708maglt26_and_pzgteq0p1.parquet')
    
    # Calculate color indices
    catgr = -2.5*np.log10(catalog['g_gaap1p0Flux_aperCorr_Merian']/catalog['r_gaap1p0Flux_aperCorr_Merian'])
    catri = -2.5*np.log10(catalog['r_gaap1p0Flux_aperCorr_Merian']/catalog['i_gaap1p0Flux_aperCorr_Merian'])
    mask = (catgr>0.)&(catri>0.)&(catgr<1.)&(catri<1.)
    
    # Create 2D histogram
    (counts,xbins,ybins,_),_=ek.hist2d(
        catgr,
        catri,
        bins=80,
        zscale='log'
    )
    counts = np.where(counts/counts.max() < 0.05, 0., counts)
    
    # Fit ridgeline to the histogram
    coeffs, func_pred, stats = fit.fit_ridgeline_image(counts.T, order=1, x=sampling.midpts(xbins), y=sampling.midpts(ybins))
    
    # Plot the results
    x = np.linspace(0., 1.,)
    ek.outlined_plot(x, func_pred(x), color='r')
    ek.outlined_plot(x, func_pred(x)+0.2, color='r', lw=1, ls='--')
    
    # Print coefficients
    print(coeffs)
    
    # Add text annotation
    ek.text (
        0.025,
        0.975,
        f'ri = {coeffs[1]:.2f}gr + {coeffs[0]:.2f}'
    )
    
    # Set labels
    plt.xlabel('g - r')
    plt.ylabel('r - i')
    plt.tight_layout()
    
    # Save figure
    os.makedirs('../local_data/figures', exist_ok=True)
    plt.savefig('../local_data/figures/colorcolor_cut.png', dpi=300, bbox_inches='tight')
    print(f"Figure saved to ../local_data/figures/colorcolor_cut.png")

if __name__ == "__main__":
    main()