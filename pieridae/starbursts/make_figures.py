import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import pandas as pd
from astropy import units as u
from astropy import cosmology
from ekfplot import plot as ek
from ekfplot import colors as ec
from ekfphys import calibrations
from carpenter import emission, conventions, pixels
from agrias import photometry, utils


cosmo = cosmology.FlatLambdaCDM(70.,0.3)
starburst_color = ec.ColorBase('#1F8AE0')
normal_color = ec.ColorBase('#787878')

def get_tree_nndist ( target_tree, neighbor_tree, rmin ):
    dd, ii = neighbor_tree.query(target_tree.data, 10)
    too_close = dd < rmin.to(u.deg).value
    dd = np.where(too_close, np.nan, dd)
    d2d = np.nanmin(dd, axis=1) * u.deg
    d2d_phys = (d2d * cosmo.kpc_comoving_per_arcmin(0.08)).to(u.Mpc)
    
    # Get the index of the minimum valid distance for each target point
    valid_idx = np.nanargmin(dd, axis=1)
    # Get the corresponding neighbor indices
    neighbor_idx = ii[np.arange(len(ii)), valid_idx]
        
    return neighbor_idx, d2d_phys
    

def mk_pairdistance ( catalog, masks,  ):
    from scipy.spatial import cKDTree
    
    rmin = 30.*u.arcsec
    
    in_band = masks['in_band'][0]
    is_good = masks['is_good'][0]
    is_starburst = masks['is_starburst'][0]
    is_massive = catalog.loc[:,'logmass'] > 10.
    
    tree_all = cKDTree(catalog.loc[masks['is_good'][0],['RA','DEC']].values)
    tree_massive_all = cKDTree(catalog.loc[masks['is_good'][0]&is_massive,['RA','DEC']].values)
    tree_sb_all = cKDTree(catalog.loc[masks['is_good'][0]&masks['is_starburst'][0],['RA','DEC']].values)
    tree_norm_all = cKDTree(catalog.loc[masks['is_good'][0]&~masks['is_starburst'][0],['RA','DEC']].values)    
    tree_spec = cKDTree(catalog.loc[masks['in_band'][0],['RA','DEC']].values)
    tree_massive_spec = cKDTree(catalog.loc[masks['in_band'][0]&is_massive,['RA','DEC']].values)
    tree_sb_spec = cKDTree(catalog.loc[masks['in_band'][0]&is_starburst,['RA','DEC']].values)
    tree_norm_spec = cKDTree(catalog.loc[masks['in_band'][0]&~is_starburst,['RA','DEC']].values)

    _, d2d_spec_phys = get_tree_nndist ( tree_norm_spec, tree_spec, rmin )
    neighbor_index, d2d_sb_spec_phys = get_tree_nndist ( tree_sb_spec, tree_spec, rmin )

    _, d2d_all_phys = get_tree_nndist ( tree_norm_all, tree_all, rmin )
    _, d2d_sb_all_phys = get_tree_nndist ( tree_sb_all, tree_all, rmin )
    
    conversion = cosmo.kpc_comoving_per_arcmin(0.08).to(u.Mpc/u.deg)
    bins = np.linspace((rmin*conversion).to(u.Mpc).value, 1.,30)
    
    fig, axarr = plt.subplots(1,2,figsize=(10,4))
    ek.hist(d2d_spec_phys.value, density=True, lw=2, alpha=0.2, bins=bins, ax=axarr[0], color=normal_color.base, label='all Merian galaxies')
    ek.hist(d2d_sb_spec_phys.value, density=True, lw=2, alpha=0.2, bins=bins, ax=axarr[0], 
            color=starburst_color.base, label='Merian starbursts')

    #ek.hist(d2d_spec_phys.value, density=True, lw=2, histtype='step', color='grey', bins=bins, ax=axarr[1], ls='--')
    ek.hist(d2d_all_phys.value, density=True, lw=2, alpha=0.2, bins=bins, ax=axarr[1], color=normal_color.base, label='all Merian galaxies')
    ek.hist(d2d_sb_all_phys.value, density=True, lw=2, alpha=0.2, bins=bins, ax=axarr[1], color=starburst_color.base,
            label='Merian starbursts')    
    
    for ax in axarr:
        ax.set_xlabel(r'd$_{\rm nearest}$ (Mpc)')
        ax.set_ylabel(r'PDF')
    axarr[0].legend(bbox_to_anchor=(1.,0.9))
    
    ek.text(0.95,0.95, 'spectroscopic sample', ax=axarr[0])
    ek.text(0.95,0.95, 'Merian emission sample', ax=axarr[1])
    
    return neighbor_index, d2d_sb_spec_phys