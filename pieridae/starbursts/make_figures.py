import numpy as np
import matplotlib.pyplot as plt
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


def mk_pairdistance ( catalog, masks ):
    from astropy import coordinates 
    
    in_band = masks['in_band'][0]
    is_good = masks['is_good'][0]
    is_starburst = masks['is_starburst'][0]
    
    catcoords = coordinates.SkyCoord(catalog['RA'].values, catalog['DEC'].values, unit='deg')
    
    _, d2d_spec,_ = catcoords[in_band].match_to_catalog_sky(catcoords[in_band], 2)
    d2d_spec_phys = (d2d_spec * cosmo.kpc_comoving_per_arcmin(0.08)).to(u.Mpc)
    
    _, d2d_all,_ = catcoords[is_good].match_to_catalog_sky(catcoords[is_good], 2)
    d2d_all_phys = (d2d_all * cosmo.kpc_comoving_per_arcmin(0.08)).to(u.Mpc)
    
    conversion = cosmo.kpc_comoving_per_arcmin(0.08).to(u.Mpc/u.deg)
    bins = np.linspace(0., 2.,30)
    
    fig, axarr = plt.subplots(1,2,figsize=(10,4))
    ek.hist(d2d_spec_phys.value, density=True, lw=2, alpha=0.2, bins=bins, ax=axarr[0], color=normal_color.base, label='all Merian galaxies')
    ek.hist(d2d_spec_phys[is_starburst[in_band]].value, density=True, lw=2, alpha=0.2, bins=bins, ax=axarr[0], 
            color=starburst_color.base, label='Merian starbursts')

    #ek.hist(d2d_spec_phys.value, density=True, lw=2, histtype='step', color='grey', bins=bins, ax=axarr[1], ls='--')
    ek.hist(d2d_all_phys.value, density=True, lw=2, alpha=0.2, bins=bins, ax=axarr[1], color=normal_color.base, label='all Merian galaxies')
    ek.hist(d2d_all_phys[is_starburst[is_good]].value, density=True, lw=2, alpha=0.2, bins=bins, ax=axarr[1], color=starburst_color.base,
            label='Merian starbursts')    
    
    for ax in axarr:
        ax.set_xlabel(r'd$_{\rm nearest}$ (Mpc)')
        ax.set_ylabel(r'PDF')
    axarr[0].legend(bbox_to_anchor=(1.,0.9))
    
    ek.text(0.95,0.95, 'spectroscopic sample', ax=axarr[0])
    ek.text(0.95,0.95, 'Merian emission sample', ax=axarr[1])