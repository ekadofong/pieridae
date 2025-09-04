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


def load_sample ():
    catalog = pd.read_parquet('../../carpenter/data/MDR1_catalogs/mdr1_n708maglt26_and_pzgteq0p1.parquet')
    catalog = catalog.set_index('objectId_Merian')
    catalog.index = [ f'M{sidx}' for sidx in catalog.index ]

    av, u_av = emission.estimate_av(catalog,)
    catalog['AV'] = av
    catalog['u_AV'] = u_av

    emission_corrections = emission.compute_emissioncorrections(catalog, logmstar_key='logmass_gaap1p0')

    ancline_correction, ge_correction, extinction_correction, catalog_apercorr = emission_corrections

    n708_fluxes, n708_luminosities, n708_eqws, n708_fcont = photometry.mbestimate_halpha(
        catalog[utils.photcols['N708']].values,
        catalog[utils.photcols['g']].values,
        catalog[utils.photcols['r']].values,
        catalog[utils.photcols['i']].values,
        catalog[utils.photcols['z']].values,
        np.full(len(catalog),0.08),
        catalog[utils.u_photcols['N708']].values,
        0.,
        0.,
        band='n708',
        apercorr=1.,
        ge_correction=ge_correction[:,2],
        ex_correction=extinction_correction[0,:,2],
        u_ex_correction = 0.*extinction_correction[1,:,2], # \\ we're actually going to do dust errors downstream
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
        np.full(len(catalog),0.08),
        catalog[utils.u_photcols['N540']].values,
        0.,
        0.,
        band='n540',
        apercorr=1.,
        ge_correction=ge_correction[:,2],
        ex_correction=extinction_correction[0,:,2],
        u_ex_correction = 0.*extinction_correction[1,:,2], # \\ we're actually going to do dust errors downstream
        ns_correction=ancline_correction[:],
        do_aperturecorrection=False,
        do_gecorrection=False,
        do_extinctioncorrection=False,
        do_linecorrection=True,
        specflux_unit=u.nJy,
        ctype='linear',
        plawbands='gr',
    ) 


    catalog['haew'] = n708_eqws[0].value
    catalog['oiiiew'] = np.where(catalog['logmass']<9.6, n540_eqws[0].value, np.nan)
    catalog['rmag'] = -2.5*np.log10(catalog['r_cModelFlux_Merian']*1e-9/3631.)
    catalog['F_Ha_obs'] = n708_fluxes[0].value * catalog_apercorr
    catalog['L_Ha'] = ( n708_luminosities[0].value * extinction_correction[0,:,2] * ge_correction[:,2] * catalog_apercorr )
    catalog['L_OIII'] = n540_luminosities[0].value * extinction_correction[0,:,3] * ge_correction[:,3] * catalog_apercorr
    catalog['n540_apercorr'] = catalog['N540_cModelFlux_Merian']/catalog['N540_gaap1p0Flux_Merian']
    catalog['i_apercorr'] = catalog['i_cModelFlux_Merian']/catalog['i_gaap1p0Flux_aperCorr_Merian']
    catalog['lineratio'] = catalog.loc[:, 'L_OIII']/catalog.loc[:, 'L_Ha']
    catalog['pz'] = catalog['pz1']+catalog['pz2']+catalog['pz3']+catalog['pz4']

    is_inmassrange = (catalog['logmass']>7.75)&(catalog['logmass']<12.)&(catalog['i_apercorr']<4.)&(catalog['n540_apercorr']<4.)
    is_emitter = (catalog['haew']>5.)&(catalog['oiiiew']>5.)
    in_band = (catalog['z_spec']>0.05)&(catalog['z_spec']<0.11)
    zphot_select = catalog['pz']>0.26
    is_good = is_inmassrange & zphot_select

    z = 0.08
    alpha = -0.13*z + 0.8
    sfr0 = 1.24*z - 1.47
    sigma = 0.22*z + 0.38
    is_starburst = catalog['L_Ha'] > calibrations.SFR2LHa(10.**(alpha * (catalog['logmass'] - 8.5) + sfr0 + 1.5*sigma))

    has_zspec = np.isfinite(catalog['z_spec'])
    in_band = (catalog['z_spec']>0.05)&(catalog['z_spec']<0.11)    
    
    masks = {'in_band':(in_band, 'in band spectroscopic redshift'), 
             'is_starburst':(is_starburst, 'is a Merian starburst'), 
             'is_good':(is_good, '7.75<logM*<12 AND i_apercorr < 4 AND n540_apercorr < 4 AND HaEW > 5 AND OIIIEW > 5. AND \int{p(z)} > 0.26')}
    
    return catalog, masks

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
    tree_spec = cKDTree(catalog.loc[masks['in_band'][0],['RA','DEC']].values)
    tree_massive_spec = cKDTree(catalog.loc[masks['in_band'][0]&is_massive,['RA','DEC']].values)
    tree_sb_spec = cKDTree(catalog.loc[masks['in_band'][0]&is_starburst,['RA','DEC']].values)

    _, d2d_spec_phys = get_tree_nndist ( tree_spec, tree_spec, rmin )
    neighbor_index, d2d_sb_spec_phys = get_tree_nndist ( tree_sb_spec, tree_spec, rmin )

    _, d2d_all_phys = get_tree_nndist ( tree_all, tree_all, rmin )
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