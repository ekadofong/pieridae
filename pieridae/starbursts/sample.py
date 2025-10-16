import numpy as np
import pandas as pd
from astropy import units as u
from ekfphys import calibrations
from carpenter import emission
from agrias import photometry, utils


def load_sample (filename=None):
    if filename is None:
        filename = '../../carpenter/data/MDR1_catalogs/mdr1_n708maglt26_and_pzgteq0p1.parquet'
    catalog = pd.read_parquet(filename)
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

    is_inmassrange = (catalog['logmass']>7.75)&(catalog['logmass']<12.)#&(catalog['i_apercorr']<4.)&(catalog['n540_apercorr']<4.)
    is_emitter = (catalog['haew']>5.)&(catalog['oiiiew']>5.)
    in_band = (catalog['z_spec']>0.05)&(catalog['z_spec']<0.11)
    zphot_select = catalog['pz']>0.26
    is_good = is_inmassrange & zphot_select

    z = 0.08
    alpha = -0.13*z + 0.8
    sfr0 = 1.24*z - 1.47
    sigma = 0.22*z + 0.38
    is_starburst = catalog['L_Ha'] > calibrations.SFR2LHa(10.**(alpha * (catalog['logmass'] - 8.5) + sfr0 + 1.5*sigma))
    is_msorabove = catalog['L_Ha'] > calibrations.SFR2LHa(10.**(alpha * (catalog['logmass'] - 8.5) + sfr0))
    is_alosbms = catalog['L_Ha'] > calibrations.SFR2LHa(10.**(alpha * (catalog['logmass'] - 8.5) + sfr0 - sigma))
    
    has_zspec = np.isfinite(catalog['z_spec'])
    in_band = (catalog['z_spec']>0.05)&(catalog['z_spec']<0.11)    
    
    masks = {
             'in_band':(in_band, 'in band spectroscopic redshift'), 
             'is_starburst':(is_starburst, 'is a Merian starburst'), 
             'is_msorabove':(is_msorabove, 'is at least at the SAGAbg SFS'),
             'is_good':(is_good, '7.75<logM*<12 AND int{p(z)} > 0.26'),            
             'is_alosbms':(is_alosbms, 'is at least one sigma below the SAGAbg SFS')
             'is_emitter':(is_emitter, 'EW(Ha)>5 Ang AND EW(OIII)>5 Ang')
             }
    
    return catalog, masks

