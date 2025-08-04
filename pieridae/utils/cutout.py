import os, sys
sys.path.append('../src')
from carpenter import handler
from carpenter.conventions import produce_merianobjectname
import numpy as np
from astropy import coordinates
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import colors
from astropy.visualization import make_lupton_rgb

from carpenter import pixels

def pull_cutouts ( coordinates, savedir, hsc_username=None, hsc_password=None ):
    handler.fetch_merian(cfile, savedir)
    handler.fetch_hsc(cfile, savedir, hsc_username=username, hsc_passwd=password)

def build_bbmb ( gid, **kwargs ):
    bbmb = pixels.BBMBImage ( )
    for band in ['g','N540','r','N708','i','z','y']:
        bbmb.add_band ( band, *load_image(gid, band) )

    fwhm_a, _ = bbmb.measure_psfsizes()
    mim, mpsf = bbmb.match_psfs ( np.argmax(fwhm_a), cbell_alpha=1., **kwargs )
    return bbmb