from __future__ import print_function
import time
import datetime
import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import copy 
import astropy.units as u
import astropy.constants as const
from scipy.interpolate import CubicSpline
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70*u.km / (u.Mpc*u.s), Om0=0.286, Ob0=0.047)

from .param_argparser import *

class empty_table():
    """
    simple class creating an empty table
    used for halo catalog and map instances
    (allows for attribute assignment)
    """
    def __init__(self):
        pass

    def copy(self):
        """@brief Creates a copy of the table."""
        return copy.copy(self)
    

### PARAMETER MANAGEMENT
class SimParameters():
    """
    class used to hold all the parameters that commonly need to be passed between functions
    arguments can be passed either using argparse or in a file with a set of default parameters
    (or can be assigned in the script after the fact)
    a list of parameters (and their defaults) is in param_argparser.py
    """
    def __init__(self):
        # read arguments in from argparser and store them in this (copyable, printable) class
        pars = parser.parse_args()
        param_dict = vars(pars)

        for key, val in param_dict.items():
            setattr(self, key, val)

        # set up cosmology object in params
        if self.cosmology == 'comap':
            self.cosmo = FlatLambdaCDM(H0=70*u.km / (u.Mpc*u.s), Om0=0.286, Ob0=0.047)
        else:
            assert self.cosmology != 'comap', "Don't recognize this cosmology"

        # other calculated metainfo
        self.z_i    = self.nu_rest/self.nu_i - 1
        self.z_f    = self.nu_rest/self.nu_f - 1
    

    def copy(self):
        """@brief Creates a copy of the table."""
        return copy.deepcopy(self)

    def print(self):
        """output the current set of attributes"""
        attrlist = []
        for i in dir(self):
            if i[0]=='_': continue
            elif i == 'copy': continue
            else: attrlist.append(i)
        print(attrlist)


def write_time(string_in):
    """
    write time info in as nicely formatted string
    """
    fmt       = '%H:%M:%S on %m/%d/%Y'
    timestamp = datetime.datetime.now().strftime(fmt)
    bar = 72*'-'
    print( '\n\n'+bar )
    print( string_in )
    print( 'Time:      '+timestamp )
    print( bar+'\n' )

    return

def timeme(method):
    """
    writes the time it takes to run a function
    To use, pput above a function definition. eg:
    @timeme
    def Lco_to_map(halos,map):
    """
    def wrapper(*args, **kw):
        startTime = int(round(time.time()))
        result = method(*args, **kw)
        endTime = int(round(time.time()))

        print('  ',endTime - startTime,'sec')
        return result

    return wrapper

def make_output_filenames(params, outputdir=None):
    """
    Uses the parameters in the input file to set up the result files to be output
    (the cube file, catalogue file and the two plot files if plotting)

    inputs:
    -------
        params: SimParameters object
            uses params.halo_catalog_file
        outputdir: str (optional)
            directory to prepend to file names (defaults to './output/')
    outputs:
    --------
        adds map_output_file, halo_output_file, plot_cube_file, plot_pspec_file

    """
    # default to a folder with the model name in a folder called output
    if not outputdir:
        outputdir = './output/' + params.model

    # make the output directory if it doesn't already exist
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    halofile = params.halo_catalog_file
    seedname = halofile[halofile.find('seed'):-4]

    params.map_output_file = outputdir + '/Lco_cube_' + params.model + '_' + seedname
    params.halo_output_file = outputdir + '/Lco_cat_' + params.model + '_' + seedname
    params.plot_cube_file = outputdir + '/cube_' + params.model + '_' + seedname
    params.plot_pspec_file = outputdir + '/pspec_' + params.model + '_' + seedname
    return


# dealing with magnitudes
def mag_abs_to_rel(Mg, z, dered=False):
    """ convert from absolute magnitudes Mg to relative magnitudes, using the Schlegel 1990 K-corrections
     following Croom et al. 2009"""
    if dered:
        kz = np.arange(2.0, 3.5, 0.1)
        kKB = np.array([-0.53, -0.55, -0.61, -0.64, -0.64, -0.60, -.56, -0.51, -0.46, -0.4,
                                -0.33, -0.25, -0.16, -0.05, 0.07])
        kinterp = CubicSpline(kz, kKB)
        kfac = kinterp(z) - kinterp(2)
    else:
        kfac = 0.
    
    return Mg + cosmo.distmod(z).value + kfac

# Cosmology Functions
# Explicitily defined here instead of using something like astropy
# in order for ease of use on any machine
def hubble(z,h,omegam):
    """
    H(z) in units of km/s
    """
    return h*100*np.sqrt(omegam*(1+z)**3+1-omegam)

def drdz(z,h,omegam):
    return 299792.458 / hubble(z,h,omegam)

def chi_to_redshift(chi, cosmo):
    """
    Transform from redshift to comoving distance
    Agrees with NED cosmology to 0.01% - http://www.astro.ucla.edu/~wright/CosmoCalc.html
    """
    zinterp = np.linspace(0,4,10000)
    dz      = zinterp[1]-zinterp[0]

    try:
        chiinterp  = np.cumsum( drdz(zinterp,cosmo.h,cosmo.Omega_M) * dz)
    except AttributeError:
        chiinterp = np.cumsum(drdz(zinterp, cosmo.h, cosmo.Om0) * dz)
    chiinterp -= chiinterp[0]
    z_of_chi   = sp.interpolate.interp1d(chiinterp,zinterp)

    return z_of_chi(chi)

def redshift_to_chi(z, cosmo):
    """
    Transform from comoving distance to redshift
    Agrees with NED cosmology to 0.01% - http://www.astro.ucla.edu/~wright/CosmoCalc.html
    """
    zinterp = np.linspace(0,4,10000)
    dz      = zinterp[1]-zinterp[0]

    try:
        chiinterp  = np.cumsum( drdz(zinterp,cosmo.h,cosmo.Omega_M) * dz)
    except AttributeError:
        chiinterp = np.cumsum(drdz(zinterp, cosmo.h, cosmo.Om0) * dz)
    chiinterp -= chiinterp[0]
    chi_of_z   = sp.interpolate.interp1d(zinterp,chiinterp)

    return chi_of_z(z)

""" DOPPLER CONVERSIONS """
def freq_to_z(nuem, nuobs):
    """
    returns a redshift given an observed and emitted frequency
    """
    zval = (nuem - nuobs) / nuobs
    return zval

def nuem_to_nuobs(nuem, z):
    """
    returns the frequency at which an emitted line at a given redshift would be
    observed
    """
    nuobs = nuem / (1 + z)
    return nuobs

def nuobs_to_nuem(nuobs, z):
    """
    returns the frequency at which an observed line at a given redshift would have
    been emitted
    """
    nuem = nuobs * (1 + z)
    return nuem


### LUMINOSITY AND MASS FUNCTIONS
def log_lum_func(halos, mapinst, params, attribute='Lcat', lumrange=None, nbins=500, unit=u.erg/u.s):
    """
    calculates a luminosity function in logspace from a halo catalog
    
    INPUTS
    ------
        halos: 
            (HaloCatalog) the halos object containing the luminosities
        mapinst:
            (SimMap) map object (for total volume of the sim) ****this could be better
        params:
            (SimParameters) parameters object *****
        attribute:
            (string, default 'Lcat') which luminosities to get the luminosity function from
        lumrange:
            (tuple, default None) the (log) luminsoities to calculate the luminosity function between
        nbins:
            (int, default 500) the number of bins to use when histogramming
        unit:
            (astropy.units object, default erg/s) the unit to return the x axis in
    OUTPUTS
    -------
        bincents:
            (array) the x-axis, centers of the numpy histogram bins
        cumhist:
            (array) the luminosity function y-axis

    """
    
    # get logarithmic luminosities, deal with inf values (from zero SFR if using Chung model)
    loglyalums = np.log10(getattr(halos, attribute))
    infidx = np.where(np.isinf(loglyalums))
    noninfidx = np.where(~np.isinf(loglyalums))
    loglyalums[infidx] = 0.
    
    if not lumrange:
        # find lims of the nonzero luminosities
        lumrange = (np.nanmin(loglyalums[noninfidx]), np.nanmax(loglyalums[noninfidx]))
    
    # histogram them up
    counts, bins = np.histogram(loglyalums, bins=nbins, range=lumrange)
    dL = np.diff(bins)
    
    # number density / lum density
    hist = counts / mapinst.totalcovol.value
    # x axis
    bincents = bins[:-1] + dL / 2
    bincents = 10**bincents * const.L_sun.to(unit)
    
    # cumulative
    cumhist = np.flip(np.cumsum(np.flip(hist)))
    
    return (bincents.value, cumhist)


def plot_results(mapinst,k,Pk,Pk_sampleerr,params):
    """
    Plot central frequency map and or powerspectrum
    """
    if params.verbose: print("\n\tPlotting results")

    ### Plot central frequency map
    plt.rcParams['font.size'] = 16
    if params.plot_cube:
        plt.figure().set_tight_layout(True)
        im = plt.imshow(np.log10(mapinst.maps[:,:,params.nmaps//2]+1e-6), extent=[-mapinst.fov_x/2,mapinst.fov_x/2,-mapinst.fov_y/2,mapinst.fov_y/2],vmin=-1,vmax=2)
        plt.colorbar(im,label=r'$log_{10}\ T_b\ [\mu K]$')
        plt.xlabel('degrees',fontsize=20)
        plt.ylabel('degrees',fontsize=20)
        plt.title('simulated map at {0:.3f} GHz'.format(mapinst.nu_bincents[params.nmaps//2]),fontsize=24)
        plt.savefig(params.plot_cube_file)

    if params.plot_pspec:
        plt.figure().set_tight_layout(True)
        plt.errorbar(k,k**3*Pk/(2*np.pi**2),k**3*Pk_sampleerr/(2*np.pi**2),
                     lw=3,capsize=0)
        plt.gca().set_xscale('log')
        plt.gca().set_yscale('log')
        plt.grid(True)
        plt.xlabel('k [1/Mpc]',fontsize=18)
        plt.ylabel('$\\Delta^2(k)$ [$\\mu$K$^2$]',fontsize=18)
        plt.title('simulated line power spectrum',fontsize=24)
        plt.savefig(params.plot_pspec_file)

    if params.plot_cube or params.plot_pspec:
        plt.show()
        backpath = os.path.normpath(os.getcwd() + os.sep + os.pardir)
        os.chdir(backpath)

    return
