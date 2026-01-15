import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import astropy.units as u
import astropy.constants as const
import sys
import os
from .tools import *

sfr_interp_tab = None

"""
Functions for converting simulated halo properties to mock luminosities
"""

@timeme
def Mhalo_to_Ls(halos, params):
    """
    wrapper function to calculate the CO luminosities and (if required) tracer
    luminosities for the full halo catalog. 

    inputs:
    -------
    halos: HaloCatalog object
        the catalog of simulated halos for which to calculate luminosities
    params: SimParameters object
        the parameters for the simulation run. uses:
            model: (str) the model to use for calculating the Lco values
            catalog_model: (str) the model to use for calculating the Lcat values (if None, don't calculate Lcat values)
            co_model_coeffs: (array-like of floats) the coeffecients for the co model
            catalog_coeffs: (array-like of floats) the coefficients for the catalog model
            codex: (float) the lognormal scatter in the Lco values in dex
            catdex: (float) the lognormal scatter in the Lcat values in dex
            rho: (float, -1 < 0 < 1) the correlation coefficient between the scatter in the two luminosities
            lum_uncert_seed: (int) the random seed for scatter in the luminosities
            save_scatterless_lums: (bool) if true, will save a copy of the luminosities with no random scatter applied
    outputs:
    --------
    adds the following attributes to the input HaloCatalog object:
        Lco: (array-like of floats) the LIM tracer luminosity (in solar luminosities)
        Lcat: (array-like of floats) the galaxy catalogue tracer luminosity (in solar luminosities)
        scatterless_Lco: (array like of floats, optional) LIM tracer luminosity with no scatter applied
        scatterless_Lcat: (array like of floats, optional) galaxy catalogue tracer luminosity with no scatter applied
    """

    # if no random number generator seed is set, give it one
    try:
        seed = params.lum_uncert_seed
    except AttributeError:
        params.lum_uncert_seed = 12345

    try:
        scatterless = params.save_scatterless_lums
    except AttributeError:
        params.save_scatterless_lums = None

    # calculate CO luminosities for each halo without any scatter
    halos.Lco = Mhalo_to_Lco(halos, params, scatter=False)
    print('done CO luminosities')

    # calculate catalog luminosities for each halo without any scatter
    if params.catalog_model:
        halos.Lcat, params = Mhalo_to_Lcatalog(halos, params)
        print('done catalog luminosities')

        # for testing--save the unscattered luminosity values
        if params.save_scatterless_lums:
            halos.scatterless_Lco = copy.deepcopy(halos.Lco)
            halos.scatterless_Lcat = copy.deepcopy(halos.Lcat)

        # calculate the joint scatter
        halos = add_co_tracer_dependant_scatter(halos, params.rho, params.codex, params.catdex, params.lum_uncert_seed)

    else: 
        # for testing--save the unscattered luminosity values
        if params.save_scatterless_lums:
            halos.scatterless_Lco = copy.deepcopy(halos.Lco)

        # co-only scatter
        halos.Lco = add_log_normal_scatter(halos.Lco, params.codex, params.lum_uncert_seed)



@timeme
def Mhalo_to_Lco(halos, params, scatter=True):
    """
    General function to get L_co(M_halo) given a certain model <model>
    if adding your own model follow this structure,
    and simply specify the model to use in the parameter file
    will output halo luminosities in **L_sun**

    Parameters
    ----------
    halos : HaloCatalog class
        Contains all halo information (position, redshift, etc..)
    model : str
        Model to use, specified in the parameter file
    coeffs : array-like
        None for default coeffs
    """
    dict = {'Li':          Mhalo_to_Lco_Li,
            'Li_sc':       Mhalo_to_Lco_Li_sigmasc,
            'Padmanabhan': Mhalo_to_Lco_Padmanabhan,
            'fiuducial':   Mhalo_to_Lco_fiuducial,
            'Yang':        Mhalo_to_Lco_Yang,
            'arbitrary':   Mhalo_to_Lco_arbitrary
            }

    if params.model in dict.keys():
        return dict[params.model](halos, params.co_model_coeffs, scatter=scatter)
    elif params.model == 'schechter': # this one is abundance-matched and needs extra params
        return Mhalo_to_Lco_schechter(halos, params, scatter=scatter)
    else:
        sys.exit('\n\n\tYour model, '+params.model+', does not seem to exist\n\t\tPlease check src/halos_to_luminosity.py to add it\n\n')


def Mhalo_to_Lco_Li(halos, coeffs, scatter=True):
    """
    halo mass to SFR to L_CO
    following the Li 2016 model (arXiv 1503.08833)
    """
    if coeffs is None:
        # Power law parameters from paper
        log_delta_mf,alpha,beta,sigma_sfr,sigma_lco,scale = (
            0.0, 1.37,-1.74, 0.3, 0.3, 1.0)
    else:
        log_delta_mf,alpha,beta,sigma_sfr,sigma_lco,scale = coeffs;
    delta_mf = 10**log_delta_mf;

    # Get Star formation rate
    if not hasattr(halos,'sfr'):
        halos.sfr = Mhalo_to_sfr_Behroozi(halos, sigma_sfr);

    # infrared luminosity
    lir      = halos.sfr * 1e10 / delta_mf
    alphainv = 1./alpha
    # Lco' (observers units)
    Lcop     = lir**alphainv * 10**(-beta * alphainv)
    # Lco in L_sun
    Lco      =  4.9e-5 * Lcop * scale
    if scatter:
        Lco      = add_log_normal_scatter(Lco, sigma_lco, 2)

    return Lco

def Mhalo_to_Lco_Li_sigmasc(halos, coeffs, scatter=True):
    """
    halo mass to SFR to L_CO
    following the Li 2016 model (arXiv 1503.08833)

    DD 2022 - updated to include a single lognormal scatter coeff
    (doing all the scatter on the luminosities directly and not on the SFR values)
    """
    if coeffs is None:
        # Power law parameters from paper
        log_delta_mf,alpha,beta,sigma_sc = (
            0.0, 1.37,-1.74, 0.3)
    else:
        log_delta_mf,alpha,beta,sigma_sc = coeffs;
    delta_mf = 10**log_delta_mf;

    # Get Star formation rate
    if not hasattr(halos,'sfr'):
        halos.sfr = Mhalo_to_sfr_Behroozi(halos, 0);

    # infrared luminosity
    lir      = halos.sfr * 1e10 / delta_mf
    alphainv = 1./alpha
    # Lco' (observers units)
    Lcop     = lir**alphainv * 10**(-beta * alphainv)
    # Lco in L_sun
    Lco      =  4.9e-5 * Lcop

    if scatter:
        Lco      = add_log_normal_scatter(Lco, sigma_sc, 2) 

    return Lco

def Mhalo_to_Lco_Padmanabhan(halos, coeffs, scatter=True):
    """
    halo mass to L_CO
    following the Padmanabhan 2017 model (arXiv 1706.01471)
    DD 2024 -- added duty fraction directly scaling the luminosity of each halo
    (this would more practically be randomly selecting a fraction (1-fduty) of halos to assign 0 luminosity)
    """
    if coeffs is None:
        m10,m11,n10,n11,b10,b11,y10,y11,fduty = (
            4.17e12,-1.17,0.0033,0.04,0.95,0.48,0.66,-0.33,1)
    else:
        m10,m11,n10,n11,b10,b11,y10,y11,fduty = coeffs

    z  = halos.redshift
    hm = halos.M

    m1 = 10**(np.log10(m10)+m11*z/(z+1))
    n  = n10 + n11 * z/(z+1)
    b  = b10 + b11 * z/(z+1)
    y  = y10 + y11 * z/(z+1)

    Lprime = 2 * n * hm / ( (hm/m1)**(-b) + (hm/m1)**y )
    Lco    = 4.9e-5 * Lprime * fduty

    return Lco

def Mhalo_to_Lco_fiuducial(halos, coeffs, scatter=True):
    """
    DD 2022, based on Chung+2022 fiuducial model (arXiv 2111.05931)
    """
    if coeffs is None:
        # default to UM+COLDz+COPSS model from Chung+22
        A, B, logC, logM, sigma = (
            -2.85, -0.42, 10.63, 12.3, 0.42)
    else:
        A,B,logC,logM,sigma = coeffs

    Mh = halos.M

    C = 10**logC
    M = 10**logM

    Lprime = C / ((Mh/M)**A + (Mh/M)**B)
    Lco = 4.9e-5 * Lprime
    if scatter:
        Lco = add_log_normal_scatter(Lco, sigma, 3)

    return Lco

def Mhalo_to_Lco_Yang(halos, coeffs, scatter=True):
    """
    DD 2022, SAM from Breysse+2022/Yang+2021
    arXiv 2111.05933/2108.07716
    Not set up for anything other than CO(1-0) at COMAP redshifts currently
    becasue the model is a pretty complicated function of redshift
    for other models edit function directly with parameters from Yang+22
    """
    if coeffs is not None:
        print('The function is only set up for CO(1-0), 1<z<4')
        return 0

    z = halos.redshift
    Mh = halos.M

    # Lco function
    logM1 = 12.13 - 0.1678*z
    logN = -6.855 + 0.2366*z - 0.05013*z**2
    alpha = 1.642 + 0.1663*z - 0.03238*z**2
    beta = 1.77*np.exp(-z/2.72) - 0.0827

    M1 = 10**logM1
    N = 10**logN

    Lco = 2*N * Mh / ((Mh/M1)**(-alpha) + (Mh/M1)**(beta))

    # fduty function
    logM2 = 11.73 + 0.6634*z
    gamma = 1.37 - 0.190*z + 0.0215*z**2

    M2 = 10**logM2

    fduty = 1 / (1 + (Mh/M2)**gamma)

    Lco = Lco * fduty

    # scatter
    sigmaco = 0.357 - 0.0701*z + 0.00621*z**2
    if scatter:
        Lco = add_log_normal_scatter(Lco, sigmaco, 4)
    return Lco

def Mhalo_to_Lco_schechter(halos, params, scatter=True):
    """ 
    wrapper to use a schechter function to generate catalog luminosities
    """

    Lco, params = abundancematch(schechter, params.co_model_coeffs, halos, params)
    Lco = Lco * 3.826e33 / params.co_model_coeffs[0]
    if scatter:
        Lco      = add_log_normal_scatter(Lco, params.codex, 2)
    return Lco

def Mhalo_to_Lco_arbitrary(halos, coeffs, scatter=True):
    """
    halo mass to L_CO
    allows for utterly arbitrary models!
    coeffs:
        coeffs[0] is a function that takes halos as its only argument
        coeffs[1] is a boolean: do we need to calculate sfr or not?
        coeffs[2] is optional sigma_sfr
        coeffs[3] is optional argument that must almost never be invoked
        alternatively, if coeffs is callable, then assume we calculate sfr
            default sigma_sfr is 0.3 dex
    if sfr is calculated, it is stored as a halos attribute
    """
    sigma_sfr = 0.3
    bad_extrapolation = False
    if callable(coeffs):
        sfr_calc = True
        lco_func = coeffs
    else:
        lco_func, sfr_calc = coeffs[:2]
        if len(coeffs)>2:
            sigma_sfr = coeffs[2]
        if len(coeffs)>3:
            bad_extrapolation = coeffs[3]
    if sfr_calc:
        halos.sfr = Mhalo_to_sfr_Behroozi(halos, sigma_sfr, bad_extrapolation)
    return lco_func(halos)

def Mhalo_to_sfr_Behroozi(halos, sigma_sfr, bad_extrapolation=False):
    global sfr_interp_tab
    if sfr_interp_tab is None:
        sfr_interp_tab = get_sfr_table(bad_extrapolation)
    sfr = sfr_interp_tab.ev(np.log10(halos.M), np.log10(halos.redshift+1))
    if sigma_sfr > 0:
        sfr = add_log_normal_scatter(sfr, sigma_sfr, 1)
    return sfr

def get_sfr_table(bad_extrapolation=False):
    """
    LOAD SFR TABLE from Behroozi+13a,b
    Columns are: z+1, logmass, logsfr, logstellarmass
    Intermediate processing of tabulated data
    with option to extrapolate to unphysical masses
    """

    tablepath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    tablepath+= '/tables/sfr_behroozi_release.dat'
    dat_zp1, dat_logm, dat_logsfr, _ = np.loadtxt(tablepath, unpack=True)

    dat_logzp1 = np.log10(dat_zp1)
    dat_sfr    = 10.**dat_logsfr

    # Reshape arrays
    dat_logzp1  = np.unique(dat_logzp1)    # log(z), 1D
    dat_logm    = np.unique(dat_logm)    # log(Mhalo), 1D
    dat_sfr     = np.reshape(dat_sfr, (dat_logm.size, dat_logzp1.size))
    dat_logsfr  = np.reshape(dat_logsfr, dat_sfr.shape)

    # optional extrapolation to masses excluded in Behroozi+13
    if bad_extrapolation:
        from scipy.interpolate import SmoothBivariateSpline
        dat_logzp1_,dat_logm_ = np.meshgrid(dat_logzp1,dat_logm)
        badspl = SmoothBivariateSpline(dat_logzp1_[-1000<(dat_logsfr)],dat_logm_[-1000<(dat_logsfr)],dat_logsfr[-1000<(dat_logsfr)],kx=4,ky=4)
        dat_sfr[dat_logsfr==-1000.] = 10**badspl(dat_logzp1,dat_logm).T[dat_logsfr==-1000.]

    # Get interpolated SFR value(s)
    sfr_interp_tab = sp.interpolate.RectBivariateSpline(
                            dat_logm, dat_logzp1, dat_sfr,
                            kx=1, ky=1)
    return sfr_interp_tab


### HELPER FUNCTIONS FOR LUMINOSITY GENERATION FOR ANOTHER TRACER (USUALLY LYA)
def schechter(L, coeffs):
    """
    generic schechter function (Schechter 1976)
    coeffs are [Lstar, phistar, alpha, min lum, max lum]
    """
    
    [Lstar, phistar, alpha, _, _] = coeffs
    
    return (phistar / Lstar) * (L/Lstar)**alpha * np.exp(-L/Lstar)

def logschechter(L, coeffs):
    """
    generic schechter function (Schechter 1976), but in dlogL form intead of dL form
    coeffs are [Lstar, phistar, alpha, minlum, maxlum]
    """

    [Lstar, phistar, alpha, _, _] = coeffs 
    
    return np.log(10) * (phistar / Lstar) * (L/Lstar)**(alpha+1) * np.exp(-L/Lstar)

def doublepowerlaw(L, coeffs):
    """
    generic double power law function (ex. reference Umeda + 2025)
    coeffs are [Lstar, phistar, alpha, beta, min lum, max lum]
    """

    [Lstar, phistar, alpha, beta, _, _] = coeffs

    return (phistar/Lstar) / (np.power((L/Lstar), -alpha-1) + np.power((L/Lstar), -beta-1))

def logdoublepowerlar(L, coeffs):
    """
    generic double power law function but in dlogL form instead of dL form
    coeffs are [Lstar, phistar, alpha, beta, minlum, maxlum]
    """

    [Lstar, phistar, alpha, beta, _, _] = coeffs 
    
    return (phistar/Lstar) / (np.power((L/Lstar), -alpha) + np.power(L/Lstar), -beta)

def schechter_dpl(L, coeffs):
    """
    helper function to do a schechter+dpl luminosity function (LAE schechter component
    and bright AGN DPL component). this is the dn/dL version that has L* divided out 
    (used for abundance matching)
    coeffs are [Lstar_sch, phistar_sch, alpha_sch, 
      Lstar_dpl, phistar_dpl, alpha_dpl, beta_dpl, 
      min lum, max lum]
    """

    [Lstar_sch, phistar_sch, alpha_sch, Lstar_dpl, phistar_dpl, alpha_dpl, beta_dpl, Lmin, Lmax] = coeffs

    sch = schechter(L, [Lstar_sch, phistar_sch, alpha_sch, Lmin, Lmax])
    dpl = doublepowerlaw(L, [Lstar_dpl, phistar_dpl, alpha_dpl, beta_dpl, Lmin, Lmax])*Lstar_dpl/Lstar_sch

    return sch + dpl

def eboss_LEDE(Mg, coeffs):
    """
    'luminsotiy-evolving, density-evolving' luminosity function for quasars, used in eboss planning
    all from Palanque-Delabrouille et al. 2016, freezing to a specific redshift (but that redshift is
    above z=2.2, the 'pivot' redshift from this work)
    coeffs are Mgstarzp, logphistar0, alpha, beta, c1a, c1b, c2, c3, zp, zcent
    """

    [Mgstarzp, logphistar0, alpha, beta, c1a, c1b, c2, c3, zp, zcent, _, _] = coeffs

    logphistarz = logphistar0 + c1a * (zcent-zp) + c1b*(zcent-zp)**2
    phistarz = np.power(10, logphistarz)
    Mgstarz = Mgstarzp + c2*(zcent-zp)
    alphaz = alpha + c3*(zcent-zp)

    return phistarz / (np.power(10, 0.4*(alphaz+1)*(Mg-Mgstarz)) + np.power(10, 0.4*(beta+1)*(Mg-Mgstarz)))

### ABUNDANCE MATCHING FUNCTIONS ### 
def halomassfunction(halos, params):
    """
    calculates the number density of dM halos per logarithmic mass bin between log10M and log10(M+dM)
    then integrates that from each M to infinity to get the halo mass function
    """
    
    # NUMBER of halos with log masses between log10M and log10(M+dM)
    N, logMprime = np.histogram(np.log10(halos.M), bins=500)
    dlogMprime = logMprime[1:] - logMprime[:-1]
    logMprimecents = logMprime[:-1] + dlogMprime / 2
    
    # VOLUME of the simulation in cMpc**3
    cosmo = halos.cosmo
    volumeslice = cosmo.comoving_volume(params.z_f) - cosmo.comoving_volume(params.z_i)
    vol = (volumeslice / (4*np.pi*u.sr) * (params.fov_x * params.fov_y * u.deg**2)).to(u.Mpc**3)
    
    # number density dn/dlog10M
    dndlogM = N / vol    
    
    # integrated from M to infinity at each value of M
    dndlogMdlogM = dndlogM * dlogMprime
    intnM = []
    for i,M in enumerate(logMprimecents):
        intval = np.sum(dndlogMdlogM[i:])
        intnM.append(intval.value)

    intnM = np.array(intnM)
    
    return (logMprimecents, intnM)

def Mbol_to_Lsun(Mbol):
    """ convert a bolometric absolute magnitude to a luminosity in Lsun """
    return 10**(0.4*(4.74-Mbol))

def Lsun_to_Mbol(Lsun):
    """ convert a luminosity in Lsun to a bolometric absolute magnitude """
    return 4.74 - np.log10(Lsun) / 0.4

def abundancematch_mags(function, coeffs, halos, params):
    """
    calculate Lcat values for each halo by abundance-matching to luminosity function
    
    inputs:
    -------
        function: function
            describes shape of luminosity function (e.g. schechter function above)
        coeffs: array-like of floats
            the coefficients to be passed to the luminosity function for the particular case
            coeffs[-2] and coeffs[-1] are the min and max luminosity to integrate over, respectively
        halos: HaloCatalog object
            information about the DM halos (mostly need mass: halos.M)
        params: SimParameters object
            holds the parameters for the simulation run
    outputs:
    --------
    Adds to the HaloCatalog object:
        Lcat: (array-like of floats) the catalogue luminosities
    
    """
    # calculate the luminosity function
    # set up bins
    logLprime = np.linspace(np.abs(coeffs[-2]), np.abs(coeffs[-1]), 401)
    dlogLprime = logLprime[1:] - logLprime[:-1]
    logLprimecents = logLprime[:-1] + dlogLprime/2

    # luminosity function
    phiLarr = function(-logLprimecents, coeffs)
    phiLdL = phiLarr*dlogLprime

    # integrate over it
    intL = []
    for i,L in enumerate(logLprimecents):
        intLval = np.sum(phiLdL[i:])
        intL.append(intLval)
    intL = np.array(intL)

    # calculate the halo mass function
    logMprimecents, intnM = halomassfunction(halos, params)

    # interpolate between the two to get luminosities
    intMforM = np.interp(np.log10(halos.M), logMprimecents, intnM)
    LforintM = -np.interp(intMforM, np.flip(intL), np.flip(logLprimecents))

    LforintM_Lsun = Mbol_to_Lsun(LforintM)

    halos.Lcat = LforintM_Lsun

    return halos.Lcat, params


def abundancematch(function, coeffs, halos, params):
    """
    calculate Lcat values for each halo by abundance-matching to luminosity function
    
    inputs:
    -------
        function: function
            describes shape of luminosity function (e.g. schechter function above)
        coeffs: array-like of floats
            the coefficients to be passed to the luminosity function for the particular case
            coeffs[-2] and coeffs[-1] are the min and max luminosity to integrate over, respectively
        halos: HaloCatalog object
            information about the DM halos (mostly need mass: halos.M)
        params: SimParameters object
            holds the parameters for the simulation run
    outputs:
    --------
    Adds to the HaloCatalog object:
        Lcat: (array-like of floats) the catalogue luminosities
    
    """
    
    # calculate the luminosity function
    logLprime = np.log10(np.logspace(coeffs[-2], coeffs[-1], 101))
    dlogLprime = logLprime[1:] - logLprime[:-1]
    dLprime = 10**logLprime[1:] - 10**logLprime[:-1]
    logLprimecents = logLprime[:-1] + dlogLprime/2
    
    phiLarr = function(10**logLprimecents, coeffs)
    phiLdL = phiLarr*dLprime*u.erg/u.s

    # integrate over it
    intL = []
    for i,L in enumerate(logLprimecents):
        intLval = np.sum(phiLdL[i:])
        intL.append(intLval.value)

    intL = np.array(intL)
    
    # claculate the halo mass function
    logMprimecents, intnM = halomassfunction(halos, params)
    
    # interpolate between the two to get luminosities
    intMforM = np.interp(np.log10(halos.M), logMprimecents, intnM)
    LforintM = np.interp(intMforM, np.flip(intL), np.flip(logLprimecents))
    
    # convert to solar luminosities and store in the halo catalog
    halos.Lcat = 10**LforintM / 3.826e33 

    return halos.Lcat, params



@timeme
def Mhalo_to_Lcatalog(halos, params):
    """
    wRAPPER function to get L_catalog(M_halo) given a certain model <model>
    if adding your own model follow this structure,
    and simply specify the model to use in the parameter file
    will output halo luminosities in **L_sun**

    Parameters
    ----------
    halos : class
        Contains all halo information (position, redshift, etc..)
    params: SimParameters object
        Uses:
            catalog_model: (str) the model to use for calculating Lcats. Options are:
            catalog_coeffs : (array-like) coefficients to be passed to model, None for default coeffs
    """

    model = params.catalog_model

    dict = {'lya_chung':            Mhalo_to_LLya_Chung,
            'schechter':           Mhalo_to_Lcatalog_schechter,
            'schechter_amp':        Mhalo_to_Lcatalog_schechter_amp,
            'schechter_dpl':    Mhalo_to_Lcatalog_schechter_dpl,
            'eboss':            Mhalo_to_Lcatalog_eboss,
            'default':          Mhalo_to_Lcatalog_test1,
            'test2':          Mhalo_to_Lcatalog_test2
            }

    if model in dict.keys():
        return dict[model](halos, params)

    else:
        sys.exit('\n\n\tYour catalog model, '+model+', does not seem to exist\n\t\tPlease check src/halos_to_luminosity.py to add it\n\n')

def Mhalo_to_LLya_Chung(halos, params):
    """
    model to get Lya luminosities from halo SFR and redshift
    based on Chung et al. 2019 (arXiv:1809.04550)
    """

    try:
        coeffs = params.catalog_coeffs
    except AttributeError:
        coeffs = None 

    # SFR scatter based on Tony Li 2016 model
    sigma_sfr = 0.3

    # this model doesn't have named coefficients yet, so will always use defaults
    # ** edit to change that in future

    # Get Star formation rate
    if not hasattr(halos,'sfr'):
        halos.sfr = Mhalo_to_sfr_Behroozi(halos, sigma_sfr);
    
    z = halos.redshift
    sfr = halos.sfr

    # escape fraction
    fesc = (1+np.exp(-1.6*z + 5))**(-0.5) * (0.18 + 0.82 / (1 + 0.8*sfr**0.875))**2

    # Llya in erg/s
    Llya = 1.6e42 * sfr * fesc

    # convert to Lsun
    Llya = Llya / 3.826e33

    # zero out NaNs
    Llya[np.where(np.isnan(Llya))] = 0.0

    params.catdex = sigma_sfr #**** this is just a placeholder

    return Llya, params

def Mhalo_to_Lcatalog_schechter(halos, params):
    """ 
    wrapper to use a schechter function to generate catalog luminosities
    """

    # default to Lya luminosity function coefficients from Ouchi et al. 2020
    if not params.catalog_coeffs:
        params.catalog_coeffs =  [0.849e43, 3.9e-4, -1.8, 39, 45]

    Llya, params = abundancematch(schechter, params.catalog_coeffs, halos, params)
    return Llya, params

def Mhalo_to_Lcatalog_schechter_amp(halos, params):
    """ 
    wrapper to use a schechter function to generate catalog luminosities
    this one is explicitly weighted with a constant value passed as the first catalog coefficient
    (used for simple amplitude modeling)
    if that constant value is one it's identical to the function above
    """

    # default to Lya luminosity function coefficients from Ouchi et al. 2020
    if not params.catalog_coeffs:
        params.catalog_coeffs =  [1, 0.849e43, 3.9e-4, -1.8, 39, 45]

    Llya, params = abundancematch(schechter, params.catalog_coeffs[1:], halos, params)
    Llya *= params.catalog_coeffs[0]
    return Llya, params

def Mhalo_to_Lcatalog_schechter_dpl(halos, params):
    """
    wrapper to use a schechter+dpl function to generate catalog luminosities
    the dpl is for the AGN component contributing to the lya luminosity function
    """

    # default to Lya luminosity function coefficients from Zhang et al. 2021
    if not params.catalog_coeffs:
        params.catalog_coeffs = [10**42.86, 10**-3.39, -1.69, 10**44.68, 10**-5.96, -1.23, -3.06, 39, 45]

    Llya, params = abundancematch(schechter_dpl, params.catalog_coeffs, halos, params)
    return Llya, params

def Mhalo_to_Lcatalog_eboss(halos, params):
    """ 
    wrapper to use the eBOSS luminosity function from Palanque-Delabrouille et al 2016
    https://www.aanda.org/articles/aa/pdf/2016/03/aa27392-15.pdf
    (specifically the high-redshift version applicable to the COMAP redshift range)
    **this one will output luminsoities in absolute g-band magnitudes, normalized to z=2,
    with a reddening correction (eqn 4 of the paper for how to do that)
    """

    # default to Lya luminosity function coefficients from Ouchi et al. 2020
    if not np.any(params.catalog_coeffs):
        params.catalog_coeffs =  [-26.71, -5.93, -3.89, -1.47, -0.46, -0.06, -0.14, 0.32, 2.2, 2.8, -15, -29]

    Mg, params = abundancematch_mags(eboss_LEDE, params.catalog_coeffs, halos, params)
    return Mg, params


def Mhalo_to_Lcatalog_test1(halos, params):
    """
    test model for assigning lums of an arbitrary tracer to halos based on M_halo
    """

    try:
        coeffs = params.catalog_coeffs
    except AttributeError:
        coeffs = None


    if coeffs is None:
        # default to scaled version of UM+COLDz+COPSS model from Chung+22 ***
        coeffs = (
            -2, -0.5, 11, 13, 0.5)
        halos.catalog_coeffs = coeffs
        A, B, logC, logM, sigma = coeffs
    else:
        A,B,logC,logM,sigma = coeffs
        halos.catalog_coeffs = coeffs

    Mh = halos.M

    C = 10**logC
    M = 10**logM

    Lprime = C / ((Mh/M)**A + (Mh/M)**B)
    Lcatalog = 4.9e-5 * Lprime

    params.catdex = sigma

    return Lcatalog, params

def Mhalo_to_Lcatalog_test2(halos, params):
    """
    test model for assigning lums of an arbitrary tracer to halos based on M_halo
    """

    try:
        coeffs = params.catalog_coeffs
    except AttributeError:
        coeffs = None


    if coeffs is None:
        # default to wildly different version of UM+COLDz+COPSS model from Chung+22 ***
        coeffs = (
            0.5, 2, 11, 12, 0.5)
        halos.catalog_coeffs = coeffs
        A, B, logC, logM, sigma = coeffs
    else:
        A,B,logC,logM,sigma = coeffs
        halos.catalog_coeffs = coeffs

    Mh = halos.M

    C = 10**logC
    M = 10**logM

    Lprime = C / ((Mh/M)**A + (Mh/M)**B)
    Lcatalog = 4.9e-5 * Lprime

    params.catdex = sigma

    return Lcatalog

### ADD RANDOM LOGNORMAL SCATTER
def add_log_normal_scatter(data,dex,seed):
    """
    Return array x, randomly scattered by a log-normal distribution with sigma=dexscatter.
    [via @tonyyli - https://github.com/dongwooc/imapper2]
    Note: scatter maintains mean in linear space (not log space).
    """
    if np.any(dex<=0):
        return data
    # Calculate random scalings
    sigma       = dex * 2.302585 # Stdev in log space (DIFFERENT from stdev in linear space), note: ln(10)=2.302585
    mu          = -0.5*sigma**2

    # Set standard seed so changing minimum mass cut
    # does not change the high mass halos
    np.random.seed(seed*13579)
    randscaling = np.random.lognormal(mu, sigma, data.shape)
    xscattered  = np.where(data > 0, data*randscaling, data)

    return xscattered

def add_co_tracer_dependant_scatter(halos, rho, codex, catdex, seed):
    """
    add correlated scatter between the CO luminosities and the other tracer luminosities
    use a passed covariance matrix to generate
    inputs:
    -------
        halos: HaloCatalog object
            stores all of the information for your catalog of DM halos
        rho: (float, -1 < rho < 1)
            correlation coefficient for relating the scatter in the two luminosity values
        codex: float
            lognormal scatter in the CO luminosities in dex
        catdex: float
            lognormal scatter in the catalog luminosities in dex
        seed:
            random generator seed for selecting the scatter values
    outputs:
        halos: HaloCatalog object
            multiplies scatter in to halos.Lco and halos.Lcat
    """
    if np.any(np.logical_or(codex <= 0, catdex <= 0)):
        print('passed a negative dex value. not scattering')
        return halos

    # set up a numpy random number generator
    scalerng = np.random.default_rng(seed=seed)

    # parameters for the CO distribution
    sigmaco = codex * 2.30285 # stdev in log space
    muco = -0.5*sigmaco**2

    # parameters for the catalog tracer distribution
    sigmatr = catdex * 2.30285
    mutr = -0.5*sigmatr**2

    # mean and convariance matrix for the joint distribution
    mean = [0,0]
    cov = [[sigmaco**2, sigmaco*sigmatr*rho],
           [sigmaco*sigmatr*rho, sigmatr**2]]
    halos.cov = cov

    # LINEAR normal scalings for co and the halo tracer
    coscale, trscale = scalerng.multivariate_normal(mean, cov, size=len(halos.Lco)).T

    # change those into lognormal scalings (output of this would be the same as pulling from
    # np.random.lognormal for a single variable)
    logscaleco = np.exp(coscale + muco)
    logscaletr = np.exp(trscale + mutr)

    # slap scalings onto existing catalog and co luminosities
    halos.Lco = halos.Lco*logscaleco
    halos.Lcat = halos.Lcat*logscaletr

    return halos
