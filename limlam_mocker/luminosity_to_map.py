
from __future__ import absolute_import, print_function
import numpy as np
import os
import scipy as sp
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from astropy.convolution import convolve, Gaussian2DKernel
import copy
from  .tools import *


"""
Functions to take a catalogue of DM halos (with 3d positions, masses, luminosities, etc)
and turn them into a 3D intensity map, complete with mock observational effects
"""

@timeme
class SimMap():
    """
    designer class to hold simulated maps and their metadata

    inputs:
    -------
        params: SimParameters object
            holds all of the simulation parameters for the map
        inputfile: str, optional
            path to file containing a saved data cube generated using this code
            if passed, will load that data cube into a map object, otherwise will create
            a blank map object
    """
    def __init__(self, params, inputfile=None):
        # if inputfile is passed, load that data cube in, otherwise start an empty object
        if inputfile:
            self.from_file(inputfile, params)
        else:
            self.setup_empty(params)


    def setup_empty(self, params):
        """
        Adds input parameters to be kept by the map class and gets map details
        (this is just params_to_mapinst from the og limlam_mocker)

        inputs:
        -------
            params: SimParameters object. 
                see param_argparser.py for descriptions of all the parameters used here
        """

        self.nmaps  = int(params.nmaps)
        self.fov_x  = float(params.fov_x)
        self.fov_y  = float(params.fov_y)
        self.npix_x = int(params.npix_x)
        self.npix_y = int(params.npix_y)
        self.nu_i   = float(params.nu_i)
        self.nu_f   = float(params.nu_f)
        self.nu_rest= float(params.nu_rest)
        self.z_i    = self.nu_rest/self.nu_i - 1
        self.z_f    = self.nu_rest/self.nu_f - 1

        # get arrays describing the final intensity map to be output
        # map sky angle dimension
        self.pix_size_x = self.fov_x/self.npix_x
        self.pix_size_y = self.fov_y/self.npix_y

        # pixel size to convert to brightness temp
        self.Ompix = (self.pix_size_x*np.pi/180)*(self.pix_size_y*np.pi/180)

        self.pix_binedges_x = np.linspace(-self.fov_x/2,self.fov_x/2,self.npix_x+1)
        self.pix_binedges_y = np.linspace(-self.fov_y/2,self.fov_y/2,self.npix_y+1)

        self.pix_bincents_x =  0.5*(self.pix_binedges_x[1:] + self.pix_binedges_x[:-1])
        self.pix_bincents_y =  0.5*(self.pix_binedges_y[1:] + self.pix_binedges_y[:-1])

        # map frequency dimension
        # use linspace to ensure nmaps channels
        self.nu_binedges = np.linspace(self.nu_i,self.nu_f,self.nmaps+1)
        self.dnu         = np.abs(np.mean(np.diff(self.nu_binedges)))
        self.nu_bincents = self.nu_binedges[:-1] - self.dnu/2

        # get full map volume
        x,y,z = self.pix_binedges_x, self.pix_binedges_y, self.nu_binedges
        zco = params.cosmo.comoving_distance(self.nu_rest/z-1)
        # assume comoving transverse distance = comoving distance
        #     (i.e. no curvature)
        avg_ctd = np.mean(zco)
        xco = x/(180)*np.pi*avg_ctd
        yco = y/(180)*np.pi*avg_ctd
        dxco, dyco, dzco = [np.abs(np.mean(np.diff(d))) for d in (xco, yco, zco)]
        self.voxcovol = dxco*dyco*dzco
        self.totalcovol = np.ptp(xco)*np.ptp(yco)*np.ptp(zco)

        return 

    def from_file(self, inputfile, params):
        """
        load in map object from a file previously saved by this code

        inputs:
        -------
            inputfile: str
                path to file containing map data
            params: SimParameters object
                pulls most of the parameters from the saved file itself, but will use
                nu_i, nuf, nu_rest, nmaps, and cosmo
        """
        
        # parameters saved in the code
        with np.load(inputfile) as f:
            self.fov_x = float(f['fov_x'])
            self.fov_y = float(f['fov_y'])
            self.pix_size_x = float(f['pix_size_x'])
            self.pix_size_y = float(f['pix_size_y'])
            self.npix_x = int(f['npix_x'])
            self.npix_y = int(f['npix_y'])
            self.pix_bincents_x = f['map_pixel_ra']
            self.pix_bincents_y = f['map_pixel_dec']
            self.nu_bincents = f['map_frequencies']
            self.map = f['map_cube']
            self.catmap = f['cat_cube']
            self.hit = f['cat_hits']

        # other map metadata from params object
        self.nu_i   = float(params.nu_i)
        self.nu_f   = float(params.nu_f)
        self.nu_rest= float(params.nu_rest)
        self.nmaps = params.nmaps
        self.z_i    = self.nu_rest/self.nu_i - 1
        self.z_f    = self.nu_rest/self.nu_f - 1

        self.nu_binedges = np.linspace(self.nu_i,self.nu_f,self.nmaps+1)
        self.dnu         = np.abs(np.mean(np.diff(self.nu_binedges)))

        # pixel size to convert to brightness temp
        self.Ompix = (self.pix_size_x*np.pi/180)*(self.pix_size_y*np.pi/180)

        # other way of binning pixels 
        self.pix_binedges_x = np.linspace(-self.fov_x/2,self.fov_x/2,self.npix_x+1)
        self.pix_binedges_y = np.linspace(-self.fov_y/2,self.fov_y/2,self.npix_y+1)

        # get full map volume
        x,y,z = self.pix_binedges_x, self.pix_binedges_y, self.nu_binedges
        zco = params.cosmo.comoving_distance(self.nu_rest/z-1)
        # assume comoving transverse distance = comoving distance
        #     (i.e. no curvature)
        avg_ctd = np.mean(zco)
        xco = x/(180)*np.pi*avg_ctd
        yco = y/(180)*np.pi*avg_ctd
        dxco, dyco, dzco = [np.abs(np.mean(np.diff(d))) for d in (xco, yco, zco)]
        self.voxcovol = dxco*dyco*dzco
        self.totalcovol = np.ptp(xco)*np.ptp(yco)*np.ptp(zco)

        return


    def copy(self):
        """
        return a deep copy of the SimMap object (to avoid overwriting confusion)
        """
        return copy.deepcopy(self)


    def mockmapmaker(self, halos, params):
        """
        wrapper function for all forms of mock mapmaking from the generated
        luminosity catalog

        most of this is stolen from the original limlam_mocker code:
        Lco_to_map, Lco_to_map_doppler, etc.
        CO beam fitting added by DD
        """

        """ SET UP """
        ### Calculate line freq from redshift
        halos.nu  = self.nu_rest/(halos.redshift+1)
        try:
            halos.nucat = self.nu_rest/(halos.zcat+1)
        except AttributeError:
            halos.offset_velocities(params)
            halos.nucat = self.nu_rest/(halos.zcat+1)

        # Transform from Luminosity to Temperature (uK)
        # ... or to flux density (Jy/sr)
        # ... or leave as mass
        match params.units:
            case 'intensity':
                if params.verbose: print('\n\tcalculating halo intensities')
                halos.Tco = I_line(halos, self)
            case 'temperature':
                if params.verbose: print('\n\tcalculating halo temperatures')
                halos.Tco = T_line(halos, self)
            case 'mass':
                if params.verbose: print('\n\tpreserving halo masses')
                halos.Tco = halos.M
            case _:
                if params.verbose: print('\n\tdefaulting to halo temperatures')
                halos.Tco = T_line(halos, self)

        # bin halos by velocity into bincount bins (will be smoothed in those chunks)
        if 1==params.bincount or not params.freqbroaden:
            subsets = [halos]
        else:
            try:
                binattr_val = getattr(halos, 'vbroaden')
            except AttributeError:
                halos.get_velocities(params)
                binattr_val = getattr(halos, 'vbroaden')
            attr_ranges = np.linspace(min(binattr_val)*(1-1e-16),max(binattr_val), params.bincount+1)
            verb = params.verbose
            params.verbose = False
            subsets = [halos.attrcut_subset('vbroaden', v1, v2, params)
                            for v1,v2 in zip(attr_ranges[:-1], attr_ranges[1:])]
            params.verbose = verb

        # set up finer binnning in RA, DEC, frequency
        # (if oversampling isn't necessary these will just be equal to the regular sampling)
        # check that oversampling isn't necessary and xrefine is appropriate
        if params.beambroaden == False:
            params.xrefine = 1
        if params.freqbroaden == False:
            params.freqrefine = 1
        bins3D_fine = [np.linspace(min(self.pix_binedges_x),
                                   max(self.pix_binedges_x),
                                   len(self.pix_binedges_x[1:])*params.xrefine+1),
                       np.linspace(min(self.pix_binedges_y),
                                   max(self.pix_binedges_y),
                                   len(self.pix_binedges_y[1:])*params.xrefine+1),
                       np.linspace(min(self.nu_binedges),
                                   max(self.nu_binedges),
                                   len(self.nu_binedges[1:])*params.freqrefine+1)]

        dx_fine = np.mean(np.diff(bins3D_fine[0]))
        dy_fine = np.mean(np.diff(bins3D_fine[1]))
        dnu_fine = np.mean(np.diff(bins3D_fine[-1]))

        """ MAKE MAP OF LIM TRACER """
        # empty array to hold the map
        maps = np.zeros((len(self.pix_bincents_x)*params.xrefine,
                         len(self.pix_bincents_y)*params.xrefine,
                         len(self.nu_bincents)))
        
        # if told to, apply line broadening
        if params.freqbroaden:
            # pull velocity values, or create them if they don't exist
            try:
                velocities = getattr(halos, 'vbroaden')
            except AttributeError:
                halos.get_velocities(params)

            # pull the relevant parameters out of the params object
            # number of velocity bins to use
            bincount = params.bincount
            # function to turn velocity halo attributes into a line width (in observed freq space)
            # default is vmax/c times observed frequency (if set to None)
            fwhmfunc = params.fwhmfunction
            # number of bins by which to oversample in frequency
            freqrefine = params.freqrefine
            # function used to broaden halo emission based on linewidth
            filterfunc = params.filterfunc
            # if true, will do a fast convolution thing (??)
            lazyfilter = params.lazyfilter

            if fwhmfunc is None:
                # a fwhmfunc is needed to turn halo attributes into a line width
                #   (in observed frequency space)
                # default fwhmfunc based on halos is vmax/c times observed frequency
                fwhmfunc = lambda h:h.nu*h.vbroaden/299792.458

            # for each velocity bin, convolve all halos with a kernel of the same width and then add onto the map
            for i,sub in enumerate(subsets):
                if sub.nhalo < 1: continue;
                # make the fine-resolution map
                maps_fine = np.histogramdd( np.c_[sub.ra, sub.dec, sub.nu],
                                              bins    = bins3D_fine,
                                              weights = sub.Tco )[0]
                # convert velocities to fwhma
                if callable(fwhmfunc):
                    sigma = 0.4246609*np.nanmedian(fwhmfunc(sub)) # in freq units (GHz)
                else: # hope it's a number
                    sigma = 0.4246609*fwhmfunc # in freq units (GHz)
                if sigma > 0:
                    if lazyfilter:
                        filteridx = np.where(np.any(maps_fine, axis=-1))
                        maps_fine[filteridx] = filterfunc(maps_fine[filteridx], sigma/dnu_fine)
                    else:
                        maps_fine = filterfunc(maps_fine, sigma/dnu_fine)
                # collapse back down to the end-stage frequency sampling
                maps += np.sum(maps_fine.reshape((maps_fine.shape[0], maps_fine.shape[1], -1, freqrefine)), axis=-1)
                if params.verbose:
                    print('\n\tsubset {} / {} complete'.format(i,len(subsets)))
            if (params.units=='intensity'):
                maps /= self.Ompix
            # flip back frequency bins and store in object
            self.map = maps[:,:,::-1]

        else:
            # bin into map (in RA, DEC, NU_obs)
            if params.verbose: print('\n\tBinning halos into map')
            maps, edges = np.histogramdd( np.c_[halos.ra.flatten(), halos.dec.flatten(), halos.nu.flatten()],
                                          bins    = bins3D_fine,
                                          weights = halos.Tco )
            if (params.units=='intensity'):
                maps /= self.Ompix
            # flip back frequency bins
            self.map = maps[:,:,::-1]

        if params.beambroaden:
            # smooth by the primary beam

            # come up with a convolution kernel approximating the beam if one isn't already passed
            if not params.beamkernel:
                # number of refined pixels corresponding to the fwhm in arcminutes
                std = params.beamfwhm / (2*np.sqrt(2*np.log(2))) / 60 # standard deviation in degrees
                std_pix = std / dx_fine

                beamkernel = Gaussian2DKernel(std_pix)

            if params.verbose:
                print('\nsmoothing by synthesized beam: {} channels total'.format(maps.shape[-1]))

            # iterate through channels, smoothing each by the beam kernel
            smoothsimlist = []
            for i in range(maps.shape[-1]):
                smoothsimlist.append(convolve(maps[:,:,i], beamkernel))
                if params.verbose:
                    if i%100 == 0:
                        print('\n\t done {} of {} channels'.format(i, maps.shape[-1]))

            maps_sm_fine = np.stack(smoothsimlist, axis=-1)
            print(maps_sm_fine.shape)

            # rebin
            mapssm = np.sum(maps_sm_fine.reshape((params.npix_x, params.xrefine,
                                                  params.npix_y, params.xrefine, -1)), axis=(1,3))

            if (params.units=='intensity'):
                mapssm/= self.Ompix
            # flip back frequency bins
            self.map = mapssm[:,:,::-1]

            
        """ MAKE A MAP OF THE LUMINOSITY VALUES FOR THE CATALOG TRACER 
            (for cross-correlating fluctuation cubes) """
        # Transform from Luminosity to Temperature (uK)
        # ... or to flux density (Jy/sr)
        if hasattr(halos, 'Lcat'):
            # do unit conversions if necessary
            if (params.units=='intensity'):
                if params.verbose: print('\n\tcalculating halo intensities')
                halos.Tcat = I_line(halos, self, attribute='Lcat')
            elif (params.units=='temperature'):
                if params.verbose: print('\n\tcalculating halo temperatures')
                halos.Tcat = T_line(halos, self, attribute='Lcat')
            else:
                if params.verbose: print('\n\tpreserving halo masses')
                halos.Tcat = halos.M 
                
                
            # flip frequency bins because np.histogram needs increasing bins
            bins3D = [self.pix_binedges_x, self.pix_binedges_y, self.nu_binedges[::-1]]

            # bin in RA, DEC, NU_obs
            if params.verbose: print('\n\tBinning catalog halos into map')
            catmaps, edges = np.histogramdd( np.c_[halos.ra, halos.dec, halos.nucat],
                                             bins    = bins3D,
                                             weights = halos.Tcat )
            if (params.units=='intensity'):
                catmaps/= self.Ompix 
            # flip back frequency bins
            self.catmap = catmaps[:,:,::-1]

        """ ADD OTHER MOCK THINGS TO MAP """
        if params.add_comap_noise:
            # add random radiometer noise (using integration time and a 19-feed focal plane array)
            self.add_random_comap_noise(params)
            if params.verbose:
                print('\nAdding COMAP radiometer noise with {} hr integration time'.format(params.noise_int_time))

        if params.add_foreground:
            # add mock spectral line contamination to the maps (from interloping foregrounds or backgrounds)
            self.add_foreground(params)
            if params.verbose:
                print('\nAdding foreground emission using permutation {} and a scale factor of {}'.format(params.fg_permutation,
                                                                                                          params.fg_scalefactor))
                
    def subtract_mean(self):
        """
        subtract off the mean in each channel and each spaxel (simulates the low-pass filter in an actual map)
        order agrees with the order used by the actual COMAP pipline (ie per-channel and then per-spaxel)
        
        inputs:
        -------
            None, uses self.map directly
        outputs:
        --------
            self.meanvals: the mean value in each spectral channel and each spaxel, summed together to create a data cube
                            (add it back on to the map to get the original one)
        """

        ogmap = self.map 
        spaceres = self.map.shape[0]
        specres = self.map.shape[2]

        specmean = np.nanmean(ogmap, axis=(0,1))
        specmean = np.tile(specmean, (spaceres,spaceres,1))

        smap = ogmap - specmean

        spacemean = np.nanmean(smap, axis=2)
        spacemean = np.tile(spacemean.T, (specres,1,1)).T

        ssmap = smap - spacemean 

        self.map = ssmap 
        self.meanvals = spacemean + specmean 


    def add_random_comap_noise(self, params):
        """
        add randomly-generated noise (in uK) to the CO map. doesn't account for any systematics or 
        asymmetries in the noise response/field coverage 

        inputs:
        -------
            params.noise_int_time: integration time in hours to simulate (radiometer) noise
            params.nfeeds: number of focal plane array feeds (default 19 for COMAP)
                (could also use this parameter for other factors which improve sensitivity by root(N) -- 
                 number of dishes, etc.)
            params.noise_seed: seed for the random number generator
        outputs:
        --------
            self.mapnoise: noise map in uK (just subtract it off the map to get the signal-only cube again)
        """
        # assuming an even spread, fraction of total integration time spent on each voxel
        voxfrac = (self.pix_size_x*self.pix_size_y)/(self.fov_x * self.fov_y)

        # calculate the noise level from the integration time
        sigma = params.Tsys*1e6 / np.sqrt(2*params.nfeeds*params.noise_int_time*3600*voxfrac*self.dnu*1e9)

        # generate noise map
        rng = np.random.default_rng(params.noise_seed)
        noise = rng.normal(loc=0., scale=sigma, size=self.map.shape)
        self.mapnoise = noise

        # add it in
        self.map = self.map + noise
        self.sigma = sigma

    def add_foreground(self, params):
        """
        quick and dirty way to add foreground/background emission. Note that this does NOT account for any changes in 
        the large scale structure with reddshift, which may be important for cross-correlations, etc
        
        inputs:
        -------
            params.fg_permutation: integer pointing to how the actual map will be permuted to turn it into a foreground map
            params.fg_scalefactor: how much fainter the foregrounds should be than the actual map
        outputs:
        --------
            self.foregroundmap: foreground map in uK (just subtract it off the map to get the signal-only cube again)
        """

        maparr = self.map 
        i = params.fg_permutation

        # permute the map along an axis of symmetry to make a foregound map that doens't line up with the 'actual' LSS
        if np.isin(i, (0,1,2)):
            permmap = np.rot90(maparr, i+1, axes=(0,1))
        elif i == 3:
            permmap = np.flip(maparr, axis=0)
        elif i == 4:
            permmap = np.flip(maparr, axis=1)
        elif i == 5:
            permmap = np.flip(maparr, axis=2)
        elif np.isin(i, (6,7,8)):
            permmap = np.flip(np.rot90(maparr, i+1, axes=(0,1)), axis=2)
        elif i == 9:
            permmap = np.flip(np.flip(maparr, axis=0), axis=2)
        elif i == 10:
            permmap = np.flip(np.flip(maparr, axis=1), axis=2)

        # scale the foreground map by the input factor
        fgmap = permmap / params.fg_scalefactor
        self.foregroundmap = fgmap 

        # add it in
        self.map = self.map + fgmap



    @timeme
    def write(self, params):
        """
        save 3D data cube in .npz format, including map header information

        inputs:
        -------
            params.map_output_file: path to file to save to
        """
        if params.verbose: print('\n\tSaving Map Data Cube to\n\t\t', params.map_output_file)
        print('saving catalog!!')
        try:
            hits = self.hit
        except AttributeError:
            hits = 0.
        try:
            sigma = self.sigma 
        except AttributeError: 
            sigma = 1
        np.savez(params.map_output_file,
                 fov_x=self.fov_x, fov_y=self.fov_y,
                 pix_size_x=self.pix_size_x, pix_size_y=self.pix_size_y,
                 npix_x=self.npix_x, npix_y=self.npix_y,
                 map_pixel_ra    = self.pix_bincents_x,
                 map_pixel_dec   = self.pix_bincents_y,
                 map_frequencies = self.nu_bincents,
                 map_cube        = self.map,
                 cat_cube        = self.catmap,
                 cat_hits        = hits,
                 sigma           = sigma)

        return


### UNIT CONVERSION FUNCTIONS
def I_line(halos, map, attribute='Lco'):
    '''
     calculates I_line = L_line/4/pi/D_L^2/dnu
     output units of Jy/sr
     assumes L_line in units of L_sun, dnu in GHz

     then 1 L_sun/Mpc**2/GHz = 4.0204e-2 Jy/sr
    '''
    lum = getattr(halos, attribute)
    convfac = 4.0204e-2 # Jy/sr per Lsol/Mpc/Mpc/GHz
    Ico     = convfac * lum/4/np.pi/halos.chi**2/(1+halos.redshift)**2/map.dnu

    return Ico

def T_line(halos, map, attribute='Lco'):
    """
    The line Temperature in Rayleigh-Jeans limit
    T_line = c^2/2/kb/nuobs^2 * I_line

     where the Intensity I_line = L_line/4/pi/D_L^2/dnu
        D_L = D_p*(1+z), I_line units of L_sun/Mpc^2/Hz

     T_line units of [L_sun/Mpc^2/GHz] * [(km/s)^2 / (J/K) / (GHz) ^2] * 1/sr
        = [ 3.48e26 W/Mpc^2/GHz ] * [ 6.50966e21 s^2/K/kg ]
        = 2.63083e-6 K = 2.63083 muK

    returns Tco in MICROKELVIN
    """
    lum = getattr(halos, attribute)
    convfac = 2.63083
    Tco     = 1./2*convfac/halos.nu**2 * lum/4/np.pi/halos.chi**2/(1+halos.redshift)**2/map.dnu/map.Ompix

    return Tco


