from __future__ import absolute_import, print_function
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u 
import astropy.constants as const
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
import copy
from  .tools import *

"""
Functions for loading and manipulating a catalogue of simulated halos
"""


class HaloCatalog():
    """
    Designer class for holding and manipulating a catalog of simulated halos.
    Used both for holding DM halos to create LIM cubes and for holding a mock galaxy catalogue
    """

    def __init__(self, params, inputfile=None, load_all=False):
        if inputfile:
            self.load(inputfile, params)
            self.cull(params)
        else:
            pass

    def copy(self):
        return copy.deepcopy(self)

    @timeme
    def load(self, filein, params):
        """
        Load peak-patch halo catalog into halos class

        inputs
        -------
        filein : string
            catalog file output from a peak-patch run
        params : SimParameters object
            Contains all parameter information, will load cosmology information into this (Omega_i, sigme_8, etc)
        """

        halo_info  = np.load(filein, allow_pickle=True)
        if params.verbose: print("\thalo catalog contains:\n\t\t", halo_info.files)

        #get cosmology from halo catalog
        params_dict    = halo_info['cosmo_header'][()]
        Omega_M  = params_dict.get('Omega_M')
        Omega_B  = params_dict.get('Omega_B')
        hvalue        = params_dict.get('h'      )

        self.cosmo = FlatLambdaCDM(H0=100*hvalue * u.km/(u.Mpc*u.s), Om0 = Omega_M, Ob0 = Omega_B)

        self.Omega_L  = params_dict.get('Omega_L')
        self.ns = params_dict.get('ns')
        self.sigma8 = params_dict.get('sigma8')

        cen_x_fov  = params_dict.get('cen_x_fov', 0.) #if the halo catalog is not centered along the z axis
        cen_y_fov  = params_dict.get('cen_y_fov', 0.) #if the halo catalog is not centered along the z axis

        self.M          = halo_info['M']     # halo mass in Msun
        self.x_pos      = halo_info['x']     # halo x position in comoving Mpc
        self.y_pos      = halo_info['y']     # halo y position in comoving Mpc
        self.z_pos      = halo_info['z']     # halo z position in comoving Mpc
        self.vx         = halo_info['vx']    # halo x velocity in km/s
        self.vy         = halo_info['vy']    # halo y velocity in km/s
        self.vz         = halo_info['vz']    # halo z velocity in km/s
        self.redshift   = halo_info['zhalo'] # observed redshift incl velocities
        self.zformation = halo_info['zform'] # formation redshift of halo

        self.nhalo = len(self.M)

        self.chi        = np.sqrt(self.x_pos**2+self.y_pos**2+self.z_pos**2)
        self.ra         = np.arctan2(-self.x_pos,self.z_pos)*180./np.pi - cen_x_fov
        self.dec        = np.arcsin(  self.y_pos/self.chi  )*180./np.pi - cen_y_fov

        assert np.max(self.M) < 1.e17,             "Halos seem too massive"
        assert np.max(self.redshift) < 4.,         "need to change max redshift interpolation in tools.py"

        if params.verbose: print('\n\t%d halos loaded' % self.nhalo)

    @timeme 
    def load_luminosities(self, inputfile, params):
        """
        load in halo luminosities generated in a previous run of the code
        #*** need to store more metainfo so these match
        """
        with np.load(inputfile) as file:
            # test to make sure that at least the lengths of the arrays are the same
            assert self.nhalo == len(file['Lco']),    "Number of halos in the file doesn't match positions"

            self.Lco = file['Lco']
            self.Lcat = file['Lcat']
            if file['vhalo'][0] > 0:
                self.vbroaden = file['vhalo']


    @timeme
    def cull(self, params):
        """
        crops the halo catalog to only include desired halos (gets rid of those out of the 
        redshift range, below the minimum mass, etc), will perform the cut in-place
        
        inputs
        ------
        params: SimParameters object
            contains all parameter information. this function will call
            z_i, z_f, min_mass, mass_cutoff, fov_x, fov_y, verbose
        """

        # convert the limits in frequency to limits in redshift
        params.z_i = freq_to_z(params.nu_rest, params.nu_i)
        params.z_f = freq_to_z(params.nu_rest, params.nu_f)

        # check that the limits are the right way round
        if params.z_i > params.z_f:
            tz = params.z_i
            params.z_i = params.z_f
            params.z_f = tz

        # relevant conditions:
        goodidx = (self.M > params.min_mass) * \
                  (self.M < params.mass_cutoff) * \
                  (self.redshift >= params.z_i) * \
                  (self.redshift <= params.z_f) * \
                  (np.abs(self.ra) <= params.fov_x/2) * \
                  (np.abs(self.dec) <= params.fov_y/2)

        goodidx = np.where(goodidx)[0]

        self.indexcut(goodidx, in_place=True)

        if params.verbose: print('\n\t%d halos remain after mass/map cut' % self.nhalo)

        # sort halos by mass, so fluctuations in luminosity
        # are the same with any given mass cut
        sortidx = np.argsort(self.M)[::-1]
        self.indexcut(sortidx, in_place=True)

    ### MESS WITH THE OBSERVATIONAL PROPERTIES OF THE CATALOG
    def get_velocities(self, params):
        """
        assigns each halo a rotation velocity based on its DM mass
        adds the velocity value to self.vbroaden and also returns it
        uses params.velocity_attr

        inputs
        ------
        params: SimParameters object
            contains all parameter information. this function will use velocity_attr, which has options:

            if 'vvir', just calculates the virial velocity
            if 'vvirincli', calculates the virial velocity and muliplies it by sin(i), i a randomly generated 
                inclination angle, to simulate the effects of inclination on line broadening
            if 'vvirincli_scaled', will scale the virial velocty by an input parameter ('vvirscalefactor').
            if 'vvir_cutoff', will cut the velocities off that are above some value 'vvircutoff'

        outputs
        -------
        returns nothing, but adds to the following attributes:
        self.sin_i: array of len nhalo
            sin(i) for randomly-chosen inclinations assigned to each halo
        self.vvir: array of len nhalo
            the virial velocity (claculated from the halo mass and redshift) of each halo
        self.vbroaden: array of len nhalo
            the 'observed' velocity of each halo, calculated using the passed method
        params.filterfunc: scipy.ndimage filter kernel
            function to generate an arbitrary one-dimensional gaussian filter
        """
        vvir = lambda M,z:35*(M*self.cosmo.H(z).value/1e10)**(1/3) # km/s

        if params.velocity_attr == 'vvirincli':
            # Calculate doppler parameters
            self.sin_i = np.sqrt(1-np.random.uniform(size=self.nhalo)**2)
            self.vvir = vvir(self.M, self.redshift) # / 2 #****
            self.vbroaden = self.vvir*self.sin_i/0.866

        elif params.velocity_attr == 'vvirincli_scaled':
            # scale the virial velocity by an input parameter for testing
            self.sin_i = np.sqrt(1-np.random.uniform(size=self.nhalo)**2)
            self.vvir = vvir(self.M, self.redshift)
            self.vbroaden = self.vvir*self.sin_i/0.866/params.vvirscalefactor

        elif params.velocity_attr == 'vvirincli_cutoff':
            # cut off the virial velocity at some cutoff value (so it's not 
            # overestimated for the most massive halos)
            self.sin_i = np.sqrt(1-np.random.uniform(size=self.nhalo)**2)
            self.vvir = vvir(self.M, self.redshift)
            vbroaden = self.vvir*self.sin_i/0.866
            # taking the remainder around the boundary condition (to avoid a big bump right there)
            ofidx = np.where(vbroaden > params.vvircutoff)
            vbroaden[ofidx] = vbroaden[ofidx] % params.vvircutoff
            self.vbroaden = vbroaden

        elif params.velocity_attr == 'vmpeak':
            # universemachine v_m,peak velocity NOT scaled by inclination
            a = 1 / (1+self.redshift)
            M200 = (1.64e12)/((a/0.378)**-0.142 + (a/0.378)**-1.79)
            vmpeak = 200 * (self.M / M200)**0.3
            rng = np.random.default_rng(12345)
            scvmpeak = 10**(np.log10(vmpeak)*rng.normal(1,0.1, len(vmpeak)))
            self.vbroaden = scvmpeak

        elif params.velocity_attr == 'vmpeakincli':
            # universemachine v_m,peak velocity scaled by inclination
            # with an additional lognormal scatter of 0.1 dex
            self.sin_i = np.sqrt(1-np.random.uniform(size=self.nhalo)**2)
            a = 1 / (1+self.redshift)
            M200 = (1.64e12)/((a/0.378)**-0.142 + (a/0.378)**-1.79)
            vmpeak = 200 * (self.M / M200)**0.3
            rng = np.random.default_rng(12345)
            scvmpeak = 10**(np.log10(vmpeak)*rng.normal(1,0.1, len(vmpeak)))
            self.vbroaden = scvmpeak*self.sin_i/0.866

        elif params.velocity_attr == 'vvir':
            # straight virial velocity
            self.sin_i = np.sqrt(1-np.random.uniform(size=self.nhalo)**2)
            self.vbroaden = vvir(self.M, self.redshift)

        params.filterfunc = gaussian_filter1d

        return self.vbroaden
    
    def offset_velocities(self, params):
        """
        offsets the catalog from the CO in redshift by some velocities. 
        
        inputs:
        -------
        params: SimParameters object
            uses specifically
                params.vcat_offset: mean offset (in km/s)
                params.vcat_scatter: scatter in the mean offset (in km/s)
        outputs:
        --------
        No output, but alters
            self.zcat: new catalog redshifts 
        """

        # velocity offsets with scatter 
        rng = np.random.default_rng(seed=params.vcat_seed)
        dv = rng.normal(loc=params.vcat_offset, scale=params.vcat_scatter, size=len(self.Lcat))

        # peculiar velocity redshift contribution
        dv_c = dv / const.c.to(u.km/u.s).value
        zpec = np.sqrt((1+dv_c) / (1-dv_c)) - 1

        # observed redshift
        zobs = (1+self.redshift)*(1+zpec) - 1

        # save in object
        self.zcat = zobs

    def observation_cull(self, params, in_place=True):
        """
        cuts the catalog by observational parameters: cuts to only objects above a certain luminosity
        and then randomly selects N objects from that cut list.
        
        inputs:
        -------
        params: SimParameters object
            uses specifically
                params.lcat_cutoff: the lower limit on catalog luminosity to include (in Lsun)
                params.goal_nobj: number of catalog objects to include once the cut is made
                params.vcat_seed: rng seed (using the velocity one)
                params.obs_weight: whether observation culling should be logarithmic or linear (or unweighted)
        in_place: bool (optional, default=True)
            if True, performs cuts on this HaloCatalog object. if False, returns a copy of the object
            with cuts applied
        outputs:
        --------
        halos: HaloCatalog object (if in_place = True)
            copy of the input HaloCatalog object with the observational cuts applied
        """

        if not in_place:
            halos = self.copy()

        # cut by luminosity (normally, just do a straightforward cutoff at Lcut. if dealing with the eboss model do that 
        #    specifically)
        if params.catalog_model == 'eboss':
            # convert from Lsun to relative g-band magnitudes
            Mg = 4.74 - np.log10(self.Lcat) / 0.4
            mg = mag_abs_to_rel(Mg, self.redshift)

            brightidx = np.where(mg < params.lcat_cutoff)

            if in_place:
                self.indexcut(brightidx, in_place=True)
            else:
                halos.indexcut(brightidx, in_place=True)

        else:
            if in_place:
                self.attrcut_subset('Lcat', params.lcat_cutoff, np.nanmax(self.Lcat)+10, params, in_place=True)
            else:
                halos.attrcut_subset('Lcat', params.lcat_cutoff, np.nanmax(self.Lcat)+10, params, in_place=True)

        if params.goal_nobj > 0:
            if in_place:
                # select nobj random objects from the leftover catalog
                rng = np.random.default_rng(params.vcat_seed)
                if params.obs_weight == 'linear':
                    weights = self.Lcat / np.sum(self.Lcat)
                elif params.obs_weight == 'log':
                    weights = np.log10(self.Lcat) / np.sum(np.log10(self.Lcat))
                elif params.obs_weight == 'none':
                    weights = np.ones(self.nhalo) / self.nhalo
                keepidx = rng.choice(self.nhalo, params.goal_nobj, replace=False, p=weights) #*** use probability here to weight selections
                # cut to these objects
                self.indexcut(keepidx, in_place=True)
            else:
                # select nobj random objects from the leftover catalog
                rng = np.random.default_rng(params.vcat_seed)
                if params.obs_weight == 'linear':
                    weights = halos.Lcat / np.sum(halos.Lcat)
                elif params.obs_weight == 'log':
                    weights = np.log10(halos.Lcat) / np.sum(np.log10(halos.Lcat))
                elif params.obs_weight == 'none':
                    weights = np.ones(halos.nhalo) / halos.nhalo
                keepidx = rng.choice(halos.nhalo, params.goal_nobj, replace=False, p=weights) #*** use probability here to weight selections
                # cut to these objects
                halos.indexcut(keepidx, in_place=True)

        if params.verbose: print('\n\t%d halos remain after observability cuts' % self.nhalo)

        if not in_place:
            return halos

    
    #### FUNCTIONS TO SLICE THE HALO catalog IN SOME WAY
    def indexcut(self, idx, in_place=False):
        """
        crops the halo catalog to only include halos included in the passed index
        array.

        inputs:
        -------
        idx: array-like, integer
            the catalog indices to cut the catalog to
        in_place: bool (optional, default=True)
            if True, performs cuts on this HaloCatalog object. if False, returns a copy of the object
            with cuts applied
        outputs:
        --------
        subset: HaloCatalog object (if in_place=False)
            a copy of the catalog object with the cuts applied
        """
        # assert np.max(idx) <= self.nhalo,   "Too many indices"

        if not in_place:
            # new halos object to hold the cut catalog
            subset = self.copy()

            # copy all the arrays over, indexing as you go
            for i in dir(self):
                if i[0]=='_': continue
                try:
                    setattr(subset, i, getattr(self,i)[idx])
                except TypeError:
                    pass
            subset.nhalo = len(subset.M)

        else:

            # replace all the arrays with an indexed version
            for i in dir(self):
                if i[0]=='_': continue
                try:
                    setattr(self, i, getattr(self,i)[idx])
                except TypeError:
                    pass
                self.nhalo = len(self.M)

        if not in_place:
            return subset


    def attrcut_subset(self, attr, minval, maxval, params, in_place=False):
        """
        crops the halo catalog to only include desired halos, based on some arbitrary
        attribute attr. will include haloes with attr from minval to maxval.

        inputs:
        -------
        attr: str
            the attribute of the HaloCatalog object to subset - will keep (minval,maxval]
        minval: float
            the minimum value to be kept
        maxval: float
            the maximum value to be kept 
        params: SimParameters object
            the parameters for the simulation run. uses:
                verbose: (bool) write more detailed messages to terminal if true
        in_place: bool
            if True, apply cuts directly to this HaloCatalog object. If False, returns a copy of the object
        outputs:
        --------
        subset: HaloCatalog object (if in_place==False)
            a copy of the input HaloCatalog object with the cuts applied
        """

        keepidx = np.where(np.logical_and(getattr(self,attr) > minval,
                                          getattr(self,attr) <= maxval))[0]

        if not in_place:
            # new halos object to hold the cut catalog
            subset = self.copy()

            # copy all the arrays over, indexing as you go
            for i in dir(self):
                if i[0]=='_': continue
                try:
                    setattr(subset, i, getattr(self,i)[keepidx])
                except TypeError:
                    pass
            nhalo = len(subset.M)
            subset.nhalo = nhalo

        else:

            # replace all the arrays with an indexed version
            for i in dir(self):
                if i[0]=='_': continue
                try:
                    setattr(self, i, getattr(self,i)[keepidx])
                except TypeError:
                    pass
                nhalo = len(self.M)
                self.nhalo = nhalo

        if params.verbose: print('\n\t%d halos remain after attribute cut' % nhalo)

        if not in_place:
            return subset


    def masscut_subset(self, min_mass, max_mass, in_place=False):
        """
        same as attrcut_subset, but a cut on mass specifically (for convenience)
        """
        if in_place:
            self.attrcut_subset('M', min_mass, max_mass, in_place=True)
        else:
            return self.attrcut_subset('M', min_mass, max_mass)

    def vmaxcut_subset(self, min_vmax, max_vmax, in_place=False):
        """
        same as attrcut_subset, but a cut  on vmax specifically (for convenience)
        """
        if in_place:
            self.attrcut_subset('vmax', min_vmax, max_vmax, in_place=True)
        else:
            return self.attrcut_subset('vmax', min_vmax, max_vmax)


    def write_cat(self, params, trim=None, writeall=False):
        """
        Write the halos to a .npz file

        inputs:
        -------
        params: parameters object
            the parameters for the simulation run. uses:
                cat_output_file: (str) the filename to be saved to (should be npz)
                verbose: (bool) writes more detailed messages to terminal
        trim: int (optional, default None)
            if this has a value, it will trim the catalog to the first N=trim objects before saving
            (this is mainly for convenience while debugging, because there can be many millions of objects
             in the peakpatch catalogues)
        writeall: bool (optional, default False)
            if True, will save all of the various values for each halo. Otherwise, just saves ra, dec, 
            redshift, mass, vhalo, Lco, and Lcat
        outputs:
        --------
        None (saves catalogue to params.cat_output_file)
        """
        if params.verbose: print('\n\tSaving Halo catalog to\n\t\t', params.cat_output_file)

        # trim the catalog to the first N=trim objects if trim is passed
        if trim:
            i = trim
        else:
            i = -1

        # fill in the velocity array with negatives if they haven't already been calculated
        try:
            velocities = self.vbroaden[:i]
        except AttributeError:
            velocities = np.ones(len(self.dec[:i])) * -99

        # write
        if writeall:
            np.savez(params.cat_output_file,
                     dec=self.dec[:i], nhalo=len(self.dec[:i]),
                     nu=self.nu[:i], ra=self.ra[:i],
                     z=self.redshift[:i], vx=self.vx[:i],
                     vz    = self.vz[:i],
                     x_pos   = self.x_pos[:i],
                     y_pos = self.y_pos[:i],
                     z_pos        = self.z_pos[:i],
                     zformation=self.zformation[:i],
                     Lco = self.Lco[:i],
                     Lcat = self.Lcat[:i],
                     vhalo = velocities,
                     M = self.M[:i])
        else:
            np.savez(params.cat_output_file,
                     dec = self.dec[:i], ra = self.ra[:i],
                     z = self.redshift[:i],
                     Lco = self.Lco[:i],
                     Lcat = self.Lcat[:i],
                     vhalo = velocities,
                     M = self.M[:i])

        return
