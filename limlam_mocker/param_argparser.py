import argparse 
import ast


class LoadFromFile(argparse.Action):
    """
    load parameters directly from a parameter file (example file is ****)
    """
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            # parse arguments in the file and store them in the target namespace
            parser.parse_args(f.read().split(), namespace)

def str2bool(v):
    """
    enables true/false statements from file (argparser doesn't handle boolean very well)
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    
def str2list(v):
    """
    enables lists to be read in as actual lists
    """
    return ast.literal_eval(v)
    
parser = argparse.ArgumentParser()

### Parameter file
parser.add_argument(
    "-p",
    "--param",
    type=open,
    action=LoadFromFile,
    help="Path to parameter file. File should have argparse syntax, and overwrites any value listed here.",
)

parser.add_argument("-f",
                    help="a dummy argument to fool ipython", default="1")

parser.add_argument("-v",
    "--verbose",
    type=str2bool,
    default=True,
    help="Enable verbose printing and turn off the terminal rewriting progress bar.")

parser.add_argument(
    "--halo_catalog_file",
    type=str,
    default='/home/deedunne/Documents/COMAP/limlam_mocker_auxcode/peakpatch_catalogues/0/COMAP_z2.39-3.44_1140Mpc_seed_13587.npz',
    help="(SimGenerator) Path to peak-patch simulation catalog. Default seed 13579."
)

parser.add_argument(
    "--mass_cutoff",
    type=float,
    default=100000000000000.,
    help="(SimGenerator) Maximum DM mass to include in the simulated cube (in M_sun). Default 1e14 M_sun."
)

parser.add_argument(
    "--min_mass",
    type=float,
    default=1000000000.,
    help="(SimGenerator) Minimum DM mass to include in the simulated cube (in M_sun). Default 1e9 M_sun."
)

parser.add_argument(
    "--model",
    type=str,
    default="fiuducial",
    help="(SimGenerator) Name of model to use for halo CO luminosities. By default 'fiuducial' is used."
)

parser.add_argument(
    "--co_model_coeffs",
    type=str2list,
    default=None,
    help="(SimGenerator) Adjusted model coefficients. If 'None', default values for the model are used."
)
parser.add_argument(
    "--catalog_model",
    type=str,
    default='schechter',
    help="(SimGenerator) Name of the function used to model emission of the other tracer. Defaults to 'default' function in generate_luminosities.py."
)

parser.add_argument(
    "--catalog_coeffs",
    type=str2list,
    default=None,
    help="(SimGenerator) Coefficients used for the emission modeling function. Defaults to None."
)

parser.add_argument(
    "--catdex",
    type=float,
    default=0.41,
    help="(SimGenerator) Size of the artificial scatter in the tracer luminosities. Defaults to 0.5."
)

parser.add_argument(
    "--codex",
    type=float,
    default=0.42,
    help="(SimGenerator) Size of the artificial scatter in the tracer luminosities. Defaults to 0.42 (chung22 fiducial value)."
)

parser.add_argument(
    "--rho",
    type=float,
    default=-0.5,
    help="(SimGenerator) Correlation between CO and tracer luminosities (-1, 1). Default to -0.5."
)

parser.add_argument(
    "--lum_uncert_seed",
    type=int,
    default=12345,
    help="(SimGenerator) Seed for the RNG determining scatter in the halo luminosities. Default 12345."
)

parser.add_argument(
    "--save_scatterless_lums",
    type=str2bool,
    default=True,
    help="(SimGenerator) Boolean: whether to keep the luminosity values calculated before scatter added. Defaults to True."
)

parser.add_argument(
    "--cosmology",
    type=str,
    default='comap',
    help="(SimGenerator) The cosmological parameters to use in generating the simulations. Defaults to the values used in Li 2016."
)

parser.add_argument(
    "--units",
    type=str,
    default='temperature',
    help="(SimGenerator) The brightness units used by the simulations. Defaults to 'temperature'."
)

parser.add_argument(
    "--nmaps",
    type=int,
    default=1024,
    help="(SimGenerator) Number of frequency channels to include in the (final) simulation cube. Default 1024."
)

parser.add_argument(
    "--npix_x",
    type=int,
    default=120,
    help="(SimGenerator) Number of pixels to include in the RA axis of the (final) simulation cube. Default 120."
)

parser.add_argument(
    "--npix_y",
    type=int,
    default=120,
    help="(SimGenerator) Number of pixels to include in the Dec axis of the (final) simulation cube. Default 120."
)

parser.add_argument(
    "--fov_x",
    type=float,
    default=4.,
    help="(SimGenerator) Size in the RA axis of the simulation cube (in degrees). Default 4.0 degrees."
)

parser.add_argument(
    "--fov_y",
    type=float,
    default=4.,
    help="(SimGenerator) Size in the Dec axis of the simulation cube (in degrees). Default 4.0 degrees."
)

parser.add_argument(
    "--nu_f",
    type=float,
    default=26.,
    help="(SimGenerator) Minimum frequency to include in the map (in GHz). Default 26.0."
)

parser.add_argument(
    "--nu_i",
    type=float,
    default=34.,
    help="(SimGenerator) Maximum frequency to include in the map (in GHz). Default 34.0."
)

parser.add_argument(
    "--nu_rest",
    type=float,
    default=115.27,
    help="(SimGenerator) Rest-frequency of the spectral line being modeled (in GHz). Default CO(1-0): 115.27 GHz."
)

parser.add_argument(
    "--xrefine",
    type=int,
    default=5,
    help="(SimGenerator) Factor by which to oversample the angular axes of the simulations. Defaults to 5."
)

parser.add_argument(
    "--freqrefine",
    type=int,
    default=5,
    help="(SimGenerator) Factor by which to oversample the frequency axes of the simulations. Defaults to 5."
)

parser.add_argument(
    "--save_hits",
    type=str2bool,
    default=True,
    help="(SimGenerator) Save the galaxy catalog as a hit map in the mapfile. Defaults to True."
)

parser.add_argument(
    "--weight_hits",
    type=str2bool,
    default=False,
    help="(SimGenerator) weight the hit map by catalog luminosities. Defaults to False."
)

### CO INSTRUMENT/ASTROPHYSICS MODIFIERS
parser.add_argument(
    "--beambroaden",
    type=str2bool,
    default=True,
    help="(SimGenerator) Whether to smooth the angular axes by a 4.5' Gaussian approximation of the COMAP primary beam. Defaults to True."
)

parser.add_argument(
    "--beamkernel",
    type=str,
    default=None,
    help="(SimGenerator) Convolution kernel approximating the primary beam. If none, use Gaussian with 4.5' FWHM."
)

parser.add_argument(
    "--beamfwhm",
    type=float,
    default=4.5,
    help="(SimGenerator) FWHM of the Gaussian beam to smooth the data by. Defaults to 4.5. If both this and beamkernel are set beamkernel will take priority."
)

parser.add_argument(
    "--freqbroaden",
    type=str2bool,
    default=True,
    help="(SimGenerator) Whether to simulate astrophysical line broadening. Defaults to True."
)

parser.add_argument(
    "--bincount",
    type=int,
    default=5,
    help="(SimGenerator) Number of mass bins to split simulated halos into before line broadening. Defaults to 5."
)

parser.add_argument(
    "--fwhmfunction",
    type=str,
    default=None,
    help="(SimGenerator) Function used to calculate FWHMa for halos. If none (default), vvirsini is used."
)

parser.add_argument(
    "--velocity_attr",
    type=str,
    default='vvirincli',
    help="(SimGenerator) Which type of per-halo velocity (stored as an attribute) to use when broadening. Default 'vvirincli'."
)

parser.add_argument(
    "--add_comap_noise",
    type=str2bool,
    default=False,
    help="(SimGenerator) Quick and dirty way to add in a white noise approximation of COMAP noise. Defaults to False."
)

parser.add_argument(
    "--noise_int_time",
    type=float,
    default=303.7,
    help="(SimGenerator) Integration time in hours used to calculate random radiometer noise. Default is 303.7 (Field 1, Season 1)."
)

parser.add_argument(
    "--Tsys",
    type=float,
    default=44.,
    help="(SimGenerator) System temperature of the instrument used to calculate noise. Defaults to COMAP Ka band (44 K)."
)

parser.add_argument(
    "--nfeeds",
    type=int,
    default=19,
    help="(SimGenerator) Number of independent FPA feeds observing. Defaults to 19 (COMAP Pathfinder value)."
)

parser.add_argument(
    "--noise_seed",
    type=int,
    default=12345,
    help="(SimGenerator) RNG seed for generating COMAP radiometer noise. Default is 12345."
)

parser.add_argument(
    "--add_foreground",
    type=str2bool,
    default=False,
    help="(SimGenerator) Quick and dirty way to add in some fake foreground/background emission. Defaults to False."
)

parser.add_argument(
    "--fg_permutation",
    type=int,
    default=0,
    help="(SimGenerator) How to permute the input map to get the foreground emission. Values 0-10, defaults to 0."
)

parser.add_argument(
    "--fg_scalefactor",
    type=float,
    default=10,
    help="(SimGenerator) Factor by which to scale down the input map to get the foreground emission. Defaults to 10."
)

### CATALOG INSTRUMENT/ASTROPHYSICS MODIFIERS
parser.add_argument(
    "--lcat_cutoff",
    type=float,
    default=0.,
    help="(SimGenerator) Lowest catalog luminosity to include in catalog files and hitmaps. Defaults to zero."
)

parser.add_argument(
    "--goal_nobj",
    type=int,
    default=-99,
    help="(SimGenerator) Number of catalog objects to keep for stats. If negative, includes all objects above luminoisty cut. Defaults to -99."
)

parser.add_argument(
    "--obs_weight",
    type=str,
    default='linear',
    help="(SimGenerator) How to weight catalog luminosities when cutting to goal_nobj observable ones. 'none', 'log' or 'linear', defaults to 'linear'." 
)

parser.add_argument(
    "--vcat_offset",
    type=float,
    default=0.,
    help="(SimGenerator) Offset the catalog redshifts by some mean velocity (in km/s). Defaults to 0."
)

parser.add_argument(
    "--vcat_scatter",
    type=float,
    default=0.,
    help="(SimGenerator) (Gaussian STD) of the scatter in the catalog velocity offset (in km/s). Defaults to zero."
)

parser.add_argument(
    "--vcat_seed",
    type=int,
    default=12345,
    help="(SimGenerator) Random seed to use when generating catalog velocity offsets."
)

parser.add_argument(
    "--lazyfilter",
    type=str2bool,
    default=True,
    help="(SimGenerator) Faster FFT when binning after line broadening. Defaults to True."
)

parser.add_argument(
    "--output_dir",
    type=str,
    default='./simulations',
    help="(SimGenerator) Path to directory in which to store all the output simulation files." #*****
)

parser.add_argument(
    "--map_output_file_name",
    type=str,
    default='sim_map.npz',
    help="(SimGenerator) File name for the final simulated map (.npz). Default sim_map.npz" #*****
)

parser.add_argument(
    "--cat_output_file_name",
    type=str,
    default='sim_cat.npz',
    help="(SimGenerator) File name for the final simulated catalog (.npz). Default sim_cat.npz" #*****
)
