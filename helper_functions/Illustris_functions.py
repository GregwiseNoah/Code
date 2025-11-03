import requests
import h5py
import numpy as np
import matplotlib.pyplot as plt
#import astropy.units as u
import cmasher as cmr
from scipy.stats import binned_statistic


###### NOTE TO SELF ######
# Most of the comoving to proper unit conversion has a factor a missing since I am using a = 0 right now
##########################
h = 0.704
msun_g = 1.989e33


def get(path, params=None):
    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically
    return r



# Custom functions
def wrap_around(array, center, box_size):
    # Take input from an array and split it from the center value like  the COM, and join the old start and end.
    # Obtain box size from hdf5 file header
    # Center is usually calculated with the particle with the lowest potential, the center of the halo.
    delta = array - center
    delta = (delta + box_size / 2.0) % box_size - box_size / 2.0
    return delta
    # box_center = np.array([box_size, box_size, box_size]) / 2.0
    # shift = box_center - center

    # array_shifted = (array + shift) % box_size
    # #print( array.min(), array_shifted.min(), center, box_center, shift )

    # return array_shifted

def density_converter(density_array, flag):
    # Converts density from units of 10**10 (M_sun / h) / (ckpc/h) ** 3  to required units
    # Flag determines if density will be converted to g / kpc**3 or g / cm**3
    msun_g = 1.989e33

    if flag==None:
        raise ValueError("Usage: flag == 1 : Msun / kpc**3 \n flag == 2 : g / kpc**3 \n flag == 3 : g / cm**3")
    elif flag == 1:
        return density_array  * 1e10  / (h**2)
    elif flag == 2:
        return density_array * 1e10 * msun_g / (h**2)
    elif flag == 3:
        return density_array * 1e10  * msun_g / (h**2 * (3.086e21 **3) )
    else:
        raise ValueError("Usage: flag == 1 : Msun / kpc**3 \n flag == 2 : g / kpc**3 \n flag == 3 : g / cm**3")

def mass_converter(mass):
    msun_g = 1.989e33  # g
    h = 0.704
    return mass * 1e10 / h * msun_g



def Temp_calculator( Part0dict ):
    # Calculates temperatures from internal energy and density
    # Part0dist is of the form halo_0['PartType0']
    # returns Temp in kelvin
    gamma = 5/3
    kb = 1.380649e-16 # cgs
    X_H = 0.76
    mp = 1.6726e-24 

    xe =  Part0dict['ElectronAbundance'][:]
    u = Part0dict['InternalEnergy'][:] * 1e10 #cgs (cm/s)**2

    print(xe.min(), u.min())
    
    # Calculate mean molecular weight, mu
    mu = 4 / (1 + 3 * X_H + 4 * X_H * xe) * mp

    Temp = (gamma - 1) * (u / kb) * mu 

    return Temp
    


def background_selector(flag):
    if flag == 'dark':
        plt.style.use('dark_background')
    elif flag == 'light':
        plt.style.use('default')
        #plt.style.use('seaborn-v0_8-pastel')


def median_binner(data_array, radius, nbins = 100):
    
    # radius_bins = np.logspace(np.log10(radius.min() + 1e-3 ) , np.log10(radius.max()), 100) # Adding a 1e-3 to avoid log10(0)
    radius_bins = np.logspace(np.log10( 5 ) , np.log10(radius.max()), nbins) 

    stat_result = binned_statistic(
        radius,               
        data_array,              
        statistic='median',   
        bins=radius_bins
    )

    binned_array = stat_result.statistic
    bin_edges = stat_result.bin_edges
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    return binned_array, bin_centers

    