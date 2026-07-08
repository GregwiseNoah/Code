import numpy as np


def radii_calc(data):
    h = np.array(data['Header'].attrs['HubbleParam'], dtype=np.float64)
    Coords = np.array(data['PartType0']['Coordinates'], dtype=np.float64) *1/h
    boxsize = data['Parameters'].attrs['BoxSize']*1/h
    shifted_Coords = Coords - boxsize * 0.5
    radii = np.linalg.norm(shifted_Coords, axis=1) * 3.085678e+21 # now in cm 
    return radii


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