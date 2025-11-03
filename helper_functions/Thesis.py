import numpy as np


def radii_calc(data):
    Coords = np.array(data['PartType0']['Coordinates'], dtype=np.float64) 
    COM = np.array(data['PartType5']['Coordinates'], dtype=np.float64) 

    boxsize = data['Parameters'].attrs['BoxSize']

    #print(f'pot = {pot.max()}, index = {index}, center = {center}')

    shifted_COM = Illustris_functions.wrap_around(Coords, COM, boxsize)
    radii = np.linalg.norm(shifted_COM, axis=1) * 3.085678e+21 # now in cm 
    return radii
