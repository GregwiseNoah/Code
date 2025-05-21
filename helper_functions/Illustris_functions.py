import numpy as np

def wrap_around(array, center, box_size):
    # Take input from an array and split it from the center value like  the COM, and join the old start and end.
    # Obtain box size from hdf5 file header
    # Center is usually calculated with the particle with the lowest potential, the center of the halo.
    box_center = np.array([box_size, box_size, box_size]) / 2.0
    shift = box_center - center

    array_shifted = (array + shift) % box_size
    print( array.min(), array_shifted.min(), center, box_center, shift )

    return array_shifted
