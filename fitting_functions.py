import numpy as np
from scipy.signal import argrelmax

def gaussian(x, a, x0, sigma, y0):
    '''

    The function creates a gaussian curve f(x), 
    where x is a normalized array

    a = amplitude parameter
    x0 = position of the center of the peak
    sigma = width parameter
    y0 = vertical offset
    '''
    return  y0 + (a / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - x0)**2 / (2 * sigma**2))


def lorentzian( x, a, x0, gamma, y0):
    '''

    The function creates a lorenztian curve f(x), 
    where x is a normalized array

    a = amplitude parameter
    x0 = position of the center of the peak
    gamma = width parameter
    y0 = vertical offset
    '''

    return y0 + (a/np.pi) * gamma / (gamma**2 + ( x - x0 )**2)

def select_2peaks(data, order_parameter, step, noise_level):
    '''
    The functions looks for maxima in the given array and selects a neighborhood
    around the first two local maxima.
    
    data = array to be inspected
    order_parameter = number of points considered when looking for the local maximum
    step = number of points considered when selecting the neighborhood on either side of the maximum
    noise_level = threshold under which the data are ignored
    '''
    loc_max = argrelmax(data[:,1], order=order_parameter)
    max_idx = np.array([])
    for kk in range(len(loc_max[0][:])):
        if data[loc_max[0][kk],1] > noise_level:
            max_idx = np.append(max_idx,loc_max[0][kk])
    if max_idx[0] > step:
        data_pk_1 = data[int(max_idx[0]-step):int(max_idx[0]+step),:]
        data_pk_2 = data[int(max_idx[1]-step):int(max_idx[1]+step),:]
    else: 
        raise ValueError("A neighborhood cannot be selected: max_idx[0] <= step")
    return data_pk_1, data_pk_2
