import numpy as np

def get_bin_line(ind, bins):
    '''
    Transforms bins into symbolicaly readed ranges.
    Inputs:
        ind - integer index of used bin;
        bins - np.array of shape (n_bins, ), 
                the points which divides one range from another.
    Output
        string in format [bins[ind - 1], bins[ind])
    '''
    
    if ind <= 0:
        return "(-inf, " + str(bins[ind]) + ")"
    if ind >= len(bins):
        return "[" + str(bins[ind - 1]) +  ",inf)"

    return "[" + str(bins[ind - 1]) +\
            "," + str(bins[ind]) + ")"

def encode_as_lines(x, bins, save_na = True):
    '''
    Transforms numeric array to lines which marks bins.
    Inputs:
        x - np.array with shape (n_sample, ), array to be transformed;
        bins - np.array with shape (n_bins, ), bins which divides one range from another;
        save_na - bool if True saves the np.NaN value from range x, else acts as np.digitise
                  i.e. NaNs belong to the latter range.
    Output
        np.array fo shape (n_sample, ), transformed x range.
    '''
    
    res = np.array(list(
        map(lambda ind: get_bin_line(ind, bins), np.digitize(x, bins).ravel())
    ))
    
    if save_na:
        res[np.isnan(x)] = np.NaN
    
    return res