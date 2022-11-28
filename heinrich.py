import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter

#inversion pipeline stage
def inv( data_dict, inv_threshold = 0.6, ** args ):
    
    '''
    inverts a signal if necessary
    
        Expects: 
            X_train (array): nan-padded
            X_test (array): nan-padded
            
        Parameters:
            inv_threshold (float): the 'invertedness' cutoff value, signals above that will be inverted
        
        Returns:
            X_train (array): nan-padded
            X_test (array): nan-padded
    '''
    
    assert "X_train" in data_dict.keys( )
    assert "X_test" in data_dict.keys( )
    
    X_train, X_test = itemgetter( "X_train", "X_test" )( data_dict )

    f = lambda signal: invert_signal_if_necessary( signal, inv_threshold )
    data_dict[ "X_train" ] = np.apply_along_axis( f, 1, X_train )
    data_dict[ "X_test" ] = np.apply_along_axis( f, 1, X_test )
    
    return data_dict

#cropping pipeline stage
def crop( data_dict, crop_location = 300, **args ):
    
    '''
    crops the first couple points of a signal
    
        Expects: 
            X_train (array): nan-padded
            X_test (array): nan-padded
            
        Parameters:
            crop_location (int): how many data points should be cut from the front 
        
        Returns:
            X_train (array): nan-padded
            X_test (array): nan-padded
    '''
    
    assert "X_train" in data_dict.keys( )
    assert "X_test" in data_dict.keys( )
    
    X_train, X_test = itemgetter( "X_train", "X_test" )( data_dict )

    f = lambda signal: crop_signal( signal, crop_location )
    data_dict[ "X_train" ] = np.apply_along_axis( f, 1, X_train )
    data_dict[ "X_test" ] = np.apply_along_axis( f, 1, X_test )
    
    return data_dict

def invert_signal_if_necessary( signal, threshold ):
    
    return - signal if criterion( signal ) > threshold else signal

def crop_signal( signal, location ):
    
    return signal[ location: ]

def criterion( signal ):

    low = np.nanquantile( signal, 0.01 )
    high = np.nanquantile( signal, 0.99 )
    mid = np.nanquantile( signal, 0.5 )

    return ( mid - low ) / ( high - low )

def signal_length( signal ):
    
    return np.count_nonzero( ~np.isnan( signal ))

def show_heartbeat( signal ):
    
    length_s = signal_length( signal ) / 300
    print( f"length: { length_s } s" )
    plt.figure( figsize = ( 15 * ( length_s / 60 ), 2 ))
    plt.axhline( np.nanmedian( signal ), color = "red" )
    plt.axhline( np.nanquantile( signal, q = 0.01 ), color = "red" )
    plt.axhline( np.nanquantile( signal, q = 0.99 ), color = "red" )
    plt.plot( signal )