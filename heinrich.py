import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
from biosppy.signals.ecg import ecg

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
            X_train_inv_indices (array): boolean array
            X_test_inv_indices (array): boolean array
    '''
    
    assert "X_train" in data_dict.keys( )
    assert "X_test" in data_dict.keys( )
    
    X_train, X_test = itemgetter( "X_train", "X_test" )( data_dict )
    is_likely_inverted = lambda signal: criterion( signal ) > inv_threshold
    
    X_train_inv_indices = np.apply_along_axis( is_likely_inverted, 1, X_train )
    X_test_inv_indices = np.apply_along_axis( is_likely_inverted, 1, X_test )
    
    data_dict[ "X_train_inv_indices" ] = X_train_inv_indices
    data_dict[ "X_test_inv_indices" ] = X_test_inv_indices
    
    f = lambda signal: - signal if is_likely_inverted( signal ) else signal
    
    data_dict[ "X_train" ] = np.apply_along_axis( f, 1, X_train )
    data_dict[ "X_test" ] = np.apply_along_axis( f, 1, X_test )
    
    #draw distribution of criterion
    print( "[inv] distribution of 'invertedness' criterion" )
    crit = np.apply_along_axis( criterion, 1, data_dict[ "X_train" ])
    plt.hist( crit )
    
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

#ecg feature extraction pipeline stage
def ecgExtract( data_dict, ** args ):
    
    for key in [ "X_train", "X_test" ]:
        
        assert key in data_dict.keys( )
        data_dict[ key ] = np.apply_along_axis( ecg_feature_extract, 1, data_dict[ key ])
        
    return data_dict

def descriptive_features( xs ):
    
    fs = [ np.min, np.max, np.mean, np.median, np.std ]
    return np.array([ f( xs ) for f in fs ])

def ecg_feature_extract( signal ):
    
    length = signal_length( signal )
    clean = signal[ :length ]
    f = lambda s: ecg( s, 300, show = False )
    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = f( clean )
    
    #descriptive statistics about the signal
    signal_f = descriptive_features( clean )
    
    #descriptive statistics about the peaks
    peak_f = descriptive_features( filtered[ rpeaks ])
    
    #descriptive statistics about the mean template
    mean_template_f = descriptive_features( np.mean( templates, axis = 0 ))
    
    
    heart_rate_failure = len( heart_rate ) == 0
    heart_rate = np.array([ 80 ]) if heart_rate_failure else heart_rate
        
    #descriptive statistics about the heart rate
    heart_rate_f = descriptive_features( heart_rate )
    
    ecg_features = np.concatenate(( signal_f, peak_f, mean_template_f, heart_rate_f ))
    
    return ecg_features

def crop_signal( signal, location ):
    
    return signal[ location: ]

def criterion( signal ):

    low = np.nanquantile( signal, 0.01 )
    high = np.nanquantile( signal, 0.99 )
    mid = np.nanquantile( signal, 0.5 )

    return ( mid - low ) / ( high - low )

def signal_length( signal ):
    
    return np.count_nonzero( ~np.isnan( signal ))

def show_signal( signal ):
    
    length_s = signal_length( signal ) / 300
    print( f"length: { length_s } s" )
    plt.figure( figsize = ( 15 * ( length_s / 60 ), 2 ))
    plt.axhline( np.nanmedian( signal ), color = "red" )
    plt.axhline( np.nanquantile( signal, q = 0.01 ), color = "red" )
    plt.axhline( np.nanquantile( signal, q = 0.99 ), color = "red" )
    plt.plot( signal )
    
def plot_heartbeat_dist( templates_ts, templates ):
    
    mu = np.mean( templates, axis = 0 )
    upper = np.quantile( templates, axis = 0, q = 0.9 )
    lower = np.quantile( templates, axis = 0, q = 0.1 )
    plt.plot( templates_ts, mu, color = "#2020a0ff" )
    plt.fill_between( templates_ts, upper, lower, color = "#2020a040" )
    plt.show( )

def plot_signal_with_peaks( ts, filtered, rpeaks ):
    
    plt.plot( ts, filtered )
    fig = plt.gcf( )
    ax = fig.gca( )
    for peak in rpeaks:
        x = peak / 300.0
        y = filtered[ peak ]
        c = plt.Circle(( x, y ), 0.2, color = "red" )
        ax.add_patch( c )