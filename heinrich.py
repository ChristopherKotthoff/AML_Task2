import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
from biosppy.signals.ecg import ecg, correct_rpeaks
from pywt import wavedec

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
    
    assert "X_train" in data_dict.keys( )
    assert "X_test" in data_dict.keys( )
    
    for key in [ "X_train", "X_test" ]:
        
        assert key in data_dict.keys( )
        print( f"extracting from {key}" )
        data_dict[ key ] = apply_along_axis_tqdm( ecg_feature_extract, 1, data_dict[ key ])
        
    return data_dict

#random forest classification
def rfClassification( data_dict, rfClassification_depth, rfClassification_useValidationSet, rfClassification_makePrediction, ** args ):
    
    assert "X_train" in data_dict.keys()
    assert "y_train" in data_dict.keys()
    
    X_train = data_dict[ "X_train" ]
    y_train = data_dict[ "y_train" ]
    X_val = None
    y_val = None
    X_test = None
    
    if rfClassification_useValidationSet:
        
        assert "X_val" in data_dict.keys()
        assert "y_val" in data_dict.keys()
        
        X_val = data_dict[ "X_val" ]
        y_val = data_dict[ "y_val" ]
        
    if rfClassification_makePrediction:
        
        assert "X_test" in data_dict.keys()
        X_test = data_dict[ "X_test" ]

    if rfClassification_useValidationSet:
        
        data_dict["train_losses"], data_dict["val_losses"], predict_funct = train_random_forest( rfClassification_depth, X_train, y_train, X_val, y_val )
        data_dict["y_val_hat"] = predict_funct( X_val )
        data_dict["y_train_hat"] = predict_funct( X_train )
        
    else:
        
        data_dict["train_losses"], predict_funct = train_random_forest( rfClassification_depth, X_train, y_train )
        data_dict["y_train_hat"] = predict_funct( X_train )

    if rfClassification_makePrediction:
        
        data_dict["y_test"] = predict_funct(X_test)

    if rfClassification_useValidationSet:

        data_dict["y_val_hat"] = predict_funct(X_val)

    return data_dict

from tqdm import tqdm

def apply_along_axis_tqdm( f, axis, arr ):

    N = arr.shape[ 1 - axis ] - 1
    it = iter( tqdm( range( N )))
    
    def g( x ): 
        
        try:
            next( it )
        except:
            pass
        return f( x )
    
    return np.apply_along_axis( g, axis, arr )

def descriptive_features( xs ):
    
    fs = [ np.min, np.max, np.mean, np.median, np.std ]
    return np.array([ f( xs ) for f in fs ])

def ecg_full( signal ):
    
    length = signal_length( signal )
    clean = signal[ :length ]
    f = lambda s: ecg( s, 300, show = False )
    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = f( clean )
    return { "ts": ts, "filtered": filtered, "rpeaks": rpeaks, "templates_ts": templates_ts, "templates": templates, "heart_rate_ts": heart_rate_ts, "heart_rate": heart_rate }

def ecg_feature_extract( signal ):
    
    length = signal_length( signal )
    clean = signal[ :length ]
    f = lambda s: ecg( s, 300, show = False )
    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = f( clean )
    rpeaks = correct_rpeaks( signal = clean, rpeaks = rpeaks, sampling_rate = 300, tol = 0.1 )
    
    #descriptive statistics about the signal
    signal_f = descriptive_features( clean )
    signal_diff_f = descriptive_features( np.diff( clean ))
    
    #descriptive statistics about the peaks
    peaks = filtered[ rpeaks ]
    peak_f = descriptive_features( peaks )
    peak_diff_f = descriptive_features( np.diff( peaks ))
    
    #descriptive statistics about the mean template
    mean_template = np.mean( templates, axis = 0 )
    mean_template_f = descriptive_features( mean_template )
    mean_template_diff_f = descriptive_features( np.diff( mean_template ))
    
    heart_rate = np.array([ 80 ]) if len( heart_rate ) == 0 else heart_rate
    heart_rate = np.concatenate(( heart_rate, heart_rate )) if len( heart_rate ) == 1 else heart_rate
        
    #descriptive statistics about the heart rate
    heart_rate_f = descriptive_features( heart_rate )
    heart_rate_diff_f = descriptive_features( np.diff( heart_rate ))
    
    heart_rate_ts = np.array([ 15. ]) if len( heart_rate_ts ) == 0 else heart_rate_ts
    heart_rate_ts = np.concatenate(( heart_rate_ts, heart_rate_ts )) if len( heart_rate_ts ) == 1 else heart_rate_ts

    #descriptive statistics about the heart rate timestamp
    heart_rate_ts_f = descriptive_features( heart_rate_ts )
    heart_rate_ts_diff_f = descriptive_features( np.diff( heart_rate_ts ))
    
    #descriptive statistics about the deviations from mean heartbeat
    dev_from_mean = lambda template: np.mean(( template - mean_template ) ** 2 )
    deviations = np.apply_along_axis( dev_from_mean, 1, templates )
    deviation_f = descriptive_features( deviations )
    deviation_diff_f = descriptive_features( np.diff( deviations ))
    
    #descriptive statistics about the oddest template
    odd_template = templates[ np.argmax( deviations )]
    odd_template_f = descriptive_features( odd_template )
    odd_template_diff_f = descriptive_features( np.diff( odd_template ))
    
    def waves( template ):
        
        cA, cD4, cD3, cD2, cD1 = wavedec( template, wavelet = "db2", level = 4 )
        return np.concatenate(( cA, cD4, cD3, cD2 ))

    #get wave coeffs for each template
    wave_matrix = np.apply_along_axis( waves, 1, templates )

    #descriptive statistics about the wave coeffs over all templates, flattened
    #this works because every template has exactly 180 datapoints
    wave_f = np.apply_along_axis( descriptive_features, 0, wave_matrix ).flatten( )
    
    ecg_features = np.concatenate(( signal_f, signal_diff_f, peak_f, peak_diff_f, mean_template_f, mean_template_diff_f, heart_rate_f, heart_rate_diff_f, deviation_f, deviation_diff_f, odd_template_f, odd_template_diff_f, heart_rate_ts_f, heart_rate_ts_diff_f, wave_f, ar_f ))
    
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

def train_random_forest( max_depth, X_train, y_train, X_val = None, y_val = None ):
    
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import f1_score
    
    score = lambda y, y_hat : 1 - f1_score( y, y_hat, average = "micro" )

    clf = GradientBoostingClassifier( max_depth = max_depth, random_state = 0, learning_rate = 0.05, n_estimators = 500, min_samples_split = 20, max_features = 0.2 )
    clf.fit( X_train, y_train )
    
    train_losses = np.repeat([ score( y_train, clf.predict( X_train ))], 2 )
    predict = lambda X: clf.predict( X )
    
    if not X_val is None and not y_val is None:
        
        val_losses = np.repeat([ score( y_val, clf.predict( X_val ))], 2 )
        return train_losses, val_losses, predict
    else:
    
        return train_losses, predict

