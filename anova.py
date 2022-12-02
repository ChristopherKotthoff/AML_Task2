import pandas as pd
import numpy as np
from operator import itemgetter

def anova( data_dict, anova_percentage, ** args ):
    
    train_X, train_y, test_X = itemgetter( "X_train", "y_train", "X_test" )( data_dict )

    n_bins = 4
    y = train_y.transpose( )[ 0 ]
    X = train_X
    indicator_f = lambda i: y == i
    indicators = [ indicator_f( i ) for i in range( n_bins )]

    #analyze statistics of each feature per bin (i.e. E[x_i | y])
    total_variance = X.var( axis = 0 )

    def bin_f( i ):

        X_bin = X[ indicators[ i ], : ]
        y_bin = y[ indicators[ i ]]

        return { 
            "X": X_bin, 
            "y": y_bin, 
            "mean": X_bin.mean( axis = 0 ), 
            "variance": X_bin.var( axis = 0 )
        }

    bins = [ bin_f( i ) for i in range( n_bins )]

    def one_way_anova( feature_index ):

        variance_across_bin_means = np.array([ b[ "mean" ][ feature_index ] for b in bins ]).var( )
        variance_across_samples = total_variance[ feature_index ]

        f_statistic = variance_across_bin_means / variance_across_samples

        #a low f statistic indicates (central limit theorem) an uninformative conditonal E[x_i | y]
        return f_statistic

    # calculate the f statistic for each feature
    f_stats = np.array([ one_way_anova( idx ) for idx in range( X.shape[ 1 ])])

    #keep the best percentage of features
    cutoff = np.sort( f_stats )[ -round( anova_percentage * len( f_stats ))]
    keeps = f_stats > cutoff
    print( f"keeping { sum( keeps )} features" )

    #visualize distribution of f statistic
    import matplotlib.pyplot as plt

    plt.hist( f_stats, bins = 7 )
    plt.show( )

    def filter_array( arr ):

        return arr[ :, keeps ]

    data_dict[ "X_train" ] = filter_array( train_X )
    data_dict[ "X_test" ] = filter_array( test_X )

    return data_dict

