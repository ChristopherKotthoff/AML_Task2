from biosppy.signals.tools import rms_error, synchronize
from biosppy.stats import pearson_correlation
import numpy as np


def allignedSimilarity(median_ts1, median_ts2):
    '''
    Alligns the two incoming median "templates". Then computes their similarity according to pearson correlation and RootMSE
    '''

    # allign the two templates as well as possible
    synched = synchronize(x=median_ts1, y=median_ts2, detrend=True)
    syn_median_ts1 = synched["synch_x"]
    syn_median_ts2 = synched["synch_y"]

    # compute the similarity of the two templates
    pearson_corr = pearson_correlation(x=syn_median_ts1, y=syn_median_ts2)
    error = rms_error(x=syn_median_ts1, y=syn_median_ts2)
    return pearson_corr[0], error[0], syn_median_ts1, syn_median_ts2


def inversionSafeAllignedSimilarity(median_ts1, median_ts2):
    normal_pearson, normal_rmse, n1, n2 = allignedSimilarity(median_ts1, median_ts2)
    inverted_pearson, inverted_rmse, i1, i2 = allignedSimilarity(-median_ts1, median_ts2)

    if normal_pearson < inverted_pearson and normal_rmse > inverted_rmse:
        # if inversion is better in both performance parameters then we assume one of the signals should be inverted
        # print("One of the signals should be inverted!")
        return inverted_pearson, inverted_rmse
    else:
        # either both performance scores are better for the non-inverted version or it is not sure. Both cases just hand over the directly applied performance score
        return normal_pearson, normal_rmse


def predictSingleDatapointSimilarity(data_dict, to_predict, **args):
    '''
    This function does a prediction by taking the similarity (in terms of smallest rmse) over the whole training data and predicting the class of the most similar 23 train datapoints

    # TODO: do smoothing first!

    '''

    train_median_curves = data_dict["X_train"]  # contains the median template of each time series
    train_labels = data_dict["y_train"]

    similarity_list = []
    for i, train_datapoint in enumerate(train_median_curves):
        pearson_corr, rmse = inversionSafeAllignedSimilarity(to_predict, train_datapoint)
        similarity_list.append((pearson_corr, rmse, train_labels[i,0]))
    
    # sort list by rmse 
    similarity_list.sort(reverse=True, key=lambda x: x[1])
    similar_labels = [lab for corr, rmse, lab in similarity_list[:6]]
    
    majority_prediction = max(set(similar_labels), key = similar_labels.count)
    
    return majority_prediction  
    

def predictSimilarity(data_dict, **args):
    '''
    This function applies predictSingleDatapointSimilarity() for every point that has to be predicted in X_test.
    '''

    test_median_curves = data_dict["X_train"]

    predictions = []
    for this_testpoint in test_median_curves:
        pred = predictSingleDatapointSimilarity(data_dict, this_testpoint, **args)
        predictions.append(pred)

    data_dict["y_test"] = np.array(predictions).reshape((-1, 1))

    return data_dict