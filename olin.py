from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np
from tqdm import tqdm

from operator import itemgetter
from biosppy.signals.ecg import ecg, correct_rpeaks
from pywt import wavedec
from heinrich import apply_along_axis_tqdm, signal_length, descriptive_features
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit


#ecg feature extraction pipeline stage
def ecgExtractOlinFFT12( data_dict, ** args ):
    
    assert "X_train" in data_dict.keys( )
    assert "X_test" in data_dict.keys( )
    
    for key in [ "X_train", "X_test" ]:
        
        assert key in data_dict.keys( )
        print( f"extracting from {key}" )
        
        data_dict[ key ] = apply_along_axis_tqdm( ecg_feature_extract_olin12, 1, data_dict[ key ])
        
    return data_dict


def ecg_feature_extract_olin12( signal ):
    
    length = signal_length( signal )
    clean = signal[ :length ]
    f = lambda s: ecg( s, 300, show = False )
    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = f( clean )
    rpeaks = correct_rpeaks( signal = clean, rpeaks = rpeaks, sampling_rate = 300, tol = 0.1 )
    
    #descriptive statistics about the signal
    signal_f = descriptive_features( clean )
    signal_diff_f = descriptive_features( np.diff( clean ))

    #descriptive statistics about the time between peaks
    period_f = descriptive_features( np.diff(rpeaks["rpeaks"]) )
    period_diff_f = descriptive_features( np.diff(np.diff(rpeaks["rpeaks"])) ) # how the periods lengths change
    
    # Fourier Transform - most relevant frequencies
    freq_f, _, _ = fft_features(signal, nr_freq=12)
    
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
    
    ecg_features = np.concatenate(( signal_f, signal_diff_f, period_f, period_diff_f, freq_f, peak_f, peak_diff_f, mean_template_f, mean_template_diff_f, heart_rate_f, heart_rate_diff_f, deviation_f, deviation_diff_f, odd_template_f, odd_template_diff_f, heart_rate_ts_f, heart_rate_ts_diff_f, wave_f ))
    
    return ecg_features




def train_stacked(X_train, y_train, X_val, y_val, use_cached_states=False, show_validation_perf=False, verbose=0):
    # TODO: take different train/test splits to evaluate the performance development, #if show_validation_perf:

    # create a dictionary for the parameters of the classifiers in the ensamble
    para_dict = {}
    para_dict["random_state"]=42

    # random forest
    para_dict["rf_use"]=True
    para_dict["rf_nestimators"]=100

    # svm classifiers with different kernels
    para_dict["linsvc_use"]=True
    para_dict["rbfsvc_use"]=True
    para_dict["rbfsvc_gamma"]="auto"
    para_dict["rbfsvc_probability"]=True
    para_dict["sigmoidsvc_use"]=True
    para_dict["sigmoidsvc_gamma"]="auto"
    para_dict["sigmoidsvc_probability"]=True
    para_dict["polysvc_use"]=True
    para_dict["polysvc_gamma"]="auto"
    para_dict["polysvc_probability"]=True

    # gradient boosting
    para_dict["gradboost_use"]=True
    para_dict["gradboost_maxdepth"]=5
    para_dict["gradboost_learnrate"]=0.08
    para_dict["gradboost_nestimators"]=500
    para_dict["gradboost_minsamplessplit"]=30
    para_dict["gradboost_maxfeatures"]=0.2

    # mlp
    para_dict["mlp_use"]=True
    para_dict["mlp_layers"]=(800)
    para_dict["mlp_maxiter"]=5000
    para_dict["mlp_learnrate"]="adaptive"
    para_dict["mlp_batchsize"]=32

    # knn classifiers with different distance metrics
    para_dict["uniformknn_use"]=True
    para_dict["uniformknn_nneighbours"]=30
    para_dict["distknn_use"]=True
    para_dict["distknn_nneighbours"]=40

    # ada boost
    para_dict["adaboost_use"]=True
    #Quadratic Discriminant Analysis
    para_dict["quadDiscAna_use"]=False

    # final estimator that combines the results of the ensamble
    para_dict["finalmlp_layers"]=(20, 20)
    para_dict["finalmlp_maxiter"]=5000
    para_dict["finalmlp_learnrate"]="adaptive"
    para_dict["finalmlp_batchsize"]=32
    
    # check if for every estimator in the ensamble it is defined to be used or not
    para_dict_keys = para_dict.keys()
    assert "rf_use" in para_dict_keys
    assert "linsvc_use" in para_dict_keys
    assert "rbfsvc_use" in para_dict_keys
    assert "sigmoidsvc_use" in para_dict_keys
    assert "polysvc_use" in para_dict_keys
    assert "gradboost_use" in para_dict_keys
    assert "mlp_use" in para_dict_keys    
    assert "uniformknn_use" in para_dict_keys
    assert "distknn_use" in para_dict_keys
    assert "adaboost_use" in para_dict_keys
    assert "quadDiscAna_use" in para_dict_keys

    # create the estimators list dependent on para_dict:
    estimators = []
    if para_dict["rf_use"]:
        estimators.append(('rf', RandomForestClassifier(n_estimators=para_dict["rf_nestimators"], random_state=para_dict["random_state"])))
    if para_dict["linsvc_use"]:
        estimators.append(('linsvc', LinearSVC(random_state=para_dict["random_state"])))
    if para_dict["rbfsvc_use"]:
        estimators.append(('rbfsvc', SVC(kernel = 'rbf', gamma=para_dict["rbfsvc_gamma"], probability=para_dict["rbfsvc_probability"], random_state=para_dict["random_state"])))
    if para_dict["sigmoidsvc_use"]:
        estimators.append(('sigmoidsvc', SVC(kernel = 'sigmoid', gamma=para_dict["sigmoidsvc_gamma"], probability=para_dict["sigmoidsvc_probability"], random_state=para_dict["random_state"])))
    if para_dict["polysvc_use"]:
        estimators.append(('polysvc', SVC(kernel = 'poly', gamma=para_dict["polysvc_gamma"], probability=para_dict["polysvc_probability"], random_state=para_dict["random_state"])))
    if para_dict["gradboost_use"]:
        estimators.append(('gradBoost', GradientBoostingClassifier( max_depth = para_dict["gradboost_maxdepth"], learning_rate = para_dict["gradboost_learnrate"], n_estimators = para_dict["gradboost_nestimators"], 
                                                 min_samples_split = para_dict["gradboost_minsamplessplit"], max_features = para_dict["gradboost_maxfeatures"], random_state = para_dict["random_state"] )))
    if para_dict["mlp_use"]:
        estimators.append(('mlp', MLPClassifier(hidden_layer_sizes=para_dict["mlp_layers"], max_iter=para_dict["mlp_maxiter"], learning_rate=para_dict["mlp_learnrate"], batch_size=para_dict["mlp_batchsize"],
                              random_state=para_dict["random_state"])))
    if para_dict["uniformknn_use"]:
        estimators.append(('uniformknn', KNeighborsClassifier(weights='uniform', n_neighbors = para_dict["uniformknn_nneighbours"])))
    if para_dict["distknn_use"]:
        estimators.append(('distknn', KNeighborsClassifier(weights='distance', n_neighbors = para_dict["distknn_nneighbours"])))
    if para_dict["adaboost_use"]:
        estimators.append(('adaboost', AdaBoostClassifier(random_state=para_dict["random_state"])))
    if para_dict["quadDiscAna_use"]:
        estimators.append(('quadDiscAna', QuadraticDiscriminantAnalysis()))
    
    # maybe to not train with all the data... X_train = X_train[:100] for testing etc
    X_train = X_train
    y_train = y_train
    
    # normalize the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_norm = scaler.transform(X_train)
    X_val_norm = scaler.transform(X_val)
    
    # combining the previously defined classifiers into one Stacked Clf
    clf = StackingClassifier(
        estimators=estimators,
        #final_estimator=LogisticRegression(), # try mlp
        final_estimator=MLPClassifier(hidden_layer_sizes=para_dict["finalmlp_layers"], max_iter=para_dict["finalmlp_maxiter"], learning_rate=para_dict["finalmlp_learnrate"], 
                                      batch_size=para_dict["finalmlp_batchsize"], random_state=para_dict["random_state"]),
        verbose = verbose
    )

    # fit the ensemble clf on the train data
    clf.fit(X_train, y_train)
        
    #create a prediction function that deploys the trained stacked classifier
    predict_funct = lambda X: clf.predict(scaler.transform(X))
    
    # define a score
    score = lambda y, y_hat : 1 - f1_score( y, y_hat, average = "micro" )
    train_loss_timeseries = np.repeat([ score( y_train, clf.predict( X_train_norm ))], 2 )
        
    if X_val is not None:
        val_loss_timeseries = np.repeat([ score( y_val, clf.predict( X_val_norm ))], 2 )
        return train_loss_timeseries, val_loss_timeseries, predict_funct
    else:
        return train_loss_timeseries, predict_funct
    
    

    
def stackedClassification(data_dict, stackedClf_useValidationSet, stackedClf_makePrediction, **args):
    
    assert "X_train" in data_dict.keys()
    assert "y_train" in data_dict.keys()
    if stackedClf_useValidationSet:
        assert "X_val" in data_dict.keys()
        assert "y_val" in data_dict.keys()
    if stackedClf_makePrediction:
        assert "X_test" in data_dict.keys()


    if stackedClf_useValidationSet:
        data_dict["train_losses"], data_dict["val_losses"], stacked_predict_funct = train_stacked( X_train=data_dict["X_train"], y_train=data_dict["y_train"], X_val=data_dict["X_val"], y_val=data_dict["y_val"],use_cached_states=False)
        data_dict["y_val_hat"]=stacked_predict_funct(data_dict["X_val"])
        data_dict["y_train_hat"]=stacked_predict_funct(data_dict["X_train"])
    else:
        data_dict["train_losses"], predict_funct = train_stacked(X_train=data_dict["X_train"], y_train=data_dict["y_train"], X_val=None, y_val=None, use_cached_states=False)
        data_dict["y_train_hat"]=stacked_predict_funct(data_dict["X_train"])

    if stackedClf_makePrediction:
        data_dict["y_test"] = stacked_predict_funct(data_dict["X_test"])

    if stackedClf_useValidationSet:
        data_dict["y_val_predicted"] = stacked_predict_funct(data_dict["X_val"])
    
    return data_dict



#ecg feature extraction pipeline stage
def ecgExtractOlinFFT( data_dict, ** args ):
    
    assert "X_train" in data_dict.keys( )
    assert "X_test" in data_dict.keys( )
    
    for key in [ "X_train", "X_test" ]:
        
        assert key in data_dict.keys( )
        print( f"extracting from {key}" )
        
        data_dict[ key ] = apply_along_axis_tqdm( ecg_feature_extract_olin, 1, data_dict[ key ])
        
    return data_dict


def ecg_feature_extract_olin( signal ):
    
    length = signal_length( signal )
    clean = signal[ :length ]
    f = lambda s: ecg( s, 300, show = False )
    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = f( clean )
    rpeaks = correct_rpeaks( signal = clean, rpeaks = rpeaks, sampling_rate = 300, tol = 0.1 )
    
    #descriptive statistics about the signal
    signal_f = descriptive_features( clean )
    signal_diff_f = descriptive_features( np.diff( clean ))

    #descriptive statistics about the time between peaks
    period_f = descriptive_features( np.diff(rpeaks["rpeaks"]) )
    period_diff_f = descriptive_features( np.diff(np.diff(rpeaks["rpeaks"])) ) # how the periods lengths change
    
    # Fourier Transform - most relevant frequencies
    freq_f, _, _ = fft_features(signal, nr_freq=35)
    
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
    
    ecg_features = np.concatenate(( signal_f, signal_diff_f, period_f, period_diff_f, freq_f, peak_f, peak_diff_f, mean_template_f, mean_template_diff_f, heart_rate_f, heart_rate_diff_f, deviation_f, deviation_diff_f, odd_template_f, odd_template_diff_f, heart_rate_ts_f, heart_rate_ts_diff_f, wave_f ))
    
    return ecg_features






def fft_features(clean_signal, nr_freq=12):
    
    if len(clean_signal) == 0:
        return 0
    
    if len(clean_signal)%2 == 1:
        clean_signal = clean_signal[:-1]
    
    SAMPLE_RATE = 300 #Hz
    DURATION = int(len(clean_signal)/SAMPLE_RATE)
    N = SAMPLE_RATE * DURATION
    cut = int((len(clean_signal) - N)/2)
    
    # cut a piece from clean_signal that has 
    if cut == 0:
        cut_signal = clean_signal
    cut_signal = clean_signal[cut:-cut]
    
    # do the Fast Fourier Transform (FFT)
    yf = fft(cut_signal)
    xf = fftfreq(N, 1 / SAMPLE_RATE)
    
    # take the most important frequencies
    pos_xf = xf[:int(len(xf)/2)]
    pos_yf = np.abs(yf)[:int(len(xf)/2)]

    tups = zip(pos_xf, pos_yf)
    sorted_tups = sorted(tups, key=lambda x: x[1], reverse=True)[:nr_freq]
    most_important_freq = [round(freq, 2) for freq, intensity in sorted_tups]   # feed this as feature
    
    #plt.plot(xf, abs(yf))
    
    return most_important_freq, xf, yf
