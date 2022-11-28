import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import find_peaks



def compute_deviation_from_avg(ts, left_window, right_window):
    # a preprocessing step for the detect_peaks() function that uses the find_peaks() from scipy.signal
    ts = np.array(ts)
    div_from_avg = np.zeros(len(ts))
    for idx in range(len(ts[left_window:-right_window])):
        window = abs(ts[idx-left_window : idx+right_window])
        if len(window) == 0:
            avg = 0
        else:
            avg = window.mean() #np.median(window)
        div_from_avg[idx] = abs(ts[idx] - avg)
    
    return div_from_avg



def score_peak_periodicity(peaks, error_margin_perc = 10):
    # evaluates how many of the intervals between the peaks have the same length (up to an error_margin)
    last_idx = 0
    period_lengths = []
    if not isinstance(peaks, list) and not isinstance(peaks, np.ndarray):
        return 0
    for i in range(len(peaks)):
        if peaks[i] == 1:
            period_lengths.append(i - last_idx)
            last_idx = i

    if len(period_lengths) == 0:
        return 0
    else: 
        #avg = sum(period_lengths)/len(period_lengths)
        avg = np.median(period_lengths)
    periods_of_proper_length = sum([1 if (avg*(1-error_margin_perc/100) < x and x < avg*(1+error_margin_perc/100)) else 0 for x in period_lengths]) 
    score = periods_of_proper_length / len(period_lengths)

    return score



def filter_out_wide_peaks(ts, peak_mask, ratio_peak_windowavg = 2, left_window = 4, right_window = 4):
    for idx in range(len(peak_mask)):
        if peak_mask[idx] == 1:
            window = ts[idx - left_window :  idx + right_window]
            window = np.clip(window, a_min = 0, a_max = None)
            if len(window) == 0:
                window_avg = 0
            else: 
                window_avg = window.mean()

            if window_avg == 0:
                ratio = 0
            else:
                ratio = ts[idx] / window_avg
            
            if ratio < ratio_peak_windowavg:
                peak_mask[idx] = 0

    return peak_mask



def refine_peaks(peak_mask, ts):
    # Wiggle the found peaks to the left/right as long as they increase in value
    changes = True
    nr_changes = 0
    while changes:
        # check if the direct neighbours have a higher value -> if yes change location of the peak
        changes = False
        for idx in range(len(peak_mask)-1):
            if peak_mask[idx] == 1:
                peak_mask_val = ts[idx]
                left_neighbour = ts[idx - 1]
                right_neighbour = ts[idx + 1]
                if left_neighbour > peak_mask_val:
                    peak_mask[idx] = 0
                    peak_mask[idx - 1] = 1
                    peak_mask_val = left_neighbour
                    changes = True
                    nr_changes += 1
                if right_neighbour > peak_mask_val:
                    peak_mask[idx] = 0
                    peak_mask[idx + 1] = 1
                    changes = True
                    nr_changes += 1
    #print("Number of wiggled indeces: "+str(nr_changes))
    return peak_mask, ts



def filter_periodicity_hazard_peaks(ts, peaks, error_margin_perc = 20):
    # filter out the points seperating two periods that together would roughly compute the median period length

    # compute the median period width    
    last_idx = 0
    period_lengths = []
    for i in range(len(peaks)):
        if peaks[i] == 1:
            period_lengths.append((i - last_idx, last_idx, i))
            last_idx = i

    # delete the first period length, since it includes noise

    if len(period_lengths) == 0:
        return np.asarray([])
    else: 
        avg_period_length = np.median([len for len, a, b, in period_lengths])
        #avg_period_length = sum(period_lengths)/len(period_lengths)
    #print("Periods: "+str(period_lengths))
    #print("Median period length: "+str(avg_period_length))

    hazard_peaks = peaks.copy()
    for period_idx in range(len(period_lengths)-1):
        this_period_len = period_lengths[period_idx][0]
        div_from_median_this_period = abs(this_period_len - avg_period_length)

        if avg_period_length*(1-error_margin_perc/100) > this_period_len:
            # this period is destroying the periodicity score due to too high length
            #print("This period is a hazard: "+str(period_idx) + ", has length "+str(this_period_len))
            next_period_len = period_lengths[period_idx + 1][0]
            combined_period_len = next_period_len + this_period_len
            #print("Combined period has length "+str(combined_period_len))
            div_from_median_this_period = abs(this_period_len - avg_period_length)
            div_from_median_combined_period = abs(combined_period_len - avg_period_length)

            if div_from_median_combined_period < div_from_median_this_period:
                #print("The combined periods are better than the single period: "+str(this_period_len)+" vs "+str(combined_period_len))
                hazardes_peak_idx = period_lengths[period_idx][2]

            #if avg_period_length*(1-error_margin_perc/100) < combined_period_len or combined_period_len < avg_period_length*(1+error_margin_perc/100):
                # if the combined period is again in the error of margin we remove the signal seperating them from our mask
                # go to the corresponding hazardes peak -> set it to 0
                hazard_peaks[hazardes_peak_idx] = 0
                unchanged_peaks_score = score_peak_periodicity(peaks, error_margin_perc = 20)
                hazard_score = score_peak_periodicity(hazard_peaks, error_margin_perc = 20)

                if hazard_score > unchanged_peaks_score:
                    # take over the peak change in case it improved the periodicity score
                    #print("Removing this peak helped: "+str(hazardes_peak_idx))
                    #print("Periodicity Score after hazard removel: "+str(hazard_score))
                    #show_detected_peaks(ts, hazard_peaks)
                    return hazard_peaks
    
    return hazard_peaks


def filter_periodicity_hazard_gaps(ts, peaks, error_margin_perc = 80):
    # WORK IN PROGRESS

    # look for periods that are exceptionally long -> there should probably be a peak inside of those gaps
    print("Now in gap detection")

    # compute the median period length
    last_idx = 0
    period_lengths = []
    for i in range(len(peaks)):
        if peaks[i] == 1:
            period_lengths.append((i - last_idx, last_idx, i))
            last_idx = i

    # delete the first period length, since it includes noise
    period_lengths = period_lengths[1:-1]
    if len(period_lengths) == 0:
        return 0
    else: 
        avg_period_length = np.median([len for len, a, b, in period_lengths])
        #avg_period_length = sum(period_lengths)/len(period_lengths)
    #print("Periods: "+str(period_lengths))
    print("Median period length: "+str(avg_period_length))

    hazard_peaks = peaks.copy()
    
    # if a gap is error_margin_perc larger than the median gap it is probable we missed a beat
    for period_idx in range(len(period_lengths)-1):
            this_period_len = period_lengths[period_idx][0]
            div_from_median_this_period = abs(this_period_len - avg_period_length)

            if avg_period_length*(1+error_margin_perc/100) < this_period_len:
                # this is a suspeciously long period -> there should probably be a peak in here
                # do a more sensitive peak detection in this filtered period
                print(period_lengths[period_idx])
                window_left_idx = period_lengths[period_idx][1]
                window_right_idx = period_lengths[period_idx][2]
                window_ts = ts[window_left_idx: window_right_idx]
                print(window_ts)
                print(window_ts.shape)

    return np.asarray([0])



def detect_peaks_hazard_filtering(input_ts, visualize, verbose=False):

    # preprocess the time series
    modified_ts = compute_deviation_from_avg(ts = input_ts, left_window = 20, right_window = 10)    # TODO: make these window sizes dependent on how high frequency the data overall is
    modified_ts= compute_deviation_from_avg(ts = modified_ts, left_window = 4, right_window = 2)

    # do peak detection on the preprocessed ts
    peaks_locations, _ = find_peaks(modified_ts, height=10, distance=100, width=(0, 10), prominence=30)     # TODO: also consider working with the threshold parameter!
    peak_mask = np.zeros(len(input_ts))
    peak_mask[peaks_locations-5] = 1 # the 5 works very well ... but kind of random ...

    # refine the position of the peaks
    refined_peak_mask, _ = refine_peaks(peak_mask, input_ts)
    periodicity_score = score_peak_periodicity(refined_peak_mask, error_margin_perc = 20)
    if verbose:
        print("Periodicity Score: "+str(round(periodicity_score, 2)))
    if visualize:
        show_detected_peaks(input_ts, peak_mask)
    # Seems like a score above 0.8 (at 20% margin) is very good -> almost always completely correct classification -> filter by this

    # keep removing hazards while the mask changes -> using filter_periodicity_hazards()
    last_peaks = refined_peak_mask.copy()
    while True:
        periodicity_hazard_peaks = filter_periodicity_hazard_peaks(input_ts, last_peaks, error_margin_perc = 20)
        hazard_score = score_peak_periodicity(periodicity_hazard_peaks, error_margin_perc = 20)

        if np.array_equal(last_peaks, periodicity_hazard_peaks):
            break    
        else:
            #print("Critical peak periodicity - deploy hazard removal - New Periodicity Score: "+str(hazard_score))
            if verbose:
                print("Used hazard detection - periodicity score: "+str(round(periodicity_score, 2))+" -> "+str(round(hazard_score,2)))
            if visualize:
                show_detected_peaks(input_ts, periodicity_hazard_peaks)
            last_peaks = periodicity_hazard_peaks.copy()
            periodicity_score = hazard_score


    # test --------
    #filter_periodicity_hazard_gaps(input_ts, last_peaks, error_margin_perc = 80)
    # test --------


    # filter out wide peaks if that increases the periodicity score
    slim_peaks_mask = filter_out_wide_peaks(input_ts, peak_mask, ratio_peak_windowavg = 5, left_window = 20, right_window = 20)
    periodicity_score_slimmed = score_peak_periodicity(slim_peaks_mask, error_margin_perc = 20)
    if periodicity_score_slimmed > periodicity_score:
        refined_peak_mask = slim_peaks_mask
        if verbose:
            print("Used slimming - periodicity score: "+str(round(periodicity_score, 2))+" -> "+str(round(periodicity_score_slimmed,2)))
        periodicity_score = periodicity_score_slimmed

    if periodicity_score < 0.6 and verbose:
        print("Having a hard time with evaluating/detecting correct peaks.")

    return last_peaks, input_ts, periodicity_score





def show_detected_peaks(ts, peak_mask):
    # plot the resulting peak detection
    plt.plot(ts)
    #plt.plot(modified_ts)
    plt.plot((peak_mask*ts), "x")
    #plt.plot(np.zeros_like(modified_ts), "--", color="gray")
    plt.show()



def execute_peak_detection(X, first_idx=0, last_idx=None, visualize = False, verbose=False):
    # iterated through all datapoint samples in train_X and shows the detected peaks

    if last_idx == None:
        last_idx = len(X)-1
    detected_peaks_mask_ts_list = []

    '''   for i in tqdm(range(first_idx, last_idx)):
        if verbose:
            print("\nIndex of datapoint: "+str(i))
        #ts = np.asarray([int(x) for x in (X[i].split(","))])
        this_ts = X[i]
        peak_mask, output_ts, score = detect_peaks_hazard_filtering(this_ts, visualize = visualize, verbose = verbose) #peak_detection
        detected_peaks_mask_ts_list.append((i+1, score, peak_mask, output_ts))
    '''
    for i,this_ts in tqdm(enumerate(X)):
        if verbose:
            print("\nIndex of datapoint: "+str(i))
        #ts = np.asarray([int(x) for x in (X[i].split(","))])
        peak_mask, output_ts, score = detect_peaks_hazard_filtering(this_ts, visualize = visualize, verbose = verbose) #peak_detection
        detected_peaks_mask_ts_list.append((i+1, score, peak_mask, output_ts))

    return detected_peaks_mask_ts_list




def pipeline_stage_detect_peaks(data_dict, feature_name_line_included_in_X= False, train_X_file_path = None, test_X_file_path= None, **args):
#def pipeline_stage_detect_peaks(data_dict: dict, feature_name_line_included_in_X:bool = False, train_X_file_path:str = None, test_X_file_path:str = None, **args):
    '''
    This function provides a basic heartbeat peak detection on time series ECG data.
    A periodicity score is introduced to evaluate how rhythmic the found peaks are. This can be used to estimate the performance of the peak detection.
    The periodicity score can also be given to a classification algorithm as a feature - assuming the peak detection is roughly correct.

    Inputs:
        input_dict (dict): either this dict contains a field "train_X" and "test_X" - each containing a list of datapoints. Each datapoint is a np.array of timeseries ECG measurements.
            -> if input dict does not contain train_X/test_X, it will be loaded from the provided paths "train_X_file_path"/"test_X_file_path" - e.g. "test_X.csv".
        
        feature_name_line_included_in_X (bool): if the feature name line in the training data is included in the given train_X/test_X, this has to be set to True, otherwise False.

    Returns:
        output_dict (dict): a copy of the input_dict, but with the added keys "train_peak_tuples" and "test_peak_tuples".
            Each of them has as value a list of elements of the following type: (datapoint_index_in_given_list: int, periodicity_score: double, peak_mask: list, analysed_datapoints_as_ts: list)
            peak_mask: 0 for indeces that are not a peak, 1 for indices that are a peak in analysed_datapoints_as_ts
    '''

    print("Loading data ...")
    if train_X_file_path != None:
        train_X_file = open(train_X_file_path, 'r')
        train_X_str = train_X_file.readlines()
        train_X = [(np.asarray([int(x) for x in (line.split(","))])) for line in train_X_str[int(feature_name_line_included_in_X):]]
    else:
        train_X = data_dict["X_train"]
        # might have to do some nan filtering here since the code takes


    if test_X_file_path != None:
        test_X_file = open(test_X_file_path, 'r')
        test_X_str = test_X_file.readlines()
        test_X = [(np.asarray([int(x) for x in (line.split(","))])) for line in test_X_str[int(feature_name_line_included_in_X):]]
    else:
        test_X = data_dict["X_test"]

    # starting from
    # first_idx and last_idx are possibilities to only work on a subset of the given X data - in default (first_idx=1, last_idx=None) the whole given dataset is covered 
    print("Detecting peaks in the train set ...")
    train_peak_tuples = execute_peak_detection(train_X, first_idx = 0, last_idx = None, visualize=False, verbose=False)
    
    print("Detecting peaks in the test set ...")
    test_peak_tuples = execute_peak_detection(test_X, first_idx = 0, last_idx = None, visualize=False, verbose=False)

    # train_peak_tuples: a list of elements (index_in_given_list: int, periodicity_score: double, peak_mask: list, analysed_datapoints_as_ts: list)
    output_dict = data_dict.copy()
    output_dict["X_train_peak_tuples"] = train_peak_tuples
    output_dict["X_test_peak_tuples"] = test_peak_tuples

    print("Done.")

    return output_dict



'''
# Lets execute this code!

# Load the data
train_X_file = open("X_train.csv", 'r')
train_X_str = train_X_file.readlines()
train_X = [(np.asarray([int(x) for x in (line.split(","))])) for line in train_X_str[1:]]  # first line has "feature names"

test_X_file = open("X_test.csv", 'r')
test_X_str = test_X_file.readlines()
test_X = [(np.asarray([int(x) for x in (line.split(","))])) for line in test_X_str[1:]]  # first line has "feature names"

# create dictionary to feed into the pipeline
input_dict = {}
input_dict["X_train"] = train_X[627:650]
input_dict["X_test"] = test_X[550:650]


# The following can be used to get an insight into the detected peaks
# visualizes all samples from sample nr (row-nr) 1234 to sample nr (row-nr) 1254
#output_tuples_of_scores_peaks_ts = execute_peak_detection(train_X, first_idx = 628, last_idx = 1254, visualize=True, verbose=True)
#print(output_tuples_of_scores_peaks_ts)


# This is how the peak detection should be called in the context of the pipline
# output_dict = pipeline_stage_detect_peaks(input_dict = {}, feature_name_line_included_in_X = True, train_X_file_path="X_train.csv", test_X_file_path="X_test.csv")
output_dict = pipeline_stage_detect_peaks(input_dict)
print([score for i, score, a1, a2 in output_dict["train_X_peak_tuples"]])
print([score for i, score, a1, a2 in output_dict["test_X_peak_tuples"]])
print(output_dict.keys())
'''
