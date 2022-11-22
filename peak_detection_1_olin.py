import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


train_X_file = open("original_data/X_train.csv", 'r')
train_X = train_X_file.readlines()



def compute_deviation_from_avg(ts, left_window, right_window):
    # a preprocessing step for the detect_peaks() function that uses the find_peaks() from scipy.signal
    div_from_avg = np.zeros(len(ts))
    for idx in range(len(ts[left_window:-right_window])):
        window = abs(ts[idx-left_window : idx+right_window])
        avg = window.mean() #np.median(window)
        div_from_avg[idx] = abs(ts[idx] - avg)
    
    return div_from_avg



def score_peak_periodicity(peaks, error_margin_perc = 10):
    # evaluates how many of the intervals between the peaks have the same length (up to an error_margin)
    last_idx = 0
    period_lengths = []
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
            window_avg = window.mean()
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


def detect_peaks(input_ts):

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
    print("Periodicity Score: "+str(round(periodicity_score, 2)))
    # Seems like a score above 0.8 (at 20% margin) is very good -> almost always completely correct classification -> filter by this

    show_detected_peaks(input_ts, peak_mask)
    if periodicity_score < 0.75:
        slim_peaks_mask = filter_out_wide_peaks(input_ts, peak_mask, ratio_peak_windowavg = 2, left_window = 20, right_window = 20)
        periodicity_score_adjusted = score_peak_periodicity(slim_peaks_mask, error_margin_perc = 20)
        print("Critical peak periodicity - deployed wide peak filter - New Periodicity Score: "+str(periodicity_score_adjusted))
        show_detected_peaks(input_ts, slim_peaks_mask)
        if periodicity_score_adjusted < 0.65:
            print("CRITICAL: please check correctness of peak detection !!!")
            '''
            show_detected_peaks(input_ts, slim_peaks_mask)
            periodicity_hazard_peaks = filter_periodicity_hazards(input_ts, slim_peaks_mask, error_margin_perc = 20)
            hazard_score = score_peak_periodicity(periodicity_hazard_peaks, error_margin_perc = 20)
            print("Critical peak periodicity - deploy hazard removal - New Periodicity Score: "+str(hazard_score))
            show_detected_peaks(input_ts, periodicity_hazard_peaks)
            '''


        # print("Using ARMA for peak prediction...")
        # arma_signals = arma_thresholding_algo(input_ts, lag = 250, threshold = 5, influence = 0)["signals"]
        # show_detected_peaks(input_ts, arma_signals)

        # try to filter out the peaks that have too much width (compute the avg over the neighbouring points - if its not way lower than the peak we sort it out)
    return refined_peak_mask, input_ts


def show_detected_peaks(ts, peak_mask):
    # plot the resulting peak detection
    plt.plot(ts)
    #plt.plot(modified_ts)
    plt.plot((peak_mask*ts), "x")
    #plt.plot(np.zeros_like(modified_ts), "--", color="gray")
    plt.show()


def present_peak_detection(train_X, first_idx=1, last_idx=10):
    # iterated through all datapoint samples in train_X and shows the detected peaks
    for i in range(first_idx, last_idx):
        print(i)
        ts = np.asarray([int(x) for x in (train_X[i].split(","))])
        detect_peaks(ts)





present_peak_detection(train_X, 1234, 2345) # visualizes all samples from sample nr (row-nr) 1234 to sample nr (row-nr) 2345
# execute file in debug mode on vs code to go through the detected peaks one by one.
# if a peak detection is not good enough according to a "periodicity score" addtional functions will be applied to change the detected peaks
# -> image will reapear with improved peak locations


'''
ts = np.asarray([int(x) for x in (train_X[2].split(","))])
peaks, _ = detect_peaks(ts)
score_peak_periodicity(peaks, error_margin_perc = 10)
'''