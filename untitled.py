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
