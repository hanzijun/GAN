from scipy import signal

def butterworth(matrix):
    b, a = signal.butter(5, 0.3, 'low')
    sf = signal.filtfilt(b, a, matrix)
    return  sf