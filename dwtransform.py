# -*- coding: utf-8 -*-

from pywt import wavedec,upcoef,Wavelet,dwt,waverec
import numpy as np

def ReconUsingUpcoef(signal, wname='db4', level=3):

    w = Wavelet(wname)
    X = signal
    #coeffs = wavedec(a, w, mode='symmetric', level=level, axis=-1)
    coeffs = []

    for i in range(level):
        a, d = dwt(X, w, mode='symmetric', axis=-1)
        coeffs.append(d)

    coeffs.append(a)
    coeffs.reverse()

    ca = coeffs[0]
    n = len(X)
    return upcoef('a', ca, w, take=n, level=level)

if __name__ == "__main__":

    x=np.array([10,12,11,14,11,13,11,12])
    print ReconUsingUpcoef(x)
    # print ()       # orignal data