# -*- coding=utf-8 -*-
# Kalman filter example demo in Python

Q = 1e-5  # process variance
R = 0.1 **  3.2# estimate of measurement variance, change to see effect
import numpy
import pylab
class kalmanfilter():
    def __init__(
            self,
            sequence):
        self.sequence = sequence
        self.n_iter = len(sequence)
        self.sz = (self.n_iter,)  # size of array
        self.noise = numpy.random.normal(self.sequence, 0.1, size=self.sz)  # observations (normal about x, sigma=0.1)
    # allocate space for arrays
        self.filterOperation()
        # self.display()

    def filterOperation(self):
        self.xhat = numpy.zeros(self.sz)  # a posteri estimate of x
        self.P = numpy.zeros(self.sz)  # a posteri error estimate
        self.xhatminus = numpy.zeros(self.sz)  # a priori estimate of x
        self.Pminus = numpy.zeros(self.sz)  # a priori error estimate
        self.KalmanGain = numpy.zeros(self.sz)  # gain or blending factor
        # intial guesses
        self.xhat[0] = self.sequence[0]
        self.P[0] = 10

        for k in range(1, self.n_iter):
            # time update
            self.xhatminus[k] = self.xhat[k - 1]  # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0
            self.Pminus[k] = self.P[k - 1] + Q  # P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1
            # measurement update
            self.KalmanGain[k] = self.Pminus[k] / (self.Pminus[k] + R)  # Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
            self.xhat[k] = self.xhatminus[k] + self.KalmanGain[k] * (self.noise[k] - self.xhatminus[k])  # X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
            self.P[k] = (1 - self.KalmanGain[k]) * self.Pminus[k]  # P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1

    def display(self):
        pylab.figure()
        pylab.plot(self.noise, 'k+', label='noisy measurements')  # 测量值
        pylab.plot(self.xhat, 'b-', label='kalman-based results')  # 过滤后的值
        pylab.plot(self.sequence, color='g', label='original value')  # 系统值
        pylab.legend(loc = 'best')
        pylab.xlabel('Iteration')
        pylab.ylabel('Voltage')
        pylab.ylim(0,300)
        pylab.show()

    def feedback(self):
        return self.xhat

if __name__ == "__main__":
    a = [10.3, 2.83, 20.62, 7.81, 9.43, 11.4, 6.08, 9.49, 5.83, 8.06, 12.04, 11.4, 5.1, 14.14, 9.06, 6.71, 8.06, 7.62, 7.28,
     8.06, 6.32, 9.49, 9.0, 7.62, 5.0, 13.45, 5.39, 9.0, 8.49, 8.94, 4.12, 3.0, 4.47, 9.22, 3.61, 5.1, 2.83, 8.06, 3.16,
     2.0, 6.71, 6.08, 8.06, 5.0, 8.0, 8.25, 2.83, 9.06, 7.07, 5.83, 5.1, 5.1, 8.0, 5.66, 0.0, 5.1, 9.22, 12.04, 8.0,
     8.54, 9.43, 6.4, 20.62, 12.04, 13.34, 7.81, 14.0, 9.22, 8.94, 6.4, 12.04, 9.49, 12.17, 35.36, 13.89, 10.3, 9.22,
     8.25, 4.47, 5.39, 11.7, 9.85, 7.07, 12.17, 12.04, 3.61, 5.83, 9.22, 12.21, 5.39, 9.85, 6.32, 24.02, 10.82, 3.16,
     7.07, 13.93, 10.77, 7.62, 13.93, 2.0, 5.83, 9.22, 9.22, 7.21, 11.4, 9.22, 6.71, 5.83, 7.07, 7.28, 8.94, 37.74,
     10.3, 11.18, 13.42, 9.43, 3.0, 5.39]
    kalmanfilter(a)
