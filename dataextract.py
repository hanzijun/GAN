#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
import pylab
from DWTfliter import   dwtfilter
import numpy as np


matrix_raw = None
with open('train_samples.pkl', 'rb') as f:
    samples = pickle.load(f)

matrixList = samples.transpose()
matrixList = matrixList[0: 56, :][0]
print matrixList

butter = dwtfilter(matrixList).butterWorth()
dwt1 = dwtfilter(matrixList).filterOperation()
dwt2 = dwtfilter(butter).filterOperation()

pylab.figure()
pylab.plot(matrixList, 'y-', label='raw data')
pylab.plot(butter, 'b-', label='butter')
pylab.plot(dwt1, 'r-', label='dwt')
pylab.plot(dwt2, 'g-', label='dwt')

pylab.show()