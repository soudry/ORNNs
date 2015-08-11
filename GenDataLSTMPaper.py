# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 16:12:37 2015

@author: Daniel
"""
# A bunch of data generation functions for some of the tasks from 
# the LSTM paper: Hochreiter, S., Hochreiter, S., Schmidhuber, J., & Schmidhuber, J. (1997). Long Short-term Memory. Neural Computation, 9(8), 1735–1780. doi:10.1.1.56.7752
# Based on the Matlab implementation in: http://minds.jacobs-university.de/sites/default/files/uploads/papers/LSTMexperimentsMatlab.zip 

import numpy as np

def AddTask(T, N):
# generating training data for the Schmidhuber addition task
# T is min length of addition sequence, N is Nr of generated samples
# dataset is array of size ceil(11 * T / 10) x 3 x N, where the first column
# per page contains the addition candidates, the second the trigger bits,
# and the last in its first row the total used length of the current
# series, and in its second row the required normalized sum.

    maxlength = np.ceil(11 * T / 10)
    dataset = np.zeros((maxlength, 3, N))
    for ii in range(N):
        # compute current useful length
        Tprime = T + np.ceil(np.random.rand() * T / 10);
        # compute first add index
        T1 = np.ceil(np.random.rand() * T / 10);
        T2 = np.ceil(np.random.rand() * T / 2);
        while T1 >= T2:
            T2 = np.ceil(np.random.rand() * T / 2)
        # fill first column
        dataset[0:Tprime,0,ii] = np.random.rand(Tprime)
        
        # fill second column
        dataset[T1,1,ii] = 1
        dataset[T2,1,ii] = -1 #here I modified matlab code where this was 1, since in the paper it was -1
        # fill third column
        dataset[Tprime-1,2,ii] = (dataset[T1,0,ii] + dataset[T2,0,ii]) / 2
        
    return dataset
        
        
