# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 21:43:53 2015

@author: Daniel
"""
from pylab import load
import matplotlib.pyplot as plt
import numpy as np

BATCH_JUMP=100
#results=load('Results/Shakespeare/TASK_D5_W200_SL1000_P_R0.save')
results=load('Results/Shakespeare/TASK_D20_W100_SL1000.save')

train_error=results[1]
orthogonality_error=results[2]
L=len(train_error)

fig, ax1 = plt.subplots()
t = range(0,L*BATCH_JUMP,BATCH_JUMP)

ax1.plot(t, train_error, 'b-')
ax1.set_xlabel('MinBatch #')
# Make the y-axis label and tick labels match the line color.
ax1.set_ylabel('Training error', color='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b')
ax1.annotate('Training error=%.3f' % (train_error[-1]), xy=(t[-1], train_error[-1]),
            xytext=(np.floor(0.6*t[-1]), train_error[-1]*1.3),
            arrowprops=dict(facecolor='black', shrink=0.05))

ax2 = ax1.twinx()
ax2.plot(t, orthogonality_error, 'r.')
ax2.set_ylabel('Orthogonality error', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
plt.show()

#%% Multiple plots
fig, ax1 = plt.subplots()
fig, ax2 = plt.subplots()
handles=[]
labels=[]
gain_list=[1,1.001,1.005,1.01,1.05,1.1]
for GAIN in gain_list:
    results=load('Results/Shakespeare/TASK_D2_W200_G%f_SL200.save' % (GAIN))
    train_error=results[1]
    orthogonality_error=results[2]
    L=len(train_error)
    t = range(0,L*BATCH_JUMP,BATCH_JUMP)
    ax1.plot(t, train_error,label=str(GAIN))
    ax2.plot(t, orthogonality_error,label=str(GAIN))
    
ax1.legend(loc=2)
ax2.legend(loc=2)
plt.show()