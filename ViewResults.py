# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 21:43:53 2015

@author: Daniel
"""
from pylab import load
import matplotlib.pyplot as plt
import numpy as np
plt.close("all")

#results=load('Results/Shakespeare/TASK_D5_W200_SL1000_P_R0.save')
#results=load('Results/Wiki_2G/TASK_D5_W727_SL250_P_R0.save')
#results=load('Results/Wiki_2G/TASK_D5_W706_SL250.save')
#results=load('Results/Wiki_2G/TASK_D5_W706_SL250_WithNANs.save')
results=load('Results/Wiki_2G/TASK_D5W706A_SL250B50.save')

BATCH_JUMP=results['params']['training']['STAT_SKIP']
VALID_JUMP=results['params']['training']['VALID_SKIP']
train_error=results['track_train_error']
orthogonality_error=results['track_orthogonality']
trace_error=results['track_trace_WW']
trace_error=[x-1 for x in trace_error]
test_error=results['track_test_error']
valid_error=results['track_valid_error']

time=results['total_time'] # in minutes
L=len(train_error)

fig, ax1 = plt.subplots()
t = range(0,L*BATCH_JUMP,BATCH_JUMP)
t_valid = range(VALID_JUMP,(len(valid_error)+1)*VALID_JUMP,VALID_JUMP)

ax1.plot(t, train_error, 'b-',label='train error')
ax1.plot(t_valid, valid_error, 'k--', linewidth=3.0,label='valid error')
ax1.plot(t_valid, test_error, 'r:', linewidth=3.0,label='test error')
ax1.set_xlabel('MinBatch #')

best_valid_error_index=valid_error.index(min(valid_error))
best_valid_error_time=(best_valid_error_index+1)*VALID_JUMP
Test_error_bVe=test_error[best_valid_error_index] # test error when best validation error is achieved
ax1.annotate('Validation-based test error=%.3f' % (Test_error_bVe), xy=(best_valid_error_time, Test_error_bVe),
            xytext=(np.floor(0.5*best_valid_error_time), Test_error_bVe*2),
            arrowprops=dict(facecolor='black', shrink=0.01))
ax1.grid()
ax1.legend(loc=2)

fig2, ax2 = plt.subplots()
ax2.plot(t, orthogonality_error, 'r.')
ax2.set_ylabel('Orthogonality error', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
#ax2.set_ylim([0,orthogonality_error[-1]*1.5])
    
ax3=ax2.twinx()
ax3.plot(t, trace_error, label='trace error', color='b')
# Make the y-axis label and tick labels match the line color.
ax3.set_ylabel('Trace error', color='b')
for tl in ax3.get_yticklabels():
    tl.set_color('b')
#ax3.set_ylim([0,trace_error[-1]*1.5])

plt.show()
#%% More analysis

#u,s,v=np.linalg.svd(jacobian[1])
#plt.plot(s)

fig, ax = plt.subplots()
ax.imshow(hidden_units[1][2,:,:])


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
plt.grid()
plt.show()