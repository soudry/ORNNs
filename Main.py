# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 21:20:37 2015

@authors: Colin Raffel & Daniel Soudry
"""

"""ORNN for next character predictionc
"""
import time,cPickle,sys,os
import numpy as np
from pylab import load

import theano,lasagne
import theano.tensor as T

## for debugging purposes:  
#mode=theano.Mode(optimizer='fast_compile')

from util import load_dataset,tangent_grad,retraction,get_file_name,oneHot2string,string2oneHot
#%% Set Parameters
# Network Architecture
HIDDEN_WIDTH = 200 # RNN Hidden layer size 
DEPTH=3 # number of Hideen RNN layers
ALL2OUTPUT=True # Are all hidden layers also directly connected to output? 
LOAD_PREVIOUS=False # should we load connectivity from a previous file?
network={'DEPTH':DEPTH,'HIDDEN_WIDTH':HIDDEN_WIDTH,'ALL2OUTPUT':ALL2OUTPUT,
'LOAD_PREVIOUS':LOAD_PREVIOUS}

# Training Parameters
BATCH_NUM=10000
BATCH_SIZE = 50
STAT_SKIP= 100 # How many Batches to wait before we update learning statistics, and retraction
VALID_SKIP = 1000 # How many Batches to wait before we update validation and test error
SEQUENCE_LENGTH =250 # Length of input sequence into RNN

BURNIN=50 # Ignore this many first characters in the output
dataset_set=['Shakespeare','Wiki_2G']
DATASET=dataset_set[0]
training={'BATCH_NUM':BATCH_NUM,'BATCH_SIZE':BATCH_SIZE,'SEQUENCE_LENGTH':SEQUENCE_LENGTH,
'BURNIN':BURNIN,'DATASET':DATASET,'STAT_SKIP':STAT_SKIP,'VALID_SKIP':VALID_SKIP}

# Algorithm Parameters
ORT_INIT=False # intialize all weights to be orthogonal?
PROJ_GRAD=False # Should we project gradient on tangent space to to the Stiefel Manifold (Orthogonal matrices)?
RETRACT=False # Should we do retraction step?
THRESHOLD=0 #error threshold in which we do the retraction step
GAIN=1 # a multiplicative constant we add to all orthogonal matrices
RETRACT_SKIP=1 # How many Batches to wait before we do retraction
opt_mathods_set=['SGD','ADAM']
OPT_METHOD=opt_mathods_set[0]
algorithm={'ORT_INIT':ORT_INIT,'PROJ_GRAD':PROJ_GRAD,'RETRACT':RETRACT,'THRESHOLD':THRESHOLD,
'GAIN':GAIN,'RETRACT_SKIP':RETRACT_SKIP,'OPT_METHOD':OPT_METHOD}

params={'network':network,'training':training,'algorithm':algorithm}
DO_SAVE=True # should we save results?
save_file_name=get_file_name(params)
#%% Intialize network model
    
data, vocab, data_ranges = load_dataset(DATASET)

# define a list of parameters to orthogonalize (recurrent connectivities)
param2orthogonlize=[]      
# The number of features is number of different letters + 1 unknown letter
FEATURES_NUM=len(vocab)+1
# Construct network

# Input layer
l_in = lasagne.layers.InputLayer(
    (BATCH_SIZE, SEQUENCE_LENGTH-1, FEATURES_NUM)) # the input has -1 sequence elength since we through away the last character (it is only predicted - in the output)
layers_to_concat = []
# All recurrent layer
for dd in range(DEPTH): 
    if ORT_INIT:
        W_in_to_hid_init=lasagne.init.Orthogonal(gain=GAIN)
        W_hid_to_hid_init=lasagne.init.Orthogonal(gain=GAIN)
    else:
        W_in_to_hid_init=lasagne.init.Normal(std=1/np.sqrt(HIDDEN_WIDTH), mean=0.0)
    if dd == 0:
        input_layer = l_in
        if not ORT_INIT:
            W_hid_to_hid_init=lasagne.init.Normal(std=1, mean=0.0)     
    else:
        if not ORT_INIT:
            W_hid_to_hid_init=lasagne.init.Normal(std=1/np.sqrt(HIDDEN_WIDTH), mean=0.0)
        input_layer = l_rec
    l_rec = lasagne.layers.RecurrentLayer(
        input_layer, HIDDEN_WIDTH,
        # Use orthogonal weight initialization
        W_in_to_hid=lasagne.init.Orthogonal(gain=GAIN),
        W_hid_to_hid=lasagne.init.Orthogonal(gain=GAIN),
        nonlinearity=lambda h: T.tanh(h),learn_init=True, name='RNN_%i' % (dd+1))
    param2orthogonlize.append(l_rec.W_hid_to_hid)
    layers_to_concat.append(l_rec)
    
    #  if we use normalized tanh nonlinearity (I think peformance is slightly worse)
    #  W_hid_to_hid nonlinearity=lambda h: 1.7159*T.tanh(2*h/3),learn_init=True)
       
if ALL2OUTPUT: #if we the output to connect to all hidden layers
    rec_outputs=lasagne.layers.ConcatLayer(layers_to_concat,axis=2)           
    output_size=DEPTH*HIDDEN_WIDTH
else: #if we only want the deepest RNN layer as output
    rec_outputs=l_rec
    output_size=HIDDEN_WIDTH

# Remove intial BURNIN steps from output 
l_sliced=lasagne.layers.SliceLayer(rec_outputs, indices=slice(BURNIN,None), axis=1)

# Squash the batch and sequence (non-feature) dimensions     
l_reshape = lasagne.layers.ReshapeLayer(l_sliced, [-1, output_size])

# Compute softmax output
l_out= lasagne.layers.DenseLayer(
    l_reshape, FEATURES_NUM,
    nonlinearity=lasagne.nonlinearities.softmax)

## for debugging purposes:        
#    lasagne.layers.get_output_shape(layers_to_concat)
#    lasagne.layers.get_output_shape(rec_outputs)

# Get Theano expression for network output
network_output = lasagne.layers.get_output(l_out)
# Symbolic vector for target
target = T.matrix('target')
# Compute categorical cross-entropy between prediction and target
loss = T.mean(lasagne.objectives.categorical_crossentropy(
    network_output, target))#/np.log(2)
# Collect all network parameters
all_params = lasagne.layers.get_all_params(l_out)

# Load previous run, if exists, and requested
if LOAD_PREVIOUS==True:
    PREVIOUS_NAME=save_file_name
    if os.path.isfile (PREVIOUS_NAME):
        results=load(PREVIOUS_NAME)  
        for (index,p) in enumerate(all_params):
            p.set_value(results['connectivity'][index])

# Calculate norm of all gradients (before clipping)
MAX_NORM=1
all_grads = T.grad(loss, all_params)
scaled_grads,grad_norm = lasagne.updates.total_norm_constraint(all_grads,MAX_NORM,return_norm=True)

if OPT_METHOD=='ADAM':
    updates = lasagne.updates.adam(loss, all_params)
elif OPT_METHOD=='SGD':
    learning_rate0=0.5
    learning_rate=learning_rate0
    lr=theano.shared(np.asarray(learning_rate,dtype=theano.config.floatX),borrow=True)
    
    updates = lasagne.updates.sgd(scaled_grads, all_params, lr)
else:
    print 'unknown optimization method'

angle=T.constant(0)        
#            angle=theano.typed_list.TypedListType(T.dscalar)('angle')
for param in all_params:
    if param in param2orthogonlize:
        delta=updates[param]-param
        tan_grad=tangent_grad(param,delta)
        angle=angle+T.sqrt((tan_grad**2).sum() / (delta**2).sum())/len(param2orthogonlize)
        if PROJ_GRAD==True:            
            updates[param] = param + tan_grad 
                
retract_updates=[]
for p in param2orthogonlize:
    retract_updates.append((p,GAIN*retraction(p)))

J=theano.gradient.jacobian(loss,param2orthogonlize) 
hidden=lasagne.layers.get_output(layers_to_concat) 

# Compile functions for training and computing output
train = theano.function([l_in.input_var, target], [loss,angle,grad_norm], updates=updates,allow_input_downcast=True)
probe_loss = theano.function([l_in.input_var, target], loss,allow_input_downcast=True)
probe_J = theano.function([l_in.input_var, target], J,allow_input_downcast=True)
retract_w = theano.function([], [], updates=retract_updates,allow_input_downcast=True)
get_output = theano.function([l_in.input_var], network_output,allow_input_downcast=True)
probe_hidden  = theano.function([l_in.input_var], hidden,allow_input_downcast=True)

track_train_error=[]
track_valid_error=[] 
track_test_error=[] 

track_orthogonality=[]
track_trace_WW=[]
track_angle=[]
track_S_min=[]
track_S_max=[]
track_grad_norm=[]


#%%  Training    
start_time = time.clock()
o_error=float("inf")

for batch in range(BATCH_NUM):
    # Sample BATCH_SIZE sequences of length SEQUENCE_LENGTH from train_data
    rand_indices=np.random.choice(data_ranges.train_end - SEQUENCE_LENGTH, BATCH_SIZE)
    next_batch = np.array([string2oneHot(data[n:n + SEQUENCE_LENGTH],vocab) for n in rand_indices])
    # Train with this batch
    loss,angle,grad_norm = train(next_batch[:, :-1], next_batch[:, BURNIN+1:].reshape(-1, next_batch.shape[-1]))
    # Update learning rate if we do SGD
    if OPT_METHOD=='SGD': learning_rate=learning_rate0*(1-batch/BATCH_NUM)  

    # Retract every RETRACT_SKIP batches  
    if (not batch % RETRACT_SKIP) and RETRACT and o_error>=THRESHOLD:
            retract_w()
                
    # Print diagnostics every STAT_SKIP batches                    
    if not batch % STAT_SKIP:
        print "\n######## Iteration: {} Loss: {}".format(batch, loss)
        #Measure time
        end_time = time.clock()
        total_time=(end_time - start_time) / 60. # in minutes
        print >> sys.stderr, ('Running time so far: %.2f minutes' % (total_time))      
        
        # Examine orhtogonality measures
        o_error=0
        trace_error=0
        for p in param2orthogonlize:
            W = p.get_value()
            o_error += np.sum((np.eye(W.shape[0]) - np.dot(W.T, W))**2)
            trace_error+=np.trace(np.dot(W.T, W))/(min(np.shape(W))*len(param2orthogonlize))
  
        print 'Gradient norm = %f' %(grad_norm) 
        print "#### Hid->hid nonorthogonality: {}".format(o_error)
        print '#### Tangent grad angle: %f' % (angle) 
        A=np.eye(HIDDEN_WIDTH)
        index=0
        for p in param2orthogonlize:
            index+=1
            W = p.get_value()
            A=np.dot(W,A)
            U,S,V=np.linalg.svd(W)
            S_mi=np.min(S)
            S_ma=np.min(S)
            print '#### W%i singular values: max=%f, min=%f' % (index,S_mi,S_ma) 
        U,S,V=np.linalg.svd(A)
        S_mi=np.min(S)
        S_ma=np.min(S)
        print '#### End2End singular values: max=%f, min=%f' % (S_mi,S_ma) 
        
        # Show text prediction examples
        print "\n#### Target:"
        print oneHot2string(next_batch[0],vocab)
        print "\n#### Predicted:"
        print oneHot2string(get_output(next_batch[:1]),vocab)
        
        # track results
        track_train_error.append(loss)
        track_angle.append(angle)
        track_orthogonality.append(o_error)
        track_trace_WW.append(trace_error)
        track_S_min.append(S_mi)
        track_S_max.append(S_ma) 
        track_grad_norm.append(grad_norm)           
        
    if ((not (batch+1) % VALID_SKIP) or (batch==(BATCH_NUM-1))):
        # Validation error            
        valid_loss=0
        valid_indices=range(data_ranges.valid_start,data_ranges.valid_end-SEQUENCE_LENGTH*BATCH_SIZE,SEQUENCE_LENGTH*BATCH_SIZE)
        if len(valid_indices)==0: print 'validation set too small! increase data size, or reduce SEQUENCE_LENGTH*BATCH_SIZE'            
        for n in valid_indices:
            batch_indices=range(n,n+SEQUENCE_LENGTH*BATCH_SIZE,SEQUENCE_LENGTH)
            next_batch = np.array([string2oneHot(data[n:n + SEQUENCE_LENGTH],vocab) for n in batch_indices])
            loss= probe_loss(next_batch[:, :-1],next_batch[:, BURNIN+1:].reshape(-1, next_batch.shape[-1]))
            valid_loss += loss/len(valid_indices)
        track_valid_error.append(valid_loss)
        
        # Test error
        test_loss=0
        test_indices=range(data_ranges.test_start,data_ranges.test_end-SEQUENCE_LENGTH*BATCH_SIZE,SEQUENCE_LENGTH*BATCH_SIZE)
        if len(test_indices)==0: print 'test set too small! increase data size, or reduce SEQUENCE_LENGTH*BATCH_SIZE'            
                    
        for n in test_indices:
            batch_indices=range(n,n+SEQUENCE_LENGTH*BATCH_SIZE,SEQUENCE_LENGTH)
            next_batch = np.array([string2oneHot(data[n:n + SEQUENCE_LENGTH],vocab) for n in batch_indices])
            loss= probe_loss(next_batch[:, :-1],next_batch[:, BURNIN+1:].reshape(-1, next_batch.shape[-1]))
            test_loss += loss/len(test_indices)
        track_test_error.append(test_loss)
        
        # Jacobian
        jacobian=probe_J(next_batch[:, :-1], next_batch[:, BURNIN+1:].reshape(-1, next_batch.shape[-1]))
        hidden_units=probe_hidden(next_batch[:, :-1])
        print '\n\n$$$$ Validation loss =', valid_loss, '$$$$'                
        print '$$$$ Test loss =', test_loss,  '$$$$\n\n'           
        
        
        #store data
        if DO_SAVE:
            connectivity = []
            for p in all_params:
                connectivity.append(p.get_value())                    
            
            f = file(save_file_name, 'wb')
            results={'track_train_error': track_train_error,'track_valid_error':track_valid_error,'track_test_error':track_test_error,
            'track_orthogonality':track_orthogonality,'track_trace_WW':track_trace_WW,'track_angle':track_angle,
            'track_grad_norm':track_grad_norm,'track_S_min':track_S_min,'track_S_max':track_S_max,
            'params':params,'connectivity': connectivity,'jacobian':jacobian,'hidden_units':hidden_units,'total_time':total_time}
            cPickle.dump(results, f, protocol=cPickle.HIGHEST_PROTOCOL)
        
            f.close()

print >> sys.stderr, ('The code for file ' +
                  os.path.split(__file__)[1] +
                  ' ran for %.2f minutes' % (total_time))


        
                    
