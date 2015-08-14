# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 21:20:37 2015

@authors: Colin Raffel & Daniel Soudry
"""

"""ORNN for next character predictionc
"""
import time,cPickle,sys,os
import numpy as np

import theano,lasagne
import theano.tensor as T

from util import load_dataset,tangent_grad,retraction,get_file_name,oneHot2string,string2oneHot
#%% Set Parameters
# Training Parameters
BATCH_NUM=500000
BATCH_SIZE = 50
STAT_SKIP= 100 # How many Batches to wait before we update learning statistics, and retraction
VALID_SKIP = 10000 # How many Batches to wait before we update validation and test error
SEQUENCE_LENGTH = 250 # Length of input sequence into RNN
opt_mathods_set=['SGD','ADAM']
OPT_METHOD=opt_mathods_set[1]
dataset_set=['Shakespeare','Wiki_2G']
DATASET=dataset_set[1]
training={'BATCH_NUM':BATCH_NUM,'BATCH_SIZE':BATCH_SIZE,'SEQUENCE_LENGTH':SEQUENCE_LENGTH,'OPT_METHOD':OPT_METHOD,
'DATASET':DATASET,'STAT_SKIP':STAT_SKIP,'VALID_SKIP':VALID_SKIP}

# Algorithm Parameters
PROJ_GRAD=True # Should we project gradient on tangent space to to the Stiefel Manifold (Orthogonal matrices)?
RETRACT=True # Should we do retraction step?
THRESHOLD=0 #error threshold in which we do the retraction step
GAIN=1 # a multiplicative constant we add to all orthogonal matrices
algorithm={'PROJ_GRAD':PROJ_GRAD,'RETRACT':RETRACT,'THRESHOLD':THRESHOLD,'GAIN':GAIN}

# Network Architecture
HIDDEN_SIZE = 706 # RNN Hidden layer size 
DEPTH=5 # number of RNNs in the middle
ALL2OUTPUT=False # Are all hidden layers also directly connected to output? 
network={'DEPTH':DEPTH,'HIDDEN_SIZE':HIDDEN_SIZE,'ALL2OUTPUT':ALL2OUTPUT}

params={'training':training,'network':network,'algorithm':algorithm}
DO_SAVE=True # should we save results?
#%% Intialize network model


if __name__ == '__main__':
    
    data, vocab, data_ranges = load_dataset(DATASET)
    
    # define a list of parameters to orthogonalize (recurrent connectivities)
    param2orthogonlize=[]      
    # The number of features is number of different letters + 1 unknown letter
    FEATURES_NUM=len(vocab)+1
    # Construct network.  
    l_in = lasagne.layers.InputLayer(
        (BATCH_SIZE, SEQUENCE_LENGTH, FEATURES_NUM))
    layers_to_concat = []
    # other recurrent layer
    for dd in range(DEPTH): 
        if dd == 0:
            input_layer = l_in
        else:
            input_layer = l_rec
        l_rec = lasagne.layers.RecurrentLayer(
            input_layer, HIDDEN_SIZE,
            # Use orthogonal weight initialization
            W_in_to_hid=(lasagne.init.Orthogonal(gain=GAIN)),
            W_hid_to_hid=lasagne.init.Orthogonal(gain=GAIN),
            nonlinearity=lambda h: T.tanh(h),learn_init=True, name='RNN_%i' % (dd+1))
        param2orthogonlize.append(l_rec.W_hid_to_hid)
        layers_to_concat.append(l_rec)
        
        #  if we use normalized tanh nonlinearity (I think peformance is slightly worse)
        #  W_hid_to_hid nonlinearity=lambda h: 1.7159*T.tanh(2*h/3),learn_init=True)
           
    if ALL2OUTPUT: #if we the output to connect to all hidden layers
        rec_outputs=lasagne.layers.ConcatLayer(layers_to_concat,axis=-1)           
        output_size=DEPTH*HIDDEN_SIZE
    else: #if we only want the deepest RNN layer as output
        rec_outputs=l_rec
        output_size=HIDDEN_SIZE
    
    # Squash the batch and sequence (non-feature) dimensions     
    l_reshape = lasagne.layers.ReshapeLayer(rec_outputs, [-1, output_size])

    # Compute softmax output
    l_out = lasagne.layers.DenseLayer(
        l_reshape, FEATURES_NUM,
        nonlinearity=lasagne.nonlinearities.softmax)

    # Get Theano expression for network output
    network_output = lasagne.layers.get_output(l_out)
    # Symbolic vector for target
    target = T.matrix('target')
    # Compute categorical cross-entropy between prediction and target
    loss = T.mean(lasagne.objectives.categorical_crossentropy(
        network_output, target))#/np.log(2)
    # Collect all network parameters
    all_params = lasagne.layers.get_all_params(l_out)
   
    if OPT_METHOD=='ADAM':
        updates = lasagne.updates.adam(loss, all_params)
        if PROJ_GRAD==True:
            for param in all_params:
                if param in param2orthogonlize:
                    updates[param] = param + tangent_grad(param, updates[param]-param)
    elif OPT_METHOD=='SGD':
        learning_rate0=0.5
        learning_rate=learning_rate0
        lr=theano.shared(np.asarray(learning_rate,dtype=theano.config.floatX),borrow=True)
        updates = []
        grads = T.grad(loss, all_params)
        for p,g in zip(all_params,grads):    
            delta=lr*g/T.sqrt(T.sum(g**2)+1) # normalize gradient size... a bit hacky
            if (p in param2orthogonlize) and PROJ_GRAD==True:
                updates.append((p,p - tangent_grad(p, delta)))
            else:
                updates.append((p,p - delta))
                
    else:
        print 'unknown optimization method'
    
    retract_updates=[]
    for p in param2orthogonlize:
        retract_updates.append((p,GAIN*retraction(p)))
    
    # Compile functions for training and computing output
    train = theano.function([l_in.input_var, target], loss, updates=updates,allow_input_downcast=True)
    probe = theano.function([l_in.input_var, target], loss,allow_input_downcast=True)
    retract_w = theano.function([], [], updates=retract_updates,allow_input_downcast=True)
    get_output = theano.function([l_in.input_var], network_output,allow_input_downcast=True)
    
    track_train_error=[]
    track_valid_error=[] 
    track_test_error=[] 
    track_orthogonality=[]
    track_trace_WW=[]
 

#%%  Training    
    start_time = time.clock()
    
    for batch in range(BATCH_NUM):
        # Sample BATCH_SIZE sequences of length SEQUENCE_LENGTH from train_data
        rand_indices=np.random.choice(data_ranges.train_end - SEQUENCE_LENGTH, BATCH_SIZE)
        next_batch = np.array([string2oneHot(data[n:n + SEQUENCE_LENGTH],vocab) for n in rand_indices])
        if OPT_METHOD=='SGD':
            learning_rate=learning_rate0*(1-batch/BATCH_NUM)
        # Train with this batch
        loss = train(next_batch[:, :-1],
                    next_batch[:, 1:].reshape(-1, next_batch.shape[-1]))
        BATCH_NUM
               
        # Print diagnostics every STAT_SKIP batches
                    
        if not batch % STAT_SKIP:
            print "#### Iteration: {} Loss: {}".format(batch, loss)
            
            o_error=0
            trace_error=0
            for p in param2orthogonlize:
                W = p.get_value()
                o_error += np.sum((np.eye(W.shape[0]) - np.dot(W.T, W))**2)
                trace_error+=np.trace(np.dot(W.T, W))/(min(np.shape(W))*len(param2orthogonlize))
                
            if RETRACT and o_error>THRESHOLD:
                retract_w()
                index=0
                for p in param2orthogonlize:
                    index+=1
                    W = p.get_value()
                    U,S,V=np.linalg.svd(W)
                    print 'W%i singular values: max=%f, min=%f' % (index,np.min(S),np.max(S))
                    
            print "#### Hid->hid nonorthogonality: {}".format(o_error)
            print "#### Target:"
            print oneHot2string(next_batch[0],vocab)
            print "#### Predicted:"
            print oneHot2string(get_output(next_batch[:1]),vocab)
            # track results
            track_train_error.append(loss)
            track_orthogonality.append(o_error)
            track_trace_WW.append(trace_error)
    
        if ((not (batch+1) % VALID_SKIP) or (batch==(BATCH_NUM-1))):
            # Validation error            
            valid_loss=0
            valid_indices=range(data_ranges.valid_start,data_ranges.valid_end-SEQUENCE_LENGTH*BATCH_SIZE,SEQUENCE_LENGTH*BATCH_SIZE)
            if len(valid_indices)==0: print 'validation set too small! increase data size, or reduce SEQUENCE_LENGTH*BATCH_SIZE'            
            for n in valid_indices:
                batch_indices=range(n,n+SEQUENCE_LENGTH*BATCH_SIZE,SEQUENCE_LENGTH)
                next_batch = np.array([string2oneHot(data[n:n + SEQUENCE_LENGTH],vocab) for n in batch_indices])
                loss= probe(next_batch[:, :-1],next_batch[:, 1:].reshape(-1, next_batch.shape[-1]))
                valid_loss += loss/len(valid_indices)
            track_valid_error.append(valid_loss)
            
            # Test error
            test_loss=0
            test_indices=range(data_ranges.test_start,data_ranges.test_end-SEQUENCE_LENGTH*BATCH_SIZE,SEQUENCE_LENGTH*BATCH_SIZE)
            if len(test_indices)==0: print 'test set too small! increase data size, or reduce SEQUENCE_LENGTH*BATCH_SIZE'            
                        
            for n in test_indices:
                batch_indices=range(n,n+SEQUENCE_LENGTH*BATCH_SIZE,SEQUENCE_LENGTH)
                next_batch = np.array([string2oneHot(data[n:n + SEQUENCE_LENGTH],vocab) for n in batch_indices])
                loss= probe(next_batch[:, :-1],next_batch[:, 1:].reshape(-1, next_batch.shape[-1]))
                test_loss += loss/len(test_indices)
            track_test_error.append(test_loss)
            print '\n\n $$$$$$$$ Validation loss =', valid_loss, '$$$$$$$$'                
            print '$$$$$$$$ Test loss =', test_loss, , '$$$$$$$$\n\n'           
            
            #Measure time
            end_time = time.clock()
            total_time=(end_time - start_time) / 60. # in minutes
            
            #store data
            if DO_SAVE:
                params_sto = []
                p_indx = 0
                while p_indx<len(all_params):
                    params_sto.append(all_params[p_indx].get_value() )
                    p_indx = p_indx + 1
                    
                save_file_name=get_file_name(params)
                f = file(save_file_name, 'wb')
                for obj in [[params_sto] + [track_train_error]+[track_valid_error] +[track_test_error]+ [track_orthogonality]+[track_trace_WW]+[total_time]+[params]]:
                    cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
            
                f.close()
    
    print >> sys.stderr, ('The code for file ' +
                      os.path.split(__file__)[1] +
                      ' ran for %.2fm' % (total_time))
    

                
                            
