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

from util import load_dataset,tangent_grad,retraction,get_file_name
#%% Set Parameters
# Training Parameters
BATCH_NUM=10000
BATCH_SIZE = 50
SEQUENCE_LENGTH = 200 # Length of input sequence into RNN
opt_mathods_set=['SGD','ADAM']
OPT_METHOD=opt_mathods_set[1]
dataset_set=['Shakespeare','Wiki_2G']
DATASET=dataset_set[0]
training={'BATCH_NUM':BATCH_NUM,'BATCH_SIZE':BATCH_SIZE,'SEQUENCE_LENGTH':SEQUENCE_LENGTH,'OPT_METHOD':OPT_METHOD,'DATASET':DATASET}

# Algorithm Parameters
PROJ_GRAD=False # Should we project gradient on tangent space to to the Stiefel Manifold (Orthogonal matrices)?
RETRACT=False # Should we do retraction step?
THRESHOLD=0 #error threshold in which we do the retraction step
GAIN=1 # a multiplicative constant we add to all orthogonal matrices
algorithm={'PROJ_GRAD':PROJ_GRAD,'RETRACT':RETRACT,'THRESHOLD':THRESHOLD,'GAIN':GAIN}

# Network Architecture
HIDDEN_SIZE = 200 # RNN Hidden layer size 
DEPTH=5 # number of RNNs in the middle
ALL2OUTPUT=False # Are all hidden layers also directly connected to output? 
network={'DEPTH':DEPTH,'HIDDEN_SIZE':HIDDEN_SIZE,'ALL2OUTPUT':ALL2OUTPUT}

params={'training':training,'network':network,'algorithm':algorithm}
DO_SAVE=True # should we save results?
#%%


if __name__ == '__main__':

    train_data, vocab = load_dataset(DATASET)
    
    # define a list of parameters to orthogonalize (recurrent connectivities)
    param2orthogonlize=[]      
    
    # Construct network.  The last dimension is the vocab size.
    l_in = lasagne.layers.InputLayer(
        (BATCH_SIZE, SEQUENCE_LENGTH, train_data.shape[-1]))
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
            nonlinearity=lambda h: T.tanh(h),learn_init=True, name='RNN_%i' % (dd+2))
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
        l_reshape, train_data.shape[-1],
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
    train = theano.function([l_in.input_var, target], loss, updates=updates)
    retract_w = theano.function([], [], updates=retract_updates)
    get_output = theano.function([l_in.input_var], network_output)
    
    train_error=[] 
    orthoganality=[]
    
    start_time = time.clock()
    
    for batch in range(BATCH_NUM):
        # Sample BATCH_SIZE sequences of length SEQUENCE_LENGTH from train_data
        next_batch = np.array([
            train_data[n:n + SEQUENCE_LENGTH]
            for n in np.random.choice(
                train_data.shape[0] - SEQUENCE_LENGTH, BATCH_SIZE)])
        if OPT_METHOD=='SGD':
            learning_rate=learning_rate0*(1-batch/BATCH_NUM)
        # Train with this batch
        loss = train(next_batch[:, :-1],
                    next_batch[:, 1:].reshape(-1, next_batch.shape[-1]))
        # Print diagnostics every 100 batches
                    
        if not batch % 100:
            print "#### Iteration: {} Loss: {}".format(batch, loss)
            
            o_error=0
            for p in param2orthogonlize:
                W = p.get_value()
                o_error += np.sum((np.eye(W.shape[0]) - np.dot(W.T, W))**2)
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
            print ''.join([vocab[n] for n in np.argmax(next_batch[0], axis=1)])
            print "#### Predicted:"
            print ''.join([vocab[n] for n in
                        np.argmax(get_output(next_batch[:1]), axis=1)])
            # save results
            train_error.append(loss)
            orthoganality.append(o_error)
            
    end_time = time.clock()
    total_time=(end_time - start_time) / 60. # in minutes
    
    print >> sys.stderr, ('The code for file ' +
                      os.path.split(__file__)[1] +
                      ' ran for %.2fm' % (total_time))
    
    #store data
    if DO_SAVE:
        params_sto = []
        p_indx = 0
        while p_indx<len(all_params):
            params_sto.append(all_params[p_indx].get_value() )
            p_indx = p_indx + 1
            
        save_file_name=get_file_name(params)
        f = file(save_file_name, 'wb')
        for obj in [[params_sto] + [train_error] + [orthoganality]+[total_time]+[params]]:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    
        f.close()
                
                            
