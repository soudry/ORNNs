# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 21:20:37 2015

@authors: Colin Raffel & Daniel Soudry
"""

"""ORNN for next character prediction
"""

import lasagne
import numpy as np
import theano
import theano.tensor as T

BATCH_SIZE = 50 
SEQUENCE_LENGTH = 100 # Length of input sequence into RNN
HIDDEN_SIZE = 100 # RNN Hidden layer size 
ORTHOGONALIZE=True # Should we project gradient on tangent space to to the Stiefel Manifold (Orthogonal matrices)?
DO_RETRACT=True # Should we do retraction step?
THRESHOLD=0.1 #error threshold in which we do the retraction step


def load_dataset(dataset_file, vocabulary=None):
    """Load in a dataset from a text file, and return the dataset as a one-hot
    matrix.

    Parameters
    ----------
    dataset_file : str
        Path to dataset file, just a text file
    vocabulary : list or None
        List of characters in the dataset file; if None, the unique
        characters in the file will be used.

    Returns
    -------
    data_matrix : np.ndarray
        One-hot encoding of the dataset.
    vocabulary : list of str
        Vocabulary of the dataset.
    """
    # Read in entire text file
    with open(dataset_file) as f:
        data = f.read()
    # Constructs vocabulary as all unique chars in text file
    if vocabulary is None:
        vocabulary = list(set(data))
    data_matrix = np.zeros(
        (len(data), len(vocabulary)), dtype=np.bool)
    # Construct one-hot encoding
    for n, char in enumerate(data):
        data_matrix[n][vocabulary.index(char)] = 1
    return data_matrix, vocabulary


def tangent_grad(X, grad):
    """Compute and project the gradient of X onto tangent space to the Stiefel Manifold
    (Orthogonal matrices)

    Parameters
    ----------
    X : theano.tensor.var.TensorVariable
        Theano variable whose gradient will be projected
    grad : theano.tensor.var.TensorVariable
        Gradient to project

    Returns
    -------
    proj_grad : theano.tensor.var.TensorVariable
        Projected gradient
    """
    XG = T.dot(T.transpose(X), grad)
    tang_grad = grad - 0.5*T.dot(X, XG + T.transpose(XG))
    return tang_grad
    
def retraction(Q):
    """ Project Matrix Q to the to the Stiefel Manifold (Orthogonal matrices)"""
    
    U, S, V = T.nlinalg.svd(Q)
       
    return T.dot(U,V)


if __name__ == '__main__':
    # Get shakespeare_input.txt from here:
    # http://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt
    train_data, vocab = load_dataset('../data/shakespeare_input.txt')
    
    # Construct network.  The last dimension is the vocab size.
    l_in = lasagne.layers.InputLayer(
        (BATCH_SIZE, SEQUENCE_LENGTH, train_data.shape[-1]))
    # Single recurrent layer
    l_rec = lasagne.layers.RecurrentLayer(
        l_in, HIDDEN_SIZE,
        # Use orthogonal weight initialization
        W_in_to_hid=lasagne.init.Orthogonal(),
        W_hid_to_hid=lasagne.init.Orthogonal(),
        nonlinearity=lambda h: T.tanh(h),learn_init=True)
        
        #  Or Use normalized tanh nonlinearity (I think peformance is less good)
        #   nonlinearity=lambda h: 1.7159*T.tanh(2*h/3),learn_init=True)
           
    # Squash the batch and sequence (non-feature) dimensions
    l_reshape = lasagne.layers.ReshapeLayer(l_rec, [-1, HIDDEN_SIZE])
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
    updates = lasagne.updates.adam(loss, all_params)
    if ORTHOGONALIZE==True:
        for param in all_params:
            if param is l_rec.W_hid_to_hid:
                updates[param] = param + tangent_grad(param, updates[param]-param)
   
    # Compile functions for training and computing output
    train = theano.function([l_in.input_var, target], loss, updates=updates)
    retract_w = theano.function([], [], updates=[(l_rec.W_hid_to_hid,retraction(l_rec.W_hid_to_hid))])
    get_output = theano.function([l_in.input_var], network_output)

    for batch in range(10000):
        # Sample BATCH_SIZE sequences of length SEQUENCE_LENGTH from train_data
        next_batch = np.array([
            train_data[n:n + SEQUENCE_LENGTH]
            for n in np.random.choice(
                train_data.shape[0] - SEQUENCE_LENGTH, BATCH_SIZE)])
        # Train with this batch
        loss = train(next_batch[:, :-1],
                    next_batch[:, 1:].reshape(-1, next_batch.shape[-1]))
        # Print diagnostics every 100 batches
        if not batch % 100:
            print "#### Iteration: {} Loss: {}".format(batch, loss)
            W_hid_to_hid = l_rec.W_hid_to_hid.get_value()
            error = np.sum((np.eye(W_hid_to_hid.shape[0]) -
                            np.dot(W_hid_to_hid.T, W_hid_to_hid))**2)
            if DO_RETRACT and error>THRESHOLD:
                retract_w()
                U,S,V=np.linalg.svd(W_hid_to_hid)
                print 'W singular values: max=%f, min=%f' % (np.min(S),np.max(S))
                            
            print "#### Hid->hid nonorthogonality: {}".format(error)
            print "#### Target:"
            print ''.join([vocab[n] for n in np.argmax(next_batch[0], axis=1)])
            print "#### Predicted:"
            print ''.join([vocab[n] for n in
                        np.argmax(get_output(next_batch[:1]), axis=1)])
                        
