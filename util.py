"""
Created on Mon Aug 12  2015

@authors: Daniel Soudry
"""

"""Auxiliary function for Main code
"""

import numpy as np
import theano.tensor as T
import os
#import theano,lasagne


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

def get_file_name(params):
    DEPTH=params['network']['DEPTH']
    HIDDEN_SIZE=params['network']['HIDDEN_SIZE']
    SEQUENCE_LENGTH=params['training']['SEQUENCE_LENGTH']
    DATASET=params['training']['DATASET']
    RETRACT=params['algorithm']['RETRACT']
    PROJ_GRAD=params['algorithm']['PROJ_GRAD']
    THRESHOLD=params['algorithm']['THRESHOLD']
    
    str1_list=[]
    if PROJ_GRAD:
        str1_list.append('_P')
    
    if RETRACT:        
        str1_list.append('_R%f' %(THRESHOLD))
        
    str1 = ''.join(str1_list)
    directory='Results/%s' % (DATASET)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    save_file_name='%s/TASK_D%i_W%i_SL%i%s.save' % (directory,DEPTH,HIDDEN_SIZE,SEQUENCE_LENGTH,str1)
    
    return save_file_name