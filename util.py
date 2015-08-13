"""
Created on Mon Aug 12  2015

@authors: Daniel Soudry
"""

"""Auxiliary function for Main code
"""

import numpy as np
import theano.tensor as T
import os
from pylab import load
#import theano,lasagne


def load_dataset_small(dataset_file, vocabulary=None):
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
        if char in vocabulary:
            data_matrix[n][vocabulary.index(char)] = 1
    return data_matrix, vocabulary
    
def load_dataset_large(dataset_file,vocabulary_str="""abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-+.,:;?/\\!@#$%&*()"\'\n """):
  #    all_chars
#    """abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-
#    +.,:;?/\\!@#$%&*()"\'\n\xe2\x80\x94\xc2\xa7\x93\x99\xe5\x9c\x98\xe9\x95\xb7
#    \x89\xaf\xe6\xe7\xb4\xa1\xb0\x8f\x9a\x8a\xc3\xa5\xc5\xb6\xb8=\xa9\xa8\xa2{\xc4
#    \x81\xa0<>\x88\x92|\xbc\xb3\xad\x8b\x9b\x96\xe8\x83\xbd\x87\xa4\x8d\x8e\xe4\x85
#    \x82\xba\xbb\x9d\xef\xe3\x91\xe1\x8c\xb9\xb2\xa6\xa3~}\xb1\xbf\xce\xe0\x84\x9f\x90
#    \xae\x86\xaa\xbe\xac\xcf\xab`^\x97\xd9\xd8\xd7\x9e\xc7\xc9\t\xd0\xd1\xb5\xcc\xca\xea
#    \xec\xeb\xcb\xc6\xda\xdb""" #all chars
    print 'loading a chunk of wikipedia...'
    (data, _) = load(dataset_file)
    data=data[1:100000000]
    Lc=len(vocabulary_str)
    Mat=np.identity(Lc)
    Mat=np.append(Mat,np.zeros([Lc,1]),axis=1)
    vocabulary_dict = dict((c,Mat[i,:]) for (i,c) in enumerate(vocabulary_str))
    other_char=np.zeros([Lc+1])
    other_char[-1]=1
    
    result=[vocabulary_dict.get(x,other_char) for x in data]
    vocabulary=[vocabulary_str[n] for n in range(len(vocabulary_str))]
    print 'loading done'
    return result,vocabulary
    
def load_dataset(dataset):
    """Load in a dataset from a text file, and return the dataset as a one-hot
    matrix.
    
    # Get shakespeare_input.txt from here:
    # http://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt
    # Get Wiki_letters_2G.txt from here: ....
    
    Parameters
    ----------
    dataset: dataset name
    
    Returns
    -------
    data_matrix : np.ndarray
        One-hot encoding of the dataset.
    vocabulary : list of str
        Vocabulary of the dataset
    """        
    dataset_file='Data/%s.txt' % (dataset)
    if dataset in ['Shakespeare']:
       data_matrix, vocabulary=load_dataset_small(dataset_file)  
       return data_matrix, vocabulary  
    elif dataset in ['Wiki_2G']:
       load_dataset_large(dataset_file)
    else:
       print 'unknown dataset!'
       


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
    # Get name of save file, based on parameters

    DEPTH=params['network']['DEPTH']
    HIDDEN_SIZE=params['network']['HIDDEN_SIZE']
    
    SEQUENCE_LENGTH=params['training']['SEQUENCE_LENGTH']
    DATASET=params['training']['DATASET']
    RETRACT=params['algorithm']['RETRACT']
    PROJ_GRAD=params['algorithm']['PROJ_GRAD']
    THRESHOLD=params['algorithm']['THRESHOLD']
    GAIN=params['algorithm']['GAIN']
    
    str1_list=[]
    if PROJ_GRAD:
        str1_list.append('_P')
    
    if RETRACT:
        if THRESHOLD>0:
            str1_list.append('_R%f' %(THRESHOLD))
        else:
            str1_list.append('_R0')
        
    str1 = ''.join(str1_list)
    directory='Results/%s' % (DATASET)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
#    if GAIN>1:
    GAIN_str='_G%f' %(GAIN)
#    else:
#        GAIN_str=''
    
    save_file_name='%s/TASK_D%i_W%i%s_SL%i%s.save' % (directory,DEPTH,HIDDEN_SIZE,GAIN_str,SEQUENCE_LENGTH,str1)
    
    return save_file_name