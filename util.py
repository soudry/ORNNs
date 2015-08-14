"""
Created on Mon Aug 12  2015

@authors: Daniel Soudry
"""

"""Auxiliary function for Main code
"""

import numpy as np
import theano.tensor as T
import os
#from pylab import load
#import theano,lasagne

def string2oneHot(data,vocabulary):
    """Load in a string from a text file, and outputs it as a one-hot matrix.

    Parameters
    ----------
    data  : str
        data
    vocabulary : list
        List of characters in the dataset file

    Returns
    -------
    data_matrix : np.ndarray
        One-hot encoding of the dataset.

    """

    Lc=len(vocabulary)
    Mat=np.identity(Lc)
    Mat=np.append(Mat,np.zeros([Lc,1]),axis=1)
    vocabulary_dict = dict((c,Mat[i,:]) for (i,c) in enumerate(vocabulary))
    other_char=np.zeros([Lc+1],dtype=np.bool)
    other_char[-1]=1
    
    data_matrix=[vocabulary_dict.get(x,other_char) for x in data]

    return data_matrix
    
class Data_ranges:
    def __init__(self,train_size,valid_size,test_size):
        self.train_end=train_size
        self.test_end=test_size+train_size+valid_size
        self.valid_end=train_size+valid_size
        self.train_start = 0
        self.valid_start =train_size
        self.test_start = train_size+valid_size
       
def oneHot2string(data_matrix,vocabulary):
    """Load in a string from a text file, and outputs it as string.

    Inputs
    ----------
    data_matrix : np.ndarray
        One-hot encoding of the dataset.

    vocabulary : list
        List of characters in the dataset file

    Returns
    -------
    data  : string
        data

    """
    vocabulary_new=vocabulary +['~'] # a letter for encoding outside vocabulary
    L=np.shape(data_matrix)[0]
    data=""    
    for kk in range(L):
        data+=vocabulary_new[np.argmax(data_matrix[kk,:])]
            
    return data

def load_dataset(dataset,part=None):
   
    """ inputs:
     dataset: string
        dataset name
    part (optional): integer
        part of the dataset to output
     
     outputs:
     datastring: very long string 
         the dataset, as a list characters
     vocabulary: list of str
        Vocabulary of the dataset.
     train_range, valid_range, test_range: 3 lists of 2 integers
         the starting point and end of each range in the dataset
     
    Available datasets:
    # Get shakespeare_input.txt from here:
    # http://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt
    # Get Wiki_letters_2G.txt from here: ....
    
    """       
    dataset_file='Data/%s.txt' % (dataset)

    print 'loading data...'
    # Read in entire text file
    with open(dataset_file) as f:
        data = f.read()
    print 'done'

    if dataset in ['Shakespeare']:
        vocabulary = list(set(data)) 
        
        if part!=None:
            L=len(data)
            data=data[:part]
    
        M=int(np.round(0.1*len(data)))
        train_size=len(data)-2*M #len(data)-2*M
        test_size=M #from  the end?
        valid_size=M # after test range 

    elif dataset in ['Wiki_2G']:
        vocabulary_str="""abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-+.,:;?/\\!@#$%&*()"\'\n """
        #    all_chars
        #    """abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-
        #    +.,:;?/\\!@#$%&*()"\'\n\xe2\x80\x94\xc2\xa7\x93\x99\xe5\x9c\x98\xe9\x95\xb7
        #    \x89\xaf\xe6\xe7\xb4\xa1\xb0\x8f\x9a\x8a\xc3\xa5\xc5\xb6\xb8=\xa9\xa8\xa2{\xc4
        #    \x81\xa0<>\x88\x92|\xbc\xb3\xad\x8b\x9b\x96\xe8\x83\xbd\x87\xa4\x8d\x8e\xe4\x85
        #    \x82\xba\xbb\x9d\xef\xe3\x91\xe1\x8c\xb9\xb2\xa6\xa3~}\xb1\xbf\xce\xe0\x84\x9f\x90
        #    \xae\x86\xaa\xbe\xac\xcf\xab`^\x97\xd9\xd8\xd7\x9e\xc7\xc9\t\xd0\xd1\xb5\xcc\xca\xea
        #    \xec\xeb\xcb\xc6\xda\xdb""" #all chars 
        
        vocabulary=[vocabulary_str[n] for n in range(len(vocabulary_str))]
        M=10000000 # original size of validation and test set in paper "Training and Analysing Deep Recurrent Neural Networks"
        if part!=None:
            L=len(data)
            data=data[:part]
            M=np.round(M*part/L)
            
        train_size=len(data)-2*M #len(data)-2*M
        test_size=M #from  the end?
        valid_size=M # after test range 
        
    else:
       print 'unknown dataset!'
       
    return data, vocabulary, Data_ranges(train_size,valid_size,test_size)
   

    


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
    
    if GAIN>1:
        GAIN_str='_G%f' %(GAIN)
    else:
        GAIN_str=''
    
    save_file_name='%s/TASK_D%i_W%i%s_SL%i%s.save' % (directory,DEPTH,HIDDEN_SIZE,GAIN_str,SEQUENCE_LENGTH,str1)
    
    return save_file_name