################################################
# Import statements
################################################

import cPickle
#import gzip
import os
import sys
import time

import numpy as np
#import h5py
#from pylab import *
#import matplotlib.pyplot as plt
from math import ceil, floor
import theano
import theano.tensor as T
from layer_classes import RNN, SoftmaxClassifier, SVMclassifier
from misc import GradClip, clip_gradient, Adam
from pylab import load
from random import sample

def String2Features(string,chars):
  
    Lc=len(chars)
    Mat=np.identity(Lc)
    Mat=np.append(Mat,np.zeros([Lc,1]),axis=1)
    chars_dict = dict((c,Mat[i,:]) for (i,c) in enumerate(chars))
    other_char=np.zeros([Lc+1])
    other_char[-1]=1
    
    result=[chars_dict.get(x,other_char) for x in string]
    return result
#    result=np.empty( shape=(0,Lc+1) )

#    for x in list(string):
#        if x in chars:
#            temp=np.reshape(chars_dict[x],[1,Lc+1])            
#            result=np.append(result,temp,axis=0)
#        else:
#            result=np.append(result,other_char,axis=0)
            
def Features2String(features,chars):
    chars+='~'
    L=np.shape(features)[0]
    result=""    
    for kk in range(L):
        result+=chars[np.argmax(features[kk,:])]
            
    return result
    

#%%    
################################################
# Parameters
################################################

#    all_chars
#    """abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-
#    +.,:;?/\\!@#$%&*()"\'\n\xe2\x80\x94\xc2\xa7\x93\x99\xe5\x9c\x98\xe9\x95\xb7
#    \x89\xaf\xe6\xe7\xb4\xa1\xb0\x8f\x9a\x8a\xc3\xa5\xc5\xb6\xb8=\xa9\xa8\xa2{\xc4
#    \x81\xa0<>\x88\x92|\xbc\xb3\xad\x8b\x9b\x96\xe8\x83\xbd\x87\xa4\x8d\x8e\xe4\x85
#    \x82\xba\xbb\x9d\xef\xe3\x91\xe1\x8c\xb9\xb2\xa6\xa3~}\xb1\xbf\xce\xe0\x84\x9f\x90
#    \xae\x86\xaa\xbe\xac\xcf\xab`^\x97\xd9\xd8\xd7\x9e\xc7\xc9\t\xd0\xd1\xb5\xcc\xca\xea
#    \xec\xeb\xcb\xc6\xda\xdb""" #all chars

#used chars    
chars = """abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-+.,:;?/\\!@#$%&*()"\'\n """    

n_epochs=10000
batch_size = 250
burnin = 50 # RNN throws away prediction before that time
n_train_batches = 28
do_save = 1

# number of hidden units
n_H = 727
# number of input units
n_in=len(chars)+1 #number of characters 
# number of output units
n_out = n_in

fname = 'RNN_' + str(n_in)
savefilename = 'C:\Users\Daniel\Copy\Columbia\Research\RNNs\Josh files' + fname + '.save'

print 'loading a chunk of wikipedia...'
(data, _) = load('C:\Users\Daniel\Copy\Columbia\Research\RNNs\mrnns\wiki_letters_2G')
_data = data


train_size=1000000
test_size=1000 #from  the end?
valid_size=1000 # after test range    

memory_threshold=1e8  # maximal size of array to allow in memory

data_test=data[(train_size+valid_size):(train_size+valid_size+test_size)]
data_valid=data[train_size:(train_size+valid_size)]
data_train=data[:train_size]
data=None # clear memory
L=train_size

splits=ceil(n_in*L/memory_threshold)
T_split=int(floor(n_in*L/splits))

# compute number of minibatches for training, validation and testing
#n_train_batches = train_set_x.get_value(borrow=True).shape[0]
#n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
#n_test_batches = test_set_x.get_value(borrow=True).shape[0]
#n_train_batches /= batch_size
#n_valid_batches /= batch_size
#n_test_batches /= batch_size

# allocate symbolic variables for the data
data_train_part=String2Features(data_train[:T_split],chars)
data_train_sym = theano.shared(np.asarray(data_train_part,dtype=theano.config.floatX),borrow=True)
data_valid_all=String2Features(data_valid,chars)
data_valid_sym = theano.shared(np.asarray(data_valid_all,dtype=theano.config.floatX),borrow=True)
data_test_all=String2Features(data_test,chars)
data_test_sym = theano.shared(np.asarray(data_test_all,dtype=theano.config.floatX),borrow=True)
x = T.matrix('x')   # the data input
y = T.matrix('y')   # the labels
index= T.lscalar('index')
batch_size_sym=theano.shared(np.asarray(batch_size,dtype='int64'),borrow=True)
burnin_sym=theano.shared(np.asarray(burnin,dtype='int64'),borrow=True)

print 'done.'

#%%
######################
# BUILD ACTUAL MODEL #
######################
print '... building the model'

# allocate symbolic variables for the data

is_train = T.iscalar('is_train') # pseudo boolean for switching between training and prediction
rng = np.random.RandomState(1234)

################################################
# Architecture: input --> hidden layers --> reconstruct + sparse activations
################################################


# input is T x features (so batch_size by n_in)
RNN_1 = RNN(rng, x ,n_in=n_in,n_out=n_H,burnin=burnin)

classifier = SoftmaxClassifier(RNN_1.output, n_H, n_out)
#classifier = SVMclassifier(RNN_1.output, n_in, n_out)

#classification_error = T.mean(T.nnet.categorical_crossentropy(y, SoftmaxClassifier.output ))
#Perplexity=T.mean(-T.log2(RNN_1.output[y, T.arange(y.shape[0])]))
#classification_error = 1-T.mean(T.eq(T.argmax(RNN_1.output, 1),y),axis=0)

cost = T.mean(classifier.objective(y))

# create a function to compute the mistakes that are made by the model
probe_model = theano.function([index], [cost, classifier.output, y],
      givens={ x: data_train_sym[index : index + batch_size_sym,:],
               y: data_train_sym[index + burnin_sym  +1 : index + batch_size_sym + 1,:]})
               
# create a function to compute the mistakes that are made by the model on the validation set
valid_model = theano.function([index], [cost, classifier.output, y],
      givens={ x: data_valid_sym[index : index + batch_size_sym,:],
               y: data_valid_sym[index + burnin_sym +1 :index + batch_size_sym + 1,:]})


# create a function to compute the mistakes that are made by the model on the validation set
test_model = theano.function([index], [cost, classifier.output, y],
      givens={ x: data_test_sym[index : index + batch_size_sym,:],
               y: data_test_sym[index + burnin_sym +1 :index + batch_size_sym + 1,:]})

# create a list of all model parameters to be fit by gradient descent
params = RNN_1.params+classifier.params

# updates from ADAM
updates = Adam(cost, params)

# create a function to train the model
train_model = theano.function([index],[cost, classifier.output, y], updates=updates,
      givens={ x: data_train_sym[index : index + batch_size_sym,:],
               y: data_train_sym[index + burnin_sym +1 :index + batch_size_sym + 1,:]})

#%%
###############
# TRAIN MODEL #
###############
print '... training'

# early-stopping parameters
patience = 5e3  # look as this many examples regardless
add_patience = 5000
#patience = train_set_x.get_value(borrow=True).shape[0] * n_epochs #no early stopping
patience_increase = 2  # wait this much longer when a new best is
                       # found
improvement_threshold = 0.995  # a relative improvement of this much is
                               # considered significant


#best_params = None
best_validation_loss = np.inf
best_iter = 0
#test_score = 0.
start_time = time.clock()

epoch = 0
done_looping = False

track_train = list()
track_valid = list()
track_test = list()

while (epoch < n_epochs) and (not done_looping):
    
    for ss in range(int(splits)):
        if ss<splits-1:
            T_split=int(floor(L/splits))
        else:
            T_split=int(L-(splits-1)*floor(L/splits))            
        
        split_loc=int(ss*floor(L/splits))
        data_train_part=String2Features(data_train[split_loc:(split_loc+T_split)],chars)
        
        epoch = epoch + 1
        n_train_batches=T_split/batch_size-1
        rand_indices=sample(range(0,T_split-batch_size,batch_size), n_train_batches)
        
        for kk in range(n_train_batches):
            # Train on minibatch
            minibatch_avg_cost, current_outputs, current_labels = train_model(rand_indices[kk])
            output_string=Features2String(current_outputs,chars)
            labels_string=Features2String(current_labels,chars)
            print ('\n----\nminibatch: %i, training error: %f, \n\nlast predicted output: %s, \n\nlast label: %s' %
                     (kk, minibatch_avg_cost,output_string,labels_string))
            # Track trainning error
            track_train.append(minibatch_avg_cost)
    
            # iterationation number
            iteration = (epoch - 1) * n_train_batches + kk
            
            validation_frequency = min(n_train_batches, patience / 2)
                              # go through this many
                              # minibatche before checking the network
                              # on the validation set; in this case we
                              # check every epoch
            
            if (iteration + 1) % validation_frequency == 0:
                # compute absolute error loss on validation set
                validation_loss=0 
                NV=valid_size
                for kk in range(valid_size-batch_size-1):
                    current_loss, junk1, junk2=valid_model(kk) 
                    validation_loss = ((NV-1)/NV)*validation_loss+(1/NV)*current_loss               
                print('--epoch %i, minibatch %i, validation error %f' %
                     (epoch, kk + 1,
                      validation_loss))
                track_valid.append(validation_loss)
    
                # if we got the best validation score until now
                if validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if validation_loss < best_validation_loss *  \
                           improvement_threshold:
                        patience = max(patience, iteration + add_patience)
                        #patience = max(patience, iteration * patience_increase)
    
                    best_validation_loss = validation_loss
                    best_iter = iteration
    
                    # test it on the test set
                    NT=test_size
                    test_loss=0
                    for kk in range(test_size-batch_size-1):
                        current_test_loss, test_pred, junk3 = test_model(kk)  
                        test_score=((NT-1)/NT)*test_loss+(1/NT)*test_loss    
    
                    print(('---- epoch %i, minibatch %i, test error of '
                           'best model %f') %
                          (epoch, kk + 1,
                           test_score))
                    track_test.append(np.mean(test_score))
        
            if do_save:
                if (iteration+1)%1000==0:
                    #store data
                    params_sto = []
                    p_indx = 0
                    while p_indx<len(params):
                        params_sto.append( params[p_indx].get_value() )
                        p_indx = p_indx + 1
                        
                        
                    f = file(savefilename, 'wb')
                    for obj in [[params_sto] + [track_train] + [track_valid] + [track_test]]:
                        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    
                    f.close()
    
            if patience <= iteration:
                done_looping = True
                break

end_time = time.clock()
print(('Optimization complete. Best validation score of %f'
       'obtained at iteration %i, with test performance %f') %
      (best_validation_loss, best_iter + 1, np.sum(test_score)))
print >> sys.stderr, ('The code for file ' +
                      os.path.split(__file__)[1] +
                      ' ran for %.2fm' % ((end_time - start_time) / 60.))

#[test_losses, recon, raw] = probe_model(1)
#plt.imshow(recon[-1,].reshape((28, 28)))
#plt.show()

#store data
if do_save:
    params_sto = []
    p_indx = 0
    while p_indx<len(params):
        params_sto.append( params[p_indx].get_value() )
        p_indx = p_indx + 1
    

    f = file(savefilename, 'wb')
    for obj in [[params_sto] + [track_train] + [track_valid] + [track_test]]:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)

    f.close()
