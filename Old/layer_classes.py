"""
Collection of layer types

"""

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

class Dropout(object):
    def __init__(self, rng, is_train, input, p=0.5):
        """
        Layer to perform dropout
        """

        srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))

        def drop(input, p=0.5, rng=rng): 
            """
            :type input: numpy.array
            :param input: layer or weight matrix on which dropout resp. dropconnect is applied
    
            :type p: float or double between 0. and 1. 
            :param p: p probability of NOT dropping out a unit or connection, therefore (1.-p) is the drop rate.
    
            """            
            mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
            return input * mask

        
        # multiply output and drop -> in an approximation the scaling effects cancel out 
        train_output = drop(numpy.cast[theano.config.floatX](1./p) * input)
        
        #is_train is a pseudo boolean theano variable for switching between training and prediction 
        self.output = T.switch(T.neq(is_train, 0), train_output, input)
        
class SoftmaxClassifier(object):


    def __init__(self, input, n_in, n_out):

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
                                
#        print self.W.get_value().shape
#        input= theano.printing.Print('i = ', attrs=['shape'])(input)
        self.n_in=n_in
        self.n_out=n_out
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(value=0*numpy.ones((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        # helper variables for adagrad
        self.W_helper = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_helper', borrow=True)
        self.b_helper = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_helper', borrow=True)
            
        # helper variables for L1
        self.W_helper2 = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_helper2', borrow=True)
        self.b_helper2 = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_helper2', borrow=True)

        # compute vector of expected values (for each output) in symbolic form
        self.lin_output = T.dot(input, self.W) + self.b
#        self.lin_output= theano.printing.Print('o = ', attrs=['shape'])(self.lin_output)
        self.lin_output = self.lin_output - T.max(self.lin_output)
#        self.lin_output= theano.printing.Print('o = ', attrs=['shape'])(self.lin_output)
    
#        e_x = T.exp(self.lin_output  - self.lin_output.max(axis=1, keepdims=True))
#        self.output = e_x / e_x.sum(axis=1, keepdims=True)
#        self.output = T.exp(self.lin_output)-T.sum(T.exp(self.lin_output)) 

        self.output = T.nnet.softmax(self.lin_output)  
#        self.output= theano.printing.Print('o = ', attrs=['shape'])(self.output)
        
        # parameters of the model
        self.params = [self.W, self.b]
        self.params_helper = [self.W_helper, self.b_helper]
        self.params_helper2 = [self.W_helper2, self.b_helper2]

    def objective(self, y):
        # Compute average log-probability of the inputs

      return T.nnet.categorical_crossentropy( self.output,y )
#        return -(T.log(self.output[:,T.argmax(y)])-T.log(T.sum(self.output)))
        
        
class SVMclassifier(object):


    def __init__(self, input, n_in, n_out):

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        self.n_in=n_in
        self.n_out=n_out
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(value=0*numpy.ones((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        # helper variables for adagrad
        self.W_helper = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_helper', borrow=True)
        self.b_helper = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_helper', borrow=True)

        # helper variables for L1
        self.W_helper2 = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_helper2', borrow=True)
        self.b_helper2 = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_helper2', borrow=True)

        # compute vector of expected values (for each output) in symbolic form
        self.lin_output = T.dot(input, self.W) + self.b

        self.output = self.lin_output*(self.lin_output>0)
        
        # parameters of the model
        self.params = [self.W, self.b]
        self.params_helper = [self.W_helper, self.b_helper]
        self.params_helper2 = [self.W_helper2, self.b_helper2]

    def objective(self, y):
        return T.mean(   ((self.n_in*y-self.output)**2)/self.n_in )
        

#This variant is the standard multiple input, multiple output version
class LinearRegression(object):
    """Linear Regression Class
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the poisson regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(value=0*numpy.ones((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        # helper variables for adagrad
        self.W_helper = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_helper', borrow=True)
        self.b_helper = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_helper', borrow=True)

        # helper variables for L1
        self.W_helper2 = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_helper2', borrow=True)
        self.b_helper2 = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_helper2', borrow=True)

        # compute vector of expected values (for each output) in symbolic form
        self.E_y_given_x = T.dot(input, self.W) + self.b

        # parameters of the model
        self.params = [self.W, self.b]
        self.params_helper = [self.W_helper, self.b_helper]
        self.params_helper2 = [self.W_helper2, self.b_helper2]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        """
        return T.mean(   (y - self.E_y_given_x)**2   , axis = 0)


#This class is to build a all-to-all hidden layer
class HiddenLayer(object):
    def __init__(self, rng, input_unlag, n_in, n_out, b=None, W=None,numlag=1):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        an activation function (see below). Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer (overwritten in body of class)
        """
        
        # Lag the input
        col=[];
        for i_col in xrange(numlag):
            if i_col == 0:
                currentcol = input_unlag
                input = currentcol
            else:
                currentcol = T.concatenate((T.zeros((i_col,input_unlag.shape[1])),input_unlag[:-i_col,:]))
                input = T.concatenate((input,currentcol),axis=1)
        #input = input_unlag
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W = numpy.asarray(rng.uniform(
                    low=-.2*numpy.sqrt(6. / (n_in + n_out)),
                    high=.2*numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
                  #if activation == theano.tensor.nnet.sigmoid:
                  #W_values *= 4
                  #W_values = numpy.random.randn(n_in, n_out).astype(theano.config.floatX)
       
       #W_unnorm = numpy.random.randn(n_in,n_out).astype(theano.config.floatX)
       # W_norms = numpy.sqrt(numpy.sum(W_unnorm**2,axis=0))
       #W = W_unnorm / W_norms
       #self.W = theano.shared(value=W,
       #                      name='W', borrow=True)
                               
        #W_values, s, v = numpy.linalg.svd(Q)
        #W_values = 1e-2*numpy.random.randn(n_in,n_out).astype(theano.config.floatX)
        self.W = theano.shared(value=W, name='W_hid', borrow=True)

        if b is None:
           b = 1e-14*numpy.ones((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b, name='b_hid', borrow=True)


        # helper variables for adagrad
        self.W_helper = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_helper', borrow=True)
        self.b_helper = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_helper', borrow=True)

        # helper variables for L1
        self.W_helper2 = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_helper2', borrow=True)
        self.b_helper2 = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_helper2', borrow=True)

        ## helper variables for lr
        #self.W_helper3 = theano.shared(value=numpy.zeros((n_in, n_out), \
        #    dtype=theano.config.floatX), name='W_helper3', borrow=True)
        #self.b_helper3 = theano.shared(value=numpy.zeros((n_out,), \
        #    dtype=theano.config.floatX), name='b_helper3', borrow=True)
        
        ## helper variables for stepsize
        #self.W_helper4 = theano.shared(value=numpy.zeros((n_in, n_out), \
        #    dtype=theano.config.floatX), name='W_helper4', borrow=True)
        #self.b_helper4 = theano.shared(value=numpy.zeros((n_out,), \
        #    dtype=theano.config.floatX), name='b_helper4', borrow=True)
            
        # parameters of this layer
        self.params = [self.W, self.b]
        self.params_helper = [self.W_helper, self.b_helper]
        self.params_helper2 = [self.W_helper2, self.b_helper2]
        #self.params_helper3 = [self.W_helper3, self.b_helper3]
        #self.params_helper4 = [self.W_helper4, self.b_helper4]
       
        lin_output = T.dot(input, self.W) + self.b

        # Hidden unit activation is given by: tanh(dot(input,W) + b)
        self.output = T.tanh(lin_output)

        # Hidden unit activation is rectified linear
        #self.output = T.minimum(lin_output*(lin_output>0),10)

        # Hidden unit activation is None (i.e. linear)
        #self.output = lin_output

    def hidden_L1norm(self):
        """Return hidden unit activity norm (L1)

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        """
        return T.mean(   T.sqrt((self.output)**2)   , axis = 0)

    def hidden_MSE(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        """
        return T.mean(   T.sqrt((y - self.output)**2)   , axis = 0)



#This class is to build the LeNet-style convolution + max pooling layers + output nonlinearity
class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), dim2 = 0):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)


        # helper variables for adagrad
        self.W_helper = theano.shared(value=numpy.zeros(filter_shape, \
            dtype=theano.config.floatX), name='W_helper', borrow=True)
        self.b_helper = theano.shared(value=numpy.zeros((filter_shape[0],), \
            dtype=theano.config.floatX), name='b_helper', borrow=True)

        # helper variables for L1
        self.W_helper2 = theano.shared(value=numpy.zeros(filter_shape, \
            dtype=theano.config.floatX), name='W_helper2', borrow=True)
        self.b_helper2 = theano.shared(value=numpy.zeros((filter_shape[0],), \
            dtype=theano.config.floatX), name='b_helper2', borrow=True)

        # parameters of this layer
        self.params = [self.W, self.b]
        self.params_helper = [self.W_helper, self.b_helper]
        self.params_helper2 = [self.W_helper2, self.b_helper2]

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape, border_mode='full')

        #convert to "same" (from full)
        s1 = numpy.floor((filter_shape[2]-1)/2.0).astype(int)
        e1 = numpy.ceil((filter_shape[2]-1)/2.0).astype(int)

        if dim2: #convert to "valid" (from full) 
            s2 = (filter_shape[3]-1)
            e2 = (filter_shape[3]-1)
            conv_out = conv_out[:,:,s1:-e1,s2:-e2]
        else:
            conv_out = conv_out[:,:,s1:-e1,:]

        #convert to "same" (from full)
        #s2 = numpy.floor((filter_shape[3]-1)/2.0).astype(int)
        #e2 = numpy.ceil((filter_shape[3]-1)/2.0).astype(int)
        #conv_out = conv_out[:,:,s1:-e1,s2:-e2]

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height

        lin_output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x');

        # Activation is given by sigmoid:
        #self.output = T.tanh(lin_output)

        # Activation is rectified linear
        self.output = lin_output*(lin_output>0)



# This class is to build a LSTM RNN layer (for now only excitatory)
# Used notation and equations from: http://deeplearning.net/tutorial/lstm.html
class LSTM(object):
    def __init__(self, rng, input, n_in, n_out):
        
        self.input = input
        
        w_mag = .01

        ##  Create parameters ##
        # initializations?? #
        # Weights from input to gates
        W_i_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_in, n_out)), dtype=theano.config.floatX) #weights from input to input gate
        W_i = theano.shared(value=W_i_values, name='W_i', borrow=True)
        W_f_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_in, n_out)), dtype=theano.config.floatX) #weights from input to forget gate
        W_f = theano.shared(value=W_f_values, name='W_f', borrow=True)
        W_c_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_in, n_out)), dtype=theano.config.floatX) #weights from input to memory cells directly
        W_c = theano.shared(value=W_c_values, name='W_c', borrow=True)
        W_o_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_in, n_out)), dtype=theano.config.floatX) #weights from input to output gate
        W_o = theano.shared(value=W_o_values, name='W_o', borrow=True)
        
        # Weights from previous output to gates
        U_i_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_out, n_out)), dtype=theano.config.floatX) #weights from last outputs to input gate
        U_i = theano.shared(value=U_i_values, name='U_i', borrow=True)
        U_f_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_out, n_out)), dtype=theano.config.floatX) #weights from last outputs to forget gate
        U_f = theano.shared(value=U_f_values, name='U_f', borrow=True)
        U_c_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_out, n_out)), dtype=theano.config.floatX) #weights from last outputs to memory cell
        U_c = theano.shared(value=U_c_values, name='U_c', borrow=True)
        U_o_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_out, n_out)), dtype=theano.config.floatX) #weights from last outputs to output gate
        U_o = theano.shared(value=U_o_values, name='U_o', borrow=True)
        
        V_o_values = numpy.asarray(rng.uniform(low=-w_mag, high=w_mag, size=(n_out, n_out)), dtype=theano.config.floatX) #weights from memory cell state to output gate
        V_o = theano.shared(value=V_o_values, name='V_o', borrow=True)
        
        # Biases of gates
        b_i_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        b_i = theano.shared(value=b_i_values, name='b_i', borrow=True)
        b_f_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        b_f = theano.shared(value=b_f_values, name='b_f', borrow=True)
        b_c_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        b_c = theano.shared(value=b_c_values, name='b_c', borrow=True)
        b_o_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        b_o = theano.shared(value=b_o_values, name='b_o', borrow=True)
        
        
        # Assign parameters to self
        self.W_i = W_i
        self.W_f = W_f
        self.W_c = W_c
        self.W_o = W_o
        self.U_i = U_i
        self.U_f = U_f
        self.U_c = U_c
        self.U_o = U_o
        self.V_o = V_o
        self.b_i = b_i
        self.b_f = b_f
        self.b_c = b_c
        self.b_o = b_o
        
        # Helper variables for adagrad
        self.W_i_helper = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_i_helper', borrow=True)
        self.W_f_helper = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_f_helper', borrow=True)
        self.W_c_helper = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_c_helper', borrow=True)
        self.W_o_helper = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_o_helper', borrow=True)
        self.U_i_helper = theano.shared(value=numpy.zeros((n_out, n_out), \
            dtype=theano.config.floatX), name='U_i_helper', borrow=True)
        self.U_f_helper = theano.shared(value=numpy.zeros((n_out, n_out), \
            dtype=theano.config.floatX), name='U_f_helper', borrow=True)
        self.U_c_helper = theano.shared(value=numpy.zeros((n_out, n_out), \
            dtype=theano.config.floatX), name='U_c_helper', borrow=True)
        self.U_o_helper = theano.shared(value=numpy.zeros((n_out, n_out), \
            dtype=theano.config.floatX), name='U_o_helper', borrow=True)
        self.V_o_helper = theano.shared(value=numpy.zeros((n_out, n_out), \
            dtype=theano.config.floatX), name='V_o_helper', borrow=True)
        self.b_i_helper = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_i_helper', borrow=True)
        self.b_f_helper = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_f_helper', borrow=True)
        self.b_c_helper = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_c_helper', borrow=True)
        self.b_o_helper = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_o_helper', borrow=True)
                                                          
        # Helper variables for L1
        self.W_i_helper2 = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_i_helper2', borrow=True)
        self.W_f_helper2 = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_f_helper2', borrow=True)
        self.W_c_helper2 = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_c_helper2', borrow=True)
        self.W_o_helper2 = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_o_helper2', borrow=True)
        self.U_i_helper2 = theano.shared(value=numpy.zeros((n_out, n_out), \
            dtype=theano.config.floatX), name='U_i_helper2', borrow=True)
        self.U_f_helper2 = theano.shared(value=numpy.zeros((n_out, n_out), \
            dtype=theano.config.floatX), name='U_f_helper2', borrow=True)
        self.U_c_helper2 = theano.shared(value=numpy.zeros((n_out, n_out), \
            dtype=theano.config.floatX), name='U_c_helper2', borrow=True)
        self.U_o_helper2 = theano.shared(value=numpy.zeros((n_out, n_out), \
            dtype=theano.config.floatX), name='U_o_helper2', borrow=True)
        self.V_o_helper2 = theano.shared(value=numpy.zeros((n_out, n_out), \
            dtype=theano.config.floatX), name='V_o_helper2', borrow=True)
        self.b_i_helper2 = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_i_helper2', borrow=True)
        self.b_f_helper2 = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_f_helper2', borrow=True)
        self.b_c_helper2 = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_c_helper2', borrow=True)
        self.b_o_helper2 = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_o_helper2', borrow=True)
                                                          
        # parameters of this layer
        self.params = [self.W_i, self.W_f, self.W_c, self.W_o, self.U_i, self.U_f, self.U_c, self.U_o, self.V_o, self.b_i, self.b_f, self.b_c, self.b_o]
        self.params_helper = [self.W_i_helper, self.W_f_helper, self.W_c_helper, self.W_o_helper, self.U_i_helper, self.U_f_helper, self.U_c_helper, self.U_o_helper, self.V_o_helper, self.b_i_helper, self.b_f_helper, self.b_c_helper, self.b_o_helper]
        self.params_helper2 = [self.W_i_helper2, self.W_f_helper2, self.W_c_helper2, self.W_o_helper2, self.U_i_helper2, self.U_f_helper2, self.U_c_helper2, self.U_o_helper2, self.V_o_helper2, self.b_i_helper2, self.b_f_helper2, self.b_c_helper2, self.b_o_helper2]
                                                          
        # initial hidden state values
        h_0 = T.zeros((n_out,))
        ##intialize memory cell (c) values with zeros?? ##
        c_0 = T.zeros((n_out,))
                                                          
        # recurrent function with rectified linear output activation function (u is input, h is hidden activity)
        def step(u_t, h_tm1, c_tm1):
            input_gate = T.nnet.sigmoid(T.dot(u_t,self.W_i)+T.dot(h_tm1,self.U_i) + self.b_i)
            forget_gate = T.nnet.sigmoid(T.dot(u_t,self.W_f)+T.dot(h_tm1,self.U_f)+self.b_f)
            c_candidate = T.tanh(T.dot(u_t,self.W_c)+T.dot(h_tm1,U_c)+self.b_c)
            c_t = c_candidate*input_gate + c_tm1*forget_gate
            output_gate = T.nnet.sigmoid(T.dot(u_t,self.W_o)+T.dot(h_tm1,self.U_o)+ T.dot(c_t,self.V_o) + b_o)
            h_t = T.tanh(c_t)*output_gate
            return h_t, c_t
                                                                                  
        # compute timeseries
        [h, c], _ = theano.scan(step,
                            sequences=self.input,
                            outputs_info=[h_0, c_0],
                            truncate_gradient=-1)
                                                                                      
       # output activity
        self.output = h


# This class is to build a RNN 
class RNN(object):
    def __init__(self, rng, input, n_in,n_out,burnin):
        """
            RNN hidden layer: units are fully-connected and have
            an activation function (see below). Weights project inputs to the 
            units which are recurrently connected.
            Weight matrix W is of shape (n_in,n_out)
            and the bias vector b is of shape (n_out,).
            
            :type rng: numpy.random.RandomState
            :param rng: a random number generator used to initialize weights
            
            :type input: theano.tensor.dmatrix
            :param input: a symbolic tensor of shape (n_examples, n_in)
            
            :type n_in: int
            :param n_in: dimensionality of input
            
            :type n_out: int
            :param n_out: dimensionality of output
            
            :type burnin: int
            :param burnin: throw away predictions before that time
            
            
            """
        self.input = input
        
        def get_orthogonal_vals(M,N):
            Q = numpy.random.randn(M, N).astype(theano.config.floatX)
            u, s, v = numpy.linalg.svd(Q)
            if M>N:
                return u[:,0:N]
            else:
                return v[0:M,:]

        W_values = get_orthogonal_vals(n_in, n_out)
        #W_values = .01*numpy.random.randn(n_in, n_out).astype(theano.config.floatX)
        W = theano.shared(value=W_values, name='W', borrow=True)
        
        b_values = 0*numpy.ones((n_out,), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, name='b', borrow=True)
        
        Q = numpy.random.randn(n_out, n_out).astype(theano.config.floatX)
        W_RNN_values, s, v = numpy.linalg.svd(Q)
        W_RNN = theano.shared(value=W_RNN_values, name='W_RNN', borrow=True)

        self.W = W
        self.b = b
        self.W_RNN = W_RNN


        # helper variables for adagrad
        self.W_helper = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_helper', borrow=True)
        self.W_RNN_helper = theano.shared(value=numpy.zeros((n_out, n_out), \
            dtype=theano.config.floatX), name='W_RNN_helper', borrow=True)
        self.b_helper = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_helper', borrow=True)

        # helper variables for L1
        self.W_helper2 = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_helper2', borrow=True)
        self.W_RNN_helper2 = theano.shared(value=numpy.zeros((n_out, n_out), \
            dtype=theano.config.floatX), name='W_RNN_helper2', borrow=True)
        self.b_helper2 = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_helper2', borrow=True)

        # parameters of this layer
        self.params = [self.W_RNN, self.W, self.b]
        self.params_helper = [self.W_RNN_helper, self.W_helper, self.b_helper]
        self.params_helper2 = [self.W_RNN_helper2, self.W_helper2, self.b_helper2]

        #initial hidden state values
        h_0 = T.zeros((n_out,))
#        print 'W_RNN= ' , W_RNN.get_value().shape
#        print 'W= ' , W.get_value().shape
        # recurrent function with rectified linear output activation function (u is input, h is hidden activity)
        def step(u_t, h_tm1):
            lin_E = T.dot(u_t, self.W) - T.dot(h_tm1, self.W_RNN) + self.b
            #h_t = lin_E*(lin_E>0)
            h_t = 1.7159*T.tanh(2*lin_E/3) # normalized tanh
            return h_t

        # compute the hidden E & I timeseries
        h, _ = theano.scan(step,
                   sequences=self.input,
                   outputs_info=h_0,
                   truncate_gradient=-1)
            
        # output activity is the hidden unit activity
        self.output = h[burnin:]
