""" Test ORNN, RNN, and LSTM effectiveness on LSTM toy problems.
"""

import lstm_problems
import numpy as np
import lasagne
import theano
import theano.tensor as T
import util
import functools
import itertools

BATCH_SIZE = 100
N_SAMPLES = 100000
RETRACT_FREQUENCY = 10000
TEST_FREQUENCY = 10000
TEST_SIZE = 10000
RESULTS_PATH = 'results'
NUM_UNITS = 100

if __name__ == '__main__':
    # Define hyperparameter space
    task_options = [lstm_problems.add, lstm_problems.multiply,
                    lstm_problems.xor]
    sequence_length_options = [80, 400]
    orthogonalize_options = [True, False]
    in_hid_std_options = [.001, .01, .1]
    compute_updates_options = [
        lasagne.updates.adam,
        functools.partial(lasagne.updates.nesterov_momentum,
                          momentum=.995)]
    learning_rate_options = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    # Create iterator over every possible hyperparameter combination
    option_iterator = itertools.product(
        task_options, sequence_length_options, orthogonalize_options,
        in_hid_std_options, compute_updates_options, learning_rate_options)
    # Iterate over hypermarameter settings
    for (task, sequence_length, orthogonalize, in_hid_std, compute_updates,
         learning_rate) in option_iterator:
        print ('####### Learning rate: {}, updates: {}, in-hid std: {}, '
               'orthogonalize: {}, sequence_length: {}, task: {}'.format(
                   learning_rate, compute_updates, in_hid_std, orthogonalize,
                   sequence_length, task))
        # Create test set
        X_test, y_test, mask_test = task(sequence_length, TEST_SIZE)
        # Construct network
        l_in = lasagne.layers.InputLayer((None, None, X_test.shape[-1]))
        l_mask = lasagne.layers.InputLayer((None, None))
        l_rec = lasagne.layers.RecurrentLayer(
            l_in, num_units=NUM_UNITS, mask_input=l_mask,
            learn_init=True, W_in_to_hid=lasagne.init.Normal(in_hid_std),
            W_hid_to_hid=lasagne.init.Orthogonal(),
            nonlinearity=lasagne.nonlinearities.tanh)
        l_slice = lasagne.layers.SliceLayer(l_rec, -1, 1)
        l_out = lasagne.layers.DenseLayer(
            l_slice, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
        # Compute symbolic expression for predicted values
        network_output = lasagne.layers.get_output(l_out)
        # Remove a dimension from the output
        predicted_values = network_output[:, -1]
        target_values = T.vector('target_values')
        # Our cost will be mean-squared error
        cost = T.mean((predicted_values - target_values)**2)
        # Retrieve all parameters from the network
        all_params = lasagne.layers.get_all_params(l_out)
        # Compute SGD updates for training
        updates = compute_updates(cost, all_params, learning_rate)
        # Project gradient updates for recurrent hid-to-hid matrix
        if orthogonalize:
            new_update = util.tangent_grad(
                l_rec.W_hid_to_hid,
                updates[l_rec.W_hid_to_hid] - l_rec.W_hid_to_hid)
            updates[l_rec.W_hid_to_hid] = l_rec.W_hid_to_hid + new_update
        # Theano functions for training and computing cost
        train = theano.function(
            [l_in.input_var, target_values, l_mask.input_var],
            cost, updates=updates)
        # Accuracy is defined as the proportion of examples whose absolute
        # error is less than .04
        accuracy = T.mean(abs(predicted_values - target_values) < .04)
        # Theano function for computing accuracy
        compute_accuracy = theano.function(
            [l_in.input_var, target_values, l_mask.input_var], accuracy)
        # Function for orthogonalizing weight matrix
        retract_w = theano.function(
            [], [],
            updates={l_rec.W_hid_to_hid: util.retraction(l_rec.W_hid_to_hid)})
        # Keep track of the number of samples used to train
        samples_trained = 0
        while samples_trained < N_SAMPLES:
            # Generate a batch of data
            X, y, mask = task(sequence_length, BATCH_SIZE)
            cost = train(X.astype(theano.config.floatX),
                         y.astype(theano.config.floatX),
                         mask.astype(theano.config.floatX))
            # Quit when a non-finite value is found
            if any([not np.isfinite(cost),
                    any([not np.all(np.isfinite(p.get_value()))
                         for p in all_params])]):
                print '####### Non-finite values found, aborting'
                break
            # Update the number of samples trained
            samples_trained += BATCH_SIZE
            if (not samples_trained % TEST_FREQUENCY):
                print samples_trained, compute_accuracy(
                    X_test.astype(theano.config.floatX),
                    y_test.astype(theano.config.floatX),
                    mask_test.astype(theano.config.floatX))
            if orthogonalize and (not samples_trained % RETRACT_FREQUENCY):
                retract_w()
