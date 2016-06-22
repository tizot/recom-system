import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

from lasagne import layers, init, nonlinearities
    

def build_multi_dssm(input_var=None, num_samples=None, num_entries=6, num_ngrams=42**3, num_hid1=300, num_hid2=300, num_out=128):
    assert (num_entries > 2)
    
    # Initialise input layer
    if num_samples is None:
        num_rows = None
    else:
        num_rows = num_samples * num_entries
        
    l_in = layers.InputLayer(shape=(num_rows, num_ngrams), input_var=input_var)
    
    # Initialise the hidden and output layers or the first DSSM
    l_hid1 = layers.DenseLayer(l_in, num_units=num_hid1, nonlinearity=nonlinearities.tanh, W=init.GlorotUniform())
    l_hid2 = layers.DenseLayer(l_hid1, num_units=num_hid2, nonlinearity=nonlinearities.tanh, W=init.GlorotUniform())
    l_out = layers.DenseLayer(l_hid2, num_units=num_out, nonlinearity=nonlinearities.tanh, W=init.GlorotUniform())
    
    l_out = layers.ExpressionLayer(l_out, lambda X: X / X.norm(2), output_shape='auto')
    
    return l_out
