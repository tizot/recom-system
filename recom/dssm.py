import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

from lasagne import layers, init, nonlinearities


def build_multi_dssm(input_var=None, num_samples=None, num_entries=6, num_ngrams=42**3, num_hid1=300, num_hid2=300, num_out=128):
    """Builds a DSSM structure in a Lasagne/Theano way.

    The built DSSM is the neural network that computes the projection of only one paper.
    The input ``input_var`` should have two dimensions: (``num_samples * num_entries``, ``num_ngrams``).
    The output is then computed in a batch way: one paper at a time, but all papers from the same sample in the dataset are grouped
    (cited papers, citing papers and ``num_entries - 2`` irrelevant papers).

    Args:
        input_var (:class:`theano.tensor.TensorType` or None): symbolic input variable of the DSSM
        num_samples (int): the number of samples in the batch input dataset (number of rows)
        num_entries (int): the number of compared papers in the DSSM structure
        num_ngrams (int): the size of the vocabulary
        num_hid1 (int): the number of units in the first hidden layer
        num_hid2 (int): the number of units in the second hidden layer
        num_out (int): the number of units in the output layer

    Returns:
        :class:`lasagne.layers.Layer`: the output layer of the DSSM
    """

    assert (num_entries > 2)

    # Initialise input layer
    if num_samples is None:
        num_rows = None
    else:
        num_rows = num_samples * num_entries

    l_in = layers.InputLayer(shape=(num_rows, num_ngrams), input_var=input_var)

    # Initialise the hidden and output layers or the DSSM
    l_hid1 = layers.DenseLayer(l_in, num_units=num_hid1, nonlinearity=nonlinearities.tanh, W=init.GlorotUniform())
    l_hid2 = layers.DenseLayer(l_hid1, num_units=num_hid2, nonlinearity=nonlinearities.tanh, W=init.GlorotUniform())
    l_out = layers.DenseLayer(l_hid2, num_units=num_out, nonlinearity=nonlinearities.tanh, W=init.GlorotUniform())

    l_out = layers.ExpressionLayer(l_out, lambda X: X / X.norm(2), output_shape='auto')

    return l_out


def compute_loss(output, num_samples, num_entries=6, gamma=500.0):
    """Compute the loss of a dataset, given the output of the DSSM.

    Args:
        output (:class:`lasagne.layers.Layer`): the output of the DSSM
        num_samples (int): the number of samples in the dataset
        num_entries (int): the number of compared papers in the DSSM structure
        gamma (float): the coefficient applied in the softmax of the similarities

    Returns:
        theano.tensor.TensorType: the loss of the dataset
    """
    assert (num_entries > 2)
    assert (num_samples > 0)

    # Post-NN operations to compute the loss
    # First, we extract the first output of each bundle
    mask = np.zeros(num_entries * num_samples)
    mask[::num_entries] = 1
    unmask = np.ones(num_entries * num_samples) - mask
    cited = T.extra_ops.compress(mask, output, axis=0)
    odocs = T.extra_ops.compress(unmask, output, axis=0)

    # We duplicate each row 'x' num_entries-1 times
    cited = T.extra_ops.repeat(cited, num_entries-1, axis=0)
    # Then we compute element-wise product of x with each y, for each bundle
    sims = T.sum(cited * odocs, axis=1)

    # We reshape the similarities
    sims = T.reshape(sims, (num_samples, num_entries-1))
    sims = gamma * sims

    # We take the softmax of each row
    probs = T.nnet.softmax(sims)

    # We compute the loss as the sum of element on the first column
    loss_mask = np.zeros(num_entries-1)
    loss_mask[0] = 1
    loss = T.extra_ops.compress(loss_mask, probs, axis=1)

    return -T.log(T.prod(loss))


def iterate_minibatches(inputs, batchsize, shuffle=False):
    """Produces an batch iterator over the input.

    Usage:
        >>> for batch in iterate_minibatches(inputs, batchsize, shuffle):
        >>>    # to stuff

    Args:
        inputs (list): the input list over which iterate
        batchsize (int): the size of each batch
        shuffle (bool): if True, ``inputs`` is shuffled before iteration
    """
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]
