#!/usr/bin/env python
# coding=utf-8

import sys
import os
import time
from argparse import ArgumentParser

import numpy as np
import theano
import theano.tensor as T
import lasagne

from dssm import *
from dataset_tools import *


def main(author_id=None, num_epochs=100, num_entries=6, num_hid1=300, num_hid2=300, num_out=128, learning_rate=0.1, input_file=None, output_file='output'):
    """Builds a DSSM and trains it with the dataset of the given author.

    The DSSM parameters are saved into a file, in order to be used for recommendation.
    You must specify either the author's id, or the dataset input file.

    Args:
        author_id (int or None): id of the author in the SQL database
        num_epochs (int): number of iterations in the training
        num_entries (int): number of compared papers in the DSSM structure
        num_hid1 (int): number of units in the first hidden layer
        num_hid2 (int): number of units in the second hidden layer
        num_out (int): number of units in the output layer
        learning_rate (float): parameter of the SGD training algorithm
        input_file (string or None): path to the dataset file of the author
        output_file (string): path to the output file (DSSM parameters)
    """

    if author_id is None and input_file is None:
        return 1

    # The input should be a tensor (3D np.array): each matrix (2D np.array) in this tensor has the structure:
    # [x (cited paper), p+ (citing paper), p1-, ..., pn- (non-citing papers)]
    if author_id is None:
        print("Retrieving dataset from file...")
        inputs, ngrams = dataset_from_file(input_file)
    else:
        print("Retrieving data from SQL DB...")
        #user_papers, citations, cited_papers = retrieve_data(author_id)
        # TODO: remove that
        print("Preparing dataset...")
        papers_feat, citations_feat, ngrams = prepare_dataset(user_papers, citations, cited_papers)
        print("Building computable dataset...")
        inputs = build_dataset(papers_feat, citations_feat, num_entries)

    # Build a DSSM (with several entries)
    print("Building the DSSM structure...")
    num_samples = 200
    gamma = 500

    input_var = T.matrix()
    network = build_multi_dssm(input_var=input_var,
                                  num_samples=num_samples,
                                  num_entries=num_entries,
                                  num_ngrams=len(ngrams),
                                  num_hid1=num_hid1,
                                  num_hid2=num_hid2,
                                  num_out=num_out)
    prediction = lasagne.layers.get_output(network)

    # Post-NN operations to compute the loss
    loss = compute_loss(prediction, num_samples, num_entries, gamma)

    # NN train function
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adadelta(loss, params, learning_rate=learning_rate)
    train_fn = theano.function([input_var], loss, updates=updates)

    # Let's train our network
    loss_values = np.zeros(num_epochs)

    print("Beginning of DSSM training...")
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch_ in iterate_minibatches(inputs, num_samples, shuffle=True):
            batch = np.reshape(batch_, (num_entries * num_samples, len(ngrams)))
            train_err += train_fn(batch)
            train_batches += 1

        loss_values[epoch] = np.float64(train_err / train_batches)

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(np.float64(train_err / train_batches)))

    # We save the DSSM parameters
    dssm_struct = {
        'num_entries': num_entries,
        'num_hid1': num_hid1,
        'num_hid2': num_hid2,
        'num_out': num_out,
        'learning_rate': learning_rate,
        'gamma': gamma
    }
    np.savez(output_file, dssm=lasagne.layers.get_all_param_values(network), dssm_struct=dssm_struct, losses=loss_values)


if __name__ == '__main__':
    # Parse command
    usage = "usage: %prog [options] args"

    parser = ArgumentParser()

    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of iterations for the training")
    parser.add_argument("-n1", "--num_hidden1", dest="num_hid1", type=int, default=300, help="Number of units in the first hidden layer")
    parser.add_argument("-n2", "--num_hidden2", dest="num_hid2", type=int, default=300, help="Number of units in the second hidden layer")
    parser.add_argument("-no", "--num_out", dest="num_out", type=int, default=18, help="Number of units in the output layer")
    parser.add_argument("-c", "--num_compare", dest="num_compare", type=int, default=4, help="Number of non relevant papers to consider for training")
    parser.add_argument("-r", "--rate", dest="learning_rate", type=float, default=0.1, help="Learning rate for the SGD")
    parser.add_argument("-o", "--output", dest="output_filename", type=str, help="Filename for output")

    group = parser.add_mutually_exclusive_group()

    group.add_argument("-a", "--author", dest="author", type=int, help="ID of the user in the database")
    group.add_argument("-i", "--input", dest="input_filename", type=str, help="Filename for dataset input")

    args = parser.parse_args()

    if not args.author and not args.input_filename:
        parser.print_help()

    kwargs = {'num_epochs': args.epochs,
              'num_hid1': args.num_hid1,
              'num_hid2': args.num_hid2,
              'num_out': args.num_out,
              'num_entries': args.num_compare+2,
              'learning_rate': args.learning_rate,
              'input_file': args.input_filename,
              'output_file': args.output_filename}

    if not args.author:
        kwargs['author_id'] = None
    else:
        kwargs['author_id'] = args.author

    main(**kwargs)
