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
from lasagne import layers, init, nonlinearities

from dataset_tools import *
from dssm import *
from string_tools import *
from data_retrieval import *


def compute_features_batch(papers, ngrams=None, verbosity=1):
    """Compute the features of the given list of papers, w.r.t. the ngrams.

    Args:
        papers (list of dicts): the list of papers whose features are to be computed
        ngrams (list of strings): the n-grams with which we compute the features
        verbosity (int): 0: quiet, 1: normal, 2: high

    Returns:
        dict: the features of each paper, identified by its id
    """
    sh = StringHasher()
    sc = StringCleaner()

    if ngrams is None:
        # Generate author's vocabulary
        tokens = generate_vocab(papers)

        # Initiate the ngrams (specific to the author)
        sh.init_ngrams(tokens)
    else:
        sh.load_ngrams(ngrams)

    # Hash user's papers' titles and abstracts
    papers_feat = {}
    total = len(papers)
    i = 1
    start_time = time.time()
    for p in papers:
        title = sc.clean_string(p['title'])
        abstract = sc.clean_string(p['abstract'])
        to_hash = title + " " + abstract
        papers_feat[p['index']] = sh.hash(to_hash)
        if verbosity > 1 and i % 100 == 0:
            print("Paper %d over %d" % (i, total))
        i += 1

    return papers_feat


def main(dataset, author_papers_file, author_name, author_slug, author_dssm, unseen_papers):
    """Given a stream of unseen papers, decides if each paper should be recommended or not.

    Args:
        dataset (string): path to the user's dataset
        author_papers_file (string): path to the user's papers file (raw text file)
        author_name (string): author's full name
        author_slug (string): short and ASCII string for the author's name
        author_dssm (string): path to the trained DSSM's parameters file
        unseen_papers (string): path to the raw file containing unseen papers' titles and abstracts
    """

    # Load DSSM params and user's papers (raw data)
    papers_loader = np.load(dataset)
    ngrams = papers_loader['ngrams']
    num_entries = papers_loader['num_entries'][0]

    user_papers_raw, _ = get_author_papers(author_name, author_slug, author_papers_file)
    papers_feat = compute_features_batch(user_papers_raw, ngrams)

    # Build DSSM, load params and compute user's papers projections
    num_samples = 1
    dssm_loader = np.load(author_dssm)
    dssm_struct = dssm_loader['dssm_struct'].reshape(1, -1)[0, 0]
    num_hid1 = dssm_struct['num_hid1']
    num_hid2 = dssm_struct['num_hid2']
    num_out = dssm_struct['num_out']
    gamma = dssm_struct['gamma']

    input_var = T.matrix()
    dssm_values = dssm_loader['dssm']
    network = build_multi_dssm(input_var=input_var,
                                  num_samples=num_samples,
                                  num_entries=num_entries,
                                  num_ngrams=len(ngrams),
                                  num_hid1=num_hid1,
                                  num_hid2=num_hid2,
                                  num_out=num_out)
    lasagne.layers.set_all_param_values(network, dssm_values)
    prediction = lasagne.layers.get_output(network, deterministic=True)
    output = prediction / prediction.norm(L=2)
    f = theano.function([input_var], output)

    user_papers = [f(x.reshape(1, -1))[0] for _, x in papers_feat.items()]

    # Compute scores for the unseen papers
    r_index = re.compile('^#index(.*)')
    r_author = re.compile('^#@(.*)')
    r_title = re.compile('^#\*(.*)')
    r_abstract = re.compile('^#!(.*)')
    r_cite = re.compile('^#%(.*)')

    unseen_papers_raw = get_irrelevant_papers(unseen_papers)
    unseen_papers = [list2paper(p, r_index, r_author, r_title, r_abstract, r_cite) for p in unseen_papers_raw]
    unseen_feats = compute_features_batch(unseen_papers, ngrams)

    # Compute similarities
    sims = [np.array([np.dot(paper, y)[0] for y in user_papers]) for paper in unseen_papers]


if __name__ == '__main__':
    # Parse command
    usage = "usage: %prog [options] args"

    parser = ArgumentParser()

    parser.add_argument("-d", "--dataset", type=str, help="Path to the user dataset")
    parser.add_argument("-n", "--name", type=str, help="Name of the author")
    parser.add_argument("-s", "--slug", type=str, help="Short ASCII string for the author's name")
    parser.add_argument("-af", "--author_file", type=str, help="Path to the author's papers file")
    parser.add_argument("-v", "--dssm_values", type=str, help="Path to the trained DSSM's parameters file")
    parser.add_argument("-u", "--unseen_papers", type=str, help="Path to the unseen papers file")

    args = parser.parse_args()

    kwargs = {'dataset': args.dataset,
              'author_papers_file': args.author_file,
              'author_name': args.name,
              'author_slug': args.slug,
              'author_dssm': args.dssm_values,
              'unseen_papers': args.unseen_papers}

    main(**kwargs)
