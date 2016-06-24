import random, time, re, MySQLdb

import pandas as pd
import numpy as np
from scipy import sparse
import nltk

from recom.string_tools import StringHasher, StringCleaner


def generate_vocab(papers):
    """Returns the vocabulary used in the papers given in parameters, after cleaning and stopwords removal.

    Args:
        papers (list of tuples): the raw list of papers from which generates the vocabulary (each element is a tuple of 3 strings: id, title and abstract)

    Returns:
        list of strings: the list of tokens forming the vocabulary
    """
    sc = StringCleaner()

    # Generate author's vocabulary
    corpus = " ".join(p[1] + " " + p[2] for p in papers)
    # Cleaning
    corpus = sc.clean_string(corpus)
    # Tokenization
    pattern = r"(?:[A-Z]\.)+|\w+(?:-\w+)*|\d+(?:\.\d+)?%?"
    #         we keep tokens that are words (with optional internal hyphens), acronyms and percentages
    tokens = set(nltk.regexp_tokenize(corpus, pattern)) - set(nltk.corpus.stopwords.words("english"))
    num_re = re.compile("^\d+$")
    tokens = set([t for t in tokens if not num_re.match(t)]) # we remove only-numeric tokens
    # Stemming
    porter = nltk.stem.PorterStemmer()

    return [porter.stem(t) for t in tokens]


def compute_features(papers, stringHasher, verbosity=1):
    """Computes the features of a list of papers, with a given list of ngrams.

    Args:
        papers (list of tuples): the list of papers (each element is a tuple of 3 strings: id, title, abstract)
        stringHasher (:class:`recom.string_tools.StringHasher`): the object which contains the list of ngrams
        verbosity (int): 0: quiet; 1: normal; 2: high

    Returns:
        dict of :class:`np.ndarray`: the list of papers represented as bag-of-words vectors
    """
    sc = StringCleaner()
    sh = stringHasher

    papers_feat = {}
    total = len(papers)
    start_time = time.time()

    i = 1
    for p_id, p_title, p_abstract in papers:
        title = sc.clean_string(p_title)
        abstract = sc.clean_string(p_abstract)
        to_hash = title + " " + abstract
        papers_feat[p_id] = sh.hash(to_hash)
        if verbosity > 1 and i % 100 == 0:
            print("Paper %d over %d" % (i, total))
        i += 1

    if verbosity > 0:
        print("Processed {} papers in {:.3f}s".format(len(papers), time.time() - start_time))

    return papers_feat


def invert_citations(citations, verbosity=1):
    """Transforms a list of citation relations into a hashtable cited_paper -> list of citing papers.

    Args:
        citations (list of tuples): the list of citation relations (each element is a tuple of 2 string: *citing* paper's id, *cited* paper's id)
        verbosity (int): 0: quiet; 1: normal; 2: high

    Returns:
        dict: a dict whose keys are cited papers ids and whose values are the lists of the ids of the papers that cite the keys (string -> list of strings)
    """
    citations_assoc = {}
    total = len(citations)
    i = 1
    start_time = time.time()
    for (cited_by, cited_paper) in citations:
        if cited_paper in citations_assoc.keys():
            citations_assoc[cited_paper].append(cited_by)
        else:
            citations_assoc[cited_paper] = [cited_by]
        if verbosity > 1 and i % 500 == 0:
            print("Citation %d over %" % (i, total))
        i += 1

    if verbosity > 0:
        print("Processed {} citation relations in {:.3f}s".format(len(citations), time.time() - start_time))

    return citations_assoc


def prepare_dataset(user_papers, citations, cited_papers, tokens, bad_papers=None, verbosity=1):
    """Prepares data from string representations of papers in order to buidl a numeric dataset.

    The result is a tuple of 4 elements:

    (1) the user's papers, as a dictionary: each key is a the id of a paper written by the user,
    and the value is the features of the paper (1D np.ndarray),
    (2) the cited papers, as a dictionary: each key is the id of a paper cited by the user,
    and the value is a tuple constituted of the list of papers id in which the paper is cited (list of strings),
    and the features of the paper (1D np.ndarray),
    (3) the irrelevant papers, as a dictionary like the first one,  
    (4) the ngrams used to compute the features (list of strings).

    Args:
        user_papers (list of 3-tuples): the papers written by the user (each element is a tuple of 3 strings: id, title, abstract)
        citations (list of 2-tuples): the list of citation relations
        cited_papers (list of 3-tuples): the papers that the user has cited (each element is a tuple of 3 strings: id, title, abstract)
        tokens (list of strings): the vocabulary to use for computing features
        bad_papers (list of 3-tuples or None): unrelated papers (each element is a tuple of 3 strings: id, title, abstract)
        verbosity (int): 0: quiet; 1: normal; 2: high

    Returns:
        tuple: data to build a dataset with
    """
    # Verbosity: 0 = None, 1 = Few details, 2 = Much details

    sh = StringHasher()
    sc = StringCleaner()

    # Initiate the ngrams (specific to the author)
    sh.init_ngrams(tokens)

    # Hash user's papers' titles and abstracts
    papers_feat = compute_features(user_papers, sh, verbosity)

    citations_assoc = invert_citations(citations, verbosity)

    # Hash cited papers' titles and abstracts
    citations_feat = {}
    total = len(cited_papers)
    i = 1
    start_time = time.time()
    for (p_id, p_title, p_abstract) in cited_papers:
        title = sc.clean_string(p_title)
        abstract = sc.clean_string(p_abstract)
        to_hash = title + " " + abstract
        citations_feat[p_id] = (citations_assoc[p_id], sh.hash(to_hash)) # [, np.ndarray]
        if verbosity > 1 and i % 500 == 0:
            print("Cited paper %d over %" % (i, total))
        i += 1

    if verbosity > 0:
        print("Processed {} cited papers in {:.3f}s".format(len(cited_papers), time.time() - start_time))

    if bad_papers is None:
        if verbosity > 0:
            print("Done.")

        return papers_feat, citations_feat, None, sh.ngrams_

    # Hash bad papers' titles and abstracts
    bad_feat = compute_features(bad_papers, sh, verbosity)

    if verbosity > 0:
        print("Done.")

    # we also return the author's specific list of ngrams (for future hashing)
    return papers_feat, citations_feat, bad_feat, sh.ngrams_


def build_dataset(papers, citations, bad_papers, num_entries=6, verbosity=1):
    """Build a dataset from features variables.

    Args:
        papers (dict): features of the user's papers (dict string -> np.ndarray)
        citations (dict): features of cited papers (dict string -> tuple(list of string, np.ndarray))
        bad_papers (dict): features of unrelated papers (dict string -> np.ndarray)
        num_entries (int): the number of compared papers in the DSSM structure
        verbosity (int): 0: quiet; 1: normal; 2: high

    Returns:
        :class:`np.ndarray`: the dataset
    """
    start_time = time.time()

    # Number of non citing papers needed to complete one block of dataset
    num_others = num_entries - 2

    # Init result
    reps = 4
    num_samples = reps*len(citations)
    num_feats = list(papers.values())[0].shape[0]
    dataset = np.empty((num_samples, num_entries, num_feats))

    sample = 0
    acceptable_idx = set(bad_papers.keys())
    for rep in range(reps):
        for c_id, val in citations.items():
            dataset[sample][0] = val[1] # features of the cited paper

            citing_papers = val[0] # indexes of papers that cite c_id
            # select the features of one of these papers, randomly
            one_citing_paper = citing_papers[random.randrange(len(citing_papers))]
            dataset[sample][1] = papers[one_citing_paper]

            # select num_others "bad" papers
            selected_idx = random.sample(acceptable_idx, num_others)
            for i in range(num_others):
                dataset[sample][2+i] = bad_papers[selected_idx[i]]

            sample += 1
            if verbosity > 1 and sample % 500 == 0:
                print("Sample {} over {}".format(sample, num_samples))

    if verbosity > 0:
        print("Generated dataset with {} samples in {:.3f}s".format(num_samples, time.time() - start_time))

    return dataset


def dataset_to_file(dataset, ngrams, filename='dataset'):
    """Save a dataset to a file.

    Args:
        dataset (:class:`np.ndarray`): the dataset to save (built with :func:`recom.dataset_tools.build_dataset`)
        ngrams (list of strings): the ngrams used to compute the features
        filename (string): the filename without extension (will be .npz)
    """
    num_samples, num_entries, num_features = dataset.shape
    # We rehaspe the ndarray from 3D to 2D in order to write it into a text file
    # Each line of the file will correspond to one cited paper
    # Therefore, on each there will be the `num_entries` sets of features
    dataset_sp = sparse.csr_matrix(dataset.reshape(num_samples*num_entries, num_features))
    np.savez(filename, num_entries=np.array([num_entries]), data=dataset_sp.data, indices=dataset_sp.indices,
             indptr=dataset_sp.indptr, shape=dataset_sp.shape, ngrams=ngrams)


def dataset_from_file(filename):
    """Load a dataset from file.

    Args:
        filename (string): the name of the file from which extract the dataset

    Returns:
        tuple: the dataset (np.ndarray) and the ngrams (list of strings)
    """
    loader = np.load(filename)
    num_entries = loader['num_entries'][0]
    sp_dataset = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])
    dataset = sp_dataset.toarray()
    samp_entries, num_features = dataset.shape
    return dataset.reshape(int(samp_entries / num_entries), num_entries, num_features), loader['ngrams']
