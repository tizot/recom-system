import os, re, time
from argparse import ArgumentParser

import pandas as pd
import MySQLdb

import data_retrieval
import dataset_tools


def main(author_name, author_slug, author_papers_file, bad_papers_file, num_entries, db_name, output_file):
    """Given an author (name, papers), generates a dataset usable by the DSSM script.

    Args:
        author_name (string): the full name of the author
        author_slug (string): a short and ASCII string for the author's name (example: "Gabriella Pasi" -> "pasi")
        author_papers_file (string): the relative path to the file containing the raw data of the author's papers
        bad_papers_file (string): the relative path to the file containing the raw data of irrelevant papers
        num_entries (int): the number of compared papers in the DSSM structure (usually, 6)
        db_name (string): the name of the SQL database in which are stored all the papers
        output_file (string): the relative path to the file in which the dataset is saved
    """
    db = MySQLdb.connect(user='root', passwd='root', db=db_name)
    c = db.cursor()

    # We parse the file containing the author's papers and generate the citations file
    print("Parsing author's papers file")
    author_papers, _ = data_retrieval.get_author_papers(author_name, author_slug, author_papers_file)
    citations = data_retrieval.generate_citations(author_papers)

    # We retrieve the cited papers from the SQL database
    print("Retrieving cited papers from SQL database")
    cited = citations['cited'].unique()
    cited_papers = data_retrieval.get_cited_papers(cited, c)

    # We parse the file containing the irrelevant papers, then we retrieve the cited papers from the db
    print("Parsing irrelevant papers file")
    bad_papers = data_retrieval.get_irrelevant_papers(bad_papers_file)
    print("Retrieving irrelevant papers citations from SQL database")
    bad_cited_papers = data_retrieval.get_irrelevant_cited_papers(bad_papers, c)
    bad_cited_papers = [{'index': p[0], 'title': p[1], 'abstract': p[2]} for p in bad_cited_papers]

    # We reformat everything as tuples
    author_papers = tuple([(p['index'], p['title'], p['abstract']) for p in author_papers])
    cites = tuple([(c[0], c[1]) for c in citations.as_matrix()])
    bad_papers = tuple([(p['index'], p['title'], p['abstract']) for p in (bad_papers + bad_cited_papers)])

    print("")

    # Generate global vocabulary
    print("Generating vocabulary")
    author_vocab = dataset_tools.generate_vocab(author_papers)
    global_vocab = dataset_tools.generate_vocab(bad_papers)
    tokens = list(set(author_vocab + global_vocab))

    print("")

    print("Preparing dataset...")
    papers_feat, citations_feat, bad_feat, ngrams = dataset_tools.prepare_dataset(author_papers, cites, cited_papers, tokens, bad_papers)

    print("Building computable dataset...")
    inputs = dataset_tools.build_dataset(papers_feat, citations_feat, bad_feat, num_entries)

    print("")

    print("Saving dataset to file: " + output_file + ".npz")
    dataset_tools.dataset_to_file(inputs, ngrams, output_file)

    print("Done.")


if __name__ == '__main__':
    # Parse command
    usage = "usage: %prog [options] args"

    parser = ArgumentParser()

    parser.add_argument("-n", "--name", type=str, help="Name of the author")
    parser.add_argument("-s", "--slug", type=str, help="Short ASCII string for the author's name")
    parser.add_argument("-af", "--author_file", type=str, help="Path to the author's papers file")
    parser.add_argument("-bf", "--irrelevant_file", type=str, help="Path to the irrelevant papers file")
    parser.add_argument("-c", "--num_compare", type=int, help="The number of irrelevant papers to compare the user's papers with, in the DSSM structure")
    parser.add_argument("-d", "--db_name", type=str, help="Name of the SQL database")
    parser.add_argument("-o", "--output_filename", type=str, help="Path to the output dataset file")

    args = parser.parse_args()

    kwargs = {'author_name': args.name,
              'author_slug': args.slug,
              'author_papers_file': args.author_file,
              'bad_papers_file': args.irrelevant_file,
              'num_entries': args.num_compare+2,
              'db_name': args.db_name,
              'output_file': args.output_filename}

    main(**kwargs)
