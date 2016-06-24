import os, re, time
import pandas as pd

import MySQLdb


def list2paper(l_paper, r_index=None, r_author=None, r_title=None, r_abstract=None, r_cite=None):
    """Transform a raw data paper (formatted as a list) into a dict.

    This function uses regular expression to match title, abstract, authors, etc. in each element of the list given in input.
    If a regex is None, then a default regex is used.

    Args:
        l_paper (list of strings): the list of elements forming the paper (title, authors, etc.), in raw format
        r_index (:class:`_sre.SRE_pattern`): a compiled regex to match an index string
        r_author (:class:`_sre.SRE_pattern`): a compiled regex to match an authors list
        r_title (:class:`_sre.SRE_pattern`): a compiled regex to match a title
        r_abstract (:class:`_sre.SRE_pattern`): a compiled regex to match an abstract
        r_cite (:class:`_sre.SRE_pattern`): a compiled regex to match a citation

    Returns:
        dict: the paper as a dict, with list of authors and list of citations
    """
    p = {'index': None, 'authors': [], 'title': None, 'abstract': None, 'citations': []}

    if r_index is None:
        r_index = re.compile('^#index(.*)')
    if r_author is None:
        r_author = re.compile('^#@(.*)')
    if r_title is None:
        r_title = re.compile('^#\*(.*)')
    if r_abstract is None:
        r_abstract = re.compile('^#!(.*)')
    if r_cite is None:
        r_cite = re.compile('^#%(.*)')

    for s in l_paper:
        m_index = r_index.match(s)
        if m_index is not None:
            p['index'] = m_index.group(1)

        m_author = r_author.match(s)
        if m_author is not None:
            p['authors'] = [a.strip() for a in m_author.group(1).split(',')]

        m_title = r_title.match(s)
        if m_title is not None:
            p['title'] = m_title.group(1)

        m_abstract = r_abstract.match(s)
        if m_abstract is not None:
            p['abstract'] = m_abstract.group(1)

        m_cite = r_cite.match(s)
        if m_cite is not None:
            p['citations'].append(m_cite.group(1))

    return p


def get_author_papers(author_name, author_slug, input_file):
    """Returns the list of papers written by the author (list of dicts) from a raw text file.

    The text file must be formatted in the following way:

    * each paper is a block of lines;
    * each line represents either the index, the title, the abstract, the list of authors or a citation reference;
    * there is a way to recognise the type of the line with a regular expression;
    * the papers are separated by a blank line.

    Args:
        author_name (string): the real name of the user
        author_slug (string): a short and ASCII string to replace the author's name
        input_file (string): the name of the file in which are stored the author's papers

    Returns:
        tuple: the author's papers as dictionaries: those with abstract and those without abstract
    """
    # Split result into a list of lists (each sublist is a paper)
    papers = []
    with open(input_file, 'r') as f:
        content = f.readlines()
        p = []
        for l in content:
            if l.strip() != '':
                p.append(l)
            else:
                papers.append(p)
                p = []

    papers = [list2paper(l) for l in papers]

    author_papers = []
    papers_without_abstract = []

    for p in papers:
        if author_name in p['authors']:
            if p['abstract'] is not None:
                author_papers.append(p)
            else:
                papers_without_abstract.append(p)

    return author_papers, papers_without_abstract


def generate_citations(author_papers):
    """Returns the citation relations.

    Args:
        author_papers (list of dicts): the author's papers, as a list of dicts produced by the function :func:`recom.data_retrieval.list2paper`

    Returns:
        :class:`pandas.DataFrame`: the citation relations
    """
    citations = []
    for p in author_papers:
        for c in p['citations']:
            citations.append([p['index'], c])

    return pd.DataFrame(citations)


def get_cited_papers(cited, db_cursor, papers_table='papers'):
    """Retrieves the cited papers data from a SQL database.

    The table ``papers_table`` must have the columns: ``id``, ``title`` and ``abstract``.

    Args:
        cited (list of strings): list of the cited papers' ids
        db_cursor (:class:`MySQLdb.cursors.Cursor`): cursor of a SQL database in which there is a papers table
        papers_table (string): name of the papers table in the SQL database

    Returns:
        tuple of tuples: the results of the SQL query
    """
    # Select papers authored by user
    db_cursor.execute("SELECT id, title, abstract FROM papers p WHERE p.abstract != '' AND p.id IN (" + ','.join(["%s"] * len(cited)) + ")", tuple(cited))
    return db_cursor.fetchall()


def get_irrelevant_papers(input_file):
    """Return the list of irrelevant papers written (list of dicts) from a raw text file.

    Args:
        input_file (string): relative path to the raw text file

    Returns:
        list of dicts: the list of irrelevant papers (with abstract) formatted as dicts
    """
    # Split result into a list of lists (each sublist is a paper)
    papers = []
    with open(input_file, 'r') as f:
        content = f.readlines()
        p = []
        for l in content:
            if l.strip() != '':
                p.append(l)
            else:
                papers.append(p)
                p = []

    papers = [list2paper(l) for l in papers]

    papers_with_abstract = []

    for p in papers:
        if p['abstract'] is not None:
            papers_with_abstract.append(p)

    return papers_with_abstract


def get_irrelevant_cited_papers(bad_papers, db_cursor, papers_table='papers'):
    """Retrieves the papers cited by the irrelevant papers given in input, from a SQL database.

    Args:
        bad_papers (list of dicts): the list of irrelevant papers, formatted as the output of :func:`recom.data_retrieval.list2paper`
        db_cursor (:class:`MySQLdb.cursors.Cursor`): cursor of a SQL database in which there is a papers table
        papers_table (string): name of the papers table in the SQL database

    Returns:
        tuple of tuples: the results of the SQL query
    """
    citations = []
    for p in bad_papers:
        for c in p['citations']:
            citations.append([p['index'], c])

    citations_df = pd.DataFrame(citations, columns=['citing', 'cited'])
    cited = citations_df['cited'].unique()

    db_cursor.execute("SELECT id, title, abstract FROM papers p WHERE p.abstract != '' AND p.id IN (" + ','.join(["%s"] * len(cited)) + ")", tuple(cited))

    return db_cursor.fetchall()
