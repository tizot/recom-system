***************************************
How to generate a dataset?
***************************************

The script ``generate_dataset.py`` transforms some raw data files
(author's papers, irrelevant papers, cited papers) into a dataset usable by a neural network.
These files *must* be formatted in a good way.


------------------
Text file format
------------------

This script uses two raw input files: the user's papers file (``author-papers.txt``) and the irrelevant papers file (``bad-papers.txt``).
They are formatted in the same way:

* each paper is a represented by several lines;
* the papers are separated by one blank line;
* each line in a block wears one kind of information (title, abstract, citation, ...);
* the type of information can be found with a regular expression.

Here is an example of formatting, taken from [Tang 2008]:

>>> #* --- paperTitle
>>> #@ --- Authors
>>> #t ---- Year
>>> #c  --- publication venue
>>> #index 00---- index id of this paper
>>> #% ---- the id of references of this paper (there are multiple lines, with each indicating a reference)
>>> #! --- Abstract

.. pull-quote::
   Jie Tang, Jing Zhang, Limin Yao, Juanzi Li, Li Zhang, and Zhong Su.
   ArnetMiner: Extraction and Mining of Academic Social Networks.
   In Proceedings of the Fourteenth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (SIGKDD'2008).
   pp.990-998.

--------
Usage
--------
In a console, you can use the following command:

>>> python generate_dataset.py -n "Gabriella Pasi" -s "pasi" -af "./data/pasi-papers.txt" -bf "./data/bad-papers.txt" -c 4 -d "dblp" -o "./data/dataset-pasi"

This will parse the files and request the SQL database in order to build a numeric dataset.
The produced dataset is stored into a file and can be reused later, for exemple in the training part.


-----
API
-----

.. automodule:: generate_dataset
   :members:
