.. Personalised Scientific Papers Recommendation System documentation master file, created by
   sphinx-quickstart on Thu Jun 23 11:46:59 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Personalised Scientific Papers Recommendation System's documentation!
================================================================================

This system aims to recommend scientific papers to a researcher, according to her field of study and her interests.
We discover the author's interests with the papers that she has written, and the papers that she has cited in her publications.

We provide three scripts to do that:

* ``generate_dataset.py`` constructs a dataset tailored for a given user, in order to train the system;
* ``train_dssm.py`` builds a neural network (more specifically, a DSSM) and train it with the dataset built with the previous script;
* ``recommend.py`` takes an unseen paper in input and decides whether this paper should be recommended or not.

.. todo::
   It seems interesting to use only a SQL database, but that requires to correctly identify authors (so, we need a *good* dataset).

   It may also be good to adapt the API and the scripts to use only CSV or JSON files, which are naturally structured and do not require parsing.

-------------
Contents
-------------

.. toctree::
   :maxdepth: 2

   install
   generate_dataset
   train_system
   recommend

   api


-------------------
Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
