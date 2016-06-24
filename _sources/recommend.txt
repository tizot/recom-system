=======================================
How to recommend an unseen paper?
=======================================

The script ``recommend.py`` takes in input a trained DSSM and a stream of unseen papers
and determines for each of them whether it should be recommended or not.


------
Usage
------

In a console, you can use the following command:

>>> python recommend.py -d "../data/dataset-pasi.npz" -n "Gabriella Pasi" -s "pasi" -af "../data/pasi-papers.txt" -v "../data/dssm-pasi.npz" -u "../data/unseen-papers.txt"


------
API
------

.. automodule:: recommend
   :members:
