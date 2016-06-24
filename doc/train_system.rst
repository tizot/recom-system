****************************
How to train the system?
****************************

The script ``train_dssm.py`` builds a DSSM and trains it with the provided dataset.

-------------------
How does it work?
-------------------



-------------------
Usage
-------------------
In a console, you can use the following command:

>>> python train_dssm.py -e 100 -n1 300 -n2 300 -no 128 -r 0.1 -i "../data/dataset-pasi.npz" -o "../data/output-pasi"

This will build a DSSM with 300 units in the first and in the second layers, and 128 units in the output layer.
The training will be done over 100 epochs, with a learning rate of 0.1.
The input dataset has been produced by the script ``generate_dataset.py``.

-------------------
API
-------------------

.. automodule:: train_dssm
   :members:
