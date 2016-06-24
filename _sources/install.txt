***************************************
Installation
***************************************

The project is developed in python 3. We used specifically the version 3.4.

----------------
Requirements
----------------

We recommend to use a virtualenv to keep projects separated on your machine.
Obviously, this is not mandatory.
To install the project, firstly clone the repository from Github, then install python dependencies.

>>> git clone https://github.com/tizot/recommendation-system.git recom
>>> cd recom
>>> virtualenv --python=python3.4 env.
>>> source .env/bin/activate
>>> pip install -r requirements.txt

If you do not want to use a virtualenv, install the following python packages:

* numpy 1.11.0
* scipy 0.17.1
* pandas 0.18.1
* matplotlib 1.5.1
* mysqlclient 1.3.7
* scikit-learn 0.17.1
* Theano: ``pip install --user https://github.com/Theano/Theano/archive/master.zip``
* Lasagne: ``pip install --user https://github.com/Lasagne/Lasagne/archive/master.zip``


------------------
SQL database
------------------

In order to use the scripts, you need a SQL database in which are stored all the papers.
You can name it as you like, the default name in the script is ``dblp``.

In this database, you must have a table called ``papers`` with at least three columns:

* ``id``: a unique identifier for each paper (INT or UUID);
* ``title``: the title of the paper (VARCHAR(255));
* ``abstract``: the abstract of the paper, that can be empty (TEXT).
