=================
Cornac Quickstart
=================

Introduction
============
Cornac is a Python library for building and training recommendation models.
It focuses on making it convenient to work with models leveraging auxiliary data 
(e.g., item descriptive text and image, social network, etc). 

Cornac enables fast experiments and straightforward implementations of new models. 
It is highly compatible with existing machine learning libraries (e.g., TensorFlow, PyTorch).

Install Python
==============
Cornac supports most versions of Python 3. This includes Python versions 3.8 to 3.11.
If you have not done so, go to the official Python download page at <https://www.python.org/downloads/> to download Python.

Install Cornac
==============
There are 3 different ways in which you could install cornac.
Depending on your environment and requirements, choose and run the corresponding codes to install Cornac:

1. Using pip (recommended)
Pip is a package manager for Python. It allows users to easily install, update, and manage 
third-party libraries and frameworks that are available on the Python Package Index (PyPI).

.. code-block:: bash

    $ pip3 install cornac

2. Using conda
Conda is an open-source package management system and environment management system for installing, 
creating, and managing software environments on Windows, macOS, Linux, and other operating systems.

.. code-block:: bash

    $ conda install cornac -c conda-forge

3. From Github Source - View <advanced section>
Should you require the latest updates of Cornac from Github, you could manually build and install using the following codes:

.. code-block:: bash

    $ pip3 install Cython numpy scipy
    $ git clone https://github.com/PreferredAI/cornac.git
    $ cd cornac
    $ python3 setup.py install

Model dependencies
------------------
Certain models in Cornac may require additional dependencies. 
The requirements.txt file shows what dependencies are required for each model.\

Take the model WMF<add hyperlink> for example.
cornac/cornac/models/wmf/requirements.txt
```
tensorflow==2.12.0
```

In order to utilize this model, this dependency needs to be installed.
To install all dependencies in a provided requirements.txt file, follow these steps:

1. Using your favourite terminal/command prompt, navigate to the models in which you want to utilize
```
cd cornac/models/wmf
```
2. Install the dependencies by using this command:
```
pip install -r requirements.txt
```


Note for MacOS users
--------------------
Some algorithm implementations use OpenMP to support multi-threading.
For MacOS users, in order to run those algorithms efficiently, you might need to install gcc from Homebrew to have an OpenMP compiler:
.. code-block:: bash
    $ brew install gcc | brew link gcc


Checking Cornac
===============
After installing Cornac, you can verify that it has been successfully installed by running the following command:
```
python -c "import cornac; print(cornac.__version__)"
```
This should output the version of Cornac installed. 
If you see a version number shown, congratulations! You now have Cornac and you're now ready to create your first experiment!


Cornac Experiment Concept
=========================
The main idea behind Cornac is to provide a simple and flexible way to experiment with different algorithms, hyperparameters, and datasets without having to manually implement and run all the code yourself.
Here are some key concepts related to Cornac:

<Add chart image>

1. Experiments
--------------
In Cornac, an "experiment" refers to a specific combination of algorithm, hyperparameters, dataset, and evaluation metric that is being tested or evaluated.
Each experiment corresponds to a single row in the database, which contains information about the experiment such as its ID, name, description, and date created/updated.

2. Datasets
-----------
In Cornac, a "dataset" refers to a specific collection of input data that is used to train or test an algorithm.
Each dataset corresponds to a single row in the database, which contains information about the dataset such as its ID, name, description, and date created/updated.

3. Algorithms
-------------
In Cornac, an "algorithm" refers to a specific computational model or technique that is being used to perform some task or function.
Each algorithm corresponds to a single row in the database, which contains information about the algorithm such as its ID, name, description, and date created/updated.

4. Hyperparameters
------------------
In Cornac, a "hyperparameter" refers to a specific parameter or setting that is being adjusted or fine-tuned during the experimentation process.
Each hyperparameter corresponds to a single row in the database, which contains information about the hyperparameter such as its ID, name, description, and date created/updated.

5. Evaluation metrics
---------------------
In Cornac, an "evaluation metric" refers to a specific performance measure or score that is being used to evaluate or compare different algorithms or models during the experimentation process.
Each evaluation metric corresponds to a single row in the database, which contains information about the evaluation metric such as its ID, name, description, and date created/updated.


The First Experiment
====================

Now that understand the concepts, we are ready for our first experiment.

<comment>
Before that, if you are unsure of how recommender systems work, head to the tutorials here <link> first to get a better understanding.
That will allow you to get up to speed.

Movies ... <add scenario>

About the MovieLens dataset
---------------------------


Splitting data
--------------


Training
--------


Testing
-------

What do this results mean?
----------------------------

Prediction
----------


Rounding it all up
------------------
.. code-block:: python
    :caption: python.py

    import cornac
    from cornac.eval_methods import RatioSplit
    from cornac.models import MF, PMF, BPR
    from cornac.metrics import MAE, RMSE, Precision, Recall, NDCG, AUC, MAP

    # load the built-in MovieLens 100K and split the data based on ratio
    ml_100k = cornac.datasets.movielens.load_feedback()
    rs = RatioSplit(data=ml_100k, test_size=0.2, rating_threshold=4.0, seed=123)

    # initialize models, here we are comparing: Biased MF, PMF, and BPR
    models = [
        MF(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02, use_bias=True, seed=123),
        PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123),
        BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123),
    ]

    # define metrics to evaluate the models
    metrics = [MAE(), RMSE(), Precision(k=10), Recall(k=10), NDCG(k=10), AUC(), MAP()]

    # put it together in an experiment, voilà!
    cornac.Experiment(eval_method=rs, models=models, metrics=metrics, user_based=True).run()

Quickstart
==========

- Installation
- Cornac Experiment Concept
- About the MovieLens Dataset
- Splitting data
- Training
- Testing
- Prediction
