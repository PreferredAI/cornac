Introduction to Cornac for Researchers
======================================

Introduction
------------
This document is intended to provide a quick introduction on how researchers like
you could use Cornac to conduct recommender systems research.

In this guide, we will cover the following topics:

- What can you do with Cornac?
- Create experiments
- Tuning parameters
- Adding your Own Model
- Development Workflow
- Analyze results

What can you do with Cornac?
-----------------------------

Cornac is a recommender systems framework that provides a wide range of recommender
models, evaluation metrics, and experimental tools. It is designed to be flexible
and extensible, allowing researchers to easily conduct experiments and compare
their models with existing ones.

Cornac is written in Python and is built on top of the popular scientific computing
libraries such as NumPy, SciPy, and scikit-learn. It is also designed to be
compatible with the popular deep learning library TensorFlow.

View the models, datasets, metrics that are currently built into Cornac:

- :doc:`/api_ref/models`
- :doc:`/api_ref/datasets`
- :doc:`/api_ref/metrics`

Create experiments
------------------

Cornac provides a set of tools to help you create experiments. The main tool is
the :class:`~cornac.experiment.Experiment` class. It is a wrapper around the
:class:`~cornac.data.Dataset` class that allows you to easily split the dataset
into training and testing sets, and to run experiments with different models and
evaluation metrics.

Recap from the :doc:`/user/quickstart` example, the following code snippet shows
how to create an experiment with the MovieLens 100K dataset, split it into 
training and testing sets, and run the experiment with the BPR recommender model
and the Precision, Recall evaluation metrics.

.. code-block:: python

    import cornac
    from cornac.eval_methods import RatioSplit
    from cornac.models import BPR, PMF
    from cornac.metrics import RMSE, Precision, Recall

    # Load a sample dataset (e.g., MovieLens)
    ml_100k = cornac.datasets.movielens.load_feedback()

    # Split the data into training and testing sets
    rs = RatioSplit(data=ml_100k, test_size=0.2, rating_threshold=4.0, seed=123)
    
    # Instantiate a matrix factorization model (e.g., BPR)
    models = [
        BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123),
        PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123),  
    ]

    # Define metrics to evaluate the models
    metrics = [RMSE(), Precision(k=10), Recall(k=10)]

    # put it together in an experiment, voilà!
    cornac.Experiment(eval_method=rs, models=models, metrics=metrics, user_based=True).run()

Tuning parameters
-----------------
Under each model, there are a number of parameters that you can tune to improve
the performance of the model.

In this example, we will use the `BPR` model and tune the `k` and `learning_rate`
hyperparameters. We will add additional variants of parameter
combinations as follows:

=====  ==============
K       Learning Rate
=====  ==============
5       0.001
10      0.001
50      0.001
5       0.01
10      0.01
50      0.01 
=====  ==============

Sample codes
^^^^^^^^^^^^

.. code-block:: python

    import cornac
    from cornac.eval_methods import RatioSplit
    from cornac.models import BPR
    from cornac.metrics import Precision, Recall

    # Load a sample dataset (e.g., MovieLens)
    ml_100k = cornac.datasets.movielens.load_feedback()

    # Split the data into training and testing sets
    rs = RatioSplit(data=ml_100k, test_size=0.2, rating_threshold=4.0, seed=123)

    # Instantiate a matrix factorization model (e.g., BPR)
    models = [
        BPR(name="BPR-K5-LR0.001", k=5, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123),
        BPR(name="BPR-K10-LR0.001", k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123),
        BPR(name="BPR-K50-LR0.001", k=50, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123),
        BPR(name="BPR-K5-LR0.01", k=5, max_iter=200, learning_rate=0.01, lambda_reg=0.01, seed=123),
        BPR(name="BPR-K10-LR0.01", k=10, max_iter=200, learning_rate=0.01, lambda_reg=0.01, seed=123),
        BPR(name="BPR-K50-LR0.01", k=50, max_iter=200, learning_rate=0.01, lambda_reg=0.01, seed=123),
    ]

    # Define metrics to evaluate the models
    metrics = [Precision(k=10), Recall(k=10)]

    # put it together in an experiment, voilà!
    cornac.Experiment(eval_method=rs, models=models, metrics=metrics, user_based=True).run()

In this example, we have defined 6 variants of the BPR model with different
hyperparameters. We then evaluate the performance of each model using the
`Precision@10` and `Recall@10` metrics. The results of the experiment will be
displayed in the console as follows:

.. code-block:: bash
    :caption: Output

                    | Precision@10 | Recall@10 | Train (s) | Test (s)
    --------------- + ------------ + --------- + --------- + --------
    BPR-K5-LR0.001  |       0.1118 |    0.1209 |    5.4062 |   0.6711
    BPR-K10-LR0.001 |       0.1110 |    0.1195 |    4.9041 |   0.7394
    BPR-K50-LR0.001 |       0.1117 |    0.1197 |    7.1869 |   0.8457
    BPR-K5-LR0.01   |       0.1710 |    0.1815 |    4.6738 |   0.8544
    BPR-K10-LR0.01  |       0.1718 |    0.1931 |    6.0954 |   0.7300
    BPR-K50-LR0.01  |       0.1630 |    0.1867 |    7.8685 |   0.9358
 
This is how Cornac could easily include multiple variants of the same model,
and have the results shown based on the metrics we have defined. You could easily
define multiple metrics, and have Cornac compute each metric for you. 


Adding your Own Model
---------------------

In order to add your own model, you need to create a class that inherits from
the :class:`~cornac.models.Recommender` class. The class must implement the
following methods:

- :meth:`~cornac.models.Recommender.__init__`
- :meth:`~cornac.models.Recommender.fit`
- :meth:`~cornac.models.Recommender.score`

Let's say we are implementing a new model called `MyModel`. The following code
snippet shows how to implement the `MyModel` class:

.. code-block:: python

    import numpy as np
    import cornac

    class MyModel(cornac.models.Recommender):
        def __init__(self, name="MyModel", k=10, max_iter=100, learning_rate=0.01, lambda_reg=0.01, seed=None):
            super().__init__(name=name, trainable=True, verbose=0)
            self.k = k
            self.max_iter = max_iter
            self.learning_rate = learning_rate
            self.lambda_reg = lambda_reg
            self.seed = seed

        def fit(self, train_set):
            # do something here
            return self

        def score(self, user_idx, item_idx):
            # do something here
            return 0.0

In the `fit` method, you need to implement the training procedure of your model.
In the `score` method, you need to implement the scoring function of your model.
The `score` method will be used to compute the predicted scores of the model
for each user-item pair in the testing set.

In order to test your model, you first have to create an example
(preferably in the examples folder). The example should contain the following
steps:

1. Load a dataset
2. Split the dataset into training and testing sets
3. Instantiate your model
4. Fit and do an experiment with the model

However, to make changes to the model, you need to rebuild Cornac. We will
discuss this in the next section.


Development Workflow
--------------------

Before we move on to the section of building a new model, let's take a look at
the development workflow of Cornac.

The main workflow of developing a new model will be to:

``Create an example, Create model files --> Build --> Run Example``

Whenever a new change is done to your model files, you are required to rebuild
Cornac using the ``setup.py`` script. This will ensure that the changes you have
made to your model files are reflected in the Cornac package.


Rebuilding Cornac
^^^^^^^^^^^^^^^^^

1. To build Cornac on your environment:

.. code-block:: bash

    python3 setup.py install


.. note::

    The following packages are required for building Cornac on your environment: ``Cython``, ``numpy``, ``scipy``.
    
    If you do not have them, install by using the following commands:

    .. code-block:: bash

        pip3 install Cython numpy scipy

2. Run an example utilising your new model.

Analyze Results
---------------
Cornac makes it easy for you to run your model alongside other existing models.
To do so, simply add you model to the list of models in the experiment.

.. code-block:: python

    # Add new model to list
    models = [
        BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123),
        PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123),
        MyModel(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123),  
    ]

    # Define metrics to evaluate the models
    metrics = [RMSE(), Precision(k=10), Recall(k=10)]

    # run the experiment and compare the results
    cornac.Experiment(eval_method=rs, models=models, metrics=metrics, user_based=True).run()

Conclusion
----------
We hope you find Cornac useful for your research. Please share with us on how
you find Cornac useful, and feel free to reach out to us if you have any
questions or suggestions.

What's Next?
------------

.. topic:: If you have already developed your model...

  Why not contribute to Cornac by including your model as part of the package?
  View :doc:`/developer/index`.

.. topic:: Keen in developing apps with Cornac?

  View a quickstart guide on how you can code and implement Cornac onto your
  application to provide recommendations for your users.

  View :doc:`/user/iamadeveloper`.






