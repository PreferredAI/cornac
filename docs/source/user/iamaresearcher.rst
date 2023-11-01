Cornac for Researchers
======================

Introduction
------------
This document is intended to provide a quick introduction on how researchers
like you could use Cornac to conduct recommender systems research.

In this guide, we will cover the following topics:

- What can you do with Cornac?
- Create experiments
- Tuning parameters
- Adding your Own Model
- Adding your own metric
- Adding your own dataset
- Development Workflow
- Analyze results

What can you do with Cornac?
-----------------------------

Cornac is a recommender systems framework that provides a wide range of
recommender models, evaluation metrics, and experimental tools.
It is designed to be flexible and extensible, allowing researchers to
easily conduct experiments and compare their models with existing ones.

Cornac is written in Python and is built on top of the popular scientific
computing libraries such as NumPy, SciPy, and scikit-learn.
It is also designed to be compatible with the popular deep learning libraries
such as PyTorch and TensorFlow.

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
In this example, we will use the `BPR` model and tune the `k` and
`learning_rate` hyperparameters. We will follow the :doc:`/user/quickstart`
guide and search for the optimal combination of hyperparameters.

In order to do this, we perform hyperparameter searches on Cornac.

Tuning the quickstart example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given the below block fo code from the :doc:`/user/quickstart` guide,
with some slight changes:

- We have added the validation set in the `RatioSplit` method
- We instantiate the `Recall@100` metric
- For this example, we only tune the BPR model

.. code-block:: python

    import cornac
    from cornac.eval_methods import RatioSplit
    from cornac.models import BPR
    from cornac.metrics import Precision, Recall

    # Load a sample dataset (e.g., MovieLens)
    ml_100k = cornac.datasets.movielens.load_feedback()

    # Split the data into training, validation and testing sets
    rs = RatioSplit(data=ml_100k, test_size=0.1, val_size=0.1, rating_threshold=4.0, seed=123)

    # Instantiate Recall@100 for evaluation
    rec100 = cornac.metrics.Recall(100)

    # Instantiate a matrix factorization model (e.g., BPR)
    bpr = BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123)


We would like to optimize the `k` and `learning_rate` hyperparameters. To do
this, we can use the `cornac.hyperopt` module to perform hyperparameter
searches.

.. code-block:: python

    from cornac.hyperopt import Discrete, Continuous
    from cornac.hyperopt import GridSearch, RandomSearch

    # Grid Search
    gs_bpr = GridSearch(
        model=bpr,
        space=[
            Discrete(name="k", values=[5, 10, 50]),
            Discrete(name="learning_rate", values=[0.001, 0.05, 0.01, 0.1])
        ],
        metric=rec100,
        eval_method=rs,
    )

    # Random Search
    rs_bpr = RandomSearch(
        model=bpr,
        space=[
            Discrete(name="k", values=[5, 10, 50]),
            Continuous(name="learning_rate", low=0.001, high=0.01)
        ],
        metric=rec100,
        eval_method=rs,
        n_trails=20,
    )

As shown in the above code, we have defined two hyperparameter search methods,
``GridSearch`` and ``RandomSearch``.

+------------------------------------------+---------------------------------------------+
| Grid Search                              | Random Search                               |
+==========================================+=============================================+
| Searches for all possible combintations  | Randomly searches for the hyperparameters   |
| of the hyperparameters                   |                                             |
+------------------------------------------+---------------------------------------------+
| Only accepts discrete values             | Accepts both discrete and continuous values |
+------------------------------------------+---------------------------------------------+

For the ``space`` parameter, we have defined the hyperparameters we want to
tune:

- We have defined the ``k`` hyperparameter to be a set of discrete values
  (5, 10, or 50). This will mean that the application would only attempt
  to tune with those set values.

- The ``learning_rate`` hyperparameter is set as continuous values between
  0.001 and 0.01. this would mean that the application would attempt any
  values in between 0.001 and 0.01.

For the ``RandomSearch`` method, we have also set the ``n_trails`` parameter to
``20``. This would mean that the application would attempt 20 random
combinations.


Running the Experiment
^^^^^^^^^^^^^^^^^^^^^^

After defining the hyperparameter search methods, we can then run the
experiments using the ``cornac.Experiment`` class.

.. code-block:: python

    # Define the experiment
    cornac.Experiment(
        eval_method=rs,
        models=[gs_bpr, rs_bpr],
        metrics=[rec100],
        user_based=False,
    ).run()

    # Obtain the best params
    print(gs_bpr.best_params)
    print(rs_bpr.best_params)

.. dropdown:: View codes for this example

    .. code-block:: python

        import cornac
        from cornac.eval_methods import RatioSplit
        from cornac.models import BPR
        from cornac.metrics import Precision, Recall
        from cornac.hyperopt import Discrete, Continuous
        from cornac.hyperopt import GridSearch, RandomSearch

        # Load a sample dataset (e.g., MovieLens)
        ml_100k = cornac.datasets.movielens.load_feedback()

        # Split the data into training and testing sets
        rs = RatioSplit(data=ml_100k, test_size=0.2, rating_threshold=4.0, seed=123)

        # Instantiate Recall@100 for evaluation
        rec100 = cornac.metrics.Recall(100)

        # Instantiate a matrix factorization model (e.g., BPR)
        bpr = BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123)

        # Grid Search
        gs_bpr = GridSearch(
            model=bpr,
            space=[
                Discrete(name="k", values=[5, 10, 50]),
                Discrete(name="learning_rate", values=[0.001, 0.05, 0.01, 0.1])
            ],
            metric=rec100,
            eval_method=rs,
        )

        # Random Search
        rs_bpr = RandomSearch(
            model=bpr,
            space=[
                Discrete(name="k", values=[5, 10, 50]),
                Continuous(name="learning_rate", low=0.001, high=0.01)
            ],
            metric=rec100,
            eval_method=rs,
            n_trails=20,
        )

        # Define the experiment
        cornac.Experiment(
            eval_method=rs,
            models=[gs_bpr, rs_bpr],
            metrics=[rec100],
            user_based=False,
        ).run()

        # Obtain the best params
        print(gs_bpr.best_params)
        print(rs_bpr.best_params)


The output of the above code could be as follows:

.. code-block:: bash
    :caption: Output

    TEST:
    ...
                    | Recall@100 | Train (s) | Test (s)
    ---------------- + ---------- + --------- + --------
    GridSearch_BPR   |     0.6953 |   77.9370 |   0.9526
    RandomSearch_BPR |     0.6988 |  147.0348 |   0.7502

    {'k': 50, 'learning_rate': 0.01}
    {'k': 50, 'learning_rate': 0.007993039950008024}

As shown in the output, the ``RandomSearch`` method has found the best
combination of hyperparameters to be ``k=50`` and ``learning_rate=0.0079``
with a Recall@100 score of 0.6988.

However, as it utilizes contains a continouous hyperparameter, the
``RandomSearch`` method may not always find the best combination of
hyperparameters. This is also the reason why we have set the ``n_trails``
parameter to 20 to increase the chances of finding the best combination of
hyperparameters.

Results may vary from dataset to dataset. Try tuning your hyperparameters
using different configurations to find the best hyperparameters for your
dataset.


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

In the `fit` function, you need to implement the training procedure of your model.
In the `score` function, you need to implement the scoring function of your model.
The `score` function will be used to compute the predicted scores of the model
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






