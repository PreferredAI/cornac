For Researchers
===============

This document is intended to provide a quick introduction on how researchers
like you could use Cornac to conduct recommender systems research.

In this guide, we will cover the following topics:

- What can you do with Cornac?
- Conducting experiments
- Tuning hyper-parameters
- Adding your own model
- Adding your own metric
- Adding your own dataset
- Development workflow
- Using additional packages

What can you do with Cornac?
----------------------------

Cornac is a recommender systems framework that provides a wide range of
recommender models, evaluation metrics, and experimental tools.
It is designed to be flexible and extensible, allowing researchers to
easily conduct experiments and compare their models with existing ones.

Cornac is written in Python and is built on top of the popular scientific
computing libraries such as NumPy and SciPy. It is also designed to be compatible 
with the popular deep learning libraries such as PyTorch and TensorFlow.

View the models, datasets, metrics that are currently built into Cornac:

- :doc:`/api_ref/models`
- :doc:`/api_ref/datasets`
- :doc:`/api_ref/metrics`

Conducting experiments
----------------------

Cornac provides a set of tools to help you conduct experiments. The main tool is
the :class:`~cornac.experiment.Experiment` class. As its name would suggests, this is
where you manage an experiment with a method to split dataset (e.g., by ratio, k-fold 
cross-validation), a set of models to be compared with, and different evaluation metrics.

Recap from the :doc:`/user/quickstart` example, the following code snippet shows
how to create an experiment with the MovieLens 100K dataset, split it into 
training and testing sets, and run the experiment with the BPR recommender model
and the Precision, Recall evaluation metrics.

.. code-block:: python

    import cornac
    from cornac.eval_methods import RatioSplit
    from cornac.models import BPR, PMF
    from cornac.metrics import Precision, Recall

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
    metrics = [Precision(k=10), Recall(k=10)]

    # put it together in an experiment, voilà!
    cornac.Experiment(eval_method=rs, models=models, metrics=metrics, user_based=True).run()

Tuning hyper-parameters
-----------------------
In this example, we will tune the number of factors `k` and the `learning_rate` of the 
`BPR` model. Given the below block fo code from the :doc:`/user/quickstart` guide,
with some slight changes:

- We have added the validation set in the `RatioSplit` method
- We instantiate the `Recall@100` metric used to track model performance

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

We would like to optimize the `k` and `learning_rate` hyper-parameters. To do
this, we can use the `cornac.hyperopt` module to perform the searches. Cornac supports two methods
for hyper-parameter search, ``GridSearch`` and ``RandomSearch``.

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

After defining the hyper-parameter search methods, we can then run the
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
        rs = RatioSplit(data=ml_100k, test_size=0.1, val_size=0.1, rating_threshold=4.0, seed=123)

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

.. topic:: View related tutorial on Github

  View the Hyperparameter Search guide on Github:
  https://github.com/PreferredAI/cornac/blob/master/tutorials/param_search_vaecf.ipynb


Adding your own model
---------------------

The Contributors Guide shows how to add a model to Cornac, see
:doc:`/developer/add_model`.

Analyze results
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

Adding a new metric
-------------------

The Contributors Guide shows how to add a metric, see :doc:`/developer/add_metric`.

Let us know!
------------
We hope you find Cornac useful for your research. Please share with us on how
you find Cornac useful, and feel free to reach out to us if you have any
questions or suggestions. If you do use Cornac in your research, we appreciate
your citation to our papers_.

.. _papers: https://github.com/PreferredAI/cornac#citation

What's Next?
------------

.. topic:: If you have already developed your model...

  Why not contribute to Cornac by including your model as part of the package?
  View :doc:`/developer/index`.

.. topic:: Keen in developing apps with Cornac?

  View a quickstart guide on how you can code and implement Cornac onto your
  application to provide recommendations for your users.

  View :doc:`/user/iamadeveloper`.






