Cornac for Developers
=====================

Introduction
------------
This document is intended for developers who want to use Cornac in their own
projects and applications.

In this guide, we will cover the following topics:

- Experimenting with Models
- Hyperparameters Tuning
- Data Loading
- Training models
- Obtaining Recommendations
- Saving a trained model
- Loading a trained model

Experimenting with Models
-------------------------

Cornac provides a set of predefined models that can be used to build
recommendation systems. These models are located in the :doc:`/api_ref/models`
module.

Each model has a set of hyperparameters that can be tuned to improve the
performance of the model. View the :doc:`/api_ref/models` documentations for
the parameters available for tuning. 

For example, some hyperparameters of the `BPR` model are as follows:

- ``k`` (int, optional, default: 10)
  - `The dimension of the latent factors.`
- ``learning_rate`` (float, optional, default: 0.001)
  – `The learning rate for SGD.`
- ``lambda_reg`` (float, optional, default: 0.001)
  – `The regularization hyper-parameter.`

As shown in our :doc:`/user/quickstart` guide, you are able to run experiments
of different models at once and compare them with the set of metrics you have
set. Also, you may also want to test different hyperparameters for each model
to find the best combination of hyperparameters for each model.

Hyperparameter Tuning
---------------------

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
`GridSearch` and `RandomSearch`.

+------------------------------------------+---------------------------------------------+
| Grid Search                              | Random Search                               |
+==========================================+=============================================+
| Searches for all possible combintations  | Randomly searches for the hyperparameters   |
| of the hyperparameters                   |                                             |
+------------------------------------------+---------------------------------------------+
| Only accepts discrete values             | Accepts both discrete and continuous values |
+------------------------------------------+---------------------------------------------+

For the `space` parameter, we have defined the hyperparameters we want to
search for. In this example, we have defined the `k` hyperparameter to be
a set of discrete values (5, 10, or 50). This will mean that the application
would only attempt those set values.

The `learning_rate` hyperparameter is set as continuous values between
0.001 and 0.01. this would mean that the application would attempt values
between 0.001 and 0.01.

We have also set the `n_trails` parameter for the `RandomSearch` method to
1.  This would mean that the application would attempt 20 random combinations.


Running the Experiment
^^^^^^^^^^^^^^^^^^^^^^

After defining the hyperparameter search methods, we can then run the
experiments using the `cornac.Experiment` class.

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

                    | Precision@10 | Recall@10 | Train (s) | Test (s)
    --------------- + ------------ + --------- + --------- + --------
    BPR-K5-LR0.001  |       0.1118 |    0.1209 |    5.4062 |   0.6711
    BPR-K10-LR0.001 |       0.1110 |    0.1195 |    4.9041 |   0.7394
    BPR-K50-LR0.001 |       0.1117 |    0.1197 |    7.1869 |   0.8457
    BPR-K5-LR0.01   |       0.1710 |    0.1815 |    4.6738 |   0.8544
    BPR-K10-LR0.01  |       0.1718 |    0.1931 |    6.0954 |   0.7300
    BPR-K50-LR0.01  |       0.1630 |    0.1867 |    7.8685 |   0.9358
 
As shown in this example, the best combination of hyperparameters for the BPR
model could be ``k=10`` and ``learning_rate=0.01``, as the results are 
``Precision@10=0.1718`` and ``Recall@10=0.1931``.

However, this may vary from dataset to dataset and you may want to try
different combinations of hyperparameters to find the best combination for your
dataset.


Data Loading
------------

While the earlier examples shows how you can use Cornac's fixed datasets to
do experiments, you may want to use your own datasets for experiments and
recommendations.

To load data into Cornac, it should be in the following format:

.. code-block:: python
    
    # Define the data as a list of UIR (user, item, rating) tuples
    data = [
        ("U1", "I1", 5),
        ("U1", "I2", 1),
        ("U2", "I2", 3),
        ("U2", "I3", 3),
        ("U3", "I4", 3),
        ("U3", "I5", 5),
        ("U4", "I1", 5)
    ]

Then, you could create the ``dataset`` object as follows:

.. code-block:: python

    from cornac.data import Dataset

    # Load the data into a dataset object
    dataset = cornac.data.Dataset.from_uir(data)


Training Models
---------------

After loading the data, you can train the models using the ``fit()`` method.
For this example, we will follow the parameters we have determined in the
earlier example.

.. note::

    Take note that different datasets could have different optimal
    hyperparameters. Therefore, you may want to try different combinations of
    hyperparameters to find the best combination for your dataset.

To train the BPR model, we can do the following:

.. code-block:: python

    from cornac.models import BPR

    # Instantiate the BPR model
    model = BPR(k=10, max_iter=200, learning_rate=0.01, lambda_reg=0.01, seed=123)

    # Train the model
    model.fit(dataset)

Obtaining Recommendations
-------------------------

Now that we have trained our model, we can obtain recommendations for users
using the ``recommend()`` method. For example, to obtain item recommendations
for user ``U1``, we can do the following:

.. code-block:: python

    # Obtain item recommendations for user U1
    recs = model.recommend(user_id="U1")
    print(r)

The output of the ``recommend()`` method is a list of item IDs containing the
recommended items for the user. For example, the output of the above code
could be as follows:

.. code-block:: bash
    :caption: Output

    ['I2', 'I1', 'I3', 'I4', 'I5']

.. dropdown:: View codes for this example

    .. code-block:: python

        import cornac
        from cornac.models import BPR
        from cornac.data import Dataset

        # Define the data as a list of UIR (user, item, rating) tuples
        data = [
            ("U1", "I1", 5),
            ("U1", "I2", 1),
            ("U2", "I2", 3),
            ("U2", "I3", 3),
            ("U3", "I4", 3),
            ("U3", "I5", 5),
            ("U4", "I1", 5)
        ]

        # Load the data into a dataset object
        dataset = Dataset.from_uir(data)

        # Instantiate the BPR model
        model = BPR(k=10, max_iter=100, learning_rate=0.01, lambda_reg=0.01, seed=123)

        # Use the fit() function to train the model
        model.fit(dataset)

        # Obtain item recommendations for user U1
        recs = model.recommend(user_id="U1")
        print(recs)


Saving a Trained Model
----------------------

To save a trained model, you can use the ``save_dir`` parameter in the experiment.
For example, to save models from the experiment in the previous section,
we can do the following:

.. code-block:: python

    # Save the trained model
    cornac.Experiment(eval_method=rs, models=models, metrics=metrics, user_based=True, save_dir="saved_models").run()

This will save the trained models in the ``saved_models`` folder of where you
executed the python code.

.. code-block:: bash
    :caption: Folder directory
    
    - example.py
    - saved_models
        |- BPR
            |- yyyy-MM-dd HH:mm:ss.SSSSSS.pkl


Loading a Trained Model
-----------------------

To load a trained model, you can use the ``load()`` function.

.. code-block:: python

    # Load the trained model
    model = BPR.load("saved_models/BPR/yyyy-MM-dd HH:mm:ss.SSSSSS.pkl")

After you have loaded the model, you can use the ``recommend()`` method to
obtain recommendations for users.

.. dropdown:: View codes for this example
    
    .. code-block:: python

        import cornac
        from cornac.models import BPR
        from cornac.data import Dataset

        # Define the data as a list of UIR (user, item, rating) tuples
        data = [
            ("U1", "I1", 5),
            ("U1", "I2", 1),
            ("U2", "I2", 3),
            ("U2", "I3", 3),
            ("U3", "I4", 3),
            ("U3", "I5", 5),
            ("U4", "I1", 5)
        ]

        # Load the data into a dataset object
        dataset = Dataset.from_uir(data)

        # Load the BPR model
        model = BPR.load("saved_models/BPR/2023-10-30_16-39-36-318863.pkl")

        # Obtain item recommendations for user U1
        recs = model.recommend(user_id="U1")
        print(recs)

What's Next?
------------

Now that you have learned how to use Cornac for your own projects and
applications, you can now start building your own recommendation systems using
Cornac.

.. topic:: View the FoodRecce example

    View the :doc:`/user/example-foodrecce` for a step by step development for a
    restaurant recommendation application.

.. topic:: View the Models API Reference

    You can also view the :doc:`/api_ref/models` documentation for more
    information about the models and its specific parameters.

Others
------

.. topic:: Are you a data scientist?

  Find out how you can have Cornac as part of your workflow to run your
  experiments, and use Cornac's many models with just a few lines of code.
  View :doc:`/user/iamaresearcher`.

.. topic:: For all the awesome people out there

  No matter who you are, you could also consider contributing to Cornac,
  with our contributors guide.
  View :doc:`/developer/index`.