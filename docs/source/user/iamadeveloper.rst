Introduction to Cornac for Developers
=====================================

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

Hyperparameter Tuning Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, we will use the `BPR` model and tune the `k` and
`learning_rate` hyperparameters. We will follow the :doc:`/user/quickstart`
guide and add additional variants of parameter combinations as follows:

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
execeuted the python code.

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