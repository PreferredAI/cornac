Quickstart
==========

Cornac is a Python library for building and training recommendation models.
It focuses on making it convenient to work with models leveraging auxiliary
data (e.g., item descriptive text and image, social network, etc).

Cornac enables fast experiments and straightforward implementations of new
models. It is highly compatible with existing machine learning libraries
(e.g., TensorFlow, PyTorch).

.. topic:: New to Recommender Systems?

   If you're new to recommender systems, this link provides a beginner-friendly
   introduction to help you understand the fundamentals and get started:
   https://github.com/PreferredAI/tutorials/tree/master/recommender-systems

The Cornac Experiment Concept
-----------------------------
The main idea behind Cornac is to provide a simple and flexible way to
experiment with different models, datasets and metrics without
having to manually implement and run all the code yourself.

**Here are some key concepts related to Cornac:**

.. grid:: 1 2 2 2
    :gutter: 4

    .. grid-item-card:: 1. Datasets
        :columns: 12 12 6 6
        :padding: 3

        A **dataset** refers to a specific collection of input data that is
        used to train or test an algorithm.

    .. grid-item-card:: 2. Models
        :columns: 12 12 6 6
        :padding: 3

        A **model** refers to a specific (machine learning) algorithm that is used to train on a
        dataset to learn user preferences and make recommendations.

    .. grid-item-card:: 3. Evaluation metrics
        :columns: 12 12 6 6
        :padding: 3

        An **evaluation metric** refers to a specific performance measure or score
        that is being used to evaluate or compare different models during the experimentation process.

    .. grid-item-card:: 4. Experiments
        :columns: 12 12 6 6
        :padding: 3

        An **experiment** is one-stop-shop where you manage how your dataset should be prepared/split, different evaluation metrics, and multiple models to be compared with.


The First Experiment
--------------------
In today's world of countless movies and TV shows at our fingertips,
finding what we truly enjoy can be a challenge.

This experiment focuses on how we could utilize a recommender system to provide
us with personalized recommendations based on our preferences.

.. _movielens-label:

About the MovieLens dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The MovieLens_ dataset, a repository of movie ratings and user preferences,
remains highly relevant today. Oftentimes, it is used as a benchmark to compare 
different recommendation algorithms.

.. _MovieLens: https://grouplens.org/datasets/movielens/

Sample data from MovieLens 100K dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The MovieLens 100K dataset contains 100,000 ratings from 943 users on 1,682
movies. Each user has rated at least 20 movies on a scale of 1 to 5.

The dataset also contains additional information about the movies, such as
genre and year of release.

+-------+-------+-------+-------+
|       |user_id|item_id| rating|
+=======+=======+=======+=======+
|   0   |   196 |   242 |  3.0  |
+-------+-------+-------+-------+
|   1   |   186 |   302 |  3.0  |
+-------+-------+-------+-------+
|   2   |    22 |   377 |  1.0  |
+-------+-------+-------+-------+
|   3   |   244 |    51 |  2.0  |
+-------+-------+-------+-------+
|   4   |   166 |   346 |  1.0  |
+-------+-------+-------+-------+


A sample of 5 records from the MovieLens 100K dataset is shown above.

The Experiment
~~~~~~~~~~~~~~

.. note::

    This tutorial assumes that you have already installed Cornac. If you have
    not done so, please refer to the installation guide in the documentation.

    See :doc:`install`.

In this experiment, we will be using the MovieLens 100K dataset to train and
evaluate a recommender system that can predict how a user would rate a movie
based on their preferences learned from past ratings.

.. image:: images/flow.jpg
   :width: 800

1. Data Loading
^^^^^^^^^^^^^^^

Create a python file called ``first_experiment.py`` and add the following code
into it:

.. code-block:: python

    import cornac

    # Load a sample dataset (e.g., MovieLens)
    ml_100k = cornac.datasets.movielens.load_feedback()

In the above code, we define a variable ``ml_100k`` that loads the
**MovieLens 100K dataset**.

MovieLens is one of the many datasets available on Cornac for use.
View the other datasets available  in :doc:`/api_ref/datasets`.


2. Data Splitting
^^^^^^^^^^^^^^^^^

We need to split the data into training and testing sets. A common way to do
this is to do it based on a specified ratio (e.g., 80% training, 20% testing).

A training set is used to train the model, while a testing set is used to
evaluate the model's performance.

.. code-block:: python

    from cornac.eval_methods import RatioSplit

    # Split the data into training and testing sets
    rs = RatioSplit(data=ml_100k, test_size=0.2, rating_threshold=4.0, seed=123)

In this example, we set various parameters for the ``RatioSplit`` object:

- ``test_size=0.2`` to split the data into **80% training** and
  **20% testing**.

- ``data=ml_100k`` to use the **MovieLens 100K dataset**.

- ``rating_threshold=4.0`` to only consider ratings that are
  greater than or equal to 4.0 to be **positive ratings**. Everything else will
  be considered as something that the user dislikes.

- ``seed=123`` to ensure that the results are **reproducible**. Setting a seed
  to a specific value will always produce the same results.


3. Define Model
^^^^^^^^^^^^^^^

We need to define a model to train and evaluate. In this example, we will be
using the **Bayesian Personalized Ranking (BPR)** model.

.. code-block:: python

    from cornac.models import BPR

    # Instantiate a recommender model (e.g., BPR)
    models = [
        BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123),
    ]

We set various parameters for the ``BPR`` object:

- ``k=10`` to set the number of latent factors to **10**. This means that each
  user and item will be represented by a vector of 10 numbers.
- ``max_iter=200`` to set the maximum number of iterations to **200**. This
  means that the model will be trained for a maximum of 200 iterations.
- ``learning_rate=0.001`` to set the learning rate to **0.001**. This
  controls how much the model will learn from each iteration.
- ``lambda_reg=0.01`` to set the regularization parameter to **0.01**. This
  controls how much the model will penalize large values in the user and item
  vectors.
- ``seed=123`` to ensure that the results are **reproducible**. Setting a seed
  to a specific value will always produce the same results. This is the same
  seed that we used for the ``RatioSplit`` object.

4. Define Metrics
^^^^^^^^^^^^^^^^^
We need to define metrics to evaluate the model. In this example, we will be
using the **Precision**, **Recall** metrics.

.. code-block:: python

    from cornac.metrics import Precision, Recall

    # Define metrics to evaluate the models
    metrics = [Precision(k=10), Recall(k=10)]

We set various metrics for the ``metrics`` object:

- The **Precision** metric measures the proportion of recommended items that
  are relevant to the user. The higher the Precision, the better the model.

- The **Recall** metric measures the proportion of relevant items that are
  recommended to the user. The higher the Recall, the better the model.

.. note::

    Certain metrics like **Precision** and **Recall** are ranking based.
    This requires a specific number of recommendations to be made in order to
    calculate the metric.

    In this example, these calculations will be done based on
    **10 recommendations** for each user. (``k=10``)


5. Run Experiment
^^^^^^^^^^^^^^^^^

We can now run the experiment by putting everything together. This will train
the model and evaluate its performance based on the metrics that we defined.

.. code-block:: python

    # Put it together in an experiment, voilà!
    cornac.Experiment(eval_method=rs, models=models, metrics=metrics, user_based=True).run()

We set various parameters for the ``Experiment`` object:

- ``eval_method=rs`` to use the ``RatioSplit`` object that we defined earlier.

- ``models=models`` to use the ``BPR`` model that we defined earlier.

- ``metrics=metrics`` to use the ``Precision``, and ``Recall``
  metrics that we defined earlier.

- ``user_based=True`` to evaluate the model on an individual user basis.
  This means that the average performance of each user will be calculated
  and averaged across users to get the final result (users are weighted equally). 
  This is opposed to evaluating based on all ratings by setting ``user_based=false``.


.. dropdown:: View codes at this point

    .. code-block:: python
        :caption: first_experiment.py
        :linenos:

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
            BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123),
        ]

        # Define metrics to evaluate the models
        metrics = [Precision(k=10), Recall(k=10)]

        # Put it together in an experiment, voilà!
        cornac.Experiment(eval_method=rs, models=models, metrics=metrics, user_based=True).run()

Run the python codes
^^^^^^^^^^^^^^^^^^^^

Finally, run the python codes you have just written by entering this into your
favourite command prompt.

.. code-block:: bash

    python first_experiment.py


What does the output mean?
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash
  :caption: output

  TEST:
  ...
      | Precision@10 | Recall@10 | Train (s) | Test (s)
  --- + ------------ + --------- + --------- + --------
  BPR |       0.1110 |    0.1195 |    4.7624 |   0.7182


After the training process, Cornac tests the trained model by using the test data
(as split by the ``RatioSplit`` function) to calculate the metrics defined.

Over in the screenshot below, we see the results for the
``Precision@10`` (k=10) and ``Recall@10`` (k=10) respectively.

Also, we see the time taken for Cornac to train, and time taken evaluate the test
data.


Adding More Models
^^^^^^^^^^^^^^^^^^

In many of the times, we may want to consider adding more models so that we can
compare results accordingly.

Let's add a second model called the Probabilistic Matrix Factorization (PMF) model.
We add the following codes to our models variable:

.. code-block:: python

    from cornac.models import BPR, PMF

    # Instantiate a matrix factorization model (e.g., BPR, PMF)
    models = [
        BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123),
        PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123),
    ]

.. dropdown:: View codes at this point

    .. code-block:: python
        :caption: first_experiment.py
        :linenos:

        import cornac
        from cornac.eval_methods import RatioSplit
        from cornac.models import BPR, PMF
        from cornac.metrics import Precision, Recall

        # Load a sample dataset (e.g., MovieLens)
        ml_100k = cornac.datasets.movielens.load_feedback()

        # Split the data into training and testing sets
        rs = RatioSplit(data=ml_100k, test_size=0.2, rating_threshold=4.0, seed=123)

        # Instantiate a matrix factorization model (e.g., BPR, PMF)
        models = [
            BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123),
            PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123),
        ]

        # Define metrics to evaluate the models
        metrics = [Precision(k=10), Recall(k=10)]

        # Put it together in an experiment, voilà!
        cornac.Experiment(eval_method=rs, models=models, metrics=metrics, user_based=True).run()

Now run it again!

.. code-block:: bash

    python first_experiment.py

.. code-block:: bash
  :caption: output

  TEST:
  ...
      | Precision@10 | Recall@10 | Train (s) | Test (s)
  --- + ------------ + --------- + --------- + --------
  BPR |       0.1110 |    0.1195 |    4.7624 |   0.7182
  PMF |       0.0813 |    0.0639 |    2.5635 |   0.4254

We are now presented with results from our different models. In this easy example,
we can see how we can easily compare the results from different models.

Depending on the results of the metrics, time taken for training and evaluation,
we can then further tweak the parameters, and also decide which model to use for
our application.

.. topic:: View example on Github

  View a related example on Github:
  https://github.com/PreferredAI/cornac/blob/master/examples/first_example.py


What's Next?
------------

.. topic:: Are you a developer?

  View a quickstart guide on how you can code and implement Cornac onto your
  application to provide recommendations for your users.

  View :doc:`/user/iamadeveloper`.

.. topic:: Are you a data scientist?

  Find out how you can have Cornac as part of your workflow to run your
  experiments, and use Cornac's many models with just a few lines of code.
  View :doc:`/user/iamaresearcher`.

.. topic:: For all the awesome people out there

  No matter who you are, you could also consider contributing to Cornac,
  with our contributors guide.
  View :doc:`/developer/index`.

