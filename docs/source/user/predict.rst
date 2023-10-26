My First Prediction
===================

In this example, we will use Cornac to predict the ratings of a user. The
data used in this example is the MovieLens 100K dataset, which is included
in the Cornac package.

.. note::

    If you have not done the quickstart tutorial, we highly recommend that you
    visit that page first before continuing with this example.

    See :doc:`quickstart`.

First, we load the Cornac package and the **MovieLens 100K dataset**.
Revisit :ref:`movielens-label` if you are unsure on what the dataset is about.


The Prediction
--------------

We will predict the ratings of a particular user. Let's say we want to
predict the ratings of the first user in our list (``index 0``).

Followed from our quickstart tutorial, we will use the PMF model to predict
the ratings of   a user. We will use the same hyperparameters as in the
quickstart tutorial.

.. image:: images/flow.jpg
   :width: 800

1. Data Loading
^^^^^^^^^^^^^^^

Create a python file called ``first_prediction.py`` and add the following code
into it:

.. code-block:: python

    import cornac

    # Load a sample dataset (e.g., MovieLens)
    ml_100k = cornac.datasets.movielens.load_feedback()

Similar to the previous tutorial, we load the MovieLens 100K dataset.
The ``ml_100k`` variable is a ``cornac.data.Dataset`` object.


2. Model Training
^^^^^^^^^^^^^^^^^

We then instantiate the PMF model and train it using the ``fit()`` function.

.. code-block:: python

    from cornac.eval_methods import RatioSplit
    from cornac.models import PMF

    # Instantiate the PMF model
    model_pmf = PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123)

    # Follow the same split as in the quickstart tutorial
    rs = RatioSplit(ml_100k, test_size=0.2, seed=123, verbose=True)

    # Use the fit() function to train the model
    model_pmf.fit(ml_100k)


3. Prediction
^^^^^^^^^^^^^

We can now use the ``score()`` function to predict the ratings of a user.
The ``score()`` function returns the predicted ratings of a user for all
items, or the predicted rating of a user for a particular item.

The ``score()`` function takes in two parameters:

- ``user_idx``: The index of the user in the dataset.
- ``item_idx``: The index of the item in the dataset.
  (Optional. If not set, will return all item scores)

.. code-block:: python

    # Predicted scores for user index 0, for all items
    all_predicted_scores = model_pmf.score(user_idx=0)
    print(f"user 0, all item scores: {all_predicted_scores}")
    print(type(all_predicted_scores))

    # Predicted score for user index 0, item index 0
    predicted_score = model_pmf.score(user_idx=0, item_idx=0)
    print(f"user 0, item 0 score: {predicted_score}")
    print(type(predicted_score))

.. dropdown:: View codes at this point

    .. code-block:: python
        :caption: first_prediction.py
        :linenos:

        import cornac
        from cornac.eval_methods import RatioSplit
        from cornac.models import PMF

        # Load a sample dataset (e.g., MovieLens)
        ml_100k = cornac.datasets.movielens.load_feedback()

        # Instantiate the PMF model
        model_pmf = PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123)

        # Follow the same split as in the quickstart tutorial
        rs = RatioSplit(ml_100k, test_size=0.2, seed=123, verbose=True)

        # Use the fit() function to train the model
        model_pmf.fit(rs.train_set)

        # Predicted scores for user index 0, for all items
        all_predicted_scores = model_pmf.score(user_idx=0)
        print(f"user 0, all item scores: {all_predicted_scores}")
        print(type(all_predicted_scores))

        # Predicted score for user index 0, item index 0
        predicted_score = model_pmf.score(user_idx=0, item_idx=0)
        print(f"user 0, item 0 score: {predicted_score}")
        print(type(predicted_score))


What do the results mean?
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash
    :caption: output

    user 0, all item scores: [1.77542631e+00 1.53287843e+00 1.33609482e+00 ... 2.24056330e-01
    1.55586646e-01 5.15620921e-04]
    <class 'numpy.ndarray'>
    user 0, item 0 score: 4.420524754279344
    <class 'numpy.float64'>

The ``all_predicted_scores`` variable is a numpy array of size ``(n_items,)``.
This means that it contains the predicted scores for all items for the
specified user. In this case, the user is ``user 0``.

The ``predicted_score`` variable is a ``numpy.float64``. This means
that it contains the predicted score for the specified user and item. In this
case, the score for ``user 0`` and ``item 0`` is ``4.42``.


Using a ranking model
^^^^^^^^^^^^^^^^^^^^^

You can also use a different model to predict the ratings of a user. For
example, you can use the BPR model, which is a ranking model.

The BPR model does not predict the ratings of a user, but instead,
it ranks the items for a user based on the user's preferences.

Import the BPR model and instantiate it:

.. code-block:: python

    from cornac.models import PMF, BPR
    # Instantiate the BPR model
    model_bpr = BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123)

Train the model:

.. code-block:: python

    # Use the fit() function to train the model
    model_bpr.fit(rs.train_set)

Get the ranking predictions:

.. code-block:: python

    # Predict the rankings
    predicted_rank = model_bpr.rank(user_idx=0)
    print(f"user 0, all item rankings: {predicted_rank}")
    print(type(predicted_rank))

.. dropdown:: View codes at this point

    .. code-block:: python
        :caption: first_prediction.py
        :linenos:

        import cornac
        from cornac.eval_methods import RatioSplit
        from cornac.models import PMF, BPR

        # Load a sample dataset (e.g., MovieLens)
        ml_100k = cornac.datasets.movielens.load_feedback()

        # Instantiate the PMF model
        model_pmf = PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123)

        # Follow the same split as in the quickstart tutorial
        rs = RatioSplit(ml_100k, test_size=0.2, seed=123, verbose=True)

        # Use the fit() function to train the model
        model_pmf.fit(rs.train_set)

        # Predicted scores for user index 0, for all items
        all_predicted_scores = model_pmf.score(user_idx=0)
        print(f"user 0, all item scores: {all_predicted_scores}")
        print(type(all_predicted_scores))

        # Predicted score for user index 0, item index 0
        predicted_score = model_pmf.score(user_idx=0, item_idx=0)
        print(f"user 0, item 0 score: {predicted_score}")
        print(type(predicted_score))


        # Instantiate the BPR model
        model_bpr = BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123)

        # use the fit() function to train the model
        model_bpr.fit(rs.train_set)

        # Predict the rankings
        predicted_rank = model_bpr.rank(user_idx=0)
        print(f"user 0, all item rankings: {predicted_rank}")
        print(type(predicted_rank))

Results:

.. code-block:: bash
    :caption: output

    user 0, all item rankings: (array([  25,  253,   54, ..., 1629, 1555, 1302]), array([ 1.5580364,  0.3658653, -0.587296 , ..., -1.4186771, -1.4275337,
       -1.4110744], dtype=float32))
    <class 'tuple'>

The ``predicted_rank`` variable is a tuple of size ``2``.

- The first element of the tuple is a numpy array of size ``n_items``.

  This means that it contains the predicted rankings for all items for the
  specified user. In this case, the user is ``user 0``.

- The second element of the tuple is a numpy array of size ``n_items``.

  This means that it contains the predicted scores for all items for the
  specified user. In this case, the user is ``user 0``.


What's Next?
------------

.. topic:: Add your own Data

  Explore how you can use your own data to make predictions.
  View :doc:`owndata`.

---------------------------------------------------------------------------

.. topic:: Are you a developer?

  Find out how you can use Cornac as a recommender system to many diferrent
  applications.
  View :doc:`applications`.

.. topic:: Are you a data scientist?

  Find out how you can use Cornac to run experiments and tweak parameters
  easily to compare against baselines already on Cornac.
  View :doc:`experiments`.

.. topic:: For all the awesome people out there

  No matter who you are, you could also consider contributing to Cornac,
  with our contributors guide.
  View :doc:`/developer/index`.
