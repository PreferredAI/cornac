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

After training our model, we now can provide recommendations.
Let's get recommendations for the user id ``10`` in the dataset.

The ``recommend()`` function returns a list of recommended items for the
user. The list is sorted in descending order of the predicted ratings.

.. code-block:: python

    # Get recommendations for user id 10
    user_id = 10
    recommended_items = model_pmf.recommend(user_id)

    # Print the recommended items
    print(len(recommended_items))
    print(recommended_items)
  
.. code-block:: bash
  :caption: output

  1656
  ['251', '169', '318', ... ,'669', '424'] # shortened for brevity

We can also recommend a specific number of items for the user by adding the ``k=5`` parameter.

.. code-block:: python

  # Get top 5 recommended items
  recommended_k5 = model_pmf.recommend(user_id, k=5)
  print(f"User 10, Top 5 item recommendations: {recommended_k5}")

.. code-block:: bash
  :caption: output

  User 10, Top 5 item recommendations: ['251', '169', '318', '408', '64']

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

    # Get recommendations for user id 10
    user_id = "10"
    recommended_items = model_pmf.recommend(user_id)

    # Print the recommended items
    print(len(recommended_items))
    print(recommended_items)

    # Get top 5 recommended items
    recommended_k5 = model_pmf.recommend(user_id, k=5)
    print(f"User 10, Top 5 item recommendations: {recommended_k5}")

What's Next?
------------

.. topic:: Adding your own Data

  Explore how you can use your own data to make predictions.
  View :doc:`owndata`.

---------------------------------------------------------------------------

.. topic:: Are you a developer?

  Find out how you can use Cornac as a recommender system to many diferrent
  applications.
  View :doc:`/user/iamadeveloper`.

.. topic:: Are you a data scientist?

  Find out how you can use Cornac to run experiments and tweak parameters
  easily to compare against baselines already on Cornac.
  View :doc:`/user/iamaresearcher`.

.. topic:: For all the awesome people out there

  No matter who you are, you could also consider contributing to Cornac,
  with our contributors guide.
  View :doc:`/developer/index`.
