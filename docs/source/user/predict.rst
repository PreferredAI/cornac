Making Recommendations
======================

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
predict the ratings of the user with ID ``"10"``.

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

Similar to the quickstart tutorial, we load the MovieLens 100K dataset.
The ``ml_100k`` variable is a list of ``(user, item, rating)`` tuples.


2. Model Training
^^^^^^^^^^^^^^^^^

We then instantiate the PMF model and train it using the ``fit()`` function.
We train it on the training part of the split, ``rs.train_set``, which is a
:class:`~cornac.data.dataset.Dataset`.

.. code-block:: python

    from cornac.eval_methods import RatioSplit
    from cornac.models import PMF

    # Instantiate the PMF model
    model_pmf = PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123)

    # Follow the same split as in the quickstart tutorial
    rs = RatioSplit(ml_100k, test_size=0.2, seed=123, verbose=True)

    # Use the fit() function to train the model
    model_pmf.fit(rs.train_set)


3. Prediction
^^^^^^^^^^^^^

After training our model, we now can provide recommendations.
Let's get recommendations for the user id ``"10"`` in the dataset.
User and item IDs in MovieLens are strings.

The ``recommend()`` function returns a list of recommended items for the
user. The list is sorted in descending order of the predicted ratings.

.. code-block:: python

    # Get recommendations for user id 10
    user_id = "10"
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

Training on your own data
-------------------------

The same steps work for your own data, loaded as shown in
:doc:`/user/owndata`. To load data into Cornac, it should be in the following
format:

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
    dataset = Dataset.from_uir(data)

After loading the data, you can train a model using ``fit()`` method.
To train a BPR model, we can do the following:

.. code-block:: python

    from cornac.models import BPR

    # Instantiate the BPR model
    model = BPR(k=10, max_iter=100, learning_rate=0.01, lambda_reg=0.01, seed=123)

    # Train the model
    model.fit(dataset)


Obtaining recommendations
-------------------------

Now that we have trained our model, we can obtain recommendations for users
using ``recommend()`` method. For example, to obtain item recommendations
for user ``U1``, we can do the following:

.. code-block:: python

    # Obtain item recommendations for user U1
    recs = model.recommend(user_id="U1", k=5)
    print(recs)

The output of ``recommend()`` method is a list of item IDs containing the
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
        recs = model.recommend(user_id="U1", k=5)
        print(recs)

Removing items already seen
---------------------------

By default, ``recommend()`` ranks every item, including the items the user has
already rated. In the output above, ``U1`` gets back ``I1`` and ``I2``, which
are in the training data. To leave them out, set ``remove_seen=True`` and pass
the dataset the model was trained on as ``train_set``:

.. code-block:: python

    # Obtain item recommendations for user U1, without the items U1 has rated
    recs = model.recommend(user_id="U1", k=5, remove_seen=True, train_set=dataset)
    print(recs)

.. code-block:: bash
    :caption: Output

    ['I3', 'I4', 'I5']

``U1`` has rated ``I1`` and ``I2``, so only three items are left to recommend.
``recommend()`` raises an error if you set ``remove_seen=True`` without a
``train_set``.

What's Next?
------------

:doc:`/user/serving` shows how to save a trained model, load it later without
training it again, and serve it to your application.
