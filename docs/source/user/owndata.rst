Using your Own Data
===================

In this example, we will use our own data to train a model. We will also
demonstrate on how we could get the predictions given your own data.

My First Data Prediction
------------------------

First, we will load the data. In this example, we will use our own data
consisting of 3 columns: `user`, `item`, `rating`. The `user` and `item`
columns are the IDs of the users and items respectively. The `rating`
column is the rating given by the user to the item.

=====  =====  =======
User   Item   Rating
=====  =====  =======
1      1      5
1      2      1
2      2      3
2      3      3
3      4      3
3      5      5
4      1      5
=====  =====  =======

1. Defining data structure
^^^^^^^^^^^^^^^^^^^^^^^^^^

The data structure is a list of tuples. Each tuple represents a row in the
data. The first element of the tuple is the user ID, the second element is
the item ID, and the third element is the rating.

First, create a file called ``my_data.py`` and add the following code:

.. code-block:: python

    import cornac

    # Define the data as a list of UIR (user, item, rating) tuples
    data = [
        (1, 1, 5),
        (1, 2, 1),
        (2, 2, 3),
        (2, 3, 3),
        (3, 4, 3),
        (3, 5, 5),
        (4, 1, 5)
    ]

2. Creating Dataset object
^^^^^^^^^^^^^^^^^^^^^^^^^^

Next, we will create a dataset object from the data. The dataset object
will be used to train the model.

.. code-block:: python

    from cornac.data import Dataset

    # Load the data into a dataset object
    dataset = cornac.data.Dataset.from_uir(data)

3. Create and train model
^^^^^^^^^^^^^^^^^^^^^^^^^

We will then create a model object and train it using the dataset object.

.. code-block:: python

    from cornac.models import PMF

    # Instantiate the PMF model
    model_pmf = PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123)

    # Use the fit() function to train the model
    model_pmf.fit(dataset)


4. Getting recommendations
^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, we will use the model to get item recommendations for the users. The ``recommend()`` 
function will recommend the top rated items from the model for the user provided.

For example, we will get all recommendations for user 4.

.. code-block:: python

    # Get recommendations for user id 4
    user_id = 4
    recommended_items = model_pmf.recommend(user_id)

    # Print the recommended items
    print(recommended_items)

.. code-block:: bash
    :caption: output

    [1, 4, 3, 5, 2]

The output is a list of item IDs. The first item in the list is the most
recommended item for the user, followed by the second item, and so on.

.. dropdown:: View codes at this point

  .. code-block:: python
    :caption: my_data.py
    :linenos:

    import cornac
    from cornac.models import PMF
    from cornac.data import Dataset

    # Define the data as a list of UIR (user, item, rating) tuples
    data = [
        (1, 1, 5),
        (1, 2, 1),
        (2, 2, 3),
        (2, 3, 3),
        (3, 4, 3),
        (3, 5, 5),
        (4, 1, 5)
    ]

    # Load the data into a dataset object
    dataset = Dataset.from_uir(data)

    # Instantiate the PMF model
    model_pmf = PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123)

    # Use the fit() function to train the model
    model_pmf.fit(dataset)

    # Get recommendations for user id 10
    user_id = 4
    recommended_items = model_pmf.recommend(user_id)

    # Print the recommended items
    print(recommended_items)

Loading data from CSV
---------------------

In this example, we will load the data from a CSV file. The CSV file
consists of 3 columns: `user`, `item`, `rating`. The `user` and `item`
columns are the IDs of the users and items respectively. The `rating`
column is the rating given by the user to the item.

=====  =====  =======
User   Item   Rating
=====  =====  =======
1      1      5
1      2      1
2      2      3
2      3      3
3      4      3
3      5      5
4      1      5
=====  =====  =======

1. Loading the data
^^^^^^^^^^^^^^^^^^^

First, create a file called ``data.csv`` and add the following code:

.. code-block::
    :caption: data.csv

    1,1,5
    1,2,1
    2,2,3
    2,3,3
    3,4,3
    3,5,5
    4,1,5

In this file, the data is separated by commas. The first column is
the user ID, the second column is the item ID, and the third column is the
rating.

Next, we have to load the data from the CSV file. We will use ``Reader``
provided by Cornac to read our CSV file.

.. code-block:: python

    from cornac.data import Reader

    data = Reader().read('data.csv', sep=',')
    print(data)

1. Creating the Dataset Object

Next, we will create a dataset object from the data. The dataset object
will be used to train the model.

.. code-block:: python

    from cornac.data import Dataset

    # Load the data into a dataset object
    dataset = Dataset.from_uir(data, sep=',', skip_lines=1)

3. Create and train model

We will then create a model object and train it using the dataset object.

.. code-block:: python

    from cornac.models import PMF

    # Instantiate the PMF model
    model_pmf = PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123)

    # Use the fit() function to train the model
    model_pmf.fit(dataset)

4. Getting recommendations

Finally, we will use the model to get item recommendations for the users. The ``recommend()`` 
function will recommend the top rated items from the model for the user provided.

For example, we will get all recommendations for user 4.

.. code-block:: python

    # Get recommendations for user id 4
    user_id = 4
    recommended_items = model_pmf.recommend(user_id)

    # Print the recommended items
    print(recommended_items)

.. code-block:: bash
    :caption: output

    [1, 4, 3, 5, 2]

The output is a list of item IDs. The first item in the list is the most
recommended item for the user, followed by the second item, and so on.

.. dropdown:: View codes at this point

  .. code-block:: python
    :caption: data_csv.py
    :linenos:

    import csv
    import cornac
    from cornac.models import PMF
    from cornac.data import Dataset

    # Load the data from the CSV file
    with open('data.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        print(data)

    # Load the data into a dataset object
    dataset = Dataset.from_uir(data, sep=',', skip_lines=1)

    # Instantiate the PMF model
    model_pmf = PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123)

    # Use the fit() function to train the model
    model_pmf.fit(dataset)

    # Get recommendations for user id 4
    user_id = 4
    recommended_items = model_pmf.recommend(user_id)

    # Print the recommended items
    print(recommended_items)


