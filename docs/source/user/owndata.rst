Loading Data
============

This page shows you how to get data into Cornac, either from one of the
built-in datasets or from your own data. Cornac reads data as a list of
tuples, such as ``(user, item, rating)``, which you pass to an evaluation
method or turn into a dataset object for a model.

Built-in datasets
-----------------

Cornac comes with loaders for many public recommendation datasets, so you can
start an experiment without preparing any files.

Loading a dataset
^^^^^^^^^^^^^^^^^

Each dataset is a module in ``cornac.datasets`` with one or more ``load_*``
functions. For example, this loads the MovieLens 100K ratings:

.. code-block:: python

    from cornac.datasets import movielens

    data = movielens.load_feedback(variant="100K")
    print(len(data))
    print(data[:3])

.. code-block:: bash
    :caption: output

    Data from https://files.grouplens.org/datasets/movielens/ml-100k/u.data
    will be cached into /home/user/.cornac/ml-100k/u.data
    1.99MB [00:18, 129kB/s]
    File cached!
    100000
    [('196', '242', 3.0), ('186', '302', 3.0), ('22', '377', 1.0)]

The first time you load a dataset, Cornac downloads it and keeps a copy; after
that it reads the copy. The result is a list of ``(user, item, rating)``
tuples, with user and item IDs as strings, which you can pass straight to an
evaluation method such as :class:`~cornac.eval_methods.ratio_split.RatioSplit`.

Depending on the dataset, a loader takes a few optional arguments:

- ``fmt`` chooses the tuple format. For MovieLens, ``fmt="UIRT"`` adds a
  timestamp to each tuple, as in ``('196', '242', 3.0, 881250949)``, which you
  need to split data by time.
- ``variant`` chooses the size of datasets that come in several versions:
  ``"100K"``, ``"1M"``, ``"10M"`` or ``"20M"`` for MovieLens, and ``"small"``
  or ``"original"`` for Netflix.
- ``reader`` takes a :class:`~cornac.data.reader.Reader` that filters the
  data while it is read (see below).

:doc:`/api_ref/datasets` lists the functions of every loader.

What is available
^^^^^^^^^^^^^^^^^

Cornac's datasets fall into five groups. The `dataset catalogue
<https://github.com/PreferredAI/cornac/blob/master/cornac/datasets/README.md>`_
on GitHub and the :doc:`/api_ref/datasets` describe each one, including its size
and where it comes from.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Kind
     - Datasets (module name)
   * - Ratings and implicit feedback, many with text, image or graph side
       information
     - Amazon Clothing (``amazon_clothing``), Amazon Digital Music
       (``amazon_digital_music``), Amazon Office (``amazon_office``), Amazon
       Toy (``amazon_toy``), Citeulike (``citeulike``), Epinions
       (``epinions``), FilmTrust (``filmtrust``), MovieLens (``movielens``),
       Netflix (``netflix``), Tradesy (``tradesy``)
   * - Next-item and sessions
     - Diginetica (``diginetica``), RetailRocket (``retailrocket``), Cosmetics
       (``cosmetics``), YooChoose (``yoochoose``), Gowalla check-ins
       (``gowalla``)
   * - Next-basket
     - Ta Feng grocery baskets with price and quantity (``tafeng``)
   * - Semantic ID models
     - Amazon reviews for the Beauty, Sports and Toys categories, with item
       text (``amazon_review``)
   * - Bundles
     - Youshu book lists, NetEase playlists, iFashion outfits and Spotify
       playlists (``bundle``)

Session, basket and bundle datasets come with their own loaders, such as
``diginetica.load_train()``, ``tafeng.load_basket()`` or
``bundle.load_feedback("Youshu", split="train")``. :doc:`/user/tasks` shows
them in use.

Filtering while you load
^^^^^^^^^^^^^^^^^^^^^^^^

A :class:`~cornac.data.reader.Reader` filters the tuples as they are read.
For example, ``bin_threshold`` turns ratings into implicit feedback: it keeps
only ratings at or above the threshold and sets them to 1. ``min_user_freq``
then drops users with fewer interactions than the given number:

.. code-block:: python

    from cornac.data import Reader

    reader = Reader(bin_threshold=4.0, min_user_freq=20)
    data = movielens.load_feedback(variant="100K", reader=reader)
    print(len(data))
    print(data[:3])

.. code-block:: bash
    :caption: output

    52244
    [('298', '474', 1.0), ('253', '465', 1.0), ('286', '1014', 1.0)]

Other options keep only some users or items (``user_set``, ``item_set``), the
most frequent ones (``num_top_freq_user``, ``num_top_freq_item``), or items
with enough interactions (``min_item_freq``). Session and basket data have
their own options, such as ``min_sequence_size`` and ``min_basket_size``.

Loading side information
^^^^^^^^^^^^^^^^^^^^^^^^

Several datasets also come with information about items or users. Load it with
the dataset's other functions, wrap it in a modality class, and pass it to the
evaluation method, which lines it up with the users and items of each split:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Side information
     - Loaders
     - Modality and argument
   * - Item text
     - ``movielens.load_plot()``, ``amazon_clothing.load_text()``,
       ``citeulike.load_text()``
     - :class:`~cornac.data.text.TextModality`, ``item_text=``
   * - Item images (CNN features)
     - ``amazon_clothing.load_visual_feature()``,
       ``tradesy.load_visual_feature()``
     - :class:`~cornac.data.image.ImageModality`, ``item_image=``
   * - Item graph
     - ``amazon_clothing.load_graph()``, ``amazon_office.load_graph()``
     - :class:`~cornac.data.graph.GraphModality`, ``item_graph=``
   * - User trust network
     - ``epinions.load_trust()``, ``filmtrust.load_trust()``
     - :class:`~cornac.data.graph.GraphModality`, ``user_graph=``
   * - Reviews
     - ``amazon_digital_music.load_review()``
     - ``ReviewModality``, ``review_text=``
   * - Aspect sentiment
     - ``amazon_toy.load_sentiment()``
     - :class:`~cornac.data.sentiment.SentimentModality`, ``sentiment=``

This example loads the MovieLens movie plots and compares
:class:`~cornac.models.ctr.recom_ctr.CTR`, which learns from the plots, with a
popularity baseline:

.. code-block:: python

    import cornac
    from cornac.data import Reader, TextModality
    from cornac.data.text import BaseTokenizer
    from cornac.datasets import movielens
    from cornac.eval_methods import RatioSplit
    from cornac.metrics import Recall
    from cornac.models import CTR, MostPop

    plots, movie_ids = movielens.load_plot()
    data = movielens.load_feedback(
        variant="100K", reader=Reader(item_set=movie_ids, bin_threshold=4.0)
    )

    item_text = TextModality(
        corpus=plots,
        ids=movie_ids,
        tokenizer=BaseTokenizer(sep="\t", stop_words="english"),
        max_vocab=5000,
        max_doc_freq=0.5,
    )

    rs = RatioSplit(
        data=data, test_size=0.2, exclude_unknowns=True, item_text=item_text, seed=123
    )

    models = [MostPop(), CTR(k=20, max_iter=20, seed=123)]

    cornac.Experiment(eval_method=rs, models=models, metrics=[Recall(k=50)]).run()

.. code-block:: bash
    :caption: output

    TEST:
    ...
            | Recall@50 | Train (s) | Test (s)
    ------- + --------- + --------- + --------
    MostPop |    0.3405 |    0.0016 |   0.0949
    CTR     |    0.4882 |    2.8118 |   0.0954

``Reader(item_set=movie_ids)`` keeps only the movies that have a plot. The
`Working with Auxiliary Data
<https://github.com/PreferredAI/cornac/blob/master/tutorials/working_with_auxiliary_data.md>`_
tutorial explains the modality classes in more detail, and the `examples
<https://github.com/PreferredAI/cornac/blob/master/examples/README.md>`_ show a
model for each kind of side information.

Where the files are kept
^^^^^^^^^^^^^^^^^^^^^^^^

Downloads go to a ``.cornac`` folder in your home directory, for example
``~/.cornac/ml-100k/u.data``. If your home directory is not writable, Cornac
uses ``/tmp/.cornac`` instead. Once a file is there, Cornac reads it without
going online, so later runs are faster and work offline. To download a dataset
again, delete its file or folder under ``~/.cornac``.

Citing a dataset
^^^^^^^^^^^^^^^^

If you use one of these datasets in your research, please follow the citation
guidelines of its original authors. The `dataset catalogue
<https://github.com/PreferredAI/cornac/blob/master/cornac/datasets/README.md>`__
and the :doc:`/api_ref/datasets` say where each one comes from.

Using your own data
-------------------

While the earlier examples shows how you can use Cornac's fixed datasets to
do experiments, you may want to use your own datasets for experiments and
recommendations. To evaluate models on your own data, including with your own
train and test splits, see :doc:`/user/evaluation`.

In this example, we will use our own data to train a model. We will also
demonstrate on how we could get the predictions given your own data.

My First Data Prediction
^^^^^^^^^^^^^^^^^^^^^^^^

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
~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we will create a dataset object from the data. The dataset object
will be used to train the model.

.. code-block:: python

    from cornac.data import Dataset

    # Load the data into a dataset object
    dataset = cornac.data.Dataset.from_uir(data)

.. note::

    Cornac also supports the UIRT format (user, item, rating, timestamp),
    loaded with ``Dataset.from_uirt()``. You need it to split data by time, as
    shown in :doc:`/user/evaluation`. Next-item, next-basket and bundle models
    use their own formats, described in :doc:`/user/tasks`.

3. Create and train model
~~~~~~~~~~~~~~~~~~~~~~~~~

We will then create a model object and train it using the dataset object.

.. code-block:: python

    from cornac.models import PMF

    # Instantiate the PMF model
    model_pmf = PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123)

    # Use the fit() function to train the model
    model_pmf.fit(dataset)


4. Getting recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~

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

    # Get recommendations for user id 4
    user_id = 4
    recommended_items = model_pmf.recommend(user_id)

    # Print the recommended items
    print(recommended_items)

Loading data from CSV
^^^^^^^^^^^^^^^^^^^^^

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
~~~~~~~~~~~~~~~~~~~

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

.. code-block:: bash
    :caption: output

    [('1', '1', 5.0), ('1', '2', 1.0), ('2', '2', 3.0), ('2', '3', 3.0), ('3', '4', 3.0), ('3', '5', 5.0), ('4', '1', 5.0)]

``Reader`` reads user and item IDs as strings and ratings as numbers.

2. Creating the Dataset Object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we will create a dataset object from the data. The dataset object
will be used to train the model.

.. code-block:: python

    from cornac.data import Dataset

    # Load the data into a dataset object
    dataset = Dataset.from_uir(data)

3. Create and train model
~~~~~~~~~~~~~~~~~~~~~~~~~~

We will then create a model object and train it using the dataset object.

.. code-block:: python

    from cornac.models import PMF

    # Instantiate the PMF model
    model_pmf = PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123)

    # Use the fit() function to train the model
    model_pmf.fit(dataset)

4. Getting recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, we will use the model to get item recommendations for the users. The ``recommend()`` 
function will recommend the top rated items from the model for the user provided.

For example, we will get all recommendations for user ``"4"``.

.. code-block:: python

    # Get recommendations for user id 4
    user_id = "4"
    recommended_items = model_pmf.recommend(user_id)

    # Print the recommended items
    print(recommended_items)

.. code-block:: bash
    :caption: output

    ['1', '4', '3', '5', '2']

The output is a list of item IDs. The first item in the list is the most
recommended item for the user, followed by the second item, and so on.

.. dropdown:: View codes at this point

  .. code-block:: python
    :caption: data_csv.py
    :linenos:

    import cornac
    from cornac.models import PMF
    from cornac.data import Dataset, Reader

    # Load the data from the CSV file
    data = Reader().read('data.csv', sep=',')
    print(data)

    # Load the data into a dataset object
    dataset = Dataset.from_uir(data)

    # Instantiate the PMF model
    model_pmf = PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123)

    # Use the fit() function to train the model
    model_pmf.fit(dataset)

    # Get recommendations for user id 4
    user_id = "4"
    recommended_items = model_pmf.recommend(user_id)

    # Print the recommended items
    print(recommended_items)

Next, :doc:`/user/tasks` shows the recommendation tasks Cornac supports and
the data each one needs.
