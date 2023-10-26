Adding your Own Data
=====================

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

Loading data from code
^^^^^^^^^^^^^^^^^^^^^^

In python form:

.. code-block:: python

    data = [
        (0, 0, 5),
        (0, 1, 1),
        (1, 1, 3),
        (1, 2, 3),
        (2, 3, 3),
        (2, 4, 5),
        (3, 0, 5)
    ]

