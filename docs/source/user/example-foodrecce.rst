FoodRecce Example
=================

Introduction
------------

``FoodRecce`` (shortform for Food Reconnaissance) is an application that allows
users to search for food places around their current location.

FoodRecce is a mobile app that allows users to provide feedback (like/dislike)
the restaurants they love,

The infrastructure of the application is as follows:

``Mobile App - Backend Server/Cornac - Database``


Scope of this example
---------------------

This example will show you how to use Cornac to build a recommendation
system for FoodRecce. The development of mobile app and database storage is 
out of scope for this example.


Loading data
------------
The data is being loaded from the database. A simple SQL statement
``SELECT * FROM feedbacks`` could be used to obtain the feedback data on Python.

For brevity, a converted version of this data into Cornac's format will look
like the following:

.. code-block:: python

    # Converted data as a list of UIR (user, item, rating) tuples
    data = [
        ("uid_001", "restaurant_032", 1),
        ("uid_954", "restaurant_012", 0),
        ("uid_022", 'resturant_027', 1),
        ...
    ]

We then convert the data to the dataset format:

.. code-block:: python

    from cornac.data import Dataset

    dataset = Dataset.from_uir(data, seed=123)


Building the recommender
------------------------

We use the BPR recommender to build the recommendation system for FoodRecce.
Based on historical data, we are able to rank other unseen restaurants based
on the user's past preferences.

Assuming that we have already experimented different parameters and values for
the BPR model, we can then train the recommender as follows:

.. code-block:: python

    from cornac.models import BPR

    bpr = BPR(k=10, max_iter=100, learning_rate=0.01, lambda_reg=0.001, seed=123)

    bpr.fit(dataset)

    # Save the model
    bpr.save(save_dir="saved_models")

Upon running this code, the model will be saved in the ``saved_models`` directory.

.. code-block:: bash
    :caption: Folder directory
    
    - example.py
    - saved_models
        |- BPR
            |- yyyy-MM-dd HH:mm:ss.SSSSSS.pkl

.. note::
    
    While it may not be essential to save the model, it is good practice to do
    so. This is because we can then load the model directly from the saved
    directory without having to retrain the model again.

Making recommendations
----------------------

After training the model, we can then use the model to make recommendations
for users. We can do this by loading the model and calling the ``recommend``
method.

We first load our saved model. For subsequent runs, we can load the model
directly from the saved directory. 

.. code-block:: python

    from cornac.models import BPR

    bpr = BPR.load("saved_models/BPR/yyyy-MM-dd HH:mm:ss.SSSSSS.pkl")

Now, given that our backend receives a load request from a user with id
``uid_001``, we can then make recommendations for this user.

We then make recommendations for a user with id ``uid_001``. We can do this
by calling the ``recommend`` method.

.. code-block:: python

    # variables are shown here for brevity
    user_id = "uid_001"
    top_k = 10

    def get_recommendations(user_id, top_k):
        # Get the top k recommendations for user with id user_id
        top_k_recs = bpr.recommend(user_id=user_id, top_k=top_k)
        return top_k_recs


The values returned for this will be as follows;

.. code-block:: bash

    [
        'restaurant_038',
        'restaurant_012',
        'restaurant_027',
        'restaurant_081',
        'restaurant_002',
        'restaurant_030',
        'restaurant_104',
        'restaurant_235',
        'restaurant_006',
        'restaurant_007'
    ]

The above list of restaurants are the top 10 recommendations for the user
with id ``uid_001``. The list is ranked in descending order, with the first
item being the most recommended item.

.. note::

    The list of recommendations are in the form of item ids. The item ids
    are the same as the item ids in the database. The mobile app will then
    use the item ids to query the database for the restaurant information.


Updating the model
------------------

As more users provide feedback on the restaurants, we can then update the
model with the new feedback data. We can do this by calling the ``fit``
method again as in the `Building the Recommender`` section.


Conclusion
----------

We have just briefly shown you how to build a recommendation system for a 
food recommendation app. There are many uses for recommender systems.

Feel free to try to building your own recommendation system for your own
application, and share them with us!


What's Next?
------------

Now that you have learned how to use Cornac for your own projects and
applications, you can now start building your own recommendation systems using
Cornac.

.. topic:: View the Developer Quickstart

  View a quickstart guide on how you can code and implement Cornac onto your
  application to provide recommendations for your users.

  View :doc:`/user/iamadeveloper`.

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