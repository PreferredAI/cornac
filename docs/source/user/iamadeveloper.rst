For Developers
==============

This document is intended for developers who want to use Cornac in their own
projects and applications.

In this guide, we will cover the following topics:

- Experimenting with models
- Hyper-parameters tuning
- Loading data
- Training model
- Obtaining recommendations
- Model persistence

Experimenting with models
-------------------------

Cornac provides a rich collection of models that can be used to build your
recommendation systems. These models are located in the :doc:`/api_ref/models`
module.

Each model has a set of hyper-parameters that can be tuned to improve its
performance. View the :doc:`/api_ref/models` documentations for
the parameters available for tuning. 

For example, some hyper-parameters of the `BPR` model are as follows:

- ``k`` (int, optional, default: 10)
  - `The dimension of the latent factors.`
- ``learning_rate`` (float, optional, default: 0.001)
  – `The learning rate for SGD.`
- ``lambda_reg`` (float, optional, default: 0.001)
  – `The regularization hyper-parameter.`

As shown in our :doc:`/user/quickstart` guide, you are able to run experiments
of different models at once and compare them with the set of metrics you have
set. Also, your model could have different optimal hyper-parameters for different 
datasets. Therefore, you may want to try different combinations of hyper-parameters 
to find the best combination on your data. Cornac support hyper-parameter tuning 
to achieve that purpose.

Hyper-parameter tuning
---------------------
In this example, we will tune the number of factors `k` and the `learning_rate`
of the `BPR` model. We will follow the :doc:`/user/quickstart`
guide and search for the optimal combination of hyper-parameters.

In order to do this, we perform hyper-parameter searches on Cornac.

Tuning the quickstart example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given the below block fo code from the :doc:`/user/quickstart` guide,
with some slight changes:

- We have added the validation set in the `RatioSplit` method
- We instantiate the `Recall@100` metric used to track performance

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


We would like to optimize the `k` and `learning_rate` hyper-parameters. To do
this, we can use the `cornac.hyperopt` module to perform the searches.

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

As shown in the above code, we have defined two methods for hyper-parameter search,
``GridSearch`` and ``RandomSearch``.

+------------------------------------------+---------------------------------------------+
| Grid Search                              | Random Search                               |
+==========================================+=============================================+
| Searches for all possible combinations   | Randomly select combinations of hyper-      |
| of the hyper-parameters provided         | parameters within a given search space      |
+------------------------------------------+---------------------------------------------+
| Only accepts discrete values             | Accepts both discrete and continuous values |
+------------------------------------------+---------------------------------------------+

For the search ``space``, we have defined the range/set of values of the hyper-parameters 
we want to tune:

- For GridSearch method, we defined the ``k`` to be a set of discrete values (5, 10, 50). 
  This means that the model will only be tuned with those values. Similarly for the set of values
  of the ``learning_rate``.

- For RandomSearch method, the searched ``learning_rate`` value will be randomized in the 
  range of continuous values between 0.001 and 0.01. We have also set the ``n_trails=20`` meaning 
  the application will attempt 20 random combinations of ``learning_rate`` and ``k``.


Running the experiment
^^^^^^^^^^^^^^^^^^^^^^

After defining the hyperparameter search methods, we can then run the
experiments using the ``cornac.Experiment`` class.

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

    TEST:
    ...
                    | Recall@100 | Train (s) | Test (s)
    ---------------- + ---------- + --------- + --------
    GridSearch_BPR   |     0.6953 |   77.9370 |   0.9526
    RandomSearch_BPR |     0.6988 |  147.0348 |   0.7502

    {'k': 50, 'learning_rate': 0.01}
    {'k': 50, 'learning_rate': 0.007993039950008024}

As shown in the output, the ``RandomSearch`` method has found the best
combination of hyper-parameters to be ``k=50`` and ``learning_rate=0.0079``
with a Recall@100 score of 0.6988.

However, as it contains a continuous hyperparameter, the
``RandomSearch`` method may technically run forever. That is why we 
have set the ``n_trails`` parameter to 20 to stop at some point. The more we try, 
the higher chances we have of finding the best combination of hyper-parameters.

Results may vary from dataset to dataset. Try tuning your hyper-parameters
using different configurations to find the best hyper-parameters for your
dataset.

.. topic:: View related tutorial on Github

  View the Hyperparameter Search guide on Github:
  https://github.com/PreferredAI/cornac/blob/master/tutorials/param_search_vaecf.ipynb


Data loading
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

.. note::

    Cornac also supports the UIRT format (user, item, rating, timestamp).
    This format is to support sequential recommender models.

Training model
---------------

After loading the data, you can train a model using ``fit()`` method.
For this example, we will follow the parameters we have determined in the
earlier example. To train a BPR model, we can do the following:

.. code-block:: python

    from cornac.models import BPR

    # Instantiate the BPR model
    model = BPR(k=10, max_iter=200, learning_rate=0.01, lambda_reg=0.01, seed=123)

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
    print(r)

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


Model persistence
----------------------

Saving a trained model
^^^^^^^^^^^^^^^^^^^^^^

There are 2 ways to saved a trained model. You can either save the model
in an experiment, or manually save the model by code.

.. dropdown:: Option 1: Saving all models in an Experiment

    To save the model in an experiment, add the ``save_dir`` parameter.
    For example, to save models from the experiment in the previous section,
    we can do the following:

    .. code-block:: python

        # Save all models in the experiment by adding
        # the 'save_dir' parameter in the experiment
        cornac.Experiment(
            eval_method=rs,
            models=models,
            metrics=metrics,
            user_based=True,
            save_dir="saved_models"
        ).run()

    This will save all trained models in the ``saved_models`` folder of where you
    executed the python code.

    .. code-block:: bash
        :caption: Folder directory

        - example.py
        - saved_models
            |- BPR
            |   |- yyyy-MM-dd HH:mm:ss.SSSSSS.pkl
            |- PMF
                |- yyyy-MM-dd HH:mm:ss.SSSSSS.pkl

.. dropdown:: Option 2: Saving the model individually

    To save the model individually, you can use the ``save()`` method.

    .. code-block:: python

        # Instantiate the BPR model
        model = BPR(k=10, max_iter=100, learning_rate=0.01, lambda_reg=0.01, seed=123)

        # Use the fit() function to train the model
        model.fit(dataset)

        # Save the trained model
        model.save(save_dir="saved_models")
    
    This will save the trained model in the ``saved_models`` folder of where you
    executed the python code.

    .. code-block:: bash
        :caption: Folder directory

        - example.py
        - saved_models
            |- BPR
                |- yyyy-MM-dd HH:mm:ss.SSSSSS.pkl


Loading from a saved model
^^^^^^^^^^^^^^^^^^^^^^^^^^
To load a model, you can use the ``load()`` function. You could either load a folder 
containing ``.pkl`` files, or load a specific ``.pkl`` file.

.. code-block:: bash
    :caption: Folder directory

    - example.py
    - saved_models
        |- BPR
            |- yyyy-MM-dd HH:mm:ss.SSSSSS.pkl

Option 1: By loading a folder containing multiple ``.pkl`` files, Cornac would pick
the latest ``.pkl`` file in the folder.

.. code-block:: python

    # Load the trained model
    model = BPR.load("saved_models/BPR/")

Option 2: By loading a specific ``.pkl`` file, Cornac would load the specific
model indicated.

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
        recs = model.recommend(user_id="U1", k=5)
        print(recs)

Running an API service
----------------------

Cornac also provides an API service that you can use to run your own
recommendation service. This is useful if you want to build a recommendation
system for your own application.

In order to do so, you need to have flask installed in your environment.
You can do so by running the following command:

.. code-block:: bash

    pip install Flask


After installing flask, you can run the API service by running the following
command:

.. code-block:: bash
    
    FLASK_APP='cornac.serving.app' \
    MODEL_PATH='save_dir/BPR' \
    MODEL_CLASS='cornac.models.BPR' \
    flask run --host localhost --port 8080

This will serve an API for the BPR model saved in the directory ``save_dir/BPR``.
The API will be launched at `localhost:8080` and following output will be shown:

.. code-block:: bash
    :caption: Output

    Model loaded
     * Serving Flask app 'cornac.serving.app'
     * Debug mode: off
    WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
     * Running on http://localhost:8080

Note that it mentions that this is a development server and should not be used
in production. If you want to use it in production, you should use a production
WSGI server instead. This will be covered in the following section.

Obtaining a Recommendation
^^^^^^^^^^^^^^^^^^^^^^^^^^

To obtain a recommendation, do a call to the API endpoint ``/recommend`` with
the following parameters:

- ``uid``: The user ID to obtain recommendations for
- ``k``: The number of recommendations to obtain
- ``remove_seen``: Whether to remove seen items during training

.. code-block:: bash

    curl -X GET "http://127.0.0.1:8080/recommend?uid=63&k=5&remove_seen=false"

    # Response: {"recommendations": ["50", "181", "100", "258", "286"], "query": {"uid": "63", "k": 5, "remove_seen": false}}

If we want to remove seen items during training, we need to provide `train_set` when starting the serving service.

Deploying in Production
^^^^^^^^^^^^^^^^^^^^^^^

In the previous section, we have shown how to run the API service in a
development environment. However, if you want to run it in a production
environment, you should use a production WSGI server instead.

We can use gunicorn to run the API service in a production environment.

Install gunicorn by running the following command:

.. code-block:: bash

    pip install gunicorn

Once installed, you can run the API service using gunicorn by running the
following command:

.. code-block:: bash

    MODEL_PATH='save_dir/BPR' \
    MODEL_CLASS='cornac.models.BPR' \
    gunicorn -b localhost:8080 -w 4 cornac.serving.app:app

Similar to the development server, the ``MODEL_PATH`` and ``MODEL_CLASS`` needs
to be specified.

The command will run the API service and bind it to port 8080. 
You can also specify the port to bind to by changing the port
in the ``-b`` parameter.

Gunicorn will run with 4 workers. You can change the number of workers by
changing the number in the ``-w`` parameter. Gunicorn recommends that the
number of workers should be ``(2 x number of CPU cores) + 1``. View more information
about gunicorn workers at https://docs.gunicorn.org/en/stable/design.html#how-many-workers.


For more information about gunicorn, you can view the documentation at
https://docs.gunicorn.org/en/stable/run.html.


Running the API service with Docker
-----------------------------------

You can also deploy the API service using Docker. To do so, you need to have
Docker installed in your environment. You can install Docker by following the
instructions at https://docs.docker.com/get-docker/.

Running with docker run command
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After installing Docker, you may run the Docker image by running the
following command:

.. code-block:: bash

    docker run \
    -dp 8080:5000 \
    -e MODEL_PATH=save_dir/bpr \
    -e MODEL_CLASS=cornac.models.BPR \
    -v $(pwd)/cornacdata:/app/cornac/serving/save_dir \
    --mount type=volume,source="cornac_vol",target=/app/cornac/serving/data \
    registry.preferred.ai/cornac/cornac-server:1.17.0-test

The above command will run the Docker image and bind it to port 8080. You can
change the port to bind to by changing the port in the ``-dp`` parameter. For
example, if you want to bind it to port 8081, you can change the ``-dp``
parameter to ``-dp 8081:5000``.

The ``MODEL_PATH`` and ``MODEL_CLASS`` needs to be specified. The ``MODEL_PATH``
is the path to the model to be loaded. The ``MODEL_CLASS`` is the class of the
model to be loaded. For example, if you want to load a BPR model, you should
set the ``MODEL_CLASS`` to ``cornac.models.BPR``.

The ``-v`` parameter is used to mount a directory in your local machine to the
Docker container. In the above example, we mounted the ``cornacdata`` folder
in the current directory to the ``save_dir`` folder in the Docker container.
This is where the trained models will be saved.

Add your saved model to the ``cornacdata`` folder in the current directory. In
the above example, we added the ``bpr`` folder to the ``cornacdata`` folder in
the current directory. This folder will be attached to the container, which
will then be loaded from.

The ``--mount`` parameter is used to mount a Docker volume to the Docker
container. In the above example, we mounted a Docker volume named ``cornac_vol``
to the ``data`` folder in the Docker container. Reason for mounting a volume is
so that we could have a persistent volume for the feedback data. You can leave
this parameter as it is.


Running with docker-compose
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Alternatively, you can also run the Docker image using docker-compose. To do
so, you need to have docker-compose installed in your environment.

To run the Docker image using docker-compose, you can run the following
command:

.. code-block:: bash

    docker-compose up 

The ``docker-compose.yml`` file contains configuration in which you can change
the port to bind to, the model path, and the model class.

.. code-block:: yaml
    :caption: docker-compose.yml

    version: "3.8"
    services:
      cornac-server:
        image: registry.preferred.ai/cornac/cornac-server:1.17.0-test
      volumes:
        - $PWD/save_dir:/app/cornac/serving/save_dir
        - cornacvol:/app/cornac/serving/data
      environment:
        - PORT=5000
        - MODEL_PATH=save_dir/bpr
        - MODEL_CLASS=cornac.models.BPR
      ports:
        - 5000:5000
    volumes:
      cornac_vol:

Similar to the ``docker run`` version, the ``PORT`` environment variable is
used to specify the port to bind to.

The ``MODEL_PATH`` and ``MODEL_CLASS`` environment variables are used to specify
the model to be loaded. 

The ``volumes`` section is used to mount the
``cornacdata`` folder in the current directory to the ``save_dir`` folder in
the Docker container.

The ``cornac_vol`` volume is used to mount a Docker volume to
the Docker container. This is so that we could have a persistent volume for the
feedback data. You can leave this parameter as it is.

Add your saved model to the ``cornacdata`` folder in the current directory. In
the above example, we added the ``bpr`` folder to the ``cornacdata`` folder in
the current directory. This folder will be attached to the container, which
will then be loaded from.

After running the above command, the API service will be launched at
`localhost:8080`.


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

------

.. topic:: Are you a data scientist?

  Find out how you can have Cornac as part of your workflow to run your
  experiments, and use Cornac's many models with just a few lines of code.
  View :doc:`/user/iamaresearcher`.

.. topic:: For all the awesome people out there

  No matter who you are, you could also consider contributing to Cornac,
  with our contributors guide.
  View :doc:`/developer/index`.