Saving and Serving
==================

Once you have trained a model, you can save it to disk, load it again later,
and serve it to your application as a web service. This page shows you how to
save and load models, run the Cornac API service with Flask or gunicorn, and
run it with Docker.

Model persistence
-----------------

Saving a trained model
^^^^^^^^^^^^^^^^^^^^^^

There are 2 ways to save a trained model. You can either save the model
in an experiment, or manually save the model by code.

.. dropdown:: Option 1: Saving all models in an Experiment

    To save the model in an experiment, add the ``save_dir`` parameter.
    For example, to save models from the experiment in the :doc:`/user/quickstart`,
    we can do the following:

    .. code-block:: python

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
    executed the python code. The experiment also writes its log file there.

    .. code-block:: bash
        :caption: Folder directory

        - example.py
        - saved_models
            |- BPR
            |   |- yyyy-MM-dd_HH-mm-ss-SSSSSS.pkl
            |   |- yyyy-MM-dd_HH-mm-ss-SSSSSS.pkl.meta
            |- PMF
            |   |- yyyy-MM-dd_HH-mm-ss-SSSSSS.pkl
            |   |- yyyy-MM-dd_HH-mm-ss-SSSSSS.pkl.meta
            |- CornacExp-yyyy-MM-dd_HH-mm-ss-SSSSSS.log

.. dropdown:: Option 2: Saving the model individually

    To save the model individually, you can use the ``save()`` method.
    Here, ``dataset`` is the small dataset from :doc:`/user/predict`.

    .. code-block:: python

        # Instantiate the BPR model
        model = BPR(k=10, max_iter=100, learning_rate=0.01, lambda_reg=0.01, seed=123)

        # Use the fit() function to train the model
        model.fit(dataset)

        # Save the trained model
        model_path = model.save(save_dir="saved_models")
        print(model_path)

    This will save the trained model in the ``saved_models`` folder of where you
    executed the python code. ``save()`` returns the path of the saved file.

    .. code-block:: bash
        :caption: output

        saved_models/BPR/2026-11-16_15-10-58-404062.pkl

    .. code-block:: bash
        :caption: Folder directory

        - example.py
        - saved_models
            |- BPR
                |- yyyy-MM-dd_HH-mm-ss-SSSSSS.pkl
                |- yyyy-MM-dd_HH-mm-ss-SSSSSS.pkl.meta

In both cases, Cornac creates a folder named after the model (``BPR``, ``PMF``)
inside ``save_dir``. The model goes into a ``.pkl`` file named after the time it
was saved, so a new save never overwrites an older one. Next to it, a ``.meta``
file records the model class and the file name.

To keep the training data with the model, pass ``save_trainset=True``:

.. code-block:: python

    model_path = model.save(save_dir="saved_models", save_trainset=True)

Cornac then also writes the train set to a ``.trainset`` file next to the model.
A served model uses it to leave out items that users have already seen, as
described in `Running an API service`_ below.

.. code-block:: bash
    :caption: Folder directory

    - example.py
    - saved_models
        |- BPR
            |- yyyy-MM-dd_HH-mm-ss-SSSSSS.pkl
            |- yyyy-MM-dd_HH-mm-ss-SSSSSS.pkl.meta
            |- yyyy-MM-dd_HH-mm-ss-SSSSSS.pkl.trainset


Loading from a saved model
^^^^^^^^^^^^^^^^^^^^^^^^^^
To load a model, you can use the ``load()`` function. You could either load a folder
containing ``.pkl`` files, or load a specific ``.pkl`` file.

.. code-block:: bash
    :caption: Folder directory

    - example.py
    - saved_models
        |- BPR
            |- yyyy-MM-dd_HH-mm-ss-SSSSSS.pkl
            |- yyyy-MM-dd_HH-mm-ss-SSSSSS.pkl.meta

Option 1: By loading a folder containing multiple ``.pkl`` files, Cornac would pick
the latest ``.pkl`` file in the folder.

.. code-block:: python

    # Load the trained model
    model = BPR.load("saved_models/BPR/")

Option 2: By loading a specific ``.pkl`` file, Cornac would load the specific
model indicated, such as the path returned by ``save()``.

.. code-block:: python

    # Load the trained model
    model = BPR.load(model_path)

After you have loaded the model, you can use the ``recommend()`` method to
obtain recommendations for users.

.. dropdown:: View code for this example

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

        # Save the trained model
        model_path = model.save(save_dir="saved_models")

        # Load the BPR model
        model = BPR.load(model_path)

        # Obtain item recommendations for user U1
        recs = model.recommend(user_id="U1", k=5)
        print(recs)

    .. code-block:: bash
        :caption: output

        ['I2', 'I1', 'I3', 'I4', 'I5']

Running an API service
----------------------

Cornac also provides an API service that you can use to run your own
recommendation service. This is useful if you want to build a recommendation
system for your own application.

In order to do so, you need to have flask installed in your environment.
You can do so by running the following command:

.. code-block:: bash

    pip install Flask

The service loads a model that was saved with ``save()``, as described in
`Model persistence`_. For example, the following code trains the BPR model
from the :doc:`/user/quickstart` and saves it, together with its train set, in
the folder ``save_dir/BPR``:

.. code-block:: python

    import cornac
    from cornac.eval_methods import RatioSplit
    from cornac.models import BPR

    # Load MovieLens 100K and split it as in the quickstart
    ml_100k = cornac.datasets.movielens.load_feedback()
    rs = RatioSplit(data=ml_100k, test_size=0.2, rating_threshold=4.0, seed=123)

    # Train a BPR model
    bpr = BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123)
    bpr.fit(rs.train_set)

    # Save the model together with its train set
    bpr.save("save_dir", save_trainset=True)

After installing flask, you can run the API service by running the following
command:

.. code-block:: bash

    MODEL_PATH="$(pwd)/save_dir/BPR" \
    MODEL_CLASS='cornac.models.BPR' \
    flask --app cornac.serving.app run --host localhost --port 8080  # requires Flask>=2.2

This will serve an API for the BPR model saved in the directory ``save_dir/BPR``.
``MODEL_PATH`` names the folder that ``save()`` created, which has the model's
name (``BPR`` here), or one ``.pkl`` file in it. Give it as an absolute path,
as above, because the service does not resolve a relative path against the
folder you start it from. ``MODEL_CLASS`` is the class of the saved model.

The API will be launched at ``localhost:8080`` and following output will be shown:

.. code-block:: bash
    :caption: Output

    Loading the 'cornac.models.BPR' model from '/path/to/save_dir/BPR'...

            Model and train set loaded. Remove seen items by adding
            'remove_seen=true' query param to the recommend endpoint.

     * Serving Flask app 'cornac.serving.app'
     * Debug mode: off
    WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
     * Running on http://localhost:8080
    Press CTRL+C to quit

Note that it mentions that this is a development server and should not be used
in production. If you want to use it in production, you should use a production
WSGI server instead, as described in `Deploying in Production`_ below.

Obtaining a Recommendation
^^^^^^^^^^^^^^^^^^^^^^^^^^

To obtain a recommendation, do a call to the API endpoint ``/recommend`` with
the following parameters:

- ``uid``: The user ID to obtain recommendations for
- ``k``: The number of recommendations to obtain (all items if left out)
- ``remove_seen``: Whether to remove items the user has in the training set (``false`` if left out)

.. code-block:: bash

    curl -X GET "http://127.0.0.1:8080/recommend?uid=63&k=5&remove_seen=false"

    # Response: {"query":{"k":5,"remove_seen":false,"uid":"63"},"recommendations":["50","181","100","258","286"]}

To remove seen items, the service needs the train set. It loads the
``.trainset`` file that ``save(..., save_trainset=True)`` wrote next to the
model. For a model saved without one, set the ``TRAIN_SET`` environment
variable to the absolute path of a train set saved with
:meth:`~cornac.data.dataset.Dataset.save`. Without a train set, the service
answers ``remove_seen=true`` with an error.

.. code-block:: bash

    curl -X GET "http://127.0.0.1:8080/recommend?uid=63&k=5&remove_seen=true"

    # Response: {"query":{"k":5,"remove_seen":true,"uid":"63"},"recommendations":["174","98","56","294","127"]}

Collecting feedback
^^^^^^^^^^^^^^^^^^^

Your application can send user feedback to the ``/feedback`` endpoint with a
``POST`` request and the parameters ``uid``, ``iid`` and ``rating`` (``1`` if
left out). The service appends each record to ``data/feedback.csv`` in the
folder you started it from.

.. code-block:: bash

    curl -X POST "http://127.0.0.1:8080/feedback?uid=63&iid=1&rating=5"

    # Response: {"data":{"iid":"1","rating":"5","time":"2026-11-16 07:08:00.409596+00:00","uid":"63"},"message":"Feedback added"}

Evaluating the served model
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``/evaluate`` endpoint scores the served model on the feedback collected so
far. Send a ``POST`` request with a JSON body that lists the ``metrics`` to
compute. To evaluate on other data instead, add it as ``data``, a list of
``[user, item, rating]`` records. Evaluation also needs the train set.

.. code-block:: bash

    curl -X POST -H "Content-Type: application/json" \
        -d '{"metrics": ["Recall(k=10)", "NDCG(k=10)"]}' \
        "http://127.0.0.1:8080/evaluate"

    # Response: {"result":{"NDCG@10":0.6309297535714575,"Recall@10":1.0},"user_result":{"NDCG@10":{"63":0.6309297535714575},"Recall@10":{"63":1.0}}}

The response holds the average of each metric (``result``) and the score of
each user (``user_result``).

Deploying in Production
^^^^^^^^^^^^^^^^^^^^^^^

So far, we have shown how to run the API service in a development environment.
However, if you want to run it in a production
environment, you should use a production WSGI server instead.

We can use gunicorn to run the API service in a production environment.

Install gunicorn by running the following command:

.. code-block:: bash

    pip install gunicorn

Once installed, you can run the API service using gunicorn by running the
following command:

.. code-block:: bash

    MODEL_PATH="$(pwd)/save_dir/BPR" \
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
    -e MODEL_PATH=save_dir/BPR \
    -e MODEL_CLASS=cornac.models.BPR \
    -v $(pwd)/cornacdata:/app/cornac/serving/save_dir \
    --mount type=volume,source="cornac_vol",target=/app/cornac/serving/data \
    ghcr.io/preferredai/cornac-server:latest

The above command will run the Docker image and bind it to port 8080. You can
change the port to bind to by changing the port in the ``-dp`` parameter. For
example, if you want to bind it to port 8081, you can change the ``-dp``
parameter to ``-dp 8081:5000``. Inside the container, the service listens on
port 5000.

The ``MODEL_PATH`` and ``MODEL_CLASS`` needs to be specified. The ``MODEL_PATH``
is the path to the model to be loaded. Inside the container, a relative path
such as ``save_dir/BPR`` points into the folder mounted with ``-v``, and its
last part is the folder that ``save()`` created, named after the model. The
``MODEL_CLASS`` is the class of the model to be loaded. For example, if you
want to load a BPR model, you should set the ``MODEL_CLASS`` to
``cornac.models.BPR``.

The ``-v`` parameter is used to mount a directory in your local machine to the
Docker container. In the above example, we mounted the ``cornacdata`` folder
in the current directory to the ``save_dir`` folder in the Docker container.
This is where the container looks for trained models.

Add your saved model to the ``cornacdata`` folder in the current directory. In
the above example, we added the ``BPR`` folder to the ``cornacdata`` folder in
the current directory. This folder will be attached to the container, which
will then be loaded from. Saving the model with
``bpr.save("cornacdata", save_trainset=True)`` creates this folder, and the
saved train set lets the container remove seen items and evaluate the model.

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
the port to bind to, the model path, and the model class. This is the file in
the Cornac repository:

.. code-block:: yaml
    :caption: docker-compose.yml

    version: '3.7'

    services:
      cornac:
        image: ghcr.io/preferredai/cornac-server:latest
        volumes:
          - $PWD/save_dir:/app/cornac/serving/save_dir
          - cornacvol:/app/cornac/serving/data
        environment:
          - MODEL_PATH=save_dir/bpr
          - MODEL_CLASS=cornac.models.BPR
        ports:
          - 8080:5000
    volumes:
      cornacvol:

Similar to the ``docker run`` version, the ``ports`` section binds port 8080 on
your machine to port 5000 in the container.

The ``MODEL_PATH`` and ``MODEL_CLASS`` environment variables are used to specify
the model to be loaded. ``MODEL_PATH`` has to match the folder that ``save()``
created, so for the model saved above, change it to ``save_dir/BPR``.

The ``volumes`` section is used to mount the
``save_dir`` folder in the current directory to the ``save_dir`` folder in
the Docker container.

The ``cornacvol`` volume is used to mount a Docker volume to
the Docker container. This is so that we could have a persistent volume for the
feedback data. You can leave this parameter as it is.

Add your saved model to the ``save_dir`` folder in the current directory. In
the above example, we added the ``BPR`` folder to the ``save_dir`` folder in
the current directory. This folder will be attached to the container, which
will then be loaded from.

After running the above command, the API service will be launched at
``localhost:8080``.


What's Next?
------------

The :doc:`/user/example-foodrecce` walks step by step through the development
of a restaurant recommendation application.
