Quickstart
=================

Cornac is a Python library for building and training recommendation models.
It focuses on making it convenient to work with models leveraging auxiliary data 
(e.g., item descriptive text and image, social network, etc). 

Cornac enables fast experiments and straightforward implementations of new models. 
It is highly compatible with existing machine learning libraries (e.g., TensorFlow, PyTorch).


Cornac Experiment Concept
-------------------------
The main idea behind Cornac is to provide a simple and flexible way to experiment with different algorithms, hyperparameters, and datasets without having to manually implement and run all the code yourself.
Here are some key concepts related to Cornac:

1. Experiments
^^^^^^^^^^^^^^
In Cornac, an "experiment" refers to a specific combination of algorithm, hyperparameters, dataset, and evaluation metric that is being tested or evaluated.
Each experiment corresponds to a single row in the database, which contains information about the experiment such as its ID, name, description, and date created/updated.

2. Datasets
^^^^^^^^^^^
In Cornac, a "dataset" refers to a specific collection of input data that is used to train or test an algorithm.
Each dataset corresponds to a single row in the database, which contains information about the dataset such as its ID, name, description, and date created/updated.

3. Algorithms
^^^^^^^^^^^^^
In Cornac, an "algorithm" refers to a specific computational model or technique that is being used to perform some task or function.
Each algorithm corresponds to a single row in the database, which contains information about the algorithm such as its ID, name, description, and date created/updated.

4. Hyperparameters
^^^^^^^^^^^^^^^^^^
In Cornac, a "hyperparameter" refers to a specific parameter or setting that is being adjusted or fine-tuned during the experimentation process.
Each hyperparameter corresponds to a single row in the database, which contains information about the hyperparameter such as its ID, name, description, and date created/updated.

5. Evaluation metrics
^^^^^^^^^^^^^^^^^^^^^
In Cornac, an "evaluation metric" refers to a specific performance measure or score that is being used to evaluate or compare different algorithms or models during the experimentation process.
Each evaluation metric corresponds to a single row in the database, which contains information about the evaluation metric such as its ID, name, description, and date created/updated.


New to recommender systems?
---------------------------

.. topic:: Recommender Systems

   If you're new to recommender systems, this link provides a beginner-friendly
   introduction to help you understand the fundamentals and get started:
   https://github.com/PreferredAI/tutorials/tree/master/recommender-systems


The First Experiment
--------------------

In today's world of countless movies and TV shows at our fingertips,
finding what we truly enjoy can be a challenge.
This experiment focuses on recommendation systems, using the MovieLens dataset,
to help us discover movies and shows we love.

About the MovieLens dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The MovieLens dataset, a repository of movie ratings and user preferences,
remains highly relevant today as it powers the personalized recommendation
algorithms crucial for modern streaming services.

It continues to drive innovation in data science and plays a pivotal role in
improving the user experience and content curation in an era of vast digital
media options.

.. image:: /flow.jpg
   :width: 600

1. Data Loading
^^^^^^^^^^^^^^^
.. code-block:: python

    import cornac

    # Load a sample dataset (e.g., MovieLens)
    ml_100k = cornac.datasets.movielens.load_feedback()

2. Data Splitting
^^^^^^^^^^^^^^^^^
.. code-block:: python

    # Split the data into training and testing sets
    rs = RatioSplit(data=ml_100k, test_size=0.2, rating_threshold=4.0, seed=123)

3. Define Model
^^^^^^^^^^^^^^^
.. code-block:: python

    from cornac.models import BPR

    # Instantiate a matrix factorization model (e.g., BPR)
    models = [
        BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123),
    ]

4. Define Metrics
^^^^^^^^^^^^^^^^^
.. code-block:: python

    from cornac.metrics import RMSE, Precision, Recall

    # Define metrics to evaluate the models
    metrics = [RMSE(), Precision(k=10), Recall(k=10)]

5. Run Experiment
^^^^^^^^^^^^^^^^^
.. code-block:: python

    # put it together in an experiment, voilà!
    cornac.Experiment(eval_method=rs, models=models, metrics=metrics, user_based=True).run()


What do this results mean?
^^^^^^^^^^^^^^^^^^^^^^^^^^
<picture of result>
description

Getting Predictions
^^^^^^^^^^^^^^^^^^^



Putting it all together
-----------------------
.. code-block:: python
    :caption: python.py

    import cornac
    from cornac.eval_methods import RatioSplit
    from cornac.models import MF, PMF, BPR
    from cornac.metrics import MAE, RMSE, Precision, Recall, NDCG, AUC, MAP

    # load the built-in MovieLens 100K and split the data based on ratio
    ml_100k = cornac.datasets.movielens.load_feedback()
    rs = RatioSplit(data=ml_100k, test_size=0.2, rating_threshold=4.0, seed=123)

    # initialize models, here we are comparing: Biased MF, PMF, and BPR
    models = [
        BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123),
    ]

    # define metrics to evaluate the models
    metrics = [MAE(), RMSE(), Precision(k=10), Recall(k=10), NDCG(k=10), AUC(), MAP()]

    # put it together in an experiment, voilà!
    cornac.Experiment(eval_method=rs, models=models, metrics=metrics, user_based=True).run()


What's Next?
------------

**Are you a developer?**

Find out how you can use Cornac as a recommender system to many diferrent applications. 
View :doc:`applications`.

**Are you a data scientist?**

Find out how you can use Cornac to run experiments and tweak parameters easily to compare against baselines already on Cornac.
View :doc:`experiments`.

**For all the awesome people out there**

No matter who you are, you could also consider contributing to Cornac, with our contributors guide.
View :doc:`/developer/index`.

