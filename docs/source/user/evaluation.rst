Evaluation and Tuning
=====================

This page helps you measure how well recommendation models work, compare them
fairly, and tune their hyper-parameters. It covers how to split your data,
which metrics to choose, how to read the results Cornac gives you, and how to
search for good hyper-parameters.

Comparing models
----------------

An :class:`~cornac.experiment.experiment.Experiment` brings together a method
to split the data, the models to compare and the evaluation metrics, as in the
:doc:`/user/quickstart`.

Cornac makes it easy for you to run your model alongside other existing models.
To do so, simply add your model to the list of models in the experiment. Below,
``MF`` takes the place of your own model; :doc:`/developer/add_model` shows how
to write one. Apart from ``MF`` and the RMSE metric, this is the Quickstart
experiment.

.. code-block:: python

    import cornac
    from cornac.eval_methods import RatioSplit
    from cornac.metrics import RMSE, Precision, Recall
    from cornac.models import BPR, MF, PMF

    # Load a sample dataset (e.g., MovieLens)
    ml_100k = cornac.datasets.movielens.load_feedback()

    # Split the data into training and testing sets
    rs = RatioSplit(data=ml_100k, test_size=0.2, rating_threshold=4.0, seed=123)

    # Add new model to list
    models = [
        BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123),
        PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123),
        MF(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02, seed=123),
    ]

    # Define metrics to evaluate the models
    metrics = [RMSE(), Precision(k=10), Recall(k=10)]

    # run the experiment and compare the results
    cornac.Experiment(eval_method=rs, models=models, metrics=metrics, user_based=True).run()

.. code-block:: bash
    :caption: output

    TEST:
    ...
        |   RMSE | Precision@10 | Recall@10 | Train (s) | Test (s)
    --- + ------ + ------------ + --------- + --------- + --------
    BPR | 2.2267 |       0.1110 |    0.1195 |    1.4414 |   0.2735
    PMF | 0.9138 |       0.0813 |    0.0639 |    1.1986 |   0.2600
    MF  | 0.8998 |       0.0675 |    0.0463 |    0.0249 |   0.2836

The rest of this page explains each part of an experiment.

Choosing an evaluation method
-----------------------------

The evaluation method decides which interactions a model learns from and which
are held out to test it. All methods take a few common arguments:
``rating_threshold`` (the lowest rating that counts as a positive for ranking
metrics), ``exclude_unknowns`` (skip test users and items that never appear in
training, ``True`` by default), and ``seed``.

**Random split.** :class:`~cornac.eval_methods.ratio_split.RatioSplit`
shuffles all interactions and holds out a fraction for testing (``test_size``)
and, if you like, for validation (``val_size``). A value above 1 is read as a
number of interactions instead of a fraction.

**Per-user split.** :class:`~cornac.eval_methods.stratified_split.StratifiedSplit`
splits each user's interactions separately, in the same proportions, so users
appear in both the training and the test set (``group_by="item"`` does the same
per item). With
``chrono=True`` and data that has timestamps, each user's earliest interactions
are used for training and the latest ones are held out.

**Split by time.** :class:`~cornac.eval_methods.timestamp_split.TimestampSplit`
cuts all interactions at the same points in time: the latest ``test_size``
fraction is the test set and the ``val_size`` fraction just before it is the
validation set. You can also give the cut-off times directly with
``val_timestamp`` and ``test_timestamp``. Models are then trained on the past
and tested on the future, as they would be in a live system.

``TimestampSplit`` needs data with timestamps (``fmt="UIRT"``), and so does
``StratifiedSplit`` by default; pass it ``fmt="UIR"`` for data without them.

.. code-block:: python

    from cornac.datasets import movielens
    from cornac.eval_methods import StratifiedSplit, TimestampSplit

    data_t = movielens.load_feedback(fmt="UIRT", variant="100K")

    ss = StratifiedSplit(
        data=data_t,
        group_by="user",
        chrono=True,
        test_size=0.2,
        rating_threshold=4.0,
        seed=123,
    )

    ts = TimestampSplit(data=data_t, test_size=0.2, val_size=0.1, rating_threshold=4.0)

    print(ss.train_set.num_ratings, ss.test_set.num_ratings)
    print(ts.train_set.num_ratings, ts.val_set.num_ratings, ts.test_set.num_ratings)

.. code-block:: bash
    :caption: output

    79619 20295
    69997 1697 2286

The time split keeps far fewer test ratings: many MovieLens users joined after
the cut-off, so the model has never seen them and ``exclude_unknowns=True``
leaves them out.

**Cross-validation.** :class:`~cornac.eval_methods.cross_validation.CrossValidation`
splits the data into ``n_folds`` parts and uses each one in turn as the test
set. The results show each fold, then the mean and standard deviation:

.. code-block:: python

    from cornac.eval_methods import CrossValidation

    cv = CrossValidation(data=ml_100k, n_folds=5, rating_threshold=4.0, seed=123)

    cornac.Experiment(
        eval_method=cv,
        models=[MF(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02, seed=123)],
        metrics=[RMSE(), Recall(k=10)],
    ).run()

.. code-block:: bash
    :caption: output

    TEST:
    ...
    [MF]
           |   RMSE | Recall@10 | Train (s) | Test (s)
    ------ + ------ + --------- + --------- + --------
    Fold 0 | 0.9126 |    0.0524 |    0.0232 |   0.2862
    Fold 1 | 0.8996 |    0.0564 |    0.0236 |   0.2819
    Fold 2 | 0.9112 |    0.0558 |    0.0234 |   0.2828
    Fold 3 | 0.8923 |    0.0551 |    0.0232 |   0.2810
    Fold 4 | 0.9113 |    0.0532 |    0.0235 |   0.2825
    ------ + ------ + --------- + --------- + --------
    Mean   | 0.9054 |    0.0546 |    0.0234 |   0.2829
    Std    | 0.0081 |    0.0015 |    0.0002 |   0.0018

**Your own splits.** If your data already comes split, for example to compare
with published results, build the evaluation method with
:meth:`~cornac.eval_methods.base_method.BaseMethod.from_splits`. Here it uses
the first of the standard MovieLens 100K splits. Unlike the other methods, it
keeps unknown users and items by default (``exclude_unknowns=False``):

.. code-block:: python

    from cornac.data import Reader
    from cornac.eval_methods import BaseMethod
    from cornac.metrics import MAE
    from cornac.utils import cache

    reader = Reader()
    train_data = reader.read(
        cache(url="https://files.grouplens.org/datasets/movielens/ml-100k/u1.base")
    )
    test_data = reader.read(
        cache(url="https://files.grouplens.org/datasets/movielens/ml-100k/u1.test")
    )

    eval_method = BaseMethod.from_splits(
        train_data=train_data, test_data=test_data, rating_threshold=4.0
    )

    cornac.Experiment(
        eval_method=eval_method,
        models=[MF(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02, seed=123)],
        metrics=[MAE(), RMSE(), Recall(k=10)],
    ).run()

.. code-block:: bash
    :caption: output

    TEST:
    ...
       |    MAE |   RMSE | Recall@10 | Train (s) | Test (s)
    -- + ------ + ------ + --------- + --------- + --------
    MF | 0.7439 | 0.9159 |    0.0450 |    0.0294 |   0.2023

To read your own files, see :doc:`/user/owndata`.

**Popularity-aware evaluation.**
:class:`~cornac.eval_methods.propensity_stratified_evaluation.PropensityStratifiedEvaluation`
estimates how likely each item was to be observed from its popularity, splits
the test set into ``n_strata`` groups by that estimate, and reports each group
next to the usual result and an unbiased estimate, so that popular items do not
dominate the result. It follows Jadidinejad et al., *The Simpson's Paradox in
the Offline Evaluation of Recommendation Systems*; see
`propensity_stratified_evaluation_example.py
<https://github.com/PreferredAI/cornac/blob/master/examples/propensity_stratified_evaluation_example.py>`_.

Next-item, next-basket and bundle data have their own evaluation methods,
covered in :doc:`/user/tasks`.

Rating and ranking metrics
--------------------------

Cornac has two kinds of metrics, listed in :doc:`/api_ref/metrics`.

**Rating metrics** (``MAE``, ``RMSE``, ``MSE``) compare the rating a model
predicts for each test pair with the actual rating. They only make sense for
models that predict ratings, such as MF. BPR learns to rank, not to predict
ratings, which is why its RMSE in `Comparing models`_ is so high even though it
ranks best.

**Ranking metrics** (``Precision``, ``Recall``, ``NDCG``, ``HitRatio``,
``FMeasure``, ``NCRR``, ``MRR``, ``AUC``, ``MAP``) ask the model to rank
the items for each test user. Test items rated ``rating_threshold`` or higher
are the relevant ones, and items the user already liked in training are left
out. ``k`` sets how many top items count: ``Recall(k=10)`` looks at the top 10,
and without ``k`` the whole ranking is used.

The ``user_based`` argument of ``Experiment`` changes how rating metrics are
averaged. With ``True`` (the default), Cornac averages the error of each user
first, then averages over users, so every user counts the same. With
``False``, it averages over all test ratings, so active users count more.
Ranking metrics are always averaged over users.

.. code-block:: python

    cornac.Experiment(
        eval_method=rs,
        models=[MF(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02, seed=123)],
        metrics=[MAE(), RMSE()],
        user_based=False,
    ).run()

.. code-block:: bash
    :caption: output

    TEST:
    ...
       |    MAE |   RMSE | Train (s) | Test (s)
    -- + ------ + ------ + --------- + --------
    MF | 0.7149 | 0.9087 |    0.0252 |   0.1001

Validation sets
---------------

When the evaluation method has a validation set (``val_size`` in
``RatioSplit``, or ``val_data`` in ``from_splits``), ``Experiment`` also
scores every model on it and prints a ``VALIDATION`` table before the ``TEST``
one, as in `Tuning hyper-parameters`_ below. Pass ``show_validation=False`` to
``Experiment`` to skip it.

Models receive the validation set during training too. Some use it, for
example to stop early (the ``early_stopping`` option of NCF models and
LightGCN) or to keep their best epoch (``model_selection="best"`` in GRU4Rec,
SASRec and TIGER). Choose hyper-parameters on the validation set rather than the
test set; `Tuning hyper-parameters`_ shows how.

Reading the results
-------------------

Each table has one row per model and one column per metric, followed by the
training and test times in seconds. Besides printing them, ``run()`` saves the
tables to a ``CornacExp-<date>_<time>.log`` file in the current folder, or in
``save_dir`` if you give one to ``Experiment``.

You can also use the results in code. Keep the ``Experiment`` object: after
``run()``, ``exp.result`` holds one :class:`~cornac.experiment.result.Result`
per model, and ``exp.val_result`` the same for the validation set:

.. code-block:: python

    exp = cornac.Experiment(
        eval_method=rs,
        models=[
            BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123),
            MF(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02, seed=123),
        ],
        metrics=[RMSE(), Recall(k=10)],
    )
    exp.run()

    for result in exp.result:
        print(result.model_name, result.metric_avg_results["Recall@10"])

    user_scores = exp.result[0].metric_user_results["Recall@10"]
    print(len(user_scores))

.. code-block:: bash
    :caption: output

    BPR 0.11952699813743203
    MF 0.04630771301678696
    918

``metric_user_results`` holds each test user's score, keyed by Cornac's
internal user index, which is useful for significance tests. For
cross-validation, each model's result is a
:class:`~cornac.experiment.result.CVResult` with one ``Result`` per fold.

Reproducible results
--------------------

Set ``seed`` on the evaluation method to get the same split every time, and on
each model to get the same initialisation and sampling. With both set, running
the examples on this page again gives the same numbers; only the times change.
For some models, such as MF and BPR, a seed also turns off multi-threaded
training, so they train more slowly.

Tuning hyper-parameters
-----------------------

Each model has a set of hyper-parameters that can be tuned to improve its
performance. View the :doc:`/api_ref/models` documentations for
the parameters available for tuning.

For example, some hyper-parameters of the ``BPR`` model are as follows:

- ``k`` (int, optional, default: 10)
  – `The dimension of the latent factors.`
- ``max_iter`` (int, optional, default: 100)
  – `Maximum number of iterations or the number of epochs for SGD.`
- ``learning_rate`` (float, optional, default: 0.001)
  – `The learning rate for SGD.`
- ``lambda_reg`` (float, optional, default: 0.01)
  – `The regularization hyper-parameter.`

As shown in our :doc:`/user/quickstart` guide, you are able to run experiments
of different models at once and compare them with the set of metrics you have
set. Also, your model could have different optimal hyper-parameters for different
datasets. Therefore, you may want to try different combinations of hyper-parameters
to find the best combination on your data. Cornac supports hyper-parameter tuning
to achieve that purpose.

In this example, we will tune the number of factors ``k`` and the ``learning_rate``
of the ``BPR`` model.

Tuning the quickstart example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given the below block of code from the :doc:`/user/quickstart` guide,
with some slight changes:

- We have added the validation set in the ``RatioSplit`` method
- We instantiate the ``Recall@100`` metric used to track model performance

.. code-block:: python

    import cornac
    from cornac.eval_methods import RatioSplit
    from cornac.models import BPR
    from cornac.metrics import Recall

    # Load a sample dataset (e.g., MovieLens)
    ml_100k = cornac.datasets.movielens.load_feedback()

    # Split the data into training, validation and testing sets
    rs = RatioSplit(data=ml_100k, test_size=0.1, val_size=0.1, rating_threshold=4.0, seed=123)

    # Instantiate Recall@100 for evaluation
    rec100 = Recall(k=100)

    # Instantiate a matrix factorization model (e.g., BPR)
    bpr = BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123)


We would like to optimize the ``k`` and ``learning_rate`` hyper-parameters. To do
this, we can use the ``cornac.hyperopt`` module to perform the searches.

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
``GridSearch`` and ``RandomSearch``. Both train the model with each setting
they try, score it on the validation set and keep the best one, so the
evaluation method needs a validation set, such as ``val_size`` in
``RatioSplit``. They also work with next-item models and ``NextItemEvaluation``.
``OptunaSearch`` takes the same arguments and runs the search with
`Optuna <https://optuna.org>`_ (``pip install optuna``).

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

.. dropdown:: View code for this example

    .. code-block:: python

        import cornac
        from cornac.eval_methods import RatioSplit
        from cornac.models import BPR
        from cornac.metrics import Recall
        from cornac.hyperopt import Discrete, Continuous
        from cornac.hyperopt import GridSearch, RandomSearch

        # Load a sample dataset (e.g., MovieLens)
        ml_100k = cornac.datasets.movielens.load_feedback()

        # Split the data into training, validation and testing sets
        rs = RatioSplit(data=ml_100k, test_size=0.1, val_size=0.1, rating_threshold=4.0, seed=123)

        # Instantiate Recall@100 for evaluation
        rec100 = Recall(k=100)

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

    VALIDATION:
    ...
                     | Recall@100 | Time (s)
    ---------------- + ---------- + --------
    GridSearch_BPR   |     0.6951 |   0.1226
    RandomSearch_BPR |     0.6949 |   0.1228

    TEST:
    ...
                     | Recall@100 | Train (s) | Test (s)
    ---------------- + ---------- + --------- + --------
    GridSearch_BPR   |     0.6953 |   21.1887 |   0.1410
    RandomSearch_BPR |     0.6988 |   37.3076 |   0.1402

    {'k': 50, 'learning_rate': 0.01}
    {'k': np.int64(50), 'learning_rate': 0.007993039950008024}

Both searches choose on the validation set, so compare them there. Both found
``k=50``, with a ``learning_rate`` of 0.01 (``GridSearch``) and about 0.008
(``RandomSearch``), and the two settings score almost the same Recall@100
(0.6951 and 0.6949). The ``TEST`` table then shows how the chosen settings do
on data that played no part in the choice.

``n_trails`` sets how many settings ``RandomSearch`` samples from the search
space (10 by default). Each setting means training the model once more, so more
trials take longer but give a better chance of finding a good setting.

Results may vary from dataset to dataset. Try tuning your hyper-parameters
using different configurations to find the best hyper-parameters for your
dataset.

.. topic:: View related tutorial on Github

  View the Hyperparameter Search guide on Github:
  https://github.com/PreferredAI/cornac/blob/master/tutorials/param_search_vaecf.ipynb

What's Next?
------------

:doc:`/user/predict` shows how to train a model and get recommendations for a
user.
