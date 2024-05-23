For Researchers
===============

This document is intended to provide a quick introduction on how researchers
like you could use Cornac to conduct recommender systems research.

In this guide, we will cover the following topics:

- What can you do with Cornac?
- Conducting experiments
- Tuning hyper-parameters
- Adding your own model
- Adding your own metric
- Adding your own dataset
- Development workflow
- Using additional packages

What can you do with Cornac?
----------------------------

Cornac is a recommender systems framework that provides a wide range of
recommender models, evaluation metrics, and experimental tools.
It is designed to be flexible and extensible, allowing researchers to
easily conduct experiments and compare their models with existing ones.

Cornac is written in Python and is built on top of the popular scientific
computing libraries such as NumPy and SciPy. It is also designed to be compatible 
with the popular deep learning libraries such as PyTorch and TensorFlow.

View the models, datasets, metrics that are currently built into Cornac:

- :doc:`/api_ref/models`
- :doc:`/api_ref/datasets`
- :doc:`/api_ref/metrics`

Conducting experiments
----------------------

Cornac provides a set of tools to help you conduct experiments. The main tool is
the :class:`~cornac.experiment.Experiment` class. As its name would suggests, this is
where you manage an experiment with a method to split dataset (e.g., by ratio, k-fold 
cross-validation), a set of models to be compared with, and different evaluation metrics.

Recap from the :doc:`/user/quickstart` example, the following code snippet shows
how to create an experiment with the MovieLens 100K dataset, split it into 
training and testing sets, and run the experiment with the BPR recommender model
and the Precision, Recall evaluation metrics.

.. code-block:: python

    import cornac
    from cornac.eval_methods import RatioSplit
    from cornac.models import BPR, PMF
    from cornac.metrics import Precision, Recall

    # Load a sample dataset (e.g., MovieLens)
    ml_100k = cornac.datasets.movielens.load_feedback()

    # Split the data into training and testing sets
    rs = RatioSplit(data=ml_100k, test_size=0.2, rating_threshold=4.0, seed=123)
    
    # Instantiate a matrix factorization model (e.g., BPR)
    models = [
        BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123),
        PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123),  
    ]

    # Define metrics to evaluate the models
    metrics = [Precision(k=10), Recall(k=10)]

    # put it together in an experiment, voilà!
    cornac.Experiment(eval_method=rs, models=models, metrics=metrics, user_based=True).run()

Tuning hyper-parameters
-----------------------
In this example, we will tune the number of factors `k` and the `learning_rate` of the 
`BPR` model. Given the below block fo code from the :doc:`/user/quickstart` guide,
with some slight changes:

- We have added the validation set in the `RatioSplit` method
- We instantiate the `Recall@100` metric used to track model performance

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
this, we can use the `cornac.hyperopt` module to perform the searches. Cornac supports two methods
for hyper-parameter search, ``GridSearch`` and ``RandomSearch``.

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

After defining the hyper-parameter search methods, we can then run the
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
        rs = RatioSplit(data=ml_100k, test_size=0.1, val_size=0.1, rating_threshold=4.0, seed=123)

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
combination of hyperparameters to be ``k=50`` and ``learning_rate=0.0079``
with a Recall@100 score of 0.6988.

.. topic:: View related tutorial on Github

  View the Hyperparameter Search guide on Github:
  https://github.com/PreferredAI/cornac/blob/master/tutorials/param_search_vaecf.ipynb


Adding your own model
---------------------

Adding your own model on Cornac is easy. Cornac is designed to be flexible and
extensible, allowing researchers to easily add their own models into the
framework.

Files to add
^^^^^^^^^^^^

Below is an example of the ``PMF`` model which was already added into Cornac.
We will use this as a reference to add our own model.

.. code-block:: bash
    
    cornac
    |-- cornac
    |   |-- models
    |       |-- pmf
    |           |-- __init__.py
    |           |-- recom_pmf.py
    |-- examples
        |-- pmf_ratio.py

.. dropdown:: 1. Create the base folder for your model

    .. code-block:: bash

        cornac
        |-- cornac
            |-- models
                |-- pmf

.. dropdown:: 2. Create the ``__init__.py`` file in the ``pmf`` folder

    Add the following line to the ``__init__.py`` file in your model folder.
    The ``.recom_pmf`` coincides with the name of the file that contains the
    model, and ``PMF`` coincides with the name of the class in the 
    ``recon_pmf.py`` file.

    .. code-block:: python
        :caption: cornac/cornac/models/pmf/__init__.py

        from .recom_pmf import PMF


.. dropdown:: 3. Create the ``recom_pmf.py`` file in the ``pmf`` folder

    The ``recom_pmf.py`` file contains the logic of the model. This includes
    the training and testing portions of the model.

    Core to the ``recom_pmf.py`` file is the ``PMF`` class. This class inherits
    from the :class:`~cornac.models.Recommender` class. The ``PMF`` class
    implements the following methods:

    - :meth:`~cornac.models.Recommender.__init__`: The constructor of the class
    - :meth:`~cornac.models.Recommender.fit`: The training procedure of the model
    - :meth:`~cornac.models.Recommender.score`: The scoring function of the model

    .. code-block:: python
        :caption: __init__ method: The constructor

        # Here we initialize parameters and variables

        def __init__(
            self,
            k=5,
            max_iter=100,
            learning_rate=0.001,
            gamma=0.9,
            lambda_reg=0.001,
            name="PMF",
            variant="non_linear",
            trainable=True,
            verbose=False,
            init_params=None,
            seed=None,
        ):
            Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
            self.k = k
            self.max_iter = max_iter
            self.learning_rate = learning_rate
            self.gamma = gamma
            self.lambda_reg = lambda_reg
            self.variant = variant
            self.seed = seed

            self.ll = np.full(max_iter, 0)
            self.eps = 0.000000001

            # Init params if provided
            self.init_params = {} if init_params is None else init_params
            self.U = self.init_params.get("U", None)  # matrix of user factors
            self.V = self.init_params.get("V", None)  # matrix of item factors

    .. code-block:: python
        :caption: fit method: The training procedure

        # Here we implement the training procedure of the model

        def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """
        Recommender.fit(self, train_set)

        from cornac.models.pmf import pmf

        if self.trainable:
            # converting data to the triplet format (needed for cython function pmf)
            (uid, iid, rat) = train_set.uir_tuple
            rat = np.array(rat, dtype="float32")
            if self.variant == "non_linear":  # need to map the ratings to [0,1]
                if [self.min_rating, self.max_rating] != [0, 1]:
                    rat = scale(rat, 0.0, 1.0, self.min_rating, self.max_rating)
            uid = np.array(uid, dtype="int32")
            iid = np.array(iid, dtype="int32")

            if self.verbose:
                print("Learning...")

            # use pre-trained params if exists, otherwise from constructor
            init_params = {"U": self.U, "V": self.V}

            if self.variant == "linear":
                res = pmf.pmf_linear(
                    uid,
                    iid,
                    rat,
                    k=self.k,
                    n_users=self.num_users,
                    n_items=self.num_items,
                    n_ratings=len(rat),
                    n_epochs=self.max_iter,
                    lambda_reg=self.lambda_reg,
                    learning_rate=self.learning_rate,
                    gamma=self.gamma,
                    init_params=init_params,
                    verbose=self.verbose,
                    seed=self.seed,
                )
            elif self.variant == "non_linear":
                res = pmf.pmf_non_linear(
                    uid,
                    iid,
                    rat,
                    k=self.k,
                    n_users=self.num_users,
                    n_items=self.num_items,
                    n_ratings=len(rat),
                    n_epochs=self.max_iter,
                    lambda_reg=self.lambda_reg,
                    learning_rate=self.learning_rate,
                    gamma=self.gamma,
                    init_params=init_params,
                    verbose=self.verbose,
                    seed=self.seed,
                )
            else:
                raise ValueError('variant must be one of {"linear","non_linear"}')

            self.U = np.asarray(res["U"])
            self.V = np.asarray(res["V"])

            if self.verbose:
                print("Learning completed")

        elif self.verbose:
            print("%s is trained already (trainable = False)" % (self.name))

        return self
    
    .. code-block:: python
        :caption: score method: The scoring function

        # Here we implement the scoring function of the model.
        # If item-idx is not provided, return scores for all known items
        # of the users. Otherwise, return the score of the user-item pair

        def score(self, user_idx, item_idx=None):
            """Predict the scores/ratings of a user for an item.

            Parameters
            ----------
            user_idx: int, required
                The index of the user for whom to perform score prediction.

            item_idx: int, optional, default: None
                The index of the item for which to perform score prediction.
                If None, scores for all known items will be returned.

            Returns
            -------
            res : A scalar or a Numpy array
                Relative scores that the user gives to the item or to all known items

            """
            if item_idx is None:
                if not self.knows_user(user_idx):
                    raise ScoreException(
                        "Can't make score prediction for (user_id=%d)" % user_idx
                    )

                known_item_scores = self.V.dot(self.U[user_idx, :])
                return known_item_scores
            else:
                if not self.knows_user(user_idx) or not self.knows_item(item_idx):
                    raise ScoreException(
                        "Can't make score prediction for (user_id=%d, item_id=%d)"
                        % (user_idx, item_idx)
                    )

                user_pred = self.V[item_idx, :].dot(self.U[user_idx, :])

                if self.variant == "non_linear":
                    user_pred = sigmoid(user_pred)
                    user_pred = scale(user_pred, self.min_rating, self.max_rating, 0.0, 1.0)

                return user_pred

    Putting everything together, below we have the whole recom_pmf.py file:

    .. code-block:: python
        :caption: cornac/cornac/models/pmf/recom_pmf.py

        import numpy as np

        from ..recommender import Recommender
        from ...utils.common import sigmoid
        from ...utils.common import scale
        from ...exception import ScoreException


        class PMF(Recommender):
            """Probabilistic Matrix Factorization.

            Parameters
            ----------
            k: int, optional, default: 5
                The dimension of the latent factors.

            max_iter: int, optional, default: 100
                Maximum number of iterations or the number of epochs for SGD.

            learning_rate: float, optional, default: 0.001
                The learning rate for SGD_RMSProp.
                
            gamma: float, optional, default: 0.9
                The weight for previous/current gradient in RMSProp.

            lambda_reg: float, optional, default: 0.001
                The regularization coefficient.

            name: string, optional, default: 'PMF'
                The name of the recommender model.
                
            variant: {"linear","non_linear"}, optional, default: 'non_linear'
                Pmf variant. If 'non_linear', the Gaussian mean is the output of a Sigmoid function.\
                If 'linear' the Gaussian mean is the output of the identity function.

            trainable: boolean, optional, default: True
                When False, the model is not trained and Cornac assumes that the model already \
                pre-trained (U and V are not None).
                
            verbose: boolean, optional, default: False
                When True, some running logs are displayed.

            init_params: dict, optional, default: None
                List of initial parameters, e.g., init_params = {'U':U, 'V':V}.
                
                U: ndarray, shape (n_users, k) 
                    User latent factors.
                
                V: ndarray, shape (n_items, k)
                    Item latent factors.

            seed: int, optional, default: None
                Random seed for parameters initialization.

            References
            ----------
            * Mnih, Andriy, and Ruslan R. Salakhutdinov. Probabilistic matrix factorization. \
            In NIPS, pp. 1257-1264. 2008.
            """

            def __init__(
                self,
                k=5,
                max_iter=100,
                learning_rate=0.001,
                gamma=0.9,
                lambda_reg=0.001,
                name="PMF",
                variant="non_linear",
                trainable=True,
                verbose=False,
                init_params=None,
                seed=None,
            ):
                Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
                self.k = k
                self.max_iter = max_iter
                self.learning_rate = learning_rate
                self.gamma = gamma
                self.lambda_reg = lambda_reg
                self.variant = variant
                self.seed = seed

                self.ll = np.full(max_iter, 0)
                self.eps = 0.000000001

                # Init params if provided
                self.init_params = {} if init_params is None else init_params
                self.U = self.init_params.get("U", None)  # matrix of user factors
                self.V = self.init_params.get("V", None)  # matrix of item factors

            def fit(self, train_set, val_set=None):
                """Fit the model to observations.

                Parameters
                ----------
                train_set: :obj:`cornac.data.Dataset`, required
                    User-Item preference data as well as additional modalities.

                val_set: :obj:`cornac.data.Dataset`, optional, default: None
                    User-Item preference data for model selection purposes (e.g., early stopping).

                Returns
                -------
                self : object
                """
                Recommender.fit(self, train_set)

                from cornac.models.pmf import pmf

                if self.trainable:
                    # converting data to the triplet format (needed for cython function pmf)
                    (uid, iid, rat) = train_set.uir_tuple
                    rat = np.array(rat, dtype="float32")
                    if self.variant == "non_linear":  # need to map the ratings to [0,1]
                        if [self.min_rating, self.max_rating] != [0, 1]:
                            rat = scale(rat, 0.0, 1.0, self.min_rating, self.max_rating)
                    uid = np.array(uid, dtype="int32")
                    iid = np.array(iid, dtype="int32")

                    if self.verbose:
                        print("Learning...")

                    # use pre-trained params if exists, otherwise from constructor
                    init_params = {"U": self.U, "V": self.V}

                    if self.variant == "linear":
                        res = pmf.pmf_linear(
                            uid,
                            iid,
                            rat,
                            k=self.k,
                            n_users=self.num_users,
                            n_items=self.num_items,
                            n_ratings=len(rat),
                            n_epochs=self.max_iter,
                            lambda_reg=self.lambda_reg,
                            learning_rate=self.learning_rate,
                            gamma=self.gamma,
                            init_params=init_params,
                            verbose=self.verbose,
                            seed=self.seed,
                        )
                    elif self.variant == "non_linear":
                        res = pmf.pmf_non_linear(
                            uid,
                            iid,
                            rat,
                            k=self.k,
                            n_users=self.num_users,
                            n_items=self.num_items,
                            n_ratings=len(rat),
                            n_epochs=self.max_iter,
                            lambda_reg=self.lambda_reg,
                            learning_rate=self.learning_rate,
                            gamma=self.gamma,
                            init_params=init_params,
                            verbose=self.verbose,
                            seed=self.seed,
                        )
                    else:
                        raise ValueError('variant must be one of {"linear","non_linear"}')

                    self.U = np.asarray(res["U"])
                    self.V = np.asarray(res["V"])

                    if self.verbose:
                        print("Learning completed")

                elif self.verbose:
                    print("%s is trained already (trainable = False)" % (self.name))

                return self

            def score(self, user_idx, item_idx=None):
                """Predict the scores/ratings of a user for an item.

                Parameters
                ----------
                user_idx: int, required
                    The index of the user for whom to perform score prediction.

                item_idx: int, optional, default: None
                    The index of the item for which to perform score prediction.
                    If None, scores for all known items will be returned.

                Returns
                -------
                res : A scalar or a Numpy array
                    Relative scores that the user gives to the item or to all known items

                """
                if item_idx is None:
                    if not self.knows_user(user_idx):
                        raise ScoreException(
                            "Can't make score prediction for (user_id=%d)" % user_idx
                        )

                    known_item_scores = self.V.dot(self.U[user_idx, :])
                    return known_item_scores
                else:
                    if not self.knows_user(user_idx) or not self.knows_item(item_idx):
                        raise ScoreException(
                            "Can't make score prediction for (user_id=%d, item_id=%d)"
                            % (user_idx, item_idx)
                        )

                    user_pred = self.V[item_idx, :].dot(self.U[user_idx, :])

                    if self.variant == "non_linear":
                        user_pred = sigmoid(user_pred)
                        user_pred = scale(user_pred, self.min_rating, self.max_rating, 0.0, 1.0)

                    return user_pred


.. dropdown:: 4. Create the example file in the ``examples`` folder

    .. code-block:: python
        :caption: cornac/examples/pmf_ratio.py
    
        """Example to run Probabilistic Matrix Factorization (PMF) model with Ratio Split evaluation strategy"""

        import cornac
        from cornac.datasets import movielens
        from cornac.eval_methods import RatioSplit
        from cornac.models import PMF


        # Load the MovieLens 100K dataset
        ml_100k = movielens.load_feedback()

        # Instantiate an evaluation method.
        ratio_split = RatioSplit(
            data=ml_100k, test_size=0.2, rating_threshold=4.0, exclude_unknowns=False
        )

        # Instantiate a PMF recommender model.
        pmf = PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001)

        # Instantiate evaluation metrics.
        mae = cornac.metrics.MAE()
        rmse = cornac.metrics.RMSE()
        rec_20 = cornac.metrics.Recall(k=20)
        pre_20 = cornac.metrics.Precision(k=20)

        # Instantiate and then run an experiment.
        cornac.Experiment(
            eval_method=ratio_split,
            models=[pmf],
            metrics=[mae, rmse, rec_20, pre_20],
            user_based=True,
        ).run()

Files to edit
^^^^^^^^^^^^^

To add your model to the overall Cornac package, you need to edit the following
file:

.. code-block:: bash
    
    cornac
    |-- cornac
        |-- models
            |-- __init__.py

.. dropdown:: Edit the models/__init__.py
    
    .. code-block:: python
        :caption: cornac/cornac/models/__init__.py

        from .amr import AMR
        ... # models removed for brevity
        from .pmf import PMF # Add this line
        ... # models removed for brevity


Now you have implemented your model, it is time to test it.
In order to do so, you have to rebuild Cornac. We will discuss on how to do
this in the next section.

.. topic:: View related tutorial on Github

  View the add model guide on Github:
  https://github.com/PreferredAI/cornac/blob/master/tutorials/add_model.md

Development workflow
--------------------

Before we move on to the section of building a new model, let's take a look at
the development workflow of Cornac.

First time setup
^^^^^^^^^^^^^^^^

As Cornac contains models which uses Cython, compilation is required before
testing could be done. In order to do so, you first need to install Cython and 
run the following command:

.. code-block:: bash

    python setup.py build_ext —inplace

This will generate C++ files from Cython files, compile the C++ files, and place the compiled binary files in the necessary folders.

The main workflow of developing a new model will be to:

1. Implement model files
2. Create an example
3. Run the example

Folder structure for testing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash
    
    cornac
    |-- cornac
    |   |-- models
    |       |-- mymodel
    |       |   |-- __init__.py
    |       |   |-- recom_mymodel.py
    |       |-- requirements.txt
    |-- mymodel_example.py <-- not in the examples folder

To run the example, ensure that your current working directory is in the top
``cornac`` folder. Then, run the following command:

.. code-block:: bash

    python mymodel_example.py

Whenever a new change is done to your model files, just run the example for
testing and debugging.

Analyze results
---------------
Cornac makes it easy for you to run your model alongside other existing models.
To do so, simply add you model to the list of models in the experiment.

.. code-block:: python

    # Add new model to list
    models = [
        BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123),
        PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123),
        MyModel(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123),  
    ]

    # Define metrics to evaluate the models
    metrics = [RMSE(), Precision(k=10), Recall(k=10)]

    # run the experiment and compare the results
    cornac.Experiment(eval_method=rs, models=models, metrics=metrics, user_based=True).run()

Using additional packages
-------------------------

Cornac is built on top of the popular scientific computing libraries such as
NumPy and SciPy. It is also designed to be compatible with the popular deep learning
libraries such as PyTorch and TensorFlow.

If you are using additional packages in your model, you can add them into the
``requirements.txt`` file. This will ensure that the packages are installed

.. code-block:: bash
    
    cornac
    |-- cornac
    |   |-- models
    |       |-- ngcf
    |           |-- __init__.py
    |           |-- recom_ngcf.py
    |           |-- requirements.txt <-- Add this file
    |-- examples
        |-- ngcf_example.py

Your requirements.txt file should look like this:

.. code-block:: bash
    :caption: cornac/cornac/models/ngcf/requirements.txt

    torch>=2.0.0
    dgl>=1.1.0

This is generated by doing a ``pip freeze > requirements.txt`` command on your
environment.

Model file structure
^^^^^^^^^^^^^^^^^^^^

Your model file should have special dependencies imported only in the
fit/score functions. This is to ensure that Cornac can be built without
installing the additional packages.

For example, in the code snippet below from the ``NGCF`` model, the ``fit``
function imports the ``torch`` package. This is to ensure that the ``torch``
package is only imported when the ``fit`` function is called.

.. code-block:: python
    :caption: cornac/cornac/models/ngcf/recom_ngcf.py

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """
        Recommender.fit(self, train_set, val_set)

        if not self.trainable:
            return self

        # model setup
        import torch
        from .ngcf import Model
        from .ngcf import construct_graph

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        graph = construct_graph(train_set, self.total_users, self.total_items).to(self.device)
        model = Model(
            graph,
            self.emb_size,
            self.layer_sizes,
            self.dropout_rates,
            self.lambda_reg,
        ).to(self.device)

        # remaining codes removed for brevity

Adding a new metric
-------------------

Cornac provides a wide range of evaluation metrics for you to use. However, if
you would like to add your own metric, you can do so by extending the
:class:`~cornac.metrics.Metric` class.

.. topic:: View related tutorial on Github

  View the add metric guide on Github:
  https://github.com/PreferredAI/cornac/blob/master/tutorials/add_metric.md

Let us know!
------------
We hope you find Cornac useful for your research. Please share with us on how
you find Cornac useful, and feel free to reach out to us if you have any
questions or suggestions. If you do use Cornac in your research, we appreciate
your citation to our papers_.

.. _papers: https://github.com/PreferredAI/cornac#citation

What's Next?
------------

.. topic:: If you have already developed your model...

  Why not contribute to Cornac by including your model as part of the package?
  View :doc:`/developer/index`.

.. topic:: Keen in developing apps with Cornac?

  View a quickstart guide on how you can code and implement Cornac onto your
  application to provide recommendations for your users.

  View :doc:`/user/iamadeveloper`.






