# Contributing a model to Cornac

This tutorial describes how to integrate a recommender model into Cornac. We assume that you have already forked the Cornac repository to your own account.

## Directory & file structure

For convenience assume that the model of interest is PMF. At this point it is useful to recall that Cornac follows the [PEP 8](https://www.python.org/dev/peps/pep-0008/) coding conventions. See [CONTRIBUTING.md](https://github.com/PreferredAI/cornac/blob/master/CONTRIBUTING.md) for how to check and format your code with Ruff. One good way to get started is to update the directory and file structure as follows.
```
cornac    
│
└───cornac
│   │
│   └───models
│       │   __init__.py
│       │   recommender.py
│       │
│       └───pmf
│           │   __init__.py
│           │   recom_pmf.py
│           │   requirements.txt
```
Note that you only need to add the `pmf` branch as the rest of the structure is already in place.

## Creating a model in 5 steps

### 1. Extending the Recommender class

The main file is `recom_pmf.py`, which will contain your recommender algorithm related codes. The name of this file should always start with the prefix `recom_`. Here is the minimal structure of such file:
```python
from ..recommender import Recommender

class PMF(Recommender):
    """Probabilistic Matrix Factorization.

    Parameters
    ----------

    References
    ----------
    """

    def __init__(self, name="PMF", trainable=True, verbose=False, ...):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)

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
``` 
Every model extends the generic class `Recommender`. All you need to do is to redefine the functions listed in the above `recom_pmf.py` file.

The `fit()` function should contain or make a call to the necessary codes to fit your model to the training data, while the `score()` function specifies how to perform predictions using your model. Start `fit()` with `Recommender.fit(self, train_set, val_set)`, which sets up the information your model needs later, such as `num_users`, `num_items`, `uid_map` and `iid_map`.
 
#### More on the score function 
When you perform rating prediction evaluation (e.g., in terms of RMSE) Cornac calls the `Recommender.rate()` function, which in turn calls the `score()` function and clips the predicted values to lie in the same range as the original ratings. Analogously, Cornac calls the `Recommender.rank()` function, which returns a ranking of items according to their scores as predicted by the function `score()`. We recommend you to take a look at the `rate()` and `rank()` functions inside the `Recommender` class, as you may wish to redefine them as well, if the default settings do not meet your needs. 

If your model cannot score a user or an item, for example one that was not in the training data, raise `ScoreException` from `cornac.exception` in `score()`. `rate()` and `rank()` catch it and fall back to a default score. The `score()` function of [PMF](https://github.com/PreferredAI/cornac/blob/master/cornac/models/pmf/recom_pmf.py) shows how.

#### Next-item and next-basket models
If your model recommends the next item of a session or the next basket of a user, extend `NextItemRecommender` or `NextBasketRecommender` instead. Both are in `cornac/models/recommender.py` and extend `Recommender`. Their `score()` function also receives the user's history, as `score(self, user_idx, history_items, **kwargs)` or `score(self, user_idx, history_baskets, **kwargs)`, and returns the scores of all known items. [GRU4Rec](https://github.com/PreferredAI/cornac/blob/master/cornac/models/gru4rec/recom_gru4rec.py) and [TIFUKNN](https://github.com/PreferredAI/cornac/blob/master/cornac/models/tifuknn/recom_tifuknn.py) are good examples to start from.

### 2. Making your recommender available to Cornac
As you may have already noticed, Cornac treats each recommender model as a separate module, so as to reduce coupling and ease the contribution of new models. This is the reason why you need to include a `pmf/__init__.py` file:
```python
from .recom_pmf import PMF
```
The `.recom_pmf` coincides with the name of the file that contains the model, and `PMF` coincides with the name of the class in the `recom_pmf.py` file.

One last step is required to make your recommender model available to Cornac, which is to update `cornac/models/__init__.py` by adding the following line:
```python
from .pmf import PMF
```

### 3. Indicating dependencies
The file `requirements.txt` is optional as opposed to the others, and it is only needed if your implementation relies on some external libraries (e.g., TensorFlow, PyTorch) which are not Cornac dependencies. In this case you should include this file and indicate which versions of the libraries of interest are required. Here is the `requirements.txt` file of the NGCF model:
```
torch>=2.0.0
dgl>=1.1.0
```

List only the packages your model needs. The file sits next to your model files:

```
cornac
|-- cornac
|   |-- models
|       |-- ngcf
|           |-- __init__.py
|           |-- ngcf.py
|           |-- recom_ngcf.py
|           |-- requirements.txt <-- Add this file
|-- examples
    |-- ngcf_example.py
```

#### Importing optional dependencies

Since `import cornac` loads the `recom_<name>.py` file of every model, your
model file imports the extra packages it needs inside `fit()`, `score()` or
other methods rather than at the top. This is to ensure that Cornac can be
imported without installing the additional packages. Backend modules that are
only imported from inside those methods, such as `ngcf.py` above, can import
them at the top as usual.

For example, in the code snippet below from the [NGCF](https://github.com/PreferredAI/cornac/blob/master/cornac/models/ngcf/recom_ngcf.py) model, the `fit`
function imports the `torch` package. This is to ensure that the `torch`
package is only imported when the `fit` function is called.

```python
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
```

### 4. Adding documentation

Cornac uses [Sphinx](http://www.sphinx-doc.org/en/master/) to generate documentation. All you need to do is to fill in the `Parameters` and `References` sections of the docstrings inside `recom_pmf.py`, which follow the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) style. If you are not familiar with numpydoc, you can just take a look at some of the existing Cornac models and follow them.

Once you have documented the parameters of your recommender, the next step, to integrate your model to the docs, is to update `docs/source/api_ref/models.rst` by adding the following code:
```
Probabilistic Matrix Factorization (PMF)
-----------------------------------------
.. automodule:: cornac.models.pmf.recom_pmf
   :members:
```
The last step is to add your algorithm into the Cornac's list of models table inside `./README.md`; the Models Available page of the documentation is built from this table.
At this point your model is part of Cornac, congratulations! Before you create a pull request, add an example as described next.

### 5. Including an example
Please add an example on how to fit your model to some dataset and report the obtained results, see for instance `examples/pcrl_example.py`. All examples, should be added to the `examples/` directory, listed in `examples/README.md` in the section that fits the model, and referenced in models table in `./README.md`. 


## Summary

In short, contributing a recommender model to Cornac involves,

- Creating new files
    - [ ] ./cornac/models/model_name/\_\_init__.py
    - [ ] ./cornac/models/model_name/recom_model_name.py
    - [ ] ./cornac/models/model_name/requirements.txt (optional)
- Implementing two functions
     - [ ] Recommender.fit()
     - [ ] Recommender.score()
- Updating existing files
     - [ ] ./cornac/models/\_\_init__.py
     - [ ] ./docs/source/api_ref/models.rst
     - [ ] ./README.md
- Adding an example
     - [ ] ./examples/model_name_example.py, listed in ./examples/README.md
     
As a concrete example, you can take a look at one of the Cornac's model implementations inside `./cornac/models/`, as this may help you save time. For instance, [recom_pmf.py](https://github.com/PreferredAI/cornac/blob/master/cornac/models/pmf/recom_pmf.py) is the complete PMF model, and [pmf_ratio.py](https://github.com/PreferredAI/cornac/blob/master/examples/pmf_ratio.py) runs it on MovieLens 100K.   
    
## Using Cython and C/C++ (optional)

- If you are interested in using [Cython](https://cython.org/) to implement the algorithmic part of your model, then the Cornac's PMF implementation `./cornac/models/pmf` is a good example to look at.
- If you already have a C/C++ implementation of your model, then you can use Cython to wrap your code for Python. In this case, you may consider `./cornac/models/hpf` or `./cornac/models/c2pf`, which are based on C++ implementations.

Note that, in this situation you will need to declare your Cython extension inside  `./setup.py`.
  