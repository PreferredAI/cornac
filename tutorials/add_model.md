# Contributing a model to Cornac

This tutorial describes how to integrate a recommender model into Cornac. We assume that you have already forked the Cornac repository to your own account.

## Directory & file structure

For convenience assume that the model of interest is PMF. At this point it is useful to recall that Cornac follows the [PEP 8](https://www.python.org/dev/peps/pep-0008/) coding conventions. One good way to get started is to update the directory and file structure as follows.
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

## Creating a model in 4 steps

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
        train_set: :obj:`cornac.data.MultimodalTrainSet`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.MultimodalTestSet`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """

    def score(self, user_id, item_id=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_id: int, required
            The index of the user for whom to perform score prediction.

        item_id: int, optional, default: None
            The index of the item for that to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items
        """
``` 
Every model extends the generic class `Recommender`. All you need to do is to redefine the functions listed in the above `recom_pmf.py` file.

The `fit()` function should contain or make a call to the necessary codes to fit your model to the training data, while the `score()` function specifies how to perform predictions using your model.
 
#### More on the score function 
When you perform rating prediction evaluation (e.g., in terms of RMSE) Cornac calls the `Recommender.rate()` function, which in turn calls the `score()` function and clips the predicted values to lie in the same range as the original ratings. Analogously, Cornac calls the `Recommender.rank()` function, which returns a ranking of items according to their scores as predicted by the function `score()`. We recommend you to take a look at the `rate()` and `rank()` functions inside the `Recommender` class, as you may wish to redefine them as well, if the default settings do not meet your needs. 

### 2. Making your recommender available to Cornac
As you may have already noticed, Cornac treats each recommender model as a separate module, so as to reduce coupling and ease the contribution of new models. This is the reason why you need to include a `pmf/__init__.py` file:
```python
from .recom_pmf import PMF
```

One last step is required to make you recommender model available to Cornac, which is to update `../models/__init__.py` by adding the following line:
```python
from .pmf import PMF
```

### 3. Indicating dependencies
The file `requirements.txt` is optional as opposed to the others, and it is only needed if your implementation relies on some external libraries (e.g., TensorFlow, PyTorch) which are not Cornac dependencies. In this case you should include this file and indicate which versions of the libraries of interest are required. Here is a sample of a `requirements.txt` file:
```
tensorflow>=1.10.0
```

### 4. Adding documentation

Cornac uses [Sphinx](http://www.sphinx-doc.org/en/master/) to generate documentation. All you need to do is to fulfill the Parameters section inside `recom_pmf.py`. If you are not familiar with Sphinx syntax conventions, you can just take a look at some of the existing Cornac models and follow them.

Once you have documented the parameters of your recommender, the next step, to integrate your model to the docs, is to update `cornac/docs/source/models.rst` by adding the following code:
```
Probabilitic Matrix Factorization (PMF)
-----------------------------------------
.. automodule:: cornac.models.pmf.recom_pmf
   :members:
```
The last step is to add your algorithm into the Cornac's list of models table inside `./README.md`, for ease of reference purposes.
At this point you are done, and ready to create a pull request, congratulations!

### Including an example (optional)
We highly encourage you to add an example on how to fit your model to some dataset and report the obtained results, see for instance `./cornac/examples/pcrl_example.py`. All examples, should be added to the `examples/` directory, listed in `examples/README.md` under section: Examples by Algorithm, and referenced in models table in `./README.md`. 


## Summary

In short, contributing a recommender model to Cornac involves,

- Creating new files
    - [x] ./cornac/models/model_name/\_\_init__.py
    - [x] ./cornac/models/model_name/recom_model_name.py
    - [x] ./cornac/models/model_name/requirement.txt (optional)
- Implementing two functions
     - [x] Recommender.fit()
     - [x] Recommender.score()
- Updating existing files
     - [x] ./cornac/models/\_\_init__.py
     - [x] ./docs/source/models.rst
     - [x] ./README.md
     
As a concrete example, you can take a look at one of the Cornac's model implementations inside `./cornac/models/`, as this may help you save time.   
    
## Using Cython and C/C++ (optional)

- If you are interested in using [Cython](https://cython.org/) to implement the algorithmic part of your model, then the Cornac's PMF implementation `./cornac/models/pmf` is a good example to look at.
- If you already have a C/C++ implementation of your model, then you can use Cython to wrap your code for Python. In this case, you may consider `./cornac/models/hpf` or `./cornac/models/c2pf`, which are based on C++ implementations.

Note that, in this situation you will need to declare your Cython extension inside  `./setup.py`.
  