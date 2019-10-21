# Contributing an evaluation metric to Cornac

This tutorial shows how to add an evaluation measure to Cornac. As with any contribution, we assume that we have already forked the Cornac repository.

Some evaluation measures may have different appellations in the literature. So it is worth it to make sure that the metric of interest is not already available in Cornac, before considering any contribution. Cornac's list of measures is [here](https://cornac.readthedocs.io/en/latest/metrics.html).

## Adding a metric

Cornac's evaluation measures are inside the ``metrics`` directory, and they are organized into two main categories, namely _ranking_ (e.g., Recall) and _rating_ (e.g., MAE) metrics as illustrated below.   
```
cornac    
│
└───cornac
│       │
│       └───metrics
│           │   __init__.py
│           │   ranking.py
│           │   rating.py 
```
It is important to follow the above categorization to avoid confusions. For the rest of the tutorial let's use MAE (Mean Absolute Error) as a running example. Note that we could have picked a ranking metric as well, e.g., Normalized Discount Cumulative Gain (NDCG), and follow the same process as below.  

### 1. Extending the RatingMetric class

The starting point is to create a class called ``MAE``, which extends the generic class ``RatingMetric`` implemented inside `metric/rating.py`. The minimal structure of our new class is as follows:  
```python
class MAE(RatingMetric):
    """Mean Absolute Error.

    Attributes
    ----------
    name: string, value: 'MAE'
        Name of the measure.

    """

    def __init__(self):
        RatingMetric.__init__(self, name='MAE')

    def compute(self, **kwargs):
        raise NotImplementedError()
```

Next, we need to implement the ``compute()`` function, as well as provide the necessary comments that will be automatically transformed into documentation by [Sphinx](http://www.sphinx-doc.org/en/master/). Here is an implementation example:
```python
def compute(self, gt_ratings, pd_ratings, weights=None, **kwargs):
    """Compute Mean Absolute Error.

    Parameters
    ----------
    gt_ratings: Numpy array
        Ground-truth rating values.

    pd_ratings: Numpy array
        Predicted rating values.

    weights: Numpy array, optional, default: None
        Weights for rating values.

    **kwargs: For compatibility

    Returns
    -------
    mae: A scalar.
        Mean Absolute Error.

    """
    mae = np.average(np.abs(gt_ratings - pd_ratings), axis=0, weights=weights)
    return mae
```
As one may have guessed, the code of the ``MAE`` class should be inside ``metrics/rating.py``.  

### 2. Making our metric available to Cornac and its Docs

To make the MAE class accessible in Cornac, we update the ``metrics/__init__.py`` file by adding the following line:
```python
from .rating import MAE
``` 

To further automatically generate the documentation for our measure, we update `cornac/docs/source/metrics.rst` by adding the following code under the _Rating_ section.
```
Mean Absolute Error (MAE)
--------------------------
.. autoclass:: MAE
```

### 3. Adding unit tests

Unit tests are required for metrics to ensure implementation quality. All tests are grouped into the ``tests`` folder, in the root of the Cornac repository. In our case we are interested in updating the rating-metrics' test file: `tests/cornac/metrics/test_rating.py`. The latter contains a class called `TestRating`, and we can simply add a function to this class to test our MAE code, as in the example below.
```python
def test_mae(self):
    mae = MAE()

    self.assertEqual(mae.type, 'rating')
    self.assertEqual(mae.name, 'MAE')

    self.assertEqual(0, mae.compute(np.asarray([0]), np.asarray([0])))
    self.assertEqual(1, mae.compute(np.asarray([0, 1]), np.asarray([1, 0])))
    self.assertEqual(2, mae.compute(np.asarray([0, 1]), np.asarray([2, 3]), np.asarray([1, 3])))
``` 
At this stage we are done and ready to create a pull request.