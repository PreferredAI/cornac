# Cornac

**Cornac** is python recommender system library for **easy**, **effective** and **efficient** experiments. Cornac is **simple** and **handy**. It is designed from the ground-up to faithfully reflect the standard steps taken by researchers to implement and evaluate personalized recommendation models.

### Quick Links

[Website](https://cornac.preferred.ai/) |
[Documentation](https://cornac.readthedocs.io/en/latest/index.html) |
[Preferred.AI](https://preferred.ai/)

[![Build Status](https://www.travis-ci.org/PreferredAI/cornac.svg?branch=master)](https://www.travis-ci.org/PreferredAI/cornac)
[![Documentation Status](https://readthedocs.org/projects/cornac/badge/?version=latest)](https://cornac.readthedocs.io/en/latest/?badge=latest)
[![Codecov](https://codecov.io/gh/PreferredAI/cornac/branch/master/graph/badge.svg)](https://codecov.io/gh/PreferredAI/cornac)
[![Python Versions](https://img.shields.io/badge/python-3.6-blue.svg)](https://cornac.preferred.ai/)
[![GitHub Version](https://badge.fury.io/gh/PreferredAI%2FCornac.svg)](https://badge.fury.io/gh/PreferredAI%2FCornac)
[![PyPI](https://badge.fury.io/py/cornac.svg)](https://badge.fury.io/py/cornac)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellowgreen.svg)](https://opensource.org/licenses/Apache-2.0)


## Installation

Currently, we are supporting Python 3 (version 3.6 is recommended). There are two ways to install Cornac:

- **Install Cornac from PyPI:**

```sh
pip3 install cornac
```

- **Install Cornac from the GitHub source (for latest updates):**

```sh
pip3 install cython
git clone https://github.com/PreferredAI/cornac.git
cd cornac
python3 setup.py install
```

**Note** 

Some installed dependencies are CPU versions. If you want to utilize your GPU, you might consider:

- [TensorFlow installation instructions](https://www.tensorflow.org/install/).
- [PyTorch installation instructions](https://pytorch.org/get-started/locally/).
- [cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/) (for Nvidia GPUs).

## Getting started: your first Cornac experiment

This example will show you how to run your very first experiment using Cornac. 

Load the [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) dataset (will be automatically downloaded if not cached).
```python
from cornac.datasets import MovieLens100K

ml_100k = MovieLens100K.load_data()
```

Instantiate an evaluation strategy.
```python
from cornac.eval_strategies import RatioSplit

ratio_split = RatioSplit(data=ml_100k, test_size=0.2, rating_threshold=4.0, exclude_unknowns=False)
```

Instantiate models that we want to evaluate. Here we use `Probabilistic Matrix Factorization (PMF)` as an example.
```python
pmf = cornac.models.PMF(k=10, max_iter=100, learning_rate=0.001, lamda=0.001)
```

Instantiate evaluation metrics.
```python
mae = cornac.metrics.MAE()
rmse = cornac.metrics.RMSE()
rec_20 = cornac.metrics.Recall(k=20)
pre_20 = cornac.metrics.Precision(k=20)
```

Instantiate and then run an experiment.
```python
exp = cornac.Experiment(eval_strategy=ratio_split,
                        models=[pmf],
                        metrics=[mae, rmse, rec_20, pre_20],
                        user_based=True)
exp.run()
```

**Output**

```
          MAE      RMSE  Recall@20  Precision@20
PMF  0.760277  0.919413   0.081803        0.0462
```

For more details, please take a look at our [examples](examples).

## License

[Apache License 2.0](LICENSE)
