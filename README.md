# Cornac

**Cornac** is python recommender system library for **easy**, **effective** and **efficient** experiments. Cornac is **simple** and **handy**. It is designed from the ground-up to faithfully reflect the standard steps taken by researchers to implement and evaluate personalized recommendation models.

### Quick Links

[Website](https://cornac.preferred.ai/) |
[Documentation](https://cornac.readthedocs.io/en/latest/index.html) |
[Models](https://github.com/PreferredAI/cornac/tree/master/cornac/models#models) |
[Examples](https://github.com/PreferredAI/cornac/tree/master/examples#cornac-examples-directory) |
[Preferred.AI](https://preferred.ai/)

[![TravisCI](https://img.shields.io/travis/PreferredAI/cornac/master.svg?logo=travis)](https://www.travis-ci.org/PreferredAI/cornac)
[![CircleCI](https://img.shields.io/circleci/project/github/PreferredAI/cornac/master.svg?logo=circleci)](https://circleci.com/gh/PreferredAI/cornac)
[![AppVeyor](https://ci.appveyor.com/api/projects/status/0yq4td1xg4kkhdwu?svg=true)](https://ci.appveyor.com/project/tqtg/cornac)
[![Codecov](https://img.shields.io/codecov/c/github/PreferredAI/cornac/master.svg?logo=codecov)](https://codecov.io/gh/PreferredAI/cornac)
[![Docs](https://img.shields.io/readthedocs/cornac/latest.svg)](https://cornac.readthedocs.io/en/latest)
<br />
[![Release](https://img.shields.io/github/release-pre/PreferredAI/cornac.svg)](https://github.com/PreferredAI/cornac/releases)
[![PyPI](https://img.shields.io/pypi/v/cornac.svg)](https://pypi.org/project/cornac/)
[![Conda](https://img.shields.io/conda/vn/conda-forge/cornac.svg)](https://anaconda.org/conda-forge/cornac)
[![Conda Recipe](https://img.shields.io/badge/recipe-cornac-green.svg)](https://github.com/conda-forge/cornac-feedstock)
<br />
[![Python](https://img.shields.io/pypi/pyversions/cornac.svg)](https://cornac.preferred.ai/)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)


## Installation

Currently, we are supporting Python 3 (version 3.6 is recommended). There are several ways to install Cornac:

  - **From PyPI (you may need a C++ compiler):**

```sh
pip3 install cornac
```

  - **From Anaconda:**

```sh
conda install cornac -c conda-forge
```

  - **From the GitHub source (for latest updates):**

```sh
pip3 install Cython
git clone https://github.com/PreferredAI/cornac.git
cd cornac
python3 setup.py install
```

**Note:** 

Additional dependencies required by models are listed [here](cornac/models/README.md).

Some of the algorithms use `OpenMP` to support multi-threading. For OSX users, in order to run those algorithms efficiently, you might need to install `gcc` from Homebrew to have an OpenMP compiler:

```sh
brew install gcc | brew link gcc
```

If you want to utilize your GPUs, you might consider:

  - [TensorFlow installation instructions](https://www.tensorflow.org/install/).
  - [PyTorch installation instructions](https://pytorch.org/get-started/locally/).
  - [cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/) (for Nvidia GPUs).

## Getting started: your first Cornac experiment

![](flow.jpg)
<p align="center"><i>Flow of an Experiment in Cornac</i></p>

Load the built-in [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) dataset (will be downloaded if not cached):

```python
from cornac.datasets import movielens

ml_100k = movielens.load_100k()
```

Split the data based on ratio:

```python
from cornac.eval_methods import RatioSplit

ratio_split = RatioSplit(data=ml_100k, test_size=0.2, rating_threshold=4.0, seed=123)
```

Here we are comparing `Biased MF`, `PMF`, and `BPR`:
  
```python
from cornac.models import MF, PMF, BPR

mf = MF(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02, use_bias=True)
pmf = PMF(k=10, max_iter=100, learning_rate=0.001, lamda=0.001)
bpr = BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01)
```

Define metrics used to evaluate the models:
  
```python
mae = cornac.metrics.MAE()
rmse = cornac.metrics.RMSE()
rec_20 = cornac.metrics.Recall(k=20)
ndcg_20 = cornac.metrics.NDCG(k=20)
auc = cornac.metrics.AUC()
```

Put everything together into an experiment and run it:
  
```python
from cornac import Experiment

exp = Experiment(eval_method=ratio_split,
                 models=[mf, pmf, bpr],
                 metrics=[mae, rmse, rec_20, ndcg_20, auc],
                 user_based=True)
exp.run()
```

**Output:**

|     |    MAE |   RMSE | Recall@20 | NDCG@20 |    AUC | Train (s) | Test (s) |
| --- | -----: | -----: | --------: | ------: | -----: | --------: | -------: |
| [MF](cornac/models/mf)  | 0.7441 | 0.9007 |    0.0622 |  0.0534 | 0.2952 |    0.0791 |   1.3119 |
| [PMF](cornac/models/pmf) | 0.7490 | 0.9093 |    0.0831 |  0.0683 | 0.4660 |    8.7645 |   2.1569 |
| [BPR](cornac/models/bpr) | N/A | N/A |    0.1449 |  0.1124 | 0.8750 |    0.8898 |   1.3769 |

For more details, please take a look at our [examples](examples).

## Models

The recommender models supported by Cornac are listed [here](cornac/models/README.md). Why don't you join us to lengthen the list?

## Support

Your contributions at any level of the library are welcome. If you intend to contribute, please:
  - Fork the Cornac repository to your own account.
  - Make changes and create pull requests.

You can also post bug reports and feature requests in [GitHub issues](https://github.com/PreferredAI/cornac/issues).

## License

[Apache License 2.0](LICENSE)
