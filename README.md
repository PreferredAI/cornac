# Cornac

**Cornac** is python recommender system library for **easy**, **effective** and **efficient** experiments. Cornac is **simple** and **handy**. It is designed from the ground-up to faithfully reflect the standard steps taken by researchers to implement and evaluate personalized recommendation models.

### Quick Links

[Website](https://cornac.preferred.ai/) |
[Documentation](https://cornac.readthedocs.io/en/latest/index.html) |
[Preferred.AI](https://preferred.ai/)

[![TravisCI](https://img.shields.io/travis/PreferredAI/cornac/master.svg?logo=travis)](https://www.travis-ci.org/PreferredAI/cornac)
[![CircleCI](https://img.shields.io/circleci/project/github/PreferredAI/cornac/master.svg?logo=circleci)](https://circleci.com/gh/PreferredAI/cornac)
[![AppVeyor](https://ci.appveyor.com/api/projects/status/0yq4td1xg4kkhdwu?svg=true)](https://ci.appveyor.com/project/tqtg/cornac)
[![Codecov](https://img.shields.io/codecov/c/github/PreferredAI/cornac/master.svg?logo=codecov)](https://codecov.io/gh/PreferredAI/cornac)
[![Docs](https://img.shields.io/readthedocs/cornac/latest.svg)](https://cornac.readthedocs.io/en/latest)
<br />
[![Release](https://img.shields.io/github/release-pre/PreferredAI/cornac.svg)](https://github.com/PreferredAI/cornac/releases)
[![PyPI](https://img.shields.io/pypi/v/cornac.svg)](https://pypi.org/project/cornac/)
[![Conda](https://img.shields.io/conda/v/qttruong/cornac.svg?label=anaconda)](https://anaconda.org/qttruong/cornac)
[![Conda Recipe](https://img.shields.io/badge/conda-recipe-green.svg)](https://github.com/tqtg/cornac-feedstock)
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
conda install cornac -c qttruong -c pytorch
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

Some of the algorithms use `OpenMP` to speed up training with multithreading. For OSX users, in order to run those algorithms efficiently, you might need to install `gcc` from Homebrew to have an OpenMP compiler and install Cornac from source:

```sh
brew install gcc | brew link gcc
```

If you want to utilize your GPUs, you might consider:

- [TensorFlow installation instructions](https://www.tensorflow.org/install/).
- [PyTorch installation instructions](https://pytorch.org/get-started/locally/).
- [cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/) (for Nvidia GPUs).

## Getting started: your first Cornac experiment

![](exp-flow.jpg)
<p align="center"><i>Flow of an Experiment in Cornac</i></p>

This example will show you how to run your very first experiment.

- Load the [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) dataset (will be automatically downloaded if not cached).
```python
from cornac.datasets import movielens

ml_100k = movielens.load_100k()
```

- Instantiate an evaluation method. Here we split the data based on ratio.
```python
from cornac.eval_methods import RatioSplit

ratio_split = RatioSplit(data=ml_100k, test_size=0.2, rating_threshold=4.0, exclude_unknowns=False)
```

- Instantiate models that we want to evaluate. Here we use `Probabilistic Matrix Factorization (PMF)` as an example.
```python
import cornac

pmf = cornac.models.PMF(k=10, max_iter=100, learning_rate=0.001, lamda=0.001)
```

- Instantiate evaluation metrics.
```python
mae = cornac.metrics.MAE()
rmse = cornac.metrics.RMSE()
rec_20 = cornac.metrics.Recall(k=20)
pre_20 = cornac.metrics.Precision(k=20)
```

- Instantiate and then run an experiment.
```python
exp = cornac.Experiment(eval_method=ratio_split,
                        models=[pmf],
                        metrics=[mae, rmse, rec_20, pre_20],
                        user_based=True)
exp.run()
```

**Output:**

```
          MAE      RMSE  Recall@20  Precision@20
PMF  0.760277  0.919413   0.081803        0.0462
```

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
