# Cornac

**Cornac** is a comparative framework for multimodal recommender systems. It focuses on making it **convenient** to work with models leveraging **auxiliary data** (e.g., item descriptive text and image, social network, etc). **Cornac** enables **fast** experiments and **straightforward** implementations of new models. It is **highly compatible** with existing machine learning libraries (e.g., TensorFlow, PyTorch).

### Quick Links

[Website](https://cornac.preferred.ai/) |
[Documentation](https://cornac.readthedocs.io/en/latest/index.html) |
[Tutorials](tutorials#tutorials) |
[Examples](https://github.com/PreferredAI/cornac/tree/master/examples#cornac-examples-directory) |
[Models](#models) |
[Datasets](./cornac/datasets/README.md#datasets) |
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

Currently, we are supporting Python 3. There are several ways to install Cornac:

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

Additional dependencies required by models are listed [here](README.md#Models).

Some algorithm implementations use `OpenMP` to support multi-threading. For OSX users, in order to run those algorithms efficiently, you might need to install `gcc` from Homebrew to have an OpenMP compiler:

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
import cornac

ml_100k = cornac.datasets.movielens.load_feedback(variant="100K")
```

Split the data based on ratio:

```python
rs = cornac.eval_methods.RatioSplit(data=ml_100k, test_size=0.2, rating_threshold=4.0, seed=123)
```

Here we are comparing `Biased MF`, `PMF`, and `BPR`:
  
```python
mf = cornac.models.MF(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02, use_bias=True, seed=123)
pmf = cornac.models.PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123)
bpr = cornac.models.BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123)
```

Define metrics used to evaluate the models:
  
```python
mae = cornac.metrics.MAE()
rmse = cornac.metrics.RMSE()
recall = cornac.metrics.Recall(k=[10, 20])
ndcg = cornac.metrics.NDCG(k=[10, 20])
auc = cornac.metrics.AUC()

```

Put everything together into an experiment and run it:
  
```python
cornac.Experiment(eval_method=rs,
                  models=[mf, pmf, bpr],
                  metrics=[mae, rmse, recall, ndcg, auc],
                  user_based=True).run()
```

**Output:**

|                          |    MAE |   RMSE |    AUC | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |  Train (s) | Test (s) |
| ------------------------ | -----: | -----: | -----: | ------: | ------: | --------: | --------: | ---------: | -------: |
| [MF](cornac/models/mf)   | 0.7430 | 0.8998 | 0.7445 |  0.0479 |  0.0556 |    0.0352 |    0.0654 |       0.13 |     1.57 |
| [PMF](cornac/models/pmf) | 0.7534 | 0.9138 | 0.7744 |  0.0617 |  0.0719 |    0.0479 |    0.0880 |       2.18 |     1.64 |
| [BPR](cornac/models/bpr) |    N/A |    N/A | 0.8695 |  0.0975 |  0.1129 |    0.0891 |    0.1449 |       3.74 |     1.49 |

For more details, please take a look at our [examples](examples).

## Models

The recommender models supported by Cornac are listed below. Why don't you join us to lengthen the list?

| Year | Model and paper | Additional dependencies | Examples |
| :---: | --- | :---: | :---: |
| 2018 | [Collaborative Context Poisson Factorization (C2PF)](cornac/models/c2pf), [paper](https://www.ijcai.org/proceedings/2018/0370.pdf) | N/A | [c2pf_exp.py](examples/c2pf_example.py)
|      | [Multi-Task Explainable Recommendation (MTER)](cornac/models/mter), [paper](https://arxiv.org/pdf/1806.03568.pdf) | [requirements.txt](cornac/models/mter/requirements.txt) | [mter_exp.py](examples/mter_example.py)
|      | [Probabilistic Collaborative Representation Learning (PCRL)](cornac/models/pcrl), [paper](http://www.hadylauw.com/publications/uai18.pdf) | [requirements.txt](cornac/models/pcrl/requirements.txt) | [pcrl_exp.py](examples/pcrl_example.py)
|      | [Variational Autoencoder for Collaborative Filtering (VAECF)](cornac/models/vaecf), [paper](https://arxiv.org/pdf/1802.05814.pdf) | [requirements.txt](cornac/models/vaecf/requirements.txt) | [vaecf_citeulike.py](examples/vaecf_citeulike.py)
| 2017 | [Collaborative Variational Autoencoder (CVAE)](cornac/models/cvae), [paper](http://eelxpeng.github.io/assets/paper/Collaborative_Variational_Autoencoder.pdf) | [requirements.txt](cornac/models/cvae/requirements.txt) | [cvae_exp.py](examples/cvae_example.py)
|      | [Generalized Matrix Factorization (GMF)](cornac/models/ncf), [paper](https://arxiv.org/pdf/1708.05031.pdf) | [requirements.txt](cornac/models/ncf/requirements.txt) | [ncf_exp.py](examples/ncf_example.py)
|      | [Indexable Bayesian Personalized Ranking (IBPR)](cornac/models/ibpr), [paper](http://www.hadylauw.com/publications/cikm17a.pdf) | [requirements.txt](cornac/models/ibpr/requirements.txt) | [ibpr_exp.py](examples/ibpr_example.py)
|      | [Matrix Co-Factorization (MCF)](cornac/models/mcf), [paper](http://papers.www2017.com.au.s3-website-ap-southeast-2.amazonaws.com/proceedings/p1113.pdf) | N/A | [mcf_office.py](examples/mcf_office.py)
|      | [Multi-Layer Perceptron (MLP)](cornac/models/ncf), [paper](https://arxiv.org/pdf/1708.05031.pdf) | [requirements.txt](cornac/models/ncf/requirements.txt) | [ncf_exp.py](examples/ncf_example.py)
|      | [Neural Matrix Factorization (NeuMF) / Neural Collaborative Filtering (NCF)](cornac/models/ncf), [paper](https://arxiv.org/pdf/1708.05031.pdf) | [requirements.txt](cornac/models/ncf/requirements.txt) | [ncf_exp.py](examples/ncf_example.py)
|      | [Online Indexable Bayesian Personalized Ranking (Online IBPR)](cornac/models/online_ibpr), [paper](http://www.hadylauw.com/publications/cikm17a.pdf) | [requirements.txt](cornac/models/online_ibpr/requirements.txt) |
|      | [Visual Matrix Factorization (VMF)](cornac/models/vmf), [paper](http://papers.www2017.com.au.s3-website-ap-southeast-2.amazonaws.com/proceedings/p1113.pdf) | [requirements.txt](cornac/models/vmf/requirements.txt) | [vmf_clothing.py](examples/vmf_clothing.py)
| 2016 | [Collaborative Deep Ranking (CDR)](cornac/models/cdr), [paper](http://inpluslab.com/chenliang/homepagefiles/paper/hao-pakdd2016.pdf) | [requirements.txt](cornac/models/cdr/requirements.txt) | [cdr_exp.py](examples/cdr_example.py)
|      | [Collaborative Ordinal Embedding (COE)](cornac/models/coe), [paper](http://www.hadylauw.com/publications/sdm16.pdf) | [requirements.txt](cornac/models/coe/requirements.txt) |
|      | [Convolutional Matrix Factorization (ConvMF)](cornac/models/conv_mf), [paper](http://uclab.khu.ac.kr/resources/publication/C_351.pdf) | [requirements.txt](cornac/models/conv_mf/requirements.txt) | [convmf_exp.py](examples/conv_mf_example.py)
|      | [Spherical K-means (SKM)](cornac/models/skm), [paper](https://www.sciencedirect.com/science/article/pii/S092523121501509X) | N/A |
|      | [Visual Bayesian Personalized Ranking (VBPR)](cornac/models/vbpr), [paper](https://arxiv.org/pdf/1510.01784.pdf) | [requirements.txt](cornac/models/vbpr/requirements.txt) | [vbpr_tradesy.py](examples/vbpr_tradesy.py)
| 2015 | [Collaborative Deep Learning (CDL)](cornac/models/cdl), [paper](https://arxiv.org/pdf/1409.2944.pdf) | [requirements.txt](cornac/models/cdl/requirements.txt) | [cdl_exp.py](examples/cdl_example.py)
|      | [Hierarchical Poisson Factorization (HPF)](cornac/models/hpf), [paper](http://jakehofman.com/inprint/poisson_recs.pdf) | N/A |
| 2014 | [Explicit Factor Model (EFM)](cornac/models/efm), [paper](http://yongfeng.me/attach/efm-zhang.pdf) | N/A | [efm_exp.py](examples/efm_example.py)
|      | [Social Bayesian Personalized Ranking (SBPR)](cornac/models/sbpr), [paper](https://cseweb.ucsd.edu/~jmcauley/pdfs/cikm14.pdf) | N/A | [sbpr_epinions.py](examples/sbpr_epinions.py)
| 2013 | [Hidden Factors and Hidden Topics (HFT)](cornac/models/hft), [paper](https://cs.stanford.edu/people/jure/pubs/reviews-recsys13.pdf) | N/A | [hft_exp.py](examples/hft_example.py)
| 2012 | [Weighted Bayesian Personalized Ranking (WBPR)](cornac/models/bpr), [paper](http://proceedings.mlr.press/v18/gantner12a/gantner12a.pdf) | N/A | [bpr_netflix.py](examples/bpr_netflix.py)
| 2011 | [Collaborative Topic Modeling (CTR)](cornac/models/ctr), [paper](http://www.cs.columbia.edu/~blei/papers/WangBlei2011.pdf) | N/A | [ctr_citeulike.py](examples/ctr_example_citeulike.py)
| Earlier | [Baseline Only](cornac/models/baseline_only), [paper](http://courses.ischool.berkeley.edu/i290-dm/s11/SECURE/a1-koren.pdf) | N/A | [svd_exp.py](examples/svd_example.py)
|      | [Bayesian Personalized Ranking (BPR)](cornac/models/bpr), [paper](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf) | N/A | [bpr_netflix.py](examples/bpr_netflix.py)
|      | [Global Average (GlobalAvg)](cornac/models/global_avg), [paper](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) | N/A | [biased_mf.py](examples/biased_mf.py)
|      | [Matrix Factorization (MF)](cornac/models/mf), [paper](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) | N/A | [biased_mf.py](examples/biased_mf.py), [given_data.py](examples/given_data.py)
|      | [Maximum Margin Matrix Factorization (MMMF)](cornac/models/mmmf), [paper](https://link.springer.com/content/pdf/10.1007/s10994-008-5073-7.pdf) | N/A | [mmmf_exp.py](examples/mmmf_exp.py)
|      | [Most Popular (MostPop)](cornac/models/most_pop), [paper](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf) | N/A | [bpr_netflix.py](examples/bpr_netflix.py)
|      | [Non-negative Matrix Factorization (NMF)](cornac/models/nmf), [paper](http://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf) | N/A | [nmf_exp.py](examples/nmf_example.py)
|      | [Probabilistic Matrix Factorization (PMF)](cornac/models/pmf), [paper](https://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf) | N/A | [pmf_ratio.py](examples/pmf_ratio.py)
|      | [Singular Value Decomposition (SVD)](cornac/models/svd), [paper](https://www.cs.rochester.edu/twiki/pub/Main/HarpSeminar/Factorization_Meets_the_Neighborhood-_a_Multifaceted_Collaborative_Filtering_Model.pdf) | N/A | [svd_exp.py](examples/svd_example.py)
|      | [Social Recommendation using PMF (SoRec)](cornac/models/sorec), [paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.304.2464&rep=rep1&type=pdf) | N/A | [sorec_filmtrust.py](examples/sorec_filmtrust.py)
|      | [Weighted Matrix Factorization (WMF)](cornac/models/wmf), [paper](http://yifanhu.net/PUB/cf.pdf) | [requirements.txt](cornac/models/wmf/requirements.txt) | [wmf_exp.py](examples/wmf_example.py)

## Support

Your contributions at any level of the library are welcome. If you intend to contribute, please:
  - Fork the Cornac repository to your own account.
  - Make changes and create pull requests.

You can also post bug reports and feature requests in [GitHub issues](https://github.com/PreferredAI/cornac/issues).

## License

[Apache License 2.0](LICENSE)
