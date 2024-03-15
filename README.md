# Cornac

**Cornac** is a comparative framework for multimodal recommender systems. It focuses on making it **convenient** to work with models leveraging **auxiliary data** (e.g., item descriptive text and image, social network, etc). **Cornac** enables **fast** experiments and **straightforward** implementations of new models. It is **highly compatible** with existing machine learning libraries (e.g., TensorFlow, PyTorch).

*Cornac is one of the frameworks recommended by [ACM RecSys 2023](https://github.com/ACMRecSys/recsys-evaluation-frameworks) for the evaluation and reproducibility of recommendation algorithms.*

### Quick Links

[Website](https://cornac.preferred.ai/) |
[Documentation](https://cornac.readthedocs.io/en/stable/index.html) |
[Tutorials](tutorials#tutorials) |
[Examples](https://github.com/PreferredAI/cornac/tree/master/examples#cornac-examples-directory) |
[Models](#models) |
[Datasets](./cornac/datasets/README.md#datasets) |
[Paper](http://www.jmlr.org/papers/volume21/19-805/19-805.pdf) |
[Preferred.AI](https://preferred.ai/)

[![.github/workflows/python-package.yml](https://github.com/PreferredAI/cornac/actions/workflows/python-package.yml/badge.svg)](https://github.com/PreferredAI/cornac/actions/workflows/python-package.yml)
[![CircleCI](https://img.shields.io/circleci/project/github/PreferredAI/cornac/master.svg?logo=circleci)](https://circleci.com/gh/PreferredAI/cornac)
[![AppVeyor](https://ci.appveyor.com/api/projects/status/0yq4td1xg4kkhdwu?svg=true)](https://ci.appveyor.com/project/tqtg/cornac)
[![Codecov](https://img.shields.io/codecov/c/github/PreferredAI/cornac/master.svg?logo=codecov)](https://codecov.io/gh/PreferredAI/cornac)
[![Docs](https://img.shields.io/readthedocs/cornac/latest.svg)](https://cornac.readthedocs.io/en/stable)
<br />
[![Release](https://img.shields.io/github/release-pre/PreferredAI/cornac.svg)](https://github.com/PreferredAI/cornac/releases)
[![PyPI](https://img.shields.io/pypi/v/cornac.svg)](https://pypi.org/project/cornac/)
[![Conda](https://img.shields.io/conda/vn/conda-forge/cornac.svg)](https://anaconda.org/conda-forge/cornac)
[![Conda Recipe](https://img.shields.io/badge/recipe-cornac-green.svg)](https://github.com/conda-forge/cornac-feedstock)
[![Downloads](https://static.pepy.tech/personalized-badge/cornac?period=month&units=international_system&left_color=grey&right_color=orange&left_text=downloads/month)](https://pepy.tech/project/cornac)
<br />
[![Python](https://img.shields.io/pypi/pyversions/cornac.svg)](https://cornac.preferred.ai/)
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/cornac.svg)](https://anaconda.org/conda-forge/cornac)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)


## Installation

Currently, we are supporting Python 3. There are several ways to install Cornac:

- **From PyPI (recommended):**
  ```bash
  pip3 install cornac
  ```

- **From Anaconda:**
  ```bash
  conda install cornac -c conda-forge
  ```

- **From the GitHub source (for latest updates):**
  ```bash
  pip3 install Cython numpy scipy
  pip3 install git+https://github.com/PreferredAI/cornac.git
  ```

**Note:** 

Additional dependencies required by models are listed [here](README.md#Models).

Some algorithm implementations use `OpenMP` to support multi-threading. For Mac OS users, in order to run those algorithms efficiently, you might need to install `gcc` from Homebrew to have an OpenMP compiler:
```bash
brew install gcc | brew link gcc
```

## Getting started: your first Cornac experiment

![](flow.jpg)
<p align="center"><i>Flow of an Experiment in Cornac</i></p>

```python
import cornac
from cornac.eval_methods import RatioSplit
from cornac.models import MF, PMF, BPR
from cornac.metrics import MAE, RMSE, Precision, Recall, NDCG, AUC, MAP

# load the built-in MovieLens 100K and split the data based on ratio
ml_100k = cornac.datasets.movielens.load_feedback()
rs = RatioSplit(data=ml_100k, test_size=0.2, rating_threshold=4.0, seed=123)

# initialize models, here we are comparing: Biased MF, PMF, and BPR
mf = MF(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02, use_bias=True, seed=123)
pmf = PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123)
bpr = BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123)
models = [mf, pmf, bpr]

# define metrics to evaluate the models
metrics = [MAE(), RMSE(), Precision(k=10), Recall(k=10), NDCG(k=10), AUC(), MAP()]

# put it together in an experiment, voilà!
cornac.Experiment(eval_method=rs, models=models, metrics=metrics, user_based=True).run()
```

**Output:**

|                          |    MAE |   RMSE |    AUC |     MAP | NDCG@10 | Precision@10 | Recall@10 |  Train (s) | Test (s) |
| ------------------------ | -----: | -----: | -----: | ------: | ------: | -----------: | --------: | ---------: | -------: |
| [MF](cornac/models/mf)   | 0.7430 | 0.8998 | 0.7445 |  0.0548 |  0.0761 |       0.0675 |    0.0463 |       0.13 |     1.57 |
| [PMF](cornac/models/pmf) | 0.7534 | 0.9138 | 0.7744 |  0.0671 |  0.0969 |       0.0813 |    0.0639 |       2.18 |     1.64 |
| [BPR](cornac/models/bpr) |    N/A |    N/A | 0.8695 |  0.1042 |  0.1500 |       0.1110 |    0.1195 |       3.74 |     1.49 |


For more details, please take a look at our [examples](examples) as well as [tutorials](tutorials). For learning purposes, this list of [tutorials on recommender systems](https://github.com/PreferredAI/tutorials/tree/master/recommender-systems) will be more organized and comprehensive. 


## Model serving

Here, we provide a simple way to serve a Cornac model by launching a standalone web service with [Flask](https://github.com/pallets/flask). It is very handy for testing or creating a demo application. First, we install the dependency:
```bash
$ pip3 install Flask
```
Supposed that we want to serve the trained BPR model from previous example, we need to save it:
```python
bpr.save("save_dir", save_trainset=True)
```
After that, the model can be deployed easily by running Cornac serving app as follows:
```bash
$ FLASK_APP='cornac.serving.app' \
  MODEL_PATH='save_dir/BPR' \
  MODEL_CLASS='cornac.models.BPR' \
  flask run --host localhost --port 8080

# Running on http://localhost:8080
```
Here we go, our model service is now ready. Let's get `top-5` item recommendations for the user `"63"`:
```bash
$ curl -X GET "http://localhost:8080/recommend?uid=63&k=5&remove_seen=false"

# Response: {"recommendations": ["50", "181", "100", "258", "286"], "query": {"uid": "63", "k": 5, "remove_seen": false}}
```
If we want to remove seen items during training, we need to provide `TRAIN_SET` which has been saved with the model earlier, when starting the serving app. We can also leverage [WSGI](https://flask.palletsprojects.com/en/3.0.x/deploying/) server for model deployment in production. Please refer to [this](https://cornac.readthedocs.io/en/stable/user/iamadeveloper.html#running-an-api-service) guide for more details.

## Efficient retrieval with ANN search

One important aspect of deploying recommender model is efficient retrieval via Approximate Nearest Neighor (ANN) search in vector space. Cornac integrates several vector similarity search frameworks for the ease of deployment. [This example](tutorials/ann_hnswlib.ipynb) demonstrates how ANN search will work seamlessly with any recommender models supporting it (e.g., MF).

| Supported framework | Cornac wrapper | Examples |
| :---: | :---: | :---: |
| [spotify/annoy](https://github.com/spotify/annoy) | [AnnoyANN](cornac/models/ann/recom_ann_annoy.py) | [ann_example.py](examples/ann_example.py), [ann_all.ipynb](examples/ann_all.ipynb)
| [meta/faiss](https://github.com/facebookresearch/faiss) | [FaissANN](cornac/models/ann/recom_ann_faiss.py) | [ann_example.py](examples/ann_example.py), [ann_all.ipynb](examples/ann_all.ipynb)
| [nmslib/hnswlib](https://github.com/nmslib/hnswlib) | [HNSWLibANN](cornac/models/ann/recom_ann_hnswlib.py) | [ann_example.py](examples/ann_example.py), [ann_hnswlib.ipynb](tutorials/ann_hnswlib.ipynb), [ann_all.ipynb](examples/ann_all.ipynb)
| [google/scann](https://github.com/google-research/google-research/tree/master/scann) | [ScaNNANN](cornac/models/ann/recom_ann_scann.py) | [ann_example.py](examples/ann_example.py), [ann_all.ipynb](examples/ann_all.ipynb)


## Models

The recommender models supported by Cornac are listed below. Why don't you join us to lengthen the list?


| Year | Model and paper | Model type | Require-ments | Examples |
| :---: | --- | :---: | :---: | :---: |
| 2022 | [Disentangled Multimodal Representation Learning for Recommendation)](cornac/models/dmrl), [paper](https://arxiv.org/pdf/2203.05406.pdf) | Collaborative Filtering / Content-Based | [reqs](cornac/models/dmrl/requirements.txt) | [exp](examples/dmrl_example.py)
| 2021 | [Bilateral Variational Autoencoder for Collaborative Filtering (BiVAECF)](cornac/models/bivaecf), [paper](https://dl.acm.org/doi/pdf/10.1145/3437963.3441759) | Collaborative Filtering / Content-Based | [reqs](cornac/models/bivaecf/requirements.txt) | [exp](https://github.com/PreferredAI/bi-vae)
|      | [Causal Inference for Visual Debiasing in Visually-Aware Recommendation (CausalRec)](cornac/models/causalrec), [paper](https://arxiv.org/abs/2107.02390) | Content-Based / Image | [reqs](cornac/models/causalrec/requirements.txt) | [exp](examples/causalrec_clothing.py)
|      | [Explainable Recommendation with Comparative Constraints on Product Aspects (ComparER)](cornac/models/comparer), [paper](https://dl.acm.org/doi/pdf/10.1145/3437963.3441754) | Explainable | N/A | [exp](https://github.com/PreferredAI/ComparER)
| 2020 | [Adversarial Multimedia Recommendation (AMR)](cornac/models/amr), [paper](https://ieeexplore.ieee.org/document/8618394) | Content-Based / Image | [reqs](cornac/models/amr/requirements.txt) | [exp](examples/amr_clothing.py)
|      | [Hybrid Deep Representation Learning of Ratings and Reviews (HRDR)](cornac/models/hrdr), [paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231219313207) | Content-Based / Text | [reqs](cornac/models/hrdr/requirements.txt) | [exp](examples/hrdr_example.py)
|      | [LightGCN: Simplifying and Powering Graph Convolution Network](cornac/models/lightgcn), [paper](https://arxiv.org/pdf/2002.02126.pdf) | Collaborative Filtering | [reqs](cornac/models/lightgcn/requirements.txt) | [exp](examples/lightgcn_example.py)
|      | [Predicting Temporal Sets with Deep Neural Networks (DNNTSP)](cornac/models/dnntsp), [paper](https://arxiv.org/pdf/2006.11483.pdf) | Next-Basket | [reqs](cornac/models/dnntsp/requirements.txt) | [exp](examples/dnntsp_tafeng.py)
|      | [Recency Aware Collaborative Filtering (UPCF)](cornac/models/upcf), [paper](https://dl.acm.org/doi/abs/10.1145/3340631.3394850) | Next-Basket | [reqs](cornac/models/upcf/requirements.txt) | [exp](examples/upcf_tafeng.py)
|      | [Temporal-Item-Frequency-based User-KNN (TIFUKNN)](cornac/models/tifuknn), [paper](https://arxiv.org/pdf/2006.00556.pdf) | Next-Basket | N/A | [exp](examples/tifuknn_tafeng.py)
|      | [Variational Autoencoder for Top-N Recommendations (RecVAE)](cornac/models/recvae), [paper](https://doi.org/10.1145/3336191.3371831) | Collaborative Filtering | [reqs](cornac/models/recvae/requirements.txt) | [exp](examples/recvae_example.py)
| 2019 | [Correlation-Sensitive Next-Basket Recommendation (Beacon)](cornac/models/beacon), [paper](https://www.ijcai.org/proceedings/2019/0389.pdf) | Next-Basket | [reqs](cornac/models/beacon/requirements.txt) | [exp](examples/beacon_tafeng.py)
|      | [Embarrassingly Shallow Autoencoders for Sparse Data (EASEᴿ)](cornac/models/ease), [paper](https://arxiv.org/pdf/1905.03375.pdf) | Collaborative Filtering | N/A | [exp](examples/ease_movielens.py)
|      | [Neural Graph Collaborative Filtering (NGCF)](cornac/models/ngcf), [paper](https://arxiv.org/pdf/1905.08108.pdf) | Collaborative Filtering | [reqs](cornac/models/ngcf/requirements.txt) | [exp](examples/ngcf_example.py)
| 2018 | [Collaborative Context Poisson Factorization (C2PF)](cornac/models/c2pf), [paper](https://www.ijcai.org/proceedings/2018/0370.pdf) | Content-Based / Graph | N/A | [exp](examples/c2pf_example.py)
|      | [Graph Convolutional Matrix Completion (GCMC)](cornac/models/gcmc), [paper](https://www.kdd.org/kdd2018/files/deep-learning-day/DLDay18_paper_32.pdf) | Collaborative Filtering | [reqs](cornac/models/gcmc/requirements.txt) | [exp](examples/gcmc_example.py)
|      | [Multi-Task Explainable Recommendation (MTER)](cornac/models/mter), [paper](https://arxiv.org/pdf/1806.03568.pdf) | Explainable | N/A | [exp](examples/mter_example.py)
|      | [Neural Attention Rating Regression with Review-level Explanations (NARRE)](cornac/models/narre), [paper](http://www.thuir.cn/group/~YQLiu/publications/WWW2018_CC.pdf) | Explainable / Content-Based | [reqs](cornac/models/narre/requirements.txt) | [exp](examples/narre_example.py)
|      | [Probabilistic Collaborative Representation Learning (PCRL)](cornac/models/pcrl), [paper](http://www.hadylauw.com/publications/uai18.pdf) | Content-Based / Graph | [reqs](cornac/models/pcrl/requirements.txt) | [exp](examples/pcrl_example.py)
|      | [Variational Autoencoder for Collaborative Filtering (VAECF)](cornac/models/vaecf), [paper](https://arxiv.org/pdf/1802.05814.pdf) | Collaborative Filtering | [reqs](cornac/models/vaecf/requirements.txt) | [exp](examples/vaecf_citeulike.py)
| 2017 | [Collaborative Variational Autoencoder (CVAE)](cornac/models/cvae), [paper](http://eelxpeng.github.io/assets/paper/Collaborative_Variational_Autoencoder.pdf) | Content-Based / Text | [reqs](cornac/models/cvae/requirements.txt) | [exp](examples/cvae_example.py)
|      | [Conditional Variational Autoencoder for Collaborative Filtering (CVAECF)](cornac/models/cvaecf), [paper](https://dl.acm.org/doi/10.1145/3132847.3132972) | Content-Based / Text | [reqs](cornac/models/cvaecf/requirements.txt) | [exp](examples/cvaecf_filmtrust.py)
|      | [Generalized Matrix Factorization (GMF)](cornac/models/ncf), [paper](https://arxiv.org/pdf/1708.05031.pdf) | Collaborative Filtering | [reqs](cornac/models/ncf/requirements.txt) | [exp](examples/ncf_example.py)
|      | [Indexable Bayesian Personalized Ranking (IBPR)](cornac/models/ibpr), [paper](http://www.hadylauw.com/publications/cikm17a.pdf) | Collaborative Filtering | [reqs](cornac/models/ibpr/requirements.txt) | [exp](examples/ibpr_example.py)
|      | [Matrix Co-Factorization (MCF)](cornac/models/mcf), [paper](https://dsail.kaist.ac.kr/files/WWW17.pdf) | Content-Based / Graph | N/A | [exp](examples/mcf_office.py)
|      | [Multi-Layer Perceptron (MLP)](cornac/models/ncf), [paper](https://arxiv.org/pdf/1708.05031.pdf) | Collaborative Filtering | [reqs](cornac/models/ncf/requirements.txt) | [exp](examples/ncf_example.py)
|      | [Neural Matrix Factorization (NeuMF) / Neural Collaborative Filtering (NCF)](cornac/models/ncf), [paper](https://arxiv.org/pdf/1708.05031.pdf) | Collaborative Filtering | [reqs](cornac/models/ncf/requirements.txt) | [exp](examples/ncf_example.py)
|      | [Online Indexable Bayesian Personalized Ranking (Online IBPR)](cornac/models/online_ibpr), [paper](http://www.hadylauw.com/publications/cikm17a.pdf) | Collaborative Filtering | [reqs](cornac/models/online_ibpr/requirements.txt) |
|      | [Visual Matrix Factorization (VMF)](cornac/models/vmf), [paper](https://dsail.kaist.ac.kr/files/WWW17.pdf) | Content-Based / Image | [reqs](cornac/models/vmf/requirements.txt) | [exp](examples/vmf_clothing.py)
| 2016 | [Collaborative Deep Ranking (CDR)](cornac/models/cdr), [paper](http://inpluslab.com/chenliang/homepagefiles/paper/hao-pakdd2016.pdf) | Content-Based / Text | [reqs](cornac/models/cdr/requirements.txt) | [exp](examples/cdr_example.py)
|      | [Collaborative Ordinal Embedding (COE)](cornac/models/coe), [paper](http://www.hadylauw.com/publications/sdm16.pdf) | Collaborative Filtering | [reqs](cornac/models/coe/requirements.txt) |
|      | [Convolutional Matrix Factorization (ConvMF)](cornac/models/conv_mf), [paper](http://uclab.khu.ac.kr/resources/publication/C_351.pdf) | Content-Based / Text | [reqs](cornac/models/conv_mf/requirements.txt) | [exp](examples/conv_mf_example.py)
|      | [Learning to Rank Features for Recommendation over Multiple Categories (LRPPM)](cornac/models/lrppm), [paper](https://www.yongfeng.me/attach/sigir16-chen.pdf) | Explainable | N/A | [exp](examples/lrppm_example.py)
|      | [Session-based Recommendations With Recurrent Neural Networks (GRU4Rec)](cornac/models/gru4rec), [paper](https://arxiv.org/pdf/1511.06939.pdf) | Next-Item | [reqs](cornac/models/gru4rec/requirements.txt) | [exp](examples/gru4rec_yoochoose.py)
|      | [Spherical K-means (SKM)](cornac/models/skm), [paper](https://www.sciencedirect.com/science/article/pii/S092523121501509X) | Collaborative Filtering | N/A | [exp](examples/skm_movielens.py)
|      | [Visual Bayesian Personalized Ranking (VBPR)](cornac/models/vbpr), [paper](https://arxiv.org/pdf/1510.01784.pdf) | Content-Based / Image | [reqs](cornac/models/vbpr/requirements.txt) | [exp](examples/vbpr_tradesy.py)
| 2015 | [Collaborative Deep Learning (CDL)](cornac/models/cdl), [paper](https://arxiv.org/pdf/1409.2944.pdf) | Content-Based / Text | [reqs](cornac/models/cdl/requirements.txt) | [exp](examples/cdl_example.py)
|      | [Hierarchical Poisson Factorization (HPF)](cornac/models/hpf), [paper](http://jakehofman.com/inprint/poisson_recs.pdf) | Collaborative Filtering | N/A | [exp](examples/hpf_movielens.py)
|      | [TriRank: Review-aware Explainable Recommendation by Modeling Aspects](cornac/models/trirank), [paper](https://wing.comp.nus.edu.sg/wp-content/uploads/Publications/PDF/TriRank-%20Review-aware%20Explainable%20Recommendation%20by%20Modeling%20Aspects.pdf) | Explainable | N/A | [exp](examples/trirank_example.py)
| 2014 | [Explicit Factor Model (EFM)](cornac/models/efm), [paper](https://www.yongfeng.me/attach/efm-zhang.pdf) | Explainable | N/A | [exp](examples/efm_example.py)
|      | [Social Bayesian Personalized Ranking (SBPR)](cornac/models/sbpr), [paper](https://cseweb.ucsd.edu/~jmcauley/pdfs/cikm14.pdf) | Content-Based / Social | N/A | [exp](examples/sbpr_epinions.py)
| 2013 | [Hidden Factors and Hidden Topics (HFT)](cornac/models/hft), [paper](https://cs.stanford.edu/people/jure/pubs/reviews-recsys13.pdf) | Content-Based / Text | N/A | [exp](examples/hft_example.py)
| 2012 | [Weighted Bayesian Personalized Ranking (WBPR)](cornac/models/bpr), [paper](http://proceedings.mlr.press/v18/gantner12a/gantner12a.pdf) | Collaborative Filtering | N/A | [exp](examples/bpr_netflix.py)
| 2011 | [Collaborative Topic Regression (CTR)](cornac/models/ctr), [paper](http://www.cs.columbia.edu/~blei/papers/WangBlei2011.pdf) | Content-Based / Text | N/A | [exp](examples/ctr_example_citeulike.py)
| Earlier | [Baseline Only](cornac/models/baseline_only), [paper](http://courses.ischool.berkeley.edu/i290-dm/s11/SECURE/a1-koren.pdf) | Baseline | N/A | [exp](examples/svd_example.py)
|      | [Bayesian Personalized Ranking (BPR)](cornac/models/bpr), [paper](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf) | Collaborative Filtering | N/A | [exp](examples/bpr_netflix.py)
|      | [Factorization Machines (FM)](cornac/models/fm), [paper](https://www.csie.ntu.edu.tw/~b97053/paper/Factorization%20Machines%20with%20libFM.pdf) | Collaborative Filtering / Content-Based | Linux only | [exp](examples/fm_example.py)
|      | [Global Average (GlobalAvg)](cornac/models/global_avg), [paper](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) | Baseline | N/A | [exp](examples/biased_mf.py)
|      | [Global Personalized Top Frequent (GPTop)](cornac/models/gp_top), [paper](https://dl.acm.org/doi/pdf/10.1145/3587153) | Next-Basket | N/A | [exp](examples/gp_top_tafeng.py)
|      | [Item K-Nearest-Neighbors (ItemKNN)](cornac/models/knn), [paper](https://dl.acm.org/doi/pdf/10.1145/371920.372071) | Neighborhood-Based | N/A | [exp](examples/knn_movielens.py)
|      | [Matrix Factorization (MF)](cornac/models/mf), [paper](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) | Collaborative Filtering | N/A | [exp1](examples/biased_mf.py), [exp2](examples/given_data.py)
|      | [Maximum Margin Matrix Factorization (MMMF)](cornac/models/mmmf), [paper](https://link.springer.com/content/pdf/10.1007/s10994-008-5073-7.pdf) | Collaborative Filtering | N/A | [exp](examples/mmmf_exp.py)
|      | [Most Popular (MostPop)](cornac/models/most_pop), [paper](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf) | Baseline | N/A | [exp](examples/bpr_netflix.py)
|      | [Non-negative Matrix Factorization (NMF)](cornac/models/nmf), [paper](http://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf) | Collaborative Filtering | N/A | [exp](examples/nmf_example.py)
|      | [Probabilistic Matrix Factorization (PMF)](cornac/models/pmf), [paper](https://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf) | Collaborative Filtering | N/A | [exp](examples/pmf_ratio.py)
|      | [Session Popular (SPop)](cornac/models/spop), [paper](https://arxiv.org/pdf/1511.06939.pdf) | Next-Item / Baseline | N/A | [exp](examples/spop_yoochoose.py)
|      | [Singular Value Decomposition (SVD)](cornac/models/svd), [paper](https://people.engr.tamu.edu/huangrh/Spring16/papers_course/matrix_factorization.pdf) | Collaborative Filtering | N/A | [exp](examples/svd_example.py)
|      | [Social Recommendation using PMF (SoRec)](cornac/models/sorec), [paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.304.2464&rep=rep1&type=pdf) | Content-Based / Social | N/A | [exp](examples/sorec_filmtrust.py)
|      | [User K-Nearest-Neighbors (UserKNN)](cornac/models/knn), [paper](https://arxiv.org/pdf/1301.7363.pdf) | Neighborhood-Based | N/A | [exp](examples/knn_movielens.py)
|      | [Weighted Matrix Factorization (WMF)](cornac/models/wmf), [paper](http://yifanhu.net/PUB/cf.pdf) | Collaborative Filtering | [reqs](cornac/models/wmf/requirements.txt) | [exp](examples/wmf_example.py)


## Contributing

This project welcomes contributions and suggestions. Before contributing, please see our [contribution guidelines](https://cornac.readthedocs.io/en/stable/developer/index.html).

## Citation

If you use Cornac in a scientific publication, we would appreciate citations to the following papers:

- [Cornac: A Comparative Framework for Multimodal Recommender Systems](http://jmlr.org/papers/v21/19-805.html), Salah *et al.*, Journal of Machine Learning Research, 21(95):1–5, 2020.

  ```
  @article{salah2020cornac,
    title={Cornac: A Comparative Framework for Multimodal Recommender Systems},
    author={Salah, Aghiles and Truong, Quoc-Tuan and Lauw, Hady W},
    journal={Journal of Machine Learning Research},
    volume={21},
    number={95},
    pages={1--5},
    year={2020}
  }
  ```

- [Exploring Cross-Modality Utilization in Recommender Systems](https://ieeexplore.ieee.org/abstract/document/9354572), Truong *et al.*, IEEE Internet Computing, 25(4):50–57, 2021.

  ```
  @article{truong2021exploring,
    title={Exploring Cross-Modality Utilization in Recommender Systems},
    author={Truong, Quoc-Tuan and Salah, Aghiles and Tran, Thanh-Binh and Guo, Jingyao and Lauw, Hady W},
    journal={IEEE Internet Computing},
    year={2021},
    publisher={IEEE}
  }
  ```

- [Multi-Modal Recommender Systems: Hands-On Exploration](https://dl.acm.org/doi/10.1145/3460231.3473324), Truong *et al.*, ACM Conference on Recommender Systems, 2021.

  ```
  @inproceedings{truong2021multi,
    title={Multi-modal recommender systems: Hands-on exploration},
    author={Truong, Quoc-Tuan and Salah, Aghiles and Lauw, Hady},
    booktitle={Fifteenth ACM Conference on Recommender Systems},
    pages={834--837},
    year={2021}
  }
  ```

## License

[Apache License 2.0](LICENSE)
