# Cornac

**Cornac** is python recommender system library for **easy**, **effective** and **efficient** experiments. Cornac is **simple** and **handy**. It is designed from the ground-up to faithfully reflect the standard steps taken by researchers to implement and evaluate personalized recommendation models.

### Quick links

[Website](https://cornac.preferred.ai/) |
[Documentation](https://cornac.readthedocs.io/en/latest/index.html) |
[Preferred.AI](https://preferred.ai/)

[![Build Status](https://www.travis-ci.org/PreferredAI/cornac.svg?branch=master)](https://www.travis-ci.org/PreferredAI/cornac)
[![Documentation Status](https://readthedocs.org/projects/cornac/badge/?version=latest)](https://cornac.readthedocs.io/en/latest/?badge=latest)
[![Python Versions](https://img.shields.io/badge/python-3.6-blue.svg)](https://cornac.preferred.ai/)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellowgreen.svg)](https://opensource.org/licenses/Apache-2.0)

## Getting started

Getting started with Cornac is simple, and you just need to install it first.

### Installation

Currently, we are supporting Python 3 (version 3.6 is recommended), please make sure that you are on the latest pip.
Then, please run the appropriate Cornac install command according to your platform.

* **Linux**:
	```
	pip3 install https://github.com/PreferredAI/cornac/archive/master.zip --process-dependency-links
	```

* **MacOS**:
	- You will need to install the Torch dependency first. Please follow the instructions [here](https://pytorch.org/) to install PyTorch on MacOS using conda. Then run the following command.
	```
	pip3 install https://github.com/PreferredAI/cornac/archive/master.zip
	```
	
* **Windows**:
 
	```python
	# Installing PyTorch is required as this dependency is not handle automatically.
	pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-win_amd64.whl 
	pip3 install https://github.com/PreferredAI/cornac/raw/master/dist/cornac-0.1.0-cp36-cp36m-win_amd64.whl
	```

### Your first Cornac experiment

This example will show you how to run your very first experiment using Cornac. 

Loading and preparing the Amazon office data (available inside folder 'data/').
```python
from scipy.io import loadmat

office = loadmat("data/office.mat")
mat_office = office['mat']
```

Instantiate an evaluation strategy.
```python
from cornac.eval_strategies import RatioSplit

ratio_split = RatioSplit(data=mat_office, test_size=0.2, rating_threshold=4.0, exclude_unknowns=False)
```

Here we use `Probabilistic Matrix Factorization (PMF)` recommender model.
```python
from cornac.models import PMF

pmf = PMF(k=10, max_iter=100, learning_rate=0.001, lamda=0.001)
```

Instantiate evaluation metrics.
```python
from cornac.metrics import Recall, Precision, MAE, RMSE

mae = MAE()
rmse = RMSE()
rec_20 = Recall(m=20)
pre_20 = Precision(m=20)
```


Instantiate and then run an experiment.
```python
from cornac.experiment import Experiment

res_pmf = Experiment(eval_strategy=ratio_split,
                     models=[pmf],
                     metrics=[mae, rmse, rec_20, pre_20],
                     user_based=True)
res_pmf.run()
```

For more details, take a look at the provided [example](examples/pmf_ratio.py).

## License

[Apache License 2.0](LICENSE.md)
