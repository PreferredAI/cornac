# Cornac

**Cornac** is python recommender system library for **easy**, **effective** and **efficient** experiments. Cornac is **simple** and **handy**. It is designed from the ground-up to faithfully reflect the standard steps taken by researchers to implement and evaluate personalized recommendation models.

[![Build Status](https://www.travis-ci.org/PreferredAI/cornac.svg?branch=master)](https://www.travis-ci.org/PreferredAI/cornac)
[![Documentation Status](https://readthedocs.org/projects/cornac/badge/?version=latest)](https://cornac.readthedocs.io/en/latest/?badge=latest)
[![Python Versions](https://img.shields.io/badge/python-3.6-blue.svg)](https://cornac.preferred.ai/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

### Quick links
[Website](https://cornac.preferred.ai/) |
[Documentation](https://cornac.readthedocs.io/en/latest/index.html) |
[Preferred.AI](https://preferred.ai/)

## Getting started

Getting started with Cornac is simple, and you just need to install it first.

### Installation

Please make sure you are using Python 3 (version >=3.6, is recommended), and you are on the latest pip.
Then, please run the appropriate Cornac install command according to your platform.

* **Windows**:
 
	```python
	#Installing PyTorch is required as this dependency is not handle automatically.
	pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-win_amd64.whl 
	pip3 install https://github.com/PreferredAI/cornac/raw/master/dist/cornac-0.1.0-cp36-cp36m-win_amd64.whl
	```

* **Linux**:
	```
	pip3 install https://github.com/PreferredAI/cornac/archive/master.zip --process-dependency-links
	```

* **MacOS**:
	- You will need to install the Torch dependency first. Please follow the instructions [here](https://pytorch.org/) to install PyTorch on MacOS using conda. Then run the following command.
	```
	pip install https://github.com/PreferredAI/cornac/archive/master.zip
	```


### Your first Cornac experiment

This example will show you how to run your very first experiment using Cornac. It consists in training and evaluating the Probabilistic Matrix Factorization (PMF) recommender model.

```python
#Importing required modules from Cornac.
from cornac.models import PMF
from cornac.experiment import Experiment
from cornac.evaluation_strategies import Split
from cornac import metrics 

#Importing some additional useful modules.
from scipy.io import loadmat

#Loading and preparing the Amazon office data,
#Available in the GitHub repository, inside folder 'data/'. 
office= loadmat("path to office.mat")
mat_office = office['mat']

#Instantiate a pfm recommender model.
#Please refer to the documentation for details on parameter settings.
rec_pmf = PMF(k=10, max_iter=100, learning_rate=0.001, lamda=0.001, init_params={'U':None,'V':None})

#Instantiate an evaluation strategy.
es_split = Split(data = mat_office, prop_test=0.2, prop_validation=0.0, good_rating=4)

#Instantiate evaluation metrics.
rec = metrics.Recall(m=20)
pre = metrics.Precision(m=20)
mae = metrics.MAE()
rmse = metrics.RMSE()

#Instantiate and then run an experiment.
res_pmf = Experiment(es_split, [rec_pmf], metrics=[mae, rmse, pre, rec])
res_pmf.run()

#Get average results.
res_pmf.average_result
```
