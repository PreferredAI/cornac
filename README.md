# Cornac

**Cornac** is python recommender system library for **easy**, **effective** and **efficient** experiments. Cornac is **simple** and **handy**. It is designed from the ground-up to faithfully reflect the standard steps taken by researchers to implement and evaluate personalized recommendation models.

## Getting started

Getting started with Cornac is simple, and you just need to install it first.

### Installation

Please make sure you are using Python 3 (version >=3.6, is recommended), and you are on the latest pip.
Then, please run the appropriate Cornac install command according to your platform.

* **Windows**:
 
	```python
	#Installing PyTorch is required as this dependency is not handle automatically.
	pip install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-win_amd64.whl 
	pip install https://github.com/PreferredAI/cornac/raw/master/dist/cornac-0.1.0-cp36-cp36m-win_amd64.whl
	```

* **Linux**:
	```
	pip install https://github.com/PreferredAI/cornac/archive/master.zip --process-dependency-links
	```

* **MacOS**:
	```
	pip install https://github.com/PreferredAI/cornac/archive/master.zip
	```


### Your first Cornac experiment

This example will show you how to run your very first experiment using Cornac. It consists in training and evaluating the Probabilistic Matrix Factorization (PMF) recommender model.

```python
#Importing required modules from Cornac.
from cornac.models import Pmf
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
rec_pmf = Pmf(k=10, max_iter=10, learning_rate=0.001, lamda=0.001, init_params={'U':None,'V':None})

#Instantiate an evaluation strategy.
es_split = Split(data = mat_office, prop_test=0.2, prop_validation=0.0, good_rating=4)

#Instantiate evaluation metrics.
rec = metrics.Recall(m=20)
pre = metrics.Precision(m=20)

#Instantiate and then run an experiment.
res_pmf = Experiment(es_split, [rec_pmf], metrics=[pre,rec])
res_pmf.run_()

#Get average results.
res_pmf.res_avg
```