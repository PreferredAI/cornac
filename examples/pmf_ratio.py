# -*- coding: utf-8 -*-

"""
Example to run Probabilistic Matrix Factorization (PMF) model with Ratio Split evaluation strategy

@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from cornac.models import PMF
from cornac.eval_strategies import RatioSplit
from cornac.metrics import Recall, Precision, MAE, RMSE
from cornac import Experiment

from scipy.io import loadmat

office = loadmat('data/office.mat')
mat_office = office['mat']

# Instantiate a PMF recommender model.
pmf = PMF(k=10, max_iter=100, learning_rate=0.001, lamda=0.001)

# Instantiate an evaluation strategy.
ratio_split = RatioSplit(data=mat_office, test_size=0.2, rating_threshold=4.0, exclude_unknowns=False)

# Instantiate evaluation metrics.
mae = MAE()
rmse = RMSE()
rec_20 = Recall(k=20)
pre_20 = Precision(k=20)

# Instantiate and then run an experiment.
res_pmf = Experiment(eval_strategy=ratio_split,
                     models=[pmf],
                     metrics=[mae, rmse, rec_20, pre_20],
                     user_based=True)
res_pmf.run()
