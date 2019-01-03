# -*- coding: utf-8 -*-

"""
Example to run Probabilistic Bayesian personalized ranking (BPR) model with Ratio Split evaluation strategy

@author: Guo Jingyao <jyguo@smu.edu.sg>
"""

import cornac
from cornac.datasets import MovieLens100K
from cornac.eval_strategies import RatioSplit
from cornac.models import Autorec

# Load the MovieLens 100K dataset
ml_100k = MovieLens100K.load_data()

# Instantiate an evaluation strategy.
ratio_split = RatioSplit(data=ml_100k, test_size=0.2, rating_threshold=1.0, exclude_unknowns=False)

# Instantiate a PMF recommender model.
autorec = Autorec(k=10, max_iter=100, learning_rate=0.001, lamda=0.001)

# Instantiate evaluation metrics.
mae = cornac.metrics.MAE()
rmse = cornac.metrics.RMSE()
rec_20 = cornac.metrics.Recall(k=20)
pre_20 = cornac.metrics.Precision(k=20)

# Instantiate and then run an experiment.
exp = cornac.Experiment(eval_strategy=ratio_split,
                        models=[autorec],
                        metrics=[mae, rmse, rec_20, pre_20],
                        user_based=True)
exp.run()
print(exp.avg_results)