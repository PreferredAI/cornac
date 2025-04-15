
import os
from datetime import datetime
import numpy as np
import cornac
from cornac.models import MF, PMF, BPR,GlobalLocalKernel


from cornac.eval_methods import RatioSplit
from cornac.metrics import MAE, RMSE, Precision, Recall, NDCG, AUC, MAP

ml_100k = cornac.datasets.movielens.load_feedback()
ml_100k = ml_100k[:500]

rs = RatioSplit(data=ml_100k, test_size=0.2, rating_threshold=4.0, seed=123)

# initialize models, here we are comparing: Biased MF, PMF, and BPR
mf = MF(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02, use_bias=True, seed=123)
pmf = PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123)
bpr = BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123)
models = [bpr]

# define metrics to evaluate the models
metrics = [MAE(), RMSE(), Precision(k=10), Recall(k=10), NDCG(k=10), AUC(), MAP()]

# put it together in an experiment, voilà!
cornac.Experiment(eval_method=rs, models=models, metrics=metrics, user_based=True).run()