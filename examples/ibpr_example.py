# -*- coding: utf-8 -*-

"""
Example to run Probabilistic Matrix Factorization (PMF) model with Ratio Split evaluation strategy

@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import cornac
from cornac.datasets import MovieLens1M
from cornac.eval_methods import RatioSplit
from cornac.models import BPR

# Load the MovieLens 100K dataset
ml_1m = MovieLens1M.load_data()

# Instantiate an evaluation method.
ratio_split = RatioSplit(data=ml_1m, test_size=0.2, rating_threshold=1.0, exclude_unknowns=False)

# Instantiate a PMF recommender model.
bpr = BPR(k=50, init_params={'U': None, 'V': None})

# Instantiate evaluation metrics.
rec_20 = cornac.metrics.Recall(k=20)
pre_20 = cornac.metrics.Precision(k=20)

# Instantiate and then run an experiment.
exp = cornac.Experiment(eval_method=ratio_split,
                        models=[bpr],
                        metrics=[rec_20, pre_20],
                        user_based=True)
exp.run()

print(exp.avg_results)