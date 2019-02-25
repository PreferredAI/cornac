# -*- coding: utf-8 -*-

"""

@author: Le Duy Dung <ddle.2015@smu.edu.sg>
"""

import cornac
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.models import IBPR

# Load the MovieLens 1M dataset
ml_1m = movielens.load_1m()

# Instantiate an evaluation method.
ratio_split = RatioSplit(data=ml_1m, test_size=0.2, rating_threshold=1.0, exclude_unknowns=True)

# Instantiate a IBPR recommender model.
ibpr = IBPR(k=10, init_params={'U': None, 'V': None})

# Instantiate evaluation metrics.
rec_20 = cornac.metrics.Recall(k=20)
pre_20 = cornac.metrics.Precision(k=20)

# Instantiate and then run an experiment.
exp = cornac.Experiment(eval_method=ratio_split,
                        models=[ibpr],
                        metrics=[rec_20, pre_20],
                        user_based=True)
exp.run()