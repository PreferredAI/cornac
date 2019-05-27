# -*- coding: utf-8 -*-

"""
Example for Collaborative Filtering for Implicit Feedback Datasets (Citeulike)

@author: Tran Thanh Binh
"""

import cornac
from cornac.data import Reader
from cornac.datasets import citeulike
from cornac.eval_methods import RatioSplit

_, item_ids = citeulike.load_text()

data = citeulike.load_data(reader=Reader(item_set=item_ids))

ratio_split = RatioSplit(data=data, test_size=0.2, exclude_unknowns=True,
                         verbose=True, seed=123, rating_threshold=0.5)

cf = cornac.models.CF(k=200, max_iter=100, learning_rate=0.001, lambda_u=0.01, lambda_v=0.01, verbose=True)

rec_300 = cornac.metrics.Recall(k=300)

cornac.Experiment(eval_method=ratio_split,
                  models=[cf],
                  metrics=[rec_300],
                  user_based=True).run()
