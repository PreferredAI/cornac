# -*- coding: utf-8 -*-

"""
Example for Bayesian Personalized Ranking

@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import cornac
from cornac.datasets import MovieLens100K
from cornac.eval_methods import RatioSplit

data = MovieLens100K.load_data()

ratio_split = RatioSplit(data=data, test_size=0.2, exclude_unknowns=False, verbose=True)

bpr = cornac.models.BPR(k=10, max_iter=100)

ndcg = cornac.metrics.NDCG()
rec_20 = cornac.metrics.Recall(k=20)

exp = cornac.Experiment(eval_method=ratio_split,
                        models=[bpr],
                        metrics=[ndcg, rec_20],
                        user_based=True)
exp.run()

print(exp.avg_results)