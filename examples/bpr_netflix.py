# -*- coding: utf-8 -*-

"""
Example for Bayesian Personalized Ranking with Netflix dataset (subset)

@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import cornac
from cornac.datasets import netflix
from cornac.eval_methods import RatioSplit


ratio_split = RatioSplit(data=netflix.load_data_small(),
                         test_size=0.1,
                         rating_threshold=1.0,
                         exclude_unknowns=True,
                         verbose=True)

bpr = cornac.models.BPR(k=10, max_iter=100, learning_rate=0.01, lambda_reg=0.01)

auc = cornac.metrics.AUC()
rec_20 = cornac.metrics.Recall(k=20)

exp = cornac.Experiment(eval_method=ratio_split,
                        models=[bpr],
                        metrics=[auc, rec_20],
                        user_based=True)
exp.run()