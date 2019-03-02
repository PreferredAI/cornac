# -*- coding: utf-8 -*-

"""
Example for Matrix Factorization with biases

@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import cornac
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit

ratio_split = RatioSplit(data=movielens.load_1m(),
                         test_size=0.2,
                         exclude_unknowns=False,
                         verbose=True)

mf = cornac.models.MF(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02,
                      use_bias=True, early_stop=True, verbose=True)

mae = cornac.metrics.MAE()
rmse = cornac.metrics.RMSE()

exp = cornac.Experiment(eval_method=ratio_split,
                        models=[mf],
                        metrics=[mae, rmse],
                        user_based=True)
exp.run()