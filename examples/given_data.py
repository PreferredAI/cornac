# -*- coding: utf-8 -*-

"""
Example to train and evaluate a model with given data

@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from cornac.data import reader
from cornac.eval_methods import BaseMethod
from cornac.models import MF
from cornac.metrics import MAE, RMSE
from cornac.utils import cache

# Download MovieLens 100K provided training and test splits
train_path = cache(url='http://files.grouplens.org/datasets/movielens/ml-100k/u1.base')
test_path = cache(url='http://files.grouplens.org/datasets/movielens/ml-100k/u1.test')
train_data = reader.read_uir(train_path)
test_data = reader.read_uir(test_path)

eval_method = BaseMethod.from_splits(train_data=train_data, test_data=test_data,
                                     exclude_unknowns=False, verbose=True)

mf = MF(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02,
        use_bias=True, early_stop=True, verbose=True)

# Evaluation
avg_results, _ = eval_method.evaluate(model=mf,
                                      metrics=[MAE(), RMSE()],
                                      user_based=True)
print(avg_results)
