# -*- coding: utf-8 -*-

"""
Example to train and evaluate a model with given data

@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import cornac as cn
from cornac.eval_methods import BaseMethod
from cornac.utils import DownloadItem

# Download MovieLens 100K provided training and test splits
train_path = DownloadItem(url='http://files.grouplens.org/datasets/movielens/ml-100k/u1.base',
                          relative_path='datasets/ml_100k/u1.base').maybe_download(True)
test_path = DownloadItem(url='http://files.grouplens.org/datasets/movielens/ml-100k/u1.test',
                         relative_path='datasets/ml_100k/u1.test').maybe_download(True)

# Load data using Reader
train_data = cn.data.Reader.read_uir_triplets(train_path)
test_data = cn.data.Reader.read_uir_triplets(test_path)

# Construct base evaluation method with given data
eval_method = BaseMethod.from_provided(train_data=train_data, test_data=test_data,
                                       exclude_unknowns=False, verbose=True)

# Model
mf = cn.models.MF(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02,
                  use_bias=True, early_stop=True, verbose=True)

# Metrics
mae = cn.metrics.MAE()
rmse = cn.metrics.RMSE()

# Evaluation
avg_results, _ = eval_method.evaluate(model=mf, metrics=[mae, rmse], user_based=True)

print(avg_results)
