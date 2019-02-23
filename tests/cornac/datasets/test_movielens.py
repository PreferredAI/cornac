# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import random, time
from cornac.datasets import movielens


def test_movielens_100k():
    # only run data download tests 20% of the time to speed up frequent testing
    random.seed(time.time())
    if random.random() > 0.8:
        ml_100k = movielens.load_100k()
        assert len(ml_100k) == 100000


def test_movielens_1m():
    # only run data download tests 20% of the time to speed up frequent testing
    random.seed(time.time())
    if random.random() > 0.8:
        ml_1m = movielens.load_1m()
        assert len(ml_1m) == 1000209
