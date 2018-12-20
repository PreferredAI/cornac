# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import random, time


def test_movielens_100k():
    # only run data download tests 10% of the time to speed up frequent testing
    random.seed(time.time())
    if random.random() > 0.9:
        from ..movielens import MovieLens100K
        ml_100k = MovieLens100K.load_data()
        assert len(ml_100k) == 100000


def test_movielens_1m():
    # only run data download tests 10% of the time to speed up frequent testing
    random.seed(time.time())
    if random.random() > 0.9:
        from ..movielens import MovieLens1M
        ml_1m = MovieLens1M.load_data()
        assert len(ml_1m) == 1000209