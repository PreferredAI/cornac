# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import random, time
from cornac.datasets import netflix


def test_load_data_small():
    # only run data download tests 20% of the time to speed up frequent testing
    random.seed(time.time())
    if random.random() > 0.8:
        data = netflix.load_data_small()
        assert len(data) == 607803