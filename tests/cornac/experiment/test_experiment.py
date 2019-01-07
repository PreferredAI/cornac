# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import os
from cornac.data.reader import Reader
from cornac.eval_methods import RatioSplit
from cornac.models import PMF
from cornac.metrics import MAE, RMSE, Recall, FMeasure
from cornac.experiment.experiment import Experiment


def test_with_ratio_split():
    data_file = './tests/data.txt'
    data = Reader.read_uir_triplets(data_file)
    exp = Experiment(eval_method=RatioSplit(data, verbose=True),
                     models=[PMF(1, 0)],
                     metrics=[MAE(), RMSE(), Recall(1), FMeasure(1)],
                     verbose=True)
    exp.run()

    assert (1, 4) == exp.avg_results.shape

    assert 1 == len(exp.user_results)
    assert 4 == len(exp.user_results['PMF'])
    assert 2 == len(exp.user_results['PMF']['MAE'])
    assert 2 == len(exp.user_results['PMF']['RMSE'])
    assert 2 == len(exp.user_results['PMF']['Recall@1'])
    assert 2 == len(exp.user_results['PMF']['F1@1'])

    try:
        Experiment(None, None, None)
    except ValueError:
        assert True

    try:
        Experiment(None, [PMF(1, 0)], None)
    except ValueError:
        assert True