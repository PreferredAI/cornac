# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""


def test_with_ratio_split():
    from ...data import reader
    from ...eval_strategies import RatioSplit
    from ...models import PMF
    from ...metrics import MAE, RMSE, Recall, FMeasure
    from ..experiment import Experiment

    data = reader.txt_to_uir_triplets('./cornac/data/tests/data.txt')
    exp = Experiment(eval_strategy=RatioSplit(data, verbose=True),
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