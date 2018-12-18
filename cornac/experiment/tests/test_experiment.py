# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""


def test_Experiment():
    from ...data.reader import txt_to_triplets
    from ...eval_strategies.ratio_split import RatioSplit
    from ...models.pmf.recom_pmf import PMF
    from ...metrics.rating import MAE, RMSE
    from ...metrics.ranking import Recall, FMeasure
    from ..experiment import Experiment

    data = txt_to_triplets('./cornac/data/tests/data.txt')
    exp = Experiment(eval_strategy=RatioSplit(data),
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