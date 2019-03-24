# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import unittest
from cornac.data import reader
from cornac.eval_methods import RatioSplit, CrossValidation
from cornac.models import PMF
from cornac.metrics import MAE, RMSE, Recall, FMeasure
from cornac.experiment.experiment import Experiment


class TestExperiment(unittest.TestCase):

    def test_with_ratio_split(self):
        data_file = './tests/data.txt'
        data = reader.read_uir(data_file)
        exp = Experiment(eval_method=RatioSplit(data, verbose=True),
                         models=[PMF(1, 0)],
                         metrics=[MAE(), RMSE(), Recall(1), FMeasure(1)],
                         verbose=True)
        exp.run()

        try:
            Experiment(None, None, None)
        except ValueError:
            assert True

        try:
            Experiment(None, [PMF(1, 0)], None)
        except ValueError:
            assert True

    def test_with_cross_validation(self):
        data_file = './tests/data.txt'
        data = reader.read_uir(data_file)
        exp = Experiment(eval_method=CrossValidation(data),
                         models=[PMF(1, 0)],
                         metrics=[MAE(), RMSE(), Recall(1), FMeasure(1)],
                         verbose=True)
        exp.run()


if __name__ == '__main__':
    unittest.main()
