# -*- coding: utf-8 -*-

"""
Example for Convolutional Matrix Factorization

@author: Tran Thanh Binh
"""

import cornac
from cornac.data import Reader
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.data import TextModule
from cornac.data.text import BaseTokenizer

plots, movie_ids = movielens.load_plot()
ml_1m = movielens.load_1m(reader=Reader(item_set=movie_ids))

# build text module
item_text_module = TextModule(corpus=plots, ids=movie_ids,
                              tokenizer=BaseTokenizer('\t'),
                              max_vocab=8000, max_doc_freq=0.5, stop_words='english')

ratio_split = RatioSplit(data=ml_1m, test_size=0.2, exclude_unknowns=True,
                         item_text=item_text_module, verbose=True, seed=123)

convmf = cornac.models.ConvMF(n_epochs=5, verbose=True, seed=123)

rmse = cornac.metrics.RMSE()

exp = cornac.Experiment(eval_method=ratio_split,
                        models=[convmf],
                        metrics=[rmse],
                        user_based=True)
exp.run()
