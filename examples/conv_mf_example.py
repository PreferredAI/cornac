# -*- coding: utf-8 -*-

"""
Example for Convolutional Matrix Factorization

@author: Tran Thanh Binh
"""

import cornac
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.data import TextModule
from cornac.data import BaseTokenizer

item_text = movielens.load_plot()
ml_1m = movielens.load_1m()
# remove items without plot
ml_1m = [(u, i, r) for (u, i, r) in ml_1m if i in item_text.keys()]

# build text module
item_text_module = TextModule(id_text=item_text, tokenizer=BaseTokenizer('\t'),
                              max_vocab=8000, max_doc_freq=0.5, stop_words='english')

ratio_split = RatioSplit(data=ml_1m, test_size=0.2, exclude_unknowns=True,
                         item_text=item_text_module, verbose=True)
convmf = cornac.models.ConvMF(n_epochs=50)
rmse = cornac.metrics.RMSE()
exp = cornac.Experiment(eval_method=ratio_split,
                        models=[convmf],
                        metrics=[rmse],
                        user_based=True)
exp.run()
