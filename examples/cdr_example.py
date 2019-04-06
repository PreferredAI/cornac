# -*- coding: utf-8 -*-

"""
Example for Collaborative Deep Ranking

@author: Tran Thanh Binh
"""

import cornac
from cornac.data import Reader
from cornac.datasets import citeulike
from cornac.eval_methods import RatioSplit
from cornac.data import TextModule
from cornac.data.text import BaseTokenizer

docs, item_ids = citeulike.load_text()
data = citeulike.load_data(reader=Reader(item_set=item_ids))

# build text module
item_text_module = TextModule(corpus=docs, ids=item_ids,
                              tokenizer=BaseTokenizer('\t'),
                              max_vocab=8000, max_doc_freq=0.5,
                              stop_words='english')

ratio_split = RatioSplit(data=data, test_size=0.2, exclude_unknowns=True,
                         item_text=item_text_module, verbose=True, seed=123, rating_threshold=0.5)

cdr = cornac.models.CDR(k=50, autoencoder_structure=[200], max_iter=100, batch_size=128,
                          lambda_u=0.01, lambda_v=0.1, lambda_w=0.0001, lambda_n=5,
                          learning_rate=0.001, vocab_size=8000)

rec_300 = cornac.metrics.Recall(k=300)

exp = cornac.Experiment(eval_method=ratio_split,
                        models=[cdr],
                        metrics=[rec_300])
exp.run()
