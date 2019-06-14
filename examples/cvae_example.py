# -*- coding: utf-8 -*-

"""
Example for Collaborative Variational Autoencoder

@author: Tran Thanh Binh
"""

import cornac
from cornac.data import Reader
from cornac.datasets import citeulike
from cornac.eval_methods import RatioSplit
from cornac.data import TextModule

docs, item_ids = citeulike.load_text()
data = citeulike.load_data(reader=Reader(item_set=item_ids))

# build text module
item_text_module = TextModule(corpus=docs, ids=item_ids,
                              max_vocab=8000, max_doc_freq=0.5,
                              stop_words='english')

ratio_split = RatioSplit(data=data, test_size=0.2, exclude_unknowns=True,
                         rating_threshold=0.5, verbose=True, seed=123,
                         item_text=item_text_module)

cvae = cornac.models.CVAE(z_dim=50, vae_layers=[200, 100], act_fn='sigmoid',
                          input_dim=8000, lr=0.001, batch_size=128, n_epochs=100,
                          lambda_u=1e-4, lambda_v=0.001, lambda_r=10, lambda_w=1e-4,
                          seed=123, verbose=True)

rec_300 = cornac.metrics.Recall(k=300)

exp = cornac.Experiment(eval_method=ratio_split,
                        models=[cvae],
                        metrics=[rec_300])
exp.run()
