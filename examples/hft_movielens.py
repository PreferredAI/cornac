# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Example for Collaborative Topic Modeling"""

from collections import defaultdict

import cornac
from cornac.data import Reader
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.data import TextModality
from cornac.data.text import BaseTokenizer

plots, movie_ids = movielens.load_plot()

# movie_plot = {}
# for plot, movie_id in  zip(plots, movie_ids):
#     movie_plot[movie_id] = plot

ml_1m = movielens.load_1m(reader=Reader(item_set=movie_ids))
# checker = defaultdict(bool)
#
# with open('data.txt', 'w') as f:
#     for u, i, r in ml_1m:
#         text = movie_plot[i].split('\t')
#         if checker[i]:
#             text = []
#         else:
#             checker[i] = True
#         f.write('{} {} {} {} {} {}\n'.format(u, i, r, 0, len(text), ' '.join(text)))
#
# build text module
item_text_modality = TextModality(corpus=plots, ids=movie_ids,
                              tokenizer=BaseTokenizer(sep='\t', stop_words='english'),
                              max_vocab=8000, max_doc_freq=0.5)

ratio_split = RatioSplit(data=ml_1m, test_size=0.2, exclude_unknowns=True,
                         item_text=item_text_modality, verbose=True, seed=123)

hft = cornac.models.HFT(max_iter=10, grad_iter=20, lambda_reg=0.0, latent_reg=0.0, seed=123)

mse = cornac.metrics.MSE()

exp = cornac.Experiment(eval_method=ratio_split,
                        models=[hft],
                        metrics=[mse],
                        user_based=False)
exp.run()
