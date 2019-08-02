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
"""Example for HFT with Movilen 1m dataset """

import cornac
from cornac.data import Reader
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.data import TextModality
from cornac.data.text import BaseTokenizer

plots, movie_ids = movielens.load_plot()
ml_1m = movielens.load_1m(reader=Reader(item_set=movie_ids))

# build text module
item_text_modality = TextModality(corpus=plots, ids=movie_ids,
                                  tokenizer=BaseTokenizer(sep='\t', stop_words='english'),
                                  max_vocab=5000, max_doc_freq=0.5)

ratio_split = RatioSplit(data=ml_1m, test_size=0.2, exclude_unknowns=True,
                         item_text=item_text_modality, verbose=True, seed=123)

hft = cornac.models.HFT(k=10, max_iter=40, grad_iter=5, l2_reg=0.001, lambda_text=0.01, vocab_size=5000, seed=123)

mse = cornac.metrics.MSE()

exp = cornac.Experiment(eval_method=ratio_split,
                        models=[hft],
                        metrics=[mse],
                        user_based=False)
exp.run()
