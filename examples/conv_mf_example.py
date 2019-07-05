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
"""Example for Convolutional Matrix Factorization"""

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
                              tokenizer=BaseTokenizer(sep='\t', stop_words='english'),
                              max_vocab=8000, max_doc_freq=0.5)

ratio_split = RatioSplit(data=ml_1m, test_size=0.2, exclude_unknowns=True,
                         item_text=item_text_module, verbose=True, seed=123)

convmf = cornac.models.ConvMF(n_epochs=5, verbose=True, seed=123)

rmse = cornac.metrics.RMSE()

exp = cornac.Experiment(eval_method=ratio_split,
                        models=[convmf],
                        metrics=[rmse],
                        user_based=True)
exp.run()
