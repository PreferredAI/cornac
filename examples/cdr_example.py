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
"""Example for Collaborative Deep Ranking"""

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
                              tokenizer=BaseTokenizer(stop_words='english'),
                              max_vocab=8000, max_doc_freq=0.5)

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
