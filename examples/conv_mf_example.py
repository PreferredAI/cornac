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
from cornac.data import TextModality
from cornac.data.text import BaseTokenizer


# ConvMF extends matrix factorization to leverage item textual information
# The necessary data can be loaded as follows
plots, movie_ids = movielens.load_plot()
ml_1m = movielens.load_feedback(variant="1M", reader=Reader(item_set=movie_ids))

# Instantiate a TextModality, it makes it convenient to work with text auxiliary information
# For more details, please refer to the tutorial on how to work with auxiliary data
item_text_modality = TextModality(
    corpus=plots,
    ids=movie_ids,
    tokenizer=BaseTokenizer(sep="\t", stop_words="english"),
    max_vocab=8000,
    max_doc_freq=0.5,
)

# Define an evaluation method to split feedback into train and test sets
ratio_split = RatioSplit(
    data=ml_1m,
    test_size=0.2,
    exclude_unknowns=True,
    item_text=item_text_modality,
    verbose=True,
    seed=123,
)

# Instantiate ConvMF model
convmf = cornac.models.ConvMF(n_epochs=5, verbose=True, seed=123)

# Instantiate RMSE for evaluation
rmse = cornac.metrics.RMSE()

# Put everything together into an experiment and run it
cornac.Experiment(
    eval_method=ratio_split, models=[convmf], metrics=[rmse], user_based=True
).run()
