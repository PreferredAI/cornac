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
"""Example for Collaborative Topic Regression"""

import cornac
from cornac.data import Reader
from cornac.datasets import citeulike
from cornac.eval_methods import RatioSplit
from cornac.data import TextModality
from cornac.data.text import BaseTokenizer


# CTR composes the LDA topic model with matrix factorization to model item (article) texts and user-item preferences
# The necessary data can be loaded as follows
docs, item_ids = citeulike.load_text()
feedback = citeulike.load_feedback(reader=Reader(item_set=item_ids))

# Instantiate a TextModality, it makes it convenient to work with text auxiliary information
# For more details, please refer to the tutorial on how to work with auxiliary data
item_text_modality = TextModality(
    corpus=docs,
    ids=item_ids,
    tokenizer=BaseTokenizer(sep=" ", stop_words="english"),
    max_vocab=8000,
    max_doc_freq=0.5,
)

# Define an evaluation method to split feedback into train and test sets
ratio_split = RatioSplit(
    data=feedback,
    test_size=0.2,
    exclude_unknowns=True,
    item_text=item_text_modality,
    verbose=True,
    seed=123,
    rating_threshold=0.5,
)

# Instantiate CTR model
ctr = cornac.models.CTR(k=50, max_iter=50, lambda_v=1)

# Use Recall@300 for evaluation
rec_300 = cornac.metrics.Recall(k=300)

# Put everything together into an experiment and run it
cornac.Experiment(eval_method=ratio_split, models=[ctr], metrics=[rec_300]).run()
