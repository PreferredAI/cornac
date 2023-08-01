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

import cornac
from cornac.data import Reader
from cornac.datasets import amazon_digital_music
from cornac.eval_methods import RatioSplit
from cornac.data import ReviewModality
from cornac.data.text import BaseTokenizer


feedback = amazon_digital_music.load_feedback()
reviews = amazon_digital_music.load_review()


review_modality = ReviewModality(
    data=reviews,
    tokenizer=BaseTokenizer(stop_words="english"),
    max_vocab=4000,
    max_doc_freq=0.5,
)

ratio_split = RatioSplit(
    data=feedback,
    test_size=0.1,
    val_size=0.1,
    exclude_unknowns=True,
    review_text=review_modality,
    verbose=True,
    seed=123,
)

pretrained_word_embeddings = {}  # You can load pretrained word embedding here

model = cornac.models.HRDR(
    embedding_size=100,
    id_embedding_size=32,
    n_factors=32,
    attention_size=16,
    kernel_sizes=[3],
    n_filters=64,
    n_user_mlp_factors=128,
    n_item_mlp_factors=128,
    dropout_rate=0.5,
    max_text_length=50,
    batch_size=64,
    max_iter=10,
    init_params={'pretrained_word_embeddings': pretrained_word_embeddings},
    verbose=True,
    seed=123,
)

cornac.Experiment(
    eval_method=ratio_split, models=[model], metrics=[cornac.metrics.RMSE()]
).run()
