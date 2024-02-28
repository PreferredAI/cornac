"""Example for Collaborative Deep Learning (CDL)"""

import cornac
from cornac.data import Reader
from cornac.data.bert_text import BertTextModality
from cornac.datasets import citeulike
from cornac.eval_methods import RatioSplit


# CDL composes an autoencoder with matrix factorization to model item (article) texts and user-item preferences
# The necessary data can be loaded as follows
docs, item_ids = citeulike.load_text()
feedback = citeulike.load_feedback(reader=Reader(item_set=item_ids))

# Instantiate a TextModality, it makes it convenient to work with text auxiliary information
# For more details, please refer to the tutorial on how to work with auxiliary data
item_text_modality = BertTextModality(
    corpus=docs,
    ids=item_ids,
    preencode=True
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

# Instantiate DMRL recommender
dmrl_recommender = cornac.models.dmrl.DMRL(
    iid_map = ratio_split.global_iid_map,
    num_users = ratio_split.total_users,
    num_items = ratio_split.total_items,
    bert_text_modality = item_text_modality,
    batch_size=2048,
    epochs=40,
    log_metrics=True,
    learning_rate=0.01,
    num_factors=1,
    decay_r=0.02,
    num_neg=2,
    embedding_dim=100)

# Use Recall@300 for evaluations
rec_300 = cornac.metrics.Recall(k=300)
prec_300 = cornac.metrics.Precision(k=300)

# Put everything together into an experiment and run it
cornac.Experiment(eval_method=ratio_split, models=[dmrl_recommender], metrics=[prec_300, rec_300]).run()
