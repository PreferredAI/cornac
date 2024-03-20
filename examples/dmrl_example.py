"""Example for Disentangled Multimodal Recommendation, with only feedback and textual modality.
For an example including image modality please see dmrl_clothes_example.py"""

import cornac
from cornac.data import Reader
from cornac.datasets import citeulike
from cornac.eval_methods import RatioSplit
from cornac.data import TextModality

# The necessary data can be loaded as follows
docs, item_ids = citeulike.load_text()
feedback = citeulike.load_feedback(reader=Reader(item_set=item_ids))

item_text_modality = TextModality(
    corpus=docs,
    ids=item_ids,
)

# Define an evaluation method to split feedback into train and test sets
ratio_split = RatioSplit(
    data=feedback,
    test_size=0.2,
    exclude_unknowns=True,
    verbose=True,
    seed=123,
    rating_threshold=0.5,
    item_text=item_text_modality,
)

# Instantiate DMRL recommender
dmrl_recommender = cornac.models.dmrl.DMRL(
    batch_size=4096,
    epochs=20,
    log_metrics=False,
    learning_rate=0.01,
    num_factors=2,
    decay_r=0.5,
    decay_c=0.01,
    num_neg=3,
    embedding_dim=100,
)

# Use Recall@300 for evaluations
rec_300 = cornac.metrics.Recall(k=300)
prec_30 = cornac.metrics.Precision(k=30)

# Put everything together into an experiment and run it
cornac.Experiment(
    eval_method=ratio_split, models=[dmrl_recommender], metrics=[prec_30, rec_300]
).run()
