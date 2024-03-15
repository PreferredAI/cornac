"""Example for Disentangled Multimodal Recommendation, with only feedback and textual modality.
For an example including image modality please see dmrl_clothes_example.py"""

import cornac
from cornac.data import Reader
from cornac.datasets import citeulike
from cornac.eval_methods import RatioSplit
from cornac.models.dmrl.recom_dmrl import TextModalityInput

# The necessary data can be loaded as follows
docs, item_id_ordering_text = citeulike.load_text()
feedback = citeulike.load_feedback(reader=Reader(item_set=item_id_ordering_text))


# Define an evaluation method to split feedback into train and test sets
ratio_split = RatioSplit(
    data=feedback,
    test_size=0.2,
    exclude_unknowns=True,
    verbose=True,
    seed=123,
    rating_threshold=0.5,
)

text_modality_input = TextModalityInput(item_id_ordering_text, docs)

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
    text_features=text_modality_input)

# Use Recall@300 for evaluations
rec_300 = cornac.metrics.Recall(k=300)
prec_30 = cornac.metrics.Precision(k=30)

# Put everything together into an experiment and run it
cornac.Experiment(eval_method=ratio_split, models=[dmrl_recommender], metrics=[prec_30, rec_300]).run()
