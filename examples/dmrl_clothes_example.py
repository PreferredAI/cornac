"""
Example for Disentangled Multimodal Recommendation, with feedback, textual and visual modality.
This example uses preencoded visual features from cornac dataset instead of TransformersVisionModality modality.
"""


import sys
sys.path.append('/workspaces/cornac')

import cornac
from cornac.data.transformer_text import TransformersTextModality
from cornac.datasets import amazon_clothing
from cornac.data import ImageModality
from cornac.eval_methods import RatioSplit



feedback = amazon_clothing.load_feedback()
features, item_ids = amazon_clothing.load_visual_feature()  # BIG file
docs, item_ids = amazon_clothing.load_text()

# only treat good feedback as positive user-item pair
new_feedback = [f for f in feedback if f[2] >=4]

# Instantiate a ImageModality, it makes it convenient to work with visual auxiliary information
# For more details, please refer to the tutorial on how to work with auxiliary data
item_image_modality = ImageModality(features=features, ids=item_ids, normalized=True)

item_text_modality = TransformersTextModality(
    corpus=docs,
    ids=item_ids,
    preencode=True
)

ratio_split = RatioSplit(
    data=new_feedback,
    test_size=0.25,
    exclude_unknowns=True,
    item_text=item_text_modality,
    item_image=item_image_modality,
    verbose=True,
    seed=123,
    rating_threshold=4,
)

dmrl_recommender = cornac.models.dmrl.DMRL(
    batch_size=1024,
    epochs=60,
    log_metrics=False,
    learning_rate=0.001,
    num_factors=2,
    decay_r=2,
    decay_c=0.1,
    num_neg=5,
    embedding_dim=100,
    image_dim=4096,
    dropout=0)


# Use Recall@300 for evaluations
rec_300 = cornac.metrics.Recall(k=300)
rec_900 = cornac.metrics.Recall(k=900)
prec_30 = cornac.metrics.Precision(k=30)

# Put everything together into an experiment and run it
cornac.Experiment(eval_method=ratio_split, models=[dmrl_recommender], metrics=[prec_30, rec_300, rec_900]).run()
