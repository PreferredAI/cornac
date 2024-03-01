import sys
sys.path.append('/workspaces/cornac')

import cornac
from cornac.data.bert_text import BertTextModality
from cornac.datasets import amazon_clothing
from cornac.data import ImageModality
from cornac.eval_methods import RatioSplit



feedback = amazon_clothing.load_feedback()
features, item_ids = amazon_clothing.load_visual_feature()  # BIG file
docs, item_ids = amazon_clothing.load_text()


# Instantiate a ImageModality, it makes it convenient to work with visual auxiliary information
# For more details, please refer to the tutorial on how to work with auxiliary data
item_image_modality = ImageModality(features=features, ids=item_ids, normalized=True)

item_image_modality = ImageModality(features=features, ids=item_ids, normalized=True)
item_text_modality = BertTextModality(
    corpus=docs,
    ids=item_ids,
    preencode=True
)

ratio_split = RatioSplit(
    data=feedback,
    test_size=0.2,
    exclude_unknowns=True,
    item_text=item_text_modality,
    item_image=item_image_modality,
    verbose=True,
    seed=123,
    rating_threshold=0.5,
)

dmrl_recommender = cornac.models.dmrl.DMRL(
    batch_size=256,
    epochs=20,
    log_metrics=True,
    learning_rate=0.01,
    num_factors=1,
    decay_r=2,
    decay_c=0.01,
    num_neg=3,
    embedding_dim=100,
    image_dim=4096)


# Use Recall@300 for evaluations
rec_300 = cornac.metrics.Recall(k=300)
prec_30 = cornac.metrics.Precision(k=30)

# Put everything together into an experiment and run it
cornac.Experiment(eval_method=ratio_split, models=[dmrl_recommender], metrics=[prec_30, rec_300]).run()
