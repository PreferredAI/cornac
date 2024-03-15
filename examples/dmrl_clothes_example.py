"""
Example for Disentangled Multimodal Recommendation, with feedback, textual and visual modality.
This example uses preencoded visual features from cornac dataset instead of TransformersVisionModality modality.
"""

import cornac
from cornac.datasets import amazon_clothing
from cornac.eval_methods import RatioSplit
from cornac.models.dmrl.recom_dmrl import ImageModalityInput, TextModalityInput


feedback = amazon_clothing.load_feedback()
encoded_image_features, item_id_ordering_image = amazon_clothing.load_visual_feature()  # BIG file
docs, item_id_ordering_text = amazon_clothing.load_text()

# only treat good feedback as positive user-item pair
new_feedback = [f for f in feedback if f[2] >=4]

ratio_split = RatioSplit(
    data=new_feedback,
    test_size=0.25,
    exclude_unknowns=True,
    verbose=True,
    seed=123,
    rating_threshold=4,
)

text_modality_input = TextModalityInput(item_id_ordering_text, docs)
image_modality_input = ImageModalityInput(item_id_ordering_image, preencoded_image_features=encoded_image_features)


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
    dropout=0,
    text_features=text_modality_input,
    image_features=image_modality_input)


# Use Recall@300 for evaluations
rec_300 = cornac.metrics.Recall(k=300)
rec_900 = cornac.metrics.Recall(k=900)
prec_30 = cornac.metrics.Precision(k=30)

# Put everything together into an experiment and run it
cornac.Experiment(eval_method=ratio_split, models=[dmrl_recommender], metrics=[prec_30, rec_300, rec_900]).run()
