# Note: This is a synthetic example purely for demonstrating the API usage.
# MovieLens ratings < 4 are mapped to 'views'. In reality, low ratings indicate negative
# preference rather than intermediate interest, which may cause VEBPR to underperform
# standard BPR here. For real performance gains, use sparse E-commerce datasets with true view logs.

import cornac
from cornac.data import PurchaseViewDataset
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.models import BPR, VEBPR
from cornac.metrics import NDCG, Recall

ml_100k = movielens.load_feedback()
purchase_data = [(u, i, 1.0) for u, i, r in ml_100k if r >= 4.0]
view_data = [(u, i, 1.0) for u, i, r in ml_100k if r < 4.0]

eval_method = RatioSplit(
    data=purchase_data, test_size=0.2, seed=123, exclude_unknowns=True
)
eval_method.train_set = PurchaseViewDataset.attach_view(
    eval_method.train_set, view_data
)

shared_params = dict(
    k=50, max_iter=1000, learning_rate=0.01, lambda_reg=0.01, seed=123, verbose=True
)
cornac.Experiment(
    eval_method=eval_method,
    models=[BPR(**shared_params), VEBPR(alpha=0.5, **shared_params)],
    metrics=[Recall(k=50), NDCG(k=50)],
).run()
