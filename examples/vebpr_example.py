
# Note: This is a synthetic example purely for demonstrating the API usage.
# MovieLens ratings < 4 are mapped to 'views'. In reality, low ratings indicate negative
# preference rather than intermediate interest, which may cause VEBPR to underperform
# standard BPR here. For real performance gains, use sparse E-commerce datasets with true view logs.

import cornac
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.models import BPR, VEBPR
from cornac.metrics import NDCG, Recall
import numpy as np
import scipy.sparse as sp

ml_100k = movielens.load_feedback()

purchase_data = []
view_data = []

for user, item, rating in ml_100k:
    if rating >= 4.0:
        purchase_data.append((user, item, 1.0))
    else:
        view_data.append((user, item, 1.0))



eval_method = RatioSplit(
    data=purchase_data,
    test_size=0.2,
    rating_threshold=0.5,
    seed=123,
    exclude_unknowns=True,
    verbose=True
)
train_set = eval_method.train_set


row_indices = []
col_indices = []

for raw_u, raw_i, _ in view_data:
    if raw_u in train_set.uid_map and raw_i in train_set.iid_map:
        inner_u = train_set.uid_map[raw_u]
        inner_i = train_set.iid_map[raw_i]
        row_indices.append(inner_u)
        col_indices.append(inner_i)

V_matrix = sp.csr_matrix(
    (np.ones(len(row_indices)), (row_indices, col_indices)),
    shape=(train_set.num_users, train_set.num_items)
)

bpr_baseline = BPR(
    name="BPR",
    k=50,
    max_iter=1000,
    learning_rate=0.01,
    lambda_reg=0.01,
    seed=123
)

vebpr_model = VEBPR(
    name="VEBPR",
    k=50,
    max_iter=1000,
    learning_rate=0.01,
    lambda_reg=0.01,
    alpha=0.5,
    view_matrix=V_matrix,
    seed=123
)

cornac.Experiment(
    eval_method=eval_method,
    models=[bpr_baseline, vebpr_model],
    metrics=[Recall(k=50), NDCG(k=50)],
).run()