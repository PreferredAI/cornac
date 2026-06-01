# Copyright 2026 The Cornac Authors. All Rights Reserved.
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
"""Validation scoring helper shared by sequential models for best-on-val
model selection (``model_selection="best"``)."""


def val_score(model, train_set, val_set, metric="recall", k=20):
    """Compute a next-item ranking metric on ``val_set`` during training.

    Delegates to :func:`cornac.eval_methods.next_item_evaluation.ranking_eval`
    so the validation score matches the eventual test protocol, letting a
    model keep the best-on-val checkpoint across epochs.

    Parameters
    ----------
    model: :obj:`cornac.models.NextItemRecommender`
        The model being trained (must implement ``score``).

    train_set, val_set: :obj:`cornac.data.SequentialDataset`
        Training set (to exclude seen items) and the validation set.

    metric: str, optional, default: 'recall'
        One of ``'recall'``, ``'ndcg'``, ``'auc'``, ``'mrr'``
        (case-insensitive). ``k`` is ignored for ``'auc'`` and ``'mrr'``.

    Returns
    -------
    float or None
        The averaged metric value, or ``None`` if ``val_set`` is ``None``.
    """
    if val_set is None:
        return None

    from ...eval_methods.next_item_evaluation import ranking_eval
    from ...metrics import AUC, MRR, NDCG, Recall

    name = metric.lower()
    if name == "recall":
        m = Recall(k=k)
    elif name == "ndcg":
        m = NDCG(k=k)
    elif name == "auc":
        m = AUC()
    elif name == "mrr":
        m = MRR()
    else:
        raise ValueError(
            f"val_metric='{metric}' not supported; choose from recall/ndcg/auc/mrr"
        )

    avg, _ = ranking_eval(
        model=model,
        metrics=[m],
        train_set=train_set,
        test_set=val_set,
        exclude_unknowns=True,
        verbose=False,
    )
    return float(avg[0]) if avg else 0.0
