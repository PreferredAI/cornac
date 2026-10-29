Evaluation Methods
==================

.. automodule:: cornac.eval_methods

Base Method
-----------
.. automodule:: cornac.eval_methods.base_method
   :members:
     
Cross Validation
----------------
.. automodule:: cornac.eval_methods.cross_validation
   :members:

Propensity Stratified Evaluation
--------------------------------
.. automodule:: cornac.eval_methods.propensity_stratified_evaluation
   :members:

Ratio Split
-----------
.. automodule:: cornac.eval_methods.ratio_split
   :members:

Stratified Split
----------------
.. automodule:: cornac.eval_methods.stratified_split
   :members:

Timestamp Split
---------------
.. automodule:: cornac.eval_methods.timestamp_split
   :members:

Next Item Evaluation
--------------------
.. automodule:: cornac.eval_methods.next_item_evaluation
   :members:

Recommendation Evaluation with Semantic IDs
------------------------------------------

Use ``NextItemEvaluation`` for both ordinary next-item models (such as SASRec)
and Semantic ID models (TIGER, LETTER, RPG and DiffGRM), including a mixed
``Experiment``. Its optional ``semantic_id_metrics`` switch defaults to False.
Set it to True in Semantic ID experiment setup to enable the extra views.
Ordinary item metric names and model-specific item resolution are preserved.
With the switch enabled, for each requested finite-K ``Recall`` or ``NDCG``,
SID models additionally report ``SID-Recall@K``, ``SID-NDCG@K``, ``CCE-Recall@K`` and ``CCE-NDCG@K``.
Other metrics, including AUC, MAP, MRR and unbounded cutoffs, still run their
ordinary item evaluation without additional views. Non-SID models have no SID
or CCE values; mixed result tables show ``N/A`` for those entries.

SID metrics match the target's complete generated ID, including any
item-disambiguating suffix. Invalid IDs, IDs absent from the admitted candidate
catalog, and repeated IDs after their first occurrence are removed before
ranking. Candidate filtering respects ``exclude_unknowns`` and also restricts
each collision bucket's size. No ungenerated fallback candidates are introduced.

Collision-Corrected Evaluation (CCE) expands surviving IDs in their original
order into their full admitted item buckets. Earlier buckets consume their full
widths. Items within a bucket have uniformly random order. CCE Recall is the
fraction of target-bucket positions inside K, and CCE NDCG averages the log
discount over those positions, divided by the full target-bucket size.

For example, suppose items a and b share the first predicted ID, c has the
second ID, and b is the target. A model that ranks a first has ordinary
Recall@1 of 0, SID-Recall@1 of 1 and CCE-Recall@1 of 0.5. CCE-Recall@2 is 1
because both positions of the target bucket fit. This is expected item credit,
not the accuracy of a concrete item resolver. A catalog whose complete IDs are
all unique yields identical SID and CCE metrics. TIGER's suffix-bearing IDs
have this property even when their tokenizer prefixes collide.

CCE follows Qian Zhang, Lech Szymanski, Haibo Zhang and Jeremiah D. Deng,
`Faithful Evaluation of Semantic-ID Tokenizers for Generative Recommendation
<https://arxiv.org/abs/2605.25330v2>`_ (CIKM 2026). Ordinary item metrics remain
the direct measure of the model's concrete recommendations; CCE evaluates the
uniform within-bucket assumption.

All three views share one model inference per query. Configured decoding budgets
are preserved. Short outputs are evaluated as returned, and one warning per
split reports the number of queries with fewer valid distinct IDs than the
largest requested SID/CCE cutoff. Increase the model's beam setting explicitly
if a larger candidate budget is desired. With ``semantic_id_metrics=False``,
only ordinary item metrics are reported; SID models issue a warning.
``mode='last'``/``mode='next'`` and per-query/per-user aggregation apply to every
view in the same way.

For example::

    evaluation = NextItemEvaluation.from_splits(
        train, test, fmt="USIT", item_feature=features,
        semantic_id_metrics=True)
    Experiment(evaluation, [TIGER(device="cpu"), SASRec()],
               [Recall(10), NDCG(10)]).run()

Adapter authors implement ``SemanticIDMixin`` alongside ``NextItemRecommender``.
``semantic_id_catalog()`` returns a two-dimensional integer array with one row
per global item, storing complete generated IDs. ``rank_with_semantic_ids`` has
the same inputs as ``rank`` and returns ordinary item ranks, item scores aligned
to the candidate indices, and a two-dimensional integer array of raw ranked SID
rows. Its full SID output is independent of item-ranking ``k``. The model keeps
its decoding and concrete SID-to-item policy. The mixin provides shared ranking
from ``_score_with_semantic_ids(user_idx, history_items, **kwargs)``, which returns
full item scores and SID rows from the same inference. Empty SID output has
shape ``(0, SID_length)``. No persistent prediction cache is required.

For exhaustive scoring, the built-in adapters order SID groups by their maximum
member score, with lexicographic ID ties. Beam adapters expose only generated
rows, preserving decoder order before evaluator filtering and compaction.
