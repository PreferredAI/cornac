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

Semantic ID Evaluation
----------------------

``SemanticIDEvaluation`` evaluates TIGER, LETTER, RPG and DiffGRM through the same
raw Semantic ID predictions. It reports SID accuracy, analytical
Collision-Corrected Evaluation (CCE), and exact-item accuracy under explicit
resolvers. Supported metrics are ``Recall(k)`` and ``NDCG(k)`` with positive
integer cutoffs; result names include ``SID/Recall@10``, ``CCE/NDCG@10`` and
``Item[popularity]/Recall@10``. ``user_based=True`` averages queries per user
before averaging users.

CCE expands each complete ID into its full item bucket and averages over uniform
member permutations. Earlier buckets consume their full widths. Invalid and
duplicate addresses preserve empty rank slots. A resolver selects one item per
bucket while preserving address order. Complete IDs include any disambiguating
suffix; injective catalogs give identical views under the same slot rules.

Popularity uses training events only. Supplying ``item_features`` adds history
cosine: normalize item vectors, sum each user's training-event vectors (including
repeats), normalize that profile and score bucket members by cosine. Features
must align with global Cornac item indices. Ties, including zero profiles, choose
the smallest item index. Pass fitted resolvers through ``resolvers`` or implement
``SemanticIDResolver.score(query_index, user_index, candidate_items)``; custom
state must also use training-only information. ``resolve_semantic_ids`` exposes
the same operation for inference.

For an existing Semantic ID model and item-feature modality::

    evaluation = SemanticIDEvaluation.from_splits(
        train, test, fmt="USIT", item_feature=features)
    Experiment(evaluation, [TIGER(device="cpu")], [Recall(10), NDCG(10)]).run()

See ``examples/semantic_id_evaluation_example.py`` for a standalone CPU example.
``Result.semantic_id_trace`` holds the predictions, and
``Result.semantic_id_evaluation`` holds per-query arrays and protocol metadata.

Adapter authors subclass ``cornac.models.recommender.SemanticIDRecommender`` and
implement ``semantic_id_catalog()`` and ``predict_semantic_ids(history_items,
**context)``. Export complete addresses separately from tokenizer prefixes and
preserve raw decoder order, invalid slots and duplicates before item conversion.
The evaluator requests ``n_candidates=max(k)``; honor it where supported and keep
shorter output without fillers. ``prediction_from_catalog_scores`` supports exact
scoring by maximum member score per address, with lexicographic address ties.
The contract covers fixed-width integer addresses, a frozen catalog and one
next-item target per query.

.. automodule:: cornac.eval_methods.semantic_id_evaluation
   :members:

.. automodule:: cornac.eval_methods.sid_evaluation
   :members:

.. automodule:: cornac.eval_methods.sid_resolvers
   :members:
