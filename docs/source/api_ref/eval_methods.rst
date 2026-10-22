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

Recommendation Evaluation with Semantic IDs
------------------------------------------

``SemanticIDEvaluation`` subclasses ``NextItemEvaluation`` and evaluates next-item
recommendation performance, rather than the intrinsic quality of Semantic IDs.
Use it when a generated ID can map to several items: SID matching alone cannot
identify which member was recommended. ``NextItemEvaluation`` instead evaluates
the item rankings returned by ``model.rank()``.

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

CCE follows Qian Zhang, Lech Szymanski, Haibo Zhang and Jeremiah D. Deng,
`Faithful Evaluation of Semantic-ID Tokenizers for Generative Recommendation
<https://arxiv.org/abs/2605.25330v2>`_ (CIKM 2026). The explicit resolvers below
are selectable item-ranking policies, distinct from CCE's expectation.

This evaluator requires ``SemanticIDRecommender`` and supports SID models only.
For ordinary next-item models such as SASRec, use ``NextItemEvaluation`` in a
separate experiment with identical train/validation/test splits and cutoffs.
Mixed SID and ordinary models in one ``Experiment`` are not supported.

Popularity uses training events only. Supplying ``item_features`` adds history
cosine: normalize item vectors, sum each user's training-event vectors (including
repeats), normalize that profile and score bucket members by cosine. Features
must align with global Cornac item indices. Ties, including zero profiles, choose
the smallest item index. Pass fitted resolvers through ``resolvers`` or implement
``SemanticIDResolver.score(query_index, user_index, candidate_items)``; custom
state must also use training-only information. Import these utilities from
``cornac.utils.semantic_id``; ``resolve_semantic_ids`` exposes the same operation
for inference. For example::

    import numpy as np
    from cornac.utils.semantic_id import (
        SemanticIDCatalog, SemanticIDPrediction, TrainingPopularity,
        resolve_semantic_ids,
    )

    codes = np.array([[0], [0], [1]])
    catalog = SemanticIDCatalog(np.arange(3), codes, codes, (2,), (2,))
    prediction = SemanticIDPrediction(
        np.array([[0], [1]]), np.array([1., .5]), np.array([True, True]))
    resolver = TrainingPopularity([1, 1, 0, 2], num_items=3)
    resolve_semantic_ids(catalog, prediction, resolver)  # (1, 2)

Items 0 and 1 share the first ID. Popularity chooses item 1 from that bucket,
then item 2 from the next, preserving generated ID order. If item 1 is the
target, SID Recall@1 is 1, CCE Recall@1 is 0.5 and popularity-resolved Recall@1
is 1. CCE Recall@2 is 1 because both members of the first bucket fit. Other
resolvers can choose differently while using the same predictions.

For an existing Semantic ID model and item-feature modality::

    evaluation = SemanticIDEvaluation.from_splits(
        train, test, fmt="USIT", item_feature=features)
    Experiment(evaluation, [TIGER(device="cpu")], [Recall(10), NDCG(10)]).run()

See ``examples/semantic_id_evaluation_example.py`` for a standalone CPU example.
``Result.semantic_id_trace`` holds the predictions, and
``Result.semantic_id_evaluation`` holds per-query arrays and protocol metadata.

Adapter authors subclass ``cornac.models.recommender.SemanticIDRecommender`` and
implement ``semantic_id_catalog()`` and ``predict_semantic_ids(history_items,
**context)``. Use the shared types in ``cornac.utils.semantic_id``. Export
complete addresses separately from tokenizer prefixes and
preserve raw decoder order, invalid slots and duplicates before item conversion.
The evaluator requests ``n_candidates=max(k)``; honor it where supported and keep
shorter output without fillers. ``prediction_from_catalog_scores`` supports exact
scoring by maximum member score per address, with lexicographic address ties.
The contract covers fixed-width integer addresses, a frozen catalog and one
next-item target per query.

The evaluation module combines trace collection, offline trace evaluation via
``evaluate_semantic_id_trace`` and the ``SemanticIDEvaluation`` experiment
integration. Utilities below provide catalogs, predictions and item resolution.

.. automodule:: cornac.eval_methods.semantic_id_evaluation
   :members:

.. automodule:: cornac.utils.semantic_id
   :members:
