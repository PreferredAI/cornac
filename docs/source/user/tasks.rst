Models and Tasks
================

This page shows you the recommendation tasks you can run with Cornac. For each
task, it explains the data you need, how to evaluate models, which models to
use, and gives an example to start from.

What can you do with Cornac?
----------------------------

Cornac is a recommender systems framework that provides a wide range of
recommender models, evaluation metrics, and experimental tools.
It is designed to be flexible and extensible, allowing researchers to
easily conduct experiments and compare their models with existing ones.

Cornac is written in Python and is built on top of the popular scientific
computing libraries such as NumPy and SciPy. It is also designed to be compatible
with the popular deep learning libraries such as PyTorch and TensorFlow.

View the models, datasets, metrics that are currently built into Cornac:

- :doc:`/api_ref/models`
- :doc:`/api_ref/datasets`
- :doc:`/api_ref/metrics`

:doc:`/models/index` lists every model by type, such as Next-Item or
Next-Basket, and the `model table <https://github.com/PreferredAI/cornac#models>`_
in the README links each model's paper and example script. The tasks are:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Task
     - Data
     - Evaluation method
   * - `Rating prediction and top-N ranking`_
     - ``(user, item, rating)``
     - :class:`~cornac.eval_methods.ratio_split.RatioSplit` and others
   * - `Next-item recommendation`_
     - ``(user, session, item, timestamp)``
     - :class:`~cornac.eval_methods.next_item_evaluation.NextItemEvaluation`
   * - `Next-basket recommendation`_
     - ``(user, basket, item, timestamp)``
     - :class:`~cornac.eval_methods.next_basket_evaluation.NextBasketEvaluation`
   * - `Generative recommendation with Semantic IDs`_
     - ``(user, item, rating, timestamp)`` and item content
     - :class:`~cornac.eval_methods.next_item_evaluation.NextItemEvaluation`
   * - `Bundle recommendation, completion and construction`_
     - ``(user, bundle)`` and ``(bundle, item)``
     - :class:`~cornac.eval_methods.bundle_evaluation.BundleEvaluation`,
       :class:`~cornac.eval_methods.bundle_evaluation.BundleConstructionEvaluation`

Rating prediction and top-N ranking
-----------------------------------

Most Cornac models learn from ``(user, item, rating)`` tuples. Some predict
the rating a user would give an item; others rank the items for each user and
recommend the top N. Ratings can be explicit, such as 1 to 5 stars, or
implicit, such as 1 for every click or purchase. :doc:`/user/owndata` shows how
to load such data from a built-in dataset or from your own files.

To evaluate these models, split the data with
:class:`~cornac.eval_methods.ratio_split.RatioSplit`,
:class:`~cornac.eval_methods.stratified_split.StratifiedSplit`,
:class:`~cornac.eval_methods.timestamp_split.TimestampSplit` or
:class:`~cornac.eval_methods.cross_validation.CrossValidation`, and measure
them with rating metrics (``MAE``, ``RMSE``) or ranking metrics
(``Precision``, ``Recall``, ``NDCG`` and others). :doc:`/user/evaluation`
explains each one.

Models for this task include matrix factorization (``MF``, ``PMF``, ``SVD``,
``NMF``), ranking models (``BPR``, ``WMF``, ``EASE``), neighborhood models
(``ItemKNN``, ``UserKNN``), and neural models (``VAECF``, ``RecVAE``,
``LightGCN``). Many others also learn from side information: item text
(``CTR``, ``CDL``, ``ConvMF``), images (``VBPR``), item or user graphs
(``MCF``, ``SoRec``, ``C2PF``) and reviews (``HFT``, ``NARRE``, ``HRDR``).
The *Type* column of :doc:`/models/index` tells them apart.

The :doc:`/user/quickstart` is a complete example: it trains BPR and PMF on
MovieLens 100K and compares their Precision and Recall. To add side
information, see the CTR example in :doc:`/user/owndata` and the `Working with
Auxiliary Data
<https://github.com/PreferredAI/cornac/blob/master/tutorials/working_with_auxiliary_data.md>`_
tutorial.

Next-item recommendation
------------------------

Sometimes what a user wants next depends on what they just did, such as the
next product to view in a shopping session. A next-item model looks at the
items in a user's current session and recommends the item that comes next.
The data is a list of ``(user, session, item, timestamp)`` tuples, in the
order the events happened. The Diginetica dataset comes already split into
training, validation and test sessions:

.. code-block:: python

    import cornac
    from cornac.datasets import diginetica
    from cornac.eval_methods import NextItemEvaluation
    from cornac.metrics import MRR, NDCG, Recall
    from cornac.models import GRU4Rec, SPop

    train_data = diginetica.load_train()
    val_data = diginetica.load_val()
    test_data = diginetica.load_test()
    print(train_data[:3])

    next_item_eval = NextItemEvaluation.from_splits(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        fmt="USIT",
        exclude_unknowns=True,
    )

    models = [
        SPop(),
        GRU4Rec(batch_size=64, learning_rate=0.1, n_epochs=10, device="cpu", seed=123),
    ]

    metrics = [NDCG(k=10), Recall(k=10), MRR()]

    cornac.Experiment(eval_method=next_item_eval, models=models, metrics=metrics).run()

.. code-block:: bash
    :caption: output

    [('3', '60', '22128', 1462723201), ('3', '60', '22128', 1462723202), ('109', '435', '206092', 1459526401)]

    VALIDATION:
    ...
            |    MRR | NDCG@10 | Recall@10 | Time (s)
    ------- + ------ + ------- + --------- + --------
    SPop    | 0.2376 |  0.2490 |    0.2926 |   0.1857
    GRU4Rec | 0.2820 |  0.3134 |    0.4244 |   0.1813

    TEST:
    ...
            |    MRR | NDCG@10 | Recall@10 | Train (s) | Test (s)
    ------- + ------ + ------- + --------- + --------- + --------
    SPop    | 0.2344 |  0.2432 |    0.2776 |    0.0005 |   0.1830
    GRU4Rec | 0.2587 |  0.2912 |    0.4114 |   32.9294 |   0.1837

:class:`~cornac.models.spop.recom_spop.SPop` recommends the most popular
items, boosting those already in the session.
:class:`~cornac.models.gru4rec.recom_gru4rec.GRU4Rec` is a recurrent neural
network that learns from the order of items, and it beats SPop on every metric.
It needs PyTorch (``pip install -r cornac/models/gru4rec/requirements.txt`` from a
clone of the repository),
and with ``device="cuda"`` it trains on a GPU. Ten epochs take under a minute
on a CPU; the `GRU4Rec example
<https://github.com/PreferredAI/cornac/blob/master/examples/gru4rec_yoochoose.py>`_
trains it on the larger YooChoose data.

How next-item evaluation works
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~cornac.eval_methods.next_item_evaluation.NextItemEvaluation` builds a
:class:`~cornac.data.dataset.SequentialDataset` for each split. For every test
session, it hides the last item and asks the model to rank all items given the
earlier ones (``mode="last"``, the default). With ``mode="next"``, it does this
for every item after the first, so a session of five items gives four
predictions. Sessions with a single item are skipped.

Items already in the session stay in the ranking, because people often come
back to an item; this is why SPop, which favours them, is a strong baseline.
With ``exclude_unknowns=True``, only items seen in training are ranked. Only
ranking metrics apply; rating metrics are ignored with a warning.

``fmt`` tells Cornac the tuple layout: ``"SIT"`` for sessions without users
(as in YooChoose), ``"USIT"`` with users, and ``"SITJson"`` or ``"USITJson"``
when each event carries extra data. Besides
:meth:`~cornac.eval_methods.next_item_evaluation.NextItemEvaluation.from_splits`
for data that is already split, there are two ways to split it for you:

- :meth:`~cornac.eval_methods.next_item_evaluation.NextItemEvaluation.from_timestamps`
  assigns whole sessions to training, validation or test by the time of their
  last event, using cut-off times that you choose. Training data then comes
  from before the test period, which is the protocol we recommend.
- :meth:`~cornac.eval_methods.next_item_evaluation.NextItemEvaluation.leave_last_out`
  takes ``(user, item, rating, timestamp)`` data, treats each user's history
  as one session, and holds out the last item for testing and the one before it
  for validation. Training then includes other users' later interactions, but
  many papers (SASRec, BERT4Rec, ...) use this protocol, so it is useful for
  comparing with their results.

The next-item models in Cornac are
:class:`~cornac.models.spop.recom_spop.SPop`,
:class:`~cornac.models.fpmc.recom_fpmc.FPMC`,
:class:`~cornac.models.gru4rec.recom_gru4rec.GRU4Rec`,
:class:`~cornac.models.sasrec.recom_sasrec.SASRec`,
:class:`~cornac.models.bert4rec.recom_bert4rec.BERT4Rec` and
:class:`~cornac.models.transformer_rec.recom_transformer_rec.TransformerRec`,
plus the Semantic ID models in
`Generative recommendation with Semantic IDs`_. Some need extra packages such
as PyTorch; install the packages in the model's ``requirements.txt`` before
you use it. The `examples
<https://github.com/PreferredAI/cornac/blob/master/examples/README.md>`_ list
scripts for them under "Next-Item Algorithms".

Next-basket recommendation
--------------------------

A next-basket model looks at a user's past baskets, such as shopping trips,
and predicts the items of the next one. The data is a list of
``(user, basket, item, timestamp)`` tuples. The Ta Feng grocery dataset adds a
fifth field with the price and quantity of each item:

.. code-block:: python

    import cornac
    from cornac.data import Reader
    from cornac.datasets import tafeng
    from cornac.eval_methods import NextBasketEvaluation
    from cornac.metrics import NDCG, HitRatio, Recall
    from cornac.models import GPTop

    data = tafeng.load_basket(
        reader=Reader(min_basket_size=3, max_basket_size=50, min_basket_sequence=2)
    )
    print(data[:2])

    next_basket_eval = NextBasketEvaluation(
        data=data, fmt="UBITJson", test_size=0.2, val_size=0.08, seed=123
    )

    models = [
        GPTop(name="GTop", use_personalized_popularity=False),
        GPTop(name="PTop", use_global_popularity=False),
        GPTop(),
    ]

    metrics = [Recall(k=10), NDCG(k=10), HitRatio(k=10)]

    cornac.Experiment(eval_method=next_basket_eval, models=models, metrics=metrics).run()

.. code-block:: bash
    :caption: output

    [('1104905', '0', '4710199010372', 973008000, {'quantity': 2, 'price': 30}), ('1104905', '0', '84501297329', 973008000, {'quantity': 2, 'price': 38})]

    VALIDATION:
    ...
          | HitRatio@10 | NDCG@10 | Recall@10 | Time (s)
    ----- + ----------- + ------- + --------- + --------
    GTop  |      0.2606 |  0.0562 |    0.0438 |   3.4698
    PTop  |      0.3686 |  0.0861 |    0.0778 |   3.6640
    GPTop |      0.4366 |  0.1052 |    0.0948 |   3.6799

    TEST:
    ...
          | HitRatio@10 | NDCG@10 | Recall@10 | Train (s) | Test (s)
    ----- + ----------- + ------- + --------- + --------- + --------
    GTop  |      0.2418 |  0.0540 |    0.0402 |    0.0409 |   9.0803
    PTop  |      0.3560 |  0.0807 |    0.0707 |    0.0002 |   9.1993
    GPTop |      0.4219 |  0.0983 |    0.0879 |    0.0400 |   9.1873

The ``Reader`` keeps baskets with 3 to 50 items and users with at least two
baskets. :class:`~cornac.models.gp_top.recom_gp_top.GPTop` recommends the
items that are most frequent overall (GTop), most frequent in the user's own
past baskets (PTop), or both (GPTop). Knowing what a user bought before helps a
lot.

How next-basket evaluation works
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~cornac.eval_methods.next_basket_evaluation.NextBasketEvaluation`
splits users, not baskets: ``test_size`` and ``val_size`` are the fractions of
users held out, and each split is stored as a
:class:`~cornac.data.dataset.BasketDataset`. For each test user, the last
basket is the target and the earlier baskets are the history given to the
model. ``fmt`` is ``"UBI"``, ``"UBIT"`` or ``"UBITJson"``, depending on
whether the tuples carry a timestamp and extra data.

Grocery shoppers often buy the same items again. To see how much of a score
comes from repeated items, set ``repetition_eval=True`` and
``exploration_eval=True``. The results then get extra ``-rep`` columns, which
count only target items the user bought before, and ``-expl`` columns, which
count only new ones:

.. code-block:: python

    rep_eval = NextBasketEvaluation(
        data=data,
        fmt="UBITJson",
        test_size=0.2,
        val_size=0.08,
        repetition_eval=True,
        exploration_eval=True,
        seed=123,
    )

    cornac.Experiment(
        eval_method=rep_eval, models=[GPTop()], metrics=[Recall(k=10)], show_validation=False
    ).run()

.. code-block:: bash
    :caption: output

    TEST:
    ...
          | Recall@10 | Recall@10-rep | Recall@10-expl | Train (s) | Test (s)
    ----- + --------- + ------------- + -------------- + --------- + --------
    GPTop |    0.0879 |        0.6439 |         0.0078 |    0.0410 |  10.1125

Almost all of GPTop's hits are repeat purchases: it finds 64% of the target
items that the user bought before, but under 1% of the new ones.

The next-basket models in Cornac are
:class:`~cornac.models.gp_top.recom_gp_top.GPTop`,
:class:`~cornac.models.tifuknn.recom_tifuknn.TIFUKNN`,
:class:`~cornac.models.upcf.recom_upcf.UPCF`,
:class:`~cornac.models.beacon.recom_beacon.Beacon` and
:class:`~cornac.models.dnntsp.recom_dnntsp.DNNTSP`. The `examples
<https://github.com/PreferredAI/cornac/blob/master/examples/README.md>`__ list
scripts for them under "Next-Basket Algorithms".

Generative recommendation with Semantic IDs
-------------------------------------------

Semantic ID models recommend the next item by generating a code that describes
it, instead of scoring every item one by one. A Semantic ID is a sequence of
integer codes given to each item. The codes are learned from the item's
content: the model takes an embedding of each item, for example of its title,
brand and categories, and quantizes it into a few codes. Items with similar
content share some of their codes, so the ID itself carries meaning, unlike an
arbitrary item number. Two items can end up with the same codes; some models
add an extra code as a suffix so that every item has its own ID.

Cornac includes five Semantic ID models. They are next-item models, so they
are evaluated with
:class:`~cornac.eval_methods.next_item_evaluation.NextItemEvaluation`, like
the models above:

- :class:`~cornac.models.tiger.recom_tiger.TIGER` builds hierarchical IDs with
  an RQ-VAE or residual k-means tokenizer; a T5-style model generates the next
  item's codes one after another
  (`README <https://github.com/PreferredAI/cornac/blob/master/cornac/models/tiger/README.md>`__,
  `example <https://github.com/PreferredAI/cornac/blob/master/examples/tiger_example.py>`__).
- :class:`~cornac.models.letter.recom_letter.LETTER` uses an RQ-VAE tokenizer
  that also aligns IDs with collaborative embeddings and spreads code usage; a
  T5 generator predicts the next item's four-code ID
  (`README <https://github.com/PreferredAI/cornac/blob/master/cornac/models/letter/README.md>`__,
  `example <https://github.com/PreferredAI/cornac/blob/master/examples/letter_example.py>`__).
  It also needs collaborative item embeddings, such as those of a trained
  SASRec, passed as ``cf_embeddings`` and ``cf_embedding_ids``.
- :class:`~cornac.models.rpg.recom_rpg.RPG` builds long, unordered IDs with
  product quantization; a GPT-2 decoder predicts all codes in parallel, and a
  similarity graph guides decoding
  (`README <https://github.com/PreferredAI/cornac/blob/master/cornac/models/rpg/README.md>`__,
  `example <https://github.com/PreferredAI/cornac/blob/master/examples/rpg_example.py>`__).
- :class:`~cornac.models.diffgrm.recom_diffgrm.DiffGRM` builds four
  independent codes with PCA and product quantization; a masked-diffusion
  model fills in all positions, most confident first
  (`README <https://github.com/PreferredAI/cornac/blob/master/cornac/models/diffgrm/README.md>`__,
  `example <https://github.com/PreferredAI/cornac/blob/master/examples/diffgrm_example.py>`__).
- :class:`~cornac.models.diger.recom_diger.DIGER` uses a residual quantizer
  whose codebooks are also updated by the recommendation loss; a T5 generator
  predicts the next item's ID
  (`README <https://github.com/PreferredAI/cornac/blob/master/cornac/models/diger/README.md>`__,
  `example <https://github.com/PreferredAI/cornac/blob/master/examples/diger_example.py>`__).

All five need PyTorch and a few other packages, listed in each model's
``requirements.txt``. Each README gives the recommended settings and the
results we measured.

A Semantic ID model needs two inputs: each user's interactions in time order,
and one content embedding per item (LETTER also takes the collaborative
embeddings above). The Amazon review datasets used in the
Semantic ID papers are built into Cornac, for the ``beauty``, ``sports`` and
``toys`` categories, together with the item text used by TIGER:

.. code-block:: python

    from cornac.datasets import amazon_review

    data = amazon_review.load_feedback(category="beauty")  # UIRT tuples, chronological per user
    texts, item_ids = amazon_review.load_text(category="beauty")  # item content text, aligned ids

    print(len(data), len(texts))
    print(data[0])
    print(texts[0])

.. code-block:: bash
    :caption: output

    198502 12101
    ('A00414041RD0BXM6WK0GX', 'B007IY97U0', 3.0, 1405296000)
    Title: 63cm Long Zipper Beige+pink Wavy Cosplay Hair Wig Rw157. Price: 11.83. Brand: Generic. Categories: Beauty, Hair Care, Styling Products, Hair Extensions & Wigs, Wigs

The first load downloads about 145 MB of reviews and product data. Next, turn
each text into an embedding. Cornac does not include a text encoder; the TIGER
paper and our examples use Sentence-T5 from the ``sentence-transformers``
package (``pip install sentence-transformers``). Pass the embeddings to the
evaluation method as a :class:`~cornac.data.modality.FeatureModality`, which
lines up its rows with the item indices.
:meth:`~cornac.eval_methods.next_item_evaluation.NextItemEvaluation.leave_last_out`
holds out each user's last item for testing and the one before it for
validation, the protocol behind the published results:

.. code-block:: python

    from sentence_transformers import SentenceTransformer

    from cornac.data import FeatureModality
    from cornac.eval_methods import NextItemEvaluation

    encoder = SentenceTransformer("sentence-t5-base")
    features = encoder.encode(texts, batch_size=64, show_progress_bar=True)

    next_item_eval = NextItemEvaluation.leave_last_out(
        data,
        exclude_unknowns=True,
        semantic_id_metrics=True,
        item_feature=FeatureModality(features=features, ids=item_ids),
    )

This example trains TIGER and SASRec, an ordinary next-item model, in the same
experiment. TIGER uses ``GRID_CONFIG``, the faster of the two training recipes
that ship with it. A full run takes an hour or more on a GPU, and much longer
on a CPU.

.. code-block:: python

    import torch

    import cornac
    from cornac.metrics import NDCG, Recall
    from cornac.models import SASRec, TIGER
    from cornac.models.tiger import GRID_CONFIG

    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = [
        TIGER(**{**GRID_CONFIG, "device": device, "seed": 123}),
        SASRec(device=device, seed=123),
    ]

    metrics = [Recall(k=10), NDCG(k=10), Recall(k=20), NDCG(k=20)]

    cornac.Experiment(eval_method=next_item_eval, models=models, metrics=metrics).run()

TIGER ranks items with a beam search, and ``n_beams`` (20 by default) should
be at least the largest cutoff of your metrics. Both models get the usual item
metrics, which measure the items they actually recommend. Because
``semantic_id_metrics=True``, for each ``Recall@K`` or ``NDCG@K`` TIGER also
reports:

- ``SID-`` metrics, which count a hit when the model generates the target
  item's ID, even if other items share that ID;
- ``CCE-`` (collision-corrected) metrics, which share that credit among all
  items with the same ID, as if one of them were picked at random.

SASRec shows ``N/A`` in these columns. When every item has its own complete
ID, as with TIGER, the SID and CCE numbers are the same. The exact rules, such
as how invalid or repeated IDs are handled, are under "Recommendation
Evaluation with Semantic IDs" in :doc:`/api_ref/eval_methods`.

Bundle recommendation, completion and construction
--------------------------------------------------

A bundle is a set of items that go together, such as a book list, a music
playlist or an outfit. Cornac covers three bundle tasks:

- **Bundle recommendation** ranks the bundles of an existing catalog for each
  user.
- **Bundle completion** gives some items of a bundle and ranks the items that
  complete it.
- **Bundle construction** builds a bundle for a user from scratch, with no
  items given.

Unlike a basket in next-basket recommendation, a catalog bundle has its own
ID, and recommending it means recommending the whole bundle.

The bundle datasets are in ``cornac.datasets.bundle``: Youshu, NetEase and
iFashion as released with CrossCBR, and Spotify as released with AttList. Each
comes with user-bundle feedback, split by its authors into training,
validation and test, the items of each bundle, and, except Spotify, user-item
feedback:

.. code-block:: python

    from cornac.datasets import bundle

    train = bundle.load_feedback("Youshu", split="train")
    validation = bundle.load_feedback("Youshu", split="validation")
    test = bundle.load_feedback("Youshu", split="test")
    memberships = bundle.load_bundle_item("Youshu")
    user_items = bundle.load_user_item("Youshu")
    ids = bundle.load_ids("Youshu")

    print(train[:2])
    print(memberships[:2])
    print([len(v) for v in ids.values()])

.. code-block:: bash
    :caption: output

    [('0', '2382', 1.0), ('0', '3026', 1.0)]
    [('0', '32604'), ('0', '30492')]
    [8039, 4771, 32770]

``train``, ``validation`` and ``test`` hold ``(user, bundle, 1.0)`` tuples,
``memberships`` holds ``(bundle, item)`` pairs, and ``user_items`` holds
``(user, item, 1.0)`` tuples. Bundle IDs and item IDs are separate: bundle
``'0'`` and item ``'0'`` are different things. ``load_ids()`` returns the
full lists of user, bundle and item IDs, including IDs without feedback.

Bundle recommendation
^^^^^^^^^^^^^^^^^^^^^

:class:`~cornac.eval_methods.bundle_evaluation.BundleEvaluation` ranks bundles
the way other evaluation methods rank items. Its training set is a
:class:`~cornac.data.dataset.BundleDataset`, with bundles in place of items,
and with the items of each bundle (``bundle_item_matrix``) and the user-item
feedback (``user_item_matrix``) alongside. Every bundle in the catalog is a
candidate, including bundles without training feedback.
``test_mask="train"`` leaves out only the training bundles of each user at
test time, as CrossCBR does; by default the validation bundles are left out
too.

.. code-block:: python

    import cornac
    from cornac.eval_methods import BundleEvaluation
    from cornac.metrics import NDCG, Recall
    from cornac.models import BundlePop

    metrics = [Recall(k=20), NDCG(k=20)]

    bundle_eval = BundleEvaluation.from_splits(
        train_data=train,
        val_data=validation,
        test_data=test,
        bundle_item_data=memberships,
        user_item_data=user_items,
        test_mask="train",
        **ids,
    )

    cornac.Experiment(
        eval_method=bundle_eval,
        models=[
            BundlePop(name="BundlePop-bundle", popularity="bundle"),
            BundlePop(name="BundlePop-item", popularity="item"),
            BundlePop(name="BundlePop-user", popularity="user"),
        ],
        metrics=metrics,
    ).run()

.. code-block:: bash
    :caption: output

    VALIDATION:
    ...
                     | NDCG@20 | Recall@20 | Time (s)
    ---------------- + ------- + --------- + --------
    BundlePop-bundle |  0.0950 |    0.2022 |   0.5415
    BundlePop-item   |  0.0006 |    0.0020 |   0.5165
    BundlePop-user   |  0.0040 |    0.0100 |   1.3501

    TEST:
    ...
                     | NDCG@20 | Recall@20 | Train (s) | Test (s)
    ---------------- + ------- + --------- + --------- + --------
    BundlePop-bundle |  0.1141 |    0.2033 |    0.0025 |   0.5554
    BundlePop-item   |  0.0008 |    0.0016 |    0.0016 |   0.5355
    BundlePop-user   |  0.0059 |    0.0104 |    0.0011 |   1.3637

:class:`~cornac.models.bundle_pop.recom_bundle_pop.BundlePop` is a baseline
that learns nothing. It scores a bundle by its number of training users
(``"bundle"``), by the mean popularity of its items (``"item"``), or by the
share of its items that the user has interacted with (``"user"``). In the run
above on Youshu, bundle popularity is by far the strongest of the three.
:class:`~cornac.models.crosscbr.recom_crosscbr.CrossCBR`,
:class:`~cornac.models.multicbr.recom_multicbr.MultiCBR` and
:class:`~cornac.models.bundlegt.recom_bundlegt.BundleGT` learn from both the
user-bundle and the user-item feedback, so they cannot run on Spotify. They
need PyTorch, and their `examples
<https://github.com/PreferredAI/cornac/blob/master/examples/README.md>`__,
listed under "Bundle Algorithms", train them on Youshu. Ordinary item models
such as BPR also work, treating bundles as items.

Bundle completion and construction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~cornac.eval_methods.bundle_evaluation.BundleConstructionEvaluation`
splits the bundles, not their rows, into training, validation and test. Each
held-out bundle becomes a query. With ``input_ratio=0.5``, half of its items
are given, and the model ranks all items to find the rest:

.. code-block:: python

    from cornac.eval_methods import BundleConstructionEvaluation

    completion_eval = BundleConstructionEvaluation(
        data=memberships,
        input_ratio=0.5,
        test_size=0.2,
        val_size=0.1,
        item_ids=ids["constituent_item_ids"],
        seed=123,
    )

    cornac.Experiment(
        eval_method=completion_eval,
        models=[
            BundlePop(name="BundlePop-item", popularity="item"),
            BundlePop(name="BundlePop-cooccurrence", popularity="cooccurrence"),
        ],
        metrics=metrics,
    ).run()

.. code-block:: bash
    :caption: output

    VALIDATION:
    ...
                           | NDCG@20 | Recall@20 | Set-F1 | Set-Jaccard | Time (s)
    ---------------------- + ------- + --------- + ------ + ----------- + --------
    BundlePop-item         |  0.0459 |    0.0435 | 0.0341 |      0.0185 |   0.5998
    BundlePop-cooccurrence |  0.1307 |    0.1210 | 0.0875 |      0.0489 |   0.8272

    TEST:
    ...
                           | NDCG@20 | Recall@20 | Set-F1 | Set-Jaccard | Train (s) | Test (s)
    ---------------------- + ------- + --------- + ------ + ----------- + --------- + --------
    BundlePop-item         |  0.0393 |    0.0382 | 0.0284 |      0.0152 |    0.0034 |   1.2091
    BundlePop-cooccurrence |  0.1224 |    0.1141 | 0.0820 |      0.0458 |    0.0018 |   1.6959

The queries take the place of users: the training set is a
:class:`~cornac.data.dataset.BundleConstructionDataset` with training bundles
and queries as rows and items as columns. The given items of each query are
its training rows, so ranking metrics leave them out, and item models such as
``MostPop`` and ``ItemKNN`` also work for completion. ``"cooccurrence"``
scores an item by how often it appears in training bundles together with the
given items, and it finds about three times as many missing items as item
popularity. The bundle rows can be ``(bundle, item)`` pairs (``fmt="BI"``,
the default), with a timestamp (``"BIT"``), or with the bundle's owner first
(``"UBI"``, ``"UBIT"``).

Next to the ranking metrics, ``Set-F1`` and ``Set-Jaccard`` compare the
missing items with as many items as the model constructs. Models that subclass
:class:`~cornac.models.recommender.BundleConstructionRecommender` construct
them with ``construct()``; for other models, Cornac takes the top-ranked
items. Pass ``set_metrics=False`` to skip them.

Construction from scratch uses ``input_ratio=0``: no items are given, so each
held-out bundle is paired with each of its users, and the model builds the
bundle from what it knows about the user. ``user_bundle_data`` says who uses
each bundle, and ``user_item_data`` adds the users' item feedback. Held-out
bundles without users are dropped.

.. code-block:: python

    construction_eval = BundleConstructionEvaluation(
        data=memberships,
        input_ratio=0,
        test_size=0.2,
        val_size=0.1,
        item_ids=ids["constituent_item_ids"],
        seed=123,
        user_bundle_data=train + validation + test,
        user_item_data=user_items,
    )

    cornac.Experiment(
        eval_method=construction_eval,
        models=[
            BundlePop(name="BundlePop-item", popularity="item"),
            BundlePop(name="BundlePop-bundle", popularity="bundle"),
            BundlePop(name="BundlePop-user", popularity="user"),
        ],
        metrics=metrics,
    ).run()

.. code-block:: bash
    :caption: output

    VALIDATION:
    ...
                     | NDCG@20 | Recall@20 | Set-F1 | Set-Jaccard | Time (s)
    ---------------- + ------- + --------- + ------ + ----------- + --------
    BundlePop-item   |  0.1846 |    0.0637 | 0.1368 |      0.0800 |   9.1786
    BundlePop-bundle |  0.1691 |    0.0575 | 0.1238 |      0.0707 |   7.5300
    BundlePop-user   |  0.1539 |    0.0503 | 0.1031 |      0.0569 |  11.8434

    TEST:
    ...
                     | NDCG@20 | Recall@20 | Set-F1 | Set-Jaccard | Train (s) | Test (s)
    ---------------- + ------- + --------- + ------ + ----------- + --------- + --------
    BundlePop-item   |  0.0881 |    0.0286 | 0.0599 |      0.0323 |    0.0038 |  14.1484
    BundlePop-bundle |  0.0848 |    0.0274 | 0.0603 |      0.0323 |    0.0031 |  11.5588
    BundlePop-user   |  0.1101 |    0.0382 | 0.0728 |      0.0392 |    0.0026 |  18.1682

The whole example, with the three experiments, is `bundle_pop_youshu.py
<https://github.com/PreferredAI/cornac/blob/master/examples/bundle_pop_youshu.py>`_.
The `dataset catalogue
<https://github.com/PreferredAI/cornac/blob/master/cornac/datasets/README.md>`__
compares the four bundle datasets.

Adding a model for a task
-------------------------

Each task has a base class for new models:
:class:`~cornac.models.recommender.Recommender` for rating prediction and
top-N ranking,
:class:`~cornac.models.recommender.NextItemRecommender`,
:class:`~cornac.models.recommender.NextBasketRecommender`,
:class:`~cornac.models.recommender.BundleRecommender` and
:class:`~cornac.models.recommender.BundleConstructionRecommender`. A Semantic
ID model extends ``NextItemRecommender`` and also implements
:class:`~cornac.models.recommender.SemanticIDMixin`, which gives the
evaluation method the generated IDs. :doc:`/developer/add_model` walks you
through adding a model to Cornac.
