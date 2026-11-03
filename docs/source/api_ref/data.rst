Data
====

	
Dataset
------------------------------------
.. automodule:: cornac.data.dataset
   :members:

Catalog Bundle Recommendation
---------------------------------
``BundleDataset`` uses inherited item fields for bundle targets. Its
``bundle_item_matrix`` and optional ``user_item_matrix`` use a distinct
constituent item ID space. Required catalog membership defines all candidates,
including bundles without training interactions. Independent user-item signals
must contain training observations only; held-out bundle interactions are never
converted into item feedback. See :class:`cornac.eval_methods.BundleEvaluation`
and ``examples/bundle_recommendation_example.py`` for the split-based workflow.

Modality
---------------------------------
.. automodule:: cornac.data.modality
   :members:

Graph Modality
---------------------------------
.. automodule:: cornac.data.graph
   :members:

Text Modality
---------------------------------
.. automodule:: cornac.data.text
   :members:

Image Modality
---------------------------------
.. automodule:: cornac.data.image
   :members:

Sentiment Modality
---------------------------------
.. automodule:: cornac.data.sentiment
   :members:

Reader
----------------------------------
.. automodule:: cornac.data.reader
   :members:
