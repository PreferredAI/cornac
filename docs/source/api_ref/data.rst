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
converted into item feedback. See :class:`cornac.eval_methods.BundleEvaluation`.

Bundle Completion and Construction
----------------------------------
``BundleConstructionDataset`` places bundle instances on the user axis and
constituent items on the item axis. An instance is a training bundle or an
evaluation query: a held-out bundle, optionally paired with one of its users.
Rows use the ``BI``, ``BIT``, ``UBI`` and ``UBIT`` formats, where ``U`` is the
bundle's single owner. ``bundle_items`` returns an instance's items in stable
timestamp order, or in row order without timestamps. Users have a separate
``owner_map``; ``owner_indices``, ``user_bundle_matrix`` and
``user_item_matrix`` relate them to instances and items using training-time data
only. See :class:`cornac.eval_methods.BundleConstructionEvaluation`.

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
