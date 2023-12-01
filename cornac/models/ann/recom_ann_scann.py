# Copyright 2023 The Cornac Authors. All Rights Reserved.
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


import os
import multiprocessing
import numpy as np

from ..recommender import MEASURE_L2, MEASURE_DOT, MEASURE_COSINE
from .recom_ann_base import BaseANN


SUPPORTED_MEASURES = {MEASURE_L2: "squared_l2", MEASURE_DOT: "dot_product"}


class ScaNNANN(BaseANN):
    """Approximate Nearest Neighbor Search with ScaNN
    (https://github.com/google-research/google-research/tree/master/scann).
    ScaNN performs vector search in three phases: paritioning, scoring, and rescoring.
    More on the algorithms and parameter description: https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md

    Parameters
    ----------------
    model: object: :obj:`cornac.models.Recommender`, required
        Trained recommender model which to get user/item vectors from.

    num_neighbors: int, optional, default: 100
        The default number of neighbors/items to be returned.

    partition_params: dict, optional
        Parameters for the partitioning phase, to send to the tree() call in ScaNN.

    score_params: dict, optional
        Parameters for the scoring phase, to send to the score_ah() call in ScaNN.
        score_brute_force() will be called if score_brute_force is True.

    score_brute_force: bool, optional, default: False
        Whether to call score_brute_force() for the scoring phase.

    rescore_params: dict, optional
        Parameters for the rescoring phase, to send to the reorder() call in ScaNN.

    num_threads: int, optional, default: -1
        Default number of threads to use when querying. If num_threads = -1, all cores will be used.

    seed: int, optional, default: None
        Random seed for reproducibility.

    name: str, required
        Name of the recommender model.

    verbose: boolean, optional, default: False
        When True, running logs are displayed.
    """

    def __init__(
        self,
        model,
        num_neighbors=100,
        partition_params=None,
        score_params=None,
        score_brute_force=False,
        rescore_params=None,
        num_threads=-1,
        seed=None,
        name="ScaNNANN",
        verbose=False,
    ):
        super().__init__(model=model, name=name, verbose=verbose)

        if score_params is None:
            score_params = {}

        self.model = model
        self.num_neighbors = num_neighbors
        self.partition_params = partition_params
        self.score_params = score_params
        self.score_brute_force = score_brute_force
        self.rescore_params = rescore_params
        self.num_threads = (
            num_threads if num_threads != -1 else multiprocessing.cpu_count()
        )
        self.seed = seed

        self.index = None
        self.ignored_attrs.extend(
            [
                "index",  # will be saved separately
                "item_vectors",  # redundant after index is built
            ]
        )

    def build_index(self):
        """Building index from the base recommender model."""
        import scann

        assert self.measure in SUPPORTED_MEASURES

        if self.measure == MEASURE_COSINE:
            self.item_vectors = (
                self.item_vectors
                / np.linalg.norm(self.item_vectors, axis=1)[:, np.newaxis]
            )
            self.measure = MEASURE_DOT

        index_builder = scann.scann_ops_pybind.builder(
            self.item_vectors, self.num_neighbors, SUPPORTED_MEASURES[self.measure]
        )

        # partitioning
        if self.partition_params:
            index_builder = index_builder.tree(**self.partition_params)

        # scoring
        if self.score_brute_force:
            index_builder = index_builder.score_brute_force(**self.score_params)
        else:
            index_builder = index_builder.score_ah(**self.score_params)

        # rescoring
        if self.rescore_params:
            index_builder = index_builder.reorder(**self.rescore_params)

        self.index = index_builder.build()

    def knn_query(self, query, k):
        """Implementing ANN search for a given query.

        Returns
        -------
        neighbors, distances: numpy.array and numpy.array
            Array of k-nearest neighbors and corresponding distances for the given query.
        """
        neighbors, distances = self.index.search_batched(query, final_num_neighbors=k)
        return neighbors, distances

    def save(self, save_dir=None):
        saved_path = super().save(save_dir)
        self.index.searcher.serialize(os.path.dirname(saved_path))
        return saved_path

    @staticmethod
    def load(model_path, trainable=False):
        from scann.scann_ops.py import scann_ops_pybind

        ann = BaseANN.load(model_path, trainable)
        ann.index = scann_ops_pybind.load_searcher(os.path.dirname(model_path))
        return ann
