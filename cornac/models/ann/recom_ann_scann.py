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


SUPPORTED_MEASURES = {
    MEASURE_L2: "squared_l2",
    MEASURE_DOT: "dot_product",
    MEASURE_COSINE: "dot_product",
}


class ScaNNANN(BaseANN):
    """Approximate Nearest Neighbor Search with ScaNN
    (https://github.com/google-research/google-research/tree/master/scann).
    ScaNN performs vector search in three phases: paritioning, scoring, and rescoring.
    More on the algorithms and parameter description: https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md

    Parameters
    ----------------
    model: object: :obj:`cornac.models.Recommender`, required
        Trained recommender model which to get user/item vectors from.

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
        Default number of threads used for training. If num_threads = -1, all cores will be used.

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

        if partition_params is None:
            partition_params = {"num_leaves": 100, "num_leaves_to_search": 50}

        if score_params is None:
            score_params = {
                "dimensions_per_block": 2,
                "anisotropic_quantization_threshold": 0.2,
            }

        if rescore_params is None:
            rescore_params = {"reordering_num_neighbors": 100}

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
        super().build_index()

        import scann

        assert self.measure in SUPPORTED_MEASURES

        if self.measure == MEASURE_COSINE:
            self.partition_params["spherical"] = True
            self.item_vectors /= np.linalg.norm(self.item_vectors, axis=1)[
                :, np.newaxis
            ]
        else:
            self.partition_params["spherical"] = False

        index_builder = scann.scann_ops_pybind.builder(
            self.item_vectors, 10, SUPPORTED_MEASURES[self.measure]
        )
        index_builder.set_n_training_threads(self.num_threads)

        # partitioning
        if self.partition_params:
            self.partition_params.setdefault(
                "training_sample_size", self.item_vectors.shape[0]
            )
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

        # make sure distances respect the notion of nearest neighbors (smaller is better)
        if self.higher_is_better:
            distances = 1.0 - distances

        return neighbors, distances

    def save(self, save_dir=None):
        saved_path = super().save(save_dir)
        idx_path = saved_path + ".index"
        os.makedirs(idx_path, exist_ok=True)
        self.index.searcher.serialize(idx_path)
        return saved_path

    @staticmethod
    def load(model_path, trainable=False):
        from scann.scann_ops.py import scann_ops_pybind

        ann = BaseANN.load(model_path, trainable)
        idx_path = ann.load_from + ".index"
        ann.index = scann_ops_pybind.load_searcher(idx_path)
        return ann
