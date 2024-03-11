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


import numpy as np

from ..recommender import MEASURE_L2, MEASURE_DOT, MEASURE_COSINE
from .recom_ann_base import BaseANN


SUPPORTED_MEASURES = {
    MEASURE_L2: "euclidean",
    MEASURE_DOT: "dot",
    MEASURE_COSINE: "angular",
}


class AnnoyANN(BaseANN):
    """Approximate Nearest Neighbor Search with Annoy (https://github.com/spotify/annoy).

    Parameters
    ----------------
    model: object: :obj:`cornac.models.Recommender`, required
        Trained recommender model which to get user/item vectors from.

    n_trees: int, default: 100
        The number of trees used to build index. It affects the build time and the
        index size. A larger value will give more accurate results, but larger indexes.

    search_k: int, default: 50
        Parameter controls the search performance and runtime. A larger value will
        give more accurate results, but will take longer time to return.

    num_threads: int, optional, default: -1
        Default number of threads used for building index. If num_threads = -1,
        all cores will be used.

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
        n_trees=100,
        search_k=50,
        num_threads=-1,
        seed=None,
        name="AnnoyANN",
        verbose=False,
    ):
        super().__init__(model=model, name=name, verbose=verbose)

        self.n_trees = n_trees
        self.search_k = search_k
        self.num_threads = num_threads
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

        from annoy import AnnoyIndex

        assert self.measure in SUPPORTED_MEASURES

        self.index = AnnoyIndex(
            self.item_vectors.shape[1], SUPPORTED_MEASURES[self.measure]
        )

        if self.seed is not None:
            self.index.set_seed(self.seed)

        for i, v in enumerate(self.item_vectors):
            self.index.add_item(i, v)

        self.index.build(self.n_trees, n_jobs=self.num_threads)

    def knn_query(self, query, k):
        """Implementing ANN search for a given query.

        Returns
        -------
        neighbors, distances: numpy.array and numpy.array
            Array of k-nearest neighbors and corresponding distances for the given query.
        """
        result = [
            self.index.get_nns_by_vector(
                q, k, search_k=self.search_k, include_distances=True
            )
            for q in query
        ]
        neighbors = np.array([r[0] for r in result], dtype="int")
        distances = np.array([r[1] for r in result], dtype="float32")

        # make sure distances respect the notion of nearest neighbors (smaller is better)
        if self.higher_is_better:
            distances = 1.0 - distances

        return neighbors, distances

    def save(self, save_dir=None):
        saved_path = super().save(save_dir)
        self.index.save(saved_path + ".index")
        return saved_path

    @staticmethod
    def load(model_path, trainable=False):
        from annoy import AnnoyIndex

        ann = BaseANN.load(model_path, trainable)
        ann.index = AnnoyIndex(
            ann.user_vectors.shape[1], SUPPORTED_MEASURES[ann.measure]
        )
        ann.index.load(ann.load_from + ".index")
        return ann
