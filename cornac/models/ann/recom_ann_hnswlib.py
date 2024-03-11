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


import sys
import random
import multiprocessing
import numpy as np

from ..recommender import MEASURE_L2, MEASURE_DOT, MEASURE_COSINE
from .recom_ann_base import BaseANN


SUPPORTED_MEASURES = {
    MEASURE_L2: "l2",
    MEASURE_DOT: "ip",
    MEASURE_COSINE: "cosine",
}


class HNSWLibANN(BaseANN):
    """Approximate Nearest Neighbor Search with HNSWLib (https://github.com/nmslib/hnswlib/).

    Parameters
    ----------------
    model: object: :obj:`cornac.models.Recommender`, required
        Trained recommender model which to get user/item vectors from.

    M: int, optional, default: 16
        Parameter that defines the maximum number of outgoing connections in the HNSW graph.
        Higher M leads to higher accuracy/run_time at fixed ef/ef_construction. Reasonable range
        for M is 2-100. Higher M work better on model with high dimensional factors, while low M
        work better for low dimensional factors. More details: https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md.

    ef_construction: int, optional, default: 100
        Parameter that controls speed/accuracy trade-off during the index construction. Bigger ef_construction leads to longer construction, but better index quality. At some point,
        increasing ef_construction does not improve the quality of the index.

    ef: int, optional, default: 50
        Parameter controlling query time/accuracy trade-off. Higher `ef` leads to more accurate but
        slower search. `ef` cannot be set lower than the number of queried nearest neighbors k. The
        value of `ef` can be anything between `k` and the total number of items.

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
        M=16,
        ef_construction=100,
        ef=50,
        num_threads=-1,
        seed=None,
        name="HNSWLibANN",
        verbose=False,
    ):
        super().__init__(model=model, name=name, verbose=verbose)

        self.M = M
        self.ef_construction = ef_construction
        self.ef = ef
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

        import hnswlib

        assert self.measure in SUPPORTED_MEASURES

        self.index = hnswlib.Index(
            space=SUPPORTED_MEASURES[self.measure], dim=self.item_vectors.shape[1]
        )

        self.index.init_index(
            max_elements=self.item_vectors.shape[0],
            ef_construction=self.ef_construction,
            M=self.M,
            random_seed=(
                np.random.randint(sys.maxsize) if self.seed is None else self.seed
            ),
        )
        self.index.add_items(
            data=self.item_vectors,
            ids=np.arange(self.item_vectors.shape[0]),
            num_threads=(-1 if self.seed is None else 1),
        )
        self.index.set_ef(self.ef)
        self.index.set_num_threads(self.num_threads)

    def knn_query(self, query, k):
        """Implementing ANN search for a given query.

        Returns
        -------
        neighbors, distances: numpy.array and numpy.array
            Array of k-nearest neighbors and corresponding distances for the given query.
        """
        neighbors, distances = self.index.knn_query(query, k=k)
        return neighbors, distances

    def save(self, save_dir=None):
        saved_path = super().save(save_dir)
        self.index.save_index(saved_path + ".index")
        return saved_path

    @staticmethod
    def load(model_path, trainable=False):
        import hnswlib

        ann = BaseANN.load(model_path, trainable)
        ann.index = hnswlib.Index(
            space=SUPPORTED_MEASURES[ann.measure], dim=ann.user_vectors.shape[1]
        )
        ann.index.load_index(ann.load_from + ".index")
        ann.index.set_ef(ann.ef)
        ann.index.set_num_threads(ann.num_threads)
        return ann
