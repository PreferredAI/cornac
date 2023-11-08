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


import multiprocessing
import numpy as np

from .recom_ann_base import BaseANN


class HNSWLibANN(BaseANN):
    """Approximate Nearest Neighbor Search with HNSWLib (https://github.com/nmslib/hnswlib/).

    Parameters
    ----------------
    recom: object: :obj:`cornac.models.Recommender`, required
        Trained recommender model which to get user/item vectors from.

    M: int, optional, default: 16
        Parameter that defines the maximum number of outgoing connections in the HNSW graph.
        Higher M leads to higher accuracy/run_time at fixed ef/efConstruction

    ef_construction: int, optional, default: 100
        Parameter that controls speed/accuracy trade-off during the index construction.

    ef: int, optional, default: 50
        Parameter controlling query time/accuracy trade-off.

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
        recom,
        M=16,
        ef_construction=100,
        ef=50,
        num_threads=-1,
        name="HNSWLibANN",
        verbose=False,
    ):
        super().__init__(recom=recom, name=name, verbose=verbose)
        self.M = M
        self.ef_construction = ef_construction
        self.ef = ef
        self.num_threads = (
            num_threads if num_threads != -1 else multiprocessing.cpu_count()
        )
        self.index = None

    def build_index(self):
        """Building index from the base recommender model."""
        import hnswlib

        item_vectors = self.recom.get_item_vectors()
        measure = self.recom.get_vector_measure()

        self.index = hnswlib.Index(space=measure, dim=item_vectors.shape[1])
        self.index.init_index(
            max_elements=item_vectors.shape[0],
            ef_construction=self.ef_construction,
            M=self.M,
        )
        self.index.add_items(item_vectors, np.arange(item_vectors.shape[0]))

        self.index.set_ef(self.ef)
        self.index.set_num_threads(self.num_threads)

    def knn_query(self, query, k):
        """Implementing ANN search for a given query.

        Returns
        -------
        neighbors: numpy.array
            Array of k-nearest neighbors for the given query.
        """
        neighbors, distances = self.index.knn_query(query, k=k)
        return neighbors
