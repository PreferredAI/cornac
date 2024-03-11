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

from ..recommender import MEASURE_L2, MEASURE_DOT, MEASURE_COSINE
from .recom_ann_base import BaseANN


class FaissANN(BaseANN):
    """Approximate Nearest Neighbor Search with Faiss (https://github.com/facebookresearch/faiss).
    Faiss provides both CPU and GPU implementation. More on the algorithms:
    https://github.com/facebookresearch/faiss/wiki

    Parameters
    ----------------
    model: object: :obj:`cornac.models.Recommender`, required
        Trained recommender model which to get user/item vectors from.

    nlist: int, default: 100
        The number of cells used for building the index.

    nprobe: int, default: 50
        The number of cells (out of nlist) that are visited to perform a search.

    use_gpu : bool, optional
        Whether or not to run Faiss on GPU. Requires faiss-gpu to be installed
        instead of faiss-cpu.

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
        nlist=100,
        nprobe=50,
        use_gpu=False,
        num_threads=-1,
        seed=None,
        name="FaissANN",
        verbose=False,
    ):
        super().__init__(model=model, name=name, verbose=verbose)

        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu = use_gpu
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

        import faiss

        faiss.omp_set_num_threads(self.num_threads)

        SUPPORTED_MEASURES = {
            MEASURE_L2: faiss.METRIC_L2,
            MEASURE_DOT: faiss.METRIC_INNER_PRODUCT,
            MEASURE_COSINE: faiss.METRIC_INNER_PRODUCT,
        }

        assert self.measure in SUPPORTED_MEASURES

        if self.measure == MEASURE_COSINE:
            self.item_vectors /= np.linalg.norm(self.item_vectors, axis=1)[
                :, np.newaxis
            ]

        self.item_vectors = self.item_vectors.astype("float32")

        self.index = faiss.IndexIVFFlat(
            faiss.IndexFlat(self.item_vectors.shape[1]),
            self.item_vectors.shape[1],
            self.nlist,
            SUPPORTED_MEASURES[self.measure],
        )

        if self.use_gpu:
            self.index = faiss.index_cpu_to_all_gpus(self.index)

        self.index.train(self.item_vectors)
        self.index.add(self.item_vectors)
        self.index.nprobe = self.nprobe

    def knn_query(self, query, k):
        """Implementing ANN search for a given query.

        Returns
        -------
        neighbors, distances: numpy.array and numpy.array
            Array of k-nearest neighbors and corresponding distances for the given query.
        """
        distances, neighbors = self.index.search(query, k)

        # make sure distances respect the notion of nearest neighbors (smaller is better)
        if self.higher_is_better:
            distances = 1.0 - distances

        return neighbors, distances

    def save(self, save_dir=None):
        import faiss

        saved_path = super().save(save_dir)
        idx_path = saved_path + ".index"
        if self.use_gpu:
            self.index = faiss.index_gpu_to_cpu(self.index)
        faiss.write_index(self.index, idx_path)
        return saved_path

    @staticmethod
    def load(model_path, trainable=False):
        import faiss

        ann = BaseANN.load(model_path, trainable)
        idx_path = ann.load_from + ".index"
        ann.index = faiss.read_index(idx_path)
        if ann.use_gpu:
            ann.index = faiss.index_cpu_to_all_gpus(ann.index)
        return ann
