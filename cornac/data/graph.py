# Copyright 2018 The Cornac Authors. All Rights Reserved.
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

import scipy.sparse as sp
import numpy as np
from tqdm import trange

from . import FeatureModule


class GraphModule(FeatureModule):
    """Graph module
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.raw_data = kwargs.get('data', None)
        self.__matrix = None
        self.__matrix_size = None

    @property
    def matrix(self):
        if self.__matrix is None:
            assert self.__matrix_size is not None
            self.__matrix = sp.csr_matrix((self.val, (self.map_rid, self.map_cid)),
                                          shape=(self.__matrix_size, self.__matrix_size))
        return self.__matrix

    def _build_triplet(self, id_map):
        """Build adjacency matrix in sparse triplet format using mapped ids
        """
        self.map_rid = []
        self.map_cid = []
        self.val = []
        for i, j, v in self.raw_data:
            if (i not in id_map) or (j not in id_map):
                continue
            self.map_rid.append(id_map[i])
            self.map_cid.append(id_map[j])
            self.val.append(v)

        self.map_rid = np.asarray(self.map_rid, dtype=np.int)
        self.map_cid = np.asarray(self.map_cid, dtype=np.int)
        self.val = np.asarray(self.val, dtype=np.float)

    def build(self, id_map=None):
        super().build(id_map)

        if id_map is not None:
            self.__matrix_size = int(max(id_map.values()) + 1)
            self._build_triplet(id_map)
        return self

    def get_train_triplet(self, train_row_ids, train_col_ids):
        """Get the training tuples
        """
        picked_idx = []
        train_row_ids = set(train_row_ids) if not isinstance(train_row_ids, set) else train_row_ids
        train_col_ids = set(train_col_ids) if not isinstance(train_col_ids, set) else train_col_ids
        for idx, (i, j) in enumerate(zip(self.map_rid, self.map_cid)):
            if (i not in train_row_ids) or (j not in train_col_ids):
                continue
            picked_idx.append(idx)

        return self.map_cid[picked_idx], \
               self.map_rid[picked_idx], \
               self.val[picked_idx]

    # TODO: add feature_fallback decorator and rename the API more meaningful
    def batch(self, batch_ids):
        """Return batch of vectors from the sparse adjacency matrix corresponding to provided batch_ids.

        Parameters
        ----------
        batch_ids: array, required
            An array contains the ids of rows to be returned from the sparse adjacency matrix.
        """

        return self.matrix[batch_ids]

    @staticmethod
    def _find_min(vect):
        """ Return the lowest number and its position (index) is a given vector (array).

        Parameters
        ----------
        vect: array, required
            A Numpy 1d array of real values.
        """

        min_index = 0
        min_ = np.inf
        for i in range(len(vect)):
            if vect[i] < min_:
                min_index = i
                min_ = vect[i]
        return min_index, min_

    @staticmethod
    def _to_triplet(mat, ids=None):
        """Return the lowest number and its position (index) is a given vector (array).

        Parameters
        ----------
        mat: 2d array, required
            A Numpy 2d array of integers.
        ids: list, optional, default: None
            A list of ids (or labels) of the objects to be used in the output triplet matrix.
        """
        tuples = set()
        n = mat.shape[0]
        k = mat.shape[1]

        if ids is None:
            ids = range(n)
        for n_ in range(n):
            for k_ in range(k):
                j = int(mat[n_, k_])
                tuples.add((ids[n_], ids[j], 1.))

        return tuples

    @staticmethod
    def _to_symmetric(triplets):
        """ Transform an asymmetric adjacency matrix to a symmetric one.

        Parameters
        ----------
        triplets: array, required
            A Numpy 1d array of real values.
        """
        triplets.update([(j, i, v) for (i, j, v) in triplets])
        return triplets

    @staticmethod
    def _build_knn(features, k=5, similarity="cosine", verbose=True):
        """Build a KNN graph of a set of objects using similarities among there features.

        Parameters
        ----------
        features: 2d array, required
            A 2d Numpy array of features (object-by-features).
        k: int, optional, default: 5
            The number of nearest neighbors
        similarity: string, optional, default: "cosine"
            The similarity measure. At this time only the cosine is supported
        """

        # Some util variables
        n = len(features)
        N = np.zeros((n, k))
        S = np.zeros((n, k))

        if similarity == "cosine":
            # Normalize features to lie on a unit hypersphere
            l2_norm = np.sqrt((features * features).sum(1))
            l2_norm = l2_norm.reshape(n, 1)
            features = features / (l2_norm + 1e-20)

        for i in trange(n, desc='Building KNN Graph', disable=not verbose):
            c_id = 0
            for j in range(n):
                if i != j:
                    sim = np.dot(features[i], features[j])
                    if c_id <= k - 1:
                        N[i, c_id] = j
                        S[i, c_id] = sim
                        c_id += 1
                    else:
                        m_id, m = GraphModule._find_min(S[i])
                        if sim > m:
                            N[i, m_id] = j
                            S[i, m_id] = sim
        return N

    @classmethod
    def from_feature(cls, features, k=5, ids=None, similarity="cosine", symmetric=False, verbose=True):
        """Instantiate a GraphModule with a KNN graph build using input features.

        Parameters
        ----------
        features: 2d Numpy array, shape: [n_objects, n_features], required
            A 2d Numpy array of features, e.g., visual, textual, etc.

        k: int, optional, default: 5
            The number of nearest neighbors

        ids: array, optional, default: None
            The list of object ids or labels. For instance if you use textual (bag-of-word) features,
            then "ids" should be the same as the input to cornac.data.TextModule.

        similarity: string, optional, default: "cosine"
            The similarity measure. At this time only the cosine is supported

        symmetric: bool, optional, default: False
            When True the resulting KNN-Graph is made symmetric

        verbose: bool, default: False
            The verbosity flag.

        Returns
        -------
        graph_module: :obj:`<cornac.data.GraphModule>`
            GraphModule object.

        """

        # build knn graph
        knn_graph_array = GraphModule._build_knn(features, k, similarity, verbose=verbose)
        knn_graph_triplet = GraphModule._to_triplet(mat=knn_graph_array, ids=ids)
        if symmetric:
            knn_graph_triplet = GraphModule._to_symmetric(knn_graph_triplet)
            if verbose:
                print("Graph is symmetrized")

        return cls(data=knn_graph_triplet)
