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
from tqdm.auto import trange

from . import FeatureModality


class GraphModality(FeatureModality):
    """Graph modality

    Parameters
    ----------
    data: List[str], required
        A list encoding an adjacency matrix, of a user or an item graph, in the sparse triplet format, \
        e.g., data=[('user1', 'user4', 1.0)].
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.raw_data = kwargs.get('data', None)
        self.__matrix = None
        self.__matrix_size = None

    @property
    def matrix(self):
        """Return the adjacency matrix in scipy csr sparse format
        """
        if self.__matrix is None:
            assert self.__matrix_size is not None
            self.__matrix = sp.csr_matrix((self.val, (self.map_rid, self.map_cid)),
                                          shape=(self.__matrix_size, self.__matrix_size))
        return self.__matrix

    def _build_triplet(self, id_map):
        """Build adjacency matrix in sparse triplet format using cornac's mapped ids
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

    def build(self, id_map=None, **kwargs):
        super().build(id_map=id_map)

        self.__matrix = None
        if id_map is not None:
            self.__matrix_size = int(max(id_map.values()) + 1)
            self._build_triplet(id_map)
        return self

    def get_train_triplet(self, train_row_ids, train_col_ids):
        """Get the subset of relations which align with the training data

        Parameters
        ----------
        train_row_ids: array, required
            An array containing the ids of training objects (users or items) for which to get the "out" relations. \

        train_col_ids: array, required
            An array containing the ids of training objects (users or items) for whom to get the "in" relations.
            Please refer to cornac/models/c2pf/recom_c2pf.py for a concrete usage example of this function.

        Returns
        -------
        A subset of the adjacency matrix, in the sparse triplet format, whose elements align with the training \
        set as specified by "train_row_ids" and "train_col_ids".
        """
        picked_idx = []
        train_row_ids = set(train_row_ids) if not isinstance(train_row_ids, set) else train_row_ids
        train_col_ids = set(train_col_ids) if not isinstance(train_col_ids, set) else train_col_ids
        for idx, (i, j) in enumerate(zip(self.map_rid, self.map_cid)):
            if (i not in train_row_ids) or (j not in train_col_ids):
                continue
            picked_idx.append(idx)

        return self.map_rid[picked_idx], \
               self.map_cid[picked_idx], \
               self.val[picked_idx]


    def get_node_degree(self, in_ids=None, out_ids=None):
        """Get the "in" and "out" degree for the desired set of nodes

        Parameters
        ----------
        in_ids: array, required
            An array containing the ids for which to get the "in" degree. \

        out_ids: array, required
            An array containing the ids for which to get the "out" degree. \

        Returns
        -------
        Dictionary of the from {node_id: [in_degree,out_degree]}
        """

        degree = {}

        if in_ids is None:
            in_ids = self.map_cid
        if out_ids is None:
            out_ids = self.map_rid

        in_ids = set(in_ids) if not isinstance(in_ids, set) else in_ids
        out_ids = set(out_ids) if not isinstance(out_ids, set) else out_ids
        for (i, j) in zip(self.map_rid, self.map_cid):
            if (i not in out_ids) or (j not in in_ids):
                continue
            degree[i] = degree.get(i,np.asarray([0,0])) + np.asarray([0,1])
            degree[j] = degree.get(j, np.asarray([0, 0])) + np.asarray([1,0])
        return  degree



    # TODO: add feature_fallback decorator and rename the API more meaningful
    def batch(self, batch_ids):
        """Return batch of vectors from the sparse adjacency matrix corresponding to provided batch_ids.

        Parameters
        ----------
        batch_ids: array, required
            An array containing the ids of rows to be returned from the sparse adjacency matrix.
        """

        return self.matrix[batch_ids]

    @staticmethod
    def _find_min(vect):
        """ Return the lowest number and its position (index) in a given vector (array).

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
        """Covert a 2d array into sparse triplet format.

        Parameters
        ----------
        mat: 2d array, required
            A Numpy 2d array of integers.
        ids: list, optional, default: None
            A list of ids (or labels) of the objects to be used in the output triplet matrix.

        Returns
        -------
        A set corresponding to the sparse triplet representation of mat.
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
        triplets: Python set, required
            A Python set representing an adjacency matrix in the sparse triplet format.

        Returns
        -------
        Python set representing a symmetric adjacency matrix.
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

        Returns
        -------
        graph_modality: :obj:`<cornac.data.GraphModality>`
            GraphModality object.
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

        for i in trange(n, desc='Building KNN graph', disable=not verbose):
            c_id = 0
            for j in range(n):
                if i != j:
                    sim = np.dot(features[i], features[j])
                    if c_id <= k - 1:
                        N[i, c_id] = j
                        S[i, c_id] = sim
                        c_id += 1
                    else:
                        m_id, m = GraphModality._find_min(S[i])
                        if sim > m:
                            N[i, m_id] = j
                            S[i, m_id] = sim
        return N

    @classmethod
    def from_feature(cls, features, k=5, ids=None, similarity="cosine", symmetric=False, verbose=True):
        """Instantiate a GraphModality with a KNN graph build using input features.

        Parameters
        ----------
        features: 2d Numpy array, shape: [n_objects, n_features], required
            A 2d Numpy array of features, e.g., visual, textual, etc.

        k: int, optional, default: 5
            The number of nearest neighbors

        ids: array, optional, default: None
            The list of object ids or labels, which align with the rows of features. \
            For instance if you use textual (bag-of-word) features, \
            then "ids" should be the same as the input to cornac.data.TextModality.

        similarity: string, optional, default: "cosine"
            The similarity measure. At this time only the cosine is supported

        symmetric: bool, optional, default: False
            When True the resulting KNN-Graph is made symmetric

        verbose: bool, default: False
            The verbosity flag.

        Returns
        -------
        graph_modality: :obj:`<cornac.data.GraphModality>`
            GraphModality object.
        """
        # build knn graph
        knn_graph_array = GraphModality._build_knn(features, k, similarity, verbose=verbose)
        knn_graph_triplet = GraphModality._to_triplet(mat=knn_graph_array, ids=ids)
        if symmetric:
            if verbose:
                print("Symmetrizing the graph")
            knn_graph_triplet = GraphModality._to_symmetric(knn_graph_triplet)

        return cls(data=knn_graph_triplet)
