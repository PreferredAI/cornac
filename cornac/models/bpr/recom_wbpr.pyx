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

# cython: language_level=3

cimport cython
from cython cimport floating, integral
from cython.parallel import parallel, prange
from libc.math cimport exp
from libcpp cimport bool
from libcpp.algorithm cimport binary_search

import numpy as np
cimport numpy as np

from ..recommender import Recommender
from ...exception import ScoreException
from ...utils import get_rng
from ...utils import fast_dot
from ...utils.common import scale

from .recom_bpr import BPR
from .recom_bpr cimport RNGVector


class WBPR(BPR):
    """Weighted Bayesian Personalized Ranking.

    Parameters
    ----------
    k: int, optional, default: 10
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    learning_rate: float, optional, default: 0.001
        The learning rate for SGD.

    lambda_reg: float, optional, default: 0.001
        The regularization hyper-parameter.

    num_threads: int, optional, default: 0
        Number of parallel threads for training. If num_threads=0, all CPU cores will be utilized.
        If seed is not None, num_threads=1 to remove randomness from parallelization.

    trainable: boolean, optional, default: True
        When False, the model will not be re-trained, and input of pre-trained parameters are required.

    verbose: boolean, optional, default: True
        When True, some running logs are displayed.

    init_params: dictionary, optional, default: None
        Initial parameters, e.g., init_params = {'U': user_factors, 'V': item_factors, 'Bi': item_biases}

    seed: int, optional, default: None
        Random seed for weight initialization.
        If specified, training will take longer because of single-thread (no parallelization).

    References
    ----------
    * Gantner, Zeno, Lucas Drumond, Christoph Freudenthaler, and Lars Schmidt-Thieme. \
    "Personalized ranking for non-uniformly sampled items." In Proceedings of KDD Cup 2011, pp. 231-247. 2012.
    """

    def __init__(self, name='WBPR', k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.01,
                 num_threads=0, trainable=True, verbose=False, init_params=None, seed=None):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.init_params = {} if init_params is None else init_params
        self.seed = seed

        import multiprocessing
        if seed is not None:
            self.num_threads = 1
        elif num_threads > 0 and num_threads < multiprocessing.cpu_count():
            self.num_threads = num_threads
        else:
            self.num_threads = multiprocessing.cpu_count()

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """
        Recommender.fit(self, train_set, val_set)

        from tqdm import trange
        from ...utils import get_rng
        from ...utils.init_utils import zeros, uniform

        rng = get_rng(self.seed)
        self.u_factors = self.init_params.get(
            'U', 
            (uniform((train_set.total_users, self.k), random_state=rng) - 0.5) / self.k
        )
        self.i_factors = self.init_params.get(
            'V', 
            (uniform((train_set.total_items, self.k), random_state=rng) - 0.5) / self.k
        )
        self.i_biases = self.init_params.get('Bi', zeros(train_set.total_items))

        if not self.trainable:
            return

        X = train_set.matrix # csr_matrix
        # this basically calculates the 'row' attribute of a COO matrix
        # without requiring us to get the whole COO matrix
        user_counts = np.ediff1d(X.indptr)
        user_ids = np.repeat(np.arange(train_set.num_users), user_counts).astype(X.indices.dtype)

        cdef:
            int num_threads = self.num_threads
            # user the same RNG for weighted sampling with negative items            
            RNGVector rng_vec = RNGVector(num_threads, len(user_ids) - 1, rng.randint(2 ** 31))

        with trange(self.max_iter, disable=not self.verbose) as progress:
            for epoch in progress:
                correct, skipped = self._fit_sgd(rng_vec, rng_vec, num_threads,
                                                 user_ids, X.indices, X.indices, X.indptr,
                                                 self.u_factors, self.i_factors, self.i_biases)
                progress.set_postfix({"correct": "%.2f%%" % (100.0 * correct / (len(user_ids) - skipped)),
                                      "skipped": "%.2f%%" % (100.0 * skipped / len(user_ids))})
        if self.verbose:
            print('Optimization finished!')

        return self