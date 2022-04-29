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

import numpy as np
cimport numpy as np
from tqdm.auto import trange

from ..recommender import Recommender
from ...utils import get_rng

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

    use_bias: boolean, optional, default: True
        When True, item bias is used.

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

    def __init__(
        self, 
        name="WBPR", 
        k=10, 
        max_iter=100, 
        learning_rate=0.001, 
        lambda_reg=0.01,
        use_bias=True,
        num_threads=0, 
        trainable=True, 
        verbose=False, 
        init_params=None, 
        seed=None
    ):
        super().__init__(
            name=name, 
            k=k, 
            max_iter=max_iter, 
            learning_rate=learning_rate, 
            lambda_reg=lambda_reg, 
            use_bias=use_bias, 
            num_threads=num_threads, 
            trainable=trainable, 
            verbose=verbose, 
            init_params=init_params, 
            seed=seed
        )

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

        self._init()

        if not self.trainable:
            return self
        
        X, user_counts, user_ids = self._prepare_data()

        cdef:
            int num_threads = self.num_threads
            # user the same RNG for weighted sampling with negative items
            RNGVector rng_vec = RNGVector(num_threads, len(user_ids) - 1, self.rng.randint(2 ** 31))

        with trange(self.max_iter, disable=not self.verbose) as progress:
            for epoch in progress:
                correct, skipped = self._fit_sgd(rng_vec, rng_vec, num_threads,
                                                 user_ids, X.indices, X.indices, X.indptr,
                                                 self.u_factors, self.i_factors, self.i_biases)
                progress.set_postfix({
                    "correct": "%.2f%%" % (100.0 * correct / (len(user_ids) - skipped)),
                    "skipped": "%.2f%%" % (100.0 * skipped / len(user_ids))
                })
        if self.verbose:
            print('Optimization finished!')

        return self