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

import random
import multiprocessing as mp
import ctypes as c

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from ..recommender import Recommender
from ...utils.common import sigmoid
from ...utils.common import scale
from ...exception import ScoreException
from ...utils import get_rng
from ...utils.init_utils import uniform


class MTER(Recommender):
    """Multi-Task Explainable Recommendation

    Parameters
    ----------
    name: string, optional, default: 'MTER'
        The name of the recommender model.

    rating_scale: float, optional, default: 5.0
        The maximum rating score of the dataset.

    n_user_factors: int, optional, default: 15
        The dimension of the user latent factors.

    n_item_factors: int, optional, default: 15
        The dimension of the item latent factors.

    n_aspect_factors: int, optional, default: 12
        The dimension of the aspect latent factors.

    n_opinion_factors: int, optional, default: 12
        The dimension of the opinion latent factors.

    n_bpr_samples: int, optional, default: 1000
        The number of samples from all BPR pairs.

    n_element_samples: int, optional, default: 50
        The number of samples from all ratings in each iteration.

    lambda_reg: float, optional, default: 0.1
        The regularization parameter.

    lambda_bpr: float, optional, default: 10.0
        The regularization parameter for BPR.

    n_epochs: int, optional, default: 200000
        Maximum number of epochs for training.

    lr: float, optional, default: 0.1
        The learning rate for optimization

    n_threads: int, optional, default: 0
        Number of parallel threads for training. If n_threads=0, all CPU cores will be utilized.
        If seed is not None, n_threads=1 to remove randomness from parallelization.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already \
        pre-trained (U, I, A, O, G1, G2, and G3 are not None).

    verbose: boolean, optional, default: False
        When True, running logs are displayed.

    init_params: dictionary, optional, default: None
        List of initial parameters, e.g., init_params = {'U':U, 'I':I, 'A':A, 'O':O, 'G1':G1, 'G2':G2, 'G3':G3}

        U: ndarray, shape (n_users, n_user_factors)
            The user latent factors, optional initialization via init_params
            
        I: ndarray, shape (n_items, n_item_factors)
            The item latent factors, optional initialization via init_params
        
        A: ndarray, shape (num_aspects+1, n_aspect_factors)
            The aspect latent factors, optional initialization via init_params
        
        O: ndarray, shape (num_opinions, n_opinion_factors)
            The opinion latent factors, optional initialization via init_params
        
        G1: ndarray, shape (n_user_factors, n_item_factors, n_aspect_factors)
            The core tensor for user, item, and aspect factors, optional initialization via init_params
        
        G2: ndarray, shape (n_user_factors, n_aspect_factors, n_opinion_factors)
            The core tensor for user, aspect, and opinion factors, optional initialization via init_params
        
        G3: ndarray, shape (n_item_factors, n_aspect_factors, n_opinion_factors)
            The core tensor for item, aspect, and opinion factors, optional initialization via init_params

    seed: int, optional, default: None
        Random seed for parameters initialization.

    References
    ----------
    Nan Wang, Hongning Wang, Yiling Jia, and Yue Yin. 2018. \
    Explainable Recommendation via Multi-Task Learning in Opinionated Text Data. \
    In The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval (SIGIR '18). \
    ACM, New York, NY, USA, 165-174. DOI: https://doi.org/10.1145/3209978.3210010
    """

    def __init__(
        self,
        name="MTER",
        rating_scale=5.0,
        n_user_factors=15,
        n_item_factors=15,
        n_aspect_factors=12,
        n_opinion_factors=12,
        n_bpr_samples=1000,
        n_element_samples=50,
        lambda_reg=0.1,
        lambda_bpr=10,
        n_epochs=200000,
        lr=0.1,
        n_threads=0,
        trainable=True,
        verbose=False,
        init_params=None,
        seed=None,
    ):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)

        self.rating_scale = rating_scale
        self.n_user_factors = n_user_factors
        self.n_item_factors = n_item_factors
        self.n_aspect_factors = n_aspect_factors
        self.n_opinion_factors = n_opinion_factors
        self.n_bpr_samples = n_bpr_samples
        self.n_element_samples = n_element_samples
        self.lambda_reg = lambda_reg
        self.lambda_bpr = lambda_bpr
        self.n_epochs = n_epochs
        self.lr = lr
        self.seed = seed

        if seed is not None:
            self.n_threads = 1
        elif n_threads > 0 and n_threads < mp.cpu_count():
            self.n_threads = n_threads
        else:
            self.n_threads = mp.cpu_count()

        # Init params if provided
        self.init_params = {} if init_params is None else init_params
        self.G1 = self.init_params.get("G1", None)
        self.G2 = self.init_params.get("G2", None)
        self.G3 = self.init_params.get("G3", None)
        self.U = self.init_params.get("U", None)
        self.I = self.init_params.get("I", None)
        self.A = self.init_params.get("A", None)
        self.O = self.init_params.get("O", None)

    def _init(self):
        U_shape = (self.train_set.num_users, self.n_user_factors)
        I_shape = (self.train_set.num_items, self.n_item_factors)
        A_shape = (self.train_set.sentiment.num_aspects + 1, self.n_aspect_factors)
        O_shape = (self.train_set.sentiment.num_opinions, self.n_opinion_factors)
        G1_shape = (self.n_user_factors, self.n_item_factors, self.n_aspect_factors)
        G2_shape = (self.n_user_factors, self.n_aspect_factors, self.n_opinion_factors)
        G3_shape = (self.n_item_factors, self.n_aspect_factors, self.n_opinion_factors)

        rng = get_rng(self.seed)
        if self.G1 is None:
            self.G1 = uniform(np.product(G1_shape), random_state=rng)
        if self.G2 is None:
            self.G2 = uniform(np.product(G2_shape), random_state=rng)
        if self.G3 is None:
            self.G3 = uniform(np.product(G3_shape), random_state=rng)
        if self.U is None:
            self.U = uniform(np.product(U_shape), random_state=rng)
        if self.I is None:
            self.I = uniform(np.product(I_shape), random_state=rng)
        if self.A is None:
            self.A = uniform(np.product(A_shape), random_state=rng)
        if self.O is None:
            self.O = uniform(np.product(O_shape), random_state=rng)

        mp_U = mp.Array(c.c_double, self.U.flatten())
        mp_I = mp.Array(c.c_double, self.I.flatten())
        mp_A = mp.Array(c.c_double, self.A.flatten())
        mp_O = mp.Array(c.c_double, self.O.flatten())
        mp_G1 = mp.Array(c.c_double, self.G1.flatten())
        mp_G2 = mp.Array(c.c_double, self.G2.flatten())
        mp_G3 = mp.Array(c.c_double, self.G3.flatten())

        self.G1 = np.frombuffer(mp_G1.get_obj()).reshape(G1_shape)
        self.G2 = np.frombuffer(mp_G2.get_obj()).reshape(G2_shape)
        self.G3 = np.frombuffer(mp_G3.get_obj()).reshape(G3_shape)
        self.U = np.frombuffer(mp_U.get_obj()).reshape(U_shape)
        self.I = np.frombuffer(mp_I.get_obj()).reshape(I_shape)
        self.A = np.frombuffer(mp_A.get_obj()).reshape(A_shape)
        self.O = np.frombuffer(mp_O.get_obj()).reshape(O_shape)

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

        if self.trainable:
            (
                rating_matrix,
                user_item_aspect,
                user_aspect_opinion,
                item_aspect_opinion,
                user_item_pairs,
            ) = self._build_data(self.train_set)

            self.G1, self.G2, self.G3, self.U, self.I, self.A, self.O = self._fit_mter(
                self.n_epochs,
                self.n_threads,
                self.lr,
                self.n_element_samples,
                self.n_bpr_samples,
                self.lambda_bpr,
                self.lambda_reg,
                rating_matrix,
                user_item_aspect,
                user_aspect_opinion,
                item_aspect_opinion,
                user_item_pairs,
                self.G1,
                self.G2,
                self.G3,
                self.U,
                self.I,
                self.A,
                self.O,
            )

        return self

    def _fit_mter(
        self,
        n_epochs,
        n_threads,
        lr,
        n_element_samples,
        n_bpr_samples,
        lambda_bpr,
        lambda_reg,
        rating_matrix,
        user_item_aspect,
        user_aspect_opinion,
        item_aspect_opinion,
        user_item_pairs,
        G1,
        G2,
        G3,
        U,
        I,
        A,
        O,
    ):
        from .mter import paraserver, grad_worker_mse, grad_worker_bpr

        mp_del_g1_arr = mp.Array(c.c_double, int(np.product(G1.shape)))
        mp_del_g2_arr = mp.Array(c.c_double, int(np.product(G2.shape)))
        mp_del_g3_arr = mp.Array(c.c_double, int(np.product(G3.shape)))
        mp_del_u_arr = mp.Array(c.c_double, int(np.product(U.shape)))
        mp_del_i_arr = mp.Array(c.c_double, int(np.product(I.shape)))
        mp_del_a_arr = mp.Array(c.c_double, int(np.product(A.shape)))
        mp_del_o_arr = mp.Array(c.c_double, int(np.product(O.shape)))

        del_g1 = np.frombuffer(mp_del_g1_arr.get_obj()).reshape(G1.shape)
        del_g2 = np.frombuffer(mp_del_g2_arr.get_obj()).reshape(G2.shape)
        del_g3 = np.frombuffer(mp_del_g3_arr.get_obj()).reshape(G3.shape)
        del_u = np.frombuffer(mp_del_u_arr.get_obj()).reshape(U.shape)
        del_i = np.frombuffer(mp_del_i_arr.get_obj()).reshape(I.shape)
        del_a = np.frombuffer(mp_del_a_arr.get_obj()).reshape(A.shape)
        del_o = np.frombuffer(mp_del_o_arr.get_obj()).reshape(O.shape)

        lock = mp.Lock()
        q_samples_mse = mp.Queue()
        q_samples_bpr = mp.Queue()

        num_grad = mp.Value("i", 0)
        error_square = mp.Value("d", 0)
        error_bpr = mp.Value("d", 0)

        processes = []
        ps = mp.Process(
            target=paraserver,
            args=(
                user_item_pairs,
                user_item_aspect,
                user_aspect_opinion,
                item_aspect_opinion,
                n_element_samples,
                n_bpr_samples,
                lambda_reg,
                n_epochs,
                lr,
                G1,
                G2,
                G3,
                U,
                I,
                A,
                O,
                error_square,
                error_bpr,
                q_samples_mse,
                q_samples_bpr,
                del_g1,
                del_g2,
                del_g3,
                del_u,
                del_i,
                del_a,
                del_o,
                num_grad,
                n_threads,
                self.seed,
                self.verbose,
            ),
        )

        ps.start()
        processes.append(ps)

        for _ in range(n_threads):
            p = mp.Process(
                target=grad_worker_mse,
                args=(
                    user_item_aspect,
                    user_aspect_opinion,
                    item_aspect_opinion,
                    G1,
                    G2,
                    G3,
                    U,
                    I,
                    A,
                    O,
                    error_square,
                    error_bpr,
                    lock,
                    q_samples_mse,
                    del_g1,
                    del_g2,
                    del_g3,
                    del_u,
                    del_i,
                    del_a,
                    del_o,
                    num_grad,
                ),
            )
            processes.append(p)
            p.start()

        for _ in range(n_threads):
            p = mp.Process(
                target=grad_worker_bpr,
                args=(
                    rating_matrix,
                    lambda_bpr,
                    G1,
                    U,
                    I,
                    A,
                    error_square,
                    error_bpr,
                    lock,
                    q_samples_bpr,
                    del_g1,
                    del_u,
                    del_i,
                    del_a,
                    num_grad,
                ),
            )
            processes.append(p)
            p.start()

        for process in processes:
            process.join()

        return G1, G2, G3, U, I, A, O

    def _build_data(self, data_set):
        import time

        start_time = time.time()
        if self.verbose:
            print("Building data started!")
        sentiment = self.train_set.sentiment
        (u_indices, i_indices, r_values) = data_set.uir_tuple
        rating_matrix = sp.csr_matrix(
            (r_values, (u_indices, i_indices)),
            shape=(self.train_set.num_users, self.train_set.num_items),
        )
        user_item_pairs = list(dict.fromkeys(zip(u_indices, i_indices)))
        user_item_aspect = {}
        user_aspect_opinion = {}
        item_aspect_opinion = {}
        for user_idx, sentiment_tup_ids_by_item in sentiment.user_sentiment.items():
            if self.train_set.is_unk_user(user_idx):
                continue
            for item_idx, tup_idx in sentiment_tup_ids_by_item.items():
                user_item_aspect[
                    (user_idx, item_idx, sentiment.num_aspects)
                ] = rating_matrix[user_idx, item_idx]
                for aspect_idx, opinion_idx, polarity in sentiment.sentiment[tup_idx]:
                    user_item_aspect[(user_idx, item_idx, aspect_idx)] = (
                        user_item_aspect.get((user_idx, item_idx, aspect_idx), 0)
                        + polarity
                    )
                    if (
                        polarity > 0
                    ):  # only include opinion with positive sentiment polarity
                        user_aspect_opinion[(user_idx, aspect_idx, opinion_idx)] = (
                            user_aspect_opinion.get(
                                (user_idx, aspect_idx, opinion_idx), 0
                            )
                            + 1
                        )
                        item_aspect_opinion[(item_idx, aspect_idx, opinion_idx)] = (
                            item_aspect_opinion.get(
                                (item_idx, aspect_idx, opinion_idx), 0
                            )
                            + 1
                        )

        for key in user_item_aspect.keys():
            if key[2] != sentiment.num_aspects:
                user_item_aspect[key] = self._compute_quality_score(
                    user_item_aspect[key]
                )

        for key in user_aspect_opinion.keys():
            user_aspect_opinion[key] = self._compute_attention_score(
                user_aspect_opinion[key]
            )

        for key in item_aspect_opinion.keys():
            item_aspect_opinion[key] = self._compute_attention_score(
                item_aspect_opinion[key]
            )

        if self.verbose:
            total_time = time.time() - start_time
            print("Building data completed in %d s" % total_time)
        return (
            rating_matrix,
            user_item_aspect,
            user_aspect_opinion,
            item_aspect_opinion,
            user_item_pairs,
        )

    def _compute_attention_score(self, count):
        return 1 + (self.rating_scale - 1) * (2 / (1 + np.exp(-count)) - 1)

    def _compute_quality_score(self, sentiment):
        return 1 + (self.rating_scale - 1) / (1 + np.exp(-sentiment))

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for that to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items

        """
        from .mter import get_value

        if item_idx is None:
            if self.train_set.is_unk_user(user_idx):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d" & user_idx
                )
            tensor_value1 = np.einsum(
                "abc,Ma->Mbc",
                self.G1,
                self.U[user_idx, :].reshape(1, self.n_user_factors),
            )
            tensor_value2 = np.einsum("Mbc,Nb->MNc", tensor_value1, self.I)
            item_scores = np.einsum("MNc,c->MN", tensor_value2, self.A[-1]).flatten()
            return item_scores
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(
                item_idx
            ):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )
            return get_value(self.G1, self.U, self.I, self.A, (user_idx, item_idx, -1))
