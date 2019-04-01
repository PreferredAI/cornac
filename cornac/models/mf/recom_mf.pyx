# -*- coding: utf-8 -*-
# cython: language_level=3

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from cornac.models.recommender import Recommender
from cornac.exception import ScoreException
from cornac.utils import fast_dot
import tqdm
import numpy as np
cimport cython
from cython.parallel import prange
from cython cimport floating, integral
from libcpp cimport bool
from libc.math cimport abs


class MF(Recommender):
    """Matrix Factorization.

    Parameters
    ----------
    k: int, optional, default: 10
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    learning_rate: float, optional, default: 0.01
        The learning rate.

    lambda_reg: float, optional, default: 0.001
        The lambda value used for regularization.

    use_bias: boolean, optional, default: True
        When True, user, item, and global biases are used.

    early_stop: boolean, optional, default: False
        When True, delta loss will be checked after each iteration to stop learning earlier.

    trainable: boolean, optional, default: True
        When False, the model will not be re-trained, and input of pre-trained parameters are required.

    verbose: boolean, optional, default: True
        When True, running logs are displayed.

    init_params: dictionary, optional, default: None
        Initial parameters, e.g., init_params = {'U': user_factors, 'V': item_factors,
        'Bu': user_biases, 'Bi': item_biases}

    References
    ----------
    * Koren, Y., Bell, R., & Volinsky, C. Matrix factorization techniques for recommender systems. \
    In Computer, (8), 30-37. 2009.
    """

    def __init__(self, k=10, max_iter=20, learning_rate=0.01, lambda_reg=0.02, use_bias=True, early_stop=False,
                 trainable=True, verbose=False, init_params=None):
        Recommender.__init__(self, name='MF', trainable=trainable, verbose=verbose)

        self.k = k
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.use_bias = use_bias
        self.early_stop = early_stop
        self.init_params = {} if init_params is None else init_params

    def fit(self, train_set):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: object of type TrainSet, required
            An object contains the user-item preference in csr scipy sparse format,\
            as well as some useful attributes such as mappings to the original user/item ids.\
            Please refer to the class TrainSet in the "data" module for details.
        """
        Recommender.fit(self, train_set)

        if not self.trainable:
            print('%s is trained already (trainable = False)' % (self.name))
            return

        self.u_factors = self.init_params.get('U', None)
        self.i_factors = self.init_params.get('V', None)
        self.u_biases = self.init_params.get('Bu', None)
        self.i_biases = self.init_params.get('Bi', None)

        if self.u_factors is None:
            self.u_factors = np.random.normal(size=[train_set.num_users, self.k], loc=0., scale=0.01).astype(np.float32)
        if self.i_factors is None:
            self.i_factors = np.random.normal(size=[train_set.num_items, self.k], loc=0., scale=0.01).astype(np.float32)
        if self.u_biases is None:
            self.u_biases = np.zeros(train_set.num_users).astype(np.float32)
        if self.i_biases is None:
            self.i_biases = np.zeros(train_set.num_items).astype(np.float32)

        self.global_mean = train_set.global_mean if self.use_bias else 0.

        (rid, cid, val) = train_set.uir_tuple
        self._fit_sgd(rid, cid, val.astype(np.float32),
                      self.u_factors, self.i_factors, self.u_biases, self.i_biases)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit_sgd(self, integral[:] rid, integral[:] cid, floating[:] val,
                 floating[:, :] U, floating[:, :] V, floating[:] Bu, floating[:] Bi):
        """Fit the model parameters (U, V, Bu, Bi) with SGD
        """
        cdef long num_users = self.train_set.num_users
        cdef long num_items = self.train_set.num_items
        cdef long num_ratings = val.shape[0]
        cdef integral num_factors = self.k
        cdef integral max_iter = self.max_iter

        cdef floating reg = self.lambda_reg
        cdef floating mu = self.global_mean

        cdef bool use_bias = self.use_bias
        cdef bool early_stop = self.early_stop
        cdef bool verbose = self.verbose

        cdef floating lr = self.learning_rate
        cdef floating loss = 0
        cdef floating last_loss = 0
        cdef floating r, r_pred, error, u_f, i_f, delta_loss
        cdef integral u, i, f, j

        cdef floating * user
        cdef floating * item

        progress = tqdm.trange(max_iter, disable=not self.verbose)
        for epoch in progress:
            last_loss = loss
            loss = 0

            for j in prange(num_ratings, nogil=True):
                u, i, r = rid[j], cid[j], val[j]
                user, item = &U[u, 0], &V[i, 0]

                # predict rating
                r_pred = mu + Bu[u] + Bi[i]
                for f in prange(num_factors):
                    r_pred += user[f] * item[f]

                error = r - r_pred
                loss += error * error

                # update factors
                for f in prange(num_factors):
                    u_f, i_f = user[f], item[f]
                    user[f] += lr * (error * i_f - reg * u_f)
                    item[f] += lr * (error * u_f - reg * i_f)

                # update biases
                if use_bias:
                    Bu[u] += lr * (error - reg * Bu[u])
                    Bi[i] += lr * (error - reg * Bi[i])

            loss = 0.5 * loss
            progress.update(1)
            progress.set_postfix({"loss": "%.2f" % loss})

            delta_loss = loss - last_loss
            if early_stop and abs(delta_loss) < 1e-5:
                if verbose:
                    print('Early stopping, delta_loss = %.4f' % delta_loss)
                break

        if verbose:
            print('Optimization finished!')


    def score(self, user_id, item_id=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_id: int, required
            The index of the user for whom to perform score prediction.

        item_id: int, optional, default: None
            The index of the item for that to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items

        """
        unk_user = self.train_set.is_unk_user(user_id)

        if item_id is None:
            known_item_scores = np.add(self.i_biases, self.global_mean)
            if not unk_user:
                known_item_scores = np.add(known_item_scores, self.u_biases[user_id])
                fast_dot(self.u_factors[user_id], self.i_factors, known_item_scores)
            return known_item_scores
        else:
            unk_item = self.train_set.is_unk_item(item_id)
            if self.use_bias:
                item_score = self.global_mean
                if not unk_user:
                    item_score += self.u_biases[user_id]
                if not unk_item:
                    item_score += self.i_biases[item_id]
                if not unk_user and not unk_item:
                    item_score += np.dot(self.u_factors[user_id], self.i_factors[item_id])
            else:
                if unk_user or unk_item:
                    raise ScoreException("Can't make score prediction for (user_id=%d, item_id=%d)" % (user_id, item_id))
                item_score = np.dot(self.u_factors[user_id], self.i_factors[item_id])
            return item_score
