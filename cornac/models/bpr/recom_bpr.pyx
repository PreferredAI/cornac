# -*- coding: utf-8 -*-
# cython: language_level=3

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from cornac.exception import ScoreException
from cornac.models.recommender import Recommender
from cornac.utils import fast_dot
from cornac.utils.common import scale
import multiprocessing
import tqdm
import numpy as np
cimport cython
from cython cimport floating, integral
from cython.parallel import parallel, prange
from libc.math cimport exp
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.algorithm cimport binary_search


cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937:
        mt19937(unsigned int)

    cdef cppclass uniform_int_distribution[T]:
        uniform_int_distribution(T, T)
        T operator()(mt19937) nogil


# thin wrapper around omp_get_thread_num (since referencing directly will cause OSX
# build to fail)
cdef extern from "recom_bpr.h" namespace "recom_bpr" nogil:
    cdef int get_thread_num()


@cython.boundscheck(False)
cdef bool has_non_zero(integral[:] indptr, integral[:] indices,
                       integral rowid, integral colid) nogil:
    """ Given a CSR matrix, returns whether the [rowid, colid] contains a non zero.
    Assumes the CSR matrix has sorted indices """
    return binary_search(&indices[indptr[rowid]], &indices[indptr[rowid + 1]], colid)


cdef class RNGVector(object):
    """ This class creates one c++ rng object per thread, and enables us to randomly sample
    positive/negative items here in a thread safe manner """
    cdef vector[mt19937] rng
    cdef vector[uniform_int_distribution[long]]  dist

    def __init__(self, int num_threads, long rows):
        for i in range(num_threads):
            self.rng.push_back(mt19937(np.random.randint(2**31)))
            self.dist.push_back(uniform_int_distribution[long](0, rows))

    cdef inline long generate(self, int thread_id) nogil:
        return self.dist[thread_id](self.rng[thread_id])



class BPR(Recommender):
    """Bayesian Personalized Ranking.

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
        Number of parallel threads for training.
        If 0, all CPU cores will be utilized.

    trainable: boolean, optional, default: True
        When False, the model will not be re-trained, and input of pre-trained parameters are required.

    verbose: boolean, optional, default: True
        When True, some running logs are displayed.

    init_params: dictionary, optional, default: None
        Initial parameters, e.g., init_params = {'U': user_factors, 'V': item_factors, 'Bi': item_biases}

    References
    ----------
    * Rendle, Steffen, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. \
    BPR: Bayesian personalized ranking from implicit feedback. In UAI, pp. 452-461. 2009.
    """

    def __init__(self, k=10, max_iter=100, learning_rate=0.01, lambda_reg=0.01, num_threads=0,
                 trainable=True, verbose=False, init_params=None):
        Recommender.__init__(self, name='BPR', trainable=trainable, verbose=verbose)
        self.factors = k
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.init_params = {} if init_params is None else init_params

        if num_threads > 0 and num_threads < multiprocessing.cpu_count():
            self.num_threads = num_threads
        else:
            self.num_threads = multiprocessing.cpu_count()

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
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

        X = train_set.matrix
        num_users, num_items = X.shape

        # this basically calculates the 'row' attribute of a COO matrix
        # without requiring us to get the whole COO matrix
        user_counts = np.ediff1d(X.indptr)
        user_ids = np.repeat(np.arange(num_users), user_counts).astype(X.indices.dtype)

        self.u_factors = self.init_params.get('U', None)  # matrix of user factors
        self.i_factors = self.init_params.get('V', None)  # matrix of item factors
        self.i_biases = self.init_params.get('Bi', None)  # vector of item biases

        # create factors if not already created.
        if self.u_factors is None:
            self.u_factors = (np.random.rand(num_users, self.factors).astype(np.float32) - .5)
            self.u_factors /= self.factors

            # set factors to all zeros for users without any ratings
            self.u_factors[user_counts == 0] = np.zeros(self.factors)

        if self.i_factors is None:
            self.i_factors = (np.random.rand(num_items, self.factors).astype(np.float32) - .5)
            self.i_factors /= self.factors

            # set factors to all zeros for items without any ratings
            item_counts = np.bincount(X.indices, minlength=num_items)
            self.i_factors[item_counts == 0] = np.zeros(self.factors)

        if self.i_biases is None:
            self.i_biases = np.zeros(num_items).astype(np.float32)

        self._fit_sgd(user_ids, X.indices, X.indptr,
                      self.u_factors, self.i_factors, self.i_biases)


    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit_sgd(self, integral[:] user_ids, integral[:] item_ids, integral[:] indptr,
                 floating[:, :] U, floating[:, :] V, floating[:] B):
        """Fit the model parameters (U, V, B) with SGD
        """
        cdef long num_samples = len(user_ids), s, i_index, j_index, correct, skipped
        cdef long num_items = self.train_set.num_items
        cdef integral f, i_id, j_id, thread_id
        cdef floating z, score, temp

        cdef floating lr = self.learning_rate
        cdef floating reg = self.lambda_reg
        cdef integral factors = self.factors
        cdef integral num_threads = self.num_threads

        cdef floating * user
        cdef floating * item_i
        cdef floating * item_j

        cdef RNGVector rng_i = RNGVector(num_threads, num_samples - 1)
        cdef RNGVector rng_j = RNGVector(num_threads, num_items - 1)

        progress = tqdm.trange(self.max_iter, disable=not self.verbose)
        for epoch in progress:
            correct = 0
            skipped = 0

            with nogil, parallel(num_threads=num_threads):
                thread_id = get_thread_num()

                for s in prange(num_samples, schedule='guided'):
                    i_index = rng_i.generate(thread_id)
                    i_id = item_ids[i_index]
                    j_id = rng_j.generate(thread_id)

                    # if the user has liked the item j, skip this for now
                    if has_non_zero(indptr, item_ids, user_ids[i_index], j_id):
                        skipped += 1
                        continue

                    # get pointers to the relevant factors
                    user, item_i, item_j = &U[user_ids[i_index], 0], &V[i_id, 0], &V[j_id, 0]

                    # compute the score
                    score = B[i_id] - B[j_id]
                    for f in range(factors):
                        score = score + user[f] * (item_i[f] - item_j[f])
                    z = 1.0 / (1.0 + exp(score))

                    if z < .5:
                        correct += 1

                    # update the factors via sgd.
                    for f in prange(factors):
                        temp = user[f]
                        user[f] += lr * (z * (item_i[f] - item_j[f]) - reg * user[f])
                        item_i[f] += lr * (z * temp - reg * item_i[f])
                        item_j[f] += lr * (-z * temp - reg * item_j[f])

                    # update item biases
                    B[i_id] += lr * (z - reg * B[i_id])
                    B[j_id] += lr * (-z - reg * B[j_id])

            progress.set_postfix({"correct": "%.2f%%" % (100.0 * correct / (num_samples - skipped)),
                                  "skipped": "%.2f%%" % (100.0 * skipped / num_samples)})

        if self.verbose:
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
            known_item_scores = self.i_biases
            if not unk_user:
                fast_dot(self.u_factors[user_id], self.i_factors, known_item_scores)
            return known_item_scores
        else:
            if self.train_set.is_unk_item(item_id):
                raise ScoreException("Can't make score prediction for (user_id=%d, item_id=%d)" % (user_id, item_id))
            item_score = self.i_biases[item_id]
            if not unk_user:
                item_score += np.dot(self.u_factors[user_id], self.i_factors[item_id])
            if self.train_set.min_rating != self.train_set.max_rating:
                item_score = scale(item_score, self.train_set.min_rating, self.train_set.max_rating, 0., 1.)
            return item_score
