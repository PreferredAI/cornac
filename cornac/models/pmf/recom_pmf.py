# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah
"""

import numpy as np
import scipy.sparse as sp
import pmf
from ..recommender import Recommender
from ...utils.generic_utils import sigmoid
from ...utils.generic_utils import map_to
from ...utils.generic_utils import intersects
from ...exception import ScoreException


class PMF(Recommender):
    """Probabilistic Matrix Factorization.

    Parameters
    ----------
    k: int, optional, default: 5
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    learning_rate: float, optional, default: 0.001
        The learning rate for SGD_RMSProp.
        
    gamma: float, optional, default: 0.9
        The weight for previous/current gradient in RMSProp.

    lamda: float, optional, default: 0.001
        The regularization parameter.

    name: string, optional, default: 'PMF'
        The name of the recommender model.
        
    variant: {"linear","non_linear"}, optional, default: 'non_linear'
        Pmf variant. If 'non_linear', the Gaussian mean is the output of a Sigmoid function.\
        If 'linear' the Gaussian mean is the output of the identity function.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already \
        pre-trained (U and V are not None).
        
    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    init_params: dictionary, optional, default: {'U':None,'V':None}
        List of initial parameters, e.g., init_params = {'U':U, 'V':V}. \
        U: a csc_matrix of shape (n_users,k), containing the user latent factors. \
        V: a csc_matrix of shape (n_items,k), containing the item latent factors.

    References
    ----------
    * Mnih, Andriy, and Ruslan R. Salakhutdinov. Probabilistic matrix factorization. \
    In NIPS, pp. 1257-1264. 2008.
    """

    def __init__(self, k=5, max_iter=100, learning_rate=0.001, gamma=0.9, lamda=0.001, name="PMF", variant='non_linear',
                 trainable=True, verbose=False, init_params={'U': None, 'V': None}):
        Recommender.__init__(self, name=name, trainable=trainable, verbose = verbose)
        self.k = k
        self.init_params = init_params
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lamda = lamda
        self.variant = variant

        self.ll = np.full(max_iter, 0)
        self.eps = 0.000000001
        self.U = init_params['U']  # matrix of user factors
        self.V = init_params['V']  # matrix of item factors


    # fit the recommender model to the traning data
    def fit(self, train_set):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: object of type TrainSet, required
            An object contraining the user-item preference in csr scipy sparse format,\
            as well as some useful attributes such as mappings to the original user/item ids.\
            Please refer to the class TrainSet in the "data" module for details.
        """

        Recommender.fit(self, train_set)

        X = self.train_set.matrix

        if self.trainable:
            # converting data to the triplet format (needed for cython function pmf)
            (rid, cid, val) = sp.find(X)
            val = np.array(val, dtype='float32')
            if self.variant == 'non_linear':  # need to map the ratings to [0,1]
                if [self.train_set.min_rating, self.train_set.max_rating] != [0, 1]:
                    val = map_to(val, 0., 1., self.train_set.min_rating, self.train_set.max_rating)
            rid = np.array(rid, dtype='int32')
            cid = np.array(cid, dtype='int32')
            tX = np.concatenate((np.concatenate(([rid], [cid]), axis=0).T, val.reshape((len(val), 1))), axis=1)
            del rid, cid, val

            if self.verbose:
                print('Learning...')

            if self.variant == 'linear':
                res = pmf.pmf_linear(tX, k=self.k, n_X=X.shape[0], d_X=X.shape[1], n_epochs=self.max_iter,
                                     lamda=self.lamda, learning_rate=self.learning_rate, gamma=self.gamma,
                                     init_params=self.init_params)
            elif self.variant == 'non_linear':
                res = pmf.pmf_non_linear(tX, k=self.k, n_X=X.shape[0], d_X=X.shape[1], n_epochs=self.max_iter,
                                         lamda=self.lamda, learning_rate=self.learning_rate, gamma=self.gamma,
                                         init_params=self.init_params)
            else:
                raise ValueError('variant must be one of {"linear","non_linear"}')

            self.U = np.asarray(res['U'])
            self.V = np.asarray(res['V'])

            if self.verbose:
                print('Learning completed')
        elif self.verbose:
            print('%s is trained already (trainable = False)' % (self.name))


    def score(self, user_id, item_id):
        """Predict the scores/ratings of a user for a list of items.

        Parameters
        ----------
        user_id: int, required
            The index of the user for whom to perform score predictions.
            
        item_id: int, required
            The index of the item to be scored by the user.

        Returns
        -------
        A scalar
            The estimated score (e.g., rating) for the user and item of interest
        """

        if self.train_set.is_unk_user(user_id) or self.train_set.is_unk_item(item_id):
            raise ScoreException("Can't make score prediction for (user_id=%d, item_id=%d)" % (user_id, item_id))

        user_pred = self.V[item_id, :].dot(self.U[user_id, :])

        if self.variant == "non_linear":
            user_pred = sigmoid(user_pred)
            user_pred = map_to(user_pred, self.train_set.min_rating, self.train_set.max_rating, 0., 1.)

        return user_pred


    def rank(self, user_id, candidate_item_ids=None):
        """Rank all test items for a given user.

        Parameters
        ----------
        user_id: int, required
            The index of the user for whom to perform item raking.

        candidate_item_ids: 1d array, optional, default: None
            A list of item indices to be ranked by the user.
            If `None`, list of ranked known item indices will be returned

        Returns
        -------
        Numpy 1d array
            Array of item indices sorted (in decreasing order) relative to some user preference scores.
        """

        if self.train_set.is_unk_user(user_id):
            return self.default_rank(candidate_item_ids)

        known_item_scores = self.V.dot(self.U[user_id, :])

        if self.variant == "non_linear":
            known_item_scores = sigmoid(known_item_scores)
            known_item_scores = map_to(known_item_scores, self.train_set.min_rating, self.train_set.max_rating, 0., 1.)

        if candidate_item_ids is None:
            ranked_item_ids = known_item_scores.argsort()[::-1]
            return ranked_item_ids
        else:
            num_items = max(self.train_set.num_items, max(candidate_item_ids) + 1)
            user_pref_scores = np.ones(num_items) * self.default_score() # use min_rating to shift unk items to the end
            user_pref_scores[:self.train_set.num_items] = known_item_scores

            ranked_item_ids = user_pref_scores.argsort()[::-1]
            ranked_item_ids = intersects(ranked_item_ids, candidate_item_ids, assume_unique=True)

            return ranked_item_ids