# -*- coding: utf-8 -*-
"""
@author: Guo Jingyao
"""

from ..recommender import Recommender
from .autorec import *
from ...utils.generic_utils import *
from ...exception import ScoreException


class Autorec(Recommender):
    """Autoencoders Meet Collaborative Filtering

    Parameters
    ----------
    k: int, optional, default: 20
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    learning_rate: float, optional, default: 0.001
        The learning rate for SGD.

    lamda: float, optional, default: 0.01
        The regularization parameter.

    batch_size: int, optional, default: 100
        The batch size of users for training.

    name: string, optional, default: 'autorec'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already \
        pre-trained (W and V are not None).

    init_params: dictionary, optional, default: None
        List of initial parameters, e.g., init_params = {'U':U, 'V':V} \
        please see below the definition of U and V.

    V: ndarray, shape (k,n_items)
        The encoder transformation matrix, optional initialization via init_params.

    W: ndarray, shape (n_items,k)
        The decoder transformation matrix, optional initialization via init_params.

    E: ndarray, shape (k,n_users)
    The encoded ratings, optional initialization via init_params.

    mu: ndarray, shape (k,1)
    The encoder bias, optional initialization via init_params.

    b: ndarray, shape (n_items,1)
    The decoder bias, optional initialization via init_params.

    g_act: encoder active function

    f_act: decoder active function

    References
    ----------
    * S. Sedhain, A. K. Menon, S. Sanner, and L. Xie. \
    Autorec: Autoencoders meet collaborative filtering.
    In Proceedings of the 24th International Conference on World Wide Web, WWW ’15 Companion, pages 111–112, New York, NY, USA, 2015. ACM.
    """

    def __init__(self, k=10, max_iter=100, learning_rate=0.001, lamda=0.01, batch_size=50, name="autorec",
                 trainable=True, g_act="Sigmoid", f_act="Identity", init_params={'V': None, 'mu': None,'W': None,'b': None,'E': None}):
        Recommender.__init__(self, name=name, trainable=trainable)
        self.k = k
        self.init_params = init_params
        self.max_iter = max_iter
        self.name = name
        self.learning_rate = learning_rate
        self.lamda = lamda
        self.batch_size = batch_size

        self.V = init_params['V']    # encoder transformation matrix
        self.mu = init_params['mu']  # encoder bias
        self.W = init_params['W']    # decoder transformation matrix
        self.b = init_params['b']    # decoder bias
        self.E = init_params['E']    # encoded ratings

        self.g_act = g_act
        self.f_act = f_act

        if (self.V is None) & (self.W is None):
            print("random initialize parameters")
        elif (self.V.shape[0]!=self.W.shape[1]) | (self.V.shape[1]!=self.W.shape[0]) | (self.V.shape[0]!=k):
            raise ValueError('initial parameters dimension error')

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
            # change the data to original user Id item Id and rating format
            data = np.ndarray(shape=(len(X.data), 3), dtype=float)
            data[:, 0] = X.tocoo().row
            data[:, 1] = X.tocoo().col
            data[:, 2] = X.data

            if self.verbose:
                print('Learning...')
            res = autorec(train_set, data, k=self.k, n_epochs=self.max_iter, lamda=self.lamda, learning_rate=self.learning_rate,
                          batch_size=self.batch_size, g_act=self.g_act, f_act=self.f_act, init_params=self.init_params)
            self.V = res['V']
            self.W = res['W']
            self.mu = res['mu']
            self.b = res['b']
            self.E = res['E']

            if self.verbose:
                print('Learning completed')
        elif self.verbose:
            print('%s is trained already (trainable = False)' % (self.name))

    # get prefiction for a single user (predictions for one user at a time for efficiency purposes)
    # predictions are not stored for the same efficiency reasons"""

    def Uscore(self, user_id):

        def idetity(x):
            return x

        def relu(x):
            return np.maximum(x, 0)

        if self.g_act == "Sigmoid":
            g_act = sigmoid
        elif self.g_act == "Relu":
            g_act = relu
        elif self.g_act == "Tanh":
            g_act = np.tanh
        elif self.g_act == "Identity":
            g_act = idetity
        else:
            raise NotImplementedError("Active function ERROR")

        if self.f_act == "Sigmoid":
            f_act = sigmoid
        elif self.f_act == "Relu":
            f_act = relu
        elif self.f_act == "Tanh":
            f_act = np.tanh
        elif self.f_act == "Identity":
            f_act = idetity
        else:
            raise NotImplementedError("Active function ERROR")

        user_pred = f_act(self.W.dot(g_act(self.E[:, user_id] + self.mu.flatten()))+self.b.flatten())

        # transform user_pred to a flatten array, but keep thinking about another possible format
        user_pred = np.array(user_pred, dtype='float64').flatten()
        user_pred = clipping(user_pred, 1, 5)

        return user_pred

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

        user_pred = self.Uscore(user_id)

        return user_pred[item_id]

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
            if candidate_item_ids is None:
                return np.arange(self.train_set.num_items)
            return candidate_item_ids

        known_item_scores = self.Uscore(user_id)

        if candidate_item_ids is None:
            ranked_item_ids = known_item_scores.argsort()[::-1]
            return ranked_item_ids
        else:
            num_items = max(self.train_set.num_items, max(candidate_item_ids) + 1)
            user_pref_scores = np.ones(num_items) * self.default_score()
            user_pref_scores[:self.train_set.num_items] = known_item_scores

            ranked_item_ids = user_pref_scores.argsort()[::-1]
            mask = np.in1d(ranked_item_ids, candidate_item_ids)
            ranked_item_ids = ranked_item_ids[mask]

            return ranked_item_ids
