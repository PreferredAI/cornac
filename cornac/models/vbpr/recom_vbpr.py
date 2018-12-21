# -*- coding: utf-8 -*-
"""
@author: Guo Jingyao <jyguo@smu.edu.sg>
"""

from .vbpr import *
from ..recommender import Recommender


class VBPR(Recommender):
    """Visual Bayesian Personalized Ranking.

    Parameters
    ----------
    k: int, optional, default: 5
        The dimension of the latent factors.

    d: int, optional, default: 5
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    aux_info：ndarray, shape (n_items, feature dimension), optional, default:None
        Image features of items

    learning_rate: float, optional, default: 0.001
        The learning rate for SGD.

    lamda: float, optional, default: 0.01
        The regularization parameter.

    batch_size: int, optional, default: 100
        The batch size for SGD.

    name: string, optional, default: 'BRP'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already \
        pre-trained (U and V are not None).

    init_params: dictionary, optional, default: None
        List of initial parameters, e.g., init_params = {'U':U, 'V':V} \
        please see below the definition of U and V.

    U: ndarray, shape (n_users,k)
        The user latent factors, optional initialization via init_params.

    V: ndarray, shape (n_items,k)
        The item latent factors, optional initialization via init_params.

    E: ndarray, shape (d, feature dimension)
        The matrix embedding deep CNN feature, optional initialization via init_params.

    Ue: ndarray, shape (n_users, d)
        The visual factors of users, optional initialization via init_params.

    References
    ----------
    * HE, Ruining et MCAULEY, Julian. VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback. In : AAAI. 2016. p. 144-150.
    """

    def __init__(self, k=10, d=10, max_iter=100, aux_info=None, learning_rate=0.001, lamda=0.01, batch_size=100,
                 name="vbpr", trainable=True,
                 init_params=None):
        Recommender.__init__(self, name=name, trainable=trainable)
        self.k = k
        self.d = d
        self.init_params = init_params
        self.aux_info = aux_info
        self.max_iter = max_iter
        self.name = name
        self.learning_rate = learning_rate
        self.lamda = lamda
        self.batch_size = batch_size

        self.U = init_params['U']  # matrix of user factors
        self.V = init_params['V']  # matrix of item factors
        self.E = init_params['E']  # matrix embedding deep CNN feature
        self.Ue = init_params['Ue']  # visual factors of users

    # fit the recommender model to the traning data
    def fit(self, X):
        """Fit the model to observations.

        Parameters
        ----------
        X: scipy sparse matrix, required
            the user-item preference matrix (traning data), in a scipy sparse format\
            (e.g., csc_matrix).
        """
        if self.trainable:
            # change the data to original user Id item Id and rating format
            cooX = X.tocoo()
            data = np.ndarray(shape=(len(cooX.data), 3), dtype=float)
            data[:, 0] = cooX.row
            data[:, 1] = cooX.col
            data[:, 2] = cooX.data

            print('Learning...')
            res = vbpr(X, data, k=self.k, d=self.d, aux_info=self.aux_info, n_epochs=self.max_iter, lamda=self.lamda,
                       learning_rate=self.learning_rate,
                       batch_size=self.batch_size, init_params=self.init_params)
            self.U = res['U']
            self.V = res['V']
            self.Ue = res['Ue']
            self.E = res['E']
            print('Learning completed')
        else:
            print('%s is trained already (trainable = False)' % (self.name))




    def score(self, user_index, item_indexes = None):
        """Predict the scores/ratings of a user for a list of items.

        Parameters
        ----------
        user_index: int, required
            The index of the user for whom to perform score predictions.
            
        item_indexes: 1d array, optional, default: None
            A list of item indexes for which to predict the rating score.\
            When "None", score prediction is performed for all test items of the given user. 

        Returns
        -------
        Numpy 1d array 
            Array containing the predicted values for the items of interest
        """

        if item_indexes is None:
            user_pred = self.U[user_index, :].dot(self.V.T) + self.Ue[user_index, :].dot(self.E).dot(self.aux_info.T)
            # user_pred = self.U[index_user, :].dot(self.V.T) + self.Ue[index_user, :]*self.E.dot(self.aux_info.T)
        else:
            user_pred = self.U[user_index, :].dot(self.V[item_indexes,:].T) + self.Ue[user_index, :].dot(self.E).dot(self.aux_info[item_indexes,:].T)
            
        # transform user_pred to a flatten array, but keep thinking about another possible format
        user_pred = np.array(user_pred, dtype='float64').flatten()

        return user_pred




    def rank(self, user_index, known_items = None):
        """Rank all test items for a given user.

        Parameters
        ----------
        user_index: int, required
            The index of the user for whom to perform item raking.
        known_items: 1d array, optional, default: None
            A list of item indices already known by the user

        Returns
        -------
        Numpy 1d array 
            Array of item indices sorted (in decreasing order) relative to some user preference scores. 
        """  
        
        u_pref_score = np.array(self.score(user_index))
        if known_items is not None:
            u_pref_score[known_items] = None
            
        rank_item_list = (-u_pref_score).argsort()  # ordering the items (in decreasing order) according to the preference score

        return rank_item_list