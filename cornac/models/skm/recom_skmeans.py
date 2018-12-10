"""
@author: Aghiles Salah <asalah@smu.edu.sg>
"""

import numpy as np
from ..recommender import Recommender
from .skmeans import *


class SKMeans(Recommender):
    """Spherical k-means based recommender.

    Parameters
    ----------
    k: int, optional, default: 5
        The number of clusters.

    max_iter: int, optional, default: 100
        Maximum number of iterations.

    name: string, optional, default: 'Skmeans'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already \
        trained. 
        
    tol : float, optional, default: 1e-6
        Relative tolerance with regards to skmeans' criterion to declare convergence.
        
    verbose: boolean, optional, default: True
        When True, the skmeans criterion (likelihood) is displayed after each iteration.

    init_par: numpy 1d array, optional, default: None
        The initial object parition, 1d array contaning the cluster label (int type starting from 0) \
        of each object (user). If par = None, then skmeans is initialized randomly.
      
    centroids: csc_matrix, shape (k,n_users)
        The maxtrix of cluster centroids.

    References
    ----------
    * Salah, Aghiles, Nicoleta Rogovschi, and Mohamed Nadif. "A dynamic collaborative filtering system \
    via a weighted clustering approach." Neurocomputing 175 (2016): 206-215.
    """

    def __init__(self, k=5, max_iter=100, name="Skmeans", trainable=True, tol=1e-6, verbose=True, init_par=None):
        Recommender.__init__(self, name=name, trainable=trainable)
        self.k = k
        self.par = init_par
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        self.centroids = None  # matrix of cluster centroids

    # fit the recommender model to the traning data
    def fit(self, X):
        """Fit the model to observations.

        Parameters
        ----------
        X: scipy sparse matrix, required
            the user-item preference matrix (traning data), in a scipy sparse format\
            (e.g., csc_matrix).
        """
        X1 = X.copy()
        X1 = X1.multiply(sp.csc_matrix(1. / (np.sqrt(X1.multiply(X1).sum(1).A1) + 1e-20)).T)
        if self.trainable:
            # Skmeans requires rows of X to have a unit L2 norm. We therefore need to make a copy of X as we should not modify the latter.
            res = skmeans(X1, k=self.k, max_iter=self.max_iter, tol=self.tol, verbose=self.verbose, init_par=self.par)
            self.centroids = res['centroids']
            self.par = res['partition']
        else:
            print('%s is trained already (trainable = False)' % (self.name))
        self.user_center_sim = X1 * self.centroids.T  # user-centroid cosine similarity matrix
        del (X1)



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
            user_pred = self.centroids.multiply(self.user_center_sim[user_index, :].T)
            # transform user_pred to a flatten array
            user_pred = user_pred.sum(0).A1 / (
                    self.user_center_sim[user_index, :].sum() + 1e-20)  # weighted average of cluster centroids
        else:
            user_pred = self.centroids[item_indexes,:].multiply(self.user_center_sim[user_index, item_indexes].T)  
            # transform user_pred to a flatten array
            user_pred = user_pred.sum(0).A1 / (
                    self.user_center_sim[user_index, item_indexes].sum() + 1e-20)  # weighted average of cluster centroids

        return user_pred




    def rank(self, user_index):
        ranking = None
        return ranking