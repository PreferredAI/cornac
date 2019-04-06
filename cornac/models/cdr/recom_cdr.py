# -*- coding: utf-8 -*-

"""
@author: Tran Thanh Binh
"""

from ..recommender import Recommender
from ...exception import ScoreException
from .model import Model
import numpy as np
from tqdm import tqdm

try:
    import tensorflow as tf
except ImportError:
    tf = None


class CDR(Recommender):
    """Collaborative Deep Learning.

    Parameters
    ----------
    k: int, optional, default: 50
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    autoencoder_structure：array, optional, default: [200]
        The number of neurons of encoder/ decoder layer for SDAE

    learning_rate: float, optional, default: 0.001
        The learning rate for AdamOptimizer.

    loss_func: string , optional, default: "squared"
        Loss function name for optimizing ranking.
        Either "squared" or "log"

    lambda_u: float, optional, default: 0.1
        The regularization parameter for users.

    lambda_v: float, optional, default: 10
        The regularization parameter for items.

    lambda_w: float, optional, default: 0.1
        The regularization parameter for SDAE weights.

    lambda_n: float, optional, default: 1000
        The regularization parameter for SDAE output.

    autoencoder_corruption: float, optional, default: 0.3
        The corruption ratio for SDAE.

    dropout_rate: float, optional, default: 0.1
        The probability that each element is removed in dropout of SDAE.

    batch_size: int, optional, default: 100
        The batch size for SGD.

    name: string, optional, default: 'CDR'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already 
        pre-trained (U and V are not None).

    init_params: dictionary, optional, default: None
        List of initial parameters, e.g., init_params = {'U':U, 'V':V}
        U: ndarray, shape (n_users,k)
            The user latent factors, optional initialization via init_params.
        V: ndarray, shape (n_items,k)
            The item latent factors, optional initialization via init_params.


    """

    def __init__(self, k=50, autoencoder_structure=None, lambda_u=0.1, lambda_v=100,
                 lambda_w=0.1, lambda_n=1000, autoencoder_corruption=0.3, learning_rate=0.001,
                 dropout_rate=0.1, batch_size=128, max_iter=100, name="CDR", trainable=True, verbose=True,
                 vocab_size=8000, loss_func="squared", init_params=None):

        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.lambda_w = lambda_w
        self.lambda_n = lambda_n
        self.autoencoder_corruption = autoencoder_corruption
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.name = name
        self.init_params = init_params
        self.max_iter = max_iter
        self.autoencoder_structure = autoencoder_structure
        self.batch_size = batch_size
        self.verbose = verbose
        self.vocab_size = vocab_size
        self.loss_func = loss_func

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
        if not self.trainable:
            print('%s is trained already (trainable = False)' % (self.name))
            return

        if self.verbose:
            print('Learning...')

        self._cdr(train_set=train_set)  # Collaborative Deep Ranking

        if self.verbose:
            print('\nLearning completed')

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

        if item_id is None:
            if self.train_set.is_unk_user(user_id):
                raise ScoreException("Can't make score prediction for (user_id=%d)" % user_id)

            known_item_scores = self.V.dot(self.U[user_id, :])
            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_id) or self.train_set.is_unk_item(item_id):
                raise ScoreException("Can't make score prediction for (user_id=%d, item_id=%d)" % (user_id, item_id))
            user_pred = self.V[item_id, :].dot(self.U[user_id, :])

            return user_pred

    def _cdr(self, train_set):

        # get user and item weight

        n_users = self.train_set.num_users
        n_items = self.train_set.num_items

        text_feature = self.train_set.item_text.batch_bow(np.arange(n_items))  # bag of word feature
        text_feature = (text_feature - text_feature.min()) / (text_feature.max() - text_feature.min())  # normalization

        layer_sizes = [self.vocab_size] + self.autoencoder_structure + [self.k] + \
                      self.autoencoder_structure + [self.vocab_size]

        # Build model
        model = Model(n_users=n_users, n_items=n_items, n_vocab=self.vocab_size, k=self.k, layers=layer_sizes,
                      lambda_u=self.lambda_u, lambda_v=self.lambda_v, lambda_w=self.lambda_w,
                      lambda_n=self.lambda_n, lr=self.learning_rate, dropout_rate=self.dropout_rate,
                      loss_func=self.loss_func, init_params=self.init_params)

        # Training model
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            loop = tqdm(range(self.max_iter), disable=not self.verbose)
            for _ in loop:
                mask_corruption_np = np.random.binomial(1, 1 - self.autoencoder_corruption,
                                                        (n_items, self.vocab_size))

                for batch_u, batch_i, batch_j in train_set.uij_iter(batch_size=self.batch_size, shuffle=True):
                    feed_dict = {
                        model.mask_input: mask_corruption_np[batch_i, :],
                        model.text_input: text_feature[batch_i, :],
                        model.batch_u: batch_u,
                        model.batch_i: batch_i,
                        model.batch_j: batch_j
                    }

                    sess.run(model.opt1, feed_dict)  # train U, V
                    _, _loss = sess.run([model.opt2, model.loss], feed_dict)  # train SDAE
                    loop.set_postfix(loss=_loss)

            self.U, self.V = sess.run([model.U, model.V])

        tf.reset_default_graph()
