# -*- coding: utf-8 -*-

"""
@author: Tran Thanh Binh
"""

from ..recommender import Recommender
from ...exception import ScoreException
import numpy as np


class CTR(Recommender):
    """Collaborative Topic Regression

    Parameters
    ----------
    name: string, default: 'CTR'
        The name of the recommender model.

    k: int, optional, default: 200
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    lambda_u: float, optional, default: 0.01
        The regularization parameter for users.

    lambda_v: float, optional, default: 0.01
        The regularization parameter for items.

    a: float, optional, default: 1
        The confidence of observed ratings.

    b: float, optional, default: 0.01
        The confidence of unseen ratings.

    eta: float, optional, default: 0.01
        Added value for smoothing phi.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already 
        pre-trained (U and V are not None).

    init_params: dictionary, optional, default: None
        List of initial parameters, e.g., init_params = {'U':U, 'V':V}
        U: ndarray, shape (n_users,k)
            The user latent factors, optional initialization via init_params.
        V: ndarray, shape (n_items,k)
            The item latent factors, optional initialization via init_params.

    seed: int, optional, default: None
        Random seed for weight initialization.

    References
    ----------
    Wang, Chong, and David M. Blei. "Collaborative topic modeling for recommending scientific articles."
    Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2011.

    """

    def __init__(self, name='CTR', k=200, lambda_u=0.01, lambda_v=0.01, eta=0.01,
                 a=1, b=0.01, max_iter=100, trainable=True, verbose=True, init_params=None,
                 seed=None):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.a = a
        self.b = b
        self.eta = eta
        self.name = name
        self.max_iter = max_iter
        self.verbose = verbose
        self.init_params = {} if not init_params else init_params
        self.seed = seed
        self.eps = 1e-100

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

        from ...utils import get_rng
        from ...utils.init_utils import xavier_uniform

        self.seed = get_rng(self.seed)
        self.n_item = self.train_set.num_items
        self.n_user = self.train_set.num_users
        self.n_voca = self.train_set.item_text.vocab.size

        self.U = self.init_params.get('U', xavier_uniform((self.n_user, self.k), self.seed))
        self.V = self.init_params.get('V', xavier_uniform((self.n_item, self.k), self.seed))

        self.theta = np.random.random([self.n_item, self.k])
        self.theta = self.theta / self.theta.sum(1)[:, np.newaxis]  # normalize
        self.beta = np.random.random([self.n_voca, self.k])
        self.beta = self.beta / self.beta.sum(0)  # normalize
        self.phi_sum = np.zeros([self.n_voca, self.k]) + self.eta

        if self.trainable:
            self._fit_ctr()

    def _fit_ctr(self, ):

        from tqdm import trange

        self.user_data = self._build_data(self.train_set.matrix)
        self.item_data = self._build_data(self.train_set.matrix.T.tocsr())
        self.doc_ids, self.doc_cnt = self._build_data(
            self.train_set.item_text.batch_bow(np.arange(self.n_item), keep_sparse=True))  # bag of word feature

        # collaborative training
        loop = trange(self.max_iter, disable=not self.verbose)
        for _ in loop:
            cf_loss = self._cf_update()  # u and v updating
            lda_loss = self._update_theta()
            self._do_m_step()
            loop.set_postfix(cf_loss=-cf_loss, lda_loss=-lda_loss)

        if self.verbose:
            print('Learning completed!')

    @staticmethod
    def _build_data(csr_mat):
        data = []
        index_list = []
        rating_list = []
        for i in range(csr_mat.shape[0]):
            j, k = csr_mat.indptr[i], csr_mat.indptr[i + 1]
            index_list.append(csr_mat.indices[j:k])
            rating_list.append(csr_mat.data[j:k])
        data.append(index_list)
        data.append(rating_list)
        return data

    def _cf_update(self):

        R_user = self.user_data[1]
        R_item = self.item_data[1]

        likelihood = 0.0
        VV = self.b * (self.V.T.dot(self.V)) + self.lambda_u * np.eye(self.k)

        # update user vector
        for i in range(self.n_user):
            idx_item = self.user_data[0][i]
            V_i = self.V[idx_item]
            R_i = R_user[i]
            A = VV + (self.a - self.b) * (V_i.T.dot(V_i))
            x = (self.a * V_i * (np.tile(R_i, (self.k, 1)).T)).sum(0)
            self.U[i] = np.linalg.solve(A, x)

            likelihood += -0.5 * self.lambda_u * np.dot(self.U[i], self.U[i])

        UU = self.b * (self.U.T.dot(self.U))

        # update item vector
        for j in range(self.n_item):
            idx_user = self.item_data[0][j]
            U_j = self.U[idx_user]
            R_j = R_item[j]

            UU_j = UU + (self.a - self.b) * (U_j.T.dot(U_j))
            A = UU_j + self.lambda_v * np.eye(self.k)
            x = (self.a * U_j * (np.tile(R_j, (self.k, 1)).T)).sum(0) + self.lambda_v * self.theta[j]
            self.V[j] = np.linalg.solve(A, x)

            likelihood += -0.5 * self.a * np.square(R_j).sum()
            likelihood += self.a * np.sum((U_j.dot(self.V[j])) * R_j)
            likelihood += - 0.5 * np.dot(self.V[j].dot(UU_j), self.V[j])

            ep = self.V[j, :] - self.theta[j, :]
            likelihood += -0.5 * self.lambda_v * np.sum(ep * ep)

        return likelihood

    def _do_m_step(self):

        self.beta = self.phi_sum / self.phi_sum.sum(0)
        self.phi_sum = np.zeros([self.n_voca, self.k]) + self.eta

    def _update_theta(self):

        loss = 0.0
        for vi in range(self.n_item):
            w = np.array(self.doc_ids[vi])
            word_beta = self.beta[w, :]
            phi = self.theta[vi, :] * word_beta + self.eps  # W x K
            phi = phi / phi.sum(1)[:, np.newaxis]
            gamma = np.array(self.doc_cnt[vi])[:, np.newaxis] * phi

            self.theta[vi, :], lda_loss = self._optimize_simplex(gamma=gamma, v=self.V[vi, :], opt_x=self.theta[vi, :],
                                                                 lambda_v=self.lambda_v, s=1)
            self.phi_sum[w, :] += gamma
            loss += lda_loss

        return loss

    def _optimize_simplex(self, gamma, v, lambda_v, opt_x, s=1):

        opt_x_old = np.copy(opt_x)
        f_old = self._fsimplex(gamma, v, lambda_v, opt_x)
        df = self._dfsimplex(gamma, v, lambda_v, opt_x)
        ab_sum = np.sum(np.absolute(df))
        if ab_sum > 1.0:
            df /= ab_sum
        opt_x -= df
        x_bar = self._simplex_project(opt_x, s=s)
        x_bar -= opt_x_old
        r = 0.5 * np.dot(df, x_bar)
        beta = 0.5
        t = beta

        for iter in range(100):
            opt_x = np.copy(opt_x_old)
            opt_x += t * x_bar
            f_new = self._fsimplex(gamma, v, lambda_v, opt_x)
            if (f_new > f_old + r * t):
                t *= beta
            else:
                break
        if not opt_x.sum() <= s + 1e-10 and np.alltrue(opt_x >= 0):
            print("Invalid values, outside simplex")

        return opt_x, f_new

    def _fsimplex(self, gamma, v, lambda_v, x):
        return -0.5 * lambda_v * np.dot((v - x).T, v - x) + np.sum(gamma * np.log(x))

    def _dfsimplex(self, gamma, v, lambda_v, x):
        return lambda_v * (v - x) + np.sum(gamma * (1 / x), axis=0)

    def _simplex_project(self, v, s=1):

        assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
        n, = v.shape  # will raise ValueError if v is not 1-D
        # check if we are already on the simplex
        if v.sum() == s and np.alltrue(v >= 0):
            # best projection: itself!
            return v
        # get the array of cumulative sums of a sorted (decreasing) copy of v
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        # get the number of > 0 components of the optimal solution
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
        # compute the Lagrange multiplier associated to the simplex constraint
        theta = (cssv[rho] - s) / (rho + 1.0)
        # compute the projection by thresholding v using theta
        w = (v - theta).clip(min=0)
        return w

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
