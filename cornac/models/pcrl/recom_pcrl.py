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

import numpy as np
import scipy.sparse as sp

from ..recommender import Recommender


# Recommender class for Probabilistic Collaborative Representation Learning (PCRL)
class PCRL(Recommender):
    """Probabilistic Collaborative Representation Learning.

    Parameters
    ----------
    k: int, optional, default: 100
        The dimension of the latent factors.
        
    z_dims: Numpy 1d array, optional, default: [300]
        The dimensions of the hidden intermdiate layers 'z' in the order \
        [dim(z_L), ...,dim(z_1)], please refer to Figure 1 in the orginal paper for more details.

    max_iter: int, optional, default: 300
        Maximum number of iterations (number of epochs) for variational PCRL.
        
    batch_size: int, optional, default: 300
        The batch size for SGD.
        
    learning_rate: float, optional, default: 0.001
        The learning rate for SGD.

    aux_info: see "cornac/examples/pcrl_example.py" in the GitHub repo for an example of how to use \
        cornac's graph module provide item auxiliary data (e.g., context, text, etc.) for PCRL.

    name: string, optional, default: 'PCRL'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already \
        pre-trained (Theta, Beta and Xi are not None). 
        
    w_determinist: boolean, optional, default: True
        When True, determinist wheights "W" are used for the generator network, \
        otherwise "W" is stochastic as in the original paper.

    init_params: dictionary, optional, default: {'G_s':None, 'G_r':None, 'L_s':None, 'L_r':None}
        List of initial parameters, e.g., init_params = {'G_s':G_s, 'G_r':G_r, 'L_s':L_s, 'L_r':L_r}, \
        where G_s and G_r are of type csc_matrix or np.array with the same shape as Theta, see below). \
        They represent respectively the "shape" and "rate" parameters of Gamma distribution over \
        Theta. It is the same for L_s, L_r and Beta.

    Theta: csc_matrix, shape (n_users,k)
        The expected user latent factors.

    Beta: csc_matrix, shape (n_items,k)
        The expected item latent factors.

    References
    ----------
    * Salah, Aghiles, and Hady W. Lauw. Probabilistic Collaborative Representation Learning for Personalized Item Recommendation. \
    In UAI 2018.
    """

    def __init__(self, k=100, z_dims=[300], max_iter=300, batch_size=300, learning_rate=0.001, name="pcrl",
                 trainable=True,
                 verbose=False, w_determinist=True, init_params={'G_s': None, 'G_r': None, 'L_s': None, 'L_r': None}):

        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)

        self.k = k
        self.z_dims = z_dims  # the dimension of the second hidden layer (we consider a 2-layers PCRL)
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.init_params = init_params
        self.w_determinist = w_determinist

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
        from .pcrl import PCRL_

        Recommender.fit(self, train_set)
        X = sp.csc_matrix(self.train_set.matrix)

        if self.trainable:
            # intanciate pcrl

            train_aux_info = train_set.item_graph.matrix[:self.train_set.num_items, :self.train_set.num_items]
            pcrl_ = PCRL_(cf_data=X, aux_data=train_aux_info, k=self.k, z_dims=self.z_dims, n_epoch=self.max_iter,
                          batch_size=self.batch_size, learning_rate=self.learning_rate, B=1,
                          w_determinist=self.w_determinist, init_params=self.init_params)
            pcrl_.learn()
            self.Theta = np.array(pcrl_.Gs) / np.array(pcrl_.Gr)
            self.Beta = np.array(pcrl_.Ls) / np.array(pcrl_.Lr)
        elif self.verbose:
            print('%s is trained already (trainable = False)' % (self.name))

    def score(self, user_id, item_id=None):
        """Predict the scores/ratings of a user for a list of items.

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
            user_pred = self.Beta * self.Theta[user_id, :].T
        else:
            user_pred = self.Beta[item_id, :] * self.Theta[user_id, :].T
        # transform user_pred to a flatten array
        user_pred = np.array(user_pred, dtype='float64').flatten()

        return user_pred
