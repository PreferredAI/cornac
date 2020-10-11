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
from ..recommender import Recommender


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
        cornac's graph modality provide item auxiliary data (e.g., context, text, etc.) for PCRL.

    name: string, optional, default: 'PCRL'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already \
        pre-trained (Theta, Beta and Xi are not None). 
        
    w_determinist: boolean, optional, default: True
        When True, determinist wheights "W" are used for the generator network, \
        otherwise "W" is stochastic as in the original paper.

    init_params: dictionary, optional, default: None
        List of initial parameters, e.g., init_params = {'G_s':G_s, 'G_r':G_r, 'L_s':L_s, 'L_r':L_r}.
        
        Theta: ndarray, shape (n_users, k)
            The expected user latent factors.

        Beta: ndarray, shape (n_items, k)
            The expected item latent factors.

        G_s: ndarray, shape (n_users, k)
            Represent the "shape" parameters of Gamma distribution over Theta.
            
        G_r: ndarray, shape (n_users, k)
            Represent the "rate" parameters of Gamma distribution over Theta. 
        
        L_s: ndarray, shape (n_items, k)
            Represent the "shape" parameters of Gamma distribution over Beta.
            
        L_r: ndarray, shape (n_items, k)
            Represent the "rate" parameters of Gamma distribution over Beta. 
        
    References
    ----------
    * Salah, Aghiles, and Hady W. Lauw. Probabilistic Collaborative Representation Learning for Personalized Item Recommendation. \
    In UAI 2018.
    """

    def __init__(
        self,
        k=100,
        z_dims=[300],
        max_iter=300,
        batch_size=300,
        learning_rate=0.001,
        name="PCRL",
        trainable=True,
        verbose=False,
        w_determinist=True,
        init_params=None,
    ):

        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)

        self.k = k
        self.z_dims = (
            z_dims
        )  # the dimension of the second hidden layer (we consider a 2-layers PCRL)
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.w_determinist = w_determinist

        # Init params if provided
        self.init_params = {} if init_params is None else init_params
        self.Theta = self.init_params.get("Theta", None)
        self.Beta = self.init_params.get("Beta", None)
        self.Gs = self.init_params.get("G_s", None)
        self.Gr = self.init_params.get("G_r", None)
        self.Ls = self.init_params.get("L_s", None)
        self.Lr = self.init_params.get("L_r", None)

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

        # X = sp.csc_matrix(self.train_set.matrix)

        if self.trainable:
            from .pcrl import PCRL_

            # use pre-trained params if exists, otherwise from constructor
            init_params = {
                "G_s": self.Gs,
                "G_r": self.Gr,
                "L_s": self.Ls,
                "L_r": self.Lr,
            }

            # instanciate pcrl
            # train_aux_info = train_set.item_graph.matrix[:self.train_set.num_items, :self.train_set.num_items]
            pcrl_ = PCRL_(
                train_set=train_set,
                k=self.k,
                z_dims=self.z_dims,
                n_epoch=self.max_iter,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                B=1,
                w_determinist=self.w_determinist,
                init_params=init_params,
            ).learn()

            self.Theta = np.array(pcrl_.Gs) / np.array(pcrl_.Gr)
            self.Beta = np.array(pcrl_.Ls) / np.array(pcrl_.Lr)

            # overwrite init_params for future fine-tuning
            self.Gs = pcrl_.Gs
            self.Gr = pcrl_.Gr
            self.Ls = pcrl_.Ls
            self.Lr = pcrl_.Lr

        elif self.verbose:
            print("%s is trained already (trainable = False)" % (self.name))

        return self

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for a list of items.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for which to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items

        """

        if item_idx is None:
            user_pred = self.Beta * self.Theta[user_idx, :].T
        else:
            user_pred = self.Beta[item_idx, :] * self.Theta[user_idx, :].T
        # transform user_pred to a flatten array
        user_pred = np.array(user_pred, dtype="float64").flatten()

        return user_pred
