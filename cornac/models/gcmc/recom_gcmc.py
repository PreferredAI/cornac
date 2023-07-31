"""Main class for GCMC recommender model"""
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

from ..recommender import Recommender

from .gcmc import process_test_set, fit_torch, get_score


class GCMC(Recommender):
    """
    Graph Convolutional Matrix Completion (GCMC)

    Parameters
    ----------
    name: string, default: 'GCMC'
        The name of the recommender model.

    max_iter: int, default: 2000
        Maximum number of iterations or the number of epochs for SGD

    learning_rate: float, default: 0.01
        The learning rate for SGD

    optimizer: string, default: 'adam'. Supported values: 'adam','sgd'.
        The optimization method used for SGD

    activation_model: string, default: 'leaky'
        The activation function used in the GCMC model. Supported values:
        ['leaky', 'linear','sigmoid','relu', 'tanh']

    gcn_agg_units: int, default: 500
        The number of units in the graph convolutional layers

    gcn_out_units: int, default: 75
        The number of units in the output layer

    gcn_dropout: float, default: 0.7
        The dropout rate for the graph convolutional layers

    gcn_agg_accum: string, default:'stack'
        The graph convolutional layer aggregation type. Supported values:
        ['stack', 'sum']

    share_param: bool, default: False
        Whether to share the parameters in the graph convolutional layers

    gen_r_num_basis_func: int, default: 2
        The number of basis functions used in the generating rating function

    train_grad_clip: float, default: 1.0
        The gradient clipping value for training

    train_valid_interval: int, default: 1
        The validation interval for training

    train_early_stopping_patience: int, default: 100
        The patience for early stopping

    train_min_learning_rate: float, default: 0.001
        The minimum learning rate for SGD

    train_decay_patience: int, default: 50
        The patience for learning rate decay

    train_lr_decay_factor: float, default: 0.5
        The learning rate decay factor

    trainable: boolean, default: True
        When False, the model is not trained and Cornac

    verbose: boolean, default: True
        When True, some running logs are displayed

    seed: int, default: None
        Random seed for parameters initialization

    References
    ----------
    * Rianne van den Berg, Thomas N. Kipf, Max Welling.
    Graph Convolutional Matrix Completion.
    """
    def __init__(
        self,
        name="GCMC",
        max_iter=2000,
        learning_rate=0.01,
        optimizer="adam",
        activation_model="leaky",
        gcn_agg_units=500,
        gcn_out_units=75,
        gcn_dropout=0.7,
        gcn_agg_accum="stack",
        share_param=False,
        gen_r_num_basis_func=2,
        train_grad_clip=1.0,
        train_valid_interval=1,
        train_early_stopping_patience=100,
        train_min_learning_rate=0.001,
        train_decay_patience=50,
        train_lr_decay_factor=0.5,
        trainable=True,
        verbose=False,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)

        self.name = name
        self.max_iter = max_iter
        self.verbose = verbose
        self.seed = seed
        self.activation_model = activation_model
        self.gcn_agg_units = gcn_agg_units
        self.gcn_out_units = gcn_out_units
        self.gcn_dropout = gcn_dropout
        self.gcn_agg_accum = gcn_agg_accum
        self.share_param = share_param
        self.gen_r_num_basis_func = gen_r_num_basis_func
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.train_grad_clip = train_grad_clip
        self.train_valid_interval = train_valid_interval
        self.train_early_stopping_patience = train_early_stopping_patience
        self.train_min_learning_rate = train_min_learning_rate
        self.train_decay_patience = train_decay_patience
        self.train_lr_decay_factor = train_lr_decay_factor
        self.u_i_rating_dict = None
        self.train_enc_graph = None
        self.net = None
        self.verbose = verbose
        self.device = None

    def fit(self, train_set, val_set=None):
        Recommender.fit(self, train_set, val_set)

        if self.trainable:
            fit_torch(self, train_set, val_set)

        return self

    def transform(self, test_set):
        self.u_i_rating_dict = process_test_set(self, test_set)

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

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
            Relative scores that the user gives to the item or
            to all known items
        """

        return get_score(self, user_idx, item_idx)
