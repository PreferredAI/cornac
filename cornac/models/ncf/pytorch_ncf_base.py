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
from ...utils import get_rng


class NCFBase_PyTorch(Recommender):
    """
    Parameters
    ----------
    num_factors: int, optional, default: 8
        Embedding size of MF model.
    
    layers: list, optional, default: [64,32,16,8]
        MLP layers configuration.
    
    act_fn: str, optional, default: 'relu'
        Activation function used for MLP layers.

    num_epochs: int, optional, default: 20
        Number of epochs.

    batch_size: int, optional, default: 256
        Batch size.

    num_neg: int, optional, default: 4
        Number of negative instances to pair with a positive instance.

    lr: float, optional, default: 0.001
        Learning rate.

    reg: float, optional, default: 0.
        Regularization (weight decay).

    learner: str, optional, default: 'adam'
        Specify an optimizer: adagrad, adam, rmsprop, sgd

    early_stopping: {min_delta: float, patience: int}, optional, default: None
        If `None`, no early stopping. Meaning of the arguments: 
        
         - `min_delta`: the minimum increase in monitored value on validation set to be considered as improvement, \
           i.e. an increment of less than min_delta will count as no improvement.
         - `patience`: number of epochs with no improvement after which training should be stopped.

    name: string, optional, default: 'Torch-NCF'
        Name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already \
        pre-trained.

    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    seed: int, optional, default: None
        Random seed for parameters initialization.
    
    use_pretrain: bool, optional, default: False
        Whether to use pretrained model(s).
    
    use_NeuMF: bool, optional, default: False
        If GMF and MLP are initialized from NeuMF model.
    
    pretrained_GMF: optional, default: None
        Pretrained GMF model.
    
    pretrained_MLP: optional, default: None
        Pretrained MLP model.
    """

    def __init__(
        self,
        name="NCF-PyTorch",
        num_factors=8,
        layers=None,
        act_fn="relu",
        num_epochs=20,
        batch_size=256,
        num_neg=4,
        lr=1e-3,
        reg=0.0,
        learner="adam",
        early_stopping=None,
        trainable=True,
        verbose=True,
        seed=None,
        use_pretrain: bool = False,
        use_NeuMF: bool = False,
        pretrained_GMF=None,
        pretrained_MLP=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.num_factors = num_factors
        self.layers = layers
        self.act_fn = act_fn
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.lr = lr
        self.reg = reg
        self.learner = learner.lower()
        self.early_stopping = early_stopping
        self.seed = seed
        self.rng = get_rng(seed)
        self.use_pretrain = use_pretrain
        self.use_NeuMF = use_NeuMF
        self.pretrained_GMF = pretrained_GMF
        self.pretrained_MLP = pretrained_MLP

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

        if self.trainable:
            if not hasattr(self, "user_embedding"):
                self.num_users = self.train_set.num_users
                self.num_items = self.train_set.num_items

        return self

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
            Relative scores that the user gives to the item or to all known items

        """
        raise NotImplementedError("The algorithm is not able to make score prediction!")

    def monitor_value(self):
        """Calculating monitored value used for early stopping on validation set (`val_set`).
        This function will be called by `early_stop()` function.

        Returns
        -------
        res : float
            Monitored value on validation set.
            Return `None` if `val_set` is `None`.
        """
        if self.val_set is None:
            return None

        from ...metrics import Recall
        from ...eval_methods import ranking_eval

        recall_20 = ranking_eval(
            model=self,
            metrics=[Recall(k=20)],
            train_set=self.train_set,
            test_set=self.val_set,
        )[0][0]

        return recall_20
