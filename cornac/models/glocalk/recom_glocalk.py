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
from ...exception import ScoreException
from ...utils import get_rng
from ...utils.init_utils import normal


class GLocalK(Recommender):
    def __init__(
        self,
        name="GLocalK",

        # Common hyperparameter settings
        n_hid = 500, # size of hidden layers
        n_dim = 5, # inner AE embedding size
        n_layers = 2, # number of hidden layers
        gk_size = 3, # width=height of kernel for convolution

        # Hyperparameters to tune for specific case
        max_epoch_p = 500, # max number of epochs for pretraining
        max_epoch_f = 1000, # max number of epochs for finetuning
        patience_p = 5, # number of consecutive rounds of early stopping condition before actual stop for pretraining
        patience_f = 10, # and finetuning
        tol_p = 1e-4, # minimum threshold for the difference between consecutive values of train rmse, used for early stopping, for pretraining
        tol_f = 1e-5 ,# and finetuning
        lambda_2 = 20., # regularisation of number or parameters
        lambda_s = 0.006, # regularisation of sparsity of the final matrix
        dot_scale = 1, # dot product weight for global kernel

        pre_min = 1.,
        pre_max = 5.,

        trainable=True,
        verbose=False,
        seed=None,
        use_gpu=True,
    ):

    

        Recommender.__init__(self,name=name, trainable=trainable, verbose=verbose)

        self.n_hid = n_hid  # size of hidden layers
        self.n_dim = n_dim
        self.n_layers = n_layers  # number of hidden layers
        self.gk_size = gk_size  # width=height of kernel for convolution
        self.max_epoch_p = max_epoch_p  # max number of epochs for pretraining
        self.max_epoch_f = max_epoch_f  # max number of epochs for finetuning
        self.patience_p = patience_p  # number of consecutive rounds of early stopping condition before actual stop for pretraining
        self.patience_f = patience_f  # and finetuning
        self.tol_p = tol_p  # minimum threshold for the difference between consecutive values of train rmse, used for early stopping, for pretraining
        self.tol_f = tol_f # and finetuning
        self.lambda_2 = lambda_2  # regularisation of number or parameters
        self.lambda_s = lambda_s  # regularisation of sparsity of the final matrix
        self.dot_scale =dot_scale   # dot product weight for global kernel

        self.pre_min = pre_min
        self.pre_max = pre_max


        self.seed = seed
        self.verbose = verbose
        
        import torch
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu") 
        

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
        import torch
        from .glocalk import GLocalK, learn
      
        if self.trainable:

            if self.verbose:
                print("Learning...")

            if self.seed is not None:
                np.random.seed(self.seed)
                torch.manual_seed(self.seed)
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.enabled = False

            self.glk = GLocalK(
                verbose=self.verbose,
                device = self.device,
                n_hid = self.n_hid,  # size of hidden layers
                n_dim = self.n_dim,
                n_layers = self.n_layers,  # number of hidden layers
                gk_size = self.gk_size,  # width=height of kernel for convolution
                max_epoch_p = self.max_epoch_p,  # max number of epochs for pretraining
                max_epoch_f = self.max_epoch_f,  # max number of epochs for finetuning
                patience_p = self.patience_p,  # number of consecutive rounds of early stopping condition before actual stop for pretraining
                patience_f = self.patience_f,  # and finetuning
                tol_p = self.tol_p,  # minimum threshold for the difference between consecutive values of train rmse, used for early stopping, for pretraining
                tol_f = self.tol_f, # and finetuning
                lambda_2 =self.lambda_2,  # regularisation of number or parameters
                lambda_s = self.lambda_s,  # regularisation of sparsity of the final matrix
                dot_scale = self.dot_scale,

                pre_min = self.pre_min,
                pre_max = self.pre_max,

            )

            self.pre=learn(self,train_set,val_set)

            if self.verbose:
                print(f"Learning completed : [{self.pre.shape}]")

        elif self.verbose:
            print("%s is trained already (trainable = False)" % (self.name))

        self.clean_pre = np.clip(self.pre, self.glk.pre_min,self.glk.pre_max)
        
        return self

    def score(self, user_idx, item_idx=None):
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
        
        if item_idx is None:
            if not self.knows_user(user_idx):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d)" % user_idx
                )

            return self.clean_pre[:,user_idx]
        else:
            
            if not (self.knows_user(user_idx) and self.knows_item(item_idx)):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )


            return self.clean_pre[:,user_idx][item_idx]
        
