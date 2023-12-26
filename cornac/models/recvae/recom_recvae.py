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
from tqdm.auto import trange

from ..recommender import Recommender
from ...exception import ScoreException


class RecVAE(Recommender):
    """
    RecVAE, a recommender system based on a Variational Autoencoder.

    Parameters
    ----------
    name : str, optional, default: 'RecVae'
        Name of the recommender model.

    hidden_dim : int, optional, default: 600
        Dimension of the hidden layer in the VAE architecture.

    latent_dim : int, optional, default: 200
        Dimension of the latent layer in the VAE architecture.

    batch_size : int, optional, default: 500
        Size of the batches used during training.

    beta : float, optional
        Weighting factor for the KL divergence term in the VAE loss function.

    gamma : float, optional, default: 0.005
        Weighting factor for the regularization term in the loss function.

    lr : float, optional, default: 5e-4
        Learning rate for the optimizer.

    n_epochs : int, optional, default: 50
        Number of epochs to train the model.

    n_enc_epochs : int, optional, default: 3
        Number of epochs to train the encoder part of VAE.

    n_dec_epochs : int, optional, default: 1
        Number of epochs to train the decoder part of VAE.

    not_alternating : boolean, optional, default: False
        If True, the model training will not alternate between encoder and decoder.

    trainable : boolean, optional, default: True
        When False, the model will not be re-trained, and input of pre-trained parameters are required.

    verbose : boolean, optional, default: False
        When True, running logs are displayed.

    seed : int, optional
        Random seed for weight initialization and training reproducibility.

    use_gpu : boolean, optional, default: True
        When True, training utilizes GPU if available.

    References
    ----------
    * RecVAE GitHub Repository: https://github.com/ilya-shenbin/RecVAE
    * Paper Link: https://arxiv.org/abs/1912.11160
    

    """
    
    def __init__(
        self,
        name="RecVae",

        hidden_dim = 600,
        latent_dim = 200,
        batch_size = 500,
        beta = None,
        gamma = 0.005,
        lr = 5e-4,
        n_epochs = 50,
        n_enc_epochs = 3,
        n_dec_epochs = 1,
        not_alternating = False,

        trainable=True,
        verbose=False,
        seed=None,
        use_gpu=True,
    ):

    

        Recommender.__init__(self,name=name, trainable=trainable, verbose=verbose)

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.beta = beta
        self.gamma = gamma
        self.lr = lr
        self.n_epochs = n_epochs
        self.n_enc_epochs = n_enc_epochs
        self.n_dec_epochs = n_dec_epochs
        self.not_alternating = not_alternating
        self.seed = seed


        import torch
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu") 


    def run(self,model, opts, train_set, my_batch_size, n_epochs, beta, gamma, dropout_rate):
        import torch
        train_data = train_set.csr_matrix 
        model.train()
        for _ in range(n_epochs):
            for i, batch_ids in enumerate(
                train_set.user_iter(my_batch_size, shuffle=True)
            ):

                ratings = torch.Tensor((train_data[batch_ids,:]).toarray()).to(self.device)                

                for optimizer in opts:
                    optimizer.zero_grad()

                _, loss = model(ratings, beta=beta, gamma=gamma, dropout_rate=dropout_rate)
                loss.backward()

                for optimizer in opts:
                    optimizer.step()
                    

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

        from .recvae import VAE
        import torch
        from torch import optim

        from ...metrics import NDCG
        from ...eval_methods import ranking_eval
                
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


            model_kwargs = {
                'hidden_dim': self.hidden_dim,
                'latent_dim': self.latent_dim,
                'input_dim': train_set.num_items,
            }

            self.recvae_model = VAE(**model_kwargs).to(self.device)



            learning_kwargs = {
                'model': self.recvae_model,
                'train_set': train_set,
                'my_batch_size': self.batch_size,
                'beta': self.beta,
                'gamma': self.gamma
            }

            self.mydata = train_set
            decoder_params = set(self.recvae_model.decoder.parameters())
            encoder_params = set(self.recvae_model.encoder.parameters())

            optimizer_encoder = optim.Adam(encoder_params, lr=self.lr)
            optimizer_decoder = optim.Adam(decoder_params, lr=self.lr)

            progress_bar = trange(1, self.n_epochs + 1, desc="RecVAE", disable=not self.verbose)

            for _ in progress_bar:
                if self.not_alternating:
                    self.run(opts=[optimizer_encoder, optimizer_decoder], n_epochs=1, dropout_rate=0.5, **learning_kwargs)
                else:
                    self.run(opts=[optimizer_encoder], n_epochs=self.n_enc_epochs, dropout_rate=0.5, **learning_kwargs)
                    self.recvae_model.update_prior()
                    self.run(opts=[optimizer_decoder], n_epochs=self.n_dec_epochs, dropout_rate=0, **learning_kwargs)

        
                ndcg_100 = ranking_eval(
                model=self,
                metrics=[NDCG(k=100)],
                train_set=train_set,
                test_set=train_set,
                )[0][0]
                        
                
                progress_bar.set_postfix(ndcg100 = ndcg_100)

            if self.verbose:
                print(f"Learning completed : [{ndcg_100}]")

        elif self.verbose:
            print("%s is trained already (trainable = False)" % (self.name))

          
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
        import torch

        ratings_in = self.mydata.matrix[user_idx,:]
        ratings_pred = self.recvae_model(torch.Tensor(ratings_in.toarray()).to(self.device), calculate_loss=False).cpu().detach().numpy().flatten()


        if item_idx is None:
            if not self.knows_user(user_idx):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d)" % user_idx
                )

            return ratings_pred
        else:
            if not (self.knows_user(user_idx) and self.knows_item(item_idx)):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )

            return ratings_pred[item_idx]
        
