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
from copy import deepcopy


from tqdm.auto import trange


class Batch:

    def __init__(self, device, idx, data_in, data_out=None):
        self._device = device
        self._idx = idx
        self._data_in = data_in
        self._data_out = data_out

    def get_idx(self):
        return self._idx

    def get_idx_to_dev(self):
        import torch
        return torch.LongTensor(self.get_idx()).to(self._device)

    def get_ratings(self, is_out=False):
        data = self._data_out if is_out else self._data_in
        return data[self._idx]

    def get_ratings_to_dev(self, is_out=False):
        import torch

        return torch.Tensor(
            self.get_ratings(is_out).toarray()
        ).to(self._device)
    

class RecVAE(Recommender):
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


    def generate(self,my_batch_size, device, data_in, data_out=None, shuffle=False, samples_perc_per_epoch=1):
        assert 0 < samples_perc_per_epoch <= 1

        total_samples = data_in.shape[0]
        samples_per_epoch = int(total_samples * samples_perc_per_epoch)

        if shuffle:
            idxlist = np.arange(total_samples)
            np.random.shuffle(idxlist)
            idxlist = idxlist[:samples_per_epoch]
        else:
            idxlist = np.arange(samples_per_epoch)

        for st_idx in range(0, samples_per_epoch, my_batch_size):
            end_idx = min(st_idx + my_batch_size, samples_per_epoch)
            idx = idxlist[st_idx:end_idx]

            # idx_len = len(idx)
            # print(f"end_idx:{end_idx}")
            # if idx_len != 500:
            #   print(f"idx:{len(idx)}")
            #   break

            yield Batch(device, idx, data_in, data_out)



    def evaluate(self,model, data_in, data_out, metrics, samples_perc_per_epoch=1, my_batch_size=500):
        metrics = deepcopy(metrics)
        model.eval()

        for m in metrics:
            m['score'] = []

        for batch in self.generate(my_batch_size,
                            self.device,
                            data_in,
                            data_out,
                            samples_perc_per_epoch=samples_perc_per_epoch
                            ):
            
            print(batch)

            ratings_in = batch.get_ratings_to_dev()
            ratings_out = batch.get_ratings(is_out=True)



            ratings_pred = model(ratings_in, calculate_loss=False).cpu().detach().numpy()



            if not (data_in is data_out):
                # print("Pred : INF")
                ratings_pred[batch.get_ratings().nonzero()] = -np.inf

            for m in metrics:
                m['score'].append(m['metric'](ratings_pred, ratings_out, k=m['k']))

        for m in metrics:
            m['score'] = np.concatenate(m['score']).mean()

        return [x['score'] for x in metrics]




    def run(self,model, opts, train_data, my_batch_size, n_epochs, beta, gamma, dropout_rate):
        model.train()
        for epoch in range(n_epochs):
            for batch in self.generate(my_batch_size,self.device, train_data, shuffle=True):
                ratings = batch.get_ratings_to_dev()
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
                'train_data': train_set.matrix,
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
                print(f"Learning completed : [{value}]")

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
        # print(f"###### {user_idx}")
        import torch

        ratings_in = self.mydata.matrix[user_idx,:]

        #print(f"ratings_in : {ratings_in}")
        #print(f"type ratings_in : {type(ratings_in)}")

        ratings_pred = self.recvae_model(torch.Tensor(ratings_in.toarray()).to(self.device), calculate_loss=False).cpu().detach().numpy().flatten()
        #print(f"ratings_pred : {ratings_pred}")

        #print(f"ratings_pred_len : {len(ratings_pred)}")

        #print(f"flattern : {ratings_pred.flatten()}")
        #print(len(ratings_pred.flatten()))  # Add this line for inspection
        #print(f"ratings_pred_len2 : {len(ratings_pred.flatten())}")
        #print(f"item_idx : {item_idx}")
        #print(f"user_idx : {user_idx}")

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
        
            # return self.clean_pre[:,user_idx][item_idx]
#print("UU")
