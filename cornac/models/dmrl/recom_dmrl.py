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

from typing import List, Tuple

import numpy as np

from cornac.data.dataset import Dataset
from cornac.data import FeatureModality, TextModality, ImageModality
from cornac.metrics.ranking import Precision, Recall
from cornac.models.recommender import Recommender


class DMRL(Recommender):
    """
    Disentangled multimodal representation learning

    Parameters
    ----------
    name: string, default: 'DMRL'
        The name of the recommender model.

    batch_size: int, optional, default: 32
        The number of samples per batch to load.

    learning_rate: float, optional, default: 1e-4
        The learning rate for the optimizer.

    decay_c: float, optional, default: 1
        The decay for the disentangled loss term in the loss function.

    decay_r: float, optional, default: 0.01
        The decay for the regularization term in the loss function.

    epochs: int, optional, default: 10
        The number of epochs to train the model.

    embedding_dim: int, optional, default: 100
        The dimension of the embeddings.

    bert_text_dim: int, optional, default: 384
        The dimension of the bert text embeddings coming from the huggingface transformer model

    image_dim: int, optional, default: None
        The dimension of the image embeddings.

    num_neg: int, optional, default: 4
        The number of negative samples to use in the training per user per batch (1 positive and num_neg negatives are used)

    num_factors: int, optional, default: 4
        The number of factors to use in the model.

    trainable: bool, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already trained.

    verbose: bool, optional, default: False
        When True, the model prints out more information during training.

    modalities_pre_built: bool, optional, default: True
        When True, the model assumes that the modalities are already built and does not build them.

    log_metrics: bool, optional, default: False
        When True, the model logs metrics to tensorboard.

    References
    ----------
    * Fan Liu, Huilin Chen,  Zhiyong Cheng, Anan Liu, Liqiang Nie, Mohan Kankanhalli. DMRL: Disentangled Multimodal Representation Learning for
        Recommendation. https://arxiv.org/pdf/2203.05406.pdf.
    """

    def __init__(
        self,
        name: str = "DMRL",
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        decay_c: float = 1,
        decay_r: float = 0.01,
        epochs: int = 10,
        embedding_dim: int = 100,
        bert_text_dim: int = 384,
        image_dim: int = None,
        dropout: float = 0,
        num_neg: int = 4,
        num_factors: int = 4,
        trainable: bool = True,
        verbose: bool = False,
        log_metrics: bool = False,
    ):

        super().__init__(name=name, trainable=trainable, verbose=verbose)

        self.learning_rate = learning_rate
        self.decay_c = decay_c
        self.decay_r = decay_r
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.embedding_dim = embedding_dim
        self.text_dim = bert_text_dim
        self.image_dim = image_dim
        self.dropout = dropout
        self.num_neg = num_neg
        self.num_factors = num_factors
        self.log_metrics = log_metrics
        if log_metrics:
            from torch.utils.tensorboard import SummaryWriter

            self.tb_writer = SummaryWriter("temp/tb_data/run_1")

        if self.num_factors == 1:
            # deactivate disentangled portion of loss if theres only 1 factor
            self.decay_c == 0

    def fit(self, train_set: Dataset, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).
        """
        Recommender.fit(self, train_set, val_set)

        if self.trainable:
            self._fit_dmrl(train_set, val_set)

        return self

    def get_item_image_embedding(self, batch):
        """
        Get the item image embeddings from the image modality. Expect the image
        modaility to be preencded and available as a numpy array.

        Parameters
        ----------

        param batch: torch.Tensor, user inidices in first column, pos item indices in second
            and all other columns are negative item indices
        """
        import torch

        if not hasattr(self, "item_image"):
            return None

        shape = batch[:, 1:].shape
        all_items = batch[:, 1:].flatten()

        item_image_embedding = self.item_image.features[all_items, :].reshape(
            (*shape, self.item_image.feature_dim)
        )

        if not isinstance(item_image_embedding, torch.Tensor):
            item_image_embedding = torch.tensor(
                item_image_embedding, dtype=torch.float32
            )

        return item_image_embedding

    def get_item_text_embeddings(self, batch):
        """
        Get the item text embeddings from the BERT model. Either by encoding the
        text on the fly or by using the preencoded text.

        Parameters
        ----------

        param batch: torch.Tensor, user inidices in first column, pos item indices in second
            and all other columns are negative item indices
        """
        import torch

        shape = batch[:, 1:].shape
        all_items = batch[:, 1:].flatten()

        if not hasattr(self, "item_text"):
            return None

        if not self.item_text.preencoded:
            item_text_embeddings = self.item_text.batch_encode(all_items)
            item_text_embeddings = item_text_embeddings.reshape(
                (*shape, self.item_text.output_dim)
            )
        else:
            item_text_embeddings = self.item_text.features[all_items]
            item_text_embeddings = item_text_embeddings.reshape(
                (*shape, self.item_text.output_dim)
            )

        if not isinstance(item_text_embeddings, torch.Tensor):
            item_text_embeddings = torch.tensor(
                item_text_embeddings, dtype=torch.float32
            )

        return item_text_embeddings

    def get_modality_embeddings(self, batch):
        """
        Get the modality embeddings for both text and image from the respectiv
        modality instances.

        Parameters
        ----------

        param batch: torch.Tensor, user inidices in first column, pos item
        indices in second
            and all other columns are negative item indices
        """
        item_text_embeddings = self.get_item_text_embeddings(batch)
        item_image_embeddings = self.get_item_image_embedding(batch)

        return item_text_embeddings, item_image_embeddings

    def _fit_dmrl(self, train_set: Dataset, val_set: Dataset = None):
        """
        Fit the model to observations.

        Parameters
        ----------
        train_set: User-Item preference data as well as additional modalities.
        """
        import torch
        from torch.utils.data import DataLoader

        from cornac.models.dmrl.dmrl import DMRLLoss, DMRLModel
        from cornac.models.dmrl.pwlearning_sampler import PWLearningSampler

        self.initialize_and_build_modalities(train_set)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device {self.device} for training")

        self.sampler = PWLearningSampler(train_set, num_neg=self.num_neg)

        self.model = DMRLModel(
            self.num_users,
            self.num_items,
            self.embedding_dim,
            self.text_dim,
            self.image_dim,
            self.dropout,
            self.num_neg,
            self.num_factors,
        ).to(self.device)

        loss_function = DMRLLoss(
            decay_c=1e-3, num_factors=self.num_factors, num_neg=self.num_neg
        )

        # add hyperparams to tensorboard
        if self.log_metrics:
            self.tb_writer.add_hparams(
                {
                    "learning_rate": self.learning_rate,
                    "decay_c": self.decay_c,
                    "decay_r": self.decay_r,
                    "batch_size": self.batch_size,
                    "epochs": self.epochs,
                    "embedding_dim": self.embedding_dim,
                    "bert_text_dim": self.text_dim,
                    "num_neg": self.num_neg,
                    "num_factors": self.num_factors,
                    "dropout": self.dropout,
                },
                {},
            )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.decay_r,
            betas=(0.9, 0.999),
        )
        # optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.decay_r)

        # Create learning rate scheduler if needed
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.25, step_size=35)

        dataloader = DataLoader(
            self.sampler,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=True,
            prefetch_factor=None,
        )

        if val_set is not None:
            self.val_sampler = PWLearningSampler(val_set, num_neg=self.num_neg)
            val_dataloader = DataLoader(
                self.val_sampler,
                batch_size=self.batch_size,
                num_workers=0,
                shuffle=True,
                prefetch_factor=None,
            )

        j = 1
        stop = False
        # Training loop
        for epoch in range(self.epochs):
            if stop:
                break
            running_loss = 0
            running_loss_val = 0
            last_loss = 0
            i = 0

            batch: torch.Tensor
            for i, batch in enumerate(dataloader):

                optimizer.zero_grad()
                item_text_embeddings, item_image_embeddings = (
                    self.get_modality_embeddings(batch)
                )

                # move the data to the device
                batch = batch.to(self.device)
                if item_text_embeddings is not None:
                    item_text_embeddings = item_text_embeddings.to(self.device)
                if item_image_embeddings is not None:
                    item_image_embeddings = item_image_embeddings.to(self.device)

                # Forward pass
                embedding_factor_lists, rating_scores = self.model(
                    batch, item_text_embeddings, item_image_embeddings
                )
                # preds = self.model(u_batch, i_batch, text)
                loss = loss_function(embedding_factor_lists, rating_scores)

                # Backward pass and optimize
                loss.backward()
                # torch.nn.utils.clip_grad_value_(self.model.parameters(), 5) # use if exploding gradient becomes an issue
                if self.log_metrics:
                    self.model.log_gradients_and_weights()

                optimizer.step()

                if val_set is not None:
                    val_batch = next(val_dataloader.__iter__())
                    item_text_embeddings_val, item_image_embeddings_val = (
                        self.get_modality_embeddings(val_batch)
                    )

                    # Forward pass
                    with torch.no_grad():
                        embedding_factor_lists_val, rating_scores_val = self.model(
                            val_batch,
                            item_text_embeddings_val,
                            item_image_embeddings_val,
                        )
                        # preds = self.model(u_batch, i_batch, text)
                        loss_val = loss_function(
                            embedding_factor_lists_val, rating_scores_val
                        )
                        running_loss_val += loss_val.item()

                # Gather data and report
                running_loss += loss.item()
                devider = 5
                if i % devider == 4:
                    last_loss = running_loss / devider  # loss per batch
                    # last_loss = running_loss / (i + 1)
                    print("  batch {} loss: {}".format(i + 1, last_loss))

                    if self.log_metrics:
                        # tb_x = epoch * len(dataloader) + i + 1
                        self.tb_writer.add_scalar("Loss/train", last_loss, j)
                        self.tb_writer.add_scalar(
                            "Loss/val", running_loss_val / devider, j
                        )
                        self.tb_writer.add_scalar(
                            "Gradient Norm/train", np.mean(self.model.grad_norms), j
                        )
                        self.tb_writer.add_scalar(
                            "Param Norm/train", np.mean(self.model.param_norms), j
                        )
                        self.tb_writer.add_scalar(
                            "User-Item based rating", np.mean(self.model.ui_ratings), j
                        )
                        self.tb_writer.add_scalar(
                            "User-Text based rating", np.mean(self.model.ut_ratings), j
                        )
                        self.tb_writer.add_scalar(
                            "User-Itm Attention", np.mean(self.model.ui_attention), j
                        )
                        self.tb_writer.add_scalar(
                            "User-Text Attention", np.mean(self.model.ut_attention), j
                        )
                        for name, param in self.model.named_parameters():
                            self.tb_writer.add_scalar(
                                name + "/grad_norm",
                                np.mean(self.model.grad_dict[name]),
                                j,
                            )
                            self.tb_writer.add_histogram(
                                name + "/grad", param.grad, global_step=epoch
                            )
                        self.tb_writer.add_scalar(
                            "Learning rate", optimizer.param_groups[0]["lr"], j
                        )
                        self.model.reset_grad_metrics()
                    running_loss = 0
                    running_loss_val = 0

                # if i % 999== 0:
                # scheduler.step()

                i += 1
                j += 1

            print(f"Epoch: {epoch} is done")
            # scheduler.step()
        print("Finished training!")
        # self.eval_train_set_performance() # evaluate the model on the training set after training if necessary

    def eval_train_set_performance(self) -> Tuple[float, float]:
        """
        Evaluate the models training set performance using Recall 300 metric.
        """
        from cornac.eval_methods.base_method import ranking_eval

        print("Evaluating training set performance at k=300")
        avg_results, _ = ranking_eval(
            self,
            [Recall(k=300), Precision(k=300)],
            self.train_set,
            self.train_set,
            verbose=True,
            rating_threshold=4,
        )
        print(f"Mean train set recall and precision: {avg_results}")
        return avg_results

    def score(self, user_index: int, item_indices = None):
        """
        Scores a user-item pair. If item_index is None, scores for all known
        items.

        Parameters
        ----------
        name: user_idx
            The index of the user for whom to perform score prediction.

        item_indices: torch.Tensor, optional, default: None
            The index of the item for which to perform score prediction.
            If None, scores for all known items will be returned.
        """
        import torch

        self.model.num_neg = 0
        self.model.eval()

        encoded_image = None
        encoded_text = None

        if item_indices is None:
            item_indices = torch.tensor(list(self.iid_map.values()), dtype=torch.long)

        user_index = user_index * torch.ones(len(item_indices), dtype=torch.long)

        if self.item_text.features is None:
            self.item_text.preencode_entire_corpus()

        # since the model expects as (batch size, 1 + num_neg, encoding dim) we just add one dim and repeat
        if hasattr(self, "item_text"):
            encoded_text: torch.Tensor = self.item_text.features[
                item_indices, :
            ]
            encoded_text = encoded_text[:, None, :]
            encoded_text = encoded_text.to(self.device)

        if hasattr(self, "item_image"):
            encoded_image = torch.tensor(
                self.item_image.features[item_indices, :], dtype=torch.float32
            )
            encoded_image = encoded_image[:, None, :]
            encoded_image = encoded_image.to(self.device)

        input_tensor = torch.stack((user_index, item_indices), axis=1)
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            _, ratings_sum_over_mods = self.model(
                input_tensor, encoded_text, encoded_image
            )

        return np.array(ratings_sum_over_mods[:, 0].detach().cpu())

    def initialize_and_build_modalities(self, trainset: Dataset):
        """
        Initializes text and image modalities for the model. Either takes in raw
        text or image and performs pre-encoding given the transformer models in
        TransformerTextModality and TransformerVisionModality. If preencoded
        features are given, it uses those instead and simply wrapes them into a
        general FeatureModality instance, as no further encoding model is
        required.
        """
        from cornac.models.dmrl.transformer_text import TransformersTextModality
        from cornac.models.dmrl.transformer_vision import TransformersVisionModality

        if trainset.item_text is not None:
            if (
                isinstance(trainset.item_text, TextModality)
                and trainset.item_text.corpus is not None
            ):
                self.item_text = TransformersTextModality(
                    corpus=trainset.item_text.corpus,
                    ids=trainset.item_text.ids,
                    preencode=True,
                )
            elif isinstance(
                trainset.item_text, FeatureModality
            ):  # already have preencoded text features from outside
                self.item_text = trainset.item_text
                assert trainset.item_text.features is not None, "No pre-encoded features found, please use TextModality"
            else:
                raise ValueError("Not supported type of modality for item text")

        if trainset.item_image is not None:
            if (
                isinstance(trainset.item_image, ImageModality)
                and trainset.item_image.images is not None
            ):
                self.item_image = TransformersVisionModality(
                    images=trainset.item_image.images,
                    ids=trainset.item_image.ids,
                    preencode=True,
                )
            elif isinstance(
                trainset.item_image, FeatureModality
            ):  # already have preencoded image features from outside
                self.item_image = trainset.item_image
                assert trainset.item_image.features is not None, "No pre-encoded features found, please use ImageModality"
            else:
                raise ValueError("Not supported type of modality for item image")
