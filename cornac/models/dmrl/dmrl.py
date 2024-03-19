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
import torch
import torch.nn as nn
from cornac.models.dmrl.d_cor_calc import DistanceCorrelationCalculator
from dataclasses import dataclass

from cornac.utils.common import get_rng
from cornac.utils.init_utils import normal, xavier_normal, xavier_uniform


@dataclass
class EmbeddingFactorLists:
    """
    A dataclass for holding the embedding factors for each modality.
    """

    user_embedding_factors: List[torch.Tensor]
    item_embedding_factors: List[torch.Tensor]
    text_embedding_factors: List[torch.Tensor] = None
    image_embedding_factors: List[torch.Tensor] = None


class DMRLModel(nn.Module):
    """
    The actual Disentangled Multi-Modal Recommendation Model neural network.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        text_dim: int,
        image_dim: int,
        dropout: float,
        num_neg: int,
        num_factors: int,
        seed: int = 123,
    ):

        super(DMRLModel, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_factors = num_factors
        self.num_neg = num_neg
        self.embedding_dim = embedding_dim
        self.num_modalities = 1 + bool(text_dim) + bool(image_dim)
        self.dropout = dropout
        self.grad_norms = []
        self.param_norms = []
        self.ui_ratings = []
        self.ut_ratings = []
        self.ui_attention = []
        self.ut_attention = []

        rng = get_rng(123)

        if text_dim:
            self.text_module = torch.nn.Sequential(
                torch.nn.Dropout(p=self.dropout),
                torch.nn.Linear(text_dim, 150),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(p=self.dropout),
                torch.nn.Linear(150, embedding_dim),
                torch.nn.LeakyReLU(),
            )
            self.text_module[1].weight.data = torch.from_numpy(
                xavier_normal([150, text_dim], random_state=rng)
            )  # , std=0.02))
            self.text_module[4].weight.data

        if image_dim:
            self.image_module = torch.nn.Sequential(
                torch.nn.Dropout(p=self.dropout),
                torch.nn.Linear(image_dim, 150),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(p=self.dropout),
                torch.nn.Linear(150, embedding_dim),
                torch.nn.LeakyReLU(),
            )

        self.user_embedding = torch.nn.Embedding(num_users, embedding_dim)
        self.item_embedding = torch.nn.Embedding(num_items, embedding_dim)

        self.user_embedding.weight.data = torch.from_numpy(
            xavier_normal([num_users, embedding_dim], random_state=rng)
        )  # , std=0.02))
        self.item_embedding.weight.data = torch.from_numpy(
            xavier_normal([num_items, embedding_dim], random_state=rng)
        )  # , std=0.02))

        self.factor_size = self.embedding_dim // self.num_factors

        self.attention_layer = torch.nn.Sequential(
            torch.nn.Dropout(p=self.dropout),
            torch.nn.Linear(
                (self.num_modalities + 1) * self.factor_size, self.num_modalities
            ),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=self.dropout),
            torch.nn.Linear(self.num_modalities, self.num_modalities, bias=False),
            torch.nn.Softmax(dim=-1),
        )
        self.attention_layer[1].weight.data = torch.from_numpy(
            xavier_normal(
                [self.num_modalities, (self.num_modalities + 1) * self.factor_size],
                random_state=rng,
            )
        )  # , std=0.02))
        self.attention_layer[4].weight.data = torch.from_numpy(
            xavier_normal([self.num_modalities, self.num_modalities], random_state=rng)
        )  # , std=0.02))

        self.grad_dict = {i[0]: [] for i in self.named_parameters()}

    def forward(
        self, batch: torch.Tensor, text: torch.Tensor, image: torch.Tensor
    ) -> Tuple[EmbeddingFactorLists, torch.Tensor]:
        """
        Forward pass of the model.

        Parameters:
        -----------
        batch: torch.Tensor
            A batch of data. The first column contains the user indices, the
            rest of the columns contain the item indices (one pos and num_neg negatives)
        text: torch.Tensor
            The text data for the items in the batch (encoded)
        image: torch.Tensor
            The image data for the items in the batch (encoded)
        """
        text_embedding_factors = [
            torch.tensor([]).to(self.device) for _ in range(self.num_factors)
        ]
        image_embedding_factors = [
            torch.tensor([]).to(self.device) for _ in range(self.num_factors)
        ]
        users = batch[:, 0]
        items = batch[:, 1:]

        # handle text
        if text is not None:
            text_embedding = self.text_module(
                torch.nn.functional.normalize(text, dim=-1)
            )
            text_embedding_factors = torch.split(
                text_embedding, self.embedding_dim // self.num_factors, dim=-1
            )

        # handle image
        if image is not None:
            image_embedding = self.image_module(
                torch.nn.functional.normalize(image, dim=-1)
            )
            image_embedding_factors = torch.split(
                image_embedding, self.embedding_dim // self.num_factors, dim=-1
            )

        # handle users
        user_embedding = self.user_embedding(users)
        # we have to get users into shape batch, 1+num_neg, embedding_dim
        # therefore we repeat the users across the 1 pos and num_neg items
        user_embedding_inflated = user_embedding.unsqueeze(1).repeat(
            1, items.shape[1], 1
        )
        user_embedding_factors = torch.split(
            user_embedding_inflated, self.embedding_dim // self.num_factors, dim=-1
        )

        # handle items
        item_embedding = self.item_embedding(items)
        item_embedding_factors = torch.split(
            item_embedding, self.embedding_dim // self.num_factors, dim=-1
        )

        embedding_factor_lists = EmbeddingFactorLists(
            user_embedding_factors,
            item_embedding_factors,
            text_embedding_factors,
            image_embedding_factors,
        )

        # attentionLayer: implemented per factor k
        batch_size = users.shape[0]
        ratings_sum_over_mods = torch.zeros((batch_size, 1 + self.num_neg)).to(
            self.device
        )
        for i in range(self.num_factors):

            concatted_features = torch.concatenate(
                [
                    user_embedding_factors[i],
                    item_embedding_factors[i],
                    text_embedding_factors[i],
                    image_embedding_factors[i],
                ],
                axis=2,
            )
            attention = self.attention_layer(
                torch.nn.functional.normalize(concatted_features, dim=-1)
            )

            r_ui = attention[:, :, 0] * torch.nn.Softplus()(
                torch.sum(
                    user_embedding_factors[i] * item_embedding_factors[i], axis=-1
                )
            )
            # log rating
            self.ui_ratings.append(torch.norm(r_ui.detach().flatten()).cpu())

            factor_rating = r_ui

            if text is not None:
                r_ut = attention[:, :, 1] * torch.nn.Softplus()(
                    torch.sum(
                        user_embedding_factors[i] * text_embedding_factors[i], axis=-1
                    )
                )
                factor_rating = factor_rating + r_ut
                # log rating
                self.ut_ratings.append(torch.norm(r_ut.detach().flatten()).cpu())

            if image is not None:
                r_ui = attention[:, :, 1] * torch.nn.Softplus()(
                    torch.sum(
                        user_embedding_factors[i] * image_embedding_factors[i], axis=-1
                    )
                )
                factor_rating = factor_rating + r_ui
                self.ui_ratings.append(torch.norm(r_ui.detach().flatten()).cpu())

            # sum up over modalities and running sum over factors
            ratings_sum_over_mods = ratings_sum_over_mods + factor_rating

        return embedding_factor_lists, ratings_sum_over_mods

    def log_gradients_and_weights(self):
        """
        Stores most recent gradient norms in a list.
        """

        for i in self.named_parameters():
            self.grad_dict[i[0]].append(torch.norm(i[1].grad.detach().flatten()).item())

        total_norm_grad = torch.norm(
            torch.cat([p.grad.detach().flatten() for p in self.parameters()])
        )
        self.grad_norms.append(total_norm_grad.item())

        total_norm_param = torch.norm(
            torch.cat([p.detach().flatten() for p in self.parameters()])
        )
        self.param_norms.append(total_norm_param.item())

    def reset_grad_metrics(self):
        """
        Reset the gradient metrics.
        """
        self.grad_norms = []
        self.param_norms = []
        self.grad_dict = {i[0]: [] for i in self.named_parameters()}
        self.ui_ratings = []
        self.ut_ratings = []
        self.ut_attention = []
        self.ut_attention = []


class DMRLLoss(nn.Module):
    """
    The disentangled multi-modal recommendation model loss function. It's a
    combination of pairwise based ranking loss and disentangled loss. For
    details see DMRL paper.
    """

    def __init__(self, decay_c, num_factors, num_neg):
        super(DMRLLoss, self).__init__()
        self.decay_c = decay_c
        self.distance_cor_calc = DistanceCorrelationCalculator(
            n_factors=num_factors, num_neg=num_neg
        )

    def forward(
        self, embedding_factor_lists: EmbeddingFactorLists, rating_scores: torch.tensor
    ) -> torch.tensor:
        """
        Calculates the loss for the batch of data.
        """
        r_pos = rating_scores[:, 0]
        # from the num_neg many negative sampled items, we want to find the one
        # with the largest score to have one negative sample per user in our
        # batch
        r_neg = torch.max(rating_scores[:, 1:], dim=1).values

        # define the ranking loss for pairwise-based ranking approach
        loss_BPR = torch.sum(torch.nn.Softplus()(-(r_pos - r_neg)))

        # regularizer loss is added as weight decay in optimization function
        if self.decay_c > 0:
            disentangled_loss = self.distance_cor_calc.calculate_disentangled_loss(
                embedding_factor_lists.user_embedding_factors,
                embedding_factor_lists.item_embedding_factors,
                embedding_factor_lists.text_embedding_factors,
                embedding_factor_lists.image_embedding_factors,
            )

            return loss_BPR + self.decay_c * disentangled_loss
        return loss_BPR
