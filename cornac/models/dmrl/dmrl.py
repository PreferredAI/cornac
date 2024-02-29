from typing import List
import numpy as np
import torch
import torch.nn as nn
from cornac.models.dmrl.d_cor_calc import DistanceCorrelationCalculator
from cornac.data.bert_text import BertTextModality
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
    text_embedding_factors: List[torch.Tensor]


class DMRLModel(nn.Module):
    """
    The actual Disentangled Multi-Modal Recommendation Model neural network.
    """
    def __init__(self, num_users, num_items, embedding_dim, bert_text_dim, num_neg, num_factors, seed=123):
        super(DMRLModel, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_factors = num_factors
        self.num_neg = num_neg
        self.embedding_dim = embedding_dim
        self.num_modalities = 2
        self.grad_norms = []
        self.param_norms = []
        self.ui_ratings = []
        self.ut_ratings = []
        self.ui_attention = []
        self.ut_attention = []
        self.text_module = torch.nn.Sequential(
                            torch.nn.Linear(bert_text_dim, 150),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(150, embedding_dim),
                            torch.nn.LeakyReLU())
        
        rng = get_rng(123)

        self.user_embedding = torch.nn.Embedding(num_users, embedding_dim)
        self.item_embedding = torch.nn.Embedding(num_items, embedding_dim)

        self.user_embedding.weight.data = torch.from_numpy(normal([num_users, embedding_dim], random_state=rng, std=0.02))
        self.item_embedding.weight.data = torch.from_numpy(normal([num_items, embedding_dim], random_state=rng, std=0.02))
        self.text_module[0].weight.data = torch.from_numpy(normal([150, bert_text_dim], random_state=rng, std=0.02))
        self.text_module[2].weight.data = torch.from_numpy(normal([embedding_dim, 150], random_state=rng, std=0.02))

        self.factor_size = self.embedding_dim // self.num_factors
        # last_factor_size = self.embedding_dim % self.num_factors

        self.attention_layer = torch.nn.Sequential(
                                                    torch.nn.Linear((self.num_modalities+1) * self.factor_size, self.num_modalities),
                                                    # nn.BatchNorm1d(self.num_neg+1), 
                                                    torch.nn.Tanh(),
                                                    torch.nn.Linear(self.num_modalities, self.num_modalities, bias=False),
                                                    # nn.BatchNorm1d(self.num_neg+1), 
                                                    torch.nn.Softmax(dim=-1)
                                                    )
        self.attention_layer[0].weight.data = torch.from_numpy(normal([self.num_modalities, (self.num_modalities+1) * self.factor_size], random_state=rng, std=0.02))
        # self.attention_layer[0].weight.data = torch.from_numpy(normal([self.num_modalities, (self.num_modalities+1) * self.factor_size], 10, 4, rng, np.float32))
        # self.attention_layer[0].bias.data = torch.zeros(self.num_modalities)
        self.attention_layer[2].weight.data = torch.from_numpy(normal([self.num_modalities, self.num_modalities], random_state=rng, std=0.02))
        # self.attention_layer[2].weight.data = torch.from_numpy(normal([self.num_modalities, self.num_modalities], 10, 4, rng, np.float32))


        # create the attention layers. One NN layer per factor, however if
        # embedding dim is not divisible into num_factors without rest the last
        # attention layer will have a different input size
        # self.attention_layer = []
        # for _ in range(self.num_factors - 1):
        #     self.attention_layer.append(self.make_attention_layer(self.factor_size))

        # if last_factor_size > 0: # last factor size is < factor size
        #     self.attention_layer.append(self.make_attention_layer(last_factor_size))
        # else: # last factor layer also has exactly factor size many input features
        #     self.attention_layer.append(self.make_attention_layer(self.factor_size))
        
        # self.attention_layer = nn.ModuleList(self.attention_layer)
        self.grad_dict = {i[0]: [] for i in self.named_parameters()}

    
    # def forward(self, u_indices, i_indices, text):
    #     """
    #     """
    #     user_embedding = self.user_embedding(u_indices)
    #     item_embedding = self.item_embedding(i_indices)

    #     text_embedding = self.text_module(text)

    #     # attention = self.attention_layer(torch.concatenate([user_embedding, item_embedding, text_embedding], axis=1))

    #     preds_ui = torch.matmul(user_embedding, item_embedding.T)
    #     preds_ut = torch.matmul(user_embedding, text_embedding.T)

    #     self.ui_ratings.append(torch.norm(preds_ui.detach().flatten()))
    #     self.ut_ratings.append(torch.norm(preds_ut.detach().flatten()))

    #     return preds_ui + preds_ut


    def forward(self, batch, text):
        text_embedding = self.text_module(text)
        users = batch[:, 0]
        items = batch[:, 1:]
        user_embedding: torch.tensor
        user_embedding = self.user_embedding(users)
        item_embedding = self.item_embedding(items)
        batch_size = users.shape[0]

        # we have to get users into shape batch, 1+num_neg, embedding_dim
        # therefore we repeat the users across the 1 pos and num_neg items
        user_embedding_inflated = user_embedding.unsqueeze(1).repeat(1, items.shape[1], 1)

        # now split into the k factors
        user_embedding_factors = torch.split(user_embedding_inflated, self.embedding_dim // self.num_factors, dim=-1)
        item_embedding_factors = torch.split(item_embedding, self.embedding_dim // self.num_factors, dim=-1)
        text_embedding_factors = torch.split(text_embedding, self.embedding_dim // self.num_factors, dim=-1)

        embedding_factor_lists = EmbeddingFactorLists(user_embedding_factors, item_embedding_factors, text_embedding_factors)

        # attentionLayer: implemented per factor k
        ratings_sum_over_mods = torch.zeros((batch_size, 1 + self.num_neg)).to(self.device)
        for i in range(self.num_factors):

            concatted_features = torch.concatenate([user_embedding_factors[i], item_embedding_factors[i], text_embedding_factors[i]], axis=2)
            attention = self.attention_layer(torch.nn.functional.normalize(concatted_features, dim=-1))

            r_ui = attention[:, :, 0] * torch.nn.Softplus()(torch.sum(user_embedding_factors[i] * item_embedding_factors[i], axis=-1))
            r_ut = attention[:, :, 1] * torch.nn.Softplus()(torch.sum(user_embedding_factors[i] * text_embedding_factors[i], axis=-1))
            # r_ui = torch.nn.Softplus()(torch.sum(user_embedding_factors[i] * item_embedding_factors[i], axis=-1))
            # r_ut = torch.nn.Softplus()(torch.sum(user_embedding_factors[i] * text_embedding_factors[i], axis=-1))
            # r_uv = torch.sum(user_embedding_inflated * visual_embedding, axis=-1)

            # sum up over modalities and factorss
            self.ui_attention.append(torch.norm(attention[:, :, 0].detach().flatten()).cpu())
            self.ut_attention.append(torch.norm(attention[:, :, 1].detach().flatten()).cpu())
            self.ui_ratings.append(torch.norm(r_ui.detach().flatten()).cpu())
            self.ut_ratings.append(torch.norm(r_ut.detach().flatten()).cpu())
            ratings_sum_over_mods = ratings_sum_over_mods + (r_ui + r_ut)

        return embedding_factor_lists, ratings_sum_over_mods


    def log_gradients_and_weights(self):
        """
        Stores most recent gradient norms in a list.
        """

        for i in self.named_parameters():
            self.grad_dict[i[0]].append(torch.norm(i[1].grad.detach().flatten()).item())

        total_norm_grad = torch.norm(torch.cat([p.grad.detach().flatten() for p in self.parameters()]))
        self.grad_norms.append(total_norm_grad.item())

        total_norm_param = torch.norm(torch.cat([p.detach().flatten() for p in self.parameters()]))
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
        self.distance_cor_calc = DistanceCorrelationCalculator(n_factors=num_factors, num_neg=num_neg)

    def forward(self, embedding_factor_lists: EmbeddingFactorLists, rating_scores: torch.tensor) -> torch.tensor:
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
        disentangled_loss = self.distance_cor_calc.calculate_disentangled_loss(embedding_factor_lists.user_embedding_factors,
                                                                               embedding_factor_lists.item_embedding_factors,
                                                                               embedding_factor_lists.text_embedding_factors)

        total_loss = loss_BPR + self.decay_c * disentangled_loss
        return total_loss
