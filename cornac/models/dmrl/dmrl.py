from typing import List
import torch
import torch.nn as nn
from cornac.models.dmrl.d_cor_calc import DistanceCorrelationCalculator
from cornac.data.bert_text import BertTextModality
from dataclasses import dataclass


@dataclass
class EmbeddingFactorLists:
    """
    A dataclass for holding the embedding factors for each modality.
    """
    user_embedding_factors: List[torch.Tensor]
    item_embedding_factors: List[torch.Tensor]
    text_embedding_factors: List[torch.Tensor]


class DMRLModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, bert_text_dim, num_neg):
        super(DMRLModel, self).__init__()
        self.num_factors = 4
        self.num_neg = num_neg
        self.embedding_dim = embedding_dim
        self.num_modalities = 2
        self.text_module = torch.nn.Sequential(
                            torch.nn.Linear(bert_text_dim, 150),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(150, embedding_dim),
                            torch.nn.LeakyReLU())

        self.user_embedding = torch.nn.Embedding(num_users, embedding_dim)
        self.item_embedding = torch.nn.Embedding(num_items, embedding_dim)

        self.factor_size = self.embedding_dim // self.num_factors
        last_factor_size = self.embedding_dim % self.num_factors

        # create the attention layers. One NN layer per factor, however if
        # embedding dim is not divisible into num_factors without rest the last
        # attention layer will have a different input size
        self.attention_layer = []
        for _ in range(self.num_factors - 1):
            self.attention_layer.append(self.make_attention_layer(self.factor_size))
        
        if last_factor_size > 0: # last factor size is < factor size
            self.attention_layer.append(self.make_attention_layer(last_factor_size))
        else: # last factor layer also has exactly factor size many input features
            self.attention_layer.append(self.make_attention_layer(self.factor_size))
        
        self.attention_layer = nn.ModuleList(self.attention_layer)

    def make_attention_layer(self, size: int) -> torch.nn.Sequential:
        """
        Creates an attention layer that takes in a tensor of size
        ((self.num_modalities+1) * size) and outputs a tensor of size
        self.num_modalities

        :param size: the factor size of each input vector that is concatenated
            before insertion here. There are (self.num_modalities+1) of the
            modality input vectors.
        :return: a torch.nn.Sequential object that is the attention layer
        """
        # add normalization layer
        return torch.nn.Sequential(
            torch.nn.Linear((self.num_modalities+1) * size, self.num_modalities),
            torch.nn.Tanh(),
            torch.nn.Linear(self.num_modalities, self.num_modalities, bias=False),
            torch.nn.Softmax()
        )

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

        # self.continue_neural_net()

        # attentionLayer: needs to be implemented per factor k
        ratings_sum_over_mods = torch.zeros((batch_size, 1 + self.num_neg))
        for i in range(self.num_factors):

            concatted_features = torch.concatenate([user_embedding_factors[i], item_embedding_factors[i], text_embedding_factors[i]], axis=2)
            attention = self.attention_layer[i](torch.nn.functional.normalize(concatted_features, dim=-1))

            r_ui = attention[:, :, 0] * torch.nn.Softplus()(torch.sum(user_embedding_factors[i] * item_embedding_factors[i], axis=-1))
            r_ut = attention[:, :, 1] * torch.nn.Softplus()(torch.sum(user_embedding_factors[i] * text_embedding_factors[i], axis=-1))
            # r_uv = torch.sum(user_embedding_inflated * visual_embedding, axis=-1)

            # sum up over modalities and factors
            ratings_sum_over_mods = ratings_sum_over_mods + (r_ui + r_ut)

        return embedding_factor_lists, ratings_sum_over_mods
    

class DMRLLoss(nn.Module):
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
