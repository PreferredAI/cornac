import torch
import torch.nn as nn

from cornac.data.bert_text import BertTextModality

class DMRL(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, bert_text_dim):
        super(DMRL, self).__init__()
        self.num_factors = 4
        self.embedding_dim = embedding_dim
        self.num_modalities = 2
        self.text_module = torch.nn.Sequential(
                            torch.nn.Linear(bert_text_dim, 150),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(150, embedding_dim),
                            torch.nn.LeakyReLU())

        self.user_embedding = torch.nn.Embedding(num_users, embedding_dim)
        self.item_embedding = torch.nn.Embedding(num_items, embedding_dim)

        self.attention = torch.nn.Sequential(
                            torch.nn.Linear(3*embedding_dim, self.num_modalities),
                            torch.nn.Tanh(),
                            torch.nn.Linear(self.num_modalities, self.num_modalities, bias=False),
                            torch.nn.Softmax()
        )


    def forward(self, batch, text):
        text_embedding = self.text_module(text)
        batch_size = users.shape[0]
        users = batch[:, 0]
        items = batch[:, 1:]
        user_embedding: torch.tensor
        user_embedding = self.user_embedding(users)
        item_embedding = self.item_embedding(items)


        # we have to get users into shape batch, 1+num_neg, embedding_dim
        # therefore we repeat the users across the 1 pos and num_neg items
        user_embedding_inflated = user_embedding.unsqueeze(1).repeat(1, items.shape[1], 1)

        # now split into the k factors
        user_embedding_factors = torch.split(user_embedding_inflated, self.embedding_dim // self.num_factors, dim=-1)
        item_embedding_factors = torch.split(item_embedding, self.embedding_dim // self.num_factors, dim=-1)
        text_embedding_factors = torch.split(text_embedding, self.embedding_dim // self.num_factors, dim=-1)

        disentangled_loss = self.calculate_disentangled_loss(user_embedding_factors, item_embedding_factors, text_embedding_factors)

        self.continue_neural_net()

        # attentionLayer:
        attention = self.attention(torch.concatenate([user_embedding_inflated, item_embedding, text_embedding], axis=2))

        r_ui = attention[:, :, 0] * torch.nn.Softplus()(torch.sum(user_embedding_inflated * item_embedding, axis=-1))
        r_ut = attention[:, :, 1] * torch.nn.Softplus()(torch.sum(user_embedding_inflated * text_embedding, axis=-1))
        # r_uv = torch.sum(user_embedding_inflated * visual_embedding, axis=-1)

        sum_over_mods = r_ui + r_ut
        r_pos = sum_over_mods[:, 0]
        # take the maximum of the negative samples
        r_neg = torch.max(sum_over_mods[:, 1:], dim=1).values

        # define the ranking loss for pairwise-based ranking approach
        loss_BPR = torch.sum(torch.nn.Softplus()(-(r_pos - r_neg)))

        # regularizer loss is added as weight decay in optimization function
        disentangled_loss




        return item_embedding
        

    def calculate_disentangled_loss(self, user_embedding_factors, item_embedding_factors, text_embedding_factors):
        """
        """
        for factor in range(self.num_factors - 1):
            for factor_plus_one in range(factor + 1, self.num_factors):
                
