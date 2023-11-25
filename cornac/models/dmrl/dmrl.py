import torch
import torch.nn as nn

from cornac.data.bert_text import BertTextModality

class DMRL(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, bert_text_dim):
        super(DMRL, self).__init__()
        self.text_module = torch.nn.Sequential(
                            torch.nn.Linear(bert_text_dim, 150),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(150, 75),
                            torch.nn.LeakyReLU())

        self.user_embedding = torch.nn.Embedding(num_users, embedding_dim)
        self.item_embedding = torch.nn.Embedding(num_items, embedding_dim)

        self.attention = torch.nn.Sequential(
                            torch.nn.Linear(75 + embedding_dim + embedding_dim, 3),
                            torch.nn.Tanh(),
                            torch.nn.Linear(3, 3, bias=False)
        )

        self.test_module =torch.nn.Sequential(torch.nn.Linear(75 + 100 + 100, 2), torch.nn.LeakyReLU())



    def forward(self, users, items, text):
        text_embedding = self.text_module(text)
        user_embedding = self.user_embedding(users)
        item_embedding = self.item_embedding(items)

        # attentionLayer:




        return item_embedding
        