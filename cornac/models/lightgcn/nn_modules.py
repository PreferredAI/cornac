import torch.nn as nn
import torch
import dgl.function as fn


class NeuralNetwork(nn.Module):
    def __init__(self, user_size, item_size, hidden_size, num_layers=1, device=None):
        super(NeuralNetwork, self).__init__()
        self.user_size = user_size
        self.item_size = item_size
        self.hidden_size = hidden_size
        self.embedding_weights = self._init_weights()
        self.layers = nn.ModuleList(
            [GCNLayer() for _ in range(num_layers)]
        )
        self.device = device

    def forward(self, graph):
        user_embedding = self.embedding_weights["user_embedding"]
        item_embedding = self.embedding_weights["item_embedding"]

        for i, layer in enumerate(self.layers, start=1):
            if i == 1:
                embeddings = layer(graph, user_embedding, item_embedding)
            else:
                embeddings = layer(graph, embeddings[:self.user_size], embeddings[self.user_size:])
            
            user_embedding = user_embedding + embeddings[: self.user_size] * (1 / (i + 1))
            item_embedding = item_embedding + embeddings[self.user_size:] * (1 / (i + 1))
        
        return user_embedding, item_embedding

    def _init_weights(self):
        initializer = nn.init.xavier_uniform_

        weights_dict = nn.ParameterDict({
            "user_embedding": nn.Parameter(
                initializer(torch.empty(self.user_size, self.hidden_size))
            ),
            "item_embedding": nn.Parameter(
                initializer(torch.empty(self.item_size, self.hidden_size))
            )
        })
        return weights_dict


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, graph, src_embedding, dst_embedding):
        with graph.local_scope():
            inner_product = torch.cat((src_embedding, dst_embedding), dim=0)
            
            # ui_out_degs = graph.out_degrees(etype="user-item")
            # iu_out_degs = graph.out_degrees(etype="item-user")
            # out_degs = torch.cat((ui_out_degs, iu_out_degs), dim=0).to(src_embedding.device).float().clamp(min=1)
            out_degs = graph.out_degrees().to(src_embedding.device).float().clamp(min=1)
            norm_out_degs = torch.pow(out_degs, -0.5).view(-1, 1)  # D^-1/2

            inner_product = inner_product * norm_out_degs

            # graph.ndata["h"] = {"user": inner_product, "item": inner_product}
            graph.ndata["h"] = inner_product
            graph.update_all(
                message_func=fn.copy_u("h", "m"),
                reduce_func=fn.sum("m", "h")
            )

            res = graph.ndata["h"]

            in_degs = graph.in_degrees().to(src_embedding.device).float().clamp(min=1)
            norm_in_degs = torch.pow(in_degs, -0.5).view(-1, 1)  # D^-1/2

            res = res * norm_in_degs
            return res

