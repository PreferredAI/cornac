import torch
import torch.nn as nn
import dgl
import dgl.function as fn


def construct_graph(data_set):
    """
    Generates graph given a cornac data set

    Parameters
    ----------
    data_set : cornac.data.dataset.Dataset
        The data set as provided by cornac
    """
    user_indices, item_indices, _ = data_set.uir_tuple
    user_nodes, item_nodes = (
        torch.from_numpy(user_indices),
        torch.from_numpy(
            item_indices + data_set.total_users
        ),  # increment item node idx by num users
    )

    u = torch.cat([user_nodes, item_nodes], dim=0)
    v = torch.cat([item_nodes, user_nodes], dim=0)

    g = dgl.graph((u, v), num_nodes=(data_set.total_users + data_set.total_items))
    return g


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, graph, src_embedding, dst_embedding):
        with graph.local_scope():
            inner_product = torch.cat((src_embedding, dst_embedding), dim=0)

            out_degs = graph.out_degrees().to(src_embedding.device).float().clamp(min=1)
            norm_out_degs = torch.pow(out_degs, -0.5).view(-1, 1)  # D^-1/2

            inner_product = inner_product * norm_out_degs

            graph.ndata["h"] = inner_product
            graph.update_all(
                message_func=fn.copy_u("h", "m"), reduce_func=fn.sum("m", "h")
            )

            res = graph.ndata["h"]

            in_degs = graph.in_degrees().to(src_embedding.device).float().clamp(min=1)
            norm_in_degs = torch.pow(in_degs, -0.5).view(-1, 1)  # D^-1/2

            res = res * norm_in_degs
            return res


class Model(nn.Module):
    def __init__(self, user_size, item_size, hidden_size, num_layers=3, device=None):
        super(Model, self).__init__()
        self.user_size = user_size
        self.item_size = item_size
        self.hidden_size = hidden_size
        self.embedding_weights = self._init_weights()
        self.layers = nn.ModuleList([GCNLayer() for _ in range(num_layers)])
        self.device = device

    def forward(self, graph):
        user_embedding = self.embedding_weights["user_embedding"]
        item_embedding = self.embedding_weights["item_embedding"]

        for i, layer in enumerate(self.layers, start=1):
            if i == 1:
                embeddings = layer(graph, user_embedding, item_embedding)
            else:
                embeddings = layer(
                    graph, embeddings[: self.user_size], embeddings[self.user_size:]
                )

            user_embedding = user_embedding + embeddings[: self.user_size] * (
                1 / (i + 1)
            )
            item_embedding = item_embedding + embeddings[self.user_size:] * (
                1 / (i + 1)
            )

        return user_embedding, item_embedding

    def _init_weights(self):
        initializer = nn.init.xavier_uniform_

        weights_dict = nn.ParameterDict(
            {
                "user_embedding": nn.Parameter(
                    initializer(torch.empty(self.user_size, self.hidden_size))
                ),
                "item_embedding": nn.Parameter(
                    initializer(torch.empty(self.item_size, self.hidden_size))
                ),
            }
        )
        return weights_dict
