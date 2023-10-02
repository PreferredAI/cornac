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
    def __init__(self, in_size, out_size, dropout):
        super(GCNLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        
        self.w1 = nn.Linear(in_size, out_size, bias=True)
        self.w2 = nn.Linear(in_size, out_size, bias=True)

        self.leaky_relu = nn.LeakyReLU(0.2)

        self.dropout = nn.Dropout(dropout)

        torch.nn.init.xavier_uniform_(self.w1.weight)
        torch.nn.init.constant_(self.w1.bias, 0)

        torch.nn.init.xavier_uniform_(self.w2.weight)
        torch.nn.init.constant_(self.w2.bias, 0)

    def forward(self, graph, norm, src_embedding, dst_embedding):
        with graph.local_scope():
            inner_product = torch.cat((src_embedding, dst_embedding), dim=0)

            in_degs, out_degs = graph.edges()
            # out_degs = graph.out_degrees().to(src_embedding.device).float().clamp(min=1)

            msgs = self.w1(inner_product[in_degs]) + self.w2(inner_product[in_degs] * inner_product[out_degs])
            # norm_msgs = torch.pow(msgs, -0.5).view(-1, 1)  # D^-1/2
            msgs = norm * msgs

            # inner_product = inner_product * norm_msgs

            graph.edata["h"] = msgs
            # graph.ndata["h"] = inner_product
            graph.update_all(
                message_func=fn.copy_e("h", "m"), reduce_func=fn.sum("m", "h")
            )

            res = graph.ndata["h"]
            res = self.leaky_relu(res)
            res = self.dropout(res)

            # in_degs = graph.in_degrees().to(src_embedding.device).float().clamp(min=1)
            # norm_in_degs = torch.pow(in_degs, -0.5).view(-1, 1)  # D^-1/2

            # res = res * norm_in_degs
            return res


class Model(nn.Module):
    def __init__(self, user_size, item_size, embed_size=64, layer_size=[64, 64, 64], dropout=[0.1, 0.1, 0.1], device=None):
        super(Model, self).__init__()
        self.user_size = user_size
        self.item_size = item_size
        self.embedding_weights = self._init_weights(embed_size)
        self.layers = nn.ModuleList([GCNLayer(embed_size, layer_size[0], dropout[0])])
        self.layers.extend(
            [
                GCNLayer(
                    layer_size[i], layer_size[i + 1], dropout=dropout[i + 1]
                ) for i in range(len(layer_size) - 1)
            ]
        )
        self.device = device

    def forward(self, graph):
        user_embedding = self.embedding_weights["user_embedding"]
        item_embedding = self.embedding_weights["item_embedding"]

        src, dst = graph.edges()
        dst_degree = graph.in_degrees(dst).float()
        src_degree = graph.out_degrees(src).float()
        norm = torch.pow(src_degree * dst_degree, -0.5).view(-1, 1)  # D^-1/2

        for i, layer in enumerate(self.layers, start=1):
            if i == 1:
                embeddings = layer(graph, norm, user_embedding, item_embedding)
            else:
                embeddings = layer(
                    graph, norm, embeddings[: self.user_size], embeddings[self.user_size:]
                )

            user_embedding = user_embedding + embeddings[: self.user_size]
            item_embedding = item_embedding + embeddings[self.user_size:]

        return user_embedding, item_embedding

    def _init_weights(self, in_size):
        initializer = nn.init.xavier_uniform_

        weights_dict = nn.ParameterDict(
            {
                "user_embedding": nn.Parameter(
                    initializer(torch.empty(self.user_size, in_size))
                ),
                "item_embedding": nn.Parameter(
                    initializer(torch.empty(self.item_size, in_size))
                ),
            }
        )
        return weights_dict
