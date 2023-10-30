# Reference: https://github.com/dmlc/dgl/blob/master/examples/pytorch/NGCF/NGCF/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn


USER_KEY = "user"
ITEM_KEY = "item"


def construct_graph(data_set, total_users, total_items):
    """
    Generates graph given a cornac data set

    Parameters
    ----------
    data_set : cornac.data.dataset.Dataset
        The data set as provided by cornac
    """
    user_indices, item_indices, _ = data_set.uir_tuple

    # construct graph from the train data and add self-loops
    user_selfs = [i for i in range(total_users)]
    item_selfs = [i for i in range(total_items)]

    data_dict = {
        (USER_KEY, "user_self", USER_KEY): (user_selfs, user_selfs),
        (ITEM_KEY, "item_self", ITEM_KEY): (item_selfs, item_selfs),
        (USER_KEY, "user_item", ITEM_KEY): (user_indices, item_indices),
        (ITEM_KEY, "item_user", USER_KEY): (item_indices, user_indices),
    }
    num_dict = {USER_KEY: total_users, ITEM_KEY: total_items}

    return dgl.heterograph(data_dict, num_nodes_dict=num_dict)


class NGCFLayer(nn.Module):
    def __init__(self, in_size, out_size, norm_dict, dropout):
        super(NGCFLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        # weights for different types of messages
        self.W1 = nn.Linear(in_size, out_size, bias=True)
        self.W2 = nn.Linear(in_size, out_size, bias=True)

        # leaky relu
        self.leaky_relu = nn.LeakyReLU(0.2)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

        # initialization
        torch.nn.init.xavier_uniform_(self.W1.weight)
        torch.nn.init.constant_(self.W1.bias, 0)
        torch.nn.init.xavier_uniform_(self.W2.weight)
        torch.nn.init.constant_(self.W2.bias, 0)

        # norm
        self.norm_dict = norm_dict

    def forward(self, g, feat_dict):
        funcs = {}  # message and reduce functions dict
        # for each type of edges, compute messages and reduce them all
        for srctype, etype, dsttype in g.canonical_etypes:
            if srctype == dsttype:  # for self loops
                messages = self.W1(feat_dict[srctype])
                g.nodes[srctype].data[etype] = messages  # store in ndata
                funcs[(srctype, etype, dsttype)] = (
                    fn.copy_u(etype, "m"),
                    fn.sum("m", "h"),
                )  # define message and reduce functions
            else:
                src, dst = g.edges(etype=(srctype, etype, dsttype))
                norm = self.norm_dict[(srctype, etype, dsttype)]
                messages = norm * (
                    self.W1(feat_dict[srctype][src])
                    + self.W2(feat_dict[srctype][src] * feat_dict[dsttype][dst])
                )  # compute messages
                g.edges[(srctype, etype, dsttype)].data[
                    etype
                ] = messages  # store in edata
                funcs[(srctype, etype, dsttype)] = (
                    fn.copy_e(etype, "m"),
                    fn.sum("m", "h"),
                )  # define message and reduce functions

        g.multi_update_all(
            funcs, "sum"
        )  # update all, reduce by first type-wisely then across different types
        feature_dict = {}
        for ntype in g.ntypes:
            h = self.leaky_relu(g.nodes[ntype].data["h"])  # leaky relu
            h = self.dropout(h)  # dropout
            h = F.normalize(h, dim=1, p=2)  # l2 normalize
            feature_dict[ntype] = h
        return feature_dict


class Model(nn.Module):
    def __init__(self, g, in_size, layer_sizes, dropout_rates, lambda_reg, device=None):
        super(Model, self).__init__()
        self.norm_dict = dict()
        self.lambda_reg = lambda_reg
        self.device = device

        for srctype, etype, dsttype in g.canonical_etypes:
            src, dst = g.edges(etype=(srctype, etype, dsttype))
            dst_degree = g.in_degrees(
                dst, etype=(srctype, etype, dsttype)
            ).float()  # obtain degrees
            src_degree = g.out_degrees(src, etype=(srctype, etype, dsttype)).float()
            norm = torch.pow(src_degree * dst_degree, -0.5).unsqueeze(1)  # compute norm
            self.norm_dict[(srctype, etype, dsttype)] = norm

        self.layers = nn.ModuleList()

        # sanity check, just to ensure layer sizes and dropout_rates have the same size
        assert len(layer_sizes) == len(dropout_rates), "'layer_sizes' and " \
            "'dropout_rates' must be of the same size"

        self.layers.append(
            NGCFLayer(in_size, layer_sizes[0], self.norm_dict, dropout_rates[0])
        )
        self.num_layers = len(layer_sizes)
        for i in range(self.num_layers - 1):
            self.layers.append(
                NGCFLayer(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                    self.norm_dict,
                    dropout_rates[i + 1],
                )
            )
        self.initializer = nn.init.xavier_uniform_

        # embeddings for different types of nodes
        self.feature_dict = nn.ParameterDict(
            {
                ntype: nn.Parameter(
                    self.initializer(torch.empty(g.num_nodes(ntype), in_size))
                )
                for ntype in g.ntypes
            }
        )

    def forward(self, g, users=None, pos_items=None, neg_items=None):
        h_dict = {ntype: self.feature_dict[ntype] for ntype in g.ntypes}
        # obtain features of each layer and concatenate them all
        user_embeds = []
        item_embeds = []
        user_embeds.append(h_dict[USER_KEY])
        item_embeds.append(h_dict[ITEM_KEY])
        for layer in self.layers:
            h_dict = layer(g, h_dict)
            user_embeds.append(h_dict[USER_KEY])
            item_embeds.append(h_dict[ITEM_KEY])
        user_embd = torch.cat(user_embeds, 1)
        item_embd = torch.cat(item_embeds, 1)

        u_g_embeddings = user_embd if users is None else user_embd[users, :]
        pos_i_g_embeddings = item_embd if pos_items is None else item_embd[pos_items, :]
        neg_i_g_embeddings = item_embd if neg_items is None else item_embd[neg_items, :]

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings

    def loss_fn(self, users, pos_items, neg_items):
        pos_scores = (users * pos_items).sum(1)
        neg_scores = (users * neg_items).sum(1)

        bpr_loss = F.softplus(neg_scores - pos_scores).mean()
        reg_loss = (
            (1 / 2)
            * (
                torch.norm(users) ** 2
                + torch.norm(pos_items) ** 2
                + torch.norm(neg_items) ** 2
            )
            / len(users)
        )

        return bpr_loss + self.lambda_reg * reg_loss, bpr_loss, reg_loss
