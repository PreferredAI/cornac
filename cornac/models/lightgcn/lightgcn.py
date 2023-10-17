# Reference: https://github.com/dmlc/dgl/blob/master/examples/pytorch/NGCF/NGCF/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn


USER_KEY = "user"
ITEM_KEY = "item"


def construct_graph(data_set):
    """
    Generates graph given a cornac data set

    Parameters
    ----------
    data_set : cornac.data.dataset.Dataset
        The data set as provided by cornac
    """
    user_indices, item_indices, _ = data_set.uir_tuple

    data_dict = {
        (USER_KEY, "user_item", ITEM_KEY): (user_indices, item_indices),
        (ITEM_KEY, "item_user", USER_KEY): (item_indices, user_indices),
    }
    num_dict = {USER_KEY: data_set.total_users, ITEM_KEY: data_set.total_items}

    return dgl.heterograph(data_dict, num_nodes_dict=num_dict)


class GCNLayer(nn.Module):
    def __init__(self, norm_dict):
        super(GCNLayer, self).__init__()

        # norm
        self.norm_dict = norm_dict

    def forward(self, g, feat_dict):
        funcs = {}  # message and reduce functions dict
        # for each type of edges, compute messages and reduce them all
        for srctype, etype, dsttype in g.canonical_etypes:
            src, dst = g.edges(etype=(srctype, etype, dsttype))
            norm = self.norm_dict[(srctype, etype, dsttype)]
            # TODO: CHECK HERE
            messages = norm * feat_dict[srctype][src]  # compute messages
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
            h = F.normalize(g.nodes[ntype].data["h"], dim=1, p=2)  # l2 normalize
            feature_dict[ntype] = h
        return feature_dict


class Model(nn.Module):
    def __init__(self, g, in_size, num_layers, lambda_reg, device=None):
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

        self.layers = nn.ModuleList([GCNLayer(self.norm_dict) for _ in range(num_layers)])

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
        user_embeds = h_dict[USER_KEY]
        item_embeds = h_dict[ITEM_KEY]

        for k, layer in enumerate(self.layers):
            h_dict = layer(g, h_dict)
            user_embeds = user_embeds + (h_dict[USER_KEY] * 1 / (k + 1))
            item_embeds = item_embeds + (h_dict[ITEM_KEY] * 1 / (k + 1))

        u_g_embeddings = user_embeds if users is None else user_embeds[users, :]
        pos_i_g_embeddings = item_embeds if pos_items is None else item_embeds[pos_items, :]
        neg_i_g_embeddings = item_embeds if neg_items is None else item_embeds[neg_items, :]

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
