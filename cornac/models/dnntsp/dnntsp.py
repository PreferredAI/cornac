import itertools
import random
from collections import defaultdict
from typing import List

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import trange


OPTIMIZER_DICT = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "rmsprop": torch.optim.RMSprop,
    "adagrad": torch.optim.Adagrad,
}


class MaskedSelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads=4, attention_aggregate="concat"):
        super(MaskedSelfAttention, self).__init__()
        # aggregate multi-heads by concatenation or mean
        self.attention_aggregate = attention_aggregate

        # the dimension of each head is dq // n_heads
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.n_heads = n_heads

        if attention_aggregate == "concat":
            self.per_head_dim = self.dq = self.dk = self.dv = output_dim // n_heads
        elif attention_aggregate == "mean":
            self.per_head_dim = self.dq = self.dk = self.dv = output_dim
        else:
            raise ValueError(f"wrong value for aggregate {attention_aggregate}")

        self.Wq = nn.Linear(input_dim, n_heads * self.dq, bias=False)
        self.Wk = nn.Linear(input_dim, n_heads * self.dk, bias=False)
        self.Wv = nn.Linear(input_dim, n_heads * self.dv, bias=False)

    def forward(self, input_tensor):
        seq_length = input_tensor.shape[1]
        # tensor, shape (nodes_num, T_max, n_heads * dim_per_head)
        Q = self.Wq(input_tensor)
        K = self.Wk(input_tensor)
        V = self.Wv(input_tensor)
        # multi_head attention
        # Q, tensor, shape (nodes_num, n_heads, T_max, dim_per_head)
        Q = Q.reshape(
            input_tensor.shape[0], input_tensor.shape[1], self.n_heads, self.dq
        ).transpose(1, 2)
        # K after transpose, tensor, shape (nodes_num, n_heads, dim_per_head, T_max)
        K = K.reshape(
            input_tensor.shape[0], input_tensor.shape[1], self.n_heads, self.dk
        ).permute(0, 2, 3, 1)
        # V, tensor, shape (nodes_num, n_heads, T_max, dim_per_head)
        V = V.reshape(
            input_tensor.shape[0], input_tensor.shape[1], self.n_heads, self.dv
        ).transpose(1, 2)

        # scaled attention_score, tensor, shape (nodes_num, n_heads, T_max, T_max)
        attention_score = Q.matmul(K) / np.sqrt(self.per_head_dim)

        # attention_mask, tensor, shape -> (T_max, T_max)  -inf in the top and right
        attention_mask = (
            torch.zeros(seq_length, seq_length)
            .masked_fill(torch.tril(torch.ones(seq_length, seq_length)) == 0, -np.inf)
            .to(input_tensor.device)
        )
        # attention_mask will be broadcast to (nodes_num, n_heads, T_max, T_max)
        attention_score = attention_score + attention_mask
        # (nodes_num, n_heads, T_max, T_max)
        attention_score = torch.softmax(attention_score, dim=-1)

        # multi_result, tensor, shape (nodes_num, n_heads, T_max, dim_per_head)
        multi_head_result = attention_score.matmul(V)
        if self.attention_aggregate == "concat":
            # multi_result, tensor, shape (nodes_num, T_max, n_heads * dim_per_head = output_dim)
            # concat multi-head attention results
            output = multi_head_result.transpose(1, 2).reshape(
                input_tensor.shape[0], seq_length, self.n_heads * self.per_head_dim
            )
        elif self.attention_aggregate == "mean":
            # multi_result, tensor, shape (nodes_num, T_max, dim_per_head = output_dim)
            # mean multi-head attention results
            output = multi_head_result.transpose(1, 2).mean(dim=2)
        else:
            raise ValueError(f"wrong value for aggregate {self.attention_aggregate}")

        return output


class GlobalGatedUpdate(nn.Module):
    def __init__(self, n_items, embedding_matrix):
        super(GlobalGatedUpdate, self).__init__()
        self.n_items = n_items
        self.embedding_matrix = embedding_matrix

        # alpha -> the weight for updating
        self.alpha = nn.Parameter(torch.rand(n_items, 1), requires_grad=True)

    def forward(self, graph, nodes, nodes_output):
        nums_nodes, id = graph.batch_num_nodes(), 0
        items_embedding = self.embedding_matrix(
            torch.tensor([i for i in range(self.n_items)]).to(nodes.device)
        )
        batch_embedding = []
        for num_nodes in nums_nodes:
            # tensor, shape, (user_nodes, item_embed_dim)
            output_node_features = nodes_output[id : id + num_nodes, :]
            # get each user's nodes
            output_nodes = nodes[id : id + num_nodes]
            # beta, tensor, (n_items, 1), indicator vector, appear item -> 1, not appear -> 0
            beta = torch.zeros(self.n_items, 1).to(nodes.device)
            beta[output_nodes] = 1
            # update global embedding by gated mechanism
            # broadcast (n_items, 1) * (n_items, item_embed_dim) -> (n_items, item_embed_dim)
            embed = (1 - beta * self.alpha) * items_embedding.clone()
            # appear items: (1 - self.alpha) * origin + self.alpha * update, not appear items: origin
            embed[output_nodes, :] = (
                embed[output_nodes, :] + self.alpha[output_nodes] * output_node_features
            )
            batch_embedding.append(embed)
            id += num_nodes
        # (B, n_items, item_embed_dim)
        batch_embedding = torch.stack(batch_embedding)
        return batch_embedding


class AggregateNodesTemporalFeature(nn.Module):
    def __init__(self, item_embed_dim):
        super(AggregateNodesTemporalFeature, self).__init__()

        self.Wq = nn.Linear(item_embed_dim, 1, bias=False)

    def forward(self, graph, lengths, nodes_output):
        nums_nodes, id = graph.batch_num_nodes(), 0
        aggregated_features = []
        for num_nodes, length in zip(nums_nodes, lengths):
            # get each user's length, tensor, shape, (user_nodes, user_length, item_embed_dim)
            output_node_features = nodes_output[id : id + num_nodes, :length, :]
            # weights for each timestamp, tensor, shape, (user_nodes, 1, user_length)
            # (user_nodes, user_length, 1) transpose to -> (user_nodes, 1, user_length)
            weights = self.Wq(output_node_features).transpose(1, 2)
            # (user_nodes, 1, user_length) matmul (user_nodes, user_length, item_embed_dim)
            # -> (user_nodes, 1, item_embed_dim) squeeze to (user_nodes, item_embed_dim)
            # aggregated_feature, tensor, shape, (user_nodes, item_embed_dim)
            aggregated_feature = weights.matmul(output_node_features).squeeze(dim=1)
            aggregated_features.append(aggregated_feature)
            id += num_nodes
        # (n_1+n_2+..., item_embed_dim)
        aggregated_features = torch.cat(aggregated_features, dim=0)
        return aggregated_features


class WeightedGraphConv(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(WeightedGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, graph, node_features, edge_weights):
        graph = graph.local_var()
        # multi W first to project the features, with bias
        # (N, F) / (N, T, F)
        graph.ndata["n"] = node_features
        # edge_weights, shape (T, N^2)
        # one way: use dgl.function is faster and less requirement of GPU memory
        graph.edata["e"] = edge_weights.t().unsqueeze(dim=-1)  # (E, T, 1)
        graph.update_all(fn.u_mul_e("n", "e", "msg"), fn.sum("msg", "h"))

        # another way: use user defined function, needs more GPU memory
        # graph.edata['e'] = edge_weights.t()
        # graph.update_all(self.gcn_message, self.gcn_reduce)

        node_features = graph.ndata.pop("h")
        output = self.linear(node_features)
        return output

    @staticmethod
    def gcn_message(edges):
        if edges.src["n"].dim() == 2:
            # (E, T, 1) (E, 1, F),  matmul ->  matmul (E, T, F)
            return {
                "msg": torch.matmul(
                    edges.data["e"].unsqueeze(dim=-1), edges.src["n"].unsqueeze(dim=1)
                )
            }

        elif edges.src["n"].dim() == 3:
            # (E, T, 1) (E, T, F),  mul -> (E, T, F)
            return {"msg": torch.mul(edges.data["e"].unsqueeze(dim=-1), edges.src["n"])}

        else:
            raise ValueError(
                f"wrong shape for edges.src['n'], the length of shape is {edges.src['n'].dim()}"
            )

    @staticmethod
    def gcn_reduce(nodes):
        # propagate, the first dimension is nodes num in a batch
        # h, tensor, shape, (N, neighbors, T, F) -> (N, T, F)
        return {"h": torch.sum(nodes.mailbox["msg"], 1)}


class WeightedGCN(nn.Module):
    def __init__(self, in_features: int, hidden_sizes: List[int], out_features: int):
        super(WeightedGCN, self).__init__()
        gcns, relus, bns = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        # layers for hidden_size
        input_size = in_features
        for hidden_size in hidden_sizes:
            gcns.append(WeightedGraphConv(input_size, hidden_size))
            relus.append(nn.ReLU())
            bns.append(nn.BatchNorm1d(hidden_size))
            input_size = hidden_size
        # output layer
        gcns.append(WeightedGraphConv(hidden_sizes[-1], out_features))
        relus.append(nn.ReLU())
        bns.append(nn.BatchNorm1d(out_features))
        self.gcns, self.relus, self.bns = gcns, relus, bns

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_features: torch.Tensor,
        edges_weight: torch.Tensor,
    ):
        h = node_features
        for gcn, relu, bn in zip(self.gcns, self.relus, self.bns):
            # (n_1+n_2+..., T, features)
            h = gcn(graph, h, edges_weight)
            h = bn(h.transpose(1, -1)).transpose(1, -1)
            h = relu(h)
        return h


class StackedWeightedGCNBlocks(nn.ModuleList):
    def __init__(self, *args, **kwargs):
        super(StackedWeightedGCNBlocks, self).__init__(*args, **kwargs)

    def forward(self, *input):
        g, nodes_feature, edge_weights = input
        h = nodes_feature
        for module in self:
            h = module(g, h, edge_weights)
        return h


class TemporalSetPrediction(nn.Module):
    def __init__(self, n_items, emb_dim, seed):
        super(TemporalSetPrediction, self).__init__()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.embedding_matrix = nn.Embedding(n_items, emb_dim)

        self.emb_dim = emb_dim
        self.n_items = n_items
        self.stacked_gcn = StackedWeightedGCNBlocks(
            [WeightedGCN(emb_dim, [emb_dim], emb_dim)]
        )

        self.masked_self_attention = MaskedSelfAttention(
            input_dim=emb_dim, output_dim=emb_dim
        )

        self.aggregate_nodes_temporal_feature = AggregateNodesTemporalFeature(
            item_embed_dim=emb_dim
        )

        self.global_gated_update = GlobalGatedUpdate(
            n_items=n_items, embedding_matrix=self.embedding_matrix
        )

        self.fc_output = nn.Linear(emb_dim, 1)

    def forward(
        self,
        batch_graph,
        batch_nodes_feature,
        batch_edges_weight,
        batch_lengths,
        batch_nodes,
    ):
        # perform weighted gcn on dynamic graphs (n_1+n_2+..., T_max, item_embed_dim)
        batch_nodes_output = [
            self.stacked_gcn(graph, nodes_feature, edges_weight)
            for graph, nodes_feature, edges_weight in zip(
                batch_graph, batch_nodes_feature, batch_edges_weight
            )
        ]

        # self-attention in time dimension, (n_1+n_2+..., T_max,  item_embed_dim)
        batch_nodes_output = [
            self.masked_self_attention(nodes_output)
            for nodes_output in batch_nodes_output
        ]

        # aggregate node features in temporal dimension, (n_1+n_2+..., item_embed_dim)
        batch_nodes_output = [
            self.aggregate_nodes_temporal_feature(graph, lengths, nodes_output)
            for graph, lengths, nodes_output in zip(
                batch_graph, batch_lengths, batch_nodes_output
            )
        ]

        # (batch_size, n_items, item_embed_dim)
        batch_nodes_output = [
            self.global_gated_update(graph, nodes, nodes_output)
            for graph, nodes, nodes_output in zip(
                batch_graph, batch_nodes, batch_nodes_output
            )
        ]

        # (batch_size, n_items)
        outputs = self.fc_output(torch.stack(batch_nodes_output)).squeeze()

        return outputs


def get_edges_weight(history_baskets):
    edges_weight_dict = defaultdict(float)
    for basket_items in history_baskets:
        for item_i, item_j in itertools.permutations(basket_items, 2):
            edges_weight_dict[(item_i, item_j)] += 1
    return edges_weight_dict


def transform_data(
    bi_batch, item_embedding, total_items, device=torch.device("cpu"), is_test=False
):
    if is_test:
        batch_history_items = [
            [np.unique(basket).tolist() for basket in basket_items]
            for basket_items in bi_batch
        ]
    else:
        batch_history_items = [
            [np.unique(basket).tolist() for basket in basket_items[:-1]]
            for basket_items in bi_batch
        ]
    batch_lengths = [
        [len(basket) for basket in history_items]
        for history_items in batch_history_items
    ]
    if is_test:
        batch_targets = None
    else:
        batch_targets = np.zeros((len(bi_batch), total_items), dtype="uint8")
        for inc, basket_items in enumerate(bi_batch):
            batch_targets[inc, basket_items[-1]] = 1
        batch_targets = torch.tensor(batch_targets, dtype=torch.uint8, device=device)
    batch_nodes = [
        torch.tensor(
            list(set(itertools.chain.from_iterable(history_items))),
            dtype=torch.int32,
            device=device,
        )
        for history_items in batch_history_items
    ]
    batch_nodes_feature = [item_embedding(nodes) for nodes in batch_nodes]

    batch_project_nodes = [
        torch.tensor(list(range(nodes.shape[0]))) for nodes in batch_nodes
    ]
    batch_src = [
        project_nodes.repeat((project_nodes.shape[0], 1)).T.flatten().tolist()
        for project_nodes in batch_project_nodes
    ]
    batch_dst = [
        project_nodes.repeat((project_nodes.shape[0],)).flatten().tolist()
        for project_nodes in batch_project_nodes
    ]
    batch_g = [
        dgl.graph((src, dst), num_nodes=project_nodes.shape[0]).to(device)
        for src, dst, project_nodes in zip(batch_src, batch_dst, batch_project_nodes)
    ]
    batch_edges_weight_dict = [
        get_edges_weight(history_items) for history_items in batch_history_items
    ]

    for i, nodes in enumerate(batch_nodes):
        for node in nodes.tolist():
            if batch_edges_weight_dict[i][(node, node)] == 0.0:
                batch_edges_weight_dict[i][(node, node)] = 1.0
        max_weight = max(batch_edges_weight_dict[i].values())
        for k, v in batch_edges_weight_dict[i].items():
            batch_edges_weight_dict[i][k] = v / max_weight

    batch_edges_weight = []
    for edges_weight_dict, history_items, nodes in zip(
        batch_edges_weight_dict, batch_history_items, batch_nodes
    ):
        edges_weight = []
        for basket in history_items:
            edge_weight = []
            for node_1 in nodes.tolist():
                for node_2 in nodes.tolist():
                    if (node_1 in basket and node_2 in basket) or (node_1 == node_2):
                        edge_weight.append(edges_weight_dict.get((node_1, node_2), 0.0))
                    else:
                        edge_weight.append(0.0)
            edges_weight.append(torch.Tensor(edge_weight))
        batch_edges_weight.append(torch.stack(edges_weight).to(device))
    return (
        batch_g,
        batch_nodes_feature,
        batch_edges_weight,
        batch_lengths,
        batch_nodes,
        batch_targets,
    )


class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, predict, truth):
        """
        Args:
            predict: (batch_size, items_total) / (batch_size, baskets_num, item_total)
            truth: (batch_size, items_total) / (batch_size, baskets_num, item_total)
        Returns:
            output: tensor
        """
        result = self.batch_bpr_loss(predict, truth)

        return result

    def batch_bpr_loss(self, predict, truth):
        """
        Args:
            predict: (batch_size, items_total)
            truth: (batch_size, items_total)
        Returns:
            output: tensor
        """
        items_total = truth.shape[1]
        nll = 0
        for user, predictUser in zip(truth, predict):
            pos_idx = user.clone().detach()
            preUser = predictUser[pos_idx]
            non_zero_list = list(itertools.chain.from_iterable(torch.nonzero(user)))
            random_list = list(set(range(0, items_total)) - set(non_zero_list))
            random.shuffle(random_list)
            neg_idx = torch.tensor(random_list[: len(preUser)])
            score = preUser - predictUser[neg_idx]
            nll += -torch.mean(torch.nn.LogSigmoid()(score))
        return nll


class WeightMSELoss(nn.Module):
    def __init__(self, weights=None):
        """
        Args:
            weights: tensor, (items_total, )
        """
        super(WeightMSELoss, self).__init__()
        self.weights = weights
        if weights is not None:
            self.weights = torch.sqrt(weights)
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, predict, truth):
        """
        Args:
            predict: tenor, (batch_size, items_total)
            truth: tensor, (batch_size, items_total)
        Returns:
            output: tensor
        """
        predict = torch.sigmoid(predict)
        truth = truth.float()
        if self.weights is not None:
            self.weights = self.weights.to(truth.device)
            predict = predict * self.weights
            truth = truth * self.weights

        loss = self.mse_loss(predict, truth)
        return loss


#######################################################################################


def scheduler_fn(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)


def loss_fn(loss_type=None, weights=None):
    if loss_type == "bpr":
        return BPRLoss()
    elif loss_type == "mse":
        return WeightMSELoss()
    elif loss_type == "weight_mse":
        assert weights is not None, f"weight_mse loss required 'weights' but {weights}"
        return WeightMSELoss(weights=weights)
    elif loss_type == "multi_label_soft_margin":
        return nn.MultiLabelSoftMarginLoss(reduction="sum")
    else:
        raise ValueError("Unknown loss function")


def get_class_weights(train_set, total_items, device):
    unique, counts = np.unique(train_set.uir_tuple[1], return_counts=True)
    item_freq = torch.ones(total_items, dtype=torch.float32, device=device)
    item_freq[unique] += torch.from_numpy(counts.astype(np.float32)).to(device)
    item_freq /= train_set.num_baskets
    weights = item_freq.max() / item_freq
    weights = weights / weights.max()
    return weights


def learn(
    model: TemporalSetPrediction,
    train_set,
    total_items,
    val_set=None,
    n_epochs=10,
    batch_size=64,
    lr=0.001,
    weight_decay=0.0,
    loss_type="bpr",
    optimizer="adam",
    device=torch.device("cpu"),
    verbose=True,
):
    model = model.to(device)
    weights = (
        get_class_weights(train_set, total_items=total_items, device=device)
        if loss_type in ["weight_mse"]
        else None
    )
    criteria = loss_fn(loss_type=loss_type, weights=weights)
    optimizer = OPTIMIZER_DICT[optimizer](
        params=model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = scheduler_fn(optimizer=optimizer)
    progress_bar = trange(1, n_epochs + 1, disable=not verbose)
    last_val_loss = np.inf
    last_loss = np.inf
    for _ in progress_bar:
        model.train()
        total_loss = 0.0
        cnt = 0
        for inc, (_, _, bi_batch) in enumerate(
            train_set.ubi_iter(batch_size, shuffle=True)
        ):
            (
                g,
                nodes_feature,
                edges_weight,
                lengths,
                nodes,
                targets,
            ) = transform_data(
                bi_batch,
                item_embedding=model.embedding_matrix,
                total_items=total_items,
                device=device,
            )
            preds = model(g, nodes_feature, edges_weight, lengths, nodes)
            loss = criteria(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.data.item()
            cnt += len(bi_batch)
            last_loss = total_loss / cnt
            if inc % 10 == 0:
                progress_bar.set_postfix(loss=last_loss, val_loss=last_val_loss)

        if val_set is not None:
            model.eval()
            total_val_loss = 0.0
            cnt = 0
            for inc, (_, _, bi_batch) in enumerate(
                val_set.ubi_iter(batch_size, shuffle=False)
            ):
                (
                    g,
                    nodes_feature,
                    edges_weight,
                    lengths,
                    nodes,
                    targets,
                ) = transform_data(
                    bi_batch,
                    item_embedding=model.embedding_matrix,
                    total_items=total_items,
                    device=device,
                )
                preds = model(g, nodes_feature, edges_weight, lengths, nodes)
                loss = criteria(preds, targets)

                total_val_loss += loss.data.item()
                cnt += len(bi_batch)
                last_val_loss = total_val_loss / cnt
                if inc % 10 == 0:
                    progress_bar.set_postfix(loss=last_loss, val_loss=last_val_loss)

            # Note that step should be called after validate
            scheduler.step(total_val_loss)
