from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Optimizer

from cornac.utils.common import get_rng


def init_parameter_matrix(
    tensor: torch.Tensor, dim0_scale: int = 1, dim1_scale: int = 1
):
    sigma = np.sqrt(
        6.0 / float(tensor.size(0) / dim0_scale + tensor.size(1) / dim1_scale)
    )
    return nn.init._no_grad_uniform_(tensor, -sigma, sigma)


class IndexedAdagradM(Optimizer):
    def __init__(self, params, lr=0.05, momentum=0.0, eps=1e-6):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if eps <= 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr, momentum=momentum, eps=eps)
        super(IndexedAdagradM, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["acc"] = torch.full_like(
                    p, 0, memory_format=torch.preserve_format
                )
                if momentum > 0:
                    state["mom"] = torch.full_like(
                        p, 0, memory_format=torch.preserve_format
                    )

    def share_memory(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["acc"].share_memory_()
                if group["momentum"] > 0:
                    state["mom"].share_memory_()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                clr = group["lr"]
                momentum = group["momentum"]
                if grad.is_sparse:
                    grad = grad.coalesce()
                    grad_indices = grad._indices()[0]
                    grad_values = grad._values()
                    accs = state["acc"][grad_indices] + grad_values.pow(2)
                    state["acc"].index_copy_(0, grad_indices, accs)
                    accs.add_(group["eps"]).sqrt_().mul_(-1 / clr)
                    if momentum > 0:
                        moma = state["mom"][grad_indices]
                        moma.mul_(momentum).add_(grad_values / accs)
                        state["mom"].index_copy_(0, grad_indices, moma)
                        p.index_add_(0, grad_indices, moma)
                    else:
                        p.index_add_(0, grad_indices, grad_values / accs)
                else:
                    state["acc"].add_(grad.pow(2))
                    accs = state["acc"].add(group["eps"])
                    accs.sqrt_()
                    if momentum > 0:
                        mom = state["mom"]
                        mom.mul_(momentum).addcdiv_(grad, accs, value=-clr)
                        p.add_(mom)
                    else:
                        p.addcdiv_(grad, accs, value=-clr)
        return loss


class GRUEmbedding(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GRUEmbedding, self).__init__()
        self.Wx0 = nn.Embedding(dim_in, dim_out * 3, sparse=True)
        self.Wrz0 = nn.Parameter(torch.empty((dim_out, dim_out * 2), dtype=torch.float))
        self.Wh0 = nn.Parameter(torch.empty((dim_out, dim_out * 1), dtype=torch.float))
        self.Bh0 = nn.Parameter(torch.zeros(dim_out * 3, dtype=torch.float))
        self.reset_parameters()

    def reset_parameters(self):
        init_parameter_matrix(self.Wx0.weight, dim1_scale=3)
        init_parameter_matrix(self.Wrz0, dim1_scale=2)
        init_parameter_matrix(self.Wh0, dim1_scale=1)
        nn.init.zeros_(self.Bh0)

    def forward(self, X, H):
        Vx = self.Wx0(X) + self.Bh0
        Vrz = torch.mm(H, self.Wrz0)
        vx_x, vx_r, vx_z = Vx.chunk(3, 1)
        vh_r, vh_z = Vrz.chunk(2, 1)
        r = torch.sigmoid(vx_r + vh_r)
        z = torch.sigmoid(vx_z + vh_z)
        h = torch.tanh(torch.mm(r * H, self.Wh0) + vx_x)
        h = (1.0 - z) * H + z * h
        return h


class GRU4RecModel(nn.Module):
    def __init__(
        self,
        n_items,
        P0=None,
        layers=[100],
        dropout_p_embed=0.0,
        dropout_p_hidden=0.0,
        embedding=0,
        constrained_embedding=True,
        logq=0.0,
        sample_alpha=0.5,
        bpreg=1.0,
        elu_param=0.5,
        loss="cross-entropy",
    ):
        super(GRU4RecModel, self).__init__()
        self.n_items = n_items
        self.P0 = P0
        self.layers = layers
        self.dropout_p_embed = dropout_p_embed
        self.dropout_p_hidden = dropout_p_hidden
        self.embedding = embedding
        self.constrained_embedding = constrained_embedding
        self.logq = logq
        self.sample_alpha = sample_alpha
        self.elu_param = elu_param
        self.bpreg = bpreg
        self.loss = loss
        self.set_loss_function(self.loss)
        self.start = 0
        if constrained_embedding:
            n_input = layers[-1]
        elif embedding:
            self.E = nn.Embedding(n_items, embedding, sparse=True)
            n_input = embedding
        else:
            self.GE = GRUEmbedding(n_items, layers[0])
            n_input = n_items
            self.start = 1
        self.DE = nn.Dropout(dropout_p_embed)
        self.G = []
        self.D = []
        for i in range(self.start, len(layers)):
            self.G.append(nn.GRUCell(layers[i - 1] if i > 0 else n_input, layers[i]))
            self.D.append(nn.Dropout(dropout_p_hidden))
        self.G = nn.ModuleList(self.G)
        self.D = nn.ModuleList(self.D)
        self.Wy = nn.Embedding(n_items, layers[-1], sparse=True)
        self.By = nn.Embedding(n_items, 1, sparse=True)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        if self.embedding:
            init_parameter_matrix(self.E.weight)
        elif not self.constrained_embedding:
            self.GE.reset_parameters()
        for i in range(len(self.G)):
            init_parameter_matrix(self.G[i].weight_ih, dim1_scale=3)
            init_parameter_matrix(self.G[i].weight_hh, dim1_scale=3)
            nn.init.zeros_(self.G[i].bias_ih)
            nn.init.zeros_(self.G[i].bias_hh)
        init_parameter_matrix(self.Wy.weight)
        nn.init.zeros_(self.By.weight)

    def set_loss_function(self, loss):
        if loss == "cross-entropy":
            self.loss_function = self.xe_loss_with_softmax
        elif loss == "bpr-max":
            self.loss_function = self.bpr_max_loss_with_elu
        elif loss == "top1":
            self.loss_function = self.top1
        else:
            raise NotImplementedError

    def xe_loss_with_softmax(self, O, Y, M):
        if self.logq > 0:
            O = O - self.logq * torch.log(
                torch.cat([self.P0[Y[:M]], self.P0[Y[M:]] ** self.sample_alpha])
            )
        X = torch.exp(O - O.max(dim=1, keepdim=True)[0])
        X = X / X.sum(dim=1, keepdim=True)
        return -torch.sum(torch.log(torch.diag(X) + 1e-24))

    def softmax_neg(self, X):
        hm = 1.0 - torch.eye(*X.shape, out=torch.empty_like(X))
        X = X * hm
        e_x = torch.exp(X - X.max(dim=1, keepdim=True)[0]) * hm
        return e_x / e_x.sum(dim=1, keepdim=True)

    def bpr_max_loss_with_elu(self, O, Y, M):
        if self.elu_param > 0:
            O = nn.functional.elu(O, self.elu_param)
        softmax_scores = self.softmax_neg(O)
        target_scores = torch.diag(O)
        target_scores = target_scores.reshape(target_scores.shape[0], -1)
        return torch.sum(
            (
                -torch.log(
                    torch.sum(torch.sigmoid(target_scores - O) * softmax_scores, dim=1)
                    + 1e-24
                )
                + self.bpreg * torch.sum((O**2) * softmax_scores, dim=1)
            )
        )

    def top1(self, O, Y, M):
        target_scores = torch.diag(O)
        target_scores = target_scores.reshape(target_scores.shape[0], -1)
        return torch.sum(
            (
                torch.mean(
                    torch.sigmoid(O - target_scores) + torch.sigmoid(O**2), axis=1
                )
                - torch.sigmoid(target_scores**2) / (M + self.n_sample)
            )
        )

    def _init_numpy_weights(self, shape):
        sigma = np.sqrt(6.0 / (shape[0] + shape[1]))
        m = np.random.rand(*shape).astype("float32") * 2 * sigma - sigma
        return m

    @torch.no_grad()
    def _reset_weights_to_compatibility_mode(self):
        np.random.seed(42)
        if self.constrained_embedding:
            n_input = self.layers[-1]
        elif self.embedding:
            n_input = self.embedding
            self.E.weight.set_(
                torch.tensor(
                    self._init_numpy_weights((self.n_items, n_input)),
                    device=self.E.weight.device,
                )
            )
        else:
            n_input = self.n_items
            m = []
            m.append(self._init_numpy_weights((n_input, self.layers[0])))
            m.append(self._init_numpy_weights((n_input, self.layers[0])))
            m.append(self._init_numpy_weights((n_input, self.layers[0])))
            self.GE.Wx0.weight.set_(
                torch.tensor(np.hstack(m), device=self.GE.Wx0.weight.device)
            )
            m2 = []
            m2.append(self._init_numpy_weights((self.layers[0], self.layers[0])))
            m2.append(self._init_numpy_weights((self.layers[0], self.layers[0])))
            self.GE.Wrz0.set_(torch.tensor(np.hstack(m2), device=self.GE.Wrz0.device))
            self.GE.Wh0.set_(
                torch.tensor(
                    self._init_numpy_weights((self.layers[0], self.layers[0])),
                    device=self.GE.Wh0.device,
                )
            )
            self.GE.Bh0.set_(
                torch.zeros((self.layers[0] * 3,), device=self.GE.Bh0.device)
            )
        for i in range(self.start, len(self.layers)):
            m = []
            m.append(self._init_numpy_weights((n_input, self.layers[i])))
            m.append(self._init_numpy_weights((n_input, self.layers[i])))
            m.append(self._init_numpy_weights((n_input, self.layers[i])))
            self.G[i].weight_ih.set_(
                torch.tensor(np.vstack(m), device=self.G[i].weight_ih.device)
            )
            m2 = []
            m2.append(self._init_numpy_weights((self.layers[i], self.layers[i])))
            m2.append(self._init_numpy_weights((self.layers[i], self.layers[i])))
            m2.append(self._init_numpy_weights((self.layers[i], self.layers[i])))
            self.G[i].weight_hh.set_(
                torch.tensor(np.vstack(m2), device=self.G[i].weight_hh.device)
            )
            self.G[i].bias_hh.set_(
                torch.zeros((self.layers[i] * 3,), device=self.G[i].bias_hh.device)
            )
            self.G[i].bias_ih.set_(
                torch.zeros((self.layers[i] * 3,), device=self.G[i].bias_ih.device)
            )
        self.Wy.weight.set_(
            torch.tensor(
                self._init_numpy_weights((self.n_items, self.layers[-1])),
                device=self.Wy.weight.device,
            )
        )
        self.By.weight.set_(
            torch.zeros((self.n_items, 1), device=self.By.weight.device)
        )

    def embed_constrained(self, X, Y=None):
        if Y is not None:
            XY = torch.cat([X, Y])
            EXY = self.Wy(XY)
            split = X.shape[0]
            E = EXY[:split]
            O = EXY[split:]
            B = self.By(Y)
        else:
            E = self.Wy(X)
            O = self.Wy.weight
            B = self.By.weight
        return E, O, B

    def embed_separate(self, X, Y=None):
        E = self.E(X)
        if Y is not None:
            O = self.Wy(Y)
            B = self.By(Y)
        else:
            O = self.Wy.weight
            B = self.By.weight
        return E, O, B

    def embed_gru(self, X, H, Y=None):
        E = self.GE(X, H)
        if Y is not None:
            O = self.Wy(Y)
            B = self.By(Y)
        else:
            O = self.Wy.weight
            B = self.By.weight
        return E, O, B

    def embed(self, X, H, Y=None):
        if self.constrained_embedding:
            E, O, B = self.embed_constrained(X, Y)
        elif self.embedding > 0:
            E, O, B = self.embed_separate(X, Y)
        else:
            E, O, B = self.embed_gru(X, H[0], Y)
        return E, O, B

    def hidden_step(self, X, H, training=False):
        for i in range(self.start, len(self.layers)):
            X = self.G[i](X, Variable(H[i]))
            if training:
                X = self.D[i](X)
            H[i] = X
        return X

    def score_items(self, X, O, B):
        O = torch.mm(X, O.T) + B.T
        return O

    def forward(self, X, H, Y, training=False):
        E, O, B = self.embed(X, H, Y)
        if training:
            E = self.DE(E)
        if not (self.constrained_embedding or self.embedding):
            H[0] = E
        Xh = self.hidden_step(E, H, training=training)
        R = self.score_items(Xh, O, B)
        return R


def io_iter(
    s_iter, uir_tuple, n_sample=0, sample_alpha=0, rng=None, batch_size=1, shuffle=False
):
    """Paralellize mini-batch of input-output items. Create an iterator over data yielding batch of input item indices, batch of output item indices,
    batch of start masking, batch of end masking, and batch of valid ids (relative positions of current sequences in the last batch).

    Parameters
    ----------
    batch_size: int, optional, default = 1

    shuffle: bool, optional, default: False
        If `True`, orders of triplets will be randomized. If `False`, default orders kept.

    Returns
    -------
    iterator : batch of input item indices, batch of output item indices, batch of starting sequence mask, batch of ending sequence mask, batch of valid ids

    """
    rng = rng if rng is not None else get_rng(None)
    start_mask = np.zeros(batch_size, dtype="int")
    end_mask = np.ones(batch_size, dtype="int")
    input_iids = None
    output_iids = None
    l_pool = []
    c_pool = [None for _ in range(batch_size)]
    sizes = np.zeros(batch_size, dtype="int")
    if n_sample > 0:
        item_count = Counter(uir_tuple[1])
        item_indices = np.array(
            [iid for iid, _ in item_count.most_common()], dtype="int"
        )
        item_dist = (
            np.array([cnt for _, cnt in item_count.most_common()], dtype="float")
            ** sample_alpha
        )
        item_dist = item_dist / item_dist.sum()
    for _, batch_mapped_ids in s_iter(batch_size, shuffle):
        l_pool += batch_mapped_ids
        while len(l_pool) > 0:
            if end_mask.sum() == 0:
                input_iids = uir_tuple[1][
                    [mapped_ids[-sizes[idx]] for idx, mapped_ids in enumerate(c_pool)]
                ]
                output_iids = uir_tuple[1][
                    [
                        mapped_ids[-sizes[idx] + 1]
                        for idx, mapped_ids in enumerate(c_pool)
                    ]
                ]
                sizes -= 1
                for idx, size in enumerate(sizes):
                    if size == 1:
                        end_mask[idx] = 1
                if n_sample > 0:
                    negative_samples = rng.choice(
                        item_indices, size=n_sample, replace=True, p=item_dist
                    )
                    output_iids = np.concatenate([output_iids, negative_samples])
                yield input_iids, output_iids, start_mask, np.arange(
                    batch_size, dtype="int"
                )
                start_mask.fill(0)  # reset start masking
            while end_mask.sum() > 0 and len(l_pool) > 0:
                next_seq = l_pool.pop()
                if len(next_seq) > 1:
                    idx = np.nonzero(end_mask)[0][0]
                    end_mask[idx] = 0
                    start_mask[idx] = 1
                    c_pool[idx] = next_seq
                    sizes[idx] = len(c_pool[idx])

    valid_id = np.ones(batch_size, dtype="int")
    while True:
        for idx, size in enumerate(sizes):
            if size == 1:
                end_mask[idx] = 1
                valid_id[idx] = 0
        input_iids = uir_tuple[1][
            [
                mapped_ids[-sizes[idx]]
                for idx, mapped_ids in enumerate(c_pool)
                if sizes[idx] > 1
            ]
        ]
        output_iids = uir_tuple[1][
            [
                mapped_ids[-sizes[idx] + 1]
                for idx, mapped_ids in enumerate(c_pool)
                if sizes[idx] > 1
            ]
        ]
        sizes -= 1
        for idx, size in enumerate(sizes):
            if size == 1:
                end_mask[idx] = 1
        start_mask = start_mask[np.nonzero(valid_id)[0]]
        end_mask = end_mask[np.nonzero(valid_id)[0]]
        sizes = sizes[np.nonzero(valid_id)[0]]
        c_pool = [_ for _, valid in zip(c_pool, valid_id) if valid > 0]
        if n_sample > 0:
            negative_samples = rng.choice(
                item_indices, size=n_sample, replace=True, p=item_dist
            )
            output_iids = np.concatenate([output_iids, negative_samples])
        yield input_iids, output_iids, start_mask, np.nonzero(valid_id)[0]
        valid_id = np.ones(len(input_iids), dtype="int")
        if end_mask.sum() == len(input_iids):
            break
        start_mask.fill(0)  # reset start masking


def score(model, layers, device, history_items):
    model.eval()
    H = []
    for i in range(len(layers)):
        H.append(
            torch.zeros(
                (1, layers[i]), dtype=torch.float32, requires_grad=False, device=device
            )
        )
    for iid in history_items:
        O = model.forward(
            torch.tensor([iid], requires_grad=False, device=device),
            H,
            None,
            training=False,
        )
    return O.squeeze().cpu().detach().numpy()
