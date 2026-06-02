# Copyright 2026 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from ..seq_utils.iterators import io_iter
from ..seq_utils.optim import IndexedAdagradM


def init_parameter_matrix(tensor: torch.Tensor, dim0_scale: int = 1, dim1_scale: int = 1):
    sigma = np.sqrt(6.0 / float(tensor.size(0) / dim0_scale + tensor.size(1) / dim1_scale))
    return nn.init._no_grad_uniform_(tensor, -sigma, sigma)


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
    """GRU4Rec PyTorch architecture.

    The model computes a hidden state from the latest item id (and previous
    hidden state) and scores all candidate items via either:

    - ``constrained_embedding=True``: tied input/output item embeddings.
    - ``embedding > 0``: separate input embedding of given size.
    - otherwise: a custom :class:`GRUEmbedding` directly producing the hidden.

    The model returns a ``(B, B+N)`` score matrix where columns 0..B-1 are
    the in-batch positives and columns B..B+N-1 are shared sampled negatives.
    Padding row ``n_items`` is included so it is safe to pass ``n_items`` as
    a fake input id for cold-start fallback at score-time.
    """

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
        self.start = 0
        if constrained_embedding:
            n_input = layers[-1]
        elif embedding:
            self.E = nn.Embedding(n_items + 1, embedding, sparse=True, padding_idx=n_items)
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
        self.Wy = nn.Embedding(n_items + 1, layers[-1], sparse=True, padding_idx=n_items)
        self.By = nn.Embedding(n_items + 1, 1, sparse=True, padding_idx=n_items)
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
        if self.Wy.padding_idx is not None:
            self.Wy.weight.data[self.Wy.padding_idx].zero_()
        if self.By.padding_idx is not None:
            self.By.weight.data[self.By.padding_idx].zero_()

    def _init_numpy_weights(self, shape):
        sigma = float(np.sqrt(6.0 / (shape[0] + shape[1])))
        m = (np.random.rand(*shape) * 2 * sigma - sigma).astype("float32")
        return m

    @torch.no_grad()
    def _reset_weights_to_compatibility_mode(self):
        """Reset weights using numpy RNG with seed 42 for reproducibility.

        Note: when ``constrained_embedding=False`` and ``embedding > 0`` the
        ``E`` embedding includes a padding row (``n_items``), and ``Wy``/``By``
        also include padding rows. We only set the first ``n_items`` rows.
        """
        np.random.seed(42)
        if self.constrained_embedding:
            n_input = self.layers[-1]
        elif self.embedding:
            n_input = self.embedding
            self.E.weight.data[: self.n_items].copy_(
                torch.tensor(
                    self._init_numpy_weights((self.n_items, n_input)),
                    device=self.E.weight.device,
                )
            )
        else:
            n_input = self.n_items
            m = [
                self._init_numpy_weights((n_input, self.layers[0])),
                self._init_numpy_weights((n_input, self.layers[0])),
                self._init_numpy_weights((n_input, self.layers[0])),
            ]
            self.GE.Wx0.weight.set_(torch.tensor(np.hstack(m), dtype=torch.float32, device=self.GE.Wx0.weight.device))
            m2 = [
                self._init_numpy_weights((self.layers[0], self.layers[0])),
                self._init_numpy_weights((self.layers[0], self.layers[0])),
            ]
            self.GE.Wrz0.set_(torch.tensor(np.hstack(m2), dtype=torch.float32, device=self.GE.Wrz0.device))
            self.GE.Wh0.set_(
                torch.tensor(
                    self._init_numpy_weights((self.layers[0], self.layers[0])),
                    dtype=torch.float32,
                    device=self.GE.Wh0.device,
                )
            )
            self.GE.Bh0.set_(torch.zeros((self.layers[0] * 3,), device=self.GE.Bh0.device))
        for i in range(self.start, len(self.layers)):
            m = [
                self._init_numpy_weights((n_input, self.layers[i])),
                self._init_numpy_weights((n_input, self.layers[i])),
                self._init_numpy_weights((n_input, self.layers[i])),
            ]
            self.G[i].weight_ih.set_(torch.tensor(np.vstack(m), dtype=torch.float32, device=self.G[i].weight_ih.device))
            m2 = [
                self._init_numpy_weights((self.layers[i], self.layers[i])),
                self._init_numpy_weights((self.layers[i], self.layers[i])),
                self._init_numpy_weights((self.layers[i], self.layers[i])),
            ]
            self.G[i].weight_hh.set_(
                torch.tensor(
                    np.vstack(m2),
                    dtype=torch.float32,
                    device=self.G[i].weight_hh.device,
                )
            )
            self.G[i].bias_hh.set_(torch.zeros((self.layers[i] * 3,), device=self.G[i].bias_hh.device))
            self.G[i].bias_ih.set_(torch.zeros((self.layers[i] * 3,), device=self.G[i].bias_ih.device))
        self.Wy.weight.data[: self.n_items].copy_(
            torch.tensor(
                self._init_numpy_weights((self.n_items, self.layers[-1])),
                dtype=torch.float32,
                device=self.Wy.weight.device,
            )
        )
        self.By.weight.data[: self.n_items].copy_(torch.zeros((self.n_items, 1), device=self.By.weight.device))
        if self.Wy.padding_idx is not None:
            self.Wy.weight.data[self.Wy.padding_idx].zero_()
        if self.By.padding_idx is not None:
            self.By.weight.data[self.By.padding_idx].zero_()

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
        out = torch.mm(X, O.T) + B.T
        return out

    def forward(self, X, H, Y, training=False):
        E, O, B = self.embed(X, H, Y)
        if training:
            E = self.DE(E)
        if not (self.constrained_embedding or self.embedding):
            H[0] = E
        Xh = self.hidden_step(E, H, training=training)
        R = self.score_items(Xh, O, B)
        return R


def score(model, layers, device, history_items):
    """Score all items given a flat ``history_items`` list of integers.

    Returns a numpy array of length ``n_items`` (or ``n_items + 1`` if the
    output embedding includes a padding row; in that case the caller is
    responsible for trimming).
    """
    model.eval()
    H = []
    for i in range(len(layers)):
        H.append(torch.zeros((1, layers[i]), dtype=torch.float32, requires_grad=False, device=device))
    O = None
    for iid in history_items:
        O = model.forward(
            torch.tensor([iid], requires_grad=False, device=device),
            H,
            None,
            training=False,
        )
    if O is None:
        return None
    return O.squeeze().cpu().detach().numpy()
