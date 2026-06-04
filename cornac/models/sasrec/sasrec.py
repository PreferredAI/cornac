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
import torch.nn as nn


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        dropout1 = nn.Dropout(p=dropout_rate)
        relu = nn.ReLU()
        conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        dropout2 = nn.Dropout(p=dropout_rate)
        self.process = nn.Sequential(conv1, dropout1, relu, conv2, dropout2)

    def forward(self, inputs):
        outputs = self.process(inputs.transpose(-1, -2))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class SASRecModel(nn.Module):
    """SASRec self-attention model (Kang & McAuley, 2018).

    Operates on sequences of past item ids, returning the last-position
    representation. Item ids are integers in ``[0, item_num)`` with
    ``item_num`` being used as the padding index.

    The model produces a ``(B, B+N)`` score matrix when called as
    ``forward(_, hist_iids, out_iids, return_hidden=False)`` where
    ``out_iids`` contains the ``B`` in-batch positives followed by ``N``
    shared negatives, matching the contract expected by the loss functions
    in :mod:`cornac.models.seq_utils`.
    """

    def __init__(
        self,
        item_num,
        embedding_dim=100,
        maxlen=20,
        n_layers=2,
        n_heads=1,
        use_pos_emb=True,
        use_biases=True,
        dropout=0.2,
        pad_idx=-1,
        init_std=0.02,
        device="cpu",
    ):
        super(SASRecModel, self).__init__()
        self.item_num = item_num
        self.pad_idx = pad_idx if pad_idx >= 0 else item_num
        self.maxlen = maxlen
        self.dev = device
        self.init_std = init_std

        # +1 row for the padding entry at pad_idx
        self.item_emb = nn.Embedding(self.item_num + 1, embedding_dim, padding_idx=self.pad_idx)
        if use_pos_emb:
            self.pos_emb = nn.Embedding(maxlen + 1, embedding_dim)
        if use_biases:
            self.item_biases = nn.Embedding(self.item_num + 1, 1, padding_idx=self.pad_idx)
        self.emb_dropout = nn.Dropout(p=dropout)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(embedding_dim, eps=1e-8)

        for _ in range(n_layers):
            self.attention_layernorms.append(nn.LayerNorm(embedding_dim, eps=1e-8))
            self.attention_layers.append(nn.MultiheadAttention(embedding_dim, n_heads, dropout))
            self.forward_layernorms.append(nn.LayerNorm(embedding_dim, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(embedding_dim, dropout))

        self.apply(self._init_weights)
        self.to(device)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _score_items(self, hidden, cand_items, biases=None):
        scores = torch.mm(hidden, cand_items.T)
        if biases is not None:
            return scores + biases.T
        return scores

    def _encode(self, hist_iids):
        # hist_iids: (B, T)
        seqs = self.item_emb(hist_iids)
        seqs = seqs * (self.item_emb.embedding_dim**0.5)
        positions = np.tile(np.arange(hist_iids.shape[1]), [hist_iids.shape[0], 1])
        if hasattr(self, "pos_emb"):
            seqs = seqs + self.pos_emb(torch.tensor(positions, dtype=torch.long, device=seqs.device))
        seqs = self.emb_dropout(seqs)

        pad_mask = hist_iids == self.pad_idx  # (B, T)
        seqs = seqs.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        B, tl, _ = seqs.shape
        future = torch.triu(torch.ones(tl, tl, dtype=torch.bool, device=seqs.device), diagonal=1)
        block = future.unsqueeze(0) | pad_mask.unsqueeze(1)  # (B, T, T)
        block = block & ~torch.eye(tl, dtype=torch.bool, device=seqs.device)
        attn_mask = torch.zeros(B, tl, tl, dtype=seqs.dtype, device=seqs.device).masked_fill(block, float("-inf"))
        n_heads = self.attention_layers[0].num_heads
        attn_mask = attn_mask.repeat_interleave(n_heads, dim=0)  # (B*n_heads, T, T)

        for i in range(len(self.attention_layers)):
            seqs_t = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs_t)
            mha_out, _ = self.attention_layers[i](Q, seqs_t, seqs_t, attn_mask=attn_mask)
            seqs_t = Q + mha_out
            seqs = torch.transpose(seqs_t, 0, 1)
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs = seqs.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        log_feats = self.last_layernorm(seqs)
        return log_feats[:, -1, :]

    def forward(self, user_ids, hist_iids, out_iids, return_hidden=False):
        hidden = self._encode(hist_iids)
        item_emb = self.item_emb(out_iids)
        biases = self.item_biases(out_iids) if hasattr(self, "item_biases") else None
        if return_hidden:
            return hidden, item_emb, biases
        return self._score_items(hidden, item_emb, biases)

    @torch.no_grad()
    def predict(self, user_ids, log_seqs, item_indices=None):
        """Score all real items for a single padded sequence.

        Returns a 1-D numpy array of size ``item_num`` (padding column is
        already stripped).
        """
        if item_indices is None:
            item_indices = torch.arange(self.item_num, device=self.dev)
        else:
            item_indices = torch.as_tensor(item_indices, dtype=torch.long, device=self.dev)
        if not isinstance(log_seqs, torch.Tensor):
            log_seqs = torch.as_tensor(log_seqs, dtype=torch.long, device=self.dev)
        hidden = self._encode(log_seqs)
        item_emb = self.item_emb(item_indices)
        biases = self.item_biases(item_indices) if hasattr(self, "item_biases") else None
        scores = self._score_items(hidden, item_emb, biases)
        return scores.squeeze().detach().cpu().numpy()
