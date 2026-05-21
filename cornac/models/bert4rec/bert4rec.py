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

import torch
import torch.nn as nn


class BERT4RecModel(nn.Module):
    """BERT4Rec: bidirectional transformer encoder for sequence rec.

    Uses HuggingFace's :class:`~transformers.BertModel` as the backbone. The
    sequence-final hidden state is used to score candidate items via the dot
    product with their output embeddings (separate "head" linear or tied to
    ``item_emb``).

    Returns ``(B, B+N)`` score matrices when called as
    ``forward(_, hist_iids, out_iids, return_hidden=False)``.
    """

    def __init__(
        self,
        item_num,
        embedding_dim=100,
        maxlen=20,
        n_layers=2,
        n_heads=1,
        dropout=0.1,
        pad_idx=-1,
        tie_weights=False,
        init_std=0.02,
        device="cpu",
    ):
        super().__init__()
        from transformers.models.bert import BertConfig, BertModel

        self.item_num = item_num
        self.pad_idx = pad_idx if pad_idx >= 0 else item_num
        self.maxlen = maxlen
        self.dev = device
        self.init_std = init_std
        self.tie_weights = tie_weights

        config = BertConfig(
            vocab_size=item_num + 1,
            hidden_size=embedding_dim,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            intermediate_size=embedding_dim * 4,
            hidden_act="gelu",
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=maxlen + 1,
            initializer_range=init_std,
            pad_token_id=self.pad_idx,
            layer_norm_eps=1e-12,
            use_cache=False,
        )

        self.item_emb = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=embedding_dim,
            padding_idx=self.pad_idx,
        )
        self.transformer_model = BertModel(config)
        self.item_biases = nn.Embedding(item_num + 1, 1, padding_idx=self.pad_idx)
        self._init_weights()
        self.to(device)

    def _init_weights(self):
        self.item_emb.weight.data.normal_(mean=0.0, std=self.init_std)
        self.item_emb.weight.data[self.pad_idx].zero_()
        self.item_biases.weight.data.zero_()

    def _encode(self, hist_iids):
        attention_mask = (hist_iids != self.pad_idx).long()
        embeds = self.item_emb(hist_iids)
        out = self.transformer_model(
            inputs_embeds=embeds, attention_mask=attention_mask
        )
        return out.last_hidden_state[:, -1, :]

    def forward(self, user_ids, hist_iids, out_iids, return_hidden=False):
        hidden = self._encode(hist_iids)
        item_e = self.item_emb(out_iids)
        bias = self.item_biases(out_iids)
        if return_hidden:
            return hidden, item_e, bias
        scores = torch.mm(hidden, item_e.T) + bias.T
        return scores

    @torch.no_grad()
    def predict(self, user_ids, log_seqs, item_indices=None):
        if item_indices is None:
            item_indices = torch.arange(self.item_num, device=self.dev)
        else:
            item_indices = torch.as_tensor(
                item_indices, dtype=torch.long, device=self.dev
            )
        if not isinstance(log_seqs, torch.Tensor):
            log_seqs = torch.as_tensor(log_seqs, dtype=torch.long, device=self.dev)
        hidden = self._encode(log_seqs)
        item_e = self.item_emb(item_indices)
        bias = self.item_biases(item_indices)
        scores = torch.mm(hidden, item_e.T) + bias.T
        return scores.squeeze().detach().cpu().numpy()
