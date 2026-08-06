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
"""PyTorch modules for DiffGRM.

This is an independent implementation of the architecture and equations in
the DiffGRM paper. The official research repository does not currently include
a license, so no source code from that repository is incorporated here.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _MultiheadAttention(nn.Module):
    """Pre-norm attention layout used by the released DiffGRM backbone."""

    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.projection = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, query, key_value=None, key_padding_mask=None):
        batch_size, query_len, _ = query.shape
        query_projection = self.qkv(query)
        q = query_projection[..., : self.d_model]
        if key_value is None:
            k = query_projection[..., self.d_model : 2 * self.d_model]
            v = query_projection[..., 2 * self.d_model :]
        else:
            key_value_projection = self.qkv(key_value)
            k = key_value_projection[..., self.d_model : 2 * self.d_model]
            v = key_value_projection[..., 2 * self.d_model :]

        key_len = k.size(1)
        q = q.view(batch_size, query_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, key_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, key_len, self.n_head, self.head_dim).transpose(1, 2)
        attention = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if key_padding_mask is not None:
            attention = attention.masked_fill(
                key_padding_mask[:, None, None, :], -torch.inf
            )
        attention = self.attention_dropout(attention.softmax(dim=-1))
        hidden = torch.matmul(attention, v)
        hidden = hidden.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.d_model
        )
        return self.residual_dropout(self.projection(hidden))


class _FeedForward(nn.Module):
    def __init__(self, d_model, n_inner, dropout, activation):
        super().__init__()
        self.input = nn.Linear(d_model, n_inner)
        self.output = nn.Linear(n_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, hidden):
        return self.dropout(self.output(self.activation(self.input(hidden))))


class _EncoderBlock(nn.Module):
    def __init__(self, d_model, n_head, n_inner, dropout, activation, norm_eps):
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model, eps=norm_eps)
        self.attention = _MultiheadAttention(d_model, n_head, dropout)
        self.feed_forward_norm = nn.LayerNorm(d_model, eps=norm_eps)
        self.feed_forward = _FeedForward(d_model, n_inner, dropout, activation)

    def forward(self, hidden, padding_mask):
        hidden = hidden + self.attention(
            self.attention_norm(hidden), key_padding_mask=padding_mask
        )
        return hidden + self.feed_forward(self.feed_forward_norm(hidden))


class _DecoderBlock(nn.Module):
    def __init__(self, d_model, n_head, n_inner, dropout, activation, norm_eps):
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(d_model, eps=norm_eps)
        self.self_attention = _MultiheadAttention(d_model, n_head, dropout)
        self.cross_attention_norm = nn.LayerNorm(d_model, eps=norm_eps)
        self.cross_attention = _MultiheadAttention(d_model, n_head, dropout)
        self.feed_forward_norm = nn.LayerNorm(d_model, eps=norm_eps)
        self.feed_forward = _FeedForward(d_model, n_inner, dropout, activation)

    def forward(self, hidden, memory):
        hidden = hidden + self.self_attention(self.self_attention_norm(hidden))
        hidden = hidden + self.cross_attention(
            self.cross_attention_norm(hidden), key_value=memory
        )
        return hidden + self.feed_forward(self.feed_forward_norm(hidden))


class DiffGRMBackbone(nn.Module):
    """Encoder-decoder backbone with on-policy code masking."""

    def __init__(
        self,
        n_digit,
        codebook_size,
        max_len,
        d_model=256,
        encoder_n_layer=1,
        decoder_n_layer=4,
        n_head=4,
        n_inner=1024,
        dropout=0.1,
        activation="gelu",
        layer_norm_eps=1e-5,
        initializer_range=0.02,
        masking_strategy="guided",
        confidence_method="msp",
        random_mask_prob=0.5,
        n_views=None,
        label_smoothing=0.1,
        view_loss_reduction="view_mean",
    ):
        super().__init__()
        self.n_digit = int(n_digit)
        self.codebook_size = int(codebook_size)
        self.max_len = int(max_len)
        self.masking_strategy = masking_strategy
        self.confidence_method = confidence_method
        self.random_mask_prob = float(random_mask_prob)
        self.n_views = self.n_digit if n_views is None else int(n_views)
        self.label_smoothing = float(label_smoothing)
        self.view_loss_reduction = view_loss_reduction

        self.code_embeddings = nn.Parameter(
            torch.empty(self.n_digit, self.codebook_size, d_model)
        )
        self.mask_embeddings = nn.Parameter(torch.empty(self.n_digit, d_model))
        self.item_projection = nn.Sequential(
            nn.Linear(self.n_digit * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.history_positions = nn.Embedding(self.max_len, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.encoder_blocks = nn.ModuleList(
            [
                _EncoderBlock(
                    d_model,
                    n_head,
                    n_inner,
                    dropout,
                    activation,
                    layer_norm_eps,
                )
                for _ in range(encoder_n_layer)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                _DecoderBlock(
                    d_model,
                    n_head,
                    n_inner,
                    dropout,
                    activation,
                    layer_norm_eps,
                )
                for _ in range(decoder_n_layer)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.register_buffer(
            "item_codes", torch.zeros(1, self.n_digit, dtype=torch.long)
        )
        self._reset_parameters(initializer_range)

    def _reset_parameters(self, initializer_range):
        nn.init.normal_(self.code_embeddings, std=initializer_range)
        nn.init.normal_(self.mask_embeddings, std=initializer_range)
        nn.init.normal_(self.history_positions.weight, std=initializer_range)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def set_item_codes(self, sid_table):
        """Set catalog semantic IDs and append one row for history padding."""
        codes = torch.as_tensor(np.asarray(sid_table), dtype=torch.long)
        if codes.ndim != 2 or codes.shape[1] != self.n_digit:
            raise ValueError(
                f"sid_table must have shape (n_items, {self.n_digit})"
            )
        if codes.numel() and (
            codes.min().item() < 0 or codes.max().item() >= self.codebook_size
        ):
            raise ValueError(
                f"semantic-ID digits must be in [0, {self.codebook_size})"
            )
        pad = torch.zeros(1, self.n_digit, dtype=torch.long)
        self.item_codes = torch.cat([codes, pad], dim=0).to(
            self.code_embeddings.device
        )

    def encode_history(self, input_iids, attention_mask):
        """Encode a right-padded item history."""
        item_codes = self.item_codes[input_iids]
        digit_embs = [
            self.code_embeddings[d][item_codes[:, :, d]]
            for d in range(self.n_digit)
        ]
        history = self.item_projection(torch.cat(digit_embs, dim=-1))
        positions = torch.arange(input_iids.size(1), device=input_iids.device)
        history = history + self.history_positions(positions).unsqueeze(0)
        history = self.embedding_dropout(history)
        padding_mask = ~attention_mask.bool()
        for block in self.encoder_blocks:
            history = block(history, padding_mask)
        history = self.final_norm(history)
        history = history * attention_mask.unsqueeze(-1)
        return history, padding_mask

    def decode_logits(self, memory, memory_padding_mask, partial_codes):
        """Predict every digit; ``-1`` denotes a currently masked digit."""
        is_masked = partial_codes < 0
        visible = partial_codes.clamp_min(0)
        digit_states = []
        for d in range(self.n_digit):
            code_state = self.code_embeddings[d][visible[:, d]]
            mask_state = self.mask_embeddings[d].expand_as(code_state)
            digit_states.append(
                torch.where(is_masked[:, d, None], mask_state, code_state)
            )
        hidden = self.embedding_dropout(torch.stack(digit_states, dim=1))
        for block in self.decoder_blocks:
            hidden = block(hidden, memory)
        hidden = self.final_norm(hidden)
        return torch.einsum("bnd,nkd->bnk", hidden, self.code_embeddings)

    def _confidence_order(self, memory, memory_padding_mask, targets):
        batch_size = targets.size(0)
        if self.masking_strategy == "fixed":
            return torch.arange(self.n_digit, device=targets.device).expand(
                batch_size, -1
            )
        if self.masking_strategy == "coherent":
            noise = torch.rand(
                batch_size, self.n_digit, device=targets.device
            )
            return noise.argsort(dim=-1)

        fully_masked = torch.full_like(targets, -1)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            probabilities = self.decode_logits(
                memory, memory_padding_mask, fully_masked
            ).softmax(dim=-1)
            if self.confidence_method == "entropy":
                confidence = (
                    probabilities
                    * probabilities.clamp_min(1e-12).log()
                ).sum(dim=-1)
            else:
                confidence = probabilities.max(dim=-1).values
        if was_training:
            self.train()
        return confidence.argsort(dim=-1, stable=True)

    def training_masks(self, memory, memory_padding_mask, targets):
        """Return nested OCN masks, hardest digit first."""
        if self.masking_strategy == "random":
            masks = (
                torch.rand(
                    targets.size(0),
                    self.n_views,
                    self.n_digit,
                    device=targets.device,
                )
                < self.random_mask_prob
            )
            no_mask = ~masks.any(dim=-1)
            masks[:, :, 0] |= no_mask
            return masks

        order = self._confidence_order(memory, memory_padding_mask, targets)
        rank = torch.empty_like(order)
        rank.scatter_(
            1,
            order,
            torch.arange(self.n_digit, device=targets.device).expand_as(order),
        )
        counts = torch.arange(
            1, self.n_views + 1, device=targets.device
        ).clamp_max(self.n_digit)
        return rank[:, None, :] < counts[None, :, None]

    def forward(self, input_iids, attention_mask, target_iids):
        """Average cross entropy over the masked digits in all OCN views."""
        memory, padding_mask = self.encode_history(input_iids, attention_mask)
        masks = self.training_masks(memory, padding_mask, target_iids)
        batch_size, n_views, _ = masks.shape
        partial = target_iids[:, None, :].expand(-1, n_views, -1).clone()
        partial[masks] = -1

        memory = memory.repeat_interleave(n_views, dim=0)
        padding_mask = padding_mask.repeat_interleave(n_views, dim=0)
        partial = partial.reshape(batch_size * n_views, self.n_digit)
        logits = self.decode_logits(memory, padding_mask, partial)
        labels = (
            target_iids[:, None, :]
            .expand(-1, n_views, -1)
            .reshape(batch_size * n_views, self.n_digit)
        )
        mask = masks.reshape(batch_size * n_views, self.n_digit)
        token_losses = F.cross_entropy(
            logits.reshape(-1, self.codebook_size),
            labels.reshape(-1),
            reduction="none",
            label_smoothing=self.label_smoothing,
        ).reshape(batch_size * n_views, self.n_digit)
        if self.view_loss_reduction == "token_mean":
            return token_losses[mask].mean()
        return (
            (token_losses * mask).sum(dim=-1)
            / mask.sum(dim=-1).clamp_min(1)
        ).mean()


@torch.no_grad()
def cpd_decode_batch(
    model,
    memory,
    memory_padding_mask,
    beam_size,
    catalog_codes=None,
    valid_code_set=None,
    constrained=False,
    digit_order=None,
    greedy_final=False,
    return_diagnostics=False,
):
    """Batched confidence-prioritized decoding over digit/code assignments.

    At every step, every still-masked digit competes globally. ``constrained``
    additionally removes partial assignments that cannot lead to a catalog
    semantic ID. Complete duplicate IDs are collapsed by maximum path score.
    """
    device = memory.device
    batch_size = memory.size(0)
    n_digit = model.n_digit
    codebook_size = model.codebook_size
    beams = torch.full(
        (batch_size, 1, n_digit), -1, dtype=torch.long, device=device
    )
    beam_scores = torch.zeros(batch_size, 1, device=device)
    catalog = None
    if constrained and catalog_codes is not None:
        catalog = torch.as_tensor(
            catalog_codes, dtype=torch.long, device=device
        )

    if digit_order is not None:
        digit_order = tuple(int(d) for d in digit_order)
        if sorted(digit_order) != list(range(n_digit)):
            raise ValueError("digit_order must be a permutation of all digits")

    for step in range(n_digit):
        n_beam = beams.size(1)
        expanded_memory = (
            memory[:, None]
            .expand(-1, n_beam, -1, -1)
            .reshape(
                batch_size * n_beam,
                memory.size(1),
                memory.size(2),
            )
        )
        expanded_padding = (
            memory_padding_mask[:, None]
            .expand(-1, n_beam, -1)
            .reshape(batch_size * n_beam, memory_padding_mask.size(1))
        )
        logits = model.decode_logits(
            expanded_memory,
            expanded_padding,
            beams.reshape(batch_size * n_beam, n_digit),
        ).reshape(batch_size, n_beam, n_digit, codebook_size)
        candidate_scores = logits.log_softmax(dim=-1)
        candidate_scores.masked_fill_(beams[..., None] >= 0, -torch.inf)
        if digit_order is not None:
            allowed_digit = digit_order[step]
            for digit in range(n_digit):
                if digit != allowed_digit:
                    candidate_scores[:, :, digit] = -torch.inf

        if constrained:
            if catalog is None:
                raise ValueError("catalog_codes are required for constrained CPD")
            for row in range(batch_size):
                for branch in range(n_beam):
                    compatible = torch.ones(
                        catalog.size(0), dtype=torch.bool, device=device
                    )
                    for digit in range(n_digit):
                        if beams[row, branch, digit] >= 0:
                            compatible &= (
                                catalog[:, digit]
                                == beams[row, branch, digit]
                            )
                    for digit in range(n_digit):
                        if beams[row, branch, digit] < 0:
                            allowed = catalog[compatible, digit].unique()
                            disallowed = torch.ones(
                                codebook_size,
                                dtype=torch.bool,
                                device=device,
                            )
                            disallowed[allowed] = False
                            candidate_scores[
                                row, branch, digit, disallowed
                            ] = -torch.inf

        candidate_scores = candidate_scores + beam_scores[:, :, None, None]
        if greedy_final and step == n_digit - 1:
            per_parent, per_parent_index = candidate_scores.reshape(
                batch_size, n_beam, -1
            ).max(dim=-1)
            keep = min(int(beam_size), n_beam)
            top_scores, parent = per_parent.topk(keep, dim=-1)
            remainder = per_parent_index.gather(1, parent)
            digit = remainder // codebook_size
            code = remainder % codebook_size
            beams = beams.gather(
                1, parent[:, :, None].expand(-1, -1, n_digit)
            ).clone()
            beams.scatter_(2, digit[:, :, None], code[:, :, None])
            beam_scores = top_scores
            continue

        flat = candidate_scores.reshape(batch_size, -1)
        keep = min(int(beam_size), flat.size(1))
        top_scores, top_indices = flat.topk(keep, dim=-1)
        parent = top_indices // (n_digit * codebook_size)
        remainder = top_indices % (n_digit * codebook_size)
        digit = remainder // codebook_size
        code = remainder % codebook_size
        beams = beams.gather(
            1, parent[:, :, None].expand(-1, -1, n_digit)
        ).clone()
        beams.scatter_(2, digit[:, :, None], code[:, :, None])
        beam_scores = top_scores

    catalog_set = valid_code_set
    if catalog_set is None and catalog_codes is not None:
        catalog_set = {
            tuple(row) for row in np.asarray(catalog_codes).tolist()
        }
    batch_codes, batch_scores, batch_diagnostics = [], [], []
    for row in range(batch_size):
        complete = {}
        for codes, score in zip(
            beams[row].cpu().tolist(), beam_scores[row].cpu().tolist()
        ):
            key = tuple(codes)
            if -1 in key or not np.isfinite(score):
                continue
            complete[key] = max(complete.get(key, -float("inf")), score)
        best = {
            key: score
            for key, score in complete.items()
            if catalog_set is None or key in catalog_set
        }
        ranked = sorted(best.items(), key=lambda pair: pair[1], reverse=True)
        batch_codes.append([codes for codes, _ in ranked])
        batch_scores.append([score for _, score in ranked])
        batch_diagnostics.append(
            {
                "complete_paths": len(beams[row]),
                "unique_complete_sids": len(complete),
                "valid_sids": len(best),
                "invalid_sids": len(complete) - len(best),
                "duplicate_paths": len(beams[row]) - len(complete),
            }
        )

    if return_diagnostics:
        return batch_codes, batch_scores, batch_diagnostics
    return batch_codes, batch_scores


@torch.no_grad()
def cpd_decode(
    model,
    memory,
    memory_padding_mask,
    beam_size,
    catalog_codes=None,
    valid_code_set=None,
    constrained=False,
    digit_order=None,
    greedy_final=False,
    return_diagnostics=False,
):
    """Single-history wrapper around :func:`cpd_decode_batch`."""
    if memory.size(0) != 1:
        raise ValueError("cpd_decode expects one encoded history")
    result = cpd_decode_batch(
        model=model,
        memory=memory,
        memory_padding_mask=memory_padding_mask,
        beam_size=beam_size,
        catalog_codes=catalog_codes,
        valid_code_set=valid_code_set,
        constrained=constrained,
        digit_order=digit_order,
        greedy_final=greedy_final,
        return_diagnostics=return_diagnostics,
    )
    if return_diagnostics:
        codes, scores, diagnostics = result
        return codes[0], scores[0], diagnostics[0]
    codes, scores = result
    return codes[0], scores[0]
