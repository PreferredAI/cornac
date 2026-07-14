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
"""Neural modules for TIGER (Rajput et al., 2023).

``RQVAE`` quantizes item content embeddings into hierarchical semantic IDs;
``TIGERSeq2Seq`` is a small T5-style encoder-decoder trained to generate the
next item's semantic ID from the session history's semantic-ID tokens.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5Stack


def _mlp(input_dim, hidden_dims, output_dim):
    dims = [input_dim, *hidden_dims, output_dim]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


@torch.no_grad()
def _kmeans(x, k, n_iters=10):
    """K-means++ seeding followed by Lloyd iterations. Returns (k, dim) centroids."""
    n = x.size(0)
    if n < k:
        idx = torch.randint(0, n, (k,), device=x.device)
        return x[idx] + 1e-4 * torch.randn(k, x.size(1), device=x.device, dtype=x.dtype)
    centroids = torch.empty(k, x.size(1), device=x.device, dtype=x.dtype)
    centroids[0] = x[torch.randint(0, n, (1,), device=x.device)]
    d2 = torch.cdist(x, centroids[:1]).squeeze(1).pow_(2)
    for i in range(1, k):
        centroids[i] = x[torch.multinomial(d2 + 1e-12, 1)]
        d2 = torch.minimum(d2, torch.cdist(x, centroids[i : i + 1]).squeeze(1).pow_(2))
    for _ in range(n_iters):
        ids = torch.cdist(x, centroids).argmin(dim=1)
        sums = torch.zeros_like(centroids)
        sums.index_add_(0, ids, x)
        counts = torch.bincount(ids, minlength=k)
        empty = counts == 0
        centroids = sums / counts.clamp_min(1).unsqueeze(1)
        if empty.any():
            centroids[empty] = x[torch.randint(0, n, (int(empty.sum()),), device=x.device)]
    return centroids


class RQVAE(nn.Module):
    """Residual-Quantized VAE over item content embeddings (TIGER paper Sec. 4.1).

    Encoder MLP -> ``num_levels`` residual codebooks (straight-through
    estimator, VQ-VAE codebook + commitment losses) -> mirror decoder MLP
    with MSE reconstruction loss.
    """

    def __init__(
        self,
        input_dim,
        hidden_dims=(512, 256, 128),
        latent_dim=32,
        num_levels=3,
        codebook_size=256,
        beta=0.25,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.beta = beta
        self.encoder = _mlp(input_dim, hidden_dims, latent_dim)
        self.decoder = _mlp(latent_dim, tuple(reversed(hidden_dims)), input_dim)
        self.codebooks = nn.Parameter(
            torch.randn(num_levels, codebook_size, latent_dim) * 0.01
        )

    def _quantize(self, z):
        ids = []
        q = torch.zeros_like(z)
        loss_rq = z.new_zeros(())
        r = z
        for level in range(self.num_levels):
            level_ids = torch.cdist(r, self.codebooks[level]).argmin(dim=1)
            e = self.codebooks[level][level_ids]
            loss_rq = loss_rq + F.mse_loss(e, r.detach()) + self.beta * F.mse_loss(r, e.detach())
            ids.append(level_ids)
            q = q + e
            # detach so each codebook is only updated by its own level's loss
            r = r - e.detach()
        return torch.stack(ids, dim=1), q, loss_rq

    def forward(self, x):
        z = self.encoder(x)
        ids, q, loss_rq = self._quantize(z)
        z_q = z + (q - z).detach()  # straight-through estimator
        x_hat = self.decoder(z_q)
        loss_recon = F.mse_loss(x_hat, x)
        return ids, x_hat, loss_recon, loss_rq

    @torch.no_grad()
    def encode(self, x):
        """Assign semantic-ID codes. Returns LongTensor of shape (B, num_levels)."""
        ids, _, _ = self._quantize(self.encoder(x))
        return ids

    @torch.no_grad()
    def kmeans_init_codebooks(self, x, n_iters=10):
        """Initialize each codebook with k-means on the encoder residuals,
        level by level, to prevent early codebook collapse."""
        r = self.encoder(x)
        for level in range(self.num_levels):
            centroids = _kmeans(r, self.codebook_size, n_iters=n_iters)
            self.codebooks[level].copy_(centroids)
            ids = torch.cdist(r, centroids).argmin(dim=1)
            r = r - centroids[ids]

    @torch.no_grad()
    def restart_dead_codes(self, x, used):
        """Reassign codes unused during the last epoch (``used``: bool (L, K))
        to random encoder residuals from ``x``. Returns #codes restarted."""
        r = self.encoder(x)
        n_restarted = 0
        for level in range(self.num_levels):
            dead = ~used[level]
            if dead.any():
                idx = torch.randint(0, r.size(0), (int(dead.sum()),), device=r.device)
                self.codebooks[level][dead] = r[idx]
                n_restarted += int(dead.sum())
            ids = torch.cdist(r, self.codebooks[level]).argmin(dim=1)
            r = r - self.codebooks[level][ids]
        return n_restarted


class TIGERSeq2Seq(nn.Module):
    """T5-style encoder-decoder over semantic-ID tokens (TIGER paper Fig. 2.b).

    A single embedding table holds all levels' tokens with cumulative offsets
    (index 0 reserved for padding) and is shared between encoder and decoder;
    each level has its own output head. A learned BOS embedding prompts the
    decoder.
    """

    def __init__(
        self,
        level_sizes,
        d_model=128,
        d_ff=1024,
        num_heads=6,
        d_kv=64,
        num_enc_layers=4,
        num_dec_layers=4,
        dropout=0.1,
    ):
        super().__init__()
        self.level_sizes = [int(s) for s in level_sizes]
        self.num_levels = len(self.level_sizes)
        # token 0 = padding; level l tokens occupy [offsets[l], offsets[l] + level_sizes[l])
        offsets = np.concatenate(([1], 1 + np.cumsum(self.level_sizes[:-1])))
        self.register_buffer("offsets", torch.as_tensor(offsets, dtype=torch.long))
        self.token_emb = nn.Embedding(1 + sum(self.level_sizes), d_model, padding_idx=0)

        cfg = T5Config(
            vocab_size=1,  # internal embed_tokens is unused; we always pass inputs_embeds
            d_model=d_model,
            d_ff=d_ff,
            d_kv=d_kv,
            num_heads=num_heads,
            num_layers=num_enc_layers,
            dropout_rate=dropout,
            is_encoder_decoder=False,
            use_cache=False,
        )
        enc_cfg = copy.deepcopy(cfg)
        enc_cfg.is_decoder = False
        dec_cfg = copy.deepcopy(cfg)
        dec_cfg.is_decoder = True
        dec_cfg.num_layers = num_dec_layers
        # T5Stack is semi-internal; if it breaks in a future transformers
        # release, fall back to T5ForConditionalGeneration(cfg).encoder/.decoder
        self.encoder = T5Stack(enc_cfg)
        self.decoder = T5Stack(dec_cfg)

        self.bos = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.heads = nn.ModuleList(
            [nn.Linear(d_model, s, bias=False) for s in self.level_sizes]
        )

    def encode_history(self, enc_tokens, enc_mask):
        """enc_tokens: (B, S) offset tokens with 0 = pad; enc_mask: (B, S) float."""
        return self.encoder(
            inputs_embeds=self.token_emb(enc_tokens), attention_mask=enc_mask
        ).last_hidden_state

    def _decode(self, dec_sids, enc_out, enc_mask):
        """Decoder pass on BOS + un-offset level tokens ``dec_sids`` (B, t) or None.
        Returns hidden states (B, t+1, d_model)."""
        inputs = self.bos.expand(enc_out.size(0), -1, -1)
        if dec_sids is not None and dec_sids.size(1) > 0:
            emb = self.token_emb(dec_sids + self.offsets[: dec_sids.size(1)])
            inputs = torch.cat([inputs, emb], dim=1)
        return self.decoder(
            inputs_embeds=inputs,
            encoder_hidden_states=enc_out,
            encoder_attention_mask=enc_mask,
            use_cache=False,
        ).last_hidden_state

    def forward(self, enc_tokens, enc_mask, target_sids):
        """Teacher-forced training loss: sum over levels of cross-entropy.
        target_sids: (B, num_levels) un-offset codes."""
        enc_out = self.encode_history(enc_tokens, enc_mask)
        h = self._decode(target_sids[:, :-1], enc_out, enc_mask)  # (B, num_levels, d)
        loss = h.new_zeros(())
        for level in range(self.num_levels):
            loss = loss + F.cross_entropy(
                self.heads[level](h[:, level]), target_sids[:, level]
            )
        return loss

    @torch.no_grad()
    def generate_beam(self, enc_tokens, enc_mask, n_beams, prefix_children):
        """Constrained beam search for a single history (batch size 1).

        At each level, candidate tokens are masked to ``prefix_children[level][prefix]``
        so every surviving beam is a valid item's semantic ID. The decoder is
        recomputed per level (length <= num_levels + 1), avoiding KV-cache
        bookkeeping. Returns (list of sid tuples, log-probs) sorted descending.
        """
        enc_out = self.encode_history(enc_tokens, enc_mask)  # (1, S, d)
        beams = [()]
        beam_lp = enc_out.new_zeros(1)
        for level, size in enumerate(self.level_sizes):
            n_b = len(beams)
            dec_sids = (
                torch.tensor(beams, dtype=torch.long, device=enc_out.device)
                if level > 0
                else None
            )
            h = self._decode(
                dec_sids, enc_out.expand(n_b, -1, -1), enc_mask.expand(n_b, -1)
            )
            logp = F.log_softmax(self.heads[level](h[:, -1]), dim=-1)  # (n_b, size)
            allowed = torch.full_like(logp, float("-inf"))
            for i, beam in enumerate(beams):
                allowed[i, prefix_children[level][beam]] = 0.0
            total = (beam_lp.unsqueeze(1) + logp + allowed).flatten()
            k = min(n_beams, int(torch.isfinite(total).sum()))
            top = total.topk(k)
            beams = [beams[j // size] + (j % size,) for j in top.indices.tolist()]
            beam_lp = top.values
        return beams, beam_lp.cpu().numpy()

    @torch.no_grad()
    def score_all_items(self, enc_tokens, enc_mask, sid_table, batch_size):
        """Exact teacher-forced log-likelihood of every item's semantic ID for a
        single history (batch size 1). sid_table: (N, num_levels) un-offset codes.
        Returns a 1-D numpy array of length N."""
        enc_out = self.encode_history(enc_tokens, enc_mask)  # (1, S, d)
        num_items = sid_table.size(0)
        scores = enc_out.new_empty(num_items)
        for start in range(0, num_items, batch_size):
            target = sid_table[start : start + batch_size]
            n = target.size(0)
            h = self._decode(
                target[:, :-1], enc_out.expand(n, -1, -1), enc_mask.expand(n, -1)
            )
            s = h.new_zeros(n)
            for level in range(self.num_levels):
                logp = F.log_softmax(self.heads[level](h[:, level]), dim=-1)
                s = s + logp.gather(1, target[:, level : level + 1]).squeeze(1)
            scores[start : start + n] = s
        return scores.cpu().numpy()
