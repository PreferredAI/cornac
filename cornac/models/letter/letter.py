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
"""Neural modules for LETTER (Wang et al., CIKM 2024).

The implementation follows the authors' released ``RQ-VAE`` and
``LETTER-TIGER`` code. In particular, the tokenizer uses constrained k-means
initialization, a Sinkhorn assignment on the last residual codebook, and the
released collaborative/diversity losses. The generator uses a tied T5
vocabulary and predicts EOS after the four semantic-ID tokens.
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def _letter_mlp(input_dim, hidden_dims, output_dim, dropout=0.0):
    """MLP used by the released LETTER tokenizer."""
    dims = [input_dim, *hidden_dims, output_dim]
    layers = []
    for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        layers.extend((nn.Dropout(dropout), nn.Linear(in_dim, out_dim)))
        if index != len(dims) - 2:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)
    return model


def _constrained_kmeans(data, n_clusters, initial=False, n_jobs=10):
    """Run LETTER's constrained k-means and return centers and labels.

    ``k-means-constrained`` is imported here so importing :mod:`cornac` does
    not require the optional tokenizer dependency.
    """
    try:
        from joblib import parallel_backend
        from k_means_constrained import KMeansConstrained
    except ImportError as exc:
        raise ImportError(
            "LETTER requires k-means-constrained. Install the dependencies in "
            "cornac/models/letter/requirements.txt."
        ) from exc

    n_samples = len(data)
    if n_samples < n_clusters:
        raise ValueError(
            "constrained codebook initialization needs at least as many items "
            f"as codes ({n_samples} < {n_clusters})"
        )
    cap = 50 if initial else 10
    size_min = min(n_samples // (n_clusters * 2), cap)
    if size_min < 1:
        raise ValueError(
            "LETTER's constrained k-means needs at least two samples per cluster"
        )
    size_max = min(
        size_min * 4 if initial else n_clusters * 6, n_samples - 1
    )
    clusterer = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=size_min,
        size_max=size_max,
        max_iter=10,
        n_init=10,
        n_jobs=n_jobs,
        verbose=False,
    )
    values = data.detach().cpu().numpy()
    # Joblib otherwise memory-maps arrays above 1 MiB as read-only for its
    # workers, but k-means-constrained's Cython center update needs a writable
    # buffer. Keep the released parallelism while passing normal arrays.
    with parallel_backend("loky", max_nbytes=None):
        clusterer.fit(values)
    centers = torch.as_tensor(
        clusterer.cluster_centers_, dtype=data.dtype, device=data.device
    )
    labels = torch.as_tensor(
        clusterer.labels_, dtype=torch.long, device=data.device
    )
    return centers, labels


def sinkhorn_assignment(distances, epsilon, n_iters):
    """Balanced assignment from the released LETTER Sinkhorn routine."""
    maximum = distances.max()
    minimum = distances.min()
    middle = (maximum + minimum) / 2
    amplitude = maximum - middle + 1e-5
    centered = ((distances - middle) / amplitude).double()

    q = torch.exp(-centered / epsilon)
    batch_size, n_codes = q.shape
    q = q / q.sum()
    for _ in range(n_iters):
        q = q / q.sum(dim=1, keepdim=True)
        q = q / batch_size
        q = q / q.sum(dim=0, keepdim=True)
        q = q / n_codes
    return (q * batch_size).argmax(dim=1)


class LETTERRQVAE(nn.Module):
    """Residual-quantized autoencoder from the official LETTER release."""

    def __init__(
        self,
        input_dim,
        hidden_dims=(2048, 1024, 512, 256, 128, 64),
        latent_dim=32,
        num_levels=4,
        codebook_size=256,
        commitment_weight=0.25,
        n_clusters=10,
        sk_epsilons=None,
        sk_iters=50,
        dropout=0.0,
        kmeans_n_jobs=10,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.commitment_weight = commitment_weight
        self.n_clusters = n_clusters
        self.sk_epsilons = tuple(
            [0.0] * (num_levels - 1) + [0.003]
            if sk_epsilons is None
            else sk_epsilons
        )
        if len(self.sk_epsilons) != num_levels:
            raise ValueError("sk_epsilons must contain one value per codebook")
        self.sk_iters = sk_iters
        self.kmeans_n_jobs = kmeans_n_jobs
        self.encoder = _letter_mlp(input_dim, hidden_dims, latent_dim, dropout)
        # nn.Embedding initializes before the official code zeros each
        # k-means-initialized table. Consume the same RNG draws so the decoder
        # and subsequent shuffled loader start from the released seed state.
        codebooks = torch.empty(num_levels, codebook_size, latent_dim)
        nn.init.normal_(codebooks)
        self.codebooks = nn.Parameter(codebooks.zero_())
        self.decoder = _letter_mlp(
            latent_dim, tuple(reversed(hidden_dims)), input_dim, dropout
        )
        self._div_labels = None

    @staticmethod
    def _distances(x, codebook):
        return (
            x.square().sum(dim=1, keepdim=True)
            + codebook.square().sum(dim=1).unsqueeze(0)
            - 2 * x @ codebook.t()
        )

    def _assign(self, distances, level, use_sinkhorn):
        epsilon = self.sk_epsilons[level]
        if use_sinkhorn and epsilon > 0:
            return sinkhorn_assignment(distances, epsilon, self.sk_iters)
        return distances.argmin(dim=1)

    @torch.no_grad()
    def initialize_codebooks(self, x):
        """Initialize every residual level on the complete item collection."""
        residual = self.encoder(x)
        for level in range(self.num_levels):
            centers, _ = _constrained_kmeans(
                residual,
                self.codebook_size,
                initial=True,
                n_jobs=self.kmeans_n_jobs,
            )
            self.codebooks[level].copy_(centers)
            distances = self._distances(residual, centers)
            ids = self._assign(distances, level, use_sinkhorn=True)
            residual = residual - centers[ids]

    @torch.no_grad()
    def update_diversity_clusters(self):
        """Refresh the ten constrained codebook groups once per epoch."""
        self._div_labels = [
            _constrained_kmeans(
                self.codebooks[level],
                self.n_clusters,
                initial=False,
                n_jobs=self.kmeans_n_jobs,
            )[1]
            for level in range(self.num_levels)
        ]

    def _diversity_loss(self, codebook, selected, ids, level):
        if self._div_labels is None:
            return codebook.new_zeros(())
        labels = self._div_labels[level].tolist()
        groups = {
            group: [index for index, label in enumerate(labels) if label == group]
            for group in range(self.n_clusters)
        }
        valid_rows = []
        positives = []
        for row, code in enumerate(ids.tolist()):
            choices = groups[labels[code]]
            # The released 256-code/10-cluster constraints guarantee siblings.
            # Keep reduced smoke-test codebooks finite when that is impossible.
            if len(choices) < 2:
                continue
            positive = random.choice(choices)
            while positive == code:
                positive = random.choice(choices)
            valid_rows.append(row)
            positives.append(positive)
        if not valid_rows:
            return codebook.new_zeros(())
        valid_rows = torch.as_tensor(
            valid_rows, dtype=torch.long, device=ids.device
        )
        targets = torch.as_tensor(positives, dtype=torch.long, device=ids.device)
        selected_ids = ids[valid_rows]
        logits = selected[valid_rows] @ codebook.t()
        logits = logits.clone()
        logits.scatter_(1, selected_ids[:, None], -1e12)
        return F.cross_entropy(logits, targets)

    def _quantize(self, z, use_sinkhorn=True):
        all_ids = []
        quantized = torch.zeros_like(z)
        residual = z
        quant_loss = z.new_zeros(())
        diversity_loss = z.new_zeros(())
        for level in range(self.num_levels):
            codebook = self.codebooks[level]
            distances = self._distances(residual, codebook)
            ids = self._assign(distances, level, use_sinkhorn)
            selected = codebook[ids]
            diversity_loss = diversity_loss + self._diversity_loss(
                codebook, selected, ids, level
            )
            quant_loss = quant_loss + F.mse_loss(
                selected, residual.detach()
            ) + self.commitment_weight * F.mse_loss(
                selected.detach(), residual
            )

            # Match VectorQuantizer.forward: straight-through at every level,
            # then subtract that value before quantizing the next residual.
            selected_st = residual + (selected - residual).detach()
            residual = residual - selected_st
            quantized = quantized + selected_st
            all_ids.append(ids)
        scale = float(self.num_levels)
        return (
            torch.stack(all_ids, dim=1),
            quantized,
            quant_loss / scale,
            diversity_loss / scale,
        )

    @staticmethod
    def _cf_loss(quantized, cf_batch):
        labels = torch.arange(quantized.size(0), device=quantized.device)
        return F.cross_entropy(quantized @ cf_batch.t(), labels)

    def forward(self, x, cf_batch=None):
        """Return IDs, reconstruction, and all released loss components."""
        z = self.encoder(x)
        ids, quantized, loss_rq, loss_div = self._quantize(
            z, use_sinkhorn=True
        )
        reconstruction = self.decoder(quantized)
        loss_recon = F.mse_loss(reconstruction, x)
        if cf_batch is None:
            loss_cf = x.new_zeros(())
        else:
            if cf_batch.size(1) != self.latent_dim:
                raise ValueError(
                    "official LETTER requires CF embeddings to match the "
                    f"{self.latent_dim}-d tokenizer latent (got {cf_batch.size(1)})"
                )
            loss_cf = self._cf_loss(quantized, cf_batch)
        return ids, reconstruction, loss_recon, loss_rq, loss_cf, loss_div

    @torch.no_grad()
    def encode(self, x, use_sinkhorn=False):
        ids, _, _, _ = self._quantize(
            self.encoder(x), use_sinkhorn=use_sinkhorn
        )
        return ids

    @torch.no_grad()
    def resolve_collisions(self, x, codes, max_iters=20):
        """Reassign colliding groups with last-level Sinkhorn, as released."""
        resolved = codes.clone()
        for _ in range(max_iters):
            groups = {}
            for item, row in enumerate(resolved.tolist()):
                groups.setdefault(tuple(row), []).append(item)
            collisions = [items for items in groups.values() if len(items) > 1]
            if not collisions:
                break
            for items in collisions:
                item_ids = torch.as_tensor(items, device=x.device)
                resolved[item_ids] = self.encode(
                    x[item_ids], use_sinkhorn=True
                )
        return resolved


class LETTERSeq2Seq(nn.Module):
    """T5 generator matching the released LETTER-TIGER parameterization."""

    eos_token_id = 1
    pad_token_id = 0

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
        base_vocab_size=32100,
        temperature=1.0,
        code_values=None,
    ):
        super().__init__()
        from transformers import T5Config, T5ForConditionalGeneration

        self.level_sizes = [int(size) for size in level_sizes]
        self.num_levels = len(self.level_sizes)
        self.temperature = temperature
        self.base_vocab_size = base_vocab_size
        if code_values is None:
            code_values = [range(size) for size in self.level_sizes]
        code_values = [list(map(int, values)) for values in code_values]
        vocab_size = base_vocab_size + sum(map(len, code_values))
        config = T5Config(
            vocab_size=vocab_size,
            d_model=d_model,
            d_ff=d_ff,
            d_kv=d_kv,
            num_heads=num_heads,
            num_layers=num_enc_layers,
            num_decoder_layers=num_dec_layers,
            dropout_rate=dropout,
            decoder_start_token_id=self.pad_token_id,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            use_cache=False,
        )
        self.t5 = T5ForConditionalGeneration(config)

        # T5Tokenizer.add_tokens receives a sorted set in the official code.
        # Reproduce its lexicographic code-token order within each level.
        token_ids = []
        offset = base_vocab_size
        for level, (size, values) in enumerate(
            zip(self.level_sizes, code_values)
        ):
            prefix = chr(ord("a") + level)
            order = sorted(values, key=lambda code: f"<{prefix}_{code}>")
            inverse = torch.zeros(size, dtype=torch.long)
            for rank, code in enumerate(order):
                inverse[code] = offset + rank
            token_ids.append(inverse)
            offset += len(values)
        self.register_buffer("code_token_ids", torch.stack(token_ids))

    def semantic_tokens(self, codes):
        levels = torch.arange(codes.size(-1), device=codes.device)
        return self.code_token_ids[levels, codes]

    def forward(self, enc_tokens, enc_mask, target_sids):
        target_tokens = self.semantic_tokens(target_sids)
        eos = target_tokens.new_full((target_tokens.size(0), 1), self.eos_token_id)
        labels = torch.cat((target_tokens, eos), dim=1)
        decoder_inputs = torch.cat(
            (labels.new_full((labels.size(0), 1), self.pad_token_id), labels[:, :-1]),
            dim=1,
        )
        output = self.t5(
            input_ids=enc_tokens,
            attention_mask=enc_mask,
            decoder_input_ids=decoder_inputs,
            use_cache=False,
            return_dict=True,
        )
        logits = output.logits / self.temperature
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

    def _decoder_logits(self, decoder_inputs, enc_out, enc_mask):
        hidden = self.t5.decoder(
            input_ids=decoder_inputs,
            encoder_hidden_states=enc_out,
            encoder_attention_mask=enc_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        if self.t5.config.tie_word_embeddings:
            hidden = hidden * (self.t5.model_dim**-0.5)
        return self.t5.lm_head(hidden[:, -1]) / self.temperature

    @torch.no_grad()
    def generate_beam(self, enc_tokens, enc_mask, n_beams, prefix_children):
        enc_out = self.t5.encoder(
            input_ids=enc_tokens, attention_mask=enc_mask, return_dict=True
        ).last_hidden_state
        beams = [()]
        beam_scores = enc_out.new_zeros(1)
        for level, size in enumerate(self.level_sizes):
            n_current = len(beams)
            decoder_inputs = torch.full(
                (n_current, level + 1),
                self.pad_token_id,
                dtype=torch.long,
                device=enc_out.device,
            )
            if level:
                previous = torch.as_tensor(beams, device=enc_out.device)
                decoder_inputs[:, 1:] = self.semantic_tokens(previous)
            logits = self._decoder_logits(
                decoder_inputs,
                enc_out.expand(n_current, -1, -1),
                enc_mask.expand(n_current, -1),
            )
            full_log_probs = F.log_softmax(logits, dim=-1)
            code_ids = self.code_token_ids[level]
            log_probs = full_log_probs.index_select(1, code_ids)
            allowed = torch.full_like(log_probs, float("-inf"))
            for row, beam in enumerate(beams):
                allowed[row, prefix_children[level][beam]] = 0.0
            totals = (beam_scores[:, None] + log_probs + allowed).flatten()
            width = min(n_beams, int(torch.isfinite(totals).sum()))
            top = totals.topk(width)
            beams = [
                beams[index // size] + (index % size,)
                for index in top.indices.tolist()
            ]
            beam_scores = top.values

        previous = torch.as_tensor(beams, device=enc_out.device)
        decoder_inputs = torch.cat(
            (
                previous.new_full((len(beams), 1), self.pad_token_id),
                self.semantic_tokens(previous),
            ),
            dim=1,
        )
        eos_logits = self._decoder_logits(
            decoder_inputs,
            enc_out.expand(len(beams), -1, -1),
            enc_mask.expand(len(beams), -1),
        )
        beam_scores = beam_scores + F.log_softmax(eos_logits, dim=-1)[
            :, self.eos_token_id
        ]
        order = beam_scores.argsort(descending=True)
        return [beams[index] for index in order.tolist()], beam_scores[order].cpu().numpy()

    @torch.no_grad()
    def score_all_items(self, enc_tokens, enc_mask, sid_table, batch_size):
        enc_out = self.t5.encoder(
            input_ids=enc_tokens, attention_mask=enc_mask, return_dict=True
        ).last_hidden_state
        scores = enc_out.new_empty(sid_table.size(0))
        for start in range(0, sid_table.size(0), batch_size):
            codes = sid_table[start : start + batch_size]
            n_items = codes.size(0)
            target_tokens = self.semantic_tokens(codes)
            decoder_inputs = torch.cat(
                (
                    target_tokens.new_full((n_items, 1), self.pad_token_id),
                    target_tokens,
                ),
                dim=1,
            )
            hidden = self.t5.decoder(
                input_ids=decoder_inputs,
                encoder_hidden_states=enc_out.expand(n_items, -1, -1),
                encoder_attention_mask=enc_mask.expand(n_items, -1),
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
            if self.t5.config.tie_word_embeddings:
                hidden = hidden * (self.t5.model_dim**-0.5)
            item_scores = hidden.new_zeros(n_items)
            labels = torch.cat(
                (
                    target_tokens,
                    target_tokens.new_full((n_items, 1), self.eos_token_id),
                ),
                dim=1,
            )
            for position in range(self.num_levels + 1):
                logits = self.t5.lm_head(hidden[:, position]) / self.temperature
                item_scores += F.log_softmax(logits, dim=-1).gather(
                    1, labels[:, position : position + 1]
                ).squeeze(1)
            scores[start : start + n_items] = item_scores
        return scores.cpu().numpy()
