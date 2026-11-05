# Copyright 2026 The Cornac Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for details.

"""DIGER computation; imported only when the optional backend is used.

The public model module must not import this module at package import time.
This implementation uses PyTorch and the public Transformers T5 implementation;
the generator is initialized locally.
"""

from collections import Counter
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers import T5Config, T5Model


@dataclass
class Quantization:
    latent: torch.Tensor
    quantized: torch.Tensor
    probabilities: torch.Tensor
    soft_probabilities: torch.Tensor
    indices: torch.Tensor
    vq_per_item: torch.Tensor
    hot_masks: torch.Tensor


def hard_forward(soft, indices):
    """Use integer assignments in forward, differentiable mixtures in backward."""
    hard = F.one_hot(indices, soft.shape[-1]).to(soft.dtype)
    return hard - soft.detach() + soft


def balanced_assignments(distances, epsilon=0.003, iterations=50):
    """Full-catalog balanced assignments on centered squared distances."""
    midpoint = (distances.max() + distances.min()) / 2
    amplitude = distances.max() - midpoint + 1e-5
    centered = ((distances - midpoint) / amplitude).double()
    mass = torch.exp(-centered / epsilon)
    mass = mass / mass.sum()
    rows, columns = mass.shape
    for _ in range(iterations):
        mass = mass / mass.sum(dim=1, keepdim=True) / rows
        mass = mass / mass.sum(dim=0, keepdim=True) / columns
    if not torch.isfinite(mass).all():
        raise RuntimeError("Non-finite balanced code assignments")
    return mass.argmax(dim=-1)


class SemanticTokenizer(nn.Module):
    """Encoder-only residual quantizer with frequency-gated Gumbel assignments."""

    def __init__(self, input_dim, hidden_dims=(2048, 1024, 512), latent_dim=256,
                 levels=3, codebook_size=256, beta=0.25, temperature=2.0,
                 usage_momentum=0.99, hot_ratio=1.5, noise_scale=1.0,
                 freeze_encoder=True):
        super().__init__()
        widths = (input_dim,) + tuple(hidden_dims) + (latent_dim,)
        modules = []
        for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
            linear = nn.Linear(left, right)
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            modules.append(linear)
            if index < len(widths) - 2:
                modules.append(nn.ReLU())
        self.encoder = nn.Sequential(*modules)
        self.encoder.requires_grad_(not freeze_encoder)
        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, latent_dim) for _ in range(levels)
        ])
        for book in self.codebooks:
            nn.init.uniform_(book.weight, -1 / codebook_size, 1 / codebook_size)
        self.register_buffer("usage", torch.full((levels, codebook_size), 1 / codebook_size))
        self.beta = beta
        self.temperature = temperature
        self.usage_momentum = usage_momentum
        self.hot_ratio = hot_ratio
        self.noise_scale = noise_scale

    @staticmethod
    def distances(residual, weight):
        return (residual.square().sum(-1, keepdim=True)
                + weight.square().sum(-1).unsqueeze(0)
                - 2 * residual @ weight.t())

    def forward(self, features, mode="frequency", noises=None):
        if mode not in ("frequency", "plain", "fixed"):
            raise ValueError("mode must be frequency, plain, or fixed")
        latent = self.encoder(features)
        residual = latent
        vectors, mixtures, softs, indices, losses, masks = [], [], [], [], [], []
        for level, book in enumerate(self.codebooks):
            logits = -self.distances(residual, book.weight)
            nearest = logits.argmax(-1)
            if mode == "frequency":
                if self.training:
                    with torch.no_grad():
                        frequency = torch.bincount(nearest, minlength=book.num_embeddings).to(self.usage.dtype)
                        frequency = frequency / frequency.sum().clamp_min(1)
                        self.usage[level].mul_(self.usage_momentum).add_(frequency, alpha=1-self.usage_momentum)
                hot = self.usage[level][nearest] > self.hot_ratio / book.num_embeddings
            else:
                hot = torch.full_like(nearest, mode == "fixed", dtype=torch.bool)
            hot = hot & self.training
            noise = torch.zeros_like(logits)
            # Consume one draw for every code even for cold items, preserving the
            # assignment RNG contract while applying noise only to hot items.
            if self.training and mode != "plain":
                if noises is None:
                    uniform = torch.rand_like(logits)
                    eps = torch.finfo(uniform.dtype).eps
                    noise = -torch.log(-torch.log(uniform.clamp(eps, 1-eps)))
                else:
                    noise = noises[level]
            soft = F.softmax((logits + hot[:, None] * self.noise_scale * noise) / self.temperature, -1)
            ids = soft.argmax(-1)
            mixture = hard_forward(soft, ids)
            vector = mixture @ book.weight
            loss = ((vector-residual.detach()).square().mean(-1)
                    + self.beta * (vector.detach()-residual).square().mean(-1))
            vectors.append(vector)
            mixtures.append(mixture)
            softs.append(soft)
            indices.append(ids)
            losses.append(loss)
            masks.append(hot)
            residual = residual - vector
        return Quantization(latent, torch.stack(vectors).sum(0),
                            torch.stack(mixtures, 1), torch.stack(softs, 1),
                            torch.stack(indices, 1), torch.stack(losses).mean(0),
                            torch.stack(masks, 1))

    @torch.no_grad()
    def catalog_codes(self, features, balanced=True, epsilon=0.003, iterations=50):
        """Assign all catalog rows jointly; never run Sinkhorn on minibatches."""
        residual = self.encoder(features)
        codes = []
        for book in self.codebooks:
            distances = self.distances(residual, book.weight)
            ids = (balanced_assignments(distances, epsilon, iterations)
                   if balanced else distances.argmin(-1))
            codes.append(ids)
            residual = residual - book(ids)
        return torch.stack(codes, 1)


def append_collision_suffix(prefixes, codebook_size=256):
    """Injective full addresses in catalog row order, without renumbering rows."""
    counts = Counter()
    suffix = []
    for prefix in prefixes.detach().cpu().tolist():
        key = tuple(prefix)
        suffix.append(counts[key])
        counts[key] += 1
    if suffix and max(suffix) >= codebook_size:
        raise ValueError("Collision bucket exceeds suffix vocabulary")
    return torch.cat((prefixes, torch.tensor(suffix, device=prefixes.device, dtype=torch.long)[:, None]), 1)


class PositionalGenerator(nn.Module):
    """T5 encoder/decoder with separate address-position embeddings and heads."""

    def __init__(self, codebook_size=256, levels=3, model_dim=128, ff_dim=512,
                 num_layers=6, num_heads=4, key_dim=64, dropout=0.1):
        super().__init__()
        config = T5Config(vocab_size=1, d_model=model_dim,
                          d_ff=ff_dim, num_layers=num_layers,
                          num_decoder_layers=num_layers, num_heads=num_heads,
                          d_kv=key_dim, dropout_rate=dropout,
                          decoder_start_token_id=0, pad_token_id=0)
        self.transformer = T5Model(config)
        self.positions = nn.ModuleList([nn.Embedding(codebook_size, model_dim)
                                        for _ in range(levels+1)])
        # DIGER initializes generator linear/embedding weights at std .02,
        # including T5 components rather than retaining the generic T5 scale.
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def history_embeddings(self, probabilities, suffix, mask):
        pieces = [probabilities[:, :, i] @ self.positions[i].weight
                  for i in range(len(self.positions)-1)]
        pieces.append(self.positions[-1](suffix.clamp_min(0)))
        embeddings = torch.stack(pieces, 2).flatten(1, 2)
        expanded = mask[:, :, None].expand(-1, -1, len(self.positions)).flatten(1)
        embeddings = torch.where(expanded[:, :, None], embeddings,
                                 self.transformer.shared.weight[0].view(1, 1, -1))
        return embeddings, expanded

    def discrete_history_embeddings(self, codes, mask):
        embeddings = torch.stack([embedding(codes[:, :, i].clamp_min(0))
                                  for i, embedding in enumerate(self.positions)], 2).flatten(1, 2)
        expanded = mask[:, :, None].expand(-1, -1, len(self.positions)).flatten(1)
        return torch.where(expanded[:, :, None], embeddings,
                           self.transformer.shared.weight[0].view(1, 1, -1)), expanded

    def decoder_embeddings(self, target_probabilities):
        start = self.transformer.shared.weight[0].expand(target_probabilities.shape[0], -1)
        return torch.stack([start] + [target_probabilities[:, i] @ self.positions[i].weight
                                     for i in range(len(self.positions)-1)], 1)

    def forward(self, history_embeddings, mask, decoder_embeddings, encoder_outputs=None):
        output = self.transformer(inputs_embeds=history_embeddings if encoder_outputs is None else None,
                                  attention_mask=mask, decoder_inputs_embeds=decoder_embeddings,
                                  encoder_outputs=encoder_outputs, output_hidden_states=True)
        return torch.stack([output.last_hidden_state[:, i] @ self.positions[i].weight.t()
                            for i in range(decoder_embeddings.shape[1])], 1)

    @torch.no_grad()
    def beam_search(self, codes, mask, beams=20, returns=None):
        """Unconstrained address beams; scores are mean positional log probability."""
        returns = beams if returns is None else returns
        if not 1 <= returns <= beams <= self.positions[0].num_embeddings:
            raise ValueError("Require 1 <= returns <= beams <= vocabulary size")
        batch = codes.shape[0]
        embeddings, expanded = self.discrete_history_embeddings(codes, mask)
        embeddings = embeddings.repeat_interleave(beams, 0)
        expanded = expanded.repeat_interleave(beams, 0)
        encoded = self.transformer.encoder(inputs_embeds=embeddings, attention_mask=expanded, return_dict=True)
        paths = torch.zeros(batch*beams, 1, dtype=torch.long, device=codes.device)
        scores = torch.full((batch, beams), -1e9, device=codes.device)
        scores[:, 0] = 0
        vocabulary = self.positions[0].num_embeddings
        for position in range(len(self.positions)):
            decoder = torch.stack([self.transformer.shared(paths[:, 0])] + [
                self.positions[i](paths[:, i+1]) for i in range(position)], 1)
            logits = self.forward(None, expanded, decoder, encoded)[:, -1]
            candidates = (F.log_softmax(logits, -1) + scores.reshape(-1, 1)).reshape(batch, -1)
            scores, choices = candidates.topk(beams, dim=1, sorted=True)
            parents = choices // vocabulary + torch.arange(batch, device=codes.device)[:, None]*beams
            paths = torch.cat([paths[parents.flatten()], (choices % vocabulary).reshape(-1, 1)], 1)
        return (paths[:, 1:].reshape(batch, beams, -1)[:, :returns],
                scores[:, :returns] / len(self.positions))


class DIGERBackend(nn.Module):
    """Joint computation with frozen input features and target-only auxiliary losses."""

    def __init__(self, features, tokenizer_options=None, generator_options=None):
        super().__init__()
        if features.ndim != 2:
            raise ValueError("features must have shape [catalog rows, semantic width]")
        self.tokenizer_options = dict(tokenizer_options or {})
        self.generator_options = dict(generator_options or {})
        self.register_buffer("features", features.detach().clone())
        self.tokenizer = SemanticTokenizer(features.shape[1], **(tokenizer_options or {}))
        options = dict(generator_options or {})
        options.setdefault("levels", len(self.tokenizer.codebooks))
        options.setdefault("codebook_size", self.tokenizer.codebooks[0].num_embeddings)
        if (options["levels"] != len(self.tokenizer.codebooks)
                or options["codebook_size"] != self.tokenizer.codebooks[0].num_embeddings):
            raise ValueError("Generator levels and codebook_size must match tokenizer")
        self.generator = PositionalGenerator(**options)

    def initialize_tokenizer(self, state, semantic_width):
        """Strictly load pretrained encoder/books/frequency state.

        A tokenizer trained for a different feature width cannot initialize this
        model. The fit controller must require a matched initializer or train
        one before freezing the encoder; random encoder construction is useful
        only for component tests.
        """
        if semantic_width != self.features.shape[1]:
            raise ValueError("Tokenizer semantic feature width differs from catalog")
        self.tokenizer.load_state_dict(state, strict=True)

    def batch_loss(self, history_ids, target_ids, catalog, mode="frequency",
                   cached_forward=True, code_weight=1.0, reconstruction_weight=1.0,
                   vq_weight=1.0, noises=None, detach_recommendation=False):
        """Catalog row0 is padding; catalog codes at padding are all -1."""
        mask = history_ids != 0
        targets = target_ids.reshape(-1)
        unique, inverse = torch.unique(torch.cat([history_ids[mask], targets]), sorted=True, return_inverse=True)
        quantized = self.tokenizer(self.features[unique], mode=mode, noises=noises)
        probabilities = (hard_forward(quantized.soft_probabilities, catalog[unique, :-1])
                         if cached_forward else quantized.probabilities)
        if detach_recommendation:
            probabilities = probabilities.detach()
        history_inverse = torch.zeros_like(history_ids)
        history_inverse[mask] = inverse[:mask.sum()]
        target_inverse = inverse[mask.sum():]
        history_probabilities = probabilities[history_inverse]
        target_probabilities = probabilities[target_inverse]
        embeddings, expanded = self.generator.history_embeddings(history_probabilities, catalog[history_ids, -1], mask)
        logits = self.generator(embeddings, expanded, self.generator.decoder_embeddings(target_probabilities))
        denominator = targets.shape[0]*len(self.generator.positions)
        semantic = -(target_probabilities * F.log_softmax(logits[:, :-1].float(), -1)).sum()/denominator
        suffix = F.cross_entropy(logits[:, -1].float(), catalog[targets, -1], reduction="sum")/denominator
        reconstruction = (quantized.quantized-quantized.latent).square().mean(-1)[target_inverse].mean()
        vq = quantized.vq_per_item[target_inverse].mean()
        loss = code_weight*(semantic+suffix) + reconstruction_weight*reconstruction + vq_weight*vq
        return {"loss": loss, "semantic": semantic, "suffix": suffix,
                "reconstruction": reconstruction, "vq": vq, "logits": logits,
                "quantization": quantized, "unique_ids": unique}

    @torch.no_grad()
    def catalog(self, balanced=True, epsilon=0.003, iterations=50):
        prefixes = self.tokenizer.catalog_codes(self.features[1:], balanced, epsilon, iterations)
        full = append_collision_suffix(prefixes, self.generator.positions[-1].num_embeddings)
        return torch.cat([full.new_full((1, full.shape[1]), -1), full], 0)
