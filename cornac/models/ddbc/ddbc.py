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
"""Neural modules for DDBC (Discrete Diffusion for Bundle Construction, ICLR 2026).

A residual vector quantizer turns item features into semantic IDs, and a
diffusion transformer (DiT) learns to unmask the semantic IDs of the missing
items of a bundle.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA:
    def __init__(self, parameters, decay):
        self.decay = decay
        self.num_updates = 0
        self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        with torch.no_grad():
            for s, p in zip(
                self.shadow_params, [p for p in parameters if p.requires_grad]
            ):
                s.sub_((1.0 - decay) * (s - p))

    def copy_to(self, parameters):
        for s, p in zip(self.shadow_params, [p for p in parameters if p.requires_grad]):
            p.data.copy_(s.data)

    def store(self, parameters):
        self.collected_params = [p.clone() for p in parameters]

    def restore(self, parameters):
        for c, p in zip(self.collected_params, parameters):
            p.data.copy_(c.data)

    def state(self):
        return [s.clone() for s in self.shadow_params]


def bias_dropout_add_scale(x, scale, residual, prob, training):
    return residual + scale * F.dropout(x, p=prob, training=training)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(qkv, cos, sin):
    cos = cos[0, :, 0, 0, :][None, :, None, :]
    sin = sin[0, :, 0, 0, :][None, :, None, :]
    q, k, v = qkv.unbind(dim=2)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return torch.stack((q, k, v), dim=2)


class Rotary(nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached or self.cos_cached.device != x.device:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.cos_cached[:, :, 2, :, :].fill_(1.0)
            self.sin_cached[:, :, 2, :, :].fill_(0.0)
        return self.cos_cached, self.sin_cached


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t, max_period=10000):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(embedding)


class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, rotary_cos_sin, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        )
        x_skip = x
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        b, s, _ = x.shape
        qkv = self.attn_qkv(x).view(b, s, 3, self.n_heads, -1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(b, s, -1)
        x = bias_dropout_add_scale(
            self.attn_out(x), gate_msa, x_skip, self.dropout, self.training
        )
        x = bias_dropout_add_scale(
            self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)),
            gate_mlp,
            x,
            self.dropout,
            self.training,
        )
        return x


class EmbeddingLayer(nn.Module):
    """Token embeddings. The semantic-ID tokens start from their codewords when dimensions match."""

    def __init__(self, dim, vocab_dim, codebook):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))
        if codebook is not None and codebook.shape[1] == dim:
            weight = torch.from_numpy(np.asarray(codebook, dtype=np.float32))
            self.embedding.data[1 : weight.shape[0] + 1] = weight

    def forward(self, x):
        return self.embedding[x]


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class DIT(nn.Module):
    def __init__(
        self,
        vocab_size,
        codebook,
        hidden_size,
        cond_dim,
        n_blocks,
        n_heads,
        dropout,
        use_bf16,
    ):
        super().__init__()
        self.use_bf16 = use_bf16
        self.vocab_embed = EmbeddingLayer(hidden_size, vocab_size, codebook)
        self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(hidden_size // n_heads)
        self.blocks = nn.ModuleList(
            [
                DDiTBlock(hidden_size, n_heads, cond_dim, dropout=dropout)
                for _ in range(n_blocks)
            ]
        )
        self.output_layer = DDitFinalLayer(hidden_size, vocab_size, cond_dim)

    def forward(self, indices, sigma):
        x = self.vocab_embed(indices)
        c = F.silu(self.sigma_map(sigma))
        rotary_cos_sin = self.rotary_emb(x)
        with torch.autocast(
            device_type=indices.device.type, dtype=torch.bfloat16, enabled=self.use_bf16
        ):
            for block in self.blocks:
                x = block(x, rotary_cos_sin, c)
            x = self.output_layer(x, c)
        return x


class DDBCNet(nn.Module):
    """Masked discrete diffusion over the semantic-ID tokens of a bundle.

    A bundle is the token sequence ``[BOS, (BOI, c_1, ..., c_L, s) * k, EOS]``,
    where ``c_l`` are the codes of the item at level ``l`` and ``s`` its
    collision suffix. Token ids: ``0`` is BOS, level ``l`` codes take
    ``l * C + 1 .. (l + 1) * C``, suffixes ``L * C + 1 .. (L + 1) * C``, then
    BOI, EOS and the mask token.
    """

    def __init__(
        self,
        codebook,
        n_levels,
        codebook_size,
        hidden_size=64,
        cond_dim=128,
        n_blocks=6,
        n_heads=8,
        dropout=0.1,
        n_diffusion_steps=25,
        use_bf16=False,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.codebook_size = codebook_size
        self.item_width = n_levels + 2
        self.bos = 0
        self.boi = (n_levels + 1) * codebook_size + 1
        self.eos = self.boi + 1
        self.mask_index = self.eos + 1
        self.vocab_size = self.mask_index + 1
        self.T = n_diffusion_steps
        self.neg_infinity = -1000000.0
        self.use_bf16 = use_bf16
        self.backbone = DIT(
            self.vocab_size,
            codebook,
            hidden_size,
            cond_dim,
            n_blocks,
            n_heads,
            dropout,
            use_bf16,
        )
        self.register_buffer(
            "special_tokens", torch.tensor([self.bos, self.eos, self.boi])
        )
        self._allowed_cache = {}

    def encode(self, items, item_tokens, with_eos=True):
        ids = [self.bos]
        for i in items:
            ids.append(self.boi)
            ids.extend(item_tokens[i])
        if with_eos:
            ids.append(self.eos)
        return ids

    def denoise(self, x, t):
        logits = self.backbone(x, t).float()
        logits[:, :, self.mask_index] += self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        unmasked = x != self.mask_index
        logits[unmasked] = self.neg_infinity
        logits[unmasked, x[unmasked]] = 0
        return logits

    def sample_t(self, n, device):
        eps_t = torch.rand(n, device=device)
        offset = torch.arange(n, device=device) / n
        eps_t = (eps_t / n + offset) % 1
        return (eps_t * self.T).floor().clamp(max=self.T - 1) + 1

    def forward(self, x0):
        """Masked-token cross-entropy weighted by ``T / t``, the negative ELBO
        of Eq. 7 under the linear schedule. Eq. 7 omits the weight, while the
        text and the reference code apply it."""
        t = self.sample_t(x0.shape[0], x0.device)
        move_chance = (
            (t / self.T)[:, None]
            .expand(x0.shape)
            .masked_fill(torch.isin(x0, self.special_tokens), 0.0)
        )
        xt = torch.where(
            torch.rand(*x0.shape, device=x0.device) < move_chance, self.mask_index, x0
        )
        log_p = torch.gather(self.denoise(xt, t), -1, x0[:, :, None]).squeeze(-1)
        return -log_p * (self.T / t)[:, None]

    def allowed_table(self, seq_len, device):
        key = (seq_len, str(device))
        if key not in self._allowed_cache:
            cb = self.codebook_size
            allowed = torch.ones(
                seq_len, self.vocab_size, dtype=torch.bool, device=device
            )
            for pos in range(1, seq_len - 1):
                level = (pos - 1) % self.item_width
                if level == 0:
                    continue
                allowed[pos] = False
                allowed[pos, (level - 1) * cb + 1 : level * cb + 1] = True
            self._allowed_cache[key] = allowed
        return self._allowed_cache[key]

    @torch.no_grad()
    def complete(self, prefix, n_items, decode="ancestral", generator=None):
        """Generate the codes and suffix ``(n, n_items, n_levels + 1)`` of ``n_items``
        missing items after ``prefix``, the BOS and the blocks of the seed items.

        ``decode='ancestral'`` reveals each masked token with probability 1/t and
        samples it. ``'confidence'`` reveals the most confident tokens first
        (Algorithm 2 of the paper).
        """
        n = prefix.shape[0]
        device = prefix.device
        tail = torch.tensor(
            ([self.boi] + [self.mask_index] * (self.item_width - 1)) * n_items
            + [self.eos],
            device=device,
        ).repeat(n, 1)
        x = torch.cat((prefix, tail), dim=1)
        start = prefix.shape[1]
        allowed = self.allowed_table(x.shape[1], device)
        for s in range(1, self.T + 1):
            t = self.T - s + 1
            masked = x == self.mask_index
            if not bool(masked.any()):
                break
            logp = self.denoise(x, torch.full((n,), float(t), device=device))
            logp = logp.masked_fill(~allowed[None], float("-inf")).log_softmax(dim=-1)
            if decode == "ancestral":
                tok = torch.multinomial(
                    logp.exp().view(-1, logp.shape[-1]), 1, generator=generator
                ).view(n, -1)
                reveal = masked & (
                    torch.rand(x.shape, device=device, generator=generator) < 1.0 / t
                )
                x = torch.where(reveal, tok, x)
            else:
                conf, tok = logp.max(dim=-1)
                conf = conf.masked_fill(~masked, float("-inf"))
                remaining = int(masked.sum(dim=1).max())
                idx = conf.topk(math.ceil(remaining / t), dim=1).indices
                x.scatter_(1, idx, tok.gather(1, idx))
        return x[:, start:-1].reshape(n, n_items, self.item_width)[:, :, 1:]

    @torch.no_grad()
    def next_item_scores(self, prefix, n_masked, item_tokens):
        """Log-likelihood ``(n, n_items)`` of every item as the first of ``n_masked``
        masked blocks after ``prefix``, with the diffusion step matching the masked
        fraction of the items.
        """
        n = prefix.shape[0]
        device = prefix.device
        tail = torch.tensor(
            ([self.boi] + [self.mask_index] * (self.item_width - 1)) * n_masked
            + [self.eos],
            device=device,
        ).repeat(n, 1)
        x = torch.cat((prefix, tail), dim=1)
        n_seed = (prefix.shape[1] - 1) // self.item_width
        t = max(1, math.ceil(self.T * n_masked / (n_seed + n_masked)))
        allowed = self.allowed_table(x.shape[1], device)
        logp = self.denoise(x, torch.full((n,), float(t), device=device))
        logp = logp.masked_fill(~allowed[None], float("-inf")).log_softmax(dim=-1)
        start = prefix.shape[1] + 1
        block = logp[:, start : start + self.item_width - 1]
        return block.gather(-1, item_tokens.T[None].expand(n, -1, -1)).sum(1)


class ResidualVQ(nn.Module):
    """Residual vector quantizer trained by reconstruction (Eq. 2 of the paper)."""

    def __init__(self, init_codebooks, beta):
        super().__init__()
        self.codebooks = nn.Parameter(init_codebooks.clone())
        self.beta = beta

    @staticmethod
    def quantize(r, cb):
        dist = (r * r).sum(1, keepdim=True) - 2 * r @ cb.T + (cb * cb).sum(1)[None]
        return dist.argmin(dim=1)

    def forward(self, e):
        r = e
        recon = torch.zeros_like(e)
        commit = torch.zeros((), device=e.device)
        codes = []
        for cb in self.codebooks:
            idx = self.quantize(r, cb)
            q = cb[idx]
            commit = (
                commit
                + ((r.detach() - q) ** 2).sum(1).mean()
                + ((r - q.detach()) ** 2).sum(1).mean()
            )
            recon = recon + q
            r = r - q
            codes.append(idx)
        loss = ((e - recon) ** 2).sum(1).mean() + self.beta * commit
        return torch.stack(codes, dim=1), recon, loss

    @torch.no_grad()
    def encode(self, e, chunk=65536):
        return torch.cat(
            [self.forward(e[i : i + chunk])[0] for i in range(0, len(e), chunk)]
        )

    def fit(self, e, steps, lr):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(steps):
            loss = self.forward(e)[2]
            opt.zero_grad()
            loss.backward()
            opt.step()


def _nearest(x, centers, chunk=65536):
    return torch.cat(
        [
            ResidualVQ.quantize(x[i : i + chunk], centers)
            for i in range(0, len(x), chunk)
        ]
    )


def residual_kmeans(e, n_levels, codebook_size, n_iters=20, seed=0):
    """Initial codebooks, shape (n_levels, codebook_size, dim), by k-means on the residuals of each level."""
    g = torch.Generator().manual_seed(seed)
    r = e.clone()
    books = []
    for _ in range(n_levels):
        if len(r) >= codebook_size:
            init = torch.randperm(len(r), generator=g)[:codebook_size]
        else:
            init = torch.randint(len(r), (codebook_size,), generator=g)
        centers = r[init.to(r.device)].clone()
        for _ in range(n_iters):
            idx = _nearest(r, centers)
            counts = torch.bincount(idx, minlength=codebook_size)
            sums = torch.zeros_like(centers).index_add_(0, idx, r)
            updated = sums / counts.clamp(min=1)[:, None].to(r.dtype)
            centers = torch.where((counts > 0)[:, None], updated, centers)
        r = r - centers[_nearest(r, centers)]
        books.append(centers)
    return torch.stack(books)


def semantic_id_tokens(codes, codebook_size):
    """Token ids ``(n_items, n_levels + 1)`` of the codes plus a suffix that tells colliding items apart."""
    n_items, n_levels = codes.shape
    tokens = np.zeros((n_items, n_levels + 1), dtype=np.int64)
    seen = {}
    for item, sid in enumerate(map(tuple, codes.tolist())):
        seen[sid] = seen.get(sid, 0) + 1
        if seen[sid] > codebook_size:
            raise ValueError(
                "{} items share the semantic ID {}, more than codebook_size={}; "
                "increase codebook_size or n_codebooks".format(
                    seen[sid], sid, codebook_size
                )
            )
        tokens[item, :n_levels] = [
            c + codebook_size * level + 1 for level, c in enumerate(sid)
        ]
        tokens[item, n_levels] = seen[sid] + codebook_size * n_levels
    return tokens


def swap_windows(bundles, swap_ratio, seq_len, rng):
    """``int(len * swap_ratio)`` copies of every bundle with at least ``seq_len``
    items, each with one random adjacent swap and cut to a random window of
    ``seq_len`` items.
    """
    out = []
    for items in bundles:
        if len(items) < seq_len:
            continue
        for _ in range(int(len(items) * swap_ratio)):
            aug = list(items)
            i = rng.randint(0, len(aug) - 1)
            aug[i], aug[i + 1] = aug[i + 1], aug[i]
            start = rng.randint(0, len(aug) - seq_len + 1)
            out.append(aug[start : start + seq_len])
    return out


@torch.no_grad()
def validation_nll(net, val_ids, batch_size, seed):
    """Mean NLL of the non-special tokens, with a fixed masking seed."""
    net.eval()
    device = next(net.parameters()).device
    total, weight = 0.0, 0.0
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for i in range(0, len(val_ids), batch_size):
            x = val_ids[i : i + batch_size].to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=net.use_bf16
            ):
                nll = net(x)
            keep = ~torch.isin(x, net.special_tokens)
            total += (nll * keep).sum().item()
            weight += keep.sum().item()
    return total / max(weight, 1.0)
