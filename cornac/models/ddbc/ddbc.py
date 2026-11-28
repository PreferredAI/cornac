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

The DiT, the loss and the ancestral sampler follow MDLM (Sahoo et al.,
NeurIPS 2024, https://github.com/kuleshov-group/mdlm, Apache-2.0).
"""

import collections
import math
import pickle
import zipfile

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


SAMPLING_EPS = 1e-3


class DDBCNet(nn.Module):
    """Masked discrete diffusion over the semantic-ID tokens of a bundle.

    A bundle is the token sequence ``[BOS, (BOI, c_1, ..., c_L, s) * k, EOS]``,
    where ``c_l`` are the codes of the item at level ``l`` and ``s`` its
    collision digit. Token ids: ``0`` is BOS, level ``l`` codes take
    ``(l - 1) * C + 1 .. l * C``, collision digits ``L * C + 1 .. L * C + S``
    with ``S = collision_size``, then BOI, EOS and the mask token.

    The mask rate is ``t / T`` in discrete time and ``(1 - 1e-3) t`` in
    continuous time.
    """

    def __init__(
        self,
        codebook,
        n_levels,
        codebook_size,
        collision_size=None,
        hidden_size=64,
        cond_dim=128,
        n_blocks=6,
        n_heads=8,
        dropout=0.1,
        n_diffusion_steps=25,
        time_conditioning=True,
        use_bf16=False,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.codebook_size = codebook_size
        self.collision_size = (
            codebook_size if collision_size is None else collision_size
        )
        self.item_width = n_levels + 2
        self.bos = 0
        self.boi = n_levels * codebook_size + self.collision_size + 1
        self.eos = self.boi + 1
        self.mask_index = self.eos + 1
        self.vocab_size = self.mask_index + 1
        self.T = n_diffusion_steps
        self.time_conditioning = time_conditioning
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

    def query(self, items, item_tokens, n_items, layout="joint"):
        """Token ids of the given items followed by ``n_items`` masked slots."""
        given = self.encode(items, item_tokens)
        slots = ([self.boi] + [self.mask_index] * (self.item_width - 1)) * n_items
        if layout == "split":
            return given + [self.bos] + slots + [self.eos]
        return given[:-1] + slots + [self.eos]

    def mask_rate(self, t):
        if self.T is None:
            return (1 - SAMPLING_EPS) * t
        return t / self.T

    def model_time(self, fraction):
        """Time ``t`` at a fraction of the schedule, 1 being fully masked."""
        return fraction if self.T is None else fraction * self.T

    def condition(self, t):
        if not self.time_conditioning:
            return torch.zeros_like(t)
        if self.T is None:
            return -torch.log1p(-self.mask_rate(t))
        return t

    def denoise(self, x, t):
        logits = self.backbone(x, self.condition(t)).float()
        logits[:, :, self.mask_index] += self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        unmasked = x != self.mask_index
        logits[unmasked] = self.neg_infinity
        logits[unmasked, x[unmasked]] = 0
        return logits

    def sample_t(self, n, device):
        u = torch.rand(n, device=device)
        u = (u / n + torch.arange(n, device=device) / n) % 1
        if self.T is None:
            return (1 - SAMPLING_EPS) * u + SAMPLING_EPS
        return (u * self.T).floor().clamp(max=self.T - 1) + 1

    def forward(self, x0):
        """Masked-token cross-entropy weighted by ``T / t`` in discrete time
        and ``1 / t`` in continuous time."""
        t = self.sample_t(x0.shape[0], x0.device)
        move_chance = (
            self.mask_rate(t)[:, None]
            .expand(x0.shape)
            .masked_fill(torch.isin(x0, self.special_tokens), 0.0)
        )
        xt = torch.where(
            torch.rand(*x0.shape, device=x0.device) < move_chance, self.mask_index, x0
        )
        log_p = torch.gather(self.denoise(xt, t), -1, x0[:, :, None]).squeeze(-1)
        weight = 1.0 / t if self.T is None else self.T / t
        return -log_p * weight[:, None]

    def slot_allowed(self, n_items, device):
        """Tokens allowed at each slot position."""
        key = (n_items, str(device))
        if key not in self._allowed_cache:
            cb = self.codebook_size
            block = torch.ones(
                self.item_width, self.vocab_size, dtype=torch.bool, device=device
            )
            for level in range(1, self.item_width):
                size = cb if level <= self.n_levels else self.collision_size
                block[level] = False
                block[level, (level - 1) * cb + 1 : (level - 1) * cb + size + 1] = True
            self._allowed_cache[key] = block.repeat(n_items, 1)
        return self._allowed_cache[key]

    @torch.no_grad()
    def complete(self, x, n_items, decode="confidence", steps=25, generator=None):
        """Unmask the ``n_items`` slots of the queries ``x`` and return their
        codes, shape ``(n, n_items, n_levels + 1)``."""
        end = x.shape[1] - 1
        start = end - n_items * self.item_width
        allowed = self.slot_allowed(n_items, x.device)
        x = x.clone()
        if decode == "ancestral":
            self._ancestral(x, start, end, allowed, steps, generator)
        else:
            self._confidence(x, start, end, allowed, steps)
        return x[:, start:end].reshape(x.shape[0], n_items, self.item_width)[:, :, 1:]

    def _times(self, n, fraction, device):
        return torch.full((n,), float(self.model_time(fraction)), device=device)

    def _ancestral(self, x, start, end, allowed, steps, generator):
        """Ancestral sampling as in the released code."""
        n, dt = x.shape[0], 1.0 / steps
        legal = allowed.clone()
        legal[:, self.mask_index] = True
        p_x0 = None
        for i in range(steps + 1):
            t = 1.0 - i * dt
            if p_x0 is None or self.time_conditioning:
                logp = self.denoise(x, self._times(n, t, x.device))
                p_x0 = logp[:, start:end].exp()
            q = p_x0 * dt
            q[:, :, self.mask_index] = t - dt
            q = q.masked_fill(~legal, -1e9)
            u = torch.rand(q.shape, device=x.device, generator=generator)
            tok = (q / (1e-10 - (u + 1e-10).log())).argmax(dim=-1)
            region = x[:, start:end]
            new = torch.where(region == self.mask_index, tok, region)
            if not torch.equal(new, region):
                x[:, start:end] = new
                p_x0 = None

    def _confidence(self, x, start, end, allowed, steps):
        """Reveal the most confident masked positions first."""
        n = x.shape[0]
        for s in range(steps, 0, -1):
            region = x[:, start:end]
            masked = region == self.mask_index
            if not bool(masked.any()):
                break
            logp = self.denoise(x, self._times(n, s / steps, x.device))
            logp = (
                logp[:, start:end]
                .masked_fill(~allowed[None], float("-inf"))
                .log_softmax(dim=-1)
            )
            conf, tok = logp.max(dim=-1)
            conf = conf.masked_fill(~masked, float("-inf"))
            k = math.ceil(int(masked.sum(dim=1).max()) / s)
            idx = conf.topk(k, dim=1).indices
            region.scatter_(1, idx, tok.gather(1, idx))


class ResidualVQ(nn.Module):
    """Residual vector quantizer trained by reconstruction."""

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
            q = F.one_hot(idx, cb.shape[0]).to(cb.dtype) @ cb
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
    """Initial codebooks by k-means on the residuals of each level."""
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


def faiss_residual_codes(e, n_levels, codebook_size):
    """Codes and codebooks of a Faiss residual quantizer, as in the released
    code."""
    import faiss

    x = np.ascontiguousarray(e, dtype=np.float32)
    n_bits = int(math.log2(codebook_size))
    index = faiss.IndexResidualQuantizer(
        x.shape[1], n_levels, n_bits, faiss.METRIC_INNER_PRODUCT
    )
    index.train(x)
    packed = index.rq.compute_codes(x)
    bits = np.unpackbits(packed, axis=1, bitorder="little")[:, : n_levels * n_bits]
    codes = bits.reshape(len(x), n_levels, n_bits).astype(np.int64) @ (
        1 << np.arange(n_bits)
    )
    codebooks = faiss.vector_to_array(index.rq.codebooks).reshape(
        n_levels, codebook_size, x.shape[1]
    )
    return codes, codebooks


def semantic_id_tokens(codes, codebook_size):
    """Token ids of the codes plus a collision digit, numbered from 1 in item
    order among the items that share the codes."""
    n_items, n_levels = codes.shape
    tokens = np.zeros((n_items, n_levels + 1), dtype=np.int64)
    seen = {}
    for item, sid in enumerate(map(tuple, codes.tolist())):
        seen[sid] = seen.get(sid, 0) + 1
        tokens[item, :n_levels] = [
            c + codebook_size * level + 1 for level, c in enumerate(sid)
        ]
        tokens[item, n_levels] = seen[sid] + codebook_size * n_levels
    return tokens


def swap_windows(bundles, swap_ratio, seq_len, rng):
    """Training windows of ``seq_len`` items with one random adjacent swap."""
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
def validation_loss(net, val_ids, batch_size, seed):
    """Mean validation loss with a fixed masking seed."""
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
                loss = net(x)
            keep = ~torch.isin(x, net.special_tokens)
            total += (loss * keep).sum().item()
            weight += keep.sum().item()
    return total / max(weight, 1.0)


_STORAGE_DTYPES = {
    "DoubleStorage": torch.float64,
    "FloatStorage": torch.float32,
    "HalfStorage": torch.float16,
    "BFloat16Storage": torch.bfloat16,
    "LongStorage": torch.int64,
    "IntStorage": torch.int32,
    "ShortStorage": torch.int16,
    "CharStorage": torch.int8,
    "ByteStorage": torch.uint8,
    "BoolStorage": torch.bool,
    "UntypedStorage": torch.uint8,
}


class _Inert:
    """Stand-in for every class a checkpoint names apart from tensors."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass

    def __setitem__(self, key, value):
        pass

    def append(self, value):
        pass

    def extend(self, values):
        pass


def _rebuild_tensor(storage, offset, size, stride, *args):
    return storage.as_strided(tuple(size), tuple(stride), offset)


def _rebuild_parameter(data, *args):
    return data


def _rebuild_from_type(func, new_type, args, state):
    return func(*args)


_SAFE_GLOBALS = {
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("torch", "Size"): torch.Size,
    ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor,
    ("torch._utils", "_rebuild_parameter"): _rebuild_parameter,
    ("torch._utils", "_rebuild_parameter_with_state"): _rebuild_parameter,
    ("torch._tensor", "_rebuild_from_type_v2"): _rebuild_from_type,
}


class _CheckpointUnpickler(pickle.Unpickler):
    def __init__(self, file, archive, prefix):
        super().__init__(file)
        self.archive = archive
        self.prefix = prefix
        self.storages = {}

    def find_class(self, module, name):
        if module == "torch" and name in _STORAGE_DTYPES:
            return _STORAGE_DTYPES[name]
        return _SAFE_GLOBALS.get((module, name), _Inert)

    def persistent_load(self, pid):
        if not (isinstance(pid, tuple) and pid[0] == "storage"):
            raise pickle.UnpicklingError("unsupported persistent id {}".format(pid))
        dtype, key = pid[1], str(pid[2])
        if not isinstance(dtype, torch.dtype):
            raise pickle.UnpicklingError("unsupported storage type")
        if key not in self.storages:
            data = bytearray(self.archive.read(self.prefix + "data/" + key))
            self.storages[key] = (
                torch.frombuffer(data, dtype=dtype)
                if len(data) > 0
                else torch.empty(0, dtype=dtype)
            )
        return self.storages[key]


def read_checkpoint(path):
    """Read a PyTorch checkpoint without running code from it."""
    if not zipfile.is_zipfile(path):
        raise ValueError("{} is not a PyTorch checkpoint archive".format(path))
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith("data.pkl")]
        if len(names) == 0:
            raise ValueError("{} is not a PyTorch checkpoint archive".format(path))
        prefix = min(names, key=len)[: -len("data.pkl")]
        if prefix + "byteorder" in archive.namelist():
            if archive.read(prefix + "byteorder").decode().strip() != "little":
                raise ValueError("only little-endian checkpoints are supported")
        with archive.open(prefix + "data.pkl") as f:
            return _CheckpointUnpickler(f, archive, prefix).load()


def released_backbone_state(checkpoint, backbone):
    """State dict of ``backbone`` with the averaged weights of a released
    checkpoint."""
    try:
        weights = checkpoint["state_dict"]
        shadow = list(checkpoint["ema"]["shadow_params"])
    except (KeyError, TypeError):
        raise ValueError("not a released DDBC checkpoint") from None
    state = collections.OrderedDict(
        (k[len("backbone.") :], v)
        for k, v in weights.items()
        if k.startswith("backbone.")
    )
    buffers = {name for name, _ in backbone.named_buffers()}
    params = [name for name in state if name not in buffers]
    if len(params) != len(shadow):
        raise ValueError(
            "the checkpoint has {} backbone parameters but {} averages".format(
                len(params), len(shadow)
            )
        )
    for name, average in zip(params, shadow):
        if average.shape != state[name].shape:
            raise ValueError("the average of {} has another shape".format(name))
        state[name] = average
    return state
