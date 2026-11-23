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

import math
import zlib

import numpy as np
from tqdm.auto import tqdm

from ..recommender import BundleConstructionRecommender
from ...exception import ScoreException
from ...utils import get_rng

SUPPORTED_DECODERS = ("ancestral", "confidence")
SUPPORTED_LAYOUTS = ("joint", "split")
SUPPORTED_TOKENIZERS = ("rvq", "faiss")


class DDBC(BundleConstructionRecommender):
    """DDBC: Discrete Diffusion for Bundle Construction.

    Item features are quantized into semantic IDs (``n_codebooks`` levels of
    ``codebook_size`` codes, plus a collision digit). A diffusion transformer
    learns to unmask the semantic-ID tokens of bundles and completes a partial
    bundle by unmasking the tokens of its missing items.

    The defaults follow the paper. ``cornac.models.ddbc`` ships the paper and
    the released-code settings as ``DDBC_PAPER_*_CONFIG`` and
    ``DDBC_RELEASE_*_CONFIG``.

    Item features must be provided through the evaluation method, e.g.::

        BundleConstructionEvaluation.from_splits(
            ..., item_feature=FeatureModality(features=feats, ids=item_ids)
        )

    With ``input_ratio=0``, a bundle is constructed from scratch. The user of a
    query is not used.

    Parameters
    ----------
    name: string, default: 'DDBC'
        The name of the recommender model.

    bundle_size: int, default: 4
        Number of items k of the training windows. Shorter training bundles
        are skipped.

    n_codebooks: int, default: 3
        Number of levels of a semantic ID.

    codebook_size: int, default: 128
        Number of codes C per level. At most C items may share the same codes.

    tokenizer: str, default: 'rvq'
        'rvq' trains a residual quantizer by gradient descent, starting from
        residual k-means. 'faiss' uses a Faiss residual quantizer with the
        inner-product metric, as the released code does, and needs a
        power-of-two ``codebook_size``. Unused when ``item_sids`` is given.

    item_sids: array-like of int, optional, default: None
        Item semantic IDs in token form, shape ``(n_items, n_codebooks + 1)``,
        in the order of ``item_ids``, e.g. the released ``clhe_token.json``.
        Replaces the tokenizer. Must be given with ``codebooks``.

    codebooks: array-like of float, optional, default: None
        Codebooks of ``item_sids``, shape ``(n_codebooks, codebook_size, dim)``
        or ``(n_codebooks * codebook_size, dim)``, e.g. the released
        ``clhe_weight.npy``.

    kmeans_iters: int, default: 20
    rvq_steps: int, default: 500
    rvq_lr: float, default: 1e-3
    rvq_beta: float, default: 0.25
        Settings of the 'rvq' tokenizer: k-means iterations per level, full-batch
        Adam steps, learning rate and commitment loss weight.

    hidden_size: int, default: 64
    cond_dim: int, default: 128
    n_blocks: int, default: 6
    n_heads: int, default: 8
    dropout: float, default: 0.1
        Transformer settings. When ``hidden_size`` equals the feature size, the
        code embeddings start from their codewords.

    n_diffusion_steps: int, optional, default: 25
        Number of diffusion steps T. None uses continuous time, as the released
        code does.

    time_conditioning: boolean, default: True
        Whether the denoiser sees the time. The released code does not.

    layout: str, default: 'joint'
        'joint' puts the given items and the masked slots in one
        ``[BOS] ... [EOS]`` sequence. 'split' gives the slots their own
        segment, as the released code does.

    decode: str, default: 'confidence'
        'confidence' reveals the most confident masked tokens first.
        'ancestral' is the sampler of the released code.

    sampling_steps: int, default: 25
        Number of decoding steps. The released code uses 256.

    fallback_levels: int, default: 3
        Number of levels whose codewords are summed for a generated ID that
        names no item. The released code uses 2.

    swap_ratio: float, default: 0.8
        Every training bundle gives ``int(len * swap_ratio)`` windows, each with
        one random adjacent swap.

    batch_size: int, default: 300
    accumulate: int, default: 2
    max_steps: int, default: 20000
    learning_rate: float, default: 3e-4
    weight_decay: float, default: 0.0
    warmup_steps: int, default: 500
    grad_clip: float, default: 1.0
        Training settings: batch size, batches per optimizer step, optimizer
        steps, AdamW learning rate and weight decay, linear warm-up steps and
        gradient-norm clipping.

    val_every: int, default: 32
        With a validation set, the validation loss is computed every
        ``val_every`` batches and at the last step.

    ema_decay: float, default: 0.9999
        Decay of the moving average of the weights used for validation and
        inference.

    use_bf16: boolean, default: False
        Use bfloat16 autocast (Ampere GPUs or newer).

    model_selection: str, default: 'best'
        'best' keeps the weights with the lowest validation loss (requires a
        validation set), 'last' keeps the final ones.

    pretrained: str, optional, default: None
        Path of a released DDBC checkpoint (``.ckpt``). ``fit`` loads its
        averaged weights and skips training. Requires ``item_sids`` and
        ``codebooks``.

    trainable: boolean, default: True
        When False, the model is not trained.

    verbose: boolean, default: False
        When True, a progress bar is displayed.

    seed: int, optional, default: None
        Random seed. If None, each call of ``fit`` draws a new seed.

    device: str, default: 'cpu'
        Torch device, e.g. 'cpu' or 'cuda'.

    References
    ----------
    * Tu, T., Li, A., Ma, Y., Xu, S., Liu, X., Ma, H., Pang, L. and Chua, T.-S.
      Discrete Diffusion for Bundle Construction. ICLR 2026.
      https://openreview.net/forum?id=dKyhgfe50H
    * Sahoo, S. S., Arriola, M., Schiff, Y., Gokaslan, A., Marroquin, E.,
      Chiu, J. T., Rush, A. and Kuleshov, V. Simple and Effective Masked
      Diffusion Language Models. NeurIPS 2024.
      https://github.com/kuleshov-group/mdlm
    """

    def __init__(
        self,
        name="DDBC",
        bundle_size=4,
        n_codebooks=3,
        codebook_size=128,
        tokenizer="rvq",
        item_sids=None,
        codebooks=None,
        kmeans_iters=20,
        rvq_steps=500,
        rvq_lr=1e-3,
        rvq_beta=0.25,
        hidden_size=64,
        cond_dim=128,
        n_blocks=6,
        n_heads=8,
        dropout=0.1,
        n_diffusion_steps=25,
        time_conditioning=True,
        layout="joint",
        decode="confidence",
        sampling_steps=25,
        fallback_levels=3,
        swap_ratio=0.8,
        batch_size=300,
        accumulate=2,
        max_steps=20000,
        val_every=32,
        learning_rate=3e-4,
        weight_decay=0.0,
        warmup_steps=500,
        grad_clip=1.0,
        ema_decay=0.9999,
        use_bf16=False,
        model_selection="best",
        pretrained=None,
        trainable=True,
        verbose=False,
        seed=None,
        device="cpu",
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        if bundle_size < 2:
            raise ValueError(
                "bundle_size must be at least 2 but {} is given".format(bundle_size)
            )
        if decode not in SUPPORTED_DECODERS:
            raise ValueError(
                "decode must be one of {} but '{}' is given".format(
                    SUPPORTED_DECODERS, decode
                )
            )
        if model_selection not in ("best", "last"):
            raise ValueError(
                "model_selection must be 'best' or 'last' but '{}' is given".format(
                    model_selection
                )
            )
        for param, value in (
            ("accumulate", accumulate),
            ("max_steps", max_steps),
            ("val_every", val_every),
        ):
            if value < 1:
                raise ValueError(
                    "{} must be at least 1 but {} is given".format(param, value)
                )
        if hidden_size % (2 * n_heads) != 0:
            raise ValueError(
                "hidden_size must be divisible by n_heads with an even size per "
                "head for the rotary embedding"
            )
        for param, value, supported in (
            ("tokenizer", tokenizer, SUPPORTED_TOKENIZERS),
            ("layout", layout, SUPPORTED_LAYOUTS),
        ):
            if value not in supported:
                raise ValueError(
                    "{} must be one of {} but '{}' is given".format(
                        param, supported, value
                    )
                )
        if tokenizer == "faiss" and codebook_size & (codebook_size - 1) != 0:
            raise ValueError("the 'faiss' tokenizer needs a power-of-two codebook_size")
        if n_diffusion_steps is not None and n_diffusion_steps < 1:
            raise ValueError(
                "n_diffusion_steps must be None or at least 1 but {} is given".format(
                    n_diffusion_steps
                )
            )
        if sampling_steps < 1:
            raise ValueError(
                "sampling_steps must be at least 1 but {} is given".format(
                    sampling_steps
                )
            )
        if not 1 <= fallback_levels <= n_codebooks:
            raise ValueError(
                "fallback_levels must be between 1 and n_codebooks={} but {} is "
                "given".format(n_codebooks, fallback_levels)
            )
        if (item_sids is None) != (codebooks is None):
            raise ValueError("item_sids and codebooks must be given together")
        if pretrained is not None and item_sids is None:
            raise ValueError(
                "pretrained requires the item_sids and codebooks the checkpoint "
                "was trained with"
            )

        self.bundle_size = bundle_size
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.tokenizer = tokenizer
        self.item_sids = item_sids
        self.codebooks = codebooks
        self.kmeans_iters = kmeans_iters
        self.rvq_steps = rvq_steps
        self.rvq_lr = rvq_lr
        self.rvq_beta = rvq_beta
        self.hidden_size = hidden_size
        self.cond_dim = cond_dim
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.dropout = dropout
        self.n_diffusion_steps = n_diffusion_steps
        self.time_conditioning = time_conditioning
        self.layout = layout
        self.decode = decode
        self.sampling_steps = sampling_steps
        self.fallback_levels = fallback_levels
        self.swap_ratio = swap_ratio
        self.batch_size = batch_size
        self.accumulate = accumulate
        self.max_steps = max_steps
        self.val_every = val_every
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.grad_clip = grad_clip
        self.ema_decay = ema_decay
        self.use_bf16 = use_bf16
        self.model_selection = model_selection
        self.pretrained = pretrained
        self.seed = seed
        self.device = device

        self.net = None
        self.item_tokens = None
        self.item_units = None
        self.code_vectors = None
        self.code_to_item = None
        self.given_items = None
        self.best_val_loss = None
        self.fit_seed = None
        self._decoded = {}

    @staticmethod
    def _training_bundles(train_set):
        """Full training bundles. The given items of held-out bundles, which
        Cornac stores in the training set, are left out."""
        partial = set(train_set.partial_bundles.tolist())
        return [
            train_set.bundle_items(i).tolist()
            for i in range(train_set.num_users)
            if i not in partial
        ]

    @staticmethod
    def _held_out_bundles(train_set, val_set):
        """Validation bundles as given items followed by missing items."""
        return [
            train_set.bundle_items(i).tolist() + val_set.bundle_items(i).tolist()
            for i in np.unique(val_set.uir_tuple[0])
        ]

    def _given_ids(self, dim):
        """Checked ``item_sids`` and ``codebooks``."""
        tokens = np.asarray(self.item_sids, dtype=np.int64)
        n_levels, size = self.n_codebooks, self.codebook_size
        if tokens.shape != (self.total_items, n_levels + 1):
            raise ValueError(
                "item_sids must have shape {} but {} is given".format(
                    (self.total_items, n_levels + 1), tokens.shape
                )
            )
        low = np.arange(n_levels + 1) * size + 1
        if ((tokens < low) | (tokens >= low + size)).any():
            raise ValueError(
                "item_sids must hold the codes of level l in "
                "[(l - 1) * codebook_size + 1, l * codebook_size] and the "
                "collision digits in [n_codebooks * codebook_size + 1, "
                "(n_codebooks + 1) * codebook_size]"
            )
        if len(np.unique(tokens, axis=0)) != len(tokens):
            raise ValueError("item_sids must give every item a distinct ID")
        codebooks = np.asarray(self.codebooks, dtype=np.float32)
        shape = (n_levels, size, dim)
        if codebooks.shape not in (shape, (n_levels * size, dim)):
            raise ValueError(
                "codebooks must have shape {} or {} but {} is given".format(
                    shape, (n_levels * size, dim), codebooks.shape
                )
            )
        return tokens, codebooks.reshape(shape)

    def fit(self, train_set, val_set=None):
        """Build the item semantic IDs, then train the model or load
        ``pretrained``.

        Parameters
        ----------
        train_set: :obj:`cornac.data.BundleConstructionDataset`, required
            Full training bundles and the given items of the held-out bundles,
            with ``item_feature``. Only the full bundles are used for training.

        val_set: :obj:`cornac.data.BundleConstructionDataset`, optional, default: None
            Missing items of the validation bundles, used for model
            selection.

        Returns
        -------
        self : object
        """
        super().fit(train_set, val_set)
        self.given_items = {
            int(i): train_set.bundle_items(i) for i in train_set.partial_bundles
        }
        if not self.trainable:
            return self

        import torch

        from .ddbc import (
            DDBCNet,
            ResidualVQ,
            faiss_residual_codes,
            read_checkpoint,
            released_backbone_state,
            residual_kmeans,
            semantic_id_tokens,
            swap_windows,
        )

        if train_set.item_feature is None or train_set.item_feature.features is None:
            raise ValueError(
                "DDBC requires item features, e.g. item_feature=FeatureModality(...)"
            )

        seed = self.seed
        if seed is None:
            seed = int(get_rng(None).randint(np.iinfo(np.int32).max))
        self.fit_seed = seed
        self._decoded = {}
        torch.manual_seed(seed)
        device = torch.device(self.device)

        features = np.asarray(
            train_set.item_feature.features[: self.total_items], dtype=np.float32
        )
        if self.item_sids is not None:
            self.item_tokens, codebooks = self._given_ids(features.shape[1])
        elif self.tokenizer == "faiss":
            codes, codebooks = faiss_residual_codes(
                features, self.n_codebooks, self.codebook_size
            )
            self.item_tokens = semantic_id_tokens(codes, self.codebook_size)
        else:
            e = torch.from_numpy(features).to(device)
            init = residual_kmeans(
                e, self.n_codebooks, self.codebook_size, self.kmeans_iters, seed
            )
            rvq = ResidualVQ(init, self.rvq_beta).to(device)
            rvq.fit(e, self.rvq_steps, self.rvq_lr)
            self.item_tokens = semantic_id_tokens(
                rvq.encode(e).cpu().numpy(), self.codebook_size
            )
            codebooks = rvq.codebooks.detach().cpu().numpy()
        self.code_to_item = {
            tuple(row): item for item, row in enumerate(self.item_tokens.tolist())
        }
        self.item_units = features / np.maximum(
            np.linalg.norm(features, axis=1, keepdims=True), 1e-12
        )
        self.code_vectors = codebooks.reshape(-1, features.shape[1])

        self.net = DDBCNet(
            codebook=self.code_vectors,
            n_levels=self.n_codebooks,
            codebook_size=self.codebook_size,
            hidden_size=self.hidden_size,
            cond_dim=self.cond_dim,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads,
            dropout=self.dropout,
            n_diffusion_steps=self.n_diffusion_steps,
            time_conditioning=self.time_conditioning,
            use_bf16=self.use_bf16,
        ).to(device)
        if self.pretrained is not None:
            self.net.backbone.load_state_dict(
                released_backbone_state(
                    read_checkpoint(self.pretrained), self.net.backbone
                )
            )
            self.net.eval()
            return self

        item_tokens = self.item_tokens.tolist()
        windows = swap_windows(
            self._training_bundles(train_set),
            self.swap_ratio,
            self.bundle_size,
            get_rng(seed),
        )
        if len(windows) == 0:
            raise ValueError(
                "no training bundle has at least bundle_size={} items".format(
                    self.bundle_size
                )
            )
        train_ids = torch.tensor([self.net.encode(w, item_tokens) for w in windows])
        val_ids = None
        if val_set is not None:
            held_out = self._held_out_bundles(train_set, val_set)
            val_rows = [
                b[: self.bundle_size] for b in held_out if len(b) >= self.bundle_size
            ]
            if len(val_rows) > 0:
                val_ids = torch.tensor(
                    [self.net.encode(r, item_tokens) for r in val_rows]
                )

        self._train(train_ids, val_ids, seed, device)
        return self

    def _train(self, train_ids, val_ids, seed, device):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from .ddbc import EMA, validation_loss

        net = self.net
        params = list(net.parameters())
        ema = EMA(params, self.ema_decay)
        optimizer = torch.optim.AdamW(
            params, lr=self.learning_rate, weight_decay=self.weight_decay
        )
        warmup = max(1.0, self.warmup_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: s / warmup if s < self.warmup_steps else 1.0
        )
        loader = DataLoader(
            TensorDataset(train_ids),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        n_epochs = math.ceil(self.max_steps * self.accumulate / len(loader))

        best_loss, best_state, step, n_batches = float("inf"), None, 0, 0
        progress = tqdm(total=self.max_steps, desc="Training", disable=not self.verbose)
        for epoch in range(1, n_epochs + 1):
            net.train()
            logs = {}
            for b, (x,) in enumerate(loader):
                n_batches += 1
                x = x.to(device)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=self.use_bf16
                ):
                    loss = net(x).sum() / x.shape[1]
                (loss / self.accumulate).backward()

                if (b + 1) % self.accumulate == 0 or b + 1 == len(loader):
                    torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    ema.update(params)
                    step += 1
                    progress.update(1)

                if val_ids is not None and (
                    n_batches % self.val_every == 0 or step >= self.max_steps
                ):
                    ema.store(params)
                    ema.copy_to(params)
                    val_loss = validation_loss(net, val_ids, self.batch_size, seed)
                    ema.restore(params)
                    logs["val_score"] = -val_loss
                    if val_loss < best_loss:
                        best_loss, best_state = val_loss, ema.state()
                    progress.set_postfix(
                        loss="{:.2f}".format(loss.item()),
                        best_val_loss="{:.4f}".format(best_loss),
                    )
                    net.train()

                if step >= self.max_steps:
                    break
            self.on_epoch_end(epoch, loss=loss.item(), **logs)
            if step >= self.max_steps:
                break
        progress.close()

        if self.model_selection == "best" and best_state is not None:
            ema.shadow_params = best_state
            self.best_val_loss = best_loss
        ema.copy_to(params)
        net.eval()

    def score(self, user_idx, item_idx=None):
        """Score items for a partial bundle from its decoded bundle

        The bundle is decoded with ``bundle_size - len(given)`` slots, the
        same bundle as :meth:`construct`. The constructed items score highest,
        and every other item scores its highest cosine similarity to the slot
        vectors.

        Parameters
        ----------
        user_idx: int, required
            The index of the bundle instance.

        item_idx: int, optional, default: None
            The index of the item to score. If None, every item is scored.

        Returns
        -------
        res : A scalar or a Numpy array
            Score of the item, or of every item with shape ``(total_items,)``.

        """
        if self.net is None:
            raise ValueError("DDBC is not fitted yet")
        given = self.given_items.get(int(user_idx))
        if given is None:
            raise ScoreException(
                "Can't make score prediction for bundle instance %d" % user_idx
            )
        if item_idx is not None and not self.knows_item(item_idx):
            raise ScoreException("Can't make score prediction for item %d" % item_idx)

        given = [int(i) for i in given]
        size = min(max(1, self.bundle_size - len(given)), self.total_items - len(given))
        scores = np.zeros(self.total_items)
        if size > 0:
            items, vectors = self._decode(user_idx, given, size)
            scores = (self.item_units @ vectors.T).max(axis=1)
            scores[items] = 2.0 + np.arange(len(items), 0, -1)
        return scores if item_idx is None else scores[item_idx]

    def construct(self, user_idx, input_items, size, **kwargs):
        """Complete a partial bundle by unmasking the semantic IDs of its missing items

        Parameters
        ----------
        user_idx: int, required
            The index of the bundle instance. It only seeds ancestral sampling.

        input_items: array-like of int, required
            Indices of the items already in the bundle, possibly none.

        size: int, required
            Number of items to add to the bundle.

        Returns
        -------
        res : a Numpy array
            At most ``size`` distinct item indices, none in ``input_items``.

        """
        if self.net is None:
            raise ValueError("DDBC is not fitted yet")

        input_items = [int(i) for i in input_items]
        size = min(size, self.total_items - len(input_items))
        if size <= 0:
            return np.zeros(0, dtype="int")
        return self._decode(user_idx, input_items, size)[0]

    def _decode(self, user_idx, input_items, size):
        """Items and slot vectors of a query, decoded once and cached."""
        key = (int(user_idx), tuple(input_items), int(size))
        if key not in self._decoded:
            import torch

            device = next(self.net.parameters()).device
            x = torch.tensor(
                [self.net.query(input_items, self.item_tokens, size, self.layout)],
                device=device,
            )
            sample_seed = (
                zlib.crc32(
                    np.asarray([user_idx] + list(input_items), dtype=np.int64).tobytes()
                )
                ^ self.fit_seed
            )
            generator = torch.Generator(device=device).manual_seed(sample_seed)
            codes = self.net.complete(
                x,
                size,
                decode=self.decode,
                steps=self.sampling_steps,
                generator=generator,
            )
            self._decoded[key] = self._slot_items(codes[0].cpu().numpy(), input_items)
        return self._decoded[key]

    def _slot_items(self, codes, input_items):
        """Catalog items and unit vectors of the generated slots. Each slot
        takes the free item closest to its vector."""
        used = set(input_items)
        items, vectors = [], []
        for row in codes.tolist():
            item = self.code_to_item.get(tuple(row))
            if item is None:
                vector = self.code_vectors[
                    np.asarray(row[: self.fallback_levels]) - 1
                ].sum(0)
                vector = vector / max(np.linalg.norm(vector), 1e-12)
            else:
                vector = self.item_units[item]
            if item is None or item in used:
                similarity = self.item_units @ vector
                similarity[list(used)] = -np.inf
                item = int(similarity.argmax())
            used.add(item)
            items.append(item)
            vectors.append(vector)
        return np.asarray(items, dtype="int"), np.asarray(vectors)
