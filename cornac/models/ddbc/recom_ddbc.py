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


class DDBC(BundleConstructionRecommender):
    """DDBC: Discrete Diffusion for Bundle Construction.

    Item features are quantized into semantic IDs by a residual vector
    quantizer (``n_codebooks`` levels of ``codebook_size`` codes, plus a suffix
    that tells colliding items apart). A diffusion transformer is trained to
    unmask the semantic-ID tokens of bundles, and completes a partial bundle by
    unmasking the tokens of its missing items given its seed items.

    Item features must be provided through the evaluation method, e.g.::

        BundleConstructionEvaluation.from_splits(
            ..., item_feature=FeatureModality(features=feats, ids=item_ids)
        )

    DDBC completes the given items of a bundle (``input_ratio`` above 0) or,
    with ``input_ratio=0``, constructs a bundle from scratch by unmasking every
    item (Appendix D.1 of the paper). It does not use the user of a query.

    :meth:`construct` generates the semantic IDs of the missing items and
    returns one catalog item for each: the item that has the generated ID, or
    the item closest by cosine to the sum of the generated codewords when no
    item has it. :meth:`score` ranks the catalog by the likelihood of an item's
    semantic ID in the next masked block.

    Parameters
    ----------
    name: string, default: 'DDBC'
        The name of the recommender model.

    bundle_size: int, default: 4
        Number of items k of the training windows. Training bundles with fewer
        items are skipped, longer ones give random windows of k items.

    n_codebooks: int, default: 3
        Number of quantization levels of a semantic ID.

    codebook_size: int, default: 256
        Number of codes per level. The paper uses 128 on POG and 256 on Spotify.

    codebook_init: numpy array, optional, default: None
        Initial codebooks of shape ``(n_codebooks * codebook_size, dim)`` or
        ``(n_codebooks, codebook_size, dim)``, where ``dim`` is the item feature
        size, e.g. the ``clhe_weight.npy`` released by the authors. If None, the codebooks
        start from residual k-means on the item features.

    kmeans_iters: int, default: 20
        Lloyd iterations per level of the residual k-means initialization.

    rvq_steps: int, default: 500
        Full-batch Adam steps of the quantizer.

    rvq_lr: float, default: 1e-3
        Learning rate of the quantizer.

    rvq_beta: float, default: 0.25
        Weight of the commitment loss of the quantizer.

    hidden_size: int, default: 64
        Hidden size of the transformer. When it equals the feature dimension,
        the token embeddings of the codes start from their codewords.

    cond_dim: int, default: 128
        Size of the diffusion-step embedding.

    n_blocks: int, default: 6
        Number of transformer blocks.

    n_heads: int, default: 8
        Number of attention heads.

    dropout: float, default: 0.1
        Dropout rate of the transformer blocks.

    n_diffusion_steps: int, default: 25
        Number of diffusion steps T.

    decode: str, default: 'confidence'
        'confidence' reveals the most confident tokens first and takes their
        argmax (Algorithm 2 of the paper). 'ancestral' reveals each masked
        token with probability 1/t and samples it, as the reference code does.

    swap_ratio: float, default: 0.8
        Every training bundle gives ``int(len * swap_ratio)`` windows, each with
        one random adjacent swap. The paper uses 0.8 on POG and 0.4 on Spotify.

    batch_size: int, default: 300
        Training batch size.

    accumulate: int, default: 2
        Number of batches per optimizer step.

    max_steps: int, default: 20000
        Number of optimizer steps.

    val_every: int, default: 32
        When a validation set is given, validation NLL is computed every
        ``val_every`` batches, counted across epochs, and at the last step.

    learning_rate: float, default: 3e-4
        AdamW learning rate.

    weight_decay: float, default: 0.0
        AdamW weight decay.

    warmup_steps: int, default: 500
        Linear warm-up steps of the learning rate.

    grad_clip: float, default: 1.0
        Gradient-norm clipping.

    ema_decay: float, default: 0.9999
        Decay of the exponential moving average of the weights, which is used
        for validation and inference.

    use_bf16: boolean, default: False
        Train and infer with bfloat16 autocast (Ampere GPUs or newer).

    model_selection: str, default: 'best'
        'best' keeps the averaged weights with the lowest validation NLL
        (requires a validation set), 'last' keeps the final ones.

    trainable: boolean, default: True
        When False, the model is not trained.

    verbose: boolean, default: False
        When True, a progress bar is displayed.

    seed: int, optional, default: None
        Random seed for initialization, augmentation, training and sampling.

    device: str, default: 'cpu'
        Torch device, e.g. 'cpu' or 'cuda'.

    References
    ----------
    Discrete Diffusion for Bundle Construction. ICLR 2026.
    https://openreview.net/forum?id=dKyhgfe50H
    """

    def __init__(
        self,
        name="DDBC",
        bundle_size=4,
        n_codebooks=3,
        codebook_size=256,
        codebook_init=None,
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
        decode="confidence",
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

        self.bundle_size = bundle_size
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_init = codebook_init
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
        self.decode = decode
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
        self.seed = seed
        self.device = device
        self.rng = get_rng(seed)

        self.net = None
        self.item_tokens = None
        self.item_units = None
        self.code_vectors = None
        self.code_to_item = None
        self.given_items = None
        self.best_val_nll = None

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
        """Validation bundles rebuilt as given items followed by missing items,
        since Cornac splits them between the training and validation sets."""
        return [
            train_set.bundle_items(i).tolist() + val_set.bundle_items(i).tolist()
            for i in np.unique(val_set.uir_tuple[0])
        ]

    def fit(self, train_set, val_set=None):
        """Train the semantic-ID quantizer and the diffusion model.

        Parameters
        ----------
        train_set: :obj:`cornac.data.BundleConstructionDataset`, required
            Full training bundles and the given items of the held-out bundles,
            with ``item_feature``. Only the full bundles are used for training.

        val_set: :obj:`cornac.data.BundleConstructionDataset`, optional, default: None
            Missing items of the validation bundles. Their NLL is reported to
            the callbacks and, with ``model_selection='best'``, selects the
            weights.

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
            residual_kmeans,
            semantic_id_tokens,
            swap_windows,
        )

        if train_set.item_feature is None or train_set.item_feature.features is None:
            raise ValueError(
                "DDBC requires item features, e.g. item_feature=FeatureModality(...)"
            )

        seed = self.seed if self.seed is not None else 0
        torch.manual_seed(seed)
        device = torch.device(self.device)

        features = np.asarray(
            train_set.item_feature.features[: self.total_items], dtype=np.float32
        )
        e = torch.from_numpy(features).to(device)
        if self.codebook_init is None:
            init = residual_kmeans(
                e, self.n_codebooks, self.codebook_size, self.kmeans_iters, seed
            )
        else:
            init = np.asarray(self.codebook_init, dtype=np.float32)
            shape = (self.n_codebooks, self.codebook_size, features.shape[1])
            if init.shape not in (shape, (shape[0] * shape[1], shape[2])):
                raise ValueError(
                    "codebook_init must have shape {} or {} but {} is given".format(
                        (shape[0] * shape[1], shape[2]), shape, init.shape
                    )
                )
            init = torch.from_numpy(init.reshape(shape)).to(device)
        rvq = ResidualVQ(init, self.rvq_beta).to(device)
        rvq.fit(e, self.rvq_steps, self.rvq_lr)
        codes = rvq.encode(e).cpu().numpy()

        self.item_tokens = semantic_id_tokens(codes, self.codebook_size)
        self.code_to_item = {
            tuple(row): item for item, row in enumerate(self.item_tokens.tolist())
        }
        self.item_units = features / np.maximum(
            np.linalg.norm(features, axis=1, keepdims=True), 1e-12
        )
        self.code_vectors = (
            rvq.codebooks.detach().reshape(-1, features.shape[1]).cpu().numpy()
        )

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
            use_bf16=self.use_bf16,
        ).to(device)

        item_tokens = self.item_tokens.tolist()
        windows = swap_windows(
            self._training_bundles(train_set),
            self.swap_ratio,
            self.bundle_size,
            self.rng,
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

        from .ddbc import EMA, validation_nll

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

        best_nll, best_state, step, n_batches = float("inf"), None, 0, 0
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
                    val_nll = validation_nll(net, val_ids, self.batch_size, seed)
                    ema.restore(params)
                    logs["val_score"] = -val_nll
                    if val_nll < best_nll:
                        best_nll, best_state = val_nll, ema.state()
                    progress.set_postfix(
                        loss="{:.2f}".format(loss.item()),
                        best_val_nll="{:.4f}".format(best_nll),
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
            self.best_val_nll = best_nll
        ema.copy_to(params)
        net.eval()

    def score(self, user_idx, item_idx=None):
        """Score items as the next item of a partial bundle

        The given items of the bundle instance, the last ``bundle_size - 1``
        ones when there are more, fill the first blocks of a window of
        ``bundle_size`` items and the other blocks are masked. An item scores
        the sum of the log-probabilities of its semantic-ID tokens in the first
        masked block.

        Parameters
        ----------
        user_idx: int, required
            The index of the bundle instance.

        item_idx: int, optional, default: None
            The index of the item to score. If None, every item is scored.

        Returns
        -------
        res : A scalar or a Numpy array
            Log-likelihood of the item, or of every item with shape
            ``(total_items,)``.

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

        import torch

        bundle = given[max(0, len(given) - self.bundle_size + 1) :].tolist()
        device = next(self.net.parameters()).device
        prefix = torch.tensor(
            [self.net.encode(bundle, self.item_tokens, with_eos=False)], device=device
        )
        tokens = torch.as_tensor(self.item_tokens, device=device)
        scores = (
            self.net.next_item_scores(prefix, self.bundle_size - len(bundle), tokens)[0]
            .cpu()
            .numpy()
        )
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

        import torch

        input_items = [int(i) for i in input_items]
        size = min(size, self.total_items - len(input_items))
        if size <= 0:
            return np.zeros(0, dtype="int")

        device = next(self.net.parameters()).device
        prefix = torch.tensor(
            [self.net.encode(input_items, self.item_tokens, with_eos=False)],
            device=device,
        )
        sample_seed = zlib.crc32(
            np.asarray([user_idx] + input_items, dtype=np.int64).tobytes()
        ) ^ (self.seed or 0)
        generator = torch.Generator(device=device).manual_seed(sample_seed)
        codes = (
            self.net.complete(prefix, size, decode=self.decode, generator=generator)[0]
            .cpu()
            .numpy()
        )
        return self._codes_to_items(codes, input_items)

    def _codes_to_items(self, codes, input_items):
        """Map generated semantic IDs to distinct catalog items, unlike the
        reference code, which maps them to a candidate pool: the item with the
        ID, else the item closest by cosine to the sum of the codewords, and the
        closest unused item when the item is already in the bundle."""
        used = set(input_items)
        items = []
        for row in codes.tolist():
            item = self.code_to_item.get(tuple(row))
            if item is None or item in used:
                if item is None:
                    vector = self.code_vectors[np.asarray(row[:-1]) - 1].sum(0)
                else:
                    vector = self.item_units[item]
                similarity = self.item_units @ (
                    vector / max(np.linalg.norm(vector), 1e-12)
                )
                similarity[list(used)] = -np.inf
                item = int(similarity.argmax())
            used.add(item)
            items.append(item)
        return np.asarray(items, dtype="int")
