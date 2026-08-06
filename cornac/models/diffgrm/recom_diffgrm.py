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

import hashlib
import math
import time

import numpy as np
from tqdm.auto import trange

from cornac.models.recommender import NextItemRecommender

from ...utils import get_rng


def _diffgrm_num_training_examples(train_set, min_history=2, max_len=50):
    """Count prefix-to-next-item examples in a sequential training split."""
    return sum(
        max(0, min(len(mapped_ids), max_len + 1) - min_history)
        for mapped_ids in train_set.sessions.values()
    )


def _diffgrm_session_iter(
    train_set,
    pad_index,
    batch_size=256,
    max_len=50,
    min_history=2,
    rng=None,
    shuffle=True,
):
    """Yield one training row per eligible next-item prefix."""
    rng = rng if rng is not None else get_rng(None)
    examples = []
    for sid, mapped_ids in train_set.sessions.items():
        target_stop = min(len(mapped_ids), max_len + 1)
        for target_position in range(min_history, target_stop):
            examples.append((sid, target_position))
    if shuffle:
        rng.shuffle(examples)

    uir_tuple = train_set.uir_tuple
    histories, masks, targets = [], [], []
    for sid, target_position in examples:
        mapped_ids = train_set.sessions[sid]
        items = np.asarray(uir_tuple[1][mapped_ids], dtype="int64")
        history = items[max(0, target_position - max_len) : target_position]
        input_iids = np.full(max_len, pad_index, dtype="int64")
        input_iids[: len(history)] = history
        attention_mask = np.zeros(max_len, dtype="float32")
        attention_mask[: len(history)] = 1.0

        histories.append(input_iids)
        masks.append(attention_mask)
        targets.append(items[target_position])
        if len(histories) == batch_size:
            yield (
                np.asarray(histories, dtype="int64"),
                np.asarray(masks, dtype="float32"),
                np.asarray(targets, dtype="int64"),
            )
            histories, masks, targets = [], [], []

    if histories:
        yield (
            np.asarray(histories, dtype="int64"),
            np.asarray(masks, dtype="float32"),
            np.asarray(targets, dtype="int64"),
        )


class DiffGRM(NextItemRecommender):
    """DiffGRM: Diffusion-based Generative Recommendation Model.

    DiffGRM combines parallel OPQ semantic IDs (PSE), on-policy
    nested masking (OCN), and confidence-prioritized decoding (CPD). Item
    content embeddings are supplied through Cornac's ``FeatureModality``.
    Precomputed semantic IDs may instead be passed through ``item_sids`` for
    artifact-reproduction and tokenizer-controlled experiments.

    Parameters
    ----------
    n_digit: int, default: 4
        Number of semantic-ID digits.
    codebook_size: int, default: 256
        Number of codes per digit. PSE uses 8-bit FAISS PQ and therefore
        requires 256. Smaller codebooks remain useful with ``item_sids``.
    pca_dim: int, default: 256
        Whitened PCA dimension before OPQ. It is reduced, if necessary, to the
        largest valid multiple of ``n_digit``.
    faiss_omp_num_threads: int, default: 32
        CPU threads used by FAISS while fitting and applying OPQ/PQ, matching
        the released configuration.
    item_sids: array-like, optional
        Precomputed un-offset semantic IDs with shape ``(n_items, n_digit)``.
    d_model, encoder_n_layer, decoder_n_layer, n_head, n_inner:
        Transformer architecture. Paper-style dataset configs are exported
        next to this class.
    max_len: int, default: 50
        Maximum number of history items.
    min_history: int, default: 2
        Shortest prefix used as a training example.
    masking_strategy: {'guided', 'random', 'coherent', 'fixed'}, default: 'guided'
        ``guided`` is OCN. ``random`` independently masks digits (without
        OCN), ``coherent`` uses nested random-order masks (without on-policy),
        and ``fixed`` always masks lower digit indices first.
    confidence_method: {'msp', 'entropy'}, default: 'msp'
        Confidence statistic used to rank the hardest digits.
    random_mask_prob: float, default: 0.5
        Per-digit probability for each independent no-OCN random view.
    n_views: int, optional
        Number of nested masks per target; defaults to ``n_digit``.
    scoring: {'released', 'paper', 'catalog', 'fixed'}, default: 'paper'
        ``released`` uses the official final-digit greedy completion and then
        filters complete IDs to the catalog. ``paper`` follows Equations 8--10
        with global beam selection at every digit.
        ``catalog`` additionally constrains every partial assignment.
        ``fixed`` is the no-CPD, fixed-permutation beam control.
    fixed_decode_order: sequence of int, optional
        Digit permutation used by ``scoring='fixed'``. A seeded permutation is
        generated once during fitting when omitted.
    collision_policy: {'all', 'first', 'last'}, default: 'all'
        Whether all items sharing a semantic ID receive its score, or only the
        lowest- or highest-index item. ``last`` matches the released reverse
        mapping's overwrite behavior; ``all`` preserves collisions explicitly.
    model_selection: {'last', 'best'}, default: 'best'
        ``best`` selects by SID-level
        ``0.8 * NDCG@k + 0.2 * Recall@k``.

    References
    ----------
    Liu et al. (2026). DiffGRM: Diffusion-based Generative Recommendation
    Model. WWW. https://arxiv.org/abs/2510.21805
    """

    def __init__(
        self,
        name="DiffGRM",
        n_digit=4,
        codebook_size=256,
        pca_dim=256,
        faiss_omp_num_threads=32,
        feature_standardize=False,
        normalize_after_pca=True,
        item_sids=None,
        d_model=256,
        encoder_n_layer=1,
        decoder_n_layer=4,
        n_head=4,
        n_inner=1024,
        dropout=0.1,
        activation="gelu",
        layer_norm_eps=1e-5,
        initializer_range=0.02,
        max_len=50,
        min_history=2,
        masking_strategy="guided",
        confidence_method="msp",
        random_mask_prob=0.5,
        n_views=None,
        label_smoothing=0.1,
        view_loss_reduction="view_mean",
        n_epochs=20,
        learning_rate=0.003,
        weight_decay=0.0,
        batch_size=256,
        max_grad_norm=1.0,
        lr_schedule="cosine",
        warmup_steps=10000,
        scoring="paper",
        beam_size=128,
        fixed_decode_order=None,
        collision_policy="all",
        model_selection="best",
        val_k=10,
        val_batch_size=32,
        val_beam_size=None,
        val_eval_start=1,
        val_eval_every=1,
        early_stopping_patience=None,
        val_sample=2000,
        device="auto",
        trainable=True,
        verbose=False,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        if n_digit <= 0 or codebook_size <= 1:
            raise ValueError("n_digit must be positive and codebook_size > 1")
        if faiss_omp_num_threads <= 0:
            raise ValueError("faiss_omp_num_threads must be positive")
        if encoder_n_layer <= 0 or decoder_n_layer <= 0:
            raise ValueError("encoder_n_layer and decoder_n_layer must be positive")
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if max_len <= 0 or min_history <= 0:
            raise ValueError("max_len and min_history must be positive")
        if n_views is not None and not 1 <= n_views <= n_digit:
            raise ValueError("n_views must be between 1 and n_digit")
        if masking_strategy not in ("guided", "random", "coherent", "fixed"):
            raise ValueError(
                "masking_strategy must be 'guided', 'random', 'coherent', or 'fixed'"
            )
        if confidence_method not in ("msp", "entropy"):
            raise ValueError("confidence_method must be 'msp' or 'entropy'")
        if not 0.0 < random_mask_prob <= 1.0:
            raise ValueError("random_mask_prob must be in (0, 1]")
        if view_loss_reduction not in ("view_mean", "token_mean"):
            raise ValueError("view_loss_reduction must be 'view_mean' or 'token_mean'")
        if scoring not in ("released", "paper", "catalog", "fixed"):
            raise ValueError(
                "scoring must be 'released', 'paper', 'catalog', or 'fixed'"
            )
        if collision_policy not in ("all", "first", "last"):
            raise ValueError("collision_policy must be 'all', 'first', or 'last'")
        if fixed_decode_order is not None and sorted(fixed_decode_order) != list(
            range(n_digit)
        ):
            raise ValueError("fixed_decode_order must be a permutation of all digits")
        if lr_schedule not in ("constant", "cosine"):
            raise ValueError("lr_schedule must be 'constant' or 'cosine'")
        if model_selection not in ("last", "best"):
            raise ValueError("model_selection must be 'last' or 'best'")
        if val_batch_size <= 0 or val_eval_start <= 0 or val_eval_every <= 0:
            raise ValueError(
                "val_batch_size, val_eval_start, and val_eval_every must be positive"
            )
        if early_stopping_patience is not None and early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive or None")
        if n_epochs <= 0 or batch_size <= 0 or beam_size <= 0 or val_k <= 0:
            raise ValueError(
                "n_epochs, batch_size, beam_size, and val_k must be positive"
            )
        if val_beam_size is not None and val_beam_size <= 0:
            raise ValueError("val_beam_size must be positive or None")

        self.n_digit = n_digit
        self.codebook_size = codebook_size
        self.pca_dim = pca_dim
        self.faiss_omp_num_threads = faiss_omp_num_threads
        self.feature_standardize = feature_standardize
        self.normalize_after_pca = normalize_after_pca
        self.item_sids = item_sids
        self.d_model = d_model
        self.encoder_n_layer = encoder_n_layer
        self.decoder_n_layer = decoder_n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.max_len = max_len
        self.min_history = min_history
        self.masking_strategy = masking_strategy
        self.confidence_method = confidence_method
        self.random_mask_prob = random_mask_prob
        self.n_views = n_digit if n_views is None else n_views
        self.label_smoothing = label_smoothing
        self.view_loss_reduction = view_loss_reduction
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.lr_schedule = lr_schedule
        self.warmup_steps = warmup_steps
        self.scoring = scoring
        self.beam_size = beam_size
        self.fixed_decode_order = fixed_decode_order
        self.collision_policy = collision_policy
        self.model_selection = model_selection
        self.val_k = val_k
        self.val_batch_size = val_batch_size
        self.val_beam_size = val_beam_size
        self.val_eval_start = val_eval_start
        self.val_eval_every = val_eval_every
        self.early_stopping_patience = early_stopping_patience
        self.val_sample = val_sample
        self.device = device
        self.seed = seed
        self.rng = get_rng(seed)

    def _get_item_features(self):
        item_feature = getattr(self.train_set, "item_feature", None)
        features = getattr(item_feature, "features", None)
        if features is None:
            raise ValueError(
                "DiffGRM requires item content embeddings unless item_sids are "
                "provided. Attach FeatureModality through "
                "NextItemEvaluation.from_splits(..., item_feature=...)."
            )
        if features.shape[0] < self.total_items:
            raise ValueError(
                f"item_feature has {features.shape[0]} rows but "
                f"{self.total_items} items are known"
            )
        features = np.asarray(features[: self.total_items], dtype="float32")
        if not np.isfinite(features).all():
            raise ValueError("item_feature contains NaN or infinite values")
        return features

    def _pse_tokenize(self, features, train_mask):
        """Whitened PCA followed by position-sensitive OPQ/PQ codes."""
        import faiss
        from sklearn.decomposition import PCA

        if self.codebook_size != 256:
            raise ValueError(
                "PSE uses 8-bit PQ; codebook_size must be 256 unless "
                "precomputed item_sids are supplied"
            )
        train_features = features[train_mask]
        if len(train_features) < 2:
            raise ValueError("PSE needs at least two training items")
        if len(train_features) < self.codebook_size:
            raise ValueError(
                "PSE needs at least codebook_size training items; provide "
                "precomputed item_sids for smaller diagnostics"
            )

        if self.feature_standardize:
            self.feature_mean_ = train_features.mean(axis=0)
            self.feature_std_ = train_features.std(axis=0)
            self.feature_std_[self.feature_std_ == 0] = 1.0
            features = (features - self.feature_mean_) / self.feature_std_
            train_features = features[train_mask]

        requested_dim = features.shape[1] if self.pca_dim <= 0 else self.pca_dim
        n_components = min(requested_dim, features.shape[1], len(train_features) - 1)
        n_components -= n_components % self.n_digit
        if n_components < self.n_digit:
            raise ValueError(
                "PCA output dimension must be at least n_digit and divisible by it"
            )
        self.pca_ = PCA(
            n_components=n_components,
            whiten=True,
            random_state=self.seed,
        )
        features = self.pca_.fit(train_features).transform(features)
        features = np.asarray(features, dtype="float32")
        if self.normalize_after_pca:
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            features = features / np.maximum(norms, 1e-12)

        features = np.ascontiguousarray(features, dtype="float32")
        train_features = np.ascontiguousarray(features[train_mask], dtype="float32")
        factory = f"OPQ{self.n_digit},IVF1,PQ{self.n_digit}x8"
        faiss.omp_set_num_threads(self.faiss_omp_num_threads)
        index = faiss.index_factory(n_components, factory, faiss.METRIC_INNER_PRODUCT)
        index_ivf = faiss.downcast_index(faiss.extract_index_ivf(index))
        index.train(train_features)
        index.add(features)

        inverted = index_ivf.invlists
        list_size = inverted.list_size(0)
        if list_size != self.total_items:
            raise RuntimeError(
                f"FAISS encoded {list_size} items, expected {self.total_items}"
            )
        code_size = inverted.code_size
        codes = faiss.rev_swig_ptr(
            inverted.get_codes(0), list_size * code_size
        ).reshape(list_size, code_size)[:, : self.n_digit]
        ids = faiss.rev_swig_ptr(inverted.get_ids(0), list_size).copy()
        sid_table = np.empty((self.total_items, self.n_digit), dtype="int64")
        sid_table[ids] = codes.astype("int64")
        return sid_table

    def _prepare_semantic_ids(self):
        if self.item_sids is not None:
            sid_table = np.asarray(self.item_sids, dtype="int64")
            expected = (self.total_items, self.n_digit)
            if sid_table.shape != expected:
                raise ValueError(
                    f"item_sids must have shape {expected}, got {sid_table.shape}"
                )
            if sid_table.size and (
                sid_table.min() < 0 or sid_table.max() >= self.codebook_size
            ):
                raise ValueError(
                    f"item_sids digits must be in [0, {self.codebook_size})"
                )
            return sid_table.copy()

        features = self._get_item_features()
        train_mask = self._training_item_mask()
        return self._pse_tokenize(features, train_mask)

    def _training_item_mask(self):
        """Items exposed by the released prefix-augmentation training rows."""
        train_mask = np.zeros(self.total_items, dtype=bool)
        item_indices = self.train_set.uir_tuple[1]
        for mapped_ids in self.train_set.sessions.values():
            stop = min(len(mapped_ids), self.max_len + 1)
            if stop <= self.min_history:
                continue
            items = np.asarray(item_indices[mapped_ids[:stop]], dtype="int64")
            train_mask[items] = True
        return train_mask

    def _build_model(self):
        from .diffgrm import DiffGRMBackbone

        model = DiffGRMBackbone(
            n_digit=self.n_digit,
            codebook_size=self.codebook_size,
            max_len=self.max_len,
            d_model=self.d_model,
            encoder_n_layer=self.encoder_n_layer,
            decoder_n_layer=self.decoder_n_layer,
            n_head=self.n_head,
            n_inner=self.n_inner,
            dropout=self.dropout,
            activation=self.activation,
            layer_norm_eps=self.layer_norm_eps,
            initializer_range=self.initializer_range,
            masking_strategy=self.masking_strategy,
            confidence_method=self.confidence_method,
            random_mask_prob=self.random_mask_prob,
            n_views=self.n_views,
            label_smoothing=self.label_smoothing,
            view_loss_reduction=self.view_loss_reduction,
        ).to(self.device_)
        model.set_item_codes(self.sid_table)
        return model

    def _make_scheduler(self, torch, optimizer):
        if self.lr_schedule == "constant":
            return None
        n_examples = _diffgrm_num_training_examples(
            self.train_set, self.min_history, self.max_len
        )
        total_steps = max(1, math.ceil(n_examples / self.batch_size) * self.n_epochs)

        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(
                1, total_steps - self.warmup_steps
            )
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _val_sessions(self, val_set):
        sessions = []
        for [_], [mapped_ids], [items] in val_set.si_iter(batch_size=1, shuffle=False):
            if len(items) < 2:
                continue
            user_idx = int(val_set.uir_tuple[0][mapped_ids[0]])
            sessions.append((user_idx, [int(i) for i in items]))
        if self.val_sample is not None and len(sessions) > self.val_sample:
            indices = self.rng.choice(len(sessions), self.val_sample, replace=False)
            sessions = [sessions[i] for i in sorted(indices)]
        return sessions

    def _validation_score(self, sessions):
        import torch

        from .diffgrm import cpd_decode_batch

        ndcg_values, recall_values = [], []
        self._ensure_device(torch)
        self.model.eval()
        for start in range(0, len(sessions), self.val_batch_size):
            batch = sessions[start : start + self.val_batch_size]
            input_iids = np.full(
                (len(batch), self.max_len), self.pad_idx, dtype="int64"
            )
            attention_mask = np.zeros((len(batch), self.max_len), dtype="float32")
            for row, (_, items) in enumerate(batch):
                history = items[:-1][-self.max_len :]
                input_iids[row, : len(history)] = history
                attention_mask[row, : len(history)] = 1.0
            inputs = torch.as_tensor(input_iids, dtype=torch.long, device=self.device_)
            masks = torch.as_tensor(
                attention_mask, dtype=torch.float32, device=self.device_
            )
            with torch.no_grad():
                memory, padding_mask = self.model.encode_history(inputs, masks)
                batch_codes, _, batch_diagnostics = cpd_decode_batch(
                    self.model,
                    memory,
                    padding_mask,
                    beam_size=(
                        self.beam_size
                        if self.val_beam_size is None
                        else self.val_beam_size
                    ),
                    catalog_codes=self.sid_table,
                    valid_code_set=self.sid_to_items_,
                    constrained=self.scoring == "catalog",
                    digit_order=self.fixed_decode_order_
                    if self.scoring == "fixed"
                    else None,
                    greedy_final=self.scoring == "released",
                    return_diagnostics=True,
                )
            self.last_decode_diagnostics_ = batch_diagnostics[-1]
            for (_, items), codes in zip(batch, batch_codes):
                target = items[-1]
                if target >= self.total_items:
                    continue
                target_sid = tuple(int(digit) for digit in self.sid_table[target])
                rank = next(
                    (
                        index
                        for index, sid in enumerate(codes)
                        if tuple(int(digit) for digit in sid) == target_sid
                    ),
                    None,
                )
                hit = rank is not None and rank < self.val_k
                recall_values.append(float(hit))
                ndcg_values.append(1.0 / np.log2(rank + 2) if hit else 0.0)
        if not ndcg_values:
            return 0.0
        return 0.8 * float(np.mean(ndcg_values)) + 0.2 * float(np.mean(recall_values))

    def _fit_model(self, torch, val_set):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = self._make_scheduler(torch, optimizer)
        select_best = self.model_selection == "best" and val_set is not None
        val_sessions = self._val_sessions(val_set) if select_best else []
        best_state, best_value = None, -float("inf")
        non_improving = 0
        self.loss_history_ = []

        progress = trange(
            1, self.n_epochs + 1, disable=not self.verbose, desc="DiffGRM"
        )
        for epoch in progress:
            self.current_epoch = epoch
            self.model.train()
            total_loss, n_batches = 0.0, 0
            for input_iids, attention_mask, target_iids in _diffgrm_session_iter(
                self.train_set,
                pad_index=self.pad_idx,
                batch_size=self.batch_size,
                max_len=self.max_len,
                min_history=self.min_history,
                rng=self.rng,
                shuffle=True,
            ):
                inputs = torch.as_tensor(
                    input_iids, dtype=torch.long, device=self.device_
                )
                masks = torch.as_tensor(
                    attention_mask, dtype=torch.float32, device=self.device_
                )
                targets = torch.as_tensor(
                    self.sid_table[target_iids],
                    dtype=torch.long,
                    device=self.device_,
                )
                optimizer.zero_grad()
                loss = self.model(inputs, masks, targets)
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                total_loss += loss.item()
                n_batches += 1
                progress.set_postfix(loss=total_loss / n_batches)
            self.loss_history_.append(total_loss / n_batches)

            if (
                select_best
                and epoch >= self.val_eval_start
                and (epoch - self.val_eval_start) % self.val_eval_every == 0
            ):
                self.model.eval()
                value = self._validation_score(val_sessions)
                if value > best_value:
                    best_value = value
                    non_improving = 0
                    self.best_value = value
                    self.best_epoch = epoch
                    self.wait = 0
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in self.model.state_dict().items()
                    }
                else:
                    non_improving += 1
                    self.wait = non_improving
                    if (
                        self.early_stopping_patience is not None
                        and non_improving >= self.early_stopping_patience
                    ):
                        self.stopped_epoch = epoch
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)
        if not self.trainable:
            return self

        import torch

        torch.manual_seed(0 if self.seed is None else self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0 if self.seed is None else self.seed)
        self.device_ = (
            "cuda"
            if self.device == "auto" and torch.cuda.is_available()
            else "cpu"
            if self.device == "auto"
            else self.device
        )
        self.pad_idx = self.total_items
        self.n_training_examples_ = _diffgrm_num_training_examples(
            self.train_set, self.min_history, self.max_len
        )
        if self.n_training_examples_ == 0:
            raise ValueError(
                "DiffGRM found no training prefixes; reduce min_history or "
                "provide longer sessions"
            )
        tokenizer_start = time.perf_counter()
        self.sid_table = self._prepare_semantic_ids()
        self.tokenizer_time_ = time.perf_counter() - tokenizer_start
        self.sid_hash_ = hashlib.sha256(self.sid_table.tobytes()).hexdigest()
        self.sid_to_items_ = {}
        for item_idx, codes in enumerate(self.sid_table):
            self.sid_to_items_.setdefault(tuple(codes.tolist()), []).append(item_idx)
        self.unique_sid_count_ = len(self.sid_to_items_)
        self.sid_collision_count_ = self.total_items - self.unique_sid_count_
        collision_sizes = [len(items) for items in self.sid_to_items_.values()]
        self.sid_collision_group_count_ = sum(size > 1 for size in collision_sizes)
        self.sid_max_collision_size_ = max(collision_sizes, default=0)
        self.sid_digit_utilization_ = np.asarray(
            [len(np.unique(self.sid_table[:, digit])) for digit in range(self.n_digit)],
            dtype="int64",
        )
        digit_entropies = []
        for digit in range(self.n_digit):
            counts = np.bincount(self.sid_table[:, digit], minlength=self.codebook_size)
            probabilities = counts[counts > 0] / counts.sum()
            digit_entropies.append(
                -float(np.sum(probabilities * np.log(probabilities)))
            )
        self.sid_digit_entropy_ = np.asarray(digit_entropies)
        self.fixed_decode_order_ = (
            tuple(int(d) for d in self.fixed_decode_order)
            if self.fixed_decode_order is not None
            else tuple(int(d) for d in get_rng(self.seed).permutation(self.n_digit))
        )

        self.model = self._build_model()
        training_start = time.perf_counter()
        self._fit_model(torch, val_set)
        self.training_time_ = time.perf_counter() - training_start
        self.model.to("cpu").eval()
        return self

    def _ensure_device(self, torch):
        requested = torch.device(self.device_)
        if requested.type == "cuda" and not torch.cuda.is_available():
            requested = torch.device("cpu")
            self.device_ = "cpu"
        if next(self.model.parameters()).device != requested:
            self.model.to(requested)

    def _score_history(self, history_items):
        import torch

        from .diffgrm import cpd_decode

        if not history_items:
            return np.ones(self.total_items, dtype="float")
        self._ensure_device(torch)
        history = list(history_items)[-self.max_len :]
        input_iids = np.full((1, self.max_len), self.pad_idx, dtype="int64")
        input_iids[0, : len(history)] = history
        attention_mask = np.zeros((1, self.max_len), dtype="float32")
        attention_mask[0, : len(history)] = 1.0
        inputs = torch.as_tensor(input_iids, dtype=torch.long, device=self.device_)
        masks = torch.as_tensor(
            attention_mask, dtype=torch.float32, device=self.device_
        )
        self.model.eval()
        decode_start = time.perf_counter()
        with torch.no_grad():
            memory, padding_mask = self.model.encode_history(inputs, masks)
            codes, path_scores, diagnostics = cpd_decode(
                self.model,
                memory,
                padding_mask,
                beam_size=self.beam_size,
                catalog_codes=self.sid_table,
                valid_code_set=self.sid_to_items_,
                constrained=self.scoring == "catalog",
                digit_order=self.fixed_decode_order_
                if self.scoring == "fixed"
                else None,
                greedy_final=self.scoring == "released",
                return_diagnostics=True,
            )
        self.last_decode_time_ = time.perf_counter() - decode_start
        self.last_decode_diagnostics_ = diagnostics
        return self._decoded_item_scores(codes, path_scores)

    def _decoded_item_scores(self, codes, path_scores):
        scores = np.full(self.total_items, -1e10, dtype="float")
        for sid, path_score in zip(codes, path_scores):
            item_indices = self.sid_to_items_[tuple(sid)]
            if self.collision_policy == "first":
                item_indices = item_indices[:1]
            elif self.collision_policy == "last":
                item_indices = item_indices[-1:]
            scores[item_indices] = path_score
        return scores

    def score(self, user_idx, history_items, **kwargs):
        return self._score_history(history_items)
