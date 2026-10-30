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

import numpy as np
from tqdm.auto import trange

from ..recommender import NextItemRecommender, SemanticIDMixin

from ...utils import get_rng

SUPPORTED_SCORING = ("graph", "exact")
SUPPORTED_LR_SCHEDULES = ("constant", "cosine")
RPG_IGNORE_INDEX = -100


def _rpg_num_training_rows(train_set, max_len):
    """Number of rows produced by the official RPG windowing scheme."""
    return sum(
        max(len(mapped_ids) - max_len, 1)
        for mapped_ids in train_set.sessions.values()
        if len(mapped_ids) >= 2
    )


def _rpg_session_iter(
    train_set,
    pad_index,
    batch_size=256,
    max_len=50,
    rng=None,
    shuffle=True,
):
    """Yield official RPG causal-session training rows.

    A session of at most ``max_len + 1`` items becomes one right-padded row:
    ``items[:-1]`` predicts ``items[1:]`` at every valid position. For a longer
    session, the first window supervises every position and each subsequent
    sliding window supervises only its last position. This covers every
    next-item target exactly once while bounding the GPT input length.
    """
    rng = rng if rng is not None else get_rng(None)
    sessions = train_set.sessions
    examples = []
    for sid, mapped_ids in sessions.items():
        if len(mapped_ids) < 2:
            continue
        for start in range(max(len(mapped_ids) - max_len, 1)):
            examples.append((sid, start))
    if shuffle:
        rng.shuffle(examples)

    uir_tuple = train_set.uir_tuple
    buffer_uids, buffer_inputs, buffer_labels = [], [], []
    for sid, start in examples:
        mapped_ids = sessions[sid]
        items = np.asarray(uir_tuple[1][mapped_ids], dtype="int64")
        window = items[start : start + max_len + 1]
        n_inputs = len(window) - 1

        input_iids = np.full(max_len, pad_index, dtype="int64")
        input_iids[:n_inputs] = window[:-1]
        labels = np.full(max_len, RPG_IGNORE_INDEX, dtype="int64")
        if start == 0:
            labels[:n_inputs] = window[1:]
        else:
            labels[n_inputs - 1] = window[-1]

        buffer_uids.append(int(uir_tuple[0][mapped_ids[0]]))
        buffer_inputs.append(input_iids)
        buffer_labels.append(labels)
        if len(buffer_uids) == batch_size:
            yield (
                np.asarray(buffer_uids, dtype="int64"),
                np.asarray(buffer_inputs, dtype="int64"),
                np.asarray(buffer_labels, dtype="int64"),
            )
            buffer_uids, buffer_inputs, buffer_labels = [], [], []

    if buffer_uids:
        yield (
            np.asarray(buffer_uids, dtype="int64"),
            np.asarray(buffer_inputs, dtype="int64"),
            np.asarray(buffer_labels, dtype="int64"),
        )


class RPG(NextItemRecommender, SemanticIDMixin):
    """RPG: Generating Long Semantic IDs in Parallel for Recommendation.

    RPG is the architectural counterpoint to TIGER. Instead of short *ordered*
    RQ-VAE semantic IDs decoded autoregressively, RPG quantizes precomputed
    item content embeddings into **long unordered** semantic IDs with **product
    quantization (OPQ, via faiss)** and predicts **all digits in parallel** with
    a multi-token-prediction (MTP) loss. A GPT-2 decoder consumes a session as a
    sequence of items (each item embedded as the mean of its ``n_codebook``
    semantic-ID token embeddings); one residual head per codebook predicts every
    digit of every supervised next item's semantic ID at once, scored by
    temperature-scaled cosine similarity to the shared token-embedding table.
    Inference uses a similarity graph over item semantic IDs to guide beam
    decoding to valid items, so scoring cost is independent of the corpus size.

    Item content embeddings must be provided through the evaluation method,
    e.g.::

        NextItemEvaluation.from_splits(
            ..., item_feature=FeatureModality(features=embs, ids=item_ids)
        )

    where ``embs`` are precomputed text/content embeddings covering every known
    item. A ready-made :data:`~cornac.models.rpg.RPG_CONFIG` (the official
    Amazon-2014 recipe) ships with the model.

    Parameters
    ----------
    name: str, default: 'RPG'
        The name of the recommender model.

    n_codebook: int, default: 32
        Number of OPQ codebooks = the semantic-ID length (number of digits).
        This is the knob behind RPG's "scaling ID length" result: set to 64 for
        the long-ID configuration.

    codebook_size: int, default: 256
        Number of codes per codebook (PQ uses 8 bits, hence 256).

    pca_dim: int, default: 512
        Target dimensionality of the whitened PCA applied to the item
        embeddings before OPQ. Values ``<= 0`` or ``>=`` the embedding
        dimension skip PCA.

    feature_standardize: bool, default: False
        When True, z-score the item features per dimension before PCA/OPQ.

    d_model: int, default: 448
    n_layer: int, default: 2
    n_head: int, default: 4
    n_inner: int, default: 1024
    activation: str, default: 'gelu_new'
    resid_dropout: float, default: 0.0
    embd_dropout: float, default: 0.5
    attn_dropout: float, default: 0.5
    layer_norm_eps: float, default: 1e-12
    initializer_range: float, default: 0.02
        GPT-2 backbone architecture settings (defaults per the paper).

    max_len: int, default: 50
        Maximum number of history items fed to the backbone.

    temperature: float, default: 0.07
        Temperature of the cosine-similarity MTP logits (train and score).

    n_epochs: int, default: 20
    learning_rate: float, default: 3e-4
    weight_decay: float, default: 0.0
    batch_size: int, default: 256
    max_grad_norm: float or None, default: 1.0
        Backbone training settings (AdamW). The official recipe trains 150
        epochs with early stopping; ``RPG_CONFIG`` carries the full recipe.

    lr_schedule: str, default: 'constant'
        'constant' keeps ``learning_rate`` fixed; 'cosine' does linear warmup
        over ``warmup_steps`` then cosine decay (as in the official trainer).

    warmup_steps: int, default: 10000
        Linear-warmup steps when ``lr_schedule='cosine'``.

    scoring: str, default: 'graph'
        'graph' (paper-faithful similarity-graph-guided beam; only the final
        ``n_beams`` candidates get real scores)
        or 'exact' (full-catalog parallel-MTP score of every item; exact full
        ranking, cheap because the backbone runs once per user).

    n_beams: int, default: 50
        Beam width for scoring='graph'.

    graph_edges: int, default: 50
        Number of kNN neighbours per item in the similarity graph.

    propagation_steps: int, default: 3
        Number of graph-propagation steps during graph decoding.

    graph_chunk_size: int, default: 1024
        Item chunk size when building the similarity graph.

    model_selection: str, default: 'last'
        'last' or 'best'. When 'best' and a ``val_set`` is given, the backbone
        weights with the highest validation score (evaluated every
        ``val_eval_every`` epochs on up to ``val_sample`` val sessions, with
        batched exact scoring) are restored at the end of ``fit``.

    val_metric: str, default: 'ndcg'
    val_eval_every: int, default: 5
    val_batch_size: int, default: 32
    early_stopping_patience: int or None, default: None
    val_k: int, default: 10
    val_sample: int, default: 2000
        Metric, cadence, validation batch size, non-improving evaluation
        patience, cutoff K and session cap for best-on-val selection.

    device: str, default: 'auto'
        'auto' selects 'cuda' if available, otherwise 'cpu'.

    trainable: bool, default: True
    verbose: bool, default: False
    seed: int, default: None
        Random seed for weight init, OPQ, and graph decoding.

    References
    ----------
    Hou, Y. et al. (2025). Generating Long Semantic IDs in Parallel for
    Recommendation. KDD. https://arxiv.org/abs/2506.05781
    (official code: https://github.com/facebookresearch/RPG_KDD2025)
    """

    def __init__(
        self,
        name="RPG",
        n_codebook=32,
        codebook_size=256,
        pca_dim=512,
        feature_standardize=False,
        d_model=448,
        n_layer=2,
        n_head=4,
        n_inner=1024,
        activation="gelu_new",
        resid_dropout=0.0,
        embd_dropout=0.5,
        attn_dropout=0.5,
        layer_norm_eps=1e-12,
        initializer_range=0.02,
        max_len=50,
        temperature=0.07,
        n_epochs=20,
        learning_rate=3e-4,
        weight_decay=0.0,
        batch_size=256,
        max_grad_norm=1.0,
        lr_schedule="constant",
        warmup_steps=10000,
        scoring="graph",
        n_beams=50,
        graph_edges=50,
        propagation_steps=3,
        graph_chunk_size=1024,
        model_selection="last",
        val_metric="ndcg",
        val_eval_every=5,
        val_batch_size=32,
        early_stopping_patience=None,
        val_k=10,
        val_sample=2000,
        device="auto",
        trainable=True,
        verbose=False,
        seed=None,
    ):
        super().__init__(name, trainable=trainable, verbose=verbose)
        if scoring not in SUPPORTED_SCORING:
            raise ValueError(
                f"scoring='{scoring}' not supported; choose from {SUPPORTED_SCORING}"
            )
        if lr_schedule not in SUPPORTED_LR_SCHEDULES:
            raise ValueError(
                f"lr_schedule='{lr_schedule}' not supported; choose from {SUPPORTED_LR_SCHEDULES}"
            )
        if model_selection not in ("last", "best"):
            raise ValueError(
                f"model_selection='{model_selection}' not supported; choose 'last' or 'best'"
            )
        if val_eval_every <= 0:
            raise ValueError("val_eval_every must be positive")
        if val_batch_size <= 0:
            raise ValueError("val_batch_size must be positive")
        if early_stopping_patience is not None and early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive or None")
        self.n_codebook = n_codebook
        self.codebook_size = codebook_size
        self.pca_dim = pca_dim
        self.feature_standardize = feature_standardize
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation = activation
        self.resid_dropout = resid_dropout
        self.embd_dropout = embd_dropout
        self.attn_dropout = attn_dropout
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.max_len = max_len
        self.temperature = temperature
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.lr_schedule = lr_schedule
        self.warmup_steps = warmup_steps
        self.scoring = scoring
        self.n_beams = n_beams
        self.graph_edges = graph_edges
        self.propagation_steps = propagation_steps
        self.graph_chunk_size = graph_chunk_size
        self.model_selection = model_selection
        self.val_metric = val_metric
        self.val_eval_every = val_eval_every
        self.val_batch_size = val_batch_size
        self.early_stopping_patience = early_stopping_patience
        self.val_k = val_k
        self.val_sample = val_sample
        self.device = device
        self.seed = seed
        self.rng = get_rng(seed)
        self.graph_rng = get_rng(seed)

    def _get_item_features(self):
        item_feature = getattr(self.train_set, "item_feature", None)
        features = getattr(item_feature, "features", None)
        if features is None:
            raise ValueError(
                "RPG requires precomputed item content embeddings. Provide them "
                "via NextItemEvaluation.from_splits(..., item_feature="
                "FeatureModality(features=..., ids=...))."
            )
        if features.shape[0] < self.total_items:
            raise ValueError(
                f"item_feature has {features.shape[0]} rows but {self.total_items} "
                "items are known; every item (train/val/test) needs a feature vector."
            )
        return np.asarray(features[: self.total_items], dtype="float32")

    def _opq_tokenize(self, feats, train_mask=None):
        """OPQ (faiss) tokenizer -> (n_items, n_codebook) un-offset codes.

        Faithful to ``genrec/models/RPG/tokenizer.py``: whitened PCA to
        ``pca_dim`` then a faiss ``OPQ{M},IVF1,PQ{M}x8`` index; the per-item
        8-bit PQ codes are read straight out of the (single) inverted list and
        reordered into item order. Each byte is one digit's code in ``[0, 256)``.
        """
        import faiss
        from sklearn.decomposition import PCA

        if self.codebook_size != 256:
            raise ValueError(
                "the faiss OPQ path uses 8-bit PQ (256 codes per codebook); set "
                f"codebook_size=256 (got {self.codebook_size})."
            )
        n_items, dim = feats.shape
        if self.pca_dim and 0 < self.pca_dim < dim:
            n_comp = min(self.pca_dim, n_items - 1, dim)
            feats = PCA(n_components=n_comp, whiten=True).fit_transform(feats)
        feats = np.ascontiguousarray(feats, dtype="float32")
        if train_mask is None:
            train_feats = feats
        else:
            train_mask = np.asarray(train_mask, dtype=bool)
            if train_mask.shape != (n_items,):
                raise ValueError(
                    f"train_mask must have shape ({n_items},), got {train_mask.shape}"
                )
            train_feats = np.ascontiguousarray(feats[train_mask], dtype="float32")

        factory = f"OPQ{self.n_codebook},IVF1,PQ{self.n_codebook}x8"
        index = faiss.index_factory(
            feats.shape[1], factory, faiss.METRIC_INNER_PRODUCT
        )
        index.train(train_feats)
        index.add(feats)

        index_ivf = faiss.extract_index_ivf(index)
        invlists = index_ivf.invlists  # IVF1 -> a single inverted list (list 0)
        list_size = invlists.list_size(0)
        code_size = invlists.code_size  # bytes per item = n_codebook for 8-bit PQ
        codes = faiss.rev_swig_ptr(
            invlists.get_codes(0), list_size * code_size
        ).reshape(list_size, code_size)[:, : self.n_codebook]
        ids = faiss.rev_swig_ptr(invlists.get_ids(0), list_size).copy()
        sid_table = np.zeros((n_items, self.n_codebook), dtype="int64")
        sid_table[ids] = codes.astype("int64")
        return sid_table

    def _build_backbone(self):
        from .rpg import RPGBackbone

        model = RPGBackbone(
            n_codebook=self.n_codebook,
            codebook_size=self.codebook_size,
            max_len=self.max_len,
            d_model=self.d_model,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_inner=self.n_inner,
            activation=self.activation,
            resid_dropout=self.resid_dropout,
            embd_dropout=self.embd_dropout,
            attn_dropout=self.attn_dropout,
            layer_norm_eps=self.layer_norm_eps,
            initializer_range=self.initializer_range,
            temperature=self.temperature,
        ).to(self.device_)
        model.set_item_tokens(self.sid_table)
        return model

    def _make_lr_scheduler(self, torch, opt):
        if self.lr_schedule != "cosine":
            return None
        n_rows = _rpg_num_training_rows(self.train_set, self.max_len)
        steps_per_epoch = max(1, math.ceil(n_rows / self.batch_size))
        total_steps = max(1, steps_per_epoch * self.n_epochs)

        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(
                1, total_steps - self.warmup_steps
            )
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    def _make_val_metric(self):
        from ...metrics import AUC, MRR, NDCG, Recall

        name = self.val_metric.lower()
        if name == "recall":
            return Recall(k=self.val_k)
        if name == "ndcg":
            return NDCG(k=self.val_k)
        if name == "auc":
            return AUC()
        if name == "mrr":
            return MRR()
        raise ValueError(
            f"val_metric='{self.val_metric}' not supported; choose from recall/ndcg/auc/mrr"
        )

    def _val_sessions(self, val_set):
        sessions = []
        for [_], [mapped_ids], [session_items] in val_set.si_iter(
            batch_size=1, shuffle=False
        ):
            if len(session_items) < 2:
                continue
            user_idx = int(val_set.uir_tuple[0][mapped_ids[0]])
            sessions.append((user_idx, [int(i) for i in session_items]))
        if self.val_sample is not None and len(sessions) > self.val_sample:
            idx = self.rng.choice(len(sessions), size=self.val_sample, replace=False)
            sessions = [sessions[i] for i in sorted(idx)]
        return sessions

    def _exact_item_scores(self, logits, n_items=None):
        """Score catalog items from batched per-codebook logits."""
        import torch

        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        n_items = self.total_items if n_items is None else n_items
        codes = torch.as_tensor(
            self.sid_table[:n_items], dtype=torch.long, device=logits.device
        )
        scores = logits.new_zeros((logits.size(0), n_items))
        for digit in range(self.n_codebook):
            scores += logits[:, digit].index_select(1, codes[:, digit])
        return scores / self.n_codebook

    def _validate(self, val_sessions, metric):
        """Mean metric over last-item sessions using batched exact scoring."""
        import torch

        num_items = self.train_set.num_items
        item_indices = np.arange(num_items)
        results = []
        self.model.eval()
        device = next(self.model.parameters()).device
        for start in range(0, len(val_sessions), self.val_batch_size):
            batch = [
                session
                for session in val_sessions[start : start + self.val_batch_size]
                if session[1][-1] < num_items
            ]
            if not batch:
                continue
            input_ids = np.full(
                (len(batch), self.max_len), self.pad_idx, dtype="int64"
            )
            for row, (_, session_items) in enumerate(batch):
                history = session_items[:-1][-self.max_len :]
                input_ids[row, -len(history) :] = history
            input_ids = torch.as_tensor(input_ids, device=device)
            attention_mask = (input_ids != self.pad_idx).float()
            with torch.no_grad():
                logits = self.model.next_item_logits(input_ids, attention_mask)
                batch_scores = (
                    self._exact_item_scores(logits, num_items).cpu().numpy()
                )

            for row, (_, session_items) in enumerate(batch):
                target = session_items[-1]
                item_scores = batch_scores[row]
                item_rank = item_indices[item_scores.argsort()[::-1]]
                results.append(
                    metric.compute(
                        gt_pos=np.array([target]),
                        gt_neg=np.delete(item_indices, target),
                        pd_rank=item_rank,
                        pd_scores=item_scores,
                        item_indices=item_indices,
                    )
                )
        return float(np.mean(results)) if results else 0.0

    def _fit_backbone(self, torch, val_set):
        self.pad_idx = self.total_items
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = self._make_lr_scheduler(torch, opt)

        best_state, best_val = None, -float("inf")
        non_improving = 0
        select_best = self.model_selection == "best" and val_set is not None
        val_sessions = self._val_sessions(val_set) if select_best else None
        val_metric = self._make_val_metric() if select_best else None

        progress_bar = trange(1, self.n_epochs + 1, disable=not self.verbose, desc="RPG")
        for epoch_id in progress_bar:
            self.current_epoch = epoch_id
            self.model.train()
            total_loss, cnt = 0.0, 0
            for inc, (_, input_iids, labels) in enumerate(
                _rpg_session_iter(
                    self.train_set,
                    pad_index=self.pad_idx,
                    batch_size=self.batch_size,
                    max_len=self.max_len,
                    rng=self.rng,
                    shuffle=True,
                )
            ):
                input_ids = torch.tensor(
                    input_iids, dtype=torch.long, device=self.device_
                )
                attn_mask = (input_ids != self.pad_idx).float()
                target_iids = torch.tensor(
                    labels, dtype=torch.long, device=self.device_
                )
                opt.zero_grad()
                loss = self.model(input_ids, attn_mask, target_iids)
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                opt.step()
                if scheduler is not None:
                    scheduler.step()
                total_loss += loss.item()
                cnt += 1
                if inc % 10 == 0 and cnt > 0:
                    progress_bar.set_postfix(loss=(total_loss / cnt))

            if select_best and epoch_id % self.val_eval_every == 0:
                score = self._validate(val_sessions, val_metric)
                if score > best_val:
                    best_val = score
                    non_improving = 0
                    self.best_value = score
                    self.best_epoch = epoch_id
                    self.wait = 0
                    best_state = {
                        n: p.detach().clone()
                        for n, p in self.model.state_dict().items()
                    }
                else:
                    non_improving += 1
                    self.wait = non_improving
                    if (
                        self.early_stopping_patience is not None
                        and non_improving >= self.early_stopping_patience
                    ):
                        self.stopped_epoch = epoch_id
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)
        if not self.trainable:
            return self

        import torch

        torch.manual_seed(self.seed if self.seed is not None else 0)
        self.device_ = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if self.device == "auto"
            else self.device
        )

        feats = self._get_item_features()
        if self.feature_standardize:
            mean = feats.mean(axis=0)
            std = feats.std(axis=0)
            std[std == 0] = 1.0
            feats = ((feats - mean) / std).astype("float32")

        train_mask = np.zeros(self.total_items, dtype=bool)
        train_mask[np.asarray(self.train_set.uir_tuple[1], dtype="int64")] = True
        self.sid_table = self._opq_tokenize(feats, train_mask)
        self.item_tokens = (
            self.sid_table + self.codebook_size * np.arange(self.n_codebook) + 1
        ).astype("int64")

        self.model = self._build_backbone()
        self._fit_backbone(torch, val_set)

        self.model.eval()
        self.adjacency = self.model.build_adjacency(
            self.total_items, self.graph_edges, self.graph_chunk_size
        )
        # keep pickles portable across GPU/CPU boxes; moved back in score()
        self.model.to("cpu").eval()
        return self

    def _ensure_device(self, torch):
        if self.device_ == "cuda" and not torch.cuda.is_available():
            self.device_ = "cpu"
        if next(self.model.parameters()).device.type != torch.device(self.device_).type:
            self.model.to(self.device_)

    def _predict_history(self, history_items, return_semantic_ids=False):
        import torch

        from .rpg import graph_decode

        if len(history_items) == 0:
            scores = np.ones(self.total_items, dtype="float")
            if return_semantic_ids:
                return scores, np.empty((0, self.sid_table.shape[1]), dtype="int64")
            return scores
        self._ensure_device(torch)
        hist = list(history_items)[-self.max_len :]
        hist = [self.pad_idx] * (self.max_len - len(hist)) + hist
        input_ids = torch.tensor([hist], dtype=torch.long, device=self.device_)
        attn_mask = (input_ids != self.pad_idx).float()
        self.model.eval()
        with torch.no_grad():
            logits = self.model.next_item_logits(input_ids, attn_mask)[0]  # (M, K)

        if self.scoring == "exact":
            scores = self._exact_item_scores(logits)[0].cpu().numpy().astype("float")
            if return_semantic_ids:
                return scores, self._semantic_ids_from_scores(scores)
            return scores

        # graph-guided beam decoding over the final propagated beam
        flat = logits.reshape(-1).cpu().numpy()
        candidates, candidate_scores = graph_decode(
            flat,
            self.adjacency,
            self.item_tokens,
            self.n_beams,
            self.propagation_steps,
            self.graph_rng,
        )
        scores = np.full(self.total_items, -1e10, dtype="float")
        scores[candidates] = candidate_scores
        if return_semantic_ids:
            return scores, self.sid_table[candidates]
        return scores

    def semantic_id_catalog(self):
        """Return complete, un-offset OPQ codes aligned to item indices."""
        if not hasattr(self, "sid_table"):
            raise ValueError("RPG must be fitted before exporting its SID catalog")
        return self.sid_table

    def _score_with_semantic_ids(self, user_idx, history_items, **kwargs):
        return self._predict_history(history_items, return_semantic_ids=True)

    def score(self, user_idx, history_items, **kwargs):
        return self._predict_history(history_items)
