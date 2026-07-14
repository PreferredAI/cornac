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
from collections import defaultdict

import numpy as np
from tqdm.auto import trange

from cornac.models.recommender import NextItemRecommender

from ...utils import get_rng
from ..seq_utils import session_seq_iter

SUPPORTED_SCORING = ("beam", "exact")
SUPPORTED_TOKENIZERS = ("rqvae", "rkmeans")
SUPPORTED_LR_SCHEDULES = ("constant", "cosine")


class TIGER(NextItemRecommender):
    """TIGER: Recommender Systems with Generative Retrieval.

    Two-stage generative retrieval with semantic IDs: (1) an RQ-VAE quantizes
    precomputed item content embeddings into hierarchical semantic IDs
    (``rqvae_num_levels`` codebooks of ``rqvae_codebook_size`` codes, plus an
    extra level that disambiguates collisions); (2) a T5-style encoder-decoder
    is trained to generate the next item's semantic ID from the session
    history's semantic-ID tokens.

    Item content embeddings must be provided through the evaluation method,
    e.g.::

        NextItemEvaluation.from_splits(
            ..., item_feature=FeatureModality(features=embs, ids=item_ids)
        )

    where ``embs`` are precomputed text/content embeddings (e.g., from
    sentence-transformers) covering every known item.

    Two ready-made configurations ship with the model:
    :data:`~cornac.models.tiger.GRID_CONFIG` (GRID handbook recipe — fast,
    no tokenizer training) and :data:`~cornac.models.tiger.PAISCHER_CONFIG`
    (Paischer et al. recipe — best documented reproduction accuracy), e.g.
    ``TIGER(**{**PAISCHER_CONFIG, "seed": 123})``.

    Parameters
    ----------
    name: string, default: 'TIGER'
        The name of the recommender model.

    tokenizer: str, optional, default: 'rqvae'
        Semantic-ID tokenizer. 'rqvae' trains the residual-quantized VAE
        (paper-faithful); 'rkmeans' runs residual k-means directly on the item
        features (level-by-level k-means on the residuals, no encoder/decoder
        and no gradient training), the simpler GRID-handbook baseline. Both
        share the collision-disambiguation level and the seq2seq stage.

    feature_standardize: bool, optional, default: False
        When True, z-score the item features per dimension (over items) before
        tokenizing. Paischer et al. standardize the RQ-VAE inputs.

    rqvae_latent_dim: int, optional, default: 32
        RQ-VAE latent (codeword) dimension.

    rqvae_hidden_dims: tuple of int, optional, default: (512, 256, 128)
        Hidden layer sizes of the RQ-VAE encoder (decoder mirrors them).

    rqvae_num_levels: int, optional, default: 3
        Number of residual codebooks (semantic-ID levels before the
        collision-disambiguation level).

    rqvae_codebook_size: int, optional, default: 256
        Number of codes per codebook.

    rqvae_beta: float, optional, default: 0.25
        Commitment loss weight of the RQ-VAE.

    rqvae_n_epochs: int, optional, default: 200
    rqvae_learning_rate: float, optional, default: 0.001
    rqvae_batch_size: int, optional, default: 1024
        RQ-VAE training settings.

    rqvae_weight_decay: float, optional, default: 0.0
        AdamW weight decay for RQ-VAE training (Paischer et al. use 0.1). The
        default 0.0 reproduces plain Adam.

    d_model: int, optional, default: 128
    d_ff: int, optional, default: 1024
    num_heads: int, optional, default: 6
    d_kv: int, optional, default: 64
    num_enc_layers: int, optional, default: 4
    num_dec_layers: int, optional, default: 4
    dropout: float, optional, default: 0.1
        Transformer (T5-style) architecture settings, defaults per the paper
        (~13M parameters).

    max_len: int, optional, default: 20
        Maximum number of history items fed to the encoder. The encoder input
        is ``max_len * (rqvae_num_levels + 1)`` tokens.

    n_epochs: int, optional, default: 20
    learning_rate: float, optional, default: 0.001
    weight_decay: float, optional, default: 0.0001
    batch_size: int, optional, default: 256
        Seq2seq training settings. Note: the paper uses Adagrad and an
        inverse-sqrt schedule; we use Adam/AdamW with constant lr (as in the
        GRID framework), which converges much faster at these model sizes.

    lr_schedule: str, optional, default: 'constant'
        Seq2seq learning-rate schedule. 'constant' keeps ``learning_rate``
        fixed; 'cosine' does linear warmup over ``warmup_steps`` steps then
        cosine decay to ~0 over the remaining total training steps
        (steps_per_epoch * n_epochs), as in Paischer et al.

    warmup_steps: int, optional, default: 10000
        Number of linear-warmup steps when ``lr_schedule='cosine'``.

    model_selection: str, optional, default: 'last'
        One of 'last' or 'best'. When 'best' and a ``val_set`` is given, the
        seq2seq weights with the highest validation score (evaluated every
        ``val_eval_every`` epochs on up to ``val_sample`` val sessions) are
        restored at the end of ``fit``. The tokenizer is fixed before seq2seq
        training, so only the seq2seq model is snapshotted.

    val_metric: str, optional, default: 'ndcg'
    val_eval_every: int, optional, default: 5
    val_k: int, optional, default: 10
        Metric, epoch cadence, and cutoff K used for best-on-val selection.
        During validation ``n_beams`` is raised to at least ``val_k``.

    val_sample: int, optional, default: 2000
        Maximum number of val sessions scored per evaluation (None = all).
        Beam-search scoring is expensive, so a fixed deterministic subsample
        (drawn once with the model's rng) is reused across epochs.

    scoring: str, optional, default: 'beam'
        'beam' (paper-faithful constrained beam search; only the top
        ``n_beams`` items receive real scores, so set ``n_beams`` >= the
        largest metric cutoff K) or 'exact' (teacher-forced log-likelihood of
        every item's semantic ID; exact full ranking, slower per user).

    n_beams: int, optional, default: 20
        Beam width for scoring='beam'.

    scoring_batch_size: int, optional, default: 2048
        Item chunk size for scoring='exact'.

    device: str, optional, default: 'auto'
        'auto' selects 'cuda' if available, otherwise 'cpu'.

    trainable: bool, optional, default: True
        When False, the model will not be re-trained.

    verbose: bool, optional, default: False
        When True, running logs are displayed.

    seed: int, optional, default: None
        Random seed for weight initialization.

    References
    ----------
    Rajput, S. et al. (2023). Recommender Systems with Generative Retrieval.
    NeurIPS. https://arxiv.org/pdf/2305.05065

    Ju, C. et al. (2025). Generative Recommendation with Semantic IDs: A
    Practitioner's Handbook (GRID). https://github.com/snap-research/GRID
    https://arxiv.org/abs/2507.22224

    Paischer, F. et al. (2024). Preference Discerning with LLM-Enhanced
    Generative Retrieval. https://arxiv.org/abs/2412.08604
    """

    def __init__(
        self,
        name="TIGER",
        tokenizer="rqvae",
        feature_standardize=False,
        rqvae_latent_dim=32,
        rqvae_hidden_dims=(512, 256, 128),
        rqvae_num_levels=3,
        rqvae_codebook_size=256,
        rqvae_beta=0.25,
        rqvae_n_epochs=200,
        rqvae_learning_rate=0.001,
        rqvae_batch_size=1024,
        rqvae_weight_decay=0.0,
        d_model=128,
        d_ff=1024,
        num_heads=6,
        d_kv=64,
        num_enc_layers=4,
        num_dec_layers=4,
        dropout=0.1,
        max_len=20,
        n_epochs=20,
        learning_rate=0.001,
        weight_decay=0.0001,
        batch_size=256,
        lr_schedule="constant",
        warmup_steps=10000,
        model_selection="last",
        val_metric="ndcg",
        val_eval_every=5,
        val_k=10,
        val_sample=2000,
        scoring="beam",
        n_beams=20,
        scoring_batch_size=2048,
        device="auto",
        trainable=True,
        verbose=False,
        seed=None,
    ):
        super().__init__(name, trainable=trainable, verbose=verbose)
        if scoring not in SUPPORTED_SCORING:
            raise ValueError(f"scoring='{scoring}' not supported; choose from {SUPPORTED_SCORING}")
        if tokenizer not in SUPPORTED_TOKENIZERS:
            raise ValueError(f"tokenizer='{tokenizer}' not supported; choose from {SUPPORTED_TOKENIZERS}")
        if lr_schedule not in SUPPORTED_LR_SCHEDULES:
            raise ValueError(f"lr_schedule='{lr_schedule}' not supported; choose from {SUPPORTED_LR_SCHEDULES}")
        if model_selection not in ("last", "best"):
            raise ValueError(f"model_selection='{model_selection}' not supported; choose 'last' or 'best'")
        self.tokenizer = tokenizer
        self.feature_standardize = feature_standardize
        self.rqvae_latent_dim = rqvae_latent_dim
        self.rqvae_hidden_dims = rqvae_hidden_dims
        self.rqvae_num_levels = rqvae_num_levels
        self.rqvae_codebook_size = rqvae_codebook_size
        self.rqvae_beta = rqvae_beta
        self.rqvae_n_epochs = rqvae_n_epochs
        self.rqvae_learning_rate = rqvae_learning_rate
        self.rqvae_batch_size = rqvae_batch_size
        self.rqvae_weight_decay = rqvae_weight_decay
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.d_kv = d_kv
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.dropout = dropout
        self.max_len = max_len
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.lr_schedule = lr_schedule
        self.warmup_steps = warmup_steps
        self.model_selection = model_selection
        self.val_metric = val_metric
        self.val_eval_every = val_eval_every
        self.val_k = val_k
        self.val_sample = val_sample
        self.scoring = scoring
        self.n_beams = n_beams
        self.scoring_batch_size = scoring_batch_size
        self.device = device
        self.seed = seed
        self.rng = get_rng(seed)

    def _get_item_features(self):
        item_feature = getattr(self.train_set, "item_feature", None)
        features = getattr(item_feature, "features", None)
        if features is None:
            raise ValueError(
                "TIGER requires precomputed item content embeddings. Provide them "
                "via NextItemEvaluation.from_splits(..., item_feature="
                "FeatureModality(features=..., ids=...))."
            )
        if features.shape[0] < self.total_items:
            raise ValueError(
                f"item_feature has {features.shape[0]} rows but {self.total_items} "
                "items are known; every item (train/val/test) needs a feature vector."
            )
        return np.asarray(features[: self.total_items], dtype="float32")

    def _fit_rqvae(self, torch, feats_t):
        from .tiger import RQVAE

        self.rqvae = RQVAE(
            input_dim=feats_t.size(1),
            hidden_dims=self.rqvae_hidden_dims,
            latent_dim=self.rqvae_latent_dim,
            num_levels=self.rqvae_num_levels,
            codebook_size=self.rqvae_codebook_size,
            beta=self.rqvae_beta,
        ).to(self.device_)
        self.rqvae.kmeans_init_codebooks(feats_t)
        opt = torch.optim.AdamW(
            self.rqvae.parameters(),
            lr=self.rqvae_learning_rate,
            weight_decay=self.rqvae_weight_decay,
        )

        n = feats_t.size(0)
        progress_bar = trange(1, self.rqvae_n_epochs + 1, disable=not self.verbose, desc="RQ-VAE")
        for _ in progress_bar:
            self.rqvae.train()
            used = torch.zeros(
                self.rqvae_num_levels,
                self.rqvae_codebook_size,
                dtype=torch.bool,
                device=self.device_,
            )
            perm = torch.randperm(n, device=self.device_)
            total_loss, cnt = 0.0, 0
            for start in range(0, n, self.rqvae_batch_size):
                batch = feats_t[perm[start : start + self.rqvae_batch_size]]
                ids, _, loss_recon, loss_rq = self.rqvae(batch)
                loss = loss_recon + loss_rq
                opt.zero_grad()
                loss.backward()
                opt.step()
                for level in range(self.rqvae_num_levels):
                    used[level, ids[:, level]] = True
                total_loss += loss.item() * len(batch)
                cnt += len(batch)
            sample = feats_t[perm[: min(n, 8192)]]
            self.rqvae.restart_dead_codes(sample, used)
            progress_bar.set_postfix(loss=(total_loss / cnt))

    def _fit_rkmeans(self, torch, feats_t):
        """Residual k-means tokenizer (GRID handbook): level-by-level k-means on
        the item features, subtracting the assigned centroid before the next
        level. Returns (N, num_levels) int64 codes; centroids are cached (numpy)
        for pickling."""
        from .tiger import _kmeans

        self.rkmeans_centroids = []
        codes = []
        r = feats_t
        for _ in range(self.rqvae_num_levels):
            centroids = _kmeans(r, self.rqvae_codebook_size)
            ids = torch.cdist(r, centroids).argmin(dim=1)
            r = r - centroids[ids]
            self.rkmeans_centroids.append(centroids.cpu().numpy())
            codes.append(ids.cpu().numpy())
        return np.stack(codes, axis=1).astype("int64")

    def _tokenize(self, torch, feats_t):
        if self.tokenizer == "rkmeans":
            return self._fit_rkmeans(torch, feats_t)
        self._fit_rqvae(torch, feats_t)
        self.rqvae.eval()
        return np.concatenate(
            [
                self.rqvae.encode(feats_t[start : start + self.rqvae_batch_size]).cpu().numpy()
                for start in range(0, feats_t.size(0), self.rqvae_batch_size)
            ]
        ).astype("int64")

    def _build_semantic_ids(self, codes):
        # extra level disambiguating items that share the same code tuple
        counters = defaultdict(int)
        dedup = np.zeros(len(codes), dtype="int64")
        for i, row in enumerate(map(tuple, codes)):
            dedup[i] = counters[row]
            counters[row] += 1
        self.sid_table = np.concatenate([codes, dedup[:, None]], axis=1)
        self.level_sizes = [self.rqvae_codebook_size] * self.rqvae_num_levels + [int(dedup.max()) + 1]

        # prefix trie for constrained beam search + sid -> item lookup
        children = [defaultdict(set) for _ in self.level_sizes]
        self.sid_to_item = {}
        for i, row in enumerate(self.sid_table):
            sid = tuple(int(v) for v in row)
            for level in range(len(sid)):
                children[level][sid[:level]].add(sid[level])
            self.sid_to_item[sid] = i
        self.prefix_children = [
            {prefix: np.fromiter(sorted(tokens), dtype="int64") for prefix, tokens in level_children.items()}
            for level_children in children
        ]
        if self.verbose:
            n_collisions = int((dedup > 0).sum())
            print(
                f"Semantic IDs assigned: {len(self.sid_table)} items, "
                f"{n_collisions} collisions, dedup level size {self.level_sizes[-1]}"
            )

    def _fit_seq2seq(self, torch, val_set):
        from .tiger import TIGERSeq2Seq

        self.model = TIGERSeq2Seq(
            level_sizes=self.level_sizes,
            d_model=self.d_model,
            d_ff=self.d_ff,
            num_heads=self.num_heads,
            d_kv=self.d_kv,
            num_enc_layers=self.num_enc_layers,
            num_dec_layers=self.num_dec_layers,
            dropout=self.dropout,
        ).to(self.device_)

        # per-item encoder tokens (offset, 0 = pad); extra all-pad row for pad_idx
        self.pad_idx = self.total_items
        offsets = self.model.offsets.cpu().numpy()
        self.enc_token_table = np.zeros((self.total_items + 1, len(self.level_sizes)), dtype="int64")
        self.enc_token_table[: self.total_items] = self.sid_table + offsets

        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = self._make_lr_scheduler(torch, opt)

        best_state, best_val = None, -float("inf")
        select_best = self.model_selection == "best" and val_set is not None
        val_sessions = self._val_sessions(val_set) if select_best else None
        val_metric = self._make_val_metric() if select_best else None

        progress_bar = trange(1, self.n_epochs + 1, disable=not self.verbose, desc="TIGER")
        for epoch_id in progress_bar:
            self.model.train()
            total_loss, cnt = 0.0, 0
            for inc, (_, hist_iids, out_iids) in enumerate(
                session_seq_iter(
                    self.train_set,
                    pad_index=self.pad_idx,
                    batch_size=self.batch_size,
                    max_len=self.max_len,
                    n_sample=0,
                    rng=self.rng,
                    shuffle=True,
                )
            ):
                enc_tokens = self.enc_token_table[hist_iids].reshape(len(hist_iids), -1)
                enc_tokens_t = torch.tensor(enc_tokens, dtype=torch.long, device=self.device_)
                enc_mask_t = (enc_tokens_t != 0).float()
                target_t = torch.tensor(self.sid_table[out_iids], dtype=torch.long, device=self.device_)
                opt.zero_grad()
                loss = self.model(enc_tokens_t, enc_mask_t, target_t)
                loss.backward()
                opt.step()
                if scheduler is not None:
                    scheduler.step()
                total_loss += loss.item() * len(hist_iids)
                cnt += len(hist_iids)
                if inc % 10 == 0 and cnt > 0:
                    progress_bar.set_postfix(loss=(total_loss / cnt))

            if select_best and epoch_id % self.val_eval_every == 0:
                score = self._validate(val_sessions, val_metric)
                if score > best_val:
                    best_val = score
                    best_state = {n: p.detach().clone() for n, p in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def _make_lr_scheduler(self, torch, opt):
        if self.lr_schedule != "cosine":
            return None
        n_samples = sum(len(m) - 1 for m in self.train_set.sessions.values() if len(m) >= 2)
        steps_per_epoch = max(1, math.ceil(n_samples / self.batch_size))
        total_steps = max(1, steps_per_epoch * self.n_epochs)

        def lr_lambda(step):
            if step < self.warmup_steps:
                return (step + 1) / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
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
        raise ValueError(f"val_metric='{self.val_metric}' not supported; choose from recall/ndcg/auc/mrr")

    def _val_sessions(self, val_set):
        """Collect (user_idx, session_items) for last-item eval, deterministically
        subsampled to ``val_sample`` sessions (fixed across epochs)."""
        sessions = []
        for [_], [mapped_ids], [session_items] in val_set.si_iter(batch_size=1, shuffle=False):
            if len(session_items) < 2:
                continue
            user_idx = int(val_set.uir_tuple[0][mapped_ids[0]])
            sessions.append((user_idx, [int(i) for i in session_items]))
        if self.val_sample is not None and len(sessions) > self.val_sample:
            idx = self.rng.choice(len(sessions), size=self.val_sample, replace=False)
            sessions = [sessions[i] for i in sorted(idx)]
        return sessions

    def _validate(self, val_sessions, metric):
        """Mean ``metric`` over ``val_sessions`` (mode 'last'), imitating
        ranking_eval. Beam scoring only fills the top ``n_beams`` items, so
        ``n_beams`` is temporarily raised to at least ``val_k``."""
        num_items = self.train_set.num_items
        orig_beams, self.n_beams = self.n_beams, max(self.n_beams, self.val_k)
        item_indices = np.arange(num_items)
        results = []
        for user_idx, session_items in val_sessions:
            target = session_items[-1]
            if target >= num_items:
                continue
            gt_pos = np.array([target])
            gt_neg = np.delete(item_indices, target)
            item_rank, item_scores = self.rank(user_idx, item_indices, history_items=session_items[:-1])
            results.append(
                metric.compute(
                    gt_pos=gt_pos,
                    gt_neg=gt_neg,
                    pd_rank=item_rank,
                    pd_scores=item_scores,
                    item_indices=item_indices,
                )
            )
        self.n_beams = orig_beams
        return float(np.mean(results)) if results else 0.0

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)
        if not self.trainable:
            return self

        import torch

        torch.manual_seed(self.seed if self.seed is not None else 0)
        self.device_ = ("cuda" if torch.cuda.is_available() else "cpu") if self.device == "auto" else self.device

        feats = self._get_item_features()
        if self.feature_standardize:
            mean = feats.mean(axis=0)
            std = feats.std(axis=0)
            std[std == 0] = 1.0
            feats = ((feats - mean) / std).astype("float32")
        feats_t = torch.tensor(feats, device=self.device_)

        codes = self._tokenize(torch, feats_t)
        self._build_semantic_ids(codes)
        self._fit_seq2seq(torch, val_set)

        # keep pickles portable across GPU/CPU boxes; moved back in score()
        self.model.to("cpu").eval()
        if self.tokenizer == "rqvae":
            self.rqvae.to("cpu").eval()
        return self

    def _ensure_device(self, torch):
        if self.device_ == "cuda" and not torch.cuda.is_available():
            self.device_ = "cpu"
        if next(self.model.parameters()).device.type != torch.device(self.device_).type:
            self.model.to(self.device_)

    def score(self, user_idx, history_items, **kwargs):
        import torch

        if len(history_items) == 0:
            return np.ones(self.total_items, dtype="float")
        self._ensure_device(torch)
        hist = list(history_items)[-self.max_len :]
        hist = [self.pad_idx] * (self.max_len - len(hist)) + hist
        enc_tokens_t = torch.tensor(
            self.enc_token_table[hist].reshape(1, -1),
            dtype=torch.long,
            device=self.device_,
        )
        enc_mask_t = (enc_tokens_t != 0).float()
        self.model.eval()
        with torch.no_grad():
            if self.scoring == "beam":
                beams, logps = self.model.generate_beam(enc_tokens_t, enc_mask_t, self.n_beams, self.prefix_children)
                scores = np.full(self.total_items, -1e10, dtype="float")
                for sid, lp in zip(beams, logps):
                    scores[self.sid_to_item[sid]] = lp
            else:  # "exact"
                sid_table_t = torch.tensor(self.sid_table, dtype=torch.long, device=self.device_)
                scores = self.model.score_all_items(
                    enc_tokens_t, enc_mask_t, sid_table_t, self.scoring_batch_size
                ).astype("float")
        return scores
