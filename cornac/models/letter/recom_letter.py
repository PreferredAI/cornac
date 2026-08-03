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
import random
from collections import defaultdict

import numpy as np
from tqdm.auto import trange

from ..tiger.recom_tiger import TIGER


def _generator_optimizer_groups(model, weight_decay):
    """Match Hugging Face Trainer's AdamW decay groups for the T5 generator."""
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = no_decay if "bias" in name or "layer_norm" in name else decay
        target.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class LETTER(TIGER):
    """LETTER: LEarnable Tokenizer for generaTivE Recommendation.

    LETTER trains a four-level RQ-VAE tokenizer with collaborative and
    code-assignment-diversity regularization, then trains the released
    LETTER-TIGER generator over the resulting semantic IDs. The official
    collaborative features are 32-dimensional SASRec item embeddings. Their
    raw item IDs are used to align the rows with Cornac's global item indices.

    This class accepts TIGER's public parameters plus the LETTER-specific
    arguments below. :data:`~cornac.models.letter.LETTER_BEAUTY_CONFIG`
    contains the authors' released Beauty recipe.

    Parameters
    ----------
    cf_embeddings: array-like or None
        Collaborative item embeddings of shape ``(n_items, rqvae_latent_dim)``.
        Required when ``cf_weight`` is non-zero.
    cf_embedding_ids: array-like or None
        Raw item IDs corresponding to the rows of ``cf_embeddings``. Required
        whenever ``cf_embeddings`` is provided. Rows are aligned to Cornac's
        global item indices during fitting.
    cf_weight: float, default: 0.02
        Collaborative loss weight (alpha).
    diversity_weight: float, default: 0.001
        Diversity loss weight (beta).
    n_clusters: int, default: 10
        Constrained codebook groups used by the diversity loss.
    rqvae_quant_loss_weight: float, default: 1.0
        Weight of the complete residual-quantization loss.
    rqvae_sk_epsilon: float, default: 0.003
        Sinkhorn epsilon on the final residual codebook. Earlier levels use
        nearest-code assignment, matching the release.
    rqvae_sk_iters: int, default: 50
        Number of Sinkhorn normalization iterations.
    rqvae_kmeans_jobs: int, default: 10
        Worker count used by the released constrained-k-means calls.
    collision_resolve_iters: int, default: 20
        Maximum official post-tokenization collision-reassignment passes.
    ranking_temperature: float, default: 1.0
        Temperature in the released ranking loss. The published value 1.0 is
        ordinary token cross-entropy.
    gradient_accumulation_steps: int, default: 1
        Number of generator minibatches per optimizer update.
    warmup_ratio: float, default: 0.01
        Fraction of generator optimizer updates used for linear warmup.
    early_stopping_patience: int or None, default: 20
        Non-improving epoch validations before stopping. Validation uses the
        released token loss and restores the lowest-loss checkpoint.
    val_batch_size: int, default: 256
        Batch size for validation loss.
    max_grad_norm: float, default: 1.0
        Generator gradient clipping threshold used by Hugging Face Trainer.
    letter_base_vocab_size: int, default: 32100
        Size of the base T5 SentencePiece vocabulary before semantic tokens.
    precomputed_semantic_ids: array-like or None
        Optional integer semantic-ID table of shape ``(n_items, num_levels)``.
        When provided, skip tokenizer training and train only the released
        LETTER-TIGER generator.
    """

    def __init__(
        self,
        name="LETTER",
        cf_embeddings=None,
        cf_embedding_ids=None,
        cf_weight=0.02,
        diversity_weight=0.001,
        n_clusters=10,
        rqvae_quant_loss_weight=1.0,
        rqvae_sk_epsilon=0.003,
        rqvae_sk_iters=50,
        rqvae_kmeans_jobs=10,
        collision_resolve_iters=20,
        ranking_temperature=1.0,
        gradient_accumulation_steps=1,
        warmup_ratio=0.01,
        early_stopping_patience=20,
        val_batch_size=256,
        max_grad_norm=1.0,
        letter_base_vocab_size=32100,
        precomputed_semantic_ids=None,
        **kwargs,
    ):
        kwargs["tokenizer"] = "rqvae"
        super().__init__(name=name, **kwargs)
        if ranking_temperature <= 0:
            raise ValueError("ranking_temperature must be positive")
        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if not 0 <= warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be between 0 and 1")
        if early_stopping_patience is not None and early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive or None")
        if val_batch_size <= 0:
            raise ValueError("val_batch_size must be positive")
        if rqvae_kmeans_jobs == 0:
            raise ValueError("rqvae_kmeans_jobs must be non-zero")
        self.cf_embeddings = (
            None
            if cf_embeddings is None
            else np.asarray(cf_embeddings, dtype="float32")
        )
        self.cf_embedding_ids = (
            None if cf_embedding_ids is None else list(cf_embedding_ids)
        )
        if (self.cf_embeddings is None) != (self.cf_embedding_ids is None):
            raise ValueError(
                "cf_embeddings and cf_embedding_ids must be provided together"
            )
        if self.cf_embeddings is not None:
            if self.cf_embeddings.ndim != 2:
                raise ValueError("cf_embeddings must be a 2-dimensional array")
            if len(self.cf_embedding_ids) != self.cf_embeddings.shape[0]:
                raise ValueError(
                    f"cf_embedding_ids has {len(self.cf_embedding_ids)} entries "
                    f"but cf_embeddings has {self.cf_embeddings.shape[0]} rows"
                )
            if len(set(self.cf_embedding_ids)) != len(self.cf_embedding_ids):
                raise ValueError("cf_embedding_ids must not contain duplicates")
        self.cf_weight = cf_weight
        self.diversity_weight = diversity_weight
        self.n_clusters = n_clusters
        self.rqvae_quant_loss_weight = rqvae_quant_loss_weight
        self.rqvae_sk_epsilon = rqvae_sk_epsilon
        self.rqvae_sk_iters = rqvae_sk_iters
        self.rqvae_kmeans_jobs = rqvae_kmeans_jobs
        self.collision_resolve_iters = collision_resolve_iters
        self.ranking_temperature = ranking_temperature
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_ratio = warmup_ratio
        self.early_stopping_patience = early_stopping_patience
        self.val_batch_size = val_batch_size
        self.max_grad_norm = max_grad_norm
        self.letter_base_vocab_size = letter_base_vocab_size
        self.precomputed_semantic_ids = (
            None
            if precomputed_semantic_ids is None
            else np.asarray(precomputed_semantic_ids, dtype="int64")
        )

    def _get_cf_embeddings(self):
        if self.cf_weight and self.cf_embeddings is None:
            raise ValueError("LETTER requires cf_embeddings when cf_weight is non-zero")
        if self.cf_embeddings is None:
            return None
        if self.cf_embeddings.shape[1] != self.rqvae_latent_dim:
            raise ValueError(
                "official LETTER uses same-dimensional collaborative and "
                f"tokenizer representations; expected {self.rqvae_latent_dim}, "
                f"got {self.cf_embeddings.shape[1]}"
            )

        row_by_id = {raw_id: row for row, raw_id in enumerate(self.cf_embedding_ids)}
        missing = [raw_id for raw_id in self.iid_map if raw_id not in row_by_id]
        if missing:
            raise ValueError(
                f"cf_embedding_ids is missing {len(missing)} item(s) known to Cornac"
            )

        aligned = np.empty((self.total_items, self.rqvae_latent_dim), dtype="float32")
        for raw_id, item_idx in self.iid_map.items():
            aligned[item_idx] = self.cf_embeddings[row_by_id[raw_id]]
        return aligned

    def _fit_rqvae(self, torch, feats_t):
        from .letter import LETTERRQVAE

        cf_embeddings = self._get_cf_embeddings()
        cf_t = (
            None
            if cf_embeddings is None
            else torch.as_tensor(cf_embeddings, device=self.device_)
        )

        seed = self.seed if self.seed is not None else 0
        random.seed(seed)
        np.random.seed(seed)
        self.rqvae = LETTERRQVAE(
            input_dim=feats_t.size(1),
            hidden_dims=self.rqvae_hidden_dims,
            latent_dim=self.rqvae_latent_dim,
            num_levels=self.rqvae_num_levels,
            codebook_size=self.rqvae_codebook_size,
            commitment_weight=self.rqvae_beta,
            n_clusters=self.n_clusters,
            sk_epsilons=[0.0] * (self.rqvae_num_levels - 1) + [self.rqvae_sk_epsilon],
            sk_iters=self.rqvae_sk_iters,
            kmeans_n_jobs=self.rqvae_kmeans_jobs,
        ).to(self.device_)
        init_loader = torch.utils.data.DataLoader(
            range(feats_t.size(0)),
            batch_size=feats_t.size(0),
            shuffle=True,
        )
        init_ids = next(iter(init_loader)).to(self.device_)
        self.rqvae.initialize_codebooks(feats_t[init_ids])
        optimizer = torch.optim.AdamW(
            self.rqvae.parameters(),
            lr=self.rqvae_learning_rate,
            weight_decay=self.rqvae_weight_decay,
        )

        train_loader = torch.utils.data.DataLoader(
            range(feats_t.size(0)),
            batch_size=self.rqvae_batch_size,
            shuffle=True,
        )
        progress = trange(
            1,
            self.rqvae_n_epochs + 1,
            disable=not self.verbose,
            desc="LETTER RQ-VAE",
        )
        for _ in progress:
            self.rqvae.train()
            if self.diversity_weight:
                self.rqvae.update_diversity_clusters()
            total_loss = 0.0
            count = 0
            for item_ids in train_loader:
                item_ids = item_ids.to(self.device_)
                batch = feats_t[item_ids]
                cf_batch = None if cf_t is None else cf_t[item_ids]
                _, _, recon, quant, collaborative, diversity = self.rqvae(
                    batch, cf_batch
                )
                loss = (
                    recon
                    + self.rqvae_quant_loss_weight
                    * (quant + self.diversity_weight * diversity)
                    + self.cf_weight * collaborative
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch)
                count += len(batch)
            progress.set_postfix(loss=total_loss / count)

    def _tokenize(self, torch, feats_t):
        if self.precomputed_semantic_ids is not None:
            codes = self.precomputed_semantic_ids
            expected = (self.total_items, self.rqvae_num_levels)
            if codes.shape != expected:
                raise ValueError(
                    f"precomputed_semantic_ids has shape {codes.shape}; expected {expected}"
                )
            if codes.size and (
                codes.min() < 0 or codes.max() >= self.rqvae_codebook_size
            ):
                raise ValueError(
                    f"precomputed_semantic_ids values must be in [0, {self.rqvae_codebook_size})"
                )
            unique = np.unique(codes, axis=0)
            self.sid_collisions_before = len(codes) - len(unique)
            self.sid_collisions_after = self.sid_collisions_before
            self.sid_code_utilization = [
                int(np.unique(codes[:, level]).size)
                for level in range(self.rqvae_num_levels)
            ]
            return codes.copy()

        self._fit_rqvae(torch, feats_t)
        self.rqvae.eval()
        codes = torch.cat(
            [
                self.rqvae.encode(
                    feats_t[start : start + self.rqvae_batch_size],
                    use_sinkhorn=False,
                )
                for start in range(0, feats_t.size(0), self.rqvae_batch_size)
            ]
        )
        self.sid_collisions_before = len(codes) - len(torch.unique(codes, dim=0))
        resolved = self.rqvae.resolve_collisions(
            feats_t, codes, max_iters=self.collision_resolve_iters
        )
        self.sid_collisions_after = len(resolved) - len(torch.unique(resolved, dim=0))
        self.sid_code_utilization = [
            int(resolved[:, level].unique().numel())
            for level in range(self.rqvae_num_levels)
        ]
        return resolved.cpu().numpy().astype("int64")

    def _build_semantic_ids(self, codes):
        """Build the official fixed-length IDs without TIGER's dedup level."""
        self.sid_table = np.asarray(codes, dtype="int64")
        self.level_sizes = [self.rqvae_codebook_size] * self.rqvae_num_levels
        children = [defaultdict(set) for _ in self.level_sizes]
        sid_to_items = defaultdict(list)
        for item, row in enumerate(self.sid_table):
            sid = tuple(int(value) for value in row)
            for level in range(len(sid)):
                children[level][sid[:level]].add(sid[level])
            sid_to_items[sid].append(item)
        # NumPy advanced indexing lets TIGER's beam scorer assign the same
        # token score to every item left in a collision, instead of dropping it.
        self.sid_to_item = {
            sid: np.asarray(items, dtype="int64") for sid, items in sid_to_items.items()
        }
        self.prefix_children = [
            {
                prefix: np.fromiter(sorted(tokens), dtype="int64")
                for prefix, tokens in level.items()
            }
            for level in children
        ]
        if self.verbose:
            collisions = sum(len(items) - 1 for items in sid_to_items.values())
            print(
                f"LETTER semantic IDs: {len(codes)} items, {collisions} unresolved collisions"
            )

    def _training_rows(self):
        rows = []
        uir_tuple = self.train_set.uir_tuple
        for mapped_ids in self.train_set.sessions.values():
            items = [int(item) for item in uir_tuple[1][mapped_ids]]
            for position in range(1, len(items)):
                rows.append((items[:position][-self.max_len :], items[position]))
        return rows

    def _encoder_batch(self, torch, histories):
        sequences = []
        for history in histories:
            item_ids = np.asarray(history[-self.max_len :], dtype="int64")
            tokens = self.enc_token_table[item_ids].reshape(-1).tolist()
            sequences.append(tokens + [self.model.eos_token_id])
        max_tokens = max(len(sequence) for sequence in sequences)
        batch = np.full(
            (len(sequences), max_tokens), self.model.pad_token_id, dtype="int64"
        )
        for row, sequence in enumerate(sequences):
            batch[row, : len(sequence)] = sequence
        tokens = torch.as_tensor(batch, dtype=torch.long, device=self.device_)
        return tokens, (tokens != self.model.pad_token_id).float()

    def _validation_loss(self, torch, val_sessions):
        was_training = self.model.training
        self.model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, len(val_sessions), self.val_batch_size):
                batch = val_sessions[start : start + self.val_batch_size]
                histories = [items[:-1] for _, items in batch]
                targets = [items[-1] for _, items in batch]
                enc_tokens, enc_mask = self._encoder_batch(torch, histories)
                target_sids = torch.as_tensor(
                    self.sid_table[targets], dtype=torch.long, device=self.device_
                )
                loss = self.model(enc_tokens, enc_mask, target_sids)
                total += loss.item() * len(batch)
                count += len(batch)
        if was_training:
            self.model.train()
        return total / count if count else float("inf")

    def _fit_seq2seq(self, torch, val_set):
        from .letter import LETTERSeq2Seq

        # Tokenizer and generator are separate programs in the release; both
        # restart from the configured seed.
        seed = self.seed if self.seed is not None else 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.rng = np.random.RandomState(seed)

        self.model = LETTERSeq2Seq(
            level_sizes=self.level_sizes,
            d_model=self.d_model,
            d_ff=self.d_ff,
            num_heads=self.num_heads,
            d_kv=self.d_kv,
            num_enc_layers=self.num_enc_layers,
            num_dec_layers=self.num_dec_layers,
            dropout=self.dropout,
            base_vocab_size=self.letter_base_vocab_size,
            temperature=self.ranking_temperature,
            code_values=[
                np.unique(self.sid_table[:, level])
                for level in range(self.rqvae_num_levels)
            ],
        ).to(self.device_)
        self.pad_idx = self.total_items
        code_table = torch.as_tensor(
            self.sid_table, dtype=torch.long, device=self.device_
        )
        self.enc_token_table = self.model.semantic_tokens(code_table).cpu().numpy()

        optimizer = torch.optim.AdamW(
            _generator_optimizer_groups(self.model, self.weight_decay),
            lr=self.learning_rate,
        )
        rows = self._training_rows()
        if not rows:
            raise ValueError("LETTER needs at least one next-item training prefix")
        train_loader = torch.utils.data.DataLoader(
            range(len(rows)), batch_size=self.batch_size, shuffle=True
        )
        batches_per_epoch = max(1, len(train_loader))
        updates_per_epoch = max(
            1, math.ceil(batches_per_epoch / self.gradient_accumulation_steps)
        )
        total_updates = max(1, updates_per_epoch * self.n_epochs)
        warmup_updates = math.ceil(total_updates * self.warmup_ratio)

        def lr_lambda(step):
            if step < warmup_updates:
                return step / max(1, warmup_updates)
            progress = (step - warmup_updates) / max(1, total_updates - warmup_updates)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scheduler = (
            torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            if self.lr_schedule == "cosine"
            else None
        )
        select_best = self.model_selection == "best" and val_set is not None
        val_sessions = self._val_sessions(val_set) if select_best else []
        best_state = None
        best_loss = float("inf")
        non_improving = 0

        progress = trange(
            1, self.n_epochs + 1, disable=not self.verbose, desc="LETTER-TIGER"
        )
        for epoch in progress:
            self.current_epoch = epoch
            self.model.train()
            optimizer.zero_grad()
            total_loss = 0.0
            count = 0
            for batch_index, indices in enumerate(train_loader, start=1):
                batch = [rows[index] for index in indices.tolist()]
                histories = [history for history, _ in batch]
                targets = [target for _, target in batch]
                enc_tokens, enc_mask = self._encoder_batch(torch, histories)
                target_sids = torch.as_tensor(
                    self.sid_table[targets], dtype=torch.long, device=self.device_
                )
                loss = self.model(enc_tokens, enc_mask, target_sids)
                (loss / self.gradient_accumulation_steps).backward()
                if (
                    batch_index % self.gradient_accumulation_steps == 0
                    or batch_index == batches_per_epoch
                ):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                total_loss += loss.item() * len(batch)
                count += len(batch)
            progress.set_postfix(loss=total_loss / count)

            if select_best and epoch % self.val_eval_every == 0:
                val_loss = self._validation_loss(torch, val_sessions)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in self.model.state_dict().items()
                    }
                    non_improving = 0
                else:
                    non_improving += 1
                if (
                    self.early_stopping_patience is not None
                    and non_improving >= self.early_stopping_patience
                ):
                    break

        self.best_val_loss = None if best_state is None else best_loss
        if best_state is not None:
            self.model.load_state_dict(best_state)

    def score(self, user_idx, history_items, **kwargs):
        import torch

        if len(history_items) == 0:
            return np.ones(self.total_items, dtype="float")
        self._ensure_device(torch)
        enc_tokens, enc_mask = self._encoder_batch(
            torch, [list(history_items)[-self.max_len :]]
        )
        self.model.eval()
        with torch.no_grad():
            if self.scoring == "beam":
                beams, log_probs = self.model.generate_beam(
                    enc_tokens, enc_mask, self.n_beams, self.prefix_children
                )
                scores = np.full(self.total_items, -1e10, dtype="float")
                for sid, log_prob in zip(beams, log_probs):
                    scores[self.sid_to_item[sid]] = log_prob
            else:
                sid_table = torch.as_tensor(
                    self.sid_table, dtype=torch.long, device=self.device_
                )
                scores = self.model.score_all_items(
                    enc_tokens, enc_mask, sid_table, self.scoring_batch_size
                ).astype("float")
        return scores
