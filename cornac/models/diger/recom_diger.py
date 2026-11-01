# Copyright 2026 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""DIGER next-item recommendation with T5 and residual quantization."""

from pathlib import Path

import numpy as np

from ..recommender import NextItemRecommender, SemanticIDMixin


class DIGER(NextItemRecommender, SemanticIDMixin):
    """DIGER with frequency-gated residual quantization and T5 generation.

    Supply precomputed item embeddings through ``FeatureModality`` in
    ``NextItemEvaluation``. Features are used without additional normalization.
    Training expands each session into all next-item prefixes; validation uses
    only each session's final target. Sessions shorter than two items are skipped.

    Parameters
    ----------
    name: str, default: "DIGER"
        Model name.
    initializer: dict or path, optional
        Bundle produced by ``training.pretrain_tokenizer`` on the same ordered
        catalog/features. If absent, train a matched-feature RQ autoencoder first.
    tokenizer_options: dict, optional
        Options for ``diger.SemanticTokenizer``, including hidden_dims, latent_dim,
        levels and codebook_size. The encoder is frozen after initialization;
        recommendation gradients update the residual codebooks.
    generator_options: dict, optional
        Options for ``diger.PositionalGenerator``, including model_dim, ff_dim,
        num_layers and num_heads. Levels and vocabulary must match the tokenizer.
    initializer_options: dict, optional
        ``pretrain_tokenizer`` options (default: 100 epochs, 20 warmup epochs,
        batch size 256, learning rate 0.001).
    initializer_seed: int, default: 2024
        Random seed for pretraining a fresh initializer.
    n_epochs: int, default: 120
        Joint training horizon; validation NDCG@10 selects paired model/catalog.
    batch_size: int, default: 256
        Number of training prefixes per batch.
    eval_batch_size: int, default: 32
        Number of validation sessions per batch.
    learning_rate: float, default: 0.0005
        Generator AdamW learning rate.
    tokenizer_learning_rate: float, default: 0.00003
        Codebook AdamW learning rate. Both rates were selected on Beauty validation.
    weight_decay: float, default: 0.05
        Weight decay for both joint-training AdamW optimizers.
    warmup_steps: int, default: 500
        Linear warmup steps before cosine learning-rate decay.
    max_len: int, default: 210
        Maximum number of history items, used consistently during fit/inference.
    n_beams: int, default: 20
        Unconstrained positional beam width. Use at least the evaluation cutoff.
    early_stopping_patience: int, default: 15
        Stop after this many epochs without strict validation improvement.
    device: str, default: 'cpu'
        PyTorch device, or 'auto' to select CUDA when available.
    trainable: bool, default: True
        Whether ``fit`` trains the model.
    verbose: bool, default: False
        Whether to print epoch loss and validation scores.
    seed: int, default: 2024
        Random seed for generator initialization and joint training.
    """

    def __init__(self, name="DIGER", initializer=None, tokenizer_options=None,
                 generator_options=None, initializer_options=None,
                 initializer_seed=2024, n_epochs=120, batch_size=256,
                 eval_batch_size=32, learning_rate=5e-4,
                 tokenizer_learning_rate=3e-5, weight_decay=0.05,
                 warmup_steps=500, max_len=210, n_beams=20,
                 early_stopping_patience=15,
                 device="cpu", trainable=True, verbose=False, seed=2024):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.initializer = initializer
        self.tokenizer_options = dict(tokenizer_options or {})
        self.generator_options = dict(generator_options or {})
        if min(n_epochs, batch_size, eval_batch_size, n_beams) < 1 or max_len < 2:
            raise ValueError("Epochs, batch sizes, max_len and n_beams must be positive")
        if min(learning_rate, tokenizer_learning_rate) <= 0 or weight_decay < 0:
            raise ValueError("Learning rates must be positive and weight_decay nonnegative")
        if warmup_steps < 0 or early_stopping_patience < 1:
            raise ValueError("warmup_steps must be nonnegative and patience positive")
        self.initializer_options = dict(initializer_options or {})
        self.initializer_seed = initializer_seed
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.tokenizer_learning_rate = tokenizer_learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_len = max_len
        self.n_beams = n_beams
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed
        self.device = device

    def _rows(self, dataset):
        rows = []
        excluded = getattr(dataset, "catalog_only_session_indices", set())
        for session, indices in dataset.sessions.items():
            if session in excluded:
                continue
            items = [int(i) + 1 for i in dataset.uir_tuple[1][indices]]
            stops = range(1, len(items)) if dataset is self.train_set else [len(items)-1]
            for stop in stops:
                if stop > 0:
                    rows.append((items[max(0, stop-self.max_len):stop], items[stop]))
        if not rows:
            raise ValueError("DIGER requires sessions containing history and target")
        return rows

    def _feature_rows(self, train_set):
        modality = getattr(train_set, "item_feature", None)
        features = getattr(modality, "features", None)
        if features is None:
            raise ValueError("DIGER requires item_feature=FeatureModality(features=..., ids=...)")
        features = np.asarray(features, dtype="float32")
        ordered_ids = [raw for raw, _ in sorted(train_set.iid_map.items(), key=lambda pair: pair[1])]
        if (features.ndim != 2 or features.shape[0] < len(ordered_ids)
                or features.shape[1] == 0 or not np.isfinite(features).all()):
            raise ValueError("Require finite 2D features covering every Cornac item")
        ids = getattr(modality, "ids", None)
        if ids is not None and list(ids[:len(ordered_ids)]) != ordered_ids:
            raise ValueError("FeatureModality IDs must cover and align with every Cornac item")
        return features[:len(ordered_ids)].copy(), ordered_ids

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)
        if not self.trainable:
            return self
        if val_set is None:
            raise ValueError("DIGER requires validation sessions for model selection")
        features, _ = self._feature_rows(train_set)
        train_rows, val_rows = self._rows(train_set), self._rows(val_set)
        import torch
        from .diger import DIGERBackend
        from .training import pretrain_tokenizer, tensor_hash, train

        self._ensure_device(torch)
        padded = np.vstack([np.zeros((1, features.shape[1]), dtype="float32"), features])
        features = torch.tensor(padded, device=self.device)
        initializer = self.initializer
        if initializer is None:
            initializer = pretrain_tokenizer(
                features, self.tokenizer_options,
                **{"seed": self.initializer_seed, **self.initializer_options})
        elif isinstance(initializer, (str, Path)):
            initializer = torch.load(initializer, map_location="cpu", weights_only=False)
        if initializer.get("format") != "cornac-diger-initializer-v1":
            raise ValueError("Require a matched pretrained tokenizer initializer")
        expected = {k: v for k, v in initializer["tokenizer_options"].items()
                    if k != "freeze_encoder"}
        actual = {k: v for k, v in self.tokenizer_options.items() if k != "freeze_encoder"}
        if expected != actual:
            raise ValueError("Initializer tokenizer recipe differs from tokenizer options")
        if initializer["feature_hash"] != tensor_hash(features):
            raise ValueError("Initializer feature hash differs from ordered catalog")
        torch.manual_seed(self.seed)
        self.backend_ = DIGERBackend(features, self.tokenizer_options,
                                     self.generator_options).to(self.device)
        if self.n_beams > self.backend_.generator.positions[0].num_embeddings:
            raise ValueError("n_beams must not exceed codebook_size")
        self.backend_.initialize_tokenizer(initializer["state"], initializer["semantic_width"])
        config = {"epochs": self.n_epochs, "batch_size": self.batch_size,
                  "eval_batch_size": self.eval_batch_size, "lr_rec": self.learning_rate,
                  "lr_id": self.tokenizer_learning_rate, "weight_decay": self.weight_decay,
                  "warmup_steps": self.warmup_steps, "max_length": self.max_len,
                  "beams": self.n_beams, "patience": self.early_stopping_patience,
                  "seed": self.seed}
        selected, self.training_summary_ = train(
            self.backend_, train_rows, val_rows, config=config, verbose=self.verbose)
        self.best_epoch, self.best_value = selected["epoch"], selected["ndcg@10"]
        self.codes_ = selected["catalog"].numpy().copy()
        self.sid_to_item_ = {tuple(code): item for item, code in enumerate(self.codes_[1:])}
        self.backend_.to("cpu")
        return self

    def _ensure_device(self, torch):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif torch.device(self.device).type == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

    def save(self, save_dir=None, save_trainset=False, metadata=None):
        if save_dir is None or not hasattr(self, "backend_"):
            return super().save(save_dir, save_trainset, metadata)
        original_device = self.backend_.features.device
        self.backend_.to("cpu")
        try:
            return super().save(save_dir, save_trainset, metadata)
        finally:
            self.backend_.to(original_device)

    def semantic_id_catalog(self):
        """Return complete collision-suffixed IDs in Cornac item order."""
        if not hasattr(self, "codes_"):
            raise ValueError("DIGER must be fitted before exporting its SID catalog")
        return self.codes_[1:]

    def _beam_predictions(self, history_items):
        """Decode raw addresses once, retaining invalid and duplicate rows."""
        import torch

        catalog = self.semantic_id_catalog()
        history = list(history_items)[-self.max_len:]
        if not history:
            return np.empty((0, catalog.shape[1]), dtype="int64"), np.empty(0)
        if any(i < 0 or i >= self.total_items for i in history):
            raise ValueError("History item outside catalog")
        self._ensure_device(torch)
        self.backend_.to(self.device).eval()
        codes = torch.tensor(self.codes_[np.asarray(history, dtype="int64") + 1],
                             device=self.device)[None]
        paths, values = self.backend_.generator.beam_search(
            codes, torch.ones(codes.shape[:2], dtype=torch.bool, device=self.device),
            beams=self.n_beams)
        return paths[0].cpu().numpy(), values[0].cpu().numpy()

    def _score_with_semantic_ids(self, user_idx, history_items, **kwargs):
        addresses, values = self._beam_predictions(history_items)
        scores = np.full(self.total_items, -np.inf, dtype="float64")
        for address, value in zip(addresses, values):
            item = self.sid_to_item_.get(tuple(address))
            if item is not None:
                scores[item] = value
        return scores, addresses

    def score(self, user_idx, history_items, item_idx=None, **kwargs):
        """Return catalog scores from unconstrained full-address beam search."""
        scores, _ = self._score_with_semantic_ids(user_idx, history_items, **kwargs)
        return scores if item_idx is None else scores[item_idx]
