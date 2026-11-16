# Copyright 2026 The Cornac Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Catalog-bundle wrapper around the unchanged RPG implementation."""

import copy
import hashlib
import warnings

import numpy as np

from ..recommender import BundleRecommender
from ..rpg import RPG
from ...exception import ScoreException
from ...utils import get_rng
from .adapter import bundle_to_sequential
from .features import collaborative_item_features, mean_bundle_features


class BundleRPG(BundleRecommender):
    """CF-only CLHE-style bundle embeddings -> OPQ -> RPG.

    Parameters
    ----------
    encoder: {'clhe', 'mean', 'random'}, default: 'clhe'
        CLHE-style CF contrastive pooling, mean CF member pooling, or a random
        bundle-feature control. This model is not a multimodal CLHE replica.
    emb_size: int, default: 64
        CF item and bundle feature width. Must fit the chosen OPQ layout.
    item_features: ndarray, optional
        Precomputed training-safe [constituent_item, emb_size] features aligned
        exactly to train_set.constituent_iid_map. Caller owns their provenance.
        If absent, LightGCN/BPR features are trained on the supplied train set.
    cf_params, encoder_params: dict, optional
        Keyword settings for collaborative_item_features and encode_bundles.
        seed/device and feature width are owned by this wrapper.
    rpg_params: dict, optional
        Existing RPG settings. The wrapper owns scoring='exact', seed, device,
        and model_selection='last'. Defaults use 8 OPQ digits and no PCA.
    epoch_budgets: sequence of int, default: (20,)
        Fresh RPG fits at each budget. With validation, select by masked full
        bundle NDCG@eval_k. Without validation, train only the largest budget.
        Features are trained once and reused across budgets.
    n_perm: int, default: 1
        Seeded training permutations per user's bundle history. Inference uses
        the fixed first order. These are synthetic orders, not timestamps.
    eval_k: int, default: 20
        Validation NDCG cutoff for external budget selection.
    device: str, default: 'cpu'
        Torch device, or 'auto' to choose CUDA when available.

    Notes
    -----
    Bundle membership is transductive catalog information; validation/test
    user-bundle labels never enter CF fitting, encoder training or histories.
    RPG's internal content-only PCA/standardization, if enabled explicitly, use
    the full feature catalog. OPQ's fit mask is supplied by unchanged RPG.
    """

    def __init__(self, name="BundleRPG", encoder="clhe", emb_size=64,
                 item_features=None, cf_params=None, encoder_params=None,
                 rpg_params=None, epoch_budgets=(20,), n_perm=1, eval_k=20,
                 device="cpu", trainable=True, verbose=False, seed=42):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        if encoder not in ("clhe", "mean", "random"):
            raise ValueError("encoder must be clhe, mean or random")
        if (not isinstance(emb_size, (int, np.integer)) or emb_size < 1
                or not isinstance(n_perm, (int, np.integer)) or n_perm < 1
                or not isinstance(eval_k, (int, np.integer)) or eval_k < 1):
            raise ValueError("emb_size, n_perm and eval_k must be positive integers")
        if (not epoch_budgets or any(not isinstance(e, (int, np.integer)) or e < 1
                                     for e in epoch_budgets)):
            raise ValueError("epoch_budgets must contain positive integers")
        self.encoder = encoder
        self.emb_size = emb_size
        self.item_features = item_features
        self.cf_params = {} if cf_params is None else copy.deepcopy(cf_params)
        self.encoder_params = {} if encoder_params is None else copy.deepcopy(encoder_params)
        self.rpg_params = {} if rpg_params is None else copy.deepcopy(rpg_params)
        self.epoch_budgets = tuple(epoch_budgets)
        self.n_perm = n_perm
        self.eval_k = eval_k
        self.device = device
        self.seed = seed
        for label, params, reserved in (
            ("cf_params", self.cf_params, {"train_set", "emb_size", "seed", "device"}),
            ("encoder_params", self.encoder_params, {"train_set", "item_features", "seed", "device"}),
            ("rpg_params", self.rpg_params, {"name", "seed", "device", "scoring", "model_selection",
                                            "n_epochs", "trainable", "verbose"}),
        ):
            if reserved.intersection(params):
                raise ValueError("{} cannot override {}".format(label, sorted(reserved.intersection(params))))

    def _bundle_features(self, train_set, device):
        self.encoder_diagnostics_ = {}
        if self.encoder == "random":
            return get_rng(self.seed).normal(size=(self.total_bundles, self.emb_size)).astype(np.float32)
        if self.item_features is None:
            features = collaborative_item_features(
                train_set, emb_size=self.emb_size, seed=self.seed, device=device,
                **self.cf_params,
            )
        else:
            features = np.asarray(self.item_features, dtype=np.float32)
        # Validate both axes and finite values before either pooling method.
        pooled = mean_bundle_features(train_set, features)
        if features.shape[1] != self.emb_size:
            raise ValueError("item_features width must equal emb_size")
        if self.encoder == "mean":
            return pooled
        from .clhe_encoder import encode_bundles

        output, self.encoder_diagnostics_ = encode_bundles(
            train_set, features, seed=self.seed, device=device, **self.encoder_params,
        )
        truncated = self.encoder_diagnostics_["truncated_bundles"]
        if truncated:
            warnings.warn("CLHE-style encoder capped members for {} bundles; see encoder_diagnostics_".format(truncated))
        return output

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)
        if not self.trainable:
            return self
        if len(np.unique(train_set.uir_tuple[1])) < 256:
            raise ValueError("real RPG OPQ needs at least 256 training-observed bundles")
        import torch

        device = ("cuda" if torch.cuda.is_available() else "cpu") if self.device == "auto" else self.device
        features = self._bundle_features(train_set, device)
        sequential, self.histories_ = bundle_to_sequential(
            train_set, features, self.n_perm, self.seed,
        )
        self.bundle_features_ = features
        self.popularity_ = np.asarray(train_set.csr_matrix.sum(axis=0)).ravel().astype(np.float32)
        self.training_diagnostics_ = {
            "synthetic_order": True,
            "empty_history_users": sum(not h for h in self.histories_),
            "singleton_history_users": sum(len(h) == 1 for h in self.histories_),
            "feature_source": "random" if self.encoder == "random" else (
                "caller_supplied" if self.item_features is not None else (
                    "supplied_user_item" if train_set.user_item_matrix is not None else "train_ub_times_bi"
                )
            ),
            "bundle_ids_sha256": hashlib.sha256(repr(list(train_set.iid_map)).encode()).hexdigest(),
            "constituent_ids_sha256": hashlib.sha256(repr(list(train_set.constituent_iid_map)).encode()).hexdigest(),
            "training_edges_sha256": hashlib.sha256(
                train_set.csr_matrix.indptr.tobytes() + train_set.csr_matrix.indices.tobytes()
            ).hexdigest(),
        }
        params = dict(n_codebook=8, pca_dim=0)
        params.update(self.rpg_params)
        # Check layout early rather than failing deep inside FAISS.
        width = features.shape[1]
        pca_dim = params.get("pca_dim", 0)
        if pca_dim and 0 < pca_dim < width:
            width = min(pca_dim, len(features) - 1, width)
        digits = params["n_codebook"]
        if not isinstance(digits, (int, np.integer)) or digits < 1 or width % digits:
            raise ValueError("effective feature width must be divisible by n_codebook")
        if params.get("codebook_size", 256) != 256:
            raise ValueError("unchanged RPG OPQ requires codebook_size=256")
        budgets = sorted(set(self.epoch_budgets)) if val_set is not None else [max(self.epoch_budgets)]
        self.val_history = []
        best_model, best_value, best_epoch = None, -np.inf, None
        for epochs in budgets:
            candidate = RPG(
                name=self.name + "_RPG", n_epochs=epochs, model_selection="last",
                scoring="exact", seed=self.seed, device=device,
                verbose=self.verbose, **params,
            )
            candidate.fit(sequential)
            self.rpg_ = candidate
            if val_set is None:
                best_model, best_epoch = candidate, epochs
                break
            from ...eval_methods import ranking_eval
            from ...metrics import NDCG

            values = ranking_eval(
                self, [NDCG(k=self.eval_k)], train_set, val_set,
            )[0]
            value = float(values[0])
            self.val_history.append((epochs, value))
            if not np.isfinite(value):
                raise ValueError("validation NDCG is not finite")
            if value > best_value:
                best_model, best_value, best_epoch = candidate, value, epochs
        self.rpg_ = best_model
        self.best_epoch = best_epoch
        self.best_value = best_value if val_set is not None else None
        codes = self.rpg_.semantic_id_catalog()
        self.training_diagnostics_["sid_collision_rate"] = 1.0 - len(np.unique(codes, axis=0)) / len(codes)
        self.training_diagnostics_["distinct_codes_per_digit"] = [len(np.unique(codes[:, i])) for i in range(codes.shape[1])]
        self.training_diagnostics_["truncated_history_users"] = sum(len(h) > self.rpg_.max_len for h in self.histories_)
        return self

    def score(self, user_idx, item_idx=None):
        if not self.knows_user(user_idx):
            raise ScoreException("Unknown user index: {}".format(user_idx))
        if item_idx is not None and not self.knows_item(item_idx):
            raise ScoreException("Unknown bundle index: {}".format(item_idx))
        if not hasattr(self, "rpg_"):
            raise ValueError("BundleRPG must be trained before scoring")
        history = self.histories_[user_idx]
        scores = self.rpg_._predict_history(history) if history else self.popularity_.copy()
        return scores if item_idx is None else scores[item_idx]
