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

import numpy as np
from tqdm.auto import trange

from cornac.models.recommender import NextItemRecommender

from ...utils import get_rng
from ..seq_utils import val_score
from .backbones import ATTENTION_TYPES

SUPPORTED_LOSSES = (
    "bce",
    "ce",
    "bpr",
    "bpr-max",
    "softmax",
    "cross-entropy",
    "xe_softmax",
    "top1",
)

VALID_OBJECTIVES = ("clm", "mlm", "plm", "rtd")
VALID_LOSS_AT = ("all", "last")


class TransformerRec(NextItemRecommender):
    """TransformerRec: unified transformer next-item recommender.

    A single item-embedding scoring head over a swappable HuggingFace
    transformer backbone (``bert``/``gpt2``/``xlnet``/``electra``), trained
    with one of four self-supervised objectives (``clm``/``mlm``/``plm``/
    ``rtd``). This subsumes BERT4Rec and the Transformers4Rec family in one
    model; see the validity matrix below for the objective/backbone
    combinations that are supported.

    Validity matrix (enforced at construction time)
    -----------------------------------------------
    ==============================  ============================  ==========
    objective                       valid backbones               loss_at
    ==============================  ============================  ==========
    clm (loss_at='all')             gpt2 (causal only)            'all'
    clm (loss_at='last', legacy)    any backbone                  'last'
    mlm                             bert, electra, xlnet          'all'
    plm                             xlnet only                    'all'
    rtd                             bert, electra                 'all'
    ==============================  ============================  ==========

    Parameters
    ----------
    name: str, default: 'TransformerRec'
        The name of the recommender model.

    backbone: str, default: 'bert'
        Transformer backbone. One of 'bert', 'gpt2', 'xlnet', 'electra'.

    objective: str, default: 'mlm'
        Self-supervised training objective. One of 'clm' (causal LM),
        'mlm' (masked LM / Cloze), 'plm' (permutation LM), 'rtd'
        (replaced-token detection).

    loss_at: str, default: 'all'
        Where the training loss is applied. 'all' scores every eligible
        position of the session; 'last' (only valid with ``objective='clm'``)
        is the legacy Cornac prefix-breakdown setting that scores the last
        position of each expanded prefix and works with any backbone.

    embedding_dim: int, optional, default: 100
        Item embedding / hidden dimension.

    loss: str, optional, default: 'ce'
        Loss function. Supported: 'bce', 'ce', 'bpr', 'bpr-max', 'softmax'
        (a.k.a 'cross-entropy' / 'xe_softmax'), 'top1'.

    batch_size: int, optional, default: 512

    learning_rate: float, optional, default: 0.001

    n_sample: int, optional, default: 2048
        Number of negative samples shared per mini-batch (0 disables extra
        sampled negatives, using in-batch negatives only).

    sample_alpha: float, optional, default: 0.5
        Popularity-based negative sampling exponent.

    n_epochs: int, optional, default: 10

    max_len: int, optional, default: 50
        Maximum session length fed to the encoder.

    num_blocks: int, optional, default: 2
        Number of transformer layers.

    num_heads: int, optional, default: 1
        Number of attention heads.

    dropout: float, optional, default: 0.2

    l2_reg: float, optional, default: 0.0

    bpreg: float, optional, default: 1.0
        Regularization coefficient for the 'bpr-max' loss.

    elu_param: float, optional, default: 0.5
        ELU parameter for the 'bpr-max' loss.

    mask_prob: float, optional, default: 0.2
        Per-position masking probability for the 'mlm', 'plm', and 'rtd'
        objectives.

    rtd_lambda: float, optional, default: 1.0
        (RTD only) weight of the discriminator loss relative to the MLM
        (generator) loss. Larger values favor the discriminative signal at
        the expense of ranking alignment; ``0`` recovers plain MLM.

    device: str, optional, default: 'cpu'
        Set to 'cuda' for GPU support.

    trainable: bool, optional, default: True
        When False, the model will not be re-trained.

    verbose: bool, optional, default: False
        When True, running logs are displayed.

    seed: int, optional, default: None
        Random seed for weight initialization and negative sampling.

    model_selection: str, optional, default: 'last'
        One of 'last' or 'best'. When 'best', the model with the highest
        validation score (evaluated every ``val_eval_every`` epochs) is
        restored at the end of ``fit``.

    val_eval_every: int, optional, default: 5
    val_k: int, optional, default: 20
    val_metric: str, optional, default: 'recall'
        Cutoff and metric used for best-on-val selection. See
        :func:`cornac.models.seq_utils.val_score`.

    Note
    ----
    * ``backbone='bert', objective='mlm'`` reproduces the canonical BERT4Rec
      Cloze training (Sun et al., 2019).
    * ``backbone='gpt2', objective='clm'`` reproduces the Transformers4Rec
      GPT-2 / causal-LM setup (Moreira et al., 2021).
    * ``objective='clm', loss_at='last'`` is the legacy Cornac prefix
      breakdown (next-item-at-last-position), valid with any backbone.
    * With ``model_selection='best'`` only the main model's ``state_dict`` is
      snapshotted/restored; RTD's auxiliary discriminator head is not
      checkpoint-restored since it is not used at serving.

    References
    ----------
    Moreira, G. de S. P., Rabhi, S., Lee, J. M., Ak, R., & Oldridge, E.
    (2021). Transformers4Rec: Bridging the gap between NLP and sequential /
    session-based recommendation. RecSys.

    Sun, F., Liu, J., Wu, J., Pei, C., Lin, X., Ou, W., & Jiang, P. (2019).
    BERT4Rec: Sequential recommendation with bidirectional encoder
    representations from transformer. CIKM.

    Clark, K., Luong, M.-T., Le, Q. V., & Manning, C. D. (2020). ELECTRA:
    Pre-training text encoders as discriminators rather than generators.
    ICLR.

    Yang, Z., Dai, Z., Yang, Y., Carbonell, J., Salakhutdinov, R., & Le,
    Q. V. (2019). XLNet: Generalized autoregressive pretraining for language
    understanding. NeurIPS.
    """

    def __init__(
        self,
        name="TransformerRec",
        backbone="bert",
        objective="mlm",
        loss_at="all",
        embedding_dim=100,
        loss="ce",
        batch_size=512,
        learning_rate=0.001,
        n_sample=2048,
        sample_alpha=0.5,
        n_epochs=10,
        max_len=50,
        num_blocks=2,
        num_heads=1,
        dropout=0.2,
        l2_reg=0.0,
        bpreg=1.0,
        elu_param=0.5,
        mask_prob=0.2,
        rtd_lambda=1.0,
        device="cpu",
        trainable=True,
        verbose=False,
        seed=None,
        model_selection="last",
        val_eval_every=5,
        val_k=20,
        val_metric="recall",
    ):
        super().__init__(name, trainable=trainable, verbose=verbose)
        if objective not in VALID_OBJECTIVES:
            raise ValueError(
                f"objective='{objective}' not supported; choose from {VALID_OBJECTIVES}"
            )
        if loss_at not in VALID_LOSS_AT:
            raise ValueError(
                f"loss_at='{loss_at}' not supported; choose from {VALID_LOSS_AT}"
            )
        if backbone not in ATTENTION_TYPES:
            raise ValueError(
                f"Unknown backbone '{backbone}'; choose from {sorted(ATTENTION_TYPES)}"
            )
        if loss not in SUPPORTED_LOSSES:
            raise ValueError(
                f"loss='{loss}' not supported; choose from {SUPPORTED_LOSSES}"
            )
        if model_selection not in ("last", "best"):
            raise ValueError(
                f"model_selection='{model_selection}' not supported; choose 'last' or 'best'"
            )

        self._validate_combo(objective, loss_at, backbone)

        self.backbone = backbone
        self.objective = objective
        self.loss_at = loss_at
        self.embedding_dim = embedding_dim
        self.loss = loss
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_sample = n_sample
        self.sample_alpha = sample_alpha
        self.n_epochs = n_epochs
        self.max_len = max_len
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.bpreg = bpreg
        self.elu_param = elu_param
        self.mask_prob = mask_prob
        self.rtd_lambda = rtd_lambda
        self.device = device
        self.seed = seed
        self.rng = get_rng(seed)
        self.model_selection = model_selection
        self.val_eval_every = val_eval_every
        self.val_k = val_k
        self.val_metric = val_metric

    @staticmethod
    def _validate_combo(objective, loss_at, backbone):
        """Enforce the objective/backbone/loss_at validity matrix."""
        attn = ATTENTION_TYPES[backbone]
        if objective == "clm":
            if loss_at == "all" and attn != "causal":
                raise ValueError(
                    f"objective='clm' with loss_at='all' requires a causal "
                    f"backbone (e.g. 'gpt2'), but backbone='{backbone}' is "
                    f"'{attn}'. Use loss_at='last' for the legacy prefix mode "
                    f"with this backbone."
                )
            return  # loss_at='last' works with any backbone

        # mlm / plm / rtd are all whole-session ('all') objectives.
        if loss_at != "all":
            raise ValueError(
                f"objective='{objective}' only supports loss_at='all', "
                f"got loss_at='{loss_at}'."
            )
        if objective == "mlm":
            if attn != "bidirectional":
                raise ValueError(
                    f"objective='mlm' requires a bidirectional backbone "
                    f"(bert, electra, xlnet), but backbone='{backbone}' is "
                    f"'{attn}'."
                )
        elif objective == "plm":
            if backbone != "xlnet":
                raise ValueError(
                    f"objective='plm' requires backbone='xlnet', "
                    f"got backbone='{backbone}'."
                )
        elif objective == "rtd":
            if backbone not in ("bert", "electra"):
                raise ValueError(
                    f"objective='rtd' requires backbone in ('bert', 'electra'), "
                    f"got backbone='{backbone}'."
                )

    def _build_objective(self):
        """Instantiate the objective implementation from its name."""
        from .objectives import (
            CLMObjective,
            MLMObjective,
            PLMObjective,
            RTDObjective,
        )

        name = self.objective
        if name == "clm":
            return CLMObjective(self.pad_idx, self.mask_idx, self.rng)
        if name == "mlm":
            return MLMObjective(
                self.pad_idx, self.mask_idx, self.rng, mask_prob=self.mask_prob
            )
        if name == "plm":
            return PLMObjective(
                self.pad_idx, self.mask_idx, self.rng, mask_prob=self.mask_prob
            )
        return RTDObjective(
            self.pad_idx,
            self.mask_idx,
            self.rng,
            mask_prob=self.mask_prob,
            rtd_lambda=self.rtd_lambda,
        )

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)
        if not self.trainable:
            return self

        import torch

        from .transformer_rec import TransformerRecModel
        from ..seq_utils import (
            build_neg_sampler,
            padded_session_iter,
            session_seq_iter,
        )
        from ..seq_utils.losses import get_loss_function

        torch.manual_seed(self.seed if self.seed is not None else 0)

        use_prefix = self.objective == "clm" and self.loss_at == "last"

        self.pad_idx = self.total_items
        self.mask_idx = self.total_items + 1
        self.model = TransformerRecModel(
            item_num=self.total_items,
            backbone=self.backbone,
            embedding_dim=self.embedding_dim,
            maxlen=self.max_len,
            n_layers=self.num_blocks,
            n_heads=self.num_heads,
            dropout=self.dropout,
            device=self.device,
        )

        # The built instance lives in ``objective_`` so ``self.objective``
        # stays the plain string hyperparameter (kept intact for clone()).
        objective = self._build_objective()
        objective.build(self.model, self.device)
        self.objective_ = objective

        loss_fn = get_loss_function(self.loss)
        loss_kwargs = dict(
            bpreg=self.bpreg, elu_param=self.elu_param, n_sample=self.n_sample
        )

        if self.n_sample > 0:
            item_indices, item_dist = build_neg_sampler(
                train_set.uir_tuple, self.sample_alpha
            )
            sample_negatives = lambda n: self.rng.choice(
                item_indices, size=n, replace=True, p=item_dist
            )
        else:
            sample_negatives = None

        opt = torch.optim.Adam(
            list(self.model.parameters()) + list(objective.parameters()),
            lr=self.learning_rate,
            betas=(0.9, 0.98),
        )

        best_val = -float("inf")
        best_state = None
        progress_bar = trange(1, self.n_epochs + 1, disable=not self.verbose)
        for epoch_id in progress_bar:
            self.model.train()
            total_loss = 0.0
            cnt = 0

            if use_prefix:
                for inc, (in_uids, hist_iids, out_iids) in enumerate(
                    session_seq_iter(
                        self.train_set,
                        pad_index=self.pad_idx,
                        batch_size=self.batch_size,
                        max_len=self.max_len,
                        n_sample=self.n_sample,
                        sample_alpha=self.sample_alpha,
                        rng=self.rng,
                        shuffle=True,
                    )
                ):
                    if len(hist_iids) < 2:
                        continue
                    hist_iids_t = torch.tensor(
                        hist_iids, dtype=torch.long, device=self.device
                    )
                    out_iids_t = torch.tensor(
                        out_iids, dtype=torch.long, device=self.device
                    )

                    self.model.zero_grad()
                    L = objective.compute_loss_prefix(
                        self.model, hist_iids_t, out_iids_t, loss_fn, loss_kwargs
                    )
                    if self.l2_reg > 0:
                        for p in self.model.parameters():
                            L = L + self.l2_reg * torch.norm(p)

                    L.backward()
                    opt.step()

                    total_loss += L.cpu().detach().numpy() * len(hist_iids)
                    cnt += len(hist_iids)
                    if inc % 10 == 0 and cnt > 0:
                        progress_bar.set_postfix(loss=(total_loss / cnt))
            else:
                for inc, (uids, seqs) in enumerate(
                    padded_session_iter(
                        self.train_set,
                        pad_index=self.pad_idx,
                        batch_size=self.batch_size,
                        max_len=self.max_len,
                        rng=self.rng,
                        shuffle=True,
                    )
                ):
                    if len(seqs) < 2:
                        continue

                    self.model.zero_grad()
                    L = objective.compute_loss(
                        self.model, seqs, sample_negatives, loss_fn, loss_kwargs
                    )
                    if self.l2_reg > 0:
                        for p in self.model.parameters():
                            L = L + self.l2_reg * torch.norm(p)

                    L.backward()
                    opt.step()

                    total_loss += L.cpu().detach().numpy() * len(seqs)
                    cnt += len(seqs)
                    if inc % 10 == 0 and cnt > 0:
                        progress_bar.set_postfix(loss=(total_loss / cnt))

            if (
                self.model_selection == "best"
                and val_set is not None
                and epoch_id % self.val_eval_every == 0
            ):
                score = val_score(
                    self, self.train_set, val_set, metric=self.val_metric, k=self.val_k
                )
                if score is not None and score > best_val:
                    best_val = score
                    best_state = {
                        n: p.detach().clone()
                        for n, p in self.model.state_dict().items()
                    }

        if self.model_selection == "best" and best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def score(self, user_idx, history_items, **kwargs):
        if len(history_items) == 0:
            return np.ones(self.total_items, dtype="float")
        inp, pos = self.objective_.prepare_score_input(
            list(history_items), self.max_len, self.pad_idx
        )
        self.model.eval()
        return self.objective_.predict_scores(self.model, inp, pos)
