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

from ..transformer_rec import TransformerRec


class BERT4Rec(TransformerRec):
    """BERT4Rec: a bidirectional transformer encoder for sequential rec.

    A light interface over :class:`~cornac.models.TransformerRec` fixed to
    ``backbone='bert', objective='clm', loss_at='last'``, i.e. the
    next-item-at-last-position training shared by the transformer family in
    Cornac. See the :class:`~cornac.models.TransformerRec` docstring for the
    shared parameters (``loss``, ``model_selection``, negative sampling,
    etc.).

    Note
    ----
    The original paper trains with the masked-language-model (Cloze) objective;
    that canonical setting is available as
    ``TransformerRec(backbone='bert', objective='mlm')``.
    In our experiments, the last-position objective gives better performance.

    References
    ----------
    Sun, F., Liu, J., Wu, J., Pei, C., Lin, X., Ou, W., & Jiang, P. (2019).
    BERT4Rec: Sequential recommendation with bidirectional encoder
    representations from transformer. CIKM.
    """

    def __init__(
        self,
        name="BERT4Rec",
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
        device="cpu",
        trainable=True,
        verbose=False,
        seed=None,
        model_selection="last",
        val_eval_every=5,
        val_k=20,
        val_metric="recall",
    ):
        super().__init__(
            name=name,
            backbone="bert",
            objective="clm",
            loss_at="last",
            embedding_dim=embedding_dim,
            loss=loss,
            batch_size=batch_size,
            learning_rate=learning_rate,
            n_sample=n_sample,
            sample_alpha=sample_alpha,
            n_epochs=n_epochs,
            max_len=max_len,
            num_blocks=num_blocks,
            num_heads=num_heads,
            dropout=dropout,
            l2_reg=l2_reg,
            bpreg=bpreg,
            elu_param=elu_param,
            device=device,
            trainable=trainable,
            verbose=verbose,
            seed=seed,
            model_selection=model_selection,
            val_eval_every=val_eval_every,
            val_k=val_k,
            val_metric=val_metric,
        )
