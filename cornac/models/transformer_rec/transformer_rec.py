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

import torch
import torch.nn as nn

from .backbones import get_backbone


class TransformerRecModel(nn.Module):
    """Unified HuggingFace-backbone encoder for next-item recommendation.

    A single item-embedding table feeds a swappable transformer backbone
    (``bert``/``gpt2``/``xlnet``/``electra``); candidate items are scored by
    the dot product of a position's hidden state with the item embeddings
    plus per-item biases. Training objectives (CLM/MLM/PLM/RTD) drive the
    model through :meth:`encode` and :meth:`score_positions`.

    Vocabulary layout (frozen): ``pad_idx = item_num`` and
    ``mask_idx = item_num + 1``. All embeddings have ``item_num + 2`` rows
    regardless of objective; only the first ``item_num`` rows are real items.

    Parameters
    ----------
    item_num : int
        Number of real items. Ids are integers in ``[0, item_num)``.
    backbone : str, default 'bert'
        Backbone name registered in :mod:`.backbones`.
    embedding_dim : int, default 100
        Item/hidden embedding dimension.
    maxlen : int, default 20
        Maximum sequence length fed to the encoder.
    n_layers : int, default 2
        Number of transformer layers.
    n_heads : int, default 1
        Number of attention heads.
    dropout : float, default 0.1
        Dropout probability inside the backbone.
    init_std : float, default 0.02
        Standard deviation for item-embedding initialization.
    device : str, default 'cpu'
        Torch device.
    """

    def __init__(
        self,
        item_num,
        backbone="bert",
        embedding_dim=100,
        maxlen=20,
        n_layers=2,
        n_heads=1,
        dropout=0.1,
        init_std=0.02,
        device="cpu",
    ):
        super().__init__()
        self.item_num = item_num
        self.pad_idx = item_num
        self.mask_idx = item_num + 1
        self.maxlen = maxlen
        self.dev = device
        self.init_std = init_std

        build_fn, attention_type = get_backbone(backbone)
        self.attention_type = attention_type

        vocab_size = item_num + 2
        self.item_emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=self.pad_idx,
        )
        self.backbone = build_fn(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_len=maxlen,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            pad_idx=self.pad_idx,
        )
        self.item_biases = nn.Embedding(vocab_size, 1, padding_idx=self.pad_idx)

        self._init_weights()
        self.to(device)

    def _init_weights(self):
        self.item_emb.weight.data.normal_(mean=0.0, std=self.init_std)
        self.item_emb.weight.data[self.pad_idx].zero_()
        self.item_biases.weight.data.zero_()

    def encode(self, hist_iids, **backbone_kwargs):
        """Encode a batch of item-id sequences.

        Parameters
        ----------
        hist_iids : torch.LongTensor, shape (B, T)
            Left-padded item ids (``pad_idx`` in padding slots). The mask
            token (``mask_idx``) counts as a real token in the attention mask
            because it is ``!= pad_idx``.
        **backbone_kwargs
            Extra keyword arguments forwarded to the backbone forward call
            (e.g. ``perm_mask`` / ``target_mapping`` for XLNet).

        Returns
        -------
        torch.Tensor
            The backbone ``last_hidden_state``. Shape ``(B, T, D)`` for the
            plain call, or whatever the backbone returns for special kwargs
            (e.g. ``(B, K, D)`` for XLNet with ``target_mapping``).
        """
        attention_mask = (hist_iids != self.pad_idx).long()
        embeds = self.item_emb(hist_iids)
        out = self.backbone(
            inputs_embeds=embeds, attention_mask=attention_mask, **backbone_kwargs
        )
        return out.last_hidden_state

    def score_positions(self, hidden, out_iids):
        """Score candidate items at each hidden position.

        Parameters
        ----------
        hidden : torch.Tensor, shape (M, D)
            Hidden states gathered at the loss positions.
        out_iids : torch.LongTensor, shape (M + N,)
            Candidate item ids: the ``M`` positives followed by ``N`` shared
            negatives.

        Returns
        -------
        torch.Tensor, shape (M, M + N)
            Score matrix (positives on the diagonal).
        """
        return torch.mm(hidden, self.item_emb(out_iids).T) + self.item_biases(out_iids).T

    @torch.no_grad()
    def predict(self, user_ids, log_seqs, read_position=-1, item_indices=None):
        """Score all real items for a single sequence.

        Parameters
        ----------
        user_ids : ignored
            Present for signature compatibility with the model family.
        log_seqs : array-like or torch.LongTensor, shape (1, T) or (T,)
            A single left-padded item-id sequence.
        read_position : int, default -1
            Position whose hidden state is used for scoring.
        item_indices : array-like, optional
            Candidate items to score. Defaults to all real items
            ``arange(item_num)`` (pad and mask rows are excluded).

        Returns
        -------
        numpy.ndarray, shape (len(item_indices),)
            Scores for the requested items.
        """
        if item_indices is None:
            item_indices = torch.arange(self.item_num, device=self.dev)
        else:
            item_indices = torch.as_tensor(
                item_indices, dtype=torch.long, device=self.dev
            )
        if not isinstance(log_seqs, torch.Tensor):
            log_seqs = torch.as_tensor(log_seqs, dtype=torch.long, device=self.dev)
        if log_seqs.dim() == 1:
            log_seqs = log_seqs.unsqueeze(0)
        hidden = self.encode(log_seqs)[:, read_position, :]
        scores = self.score_positions(hidden, item_indices)
        return scores.squeeze().detach().cpu().numpy()
