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


class FPMC_Model(nn.Module):
    """Factorizing Personalized Markov Chains (Rendle et al., 2010).

    Score for user ``u``, last item ``i``, candidate ``j``:

        s(u, i, j) = <UI_u, IU_j> + <IL_j, LI_i>

    Implemented to return ``(B, B+N)`` matrices when out_iids contains the
    in-batch positives followed by ``N`` shared negatives.
    """

    def __init__(self, user_num, item_num, factor_num, pad_idx=None, device="cpu"):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.factor_num = factor_num
        self.device = device
        self.pad_idx = pad_idx if pad_idx is not None else item_num

        # User-Item (target) embeddings
        self.UI_emb = nn.Embedding(user_num + 1, factor_num, padding_idx=user_num)
        # Item-User factor: scoring with UI
        self.IU_emb = nn.Embedding(item_num + 1, factor_num, padding_idx=self.pad_idx)
        # Last-Item factor (input)
        self.LI_emb = nn.Embedding(item_num + 1, factor_num, padding_idx=self.pad_idx)
        # Item-Last factor: scoring with LI
        self.IL_emb = nn.Embedding(item_num + 1, factor_num, padding_idx=self.pad_idx)
        self.item_biases = nn.Embedding(item_num + 1, 1, padding_idx=self.pad_idx)

        nn.init.normal_(self.UI_emb.weight, std=0.01)
        nn.init.normal_(self.IU_emb.weight, std=0.01)
        nn.init.normal_(self.LI_emb.weight, std=0.01)
        nn.init.normal_(self.IL_emb.weight, std=0.01)
        nn.init.constant_(self.item_biases.weight, 0)
        for emb in (
            self.UI_emb,
            self.IU_emb,
            self.LI_emb,
            self.IL_emb,
            self.item_biases,
        ):
            if emb.padding_idx is not None:
                emb.weight.data[emb.padding_idx].zero_()

    def forward(self, in_uids, in_iids, out_iids):
        last_item_emb = self.LI_emb(in_iids)  # (B, D)
        user_emb = self.UI_emb(in_uids)  # (B, D)
        iu_emb = self.IU_emb(out_iids)  # (B+N, D)
        il_emb = self.IL_emb(out_iids)  # (B+N, D)
        bias = self.item_biases(out_iids)  # (B+N, 1)

        mf = torch.einsum("be,ne->bn", user_emb, iu_emb)
        fmc = torch.einsum("ne,be->bn", il_emb, last_item_emb)
        return mf + fmc + bias.T

    @torch.no_grad()
    def predict(self, user_idx, last_iid, item_indices=None):
        if item_indices is None:
            item_indices = torch.arange(self.item_num, device=self.device)
        else:
            item_indices = torch.as_tensor(
                item_indices, dtype=torch.long, device=self.device
            )
        scores = self.forward(user_idx, last_iid, item_indices).squeeze()
        return scores.detach().cpu().numpy()
