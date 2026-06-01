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
"""
Custom optimizer(s) for sequential recommendation models.
"""

import torch
from torch.optim import Optimizer


class IndexedAdagradM(Optimizer):
    """Sparse-aware Adagrad with momentum, used by GRU4Rec and FPMC."""

    def __init__(self, params, lr=0.05, momentum=0.0, eps=1e-6):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if eps <= 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr, momentum=momentum, eps=eps)
        super(IndexedAdagradM, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["acc"] = torch.full_like(p, 0, memory_format=torch.preserve_format)
                if momentum > 0:
                    state["mom"] = torch.full_like(p, 0, memory_format=torch.preserve_format)

    def share_memory(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["acc"].share_memory_()
                if group["momentum"] > 0:
                    state["mom"].share_memory_()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                clr = group["lr"]
                momentum = group["momentum"]
                if grad.is_sparse:
                    grad = grad.coalesce()
                    grad_indices = grad._indices()[0]
                    grad_values = grad._values()
                    accs = state["acc"][grad_indices] + grad_values.pow(2)
                    state["acc"].index_copy_(0, grad_indices, accs)
                    accs.add_(group["eps"]).sqrt_().mul_(-1 / clr)
                    if momentum > 0:
                        moma = state["mom"][grad_indices]
                        moma.mul_(momentum).add_(grad_values / accs)
                        state["mom"].index_copy_(0, grad_indices, moma)
                        p.index_add_(0, grad_indices, moma)
                    else:
                        p.index_add_(0, grad_indices, grad_values / accs)
                else:
                    state["acc"].add_(grad.pow(2))
                    accs = state["acc"].add(group["eps"])
                    accs.sqrt_()
                    if momentum > 0:
                        mom = state["mom"]
                        mom.mul_(momentum).addcdiv_(grad, accs, value=-clr)
                        p.add_(mom)
                    else:
                        p.addcdiv_(grad, accs, value=-clr)
        return loss
