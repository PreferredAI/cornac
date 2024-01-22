import itertools
import random

import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.nn.init import constant_, xavier_normal_
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm.auto import trange


class Wloss(nn.modules.loss._Loss):
    def __init__(self, p, n):
        super(Wloss, self).__init__()
        self.p = p
        self.n = n
        if p > n:
            self.mode = "positive"
        else:
            self.mode = "negative"

    def forward(self, pred, tgt, cand):
        loss = 0.0
        if self.mode == "positive":
            for ind in range(pred.size(0)):
                if ind in tgt:
                    loss += -torch.log(pred[ind]) * self.p
                else:
                    loss += -torch.log(1 - pred[ind]) * self.n
        elif self.mode == "negative":
            for ind in range(pred.size(0)):
                if ind in tgt:
                    loss += -torch.log(pred[ind]) * self.p
                else:
                    if ind in cand:
                        loss += -torch.log(1 - pred[ind]) * self.n
                    else:
                        loss += -torch.log(1 - pred[ind])
        return loss / pred.size(0)


class DREAM(nn.Module):
    def __init__(
        self,
        n_items,
        emb_size,
        emb_type,
        hidden_size,
        dropout_prob,
        max_seq_length,
        loss_mode,
        loss_uplift,
        attention,
        device="cpu",
        seed=None,
    ):
        super(DREAM, self).__init__()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # device setting
        self.device = device

        # dataset features
        self.n_items = n_items

        # model parameters
        self.emb_size = emb_size
        self.emb_type = emb_type
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.max_seq_length = max_seq_length  # max sequence length
        self.loss_mode = loss_mode
        self.loss_uplift = loss_uplift

        self.BasketEmbedding = BasketEmbedding(
            hidden_size=self.emb_size,
            n_items=self.n_items,
            max_seq_length=self.max_seq_length,
            type=self.emb_type,
            device=self.device,
        )
        self.gru = nn.GRU(
            input_size=self.emb_size,
            hidden_size=self.hidden_size,
            batch_first=True,
        )
        self.attention = attention
        self.decoder = Decoder(
            hidden_size=self.hidden_size,
            max_seq_length=self.max_seq_length,
            num_item=self.n_items,
            dropout_prob=self.dropout_prob,
            attention=self.attention,
            device=self.device,
        )

        self.loss_fct = nn.BCELoss()
        self.p_loss_fct = Wloss(self.loss_uplift, 1)
        self.n_loss_fct = Wloss(1, self.loss_uplift)
        self.meta_loss_fct = nn.MSELoss()
        self.sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, basket_seq):
        basket_seq_len = []
        for b in basket_seq:
            basket_seq_len.append(len(b))
        basket_seq_len = torch.as_tensor(basket_seq_len).to(self.device)
        batch_basket_seq_embed = self.BasketEmbedding(basket_seq)
        all_memory, _ = self.gru(batch_basket_seq_embed)
        last_memory = self.gather_indexes(all_memory, basket_seq_len - 1)
        timeline_mask = get_timeline_mask(
            batch_basket_seq_embed, self.device, self.emb_size
        )
        pred = self.decoder.forward(all_memory, last_memory, timeline_mask)
        return pred

    def get_batch_loss(self, pred, tgt, cand, tag, device):
        batch_size = pred.size(0)
        tmp_tgt = get_label_tensor(tgt, device, self.n_items)
        loss = 0.0
        if self.loss_mode == 0:
            for ind in range(batch_size):
                pred_ind = torch.clamp(pred[ind], 0.001, 0.999)
                loss += self.loss_fct(pred_ind.unsqueeze(0), tmp_tgt[ind].unsqueeze(0))
        if self.loss_mode == 1:
            if tag == "negative":
                for ind in range(batch_size):
                    user_pred_ind = torch.clamp(pred[ind], 0.001, 0.999)
                    user_tgt = torch.tensor(tgt[ind])
                    user_cand = torch.tensor(cand[ind])
                    loss += self.n_loss_fct(user_pred_ind, user_tgt, user_cand)
            if tag == "positive":
                for ind in range(batch_size):
                    user_pred_ind = torch.clamp(pred[ind], 0.001, 0.999)
                    user_tgt = torch.tensor(tgt[ind])
                    user_cand = torch.tensor(cand[ind])
                    loss += self.p_loss_fct(user_pred_ind, user_tgt, user_cand)
        return loss / batch_size  # compute average

    def global_loss(self, basket_seq, tgt_basket, cand_basket):
        prediction = self.forward(basket_seq)
        cand = [
            l1 + l2 for l1, l2 in zip(cand_basket["repeat"], cand_basket["explore"])
        ]
        loss = self.get_batch_loss(
            prediction, tgt_basket, cand, "positive", self.device
        )
        return loss

    def calculate_loss(self, basket_seq, tgt_basket, cand_basket):
        global_loss = self.global_loss(basket_seq, tgt_basket, cand_basket)
        return global_loss

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)


class BasketEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size,
        n_items,
        max_seq_length,
        type,
        device,
    ):  # hidden_size is the emb_size
        super(BasketEmbedding, self).__init__()
        self.hidden_size = hidden_size
        self.n_items = n_items
        self.max_seq_length = max_seq_length
        self.type = type
        self.device = device
        self.padding_idx = n_items
        self.item_embedding = nn.Embedding(n_items + 1, hidden_size)
        self.item_embedding.weight.data[-1] = torch.zeros(hidden_size)

    def forward(self, batch_basket):
        # need to padding here
        batch_embed_seq = []  # batch * max_seq_length * hidden size
        for basket_seq in batch_basket:
            embed_baskets = []
            for basket in basket_seq:
                basket = torch.LongTensor(basket).resize_(1, len(basket))
                basket = Variable(basket).to(self.device)
                basket = self.item_embedding(basket).squeeze(0)
                if self.type == "mean":
                    embed_baskets.append(torch.mean(basket, 0))
                if self.type == "max":
                    embed_baskets.append(torch.max(basket, 0)[0])
                if self.type == "sum":
                    embed_baskets.append(torch.sum(basket, 0))
            # padding the seq
            pad_num = self.max_seq_length - len(embed_baskets)
            for _ in range(pad_num):
                embed_baskets.append(
                    torch.tile(
                        torch.tensor([self.padding_idx], device=self.device),
                        dims=(self.hidden_size,),
                    )
                )
            embed_seq = torch.stack(embed_baskets, 0)
            embed_seq = torch.as_tensor(embed_seq)
            batch_embed_seq.append(embed_seq)
        batch_embed_output = torch.stack(batch_embed_seq, 0).to(self.device)
        return batch_embed_output


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        max_seq_length,
        num_item,
        dropout_prob,
        attention,
        device,
    ):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden_size = hidden_size
        self.device = device
        self.max_seq_length = max_seq_length
        self.n_items = num_item
        self.attention = attention

        if self.attention == "attention":
            self.W_repeat = nn.Linear(hidden_size, hidden_size, bias=False)
            self.U_repeat = nn.Linear(hidden_size, hidden_size, bias=False)
            self.tanh = nn.Tanh()
            self.V_repeat = nn.Linear(hidden_size, 1)
            self.Repeat = nn.Linear(hidden_size * 2, num_item)
        else:
            self.Repeat = nn.Linear(hidden_size, num_item)

    def forward(self, all_memory, last_memory, mask=None):
        """item_seq is the appared items or candidate items"""
        if self.attention == "attention":
            all_memory_values, last_memory_values = all_memory, last_memory
            all_memory = self.dropout(self.U_repeat(all_memory))
            last_memory = self.dropout(self.W_repeat(last_memory))
            last_memory = last_memory.unsqueeze(1)
            last_memory = last_memory.repeat(1, self.max_seq_length, 1)

            output_er = self.tanh(all_memory + last_memory)
            output_er = self.V_repeat(output_er).squeeze(-1)

            if mask is not None:
                output_er.masked_fill_(mask, -1e9)

            output_er = output_er.unsqueeze(-1)

            alpha_r = nn.Softmax(dim=1)(output_er)
            alpha_r = alpha_r.repeat(1, 1, self.hidden_size)
            output_r = (all_memory_values * alpha_r).sum(dim=1)
            output_r = torch.cat([output_r, last_memory_values], dim=1)
            output_r = self.dropout(self.Repeat(output_r))

            decoder = torch.sigmoid(output_r)
        else:
            decoder = torch.sigmoid(self.dropout(self.Repeat(last_memory)))

        return decoder


def get_timeline_mask(batch_basket_emb, device, emb_size):
    batch_mask = []
    for basket_seq in batch_basket_emb:
        mask = []
        for basket_emb in basket_seq:
            if torch.equal(basket_emb, torch.zeros(emb_size).to(device)):
                mask.append(1)
            else:
                mask.append(0)
        batch_mask.append(torch.as_tensor(mask).bool())
    batch_mask = torch.stack(batch_mask, 0).to(device)
    return batch_mask.bool()


def get_label_tensor(labels, device, max_index=None):
    """Candidates is the output of basic models or repeat or popular
    labels is list[]"""
    batch_size = len(labels)
    if torch.cuda.is_available():
        label_tensor = torch.FloatTensor(batch_size, max_index).fill_(0.0).to(device)
    else:
        label_tensor = torch.zeros(batch_size, max_index)
    for ind in range(batch_size):
        if len(labels[ind]) != 0:
            label_tensor[ind].scatter_(0, torch.as_tensor(labels[ind]).to(device), 1)
    label_tensor.requires_grad = False  # because this is not trainable
    return label_tensor


def transform_data(batch_basket_items, max_seq_length):
    batch_history_basket_items = []
    batch_target_items = []
    candidates = {"repeat": [], "explore": []}
    for basket_items in batch_basket_items:
        history_basket_items = basket_items[-max_seq_length - 1 : -1]
        target_items = basket_items[-1]
        batch_history_basket_items.append(history_basket_items)
        batch_target_items.append(target_items)
        history_items = set(itertools.chain.from_iterable(history_basket_items))
        repeat_items = [iid for iid in target_items if iid in history_items]
        explore_items = [iid for iid in target_items if iid not in repeat_items]
        candidates["repeat"].append(repeat_items)
        candidates["explore"].append(explore_items)
    return batch_history_basket_items, batch_target_items, candidates


def learn(
    model,
    train_set,
    val_set,
    max_seq_length,
    lr,
    weight_decay,
    n_epochs,
    batch_size,
    verbose,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = model.calculate_loss
    last_loss = np.inf
    last_val_loss = np.inf
    progress_bar = trange(1, n_epochs + 1, disable=not verbose)
    for _ in progress_bar:
        model.train()
        for inc, (_, _, batch_basket_items) in enumerate(
            train_set.ubi_iter(batch_size=batch_size, shuffle=True)
        ):
            batch_history_basket_items, batch_target_items, candidates = transform_data(
                batch_basket_items, max_seq_length=max_seq_length
            )
            optimizer.zero_grad()
            loss = loss_func(batch_history_basket_items, batch_target_items, candidates)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            last_loss = loss.data.item()
            if inc % 10:
                progress_bar.set_postfix(loss=last_loss, val_loss=last_val_loss)

        if val_set is not None:
            model.eval()
            for inc, (_, _, batch_basket_items) in enumerate(
                val_set.ubi_iter(batch_size=batch_size, shuffle=True)
            ):
                (
                    batch_history_basket_items,
                    batch_target_items,
                    candidates,
                ) = transform_data(batch_basket_items, max_seq_length=max_seq_length)
                loss = loss_func(
                    batch_history_basket_items, batch_target_items, candidates
                )
                last_val_loss = loss.data.item()
                if inc % 10:
                    progress_bar.set_postfix(loss=last_loss, val_loss=last_val_loss)
