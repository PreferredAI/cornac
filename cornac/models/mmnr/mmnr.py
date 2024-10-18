import math
from collections import Counter
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from tqdm.auto import trange

OPTIMIZER_DICT = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "rmsprop": torch.optim.RMSprop,
    "adagrad": torch.optim.Adagrad,
}


class Model(nn.Module):
    def __init__(
        self,
        n_items,
        emb_dim=32,
        n_aspects=11,
        padding_idx=None,
        ctx=3,
        d1=5,
        d2=5,
    ):
        super(Model, self).__init__()

        self.emb_dim = emb_dim
        self.n_aspects = n_aspects
        self.padding_idx = padding_idx if padding_idx is not None else n_items
        self.ctx = ctx
        self.d1 = d1
        self.d2 = d2

        self.item_embedding = nn.Embedding(
            n_items + 1, self.emb_dim, padding_idx=self.padding_idx
        )

        # Aspect-Specific Projection Matrices (K different aspects)
        self.aspProj = nn.Parameter(
            torch.Tensor(self.n_aspects, self.emb_dim, self.d1), requires_grad=True
        )
        self.aspProjSeq = nn.Parameter(
            torch.Tensor(2 * self.n_aspects, self.d1, self.d2), requires_grad=True
        )
        torch.nn.init.xavier_normal_(self.aspProj.data, gain=1)
        torch.nn.init.xavier_normal_(self.aspProjSeq.data, gain=1)

        self.out = nn.Linear(self.d2, n_items)
        self.his_linear_embds = nn.Linear(n_items, self.d2)
        self.his_nn_embds = nn.Embedding(
            n_items + 1, self.d2, padding_idx=self.padding_idx
        )
        self.gate_his = nn.Linear(self.d2, 1)

        self.asp_h1_h2 = nn.Linear(self.d1, self.d2)

    def forward(self, seq, decay, uHis, iHis, device):

        batch = seq.shape[0]  # batch
        self.max_seq = seq.shape[1]  # L
        self.max_bas = seq.shape[2]  # B

        # Multi-view Embedding
        uEmbs, iEmbs = self.EmbeddingLayer(
            batch, seq, uHis, iHis, device
        )  # [batch, L, B, d]

        # Multi-aspect Representation Learning
        uEmbsAsp = self.AspectLearning(uEmbs, batch, device)  # [batch, asp, L, h1]
        iEmbsAsp = self.AspectLearning(iEmbs, batch, device)

        # decay [batch, L, 1]
        decay = decay.unsqueeze(1)  # [batch, 1, L, 1]
        decay = decay.repeat(1, self.n_aspects, 1, 1)  # [batch, asp, L, 1]
        uEmbsAspDec = uEmbsAsp * decay  # decay[batch, asp, L, 1]->[batch, asp, L, h1]
        iEmbsAspDec = iEmbsAsp * decay  # decay[batch, asp, L, 1]->[batch, asp, L, h1]

        uAsp = self.asp_h1_h2(torch.sum(uEmbsAspDec, dim=2) / self.max_seq)
        iAsp = self.asp_h1_h2(torch.sum(iEmbsAspDec, dim=2) / self.max_seq)

        result, loss_cl = self.PredictionLayer(uAsp, iAsp, uHis)

        return result, loss_cl

    def EmbeddingLayer(self, batch, seq, uHis, iHis, device):
        """
        input:
            seq [batch, L, B, d]
        output:
            userEmbs [batch, L, B, d]
            itemEmbs [batch, L, B, d]
        """
        embs = self.item_embedding(seq)

        # [batch*max_num_seq*max_bas]
        row = (
            torch.arange(batch)
            .repeat(self.max_seq * self.max_bas, 1)
            .transpose(0, 1)
            .reshape(-1)
        )
        col = seq.reshape(len(seq), -1).reshape(-1)  # [batch, L, B]

        # padded = torch.zeros(batch, 1).to(device)  # [batch, 1]
        padded = torch.zeros(batch, 1).fill_(0).to(device)  # [batch, 1]
        userHis = torch.cat((uHis, padded), dim=1)  # [batch, n_items+1]
        itemHis = torch.cat((iHis, padded), dim=1)  # [batch, n_items+1]

        uMatrix = userHis[row, col].reshape(
            batch, self.max_seq, -1, 1
        )  # [batch, L, B, 1]
        iMatrix = itemHis[row, col].reshape(
            batch, self.max_seq, -1, 1
        )  # [batch, L, B, 1]

        uEmbs = embs * uMatrix
        iEmbs = embs * iMatrix

        return uEmbs, iEmbs

    def AspectLearning(self, embs, batch, device):
        """
        input:
            uEmbs [batch, L, B, d]
            iEmbs [batch, L, B, d]
        output:
            basketAsp  [batch, asp, L, h1]
        """

        # Aspect Embeddings (basket)
        self.aspEmbed = nn.Embedding(self.n_aspects, self.ctx * self.d1).to(device)
        self.aspEmbed.weight.requires_grad = True
        torch.nn.init.xavier_normal_(self.aspEmbed.weight.data, gain=1)

        # Loop over all aspects
        asp_lst = []
        for a in range(self.n_aspects):
            self.norm = nn.LayerNorm(self.aspProj[a].shape[1]).to(device)

            # [batch, L, B, d] × [d, h1] = [batch, L, B, h1]
            aspProj = torch.tanh(torch.matmul(embs, self.norm(self.aspProj[a])))

            # [batch, L, 1] -> [batch, L, 1, h1]
            aspEmbed = self.aspEmbed(
                torch.LongTensor(batch, self.max_seq, 1).fill_(a).to(device)
            )
            aspEmbed = torch.transpose(aspEmbed, 2, 3)  # [batch, L, h1, 1]

            if self.ctx == 1:
                # [batch, L, B, (1*h1)] × [batch, L, (1*h1), 1] = [batch, L, B, 1]
                aspAttn = torch.matmul(aspProj, aspEmbed)
                aspAttn = F.softmax(aspAttn, dim=2)  # [batch,L,B,1]
            else:
                pad_size = int((self.ctx - 1) / 2)

                # [batch, max_len, max_bas+1+1, h1]; pad_size=1
                aspProj_padded = F.pad(
                    aspProj, (0, 0, pad_size, pad_size), "constant", 0
                )

                # [batch,L,B+1+1,h1]->[batch,L,B,h1,ctx]
                aspProj_padded = aspProj_padded.unfold(2, self.ctx, 1)  # sliding
                aspProj_padded = torch.transpose(aspProj_padded, 3, 4)
                # [batch, max_len, max_bas, ctx*h1]
                aspProj_padded = aspProj_padded.contiguous().view(
                    -1, self.max_seq, self.max_bas, self.ctx * self.d1
                )

                # Calculate Attention: Inner Product & Softmax
                # [batch, L,B, (ctx*h1)] x [batch, L, (ctx*h1), 1] -> [batch, L, B, 1]
                aspAttn = torch.matmul(aspProj_padded, aspEmbed)
                aspAttn = F.softmax(aspAttn, dim=2)  # [batch, max_len, max_bas, 1]

            # [batch, L, B, h1] x [batch, L, B, 1]
            aspItem = aspProj * aspAttn.expand_as(aspProj)  # [batch, L, B, h1]
            batch_asp = torch.sum(aspItem, dim=2)  # [batch, L, h1]

            # [batch, L, h1] -> [batch, 1, L, h1]
            asp_lst.append(torch.unsqueeze(batch_asp, 1))

        # [batch, asp, L, h1]
        basketAsp = torch.cat(asp_lst, dim=1)

        return basketAsp

    def PredictionLayer(self, uuAsp, iiAsp, his):
        intent = []
        loss_cl = 0
        # Over loop each aspect
        for b in range(uuAsp.shape[1]):
            uInterest = torch.tanh(uuAsp[:, b, :])  # [batch, h2]
            iInterest = torch.tanh(iiAsp[:, b, :])  # [batch, h2]

            uLoss = self.cl_loss(uInterest, iInterest)  # [batch, h2]
            iLoss = self.cl_loss(iInterest, uInterest)  # [batch, h2]
            cLoss = uLoss + iLoss

            Interest = torch.cat(
                [uInterest.unsqueeze(2), iInterest.unsqueeze(2)], dim=2
            )  # [batch,h2,2]
            Interests = torch.sum(Interest, dim=2)  # [batch,h2]
            scores_trans = self.out(Interests)  # [batch,h2] -> [batch,n_items]
            scores_trans = F.softmax(scores_trans, dim=-1)  # [batch, n_items]

            hisEmb = self.his_linear_embds(his)  # [batch,n_items] -> [batch,h2]

            # [h1 -> 1]
            gate = torch.sigmoid(
                self.gate_his(hisEmb) + self.gate_his(Interests)
            )  # value

            res = gate * scores_trans + (1 - gate) * his  # [batch, n_items]
            res = res / math.sqrt(self.emb_dim)

            intent.append(res.unsqueeze(2))
            loss_cl += cLoss.mean()

        results = torch.cat(intent, dim=2)  # [batch, n_items, asp]
        result = F.max_pool1d(results, int(results.size(2))).squeeze(
            2
        )  # [batch, n_items]
        loss_cl = loss_cl / self.n_aspects

        return result, loss_cl

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def cl_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        tau = 0.6
        f = lambda x: torch.exp(x / tau)

        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())
        )


def transform_data(
    batch_users,
    batch_basket_items,
    user_history_matrix,
    item_history_matrix,
    total_items,
    decay,
    device,
    is_test=False,
):
    padding_idx = total_items
    if is_test:
        batch_history_items = [
            [np.unique(basket).tolist() for basket in basket_items]
            for basket_items in batch_basket_items
        ]
        batch_targets = None
    else:
        batch_history_items = [
            [np.unique(basket).tolist() for basket in basket_items[:-1]]
            for basket_items in batch_basket_items
        ]
        batch_targets = np.zeros((len(batch_basket_items), total_items), dtype="uint8")
        for inc, basket_items in enumerate(batch_basket_items):
            batch_targets[inc, basket_items[-1]] = 1
        batch_targets = torch.tensor(batch_targets, dtype=torch.uint8, device=device)

    batch_lengths = [
        [len(basket) for basket in history_items]
        for history_items in batch_history_items
    ]

    max_sequence_size = max([len(lengths) for lengths in batch_lengths])
    max_basket_size = max([max(lengths) for lengths in batch_lengths])
    padded_samples = []
    padded_decays = []
    for history_items in batch_history_items:
        padded_samples.append(
            [
                basket + [padding_idx] * (max_basket_size - len(basket))
                for basket in history_items
            ]
            + [[padding_idx] * max_basket_size]
            * (max_sequence_size - len(history_items))
        )
        padded_decays.append(
            [
                decay ** (len(history_items) - 1 - inc)
                for inc, _ in enumerate(history_items)
            ]
            + [0] * (max_sequence_size - len(history_items))
        )
    padded_samples = (
        torch.from_numpy(np.asarray(padded_samples, dtype=np.int32))
        .type(torch.LongTensor)
        .to(device)
    )
    padded_decays = (
        torch.from_numpy(
            np.asarray(padded_decays, dtype=np.float32).reshape(
                len(batch_history_items), -1, 1
            )
        )
        .type(torch.FloatTensor)
        .to(device)
    )
    userhis = (
        torch.from_numpy(user_history_matrix[batch_users].todense())
        .type(torch.FloatTensor)
        .to(device)
    )
    itemhis = (
        torch.from_numpy(item_history_matrix[batch_users].todense())
        .type(torch.FloatTensor)
        .to(device)
    )
    return padded_samples, padded_decays, userhis, itemhis, batch_targets


def build_history_matrix(
    train_set,
    val_set,
    test_set,
    total_users,
    total_items,
    mode="train",
):
    counter = Counter()
    for [user], _, [basket_items] in train_set.ubi_iter(1, shuffle=False):
        if mode == "train":
            user_items = chain.from_iterable(basket_items[:-1])
        else:
            user_items = chain.from_iterable(basket_items)
        counter.update((user, item) for item in user_items)
    if val_set is not None and mode != "train":
        for [user], _, [basket_items] in val_set.ubi_iter(1, shuffle=False):
            if mode == "validation":
                user_items = chain.from_iterable(basket_items[:-1])
            else:
                user_items = chain.from_iterable(basket_items)
            counter.update((user, item) for item in user_items)
    if test_set is not None and mode == "test":
        for [user], _, [basket_items] in test_set.ubi_iter(1, shuffle=False):
            user_items = chain.from_iterable(basket_items[:-1])
            counter.update((user, item) for item in user_items)
    users = []
    items = []
    counts = []
    for (user, item), count in counter.items():
        users.append(user)
        items.append(item)
        counts.append(count)
    users = np.asarray(users, dtype=np.int32)
    items = np.asarray(items, dtype=np.int32)
    scores = np.asarray(counts, dtype=np.float32)
    history_matrix = csr_matrix(
        (scores, (users, items)), shape=(total_users, total_items)
    )
    user_history_matrix = normalize(history_matrix, norm="l1", axis=1)
    item_history_matrix = normalize(history_matrix, norm="l1", axis=0)
    return user_history_matrix, item_history_matrix


def learn(
    model,
    train_set,
    total_users,
    total_items,
    val_set,
    n_epochs,
    batch_size,
    lr,
    l2,
    decay,
    m,
    n,
    optimizer,
    device,
    verbose=False,
):
    model.to(device)

    optimizer = OPTIMIZER_DICT[optimizer](
        params=model.parameters(),
        lr=lr,
        weight_decay=l2,
    )
    train_user_history_matrix, train_item_history_matrix = build_history_matrix(
        train_set=train_set,
        val_set=val_set,
        test_set=None,
        total_users=total_users,
        total_items=total_items,
        mode="train",
    )
    val_user_history_matrix, val_item_history_matrix = build_history_matrix(
        train_set=train_set,
        val_set=val_set,
        test_set=None,
        total_users=total_users,
        total_items=total_items,
        mode="validation",
    )
    progress_bar = trange(1, n_epochs + 1, disable=not verbose)
    last_val_loss = np.inf
    last_loss = np.inf
    for _ in progress_bar:
        model.train()
        total_loss = 0.0
        cnt = 0
        for inc, (u_batch, _, bi_batch) in enumerate(
            train_set.ubi_iter(batch_size, shuffle=True)
        ):
            (samples, decays, userhis, itemhis, target) = transform_data(
                u_batch,
                bi_batch,
                total_items=total_items,
                user_history_matrix=train_user_history_matrix,
                item_history_matrix=train_item_history_matrix,
                decay=decay,
                device=device,
            )
            scores, loss_cl = model(samples, decays, userhis, itemhis, device)
            loss_ce = (
                -(
                    m * target * torch.log(scores)
                    + n * (1 - target) * torch.log(1 - scores)
                )
                .sum(-1)
                .mean()
            )
            loss = loss_ce + loss_cl
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cnt += len(bi_batch)
            last_loss = total_loss / cnt
            if inc % 10 == 0:
                progress_bar.set_postfix(loss=last_loss, val_loss=last_val_loss)

        if val_set is not None:
            model.eval()
            total_val_loss = 0.0
            cnt = 0
            for inc, (u_batch, _, bi_batch) in enumerate(
                val_set.ubi_iter(batch_size, shuffle=False)
            ):
                (samples, decays, userhis, itemhis, target) = transform_data(
                    u_batch,
                    bi_batch,
                    total_items=total_items,
                    user_history_matrix=val_user_history_matrix,
                    item_history_matrix=val_item_history_matrix,
                    decay=decay,
                    device=device,
                )
                scores, loss_cl = model(samples, decays, userhis, itemhis, device)
                loss_ce = (
                    -(
                        m * target * torch.log(scores)
                        + n * (1 - target) * torch.log(1 - scores)
                    )
                    .sum(-1)
                    .mean()
                )
                loss = loss_ce + loss_cl
                total_val_loss += loss.item()
                cnt += len(bi_batch)
                last_val_loss = total_val_loss / cnt
                if inc % 10 == 0:
                    progress_bar.set_postfix(loss=last_loss, val_loss=last_val_loss)


def score(
    model,
    user_history_matrix,
    item_history_matrix,
    total_items,
    user_idx,
    history_baskets,
    decay,
    device,
):
    model.eval()
    (samples, decays, userhis, itemhis, _) = transform_data(
        [user_idx],
        [history_baskets],
        total_items=total_items,
        user_history_matrix=user_history_matrix,
        item_history_matrix=item_history_matrix,
        decay=decay,
        device=device,
        is_test=True,
    )
    scores, _ = model(samples, decays, userhis, itemhis, device)
    return scores.cpu().detach().numpy().squeeze()
