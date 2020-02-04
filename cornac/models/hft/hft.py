# Copyright 2018 The Cornac Authors. All Rights Reserved.
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
import scipy.optimize as opt


class Model:
    def __init__(
        self,
        n_user,
        n_item,
        n_vocab,
        alpha,
        beta_u,
        beta_i,
        gamma_u,
        gamma_i,
        k=10,
        lambda_text=0.1,
        l2_reg=0.001,
        max_iter=50,
        grad_iter=50,
    ):
        self.k = k
        self.lambda_text = lambda_text
        self.l2_reg = l2_reg
        self.grad_iter = grad_iter
        self.max_iter = max_iter
        self.n_item = n_item
        self.n_user = n_user
        self.n_vocab = n_vocab

        # Model parameters
        self.alpha = alpha
        self.beta_u = beta_u
        self.beta_i = beta_i
        self.gamma_u = gamma_u
        self.gamma_i = gamma_i

        self._init_params()

    def _init_params(self):
        params_length = np.array(
            [
                1,
                1,
                self.n_user,
                self.n_item,
                self.n_user * self.k,
                self.n_item * self.k,
                self.n_vocab * self.k,
            ]
        )
        self.params_idx = params_length.cumsum()

        self.params = np.zeros(params_length.sum())
        self.params[0] = self.alpha
        self.params[1] = 1.0  # kappa init
        self.params[self.params_idx[3] : self.params_idx[4]] = self.gamma_u.ravel()
        self.params[self.params_idx[4] : self.params_idx[5]] = self.gamma_i.ravel()

    def init_count(self, docs):
        # Counter
        self.item_topic_cnt = np.zeros(
            shape=(self.n_item, self.k), dtype=int
        )  # given item, count number of time each topic occur

        self.word_topic_cnt = np.zeros(
            shape=(self.n_vocab, self.k), dtype=int
        )  # given word, count number of time each topic occur

        self.item_word = np.zeros(
            shape=(self.n_item, 1), dtype=int
        )  # number of word for each item
        self.topic_cnt = np.zeros(
            shape=(1, self.k), dtype=int
        )  # number of time topic occur
        self.total_word = 0  # total number word in corpus
        self.background_word = np.zeros(
            shape=(self.n_vocab, 1), dtype=float
        )  # background weights for each word [n_vocab, 1]

        self.topic_assignment = list()

        for di in range(len(docs)):
            doc = docs[di]
            doc_len = len(doc)
            topics = np.random.randint(self.k, size=doc_len)
            self.topic_assignment.append(topics)
            self.item_word[di] = doc_len
            self.total_word += doc_len
            for wi in range(doc_len):
                topic = topics[wi]
                word = doc[wi]
                self.word_topic_cnt[word, topic] += 1
                self.item_topic_cnt[di, topic] += 1
                self.background_word[word] += 1
                self.topic_cnt[0, topic] += 1

        self.background_word /= self.total_word

    @staticmethod
    def _sampling_from_dist(prob):
        thr = prob.sum() * np.random.rand()
        new_topic = 0
        tmp = prob[new_topic]
        while tmp < thr:
            new_topic += 1
            tmp += prob[new_topic]
        return new_topic

    def assign_word_topics(self, docs):
        _, self.kappa, _, _, _, self.gamma_i, self.topic_word = self._get_view(
            self.params
        )

        for di in range(len(docs)):
            doc = docs[di]
            doc_len = len(doc)
            topics = self.topic_assignment[di]

            for wi in range(doc_len):
                old_topic = topics[wi]
                word = doc[wi]
                topic_score = np.exp(
                    self.kappa * self.gamma_i[di]
                    + self.background_word[word]
                    + self.topic_word[word]
                )
                topic_score = topic_score / np.sum(topic_score)
                new_topic = self._sampling_from_dist(topic_score)
                if new_topic != old_topic:
                    self.word_topic_cnt[word, old_topic] -= 1
                    self.word_topic_cnt[word, new_topic] += 1
                    self.topic_cnt[0, old_topic] -= 1
                    self.topic_cnt[0, new_topic] += 1
                    self.item_topic_cnt[di, old_topic] -= 1
                    self.item_topic_cnt[di, new_topic] += 1
                    self.topic_assignment[di][wi] = new_topic

        average = self.topic_word.sum(axis=1)[:, np.newaxis] / self.k
        self.topic_word -= average
        self.background_word += average

    def update_params(self, rating_data):
        res = opt.fmin_l_bfgs_b(
            self._func, x0=self.params, args=rating_data, maxiter=self.grad_iter
        )
        self.params = res[0]
        return res[1]

    def get_parameter(self):
        alpha, _, beta_u, beta_i, gamma_u, gamma_i, _ = self._get_view(self.params)
        return alpha.item(), beta_u, beta_i, gamma_u, gamma_i

    def _get_view(self, params):
        idx = self.params_idx

        alpha = params[0 : idx[0],]
        kappa = params[idx[0] : idx[1],]
        beta_u = params[idx[1] : idx[2],]
        beta_i = params[idx[2] : idx[3],]
        gamma_u = params[idx[3] : idx[4],].reshape(self.n_user, self.k)
        gamma_i = params[idx[4] : idx[5],].reshape(self.n_item, self.k)
        topic_word = params[idx[5] :,].reshape(self.n_vocab, self.k)

        return alpha, kappa, beta_u, beta_i, gamma_u, gamma_i, topic_word

    def _func(self, X, *args):
        user_data = args[0]
        item_data = args[1]
        R_user = user_data[1]
        R_item = item_data[1]
        grad = np.zeros_like(X)
        alpha, kappa, beta_u, beta_i, gamma_u, gamma_i, topic_word = self._get_view(
            params=X
        )
        dalpha, dkappa, dbeta_u, dbeta_i, dgamma_u, dgamma_i, dtopic_word = self._get_view(
            params=grad
        )
        cf_loss = 0.0
        reg_loss = 0.0
        corpus_likelihood = 0.0

        for i in range(self.n_user):
            idx_item = user_data[0][i]
            if not len(idx_item):  # user without any rating
                continue
            gamma_items = gamma_i[idx_item]
            R_i = R_user[i]
            pred = alpha + beta_u[i] + beta_i[idx_item] + gamma_items.dot(gamma_u[i])
            err = (pred - R_i).reshape(-1, 1)
            cf_loss += np.sum(err ** 2)

            total_err = np.sum(err)
            dalpha += 2 * total_err
            dbeta_u[i] += 2 * total_err
            dgamma_u[i] += 2 * np.sum(err * gamma_items, axis=0)

        for j in range(self.n_item):
            idx_user = item_data[0][j]
            if not len(idx_user):  # item without any rating
                continue
            gamma_users = gamma_u[idx_user]
            R_j = R_item[j]
            pred = alpha + beta_u[idx_user] + beta_i[j] + gamma_users.dot(gamma_i[j])
            err = (pred - R_j).reshape(-1, 1)
            total_err = np.sum(err)
            dbeta_i[j] += 2 * total_err
            dgamma_i[j] += 2 * np.sum(err * gamma_users, axis=0)

        if self.l2_reg > 0:
            reg_loss += self.l2_reg * np.sum(gamma_u ** 2)
            dgamma_u += 2 * self.l2_reg * gamma_u
            reg_loss += self.l2_reg * np.sum(gamma_i ** 2)
            dgamma_i += 2 * self.l2_reg * gamma_i

        e_theta = np.exp(self.kappa * self.gamma_i)
        t_z = e_theta.sum(axis=1, keepdims=True)
        corpus_likelihood += self.lambda_text * np.sum(
            self.item_topic_cnt * (self.kappa * self.gamma_i - np.log(t_z))
        )

        e_phi = np.exp(self.background_word + topic_word)
        word_z = e_phi.sum(axis=0, keepdims=True)
        corpus_likelihood += self.lambda_text * np.sum(
            self.word_topic_cnt * (self.background_word + topic_word - np.log(word_z))
        )

        q = -self.lambda_text * (self.item_topic_cnt - self.item_word * e_theta / t_z)
        dgamma_i += kappa * q
        dkappa += np.sum(gamma_i * q)
        dtopic_word += -self.lambda_text * (
            self.word_topic_cnt - self.topic_cnt * e_phi / word_z
        )

        loss = cf_loss + reg_loss - corpus_likelihood

        return loss, grad
