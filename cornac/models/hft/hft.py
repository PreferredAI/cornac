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
import scipy.optimize


class Model:

    def __init__(self, params, n_user, n_item, n_vocab, k=10,
                 lambda_reg=0.01, latent_reg=0.1, grad_iter=50, max_iter=50):

        self.k = k
        self.lambda_reg = lambda_reg
        self.latent_reg = latent_reg
        self.grad_iter = grad_iter
        self.max_iter = max_iter
        self.n_item = n_item
        self.n_user = n_user
        self.n_vocab = n_vocab
        self.params = params
        self.alpha, self.kappa, self.beta_u, self.beta_i, \
        self.gamma_u, self.gamma_i, self.topic_word = self._get_view(params)  # Parameter views
        # Counter
        self.item_topic_cnt = np.zeros(shape=(self.n_item, self.k),
                                       dtype=int)  # given item, count number of time each topic occur
        self.word_topic_cnt = np.zeros(shape=(self.n_vocab, self.k),
                                       dtype=int)  # given word, count number of time each topic occur
        self.item_word = np.zeros(shape=(self.n_item, 1), dtype=int)  # number of word for each item
        self.topic_cnt = np.zeros(shape=(self.k, 1), dtype=int)  # number of time topic occur
        self.total_word = 0  # total number word in corpus
        self.background_word = np.zeros(shape=(self.n_vocab, 1),
                                        dtype=float)  # background weights for each word [n_vocab, 1]
        self.topic_assignment = list()

    def random_int_topic(self, docs):

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
                self.topic_cnt[topic] += 1

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
        for di in range(len(docs)):
            doc = docs[di]
            doc_len = len(doc)
            topics = self.topic_assignment[di]

            for wi in range(doc_len):
                topic = topics[wi]
                word = doc[wi]
                topic_score = np.exp(
                    self.kappa.item() * self.gamma_i[di] + self.background_word[word].item() + self.topic_word[word])
                topic_score = topic_score / np.sum(topic_score)
                new_topic = self._sampling_from_dist(topic_score)
                if new_topic != topic:
                    self.word_topic_cnt[word, topic] -= 1
                    self.word_topic_cnt[word, new_topic] += 1
                    self.topic_cnt[topic] -= 1
                    self.topic_cnt[new_topic] += 1
                    self.item_topic_cnt[di, topic] -= 1
                    self.item_topic_cnt[di, new_topic] += 1
                    self.topic_assignment[di][wi] = new_topic

        average = self.topic_word.sum(axis=1)[:, np.newaxis] / self.k
        self.topic_word -= average
        self.background_word += average

    def update_params(self, rating_data):

        res = scipy.optimize.fmin_l_bfgs_b(self._func, x0=self.params, disp=True,
                                           args=rating_data, maxiter=self.grad_iter)
        self.params = res[0]
        self.alpha, self.kappa, self.beta_u, self.beta_i, \
        self.gamma_u, self.gamma_i, self.topic_word = self._get_view(self.params)

        return res[1]

    def get_parameter(self):

        return self.params

    def _get_view(self, params):

        params_length = np.array(
            [1, 1, self.n_user, self.n_item, self.n_user * self.k, self.n_item * self.k, self.n_vocab * self.k])
        idx = params_length.cumsum()

        alpha = params[0:idx[0], ]
        kappa = params[idx[0]:idx[1], ]
        beta_u = params[idx[1]:idx[2], ]
        beta_i = params[idx[2]:idx[3], ]
        gamma_u = params[idx[3]:idx[4], ].reshape(self.n_user, self.k)
        gamma_i = params[idx[4]:idx[5], ].reshape(self.n_item, self.k)
        topic_word = params[idx[5]:idx[6], ].reshape(self.n_vocab, self.k)

        return alpha, kappa, beta_u, beta_i, gamma_u, gamma_i, topic_word

    def _func(self, X, *args):
        user_data = args[0]
        item_data = args[1]
        R_user = user_data[1]
        R_item = item_data[1]
        alpha, kappa, beta_u, beta_i, gamma_u, gamma_i, topic_word = self._get_view(params=X)
        grad = np.zeros_like(X)
        dalpha, dkappa, dbeta_u, dbeta_i, dgamma_u, dgamma_i, dtopic_word = self._get_view(params=grad)

        cf_loss = 0.0
        reg_loss = 0.0
        corpus_likelihood = 0.0

        predictions = [{} for _ in range(self.n_item)]

        for i in range(self.n_user):
            idx_item = user_data[0][i]
            if not len(idx_item):  # user without any rating
                continue
            gamma_items = gamma_i[idx_item]
            R_i = R_user[i]
            pred = alpha + beta_u[i] + beta_i[idx_item] + np.sum(gamma_u[i] * gamma_items, axis=1)
            for idx, pd in zip(idx_item, pred):
                predictions[idx][i] = pd
            err = (pred - R_i).reshape(-1, 1)
            cf_loss += 0.5 * np.sum(err ** 2)

            total_err = np.sum(err)
            dalpha += total_err
            dbeta_u[i] += total_err
            dgamma_u[i] += np.sum(err * gamma_items, axis=0)

        for j in range(self.n_item):
            idx_user = item_data[0][j]
            if not len(idx_user):  # item without any rating
                continue
            gamma_users = gamma_u[idx_user]
            R_j = R_item[j]
            pred = np.zeros(len(idx_user))
            for i, idx in enumerate(idx_user):
                pred[i] = predictions[j][idx]

            err = (pred - R_j).reshape(-1, 1)
            total_err = np.sum(err)
            dbeta_i[j] += total_err
            dgamma_i[j] += np.sum(err * gamma_users, axis=0)

        if self.latent_reg > 0:
            reg_loss += self.latent_reg * np.linalg.norm(gamma_u)
            dgamma_u += self.latent_reg * gamma_u
            reg_loss += self.latent_reg * np.linalg.norm(gamma_i)
            dgamma_i += self.latent_reg * gamma_i

        e_theta = np.exp(self.kappa * self.gamma_i)
        t_z = e_theta.sum(axis=1, keepdims=True)
        corpus_likelihood += self.lambda_reg * np.sum(self.item_topic_cnt * (self.kappa * self.gamma_i - np.log(t_z)))

        e_phi = np.exp(self.background_word + topic_word)
        word_z = e_phi.sum(axis=0, keepdims=True)
        corpus_likelihood += self.lambda_reg * np.sum(self.word_topic_cnt *
                                                      (self.background_word + topic_word - np.log(word_z)))
        q = - self.lambda_reg * (self.item_topic_cnt - self.item_word * e_theta / t_z)
        dgamma_i += kappa * q
        dkappa += np.sum(gamma_i * q)
        dtopic_word += -self.lambda_reg * (
                self.word_topic_cnt - self.topic_cnt.reshape(1, -1) * e_phi / word_z)

        loss = cf_loss + reg_loss - corpus_likelihood

        return loss, grad
