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

import time
import multiprocessing as mp
import ctypes as c

import numpy as np
from autograd import grad


EPS = 1e-9


def get_value(G, U, I, F, key):
    tensor_value1 = np.einsum("abc,a->bc", G, U[key[0]])
    tensor_value2 = np.einsum("bc,b->c", tensor_value1, I[key[1]])
    return np.einsum("c,c-> ", tensor_value2, F[key[2]])


def sign(a, b):
    return 1 if a > b else -1


def grad_worker_mse(
    user_item_aspect,
    user_aspect_opinion,
    item_aspect_opinion,
    G1,
    G2,
    G3,
    U,
    I,
    A,
    O,
    error_square,
    error_bpr,
    lock,
    q_samples_mse,
    del_g1,
    del_g2,
    del_g3,
    del_u,
    del_i,
    del_a,
    del_o,
    num_grad,
):
    while 1:
        if not q_samples_mse.empty():
            sample = q_samples_mse.get()
            if not sample:
                break

            uia_samples, uao_samples, iao_samples = sample

            for [user_idx, item_idx, aspect_idx] in uia_samples:
                key = (user_idx, item_idx, aspect_idx)
                pred_rating = get_value(G1, U, I, A, key)
                score = user_item_aspect[key]
                del_sqerror = 2 * (pred_rating - score)
                lock.acquire()
                error_square.value += (pred_rating - score) ** 2
                del_g1 += del_sqerror * np.einsum(
                    "ab,c->abc",
                    np.einsum("a,b->ab", U[user_idx], I[item_idx]),
                    A[aspect_idx],
                )
                del_u[user_idx] += del_sqerror * np.einsum(
                    "ac,c->a", np.einsum("abc,b->ac", G1, I[item_idx]), A[aspect_idx]
                )
                del_i[item_idx] += del_sqerror * np.einsum(
                    "bc,c->b", np.einsum("abc,a->bc", G1, U[user_idx]), A[aspect_idx]
                )
                del_a[aspect_idx] += del_sqerror * np.einsum(
                    "bc,b->c", np.einsum("abc,a->bc", G1, U[user_idx]), I[item_idx]
                )
                lock.release()

            for [user_idx, aspect_idx, opinion_idx] in uao_samples:
                key = (user_idx, aspect_idx, opinion_idx)
                pred_rating = get_value(G2, U, A, O, key)
                score = user_aspect_opinion[key]
                del_sqerror = 2 * (pred_rating - score)
                lock.acquire()
                error_square.value += (pred_rating - score) ** 2
                del_g2 += del_sqerror * np.einsum(
                    "ab,c->abc",
                    np.einsum("a,b->ab", U[user_idx], A[aspect_idx]),
                    O[opinion_idx],
                )
                del_u[user_idx] += del_sqerror * np.einsum(
                    "ac,c->a", np.einsum("abc,b->ac", G2, A[aspect_idx]), O[opinion_idx]
                )
                del_a[aspect_idx] += del_sqerror * np.einsum(
                    "bc,c->b", np.einsum("abc,a->bc", G2, U[user_idx]), O[opinion_idx]
                )
                del_o[opinion_idx] += del_sqerror * np.einsum(
                    "bc,b->c", np.einsum("abc,a->bc", G2, U[user_idx]), A[aspect_idx]
                )
                lock.release()

            for [item_idx, aspect_idx, opinion_idx] in iao_samples:
                key = (item_idx, aspect_idx, opinion_idx)
                pred_rating = get_value(G3, I, A, O, key)
                score = item_aspect_opinion[key]
                del_sqerror = 2 * (pred_rating - score)
                lock.acquire()
                error_square.value += (pred_rating - score) ** 2
                del_g3 += del_sqerror * np.einsum(
                    "ab,c->abc",
                    np.einsum("a,b->ab", I[item_idx], A[aspect_idx]),
                    O[opinion_idx],
                )
                del_i[item_idx] += del_sqerror * np.einsum(
                    "ac,c->a", np.einsum("abc,b->ac", G3, A[aspect_idx]), O[opinion_idx]
                )
                del_a[aspect_idx] += del_sqerror * np.einsum(
                    "bc,c->b", np.einsum("abc,a->bc", G3, I[item_idx]), O[opinion_idx]
                )
                del_o[opinion_idx] += del_sqerror * np.einsum(
                    "bc,b->c", np.einsum("abc,a->bc", G3, I[item_idx]), A[aspect_idx]
                )
                lock.release()

            lock.acquire()
            num_grad.value += 1
            lock.release()


def grad_worker_bpr(
    rating_matrix,
    lambda_bpr,
    G1,
    U,
    I,
    A,
    error_square,
    error_bpr,
    lock,
    q_samples_bpr,
    del_g1,
    del_u,
    del_i,
    del_a,
    num_grad,
):
    while 1:
        if not q_samples_bpr.empty():
            sample = q_samples_bpr.get()
            if not sample:
                break

            bpr_sample_ele, item2_sample = sample

            for i, [user_idx, item_idx1] in enumerate(bpr_sample_ele):
                item_idx2 = item2_sample[i]
                user_item_vector = rating_matrix[user_idx, :].A.flatten()

                if user_item_vector[item_idx1] != user_item_vector[item_idx2]:
                    pred_x_ij = (
                        get_value(G1, U, I, A, (user_idx, item_idx1, -1))
                        - get_value(G1, U, I, A, (user_idx, item_idx2, -1))
                    ) * sign(user_item_vector[item_idx1], user_item_vector[item_idx2])
                    del_bpr = (
                        lambda_bpr
                        * (np.exp(-pred_x_ij) / (1 + np.exp(-pred_x_ij)))
                        * sign(user_item_vector[item_idx1], user_item_vector[item_idx2])
                    )

                    lock.acquire()
                    error_bpr.value += np.log(1 / (1 + np.exp(-pred_x_ij)))
                    item_diff = I[item_idx1] - I[item_idx2]
                    del_g1 -= del_bpr * np.einsum(
                        "ab,c->abc", np.einsum("a,b->ab", U[user_idx], item_diff), A[-1]
                    )
                    del_u[user_idx] -= del_bpr * np.einsum(
                        "ac,c->a", np.einsum("abc,b->ac", G1, item_diff), A[-1]
                    )
                    del_i[item_idx1] -= del_bpr * np.einsum(
                        "bc,c->b", np.einsum("abc,a->bc", G1, U[user_idx]), A[-1]
                    )
                    del_i[item_idx2] += del_bpr * np.einsum(
                        "bc,c->b", np.einsum("abc,a->bc", G1, U[user_idx]), A[-1]
                    )
                    del_a[-1] -= del_bpr * np.einsum(
                        "bc,b->c", np.einsum("abc,a->bc", G1, U[user_idx]), item_diff
                    )
                    lock.release()

            lock.acquire()
            num_grad.value += 1
            lock.release()


def paraserver(
    user_item_pairs,
    user_item_aspect,
    user_aspect_opinion,
    item_aspect_opinion,
    n_element_samples,
    n_bpr_samples,
    lambda_reg,
    n_epochs,
    lr,
    G1,
    G2,
    G3,
    U,
    I,
    A,
    O,
    error_square,
    error_bpr,
    q_samples_mse,
    q_samples_bpr,
    del_g1,
    del_g2,
    del_g3,
    del_u,
    del_i,
    del_a,
    del_o,
    num_grad,
    n_threads,
    seed=None,
    verbose=False,
):

    from ...utils import get_rng

    rng = get_rng(seed)

    sum_square_gradients_G1 = np.zeros_like(G1)
    sum_square_gradients_G2 = np.zeros_like(G2)
    sum_square_gradients_G3 = np.zeros_like(G3)
    sum_square_gradients_U = np.zeros_like(U)
    sum_square_gradients_I = np.zeros_like(I)
    sum_square_gradients_A = np.zeros_like(A)
    sum_square_gradients_O = np.zeros_like(O)

    mse_per_proc = int(n_element_samples / n_threads)
    bpr_per_proc = int(n_bpr_samples / n_threads)

    user_item_aspect_keys = np.array(list(user_item_aspect.keys()))
    user_aspect_opinion_keys = np.array(list(user_aspect_opinion.keys()))
    item_aspect_opinion_keys = np.array(list(item_aspect_opinion.keys()))
    user_item_pairs_keys = np.array(user_item_pairs)
    for epoch in range(n_epochs):
        start_time = time.time()
        if verbose:
            print("iteration:", epoch + 1, "/", n_epochs)

        error_square.value = 0
        error_bpr.value = 0
        uia_samples = user_item_aspect_keys[
            rng.choice(len(user_item_aspect_keys), size=n_element_samples)
        ]
        uao_samples = user_aspect_opinion_keys[
            rng.choice(len(user_aspect_opinion_keys), size=n_element_samples)
        ]
        iao_samples = item_aspect_opinion_keys[
            rng.choice(len(item_aspect_opinion_keys), size=n_element_samples)
        ]
        bpr_sample_ele = user_item_pairs_keys[
            rng.choice(len(user_item_pairs_keys), size=n_bpr_samples)
        ]
        item2_sample = rng.choice(range(0, I.shape[0]), size=n_bpr_samples)

        num_grad.value = 0
        del_g1[:] = 0
        del_g2[:] = 0
        del_g3[:] = 0
        del_u[:] = 0
        del_i[:] = 0
        del_a[:] = 0
        del_o[:] = 0

        for i in range(n_threads):
            q_samples_mse.put(
                (
                    uia_samples[mse_per_proc * i : mse_per_proc * (i + 1)],
                    uao_samples[mse_per_proc * i : mse_per_proc * (i + 1)],
                    iao_samples[mse_per_proc * i : mse_per_proc * (i + 1)],
                )
            )
            q_samples_bpr.put(
                (
                    bpr_sample_ele[bpr_per_proc * i : bpr_per_proc * (i + 1)],
                    item2_sample[bpr_per_proc * i : bpr_per_proc * (i + 1)],
                )
            )

        while 1:
            if num_grad.value == 2 * n_threads:
                break

        del_g1_reg = del_g1 + lambda_reg * G1 * (del_g1 != 0)
        del_g2_reg = del_g2 + lambda_reg * G2 * (del_g2 != 0)
        del_g3_reg = del_g3 + lambda_reg * G3 * (del_g3 != 0)
        del_u_reg = del_u + lambda_reg * U * (del_u != 0)
        del_i_reg = del_i + lambda_reg * I * (del_i != 0)
        del_a_reg = del_a + lambda_reg * A * (del_a != 0)
        del_o_reg = del_o + lambda_reg * O * (del_o != 0)

        sum_square_gradients_G1 += EPS + np.square(del_g1_reg)
        sum_square_gradients_G2 += EPS + np.square(del_g2_reg)
        sum_square_gradients_G3 += EPS + np.square(del_g3_reg)
        sum_square_gradients_U += EPS + np.square(del_u_reg)
        sum_square_gradients_I += EPS + np.square(del_i_reg)
        sum_square_gradients_A += EPS + np.square(del_a_reg)
        sum_square_gradients_O += EPS + np.square(del_o_reg)

        lr_g1 = np.divide(lr, np.sqrt(sum_square_gradients_G1))
        lr_g2 = np.divide(lr, np.sqrt(sum_square_gradients_G2))
        lr_g3 = np.divide(lr, np.sqrt(sum_square_gradients_G3))
        lr_u = np.divide(lr, np.sqrt(sum_square_gradients_U))
        lr_i = np.divide(lr, np.sqrt(sum_square_gradients_I))
        lr_a = np.divide(lr, np.sqrt(sum_square_gradients_A))
        lr_o = np.divide(lr, np.sqrt(sum_square_gradients_O))

        G1 -= lr_g1 * del_g1_reg
        G2 -= lr_g2 * del_g2_reg
        G3 -= lr_g3 * del_g3_reg
        U -= lr_u * del_u_reg
        I -= lr_i * del_i_reg
        A -= lr_a * del_a_reg
        O -= lr_o * del_o_reg

        # Projection to non-negative space
        G1[G1 < 0] = 0
        G2[G2 < 0] = 0
        G3[G3 < 0] = 0
        U[U < 0] = 0
        I[I < 0] = 0
        A[A < 0] = 0
        O[O < 0] = 0

        if verbose:
            if n_element_samples:
                print("RMSE:", np.sqrt(error_square.value / 3 / n_element_samples))
            if n_bpr_samples:
                print("BPR:", error_bpr.value / n_bpr_samples)

            timeleft = (time.time() - start_time) * (n_epochs - epoch - 1)

            if (timeleft / 60) > 60:
                print(
                    "time left: "
                    + str(int(timeleft / 3600))
                    + " hr "
                    + str(int(timeleft / 60 % 60))
                    + " min "
                    + str(int(timeleft % 60))
                    + " s"
                )
            else:
                print(
                    "time left: "
                    + str(int(timeleft / 60))
                    + " min "
                    + str(int(timeleft % 60))
                    + " s"
                )

    for _ in range(n_threads):
        q_samples_bpr.put(0)
        q_samples_mse.put(0)
