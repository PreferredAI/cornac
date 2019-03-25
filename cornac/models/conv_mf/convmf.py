"""
reference https://github.com/cartopy/ConvMF/blob/master/models.py
"""
import time
import math
import numpy as np
from .cnn_module.models import CNN_module

def convmf(train_user, train_item, document,
            give_item_weight=True,
            max_iter=50, lambda_u=1, lambda_v=100,
            init_W=None,dimension=50, vocab_size=8000,
            dropout_rate=0.2, emb_dim=200, max_len=300, num_kernel_per_ws=100):
    
    # explicit setting
    a = 1
    b = 0

    num_user = len(train_user[0])
    num_item = len(train_item[0])
    endure = 3

    PREV_LOSS = 1e-50
    
    Train_R_I = train_user[1]
    Train_R_J = train_item[1]
    
    if give_item_weight is True:
        item_weight = np.array([math.sqrt(len(i))
                                for i in Train_R_J], dtype=float)
        item_weight = (float(num_item) / item_weight.sum()) * item_weight
    else:
        item_weight = np.ones(num_item, dtype=float)

    cnn_module = CNN_module(output_dimesion=dimension, vocab_size=vocab_size, dropout_rate=dropout_rate,
                            emb_dim=emb_dim, max_len=max_len, nb_filters=num_kernel_per_ws, init_W=init_W)

    theta = cnn_module.get_projection_layer(document)

    U = np.random.uniform(size=(num_user, dimension))
    V = theta
    
    for iteration in range(max_iter):
        loss = 0
        tic = time.time()
        print("Iteration {}".format(iteration))

        VV = b * (V.T.dot(V)) + lambda_u * np.eye(dimension)
        sub_loss = np.zeros(num_user)

        for i in range(num_user):
            idx_item = train_user[0][i]
            V_i = V[idx_item]
            R_i = Train_R_I[i]
            A = VV + (a - b) * (V_i.T.dot(V_i))
            B = (a * V_i * (np.tile(R_i, (dimension, 1)).T)).sum(0)

            U[i] = np.linalg.solve(A, B)

            sub_loss[i] = -0.5 * lambda_u * np.dot(U[i], U[i])

        loss = loss + np.sum(sub_loss)

        sub_loss = np.zeros(num_item)
        UU = b * (U.T.dot(U))
        for j in range(num_item):
            idx_user = train_item[0][j]
            U_j = U[idx_user]
            R_j = Train_R_J[j]

            tmp_A = UU + (a - b) * (U_j.T.dot(U_j))
            A = tmp_A + lambda_v * item_weight[j] * np.eye(dimension)
            B = (a * U_j * (np.tile(R_j, (dimension, 1)).T)
                 ).sum(0) + lambda_v * item_weight[j] * theta[j]
            V[j] = np.linalg.solve(A, B)

            sub_loss[j] = -0.5 * np.square(R_j * a).sum()
            sub_loss[j] = sub_loss[j] + a * np.sum((U_j.dot(V[j])) * R_j)
            sub_loss[j] = sub_loss[j] - 0.5 * np.dot(V[j].dot(tmp_A), V[j])

        loss = loss + np.sum(sub_loss)
        cnn_loss = cnn_module.train(X_train=document, V=V, item_weight=item_weight)
        theta = cnn_module.get_projection_layer(document)

        loss = loss - 0.5 * lambda_v * cnn_loss * num_item
        
        toc = time.time()
        elapsed = toc - tic

        converge = abs((loss - PREV_LOSS) / PREV_LOSS)

        print("Loss: %.5f Elpased: %.4fs Converge: %.6f "% (loss, elapsed, converge))

        PREV_LOSS = loss

        if(converge< 0.01):
            endure-=1
            if endure==0:
                break
    res = {'U': U, 'V': V}
    return res
