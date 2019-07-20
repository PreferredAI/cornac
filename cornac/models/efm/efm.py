import numpy as np
from numba import jit

@jit(nopython=True)
def sgd_efm(A, X, Y, U1, U2, V, H1, H2,
            global_mean, Bu, Bi,
            num_explicit_factors, num_latent_factors,
            lambda_x, lambda_y, lambda_u, lambda_h, lambda_v,
            use_bias, max_iter, learning_rate, verbose):

    t = 0
    while t < max_iter:
        t += 1
        loss = 0
        for i in range(len(A)):
            for j in range(len(A[i])):
                if A[i, j] > 0:
                    prediction = global_mean + Bu[i] + Bi[j] + U1[i, :].dot(U2.T[:, j]) + H1[i, :].dot(H2.T[:, j])
                    e1_ij = A[i,j] - prediction
                    loss += pow(e1_ij, 2)
                    if use_bias:
                        Bu[i] += learning_rate * (e1_ij - lambda_x * Bu[i])
                        Bi[j] += learning_rate * (e1_ij - lambda_y * Bi[j])
                    U1[i, :] = U1[i, :] + learning_rate * (e1_ij * U2[j, :] - lambda_u * U1[i, :])
                    U2[j, :] = U2[j, :] + learning_rate * (e1_ij * U1[i, :] - lambda_u * U2[j, :])
                    H1[i, :] = H1[i, :] + learning_rate * (e1_ij * H2[j, :] - lambda_h * H1[i, :])
                    H2[j, :] = H2[j, :] + learning_rate * (e1_ij * H1[i, :] - lambda_h * H2[j, :])

        for i in range(len(X)):
            for j in range(len(X[i])):
                if X[i, j] > 0:
                    e2_ij = X[i, j] - U1[i, :].dot(V.T[:, j])
                    loss += pow(e2_ij, 2)
                    U1[i, :] = U1[i, :] + learning_rate * (e2_ij * V[j, :] - lambda_u * U1[i, :])
                    V[j, :] = V[j, :] + learning_rate * (e2_ij * U1[i, :] - lambda_v * V[j, :])
        for i in range(len(Y)):
            for j in range(len(Y[i])):
                if Y[i, j] > 0:
                    e3_ij = Y[i, j] - U2[i, :].dot(V.T[:, j])
                    loss += pow(e3_ij, 2)
                    U2[i, :] = U2[i, :] + learning_rate * (e3_ij * V[j, :] - lambda_u * U2[j, :])
                    V[j, :] = V[j, :] + learning_rate * (e3_ij * U2[i, :] - lambda_v * V[j, :])
        e4 = lambda_u * (np.sum(U1 ** 2) + np.sum(U2 ** 2))
        e5 = lambda_h * (np.sum(H1 ** 2) + np.sum(H2 ** 2))
        e6 = lambda_v * (np.sum(V ** 2))
        loss += e4 + e5 + e6
        if verbose:
            print('iter:', t, ', loss:', loss)

    return U1, U2, V, H1, H2, Bu, Bi
