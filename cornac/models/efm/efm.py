import numpy as np
from numba import jit

@jit(nopython=True)
def sgd_efm(A, X, Y, U1, U2, V, H1, H2,
            num_explicit_factors, num_latent_factors,
            lambda_x, lambda_y, lambda_u, lambda_h, lambda_v,
            max_iter, learning_rate, verbose):

    t = 0
    while t < max_iter:
        t += 1
        loss = 0
        for i in range(len(A)):
            for j in range(len(A[i])):
                if A[i, j] > 0:
                    prediction = U1[i, :].dot(U2.T[:, j]) + H1[i, :].dot(H2.T[:, j])
                    e1_ij = A[i,j] - prediction
                    loss += pow(e1_ij, 2)
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

    return U1, U2, V, H1, H2


def explicit_factor_model(A, X, Y, U1, U2, V, H1, H2,
                          num_explicit_factors, num_latent_factors,
                          lambda_x, lambda_y, lambda_u, lambda_h, lambda_v,
                          max_iter, learning_rate, verbose):

    t = 0
    I = np.eye(num_explicit_factors)
    while t < max_iter:
        t += 1
        # update V
        tmp1 = lambda_x * (X.T.dot(U1)) + lambda_y * (Y.T.dot(U2))
        tmp2 = V.dot(lambda_x * U1.T.dot(U1) + lambda_y * U2.T.dot(U2) + lambda_v * I)
        tmp3 = np.sqrt(np.divide(tmp1, tmp2))
        V = np.multiply(V, tmp3)
        # update U1
        tmp1 = A.dot(U2) + lambda_x * X.dot(V)
        tmp2 = (U1.dot(U2.T) + H1.dot(H2.T)).dot(U2) + U1.dot(lambda_x * V.T.dot(V) + lambda_u * I)
        tmp3 = np.sqrt(np.divide(tmp1, tmp2))
        U1 = np.multiply(U1, tmp3)
        # update U2
        tmp1 = A.T.dot(U1) + lambda_y * Y.dot(V)
        tmp2 = (U2.dot(U1.T) + H2.dot(H1.T)).dot(U1) + U2.dot(lambda_y * V.T.dot(V) + lambda_u * I)
        tmp3 = np.sqrt(np.divide(tmp1, tmp2))
        U2 = np.multiply(U2, tmp3)
        # update H1
        tmp1 = A.dot(H2)
        tmp2 = (U1.dot(U2.T) + H1.dot(H2.T)).dot(H2) + lambda_h * H1
        tmp3 = np.sqrt(np.divide(tmp1, tmp2))
        H1 = np.multiply(H1, tmp3)
        # update H2
        tmp1 = A.T.dot(H1)
        tmp2 = (U2.dot(U1.T) + H2.dot(H1.T)).dot(H1) + lambda_h * H2
        tmp3 = np.sqrt(np.divide(tmp1, tmp2))
        H2 = np.multiply(H2, tmp3)

        if verbose:
            e1 = ((U1.dot(U2.T) + H1.dot(H2.T) - A) ** 2).mean()
            e2 = lambda_x * (((U1.dot(V.T) - X) ** 2).mean())
            e3 = lambda_y * (((U2.dot(V.T) - Y) ** 2).mean())
            e4 = lambda_u * ((U1 ** 2).mean() + (U2 ** 2).mean())
            e5 = lambda_h * ((H1 ** 2).mean() + (H2 ** 2).mean())
            e6 = lambda_v * ((V ** 2).mean())
            loss = e1 + e2 + e3 + e4 + e5 + e6
            print('iter:', t, ', loss:', loss)

    return U1, U2, V, H1, H2