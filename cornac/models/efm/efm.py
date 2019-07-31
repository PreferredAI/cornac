import numpy as np
from numba import jit

@jit(nopython=True)
def sgd_efm(A, iA, jA, X, iX, jX, Y, iY, jY,
            U1, U2, V, H1, H2,
            global_mean, Bu, Bi,
            num_explicit_factors, num_latent_factors,
            lambda_x, lambda_y, lambda_u, lambda_h, lambda_v, lambda_reg,
            use_bias, max_iter, lr, verbose):

    t = 0
    while t < max_iter:
        t += 1
        loss = 0
        for i in range(len(iA)-1):
            for idx in range(iA[i], iA[i+1]):
                j = jA[idx]
                rating = A[idx]
                prediction = global_mean + Bu[i] + Bi[j] + U1[i, :].dot(U2.T[:, j]) + H1[i, :].dot(H2.T[:, j])
                eA_ij = rating - prediction
                loss += pow(eA_ij, 2)
                if use_bias:
                    Bu[i] += lr * (eA_ij - lambda_reg * Bu[i])
                    Bi[j] += lr * (eA_ij - lambda_reg * Bi[j])
                U1[i, :] += lr * (eA_ij * U2[j, :] - lambda_u * U1[i, :])
                U2[j, :] += lr * (eA_ij * U1[i, :] - lambda_u * U2[j, :])
                H1[i, :] += lr * (eA_ij * H2[j, :] - lambda_h * H1[i, :])
                H2[j, :] += lr * (eA_ij * H1[i, :] - lambda_h * H2[j, :])
        for i in range(len(iX) - 1):
            for idx in range(iX[i], iX[i+1]):
                j = jX[idx]
                score = X[idx]
                eX_ij = score - U1[i, :].dot(V.T[:, j])
                loss += pow(eX_ij, 2)
                U1[i, :] += lr * (eX_ij * V[j, :] - lambda_u * U1[i, :])
                V[j, :] += lr * (eX_ij * U1[i, :] - lambda_v * V[j, :])
        for i in range(len(iY) - 1):
            for idx in range(iY[i], iY[i+1]):
                j = jY[idx]
                score = Y[idx]
                eY_ij = score - U2[i, :].dot(V.T[:, j])
                loss += pow(eY_ij, 2)
                U2[i, :] += lr * (eY_ij * V[j, :] - lambda_u * U2[j, :])
                V[j, :] += lr * (eY_ij * U2[i, :] - lambda_v * V[j, :])
        eU = lambda_u * (np.sum(U1 ** 2) + np.sum(U2 ** 2))
        eH = lambda_h * (np.sum(H1 ** 2) + np.sum(H2 ** 2))
        eV = lambda_v * (np.sum(V ** 2))
        loss += eU + eH + eV
        if verbose:
            print('iter:', t, ', loss:', loss)

    return U1, U2, V, H1, H2, Bu, Bi
