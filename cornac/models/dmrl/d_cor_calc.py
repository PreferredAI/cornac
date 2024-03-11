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

import torch


class DistanceCorrelationCalculator:
    """
    Calculates the disentangled loss for DMRL model.
    Please see https://arxiv.org/pdf/2203.05406.pdf for more details.
    """

    def __init__(self, n_factors, num_neg) -> None:
        self.n_factors = n_factors
        self.num_neg = num_neg

    def calculate_cov(self, X, Y):
        """
        Computes the distance covariance between X and Y.
        :param X: A 3D torch tensor.
        :param Y: A 3D torch tensor.
        :return: A 1D torch tensor of len 1+num_neg.
        """
        # first create centered distance matrices
        X = self.cent_dist(X)
        Y = self.cent_dist(Y)

        # batch_size is dim 1, as dim 0 is one positive and num_neg negative samples
        n_samples = X.shape[1]
        # then calculate the covariance as a 1D array of length 1+num_neg
        cov = torch.sqrt(torch.max(torch.sum(X * Y, dim=(1, 2)) / (n_samples * n_samples), torch.tensor(1e-5)))
        return cov

    def calculate_var(self, X):
        """
        Computes the distance variance of X.
        :param X: A 3D torch tensor.
        :return: A 1D torch tensor of len 1+mum_neg.
        """
        return self.calculate_cov(X, X)

    def calculate_cor(self, X, Y):
        """
        Computes the distance correlation between X and Y.

        :param X: A 3D torch tensor.
        :param Y: A 3D torch tensor.
        :return: A 1D torch tensor of len 1+mum_neg.
        """
        return self.calculate_cov(X, Y) / torch.sqrt(torch.max(self.calculate_var(X) * self.calculate_var(Y), torch.tensor(0.0)))

    def cent_dist(self, X):
        """
        Computes the pairwise euclidean distance between rows of X and centers
        each cell of the distance matrix with row mean, column mean, and grand mean.
        """
        # put the samples from dim 1 into dim 0
        X = torch.transpose(X, dim0=0, dim1=1)

        # Now use pythagoras to calculate the distance matrix
        first_part = torch.sum(torch.square(X), dim=-1, keepdims=True)
        middle_part = torch.matmul(X, torch.transpose(X, dim0=1, dim1=2))
        last_part = torch.transpose(first_part, dim0=1, dim1=2)

        D = torch.sqrt(torch.max(first_part - 2 * middle_part + last_part, torch.tensor(1e-5)))
        # dim0 is the negative samples, dim1 is batch_size, dim2 is the kth factor of the embedding_dim

        row_mean = torch.mean(D, dim=2, keepdim=True)
        column_mean = torch.mean(D, dim=1, keepdim=True)
        global_mean = torch.mean(D, dim=(1, 2), keepdim=True)
        D = D - row_mean - column_mean + global_mean
        return D

    def  calculate_disentangled_loss(
            self,
            item_embedding_factors: torch.Tensor,
            user_embedding_factors: torch.Tensor,
            text_embedding_factors: torch.Tensor,
            image_embedding_factors: torch.Tensor):
        """
        Calculates the disentangled loss for the given factors.

        :param item_embedding_factors: A list of 3D torch tensors.
        :param user_embedding_factors: A list of 3D torch tensors.
        :param text_embedding_factors: A list of 3D torch tensors.
        :return: A 1D torch tensor of len 1+mum_neg.
        """
        cor_loss = torch.tensor([0.0] * (1 + self.num_neg))
        for i in range(0, self.n_factors - 2):
            for j in range(i + 1, self.n_factors - 1):
                cor_loss += self.calculate_cor(item_embedding_factors[i], item_embedding_factors[j])
                cor_loss += self.calculate_cor(user_embedding_factors[i], user_embedding_factors[j])
                if text_embedding_factors[i].numel() > 0:
                    cor_loss += self.calculate_cor(text_embedding_factors[i], text_embedding_factors[j])
                if image_embedding_factors[i].numel() > 0:
                    cor_loss += self.calculate_cor(image_embedding_factors[i], image_embedding_factors[j])

        cor_loss = cor_loss / ((self.n_factors + 1.0) * self.n_factors / 2)

        # two options, we can either return the sum over the 1 positive and num_neg negative samples.
        # or we can return only the loss of the one positive sample, as they did in the paper

        # return torch.sum(cor_loss)
        return cor_loss[0]
