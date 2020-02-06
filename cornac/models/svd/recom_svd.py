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


from ..mf import MF


class SVD(MF):
    """Singular Value Decomposition (SVD).
    The implementation is based on Matrix Factorization with biases.

    Parameters
    ----------
    k: int, optional, default: 10
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    learning_rate: float, optional, default: 0.01
        The learning rate.

    lambda_reg: float, optional, default: 0.001
        The lambda value used for regularization.

    early_stop: boolean, optional, default: False
        When True, delta loss will be checked after each iteration to stop learning earlier.

    num_threads: int, optional, default: 0
        Number of parallel threads for training. If num_threads=0, all CPU cores will be utilized.
        If seed is not None, num_threads=1 to remove randomness from parallelization.

    trainable: boolean, optional, default: True
        When False, the model will not be re-trained, and input of pre-trained parameters are required.

    verbose: boolean, optional, default: True
        When True, running logs are displayed.

    init_params: dictionary, optional, default: None
        Initial parameters, e.g., init_params = {'U': user_factors, 'V': item_factors, \
        'Bu': user_biases, 'Bi': item_biases}

    seed: int, optional, default: None
        Random seed for weight initialization.
        If specified, training will take longer because of single-thread (no parallelization).

    References
    ----------
    * Koren, Y. Factorization meets the neighborhood: a multifaceted collaborative filtering model. \
    In SIGKDD, pp. 426-434. 2008.
    * Koren, Y. Factor in the neighbors: Scalable and accurate collaborative filtering. \
    In TKDD, 2010.
    """

    def __init__(
        self,
        name="SVD",
        k=10,
        max_iter=20,
        learning_rate=0.01,
        lambda_reg=0.02,
        early_stop=False,
        num_threads=0,
        trainable=True,
        verbose=False,
        init_params=None,
        seed=None,
    ):
        super().__init__(
            name=name,
            k=k,
            max_iter=max_iter,
            learning_rate=learning_rate,
            lambda_reg=lambda_reg,
            use_bias=True,
            early_stop=early_stop,
            num_threads=num_threads,
            trainable=trainable,
            verbose=verbose,
            init_params=init_params,
            seed=seed,
        )
