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

from .recom_ncf_base import NCFBase


class MLP(NCFBase):
    """Multi-Layer Perceptron.

    Parameters
    ----------
    layers: list, optional, default: [64, 32, 16, 8]
        MLP layers. Note that the first layer is the concatenation of
        user and item embeddings. So layers[0]/2 is the embedding size.

    act_fn: str, default: 'relu'
        Name of the activation function used for the MLP layers.
        Supported functions: ['sigmoid', 'tanh', 'elu', 'relu', 'selu, 'relu6', 'leaky_relu']

    reg: float, optional, default: 0.
        Regularization (weight_decay).

    num_epochs: int, optional, default: 20
        Number of epochs.

    batch_size: int, optional, default: 256
        Batch size.

    num_neg: int, optional, default: 4
        Number of negative instances to pair with a positive instance.

    lr: float, optional, default: 0.001
        Learning rate.

    learner: str, optional, default: 'adam'
        Specify an optimizer: adagrad, adam, rmsprop, sgd
    
    backend: str, optional, default: 'tensorflow'
        Backend used for model training: tensorflow, pytorch
        
    early_stopping: {min_delta: float, patience: int}, optional, default: None
        If `None`, no early stopping. Meaning of the arguments:
        
        - `min_delta`: the minimum increase in monitored value on validation set to be considered as improvement, \
           i.e. an increment of less than min_delta will count as no improvement.
        
        - `patience`: number of epochs with no improvement after which training should be stopped.

    name: string, optional, default: 'MLP'
        Name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already \
        pre-trained.

    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    seed: int, optional, default: None
        Random seed for parameters initialization.

    References
    ----------
    * He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017, April). Neural collaborative filtering. \
    In Proceedings of the 26th international conference on world wide web (pp. 173-182).
    """

    def __init__(
        self,
        name="MLP",
        layers=(64, 32, 16, 8),
        act_fn="relu",
        reg=0.0,
        num_epochs=20,
        batch_size=256,
        num_neg=4,
        lr=0.001,
        learner="adam",
        backend="tensorflow",
        early_stopping=None,
        trainable=True,
        verbose=True,
        seed=None,
    ):
        super().__init__(
            name=name,
            trainable=trainable,
            verbose=verbose,
            num_epochs=num_epochs,
            batch_size=batch_size,
            num_neg=num_neg,
            lr=lr,
            learner=learner,
            backend=backend,
            early_stopping=early_stopping,
            seed=seed,
        )
        self.layers = layers
        self.act_fn = act_fn
        self.reg = reg

    ########################
    ## TensorFlow backend ##
    ########################
    def _build_model_tf(self):
        import tensorflow as tf
        from .backend_tf import MLPLayer
        
        # Define inputs
        user_input = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name="user_input")
        item_input = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name="item_input")
        
        # MLP layer
        mlp_layer = MLPLayer(
            num_users=self.num_users,
            num_items=self.num_items,
            layers=self.layers,
            reg_layers=[self.reg] * len(self.layers),
            act_fn=self.act_fn,
            seed=self.seed,
            name="mlp_layer"
        )
        
        # Get MLP vector
        mlp_vector = mlp_layer([user_input, item_input])
        
        # Output layer
        logits = tf.keras.layers.Dense(
            1,
            kernel_initializer=tf.keras.initializers.LecunUniform(seed=self.seed),
            name="logits"
        )(mlp_vector)
        
        prediction = tf.keras.layers.Activation('sigmoid', name="prediction")(logits)
        
        # Create model
        model = tf.keras.Model(
            inputs=[user_input, item_input],
            outputs=prediction,
            name="MLP"
        )
        
        return model

    #####################
    ## PyTorch backend ##
    #####################
    def _build_model_pt(self):
        from .backend_pt import MLP

        return MLP(
            num_users=self.num_users,
            num_items=self.num_items,
            layers=self.layers,
            act_fn=self.act_fn,
        )

    def _score_pt(self, user_idx, item_idx):
        import torch

        with torch.no_grad():
            if item_idx is None:
                users = torch.from_numpy(np.ones(self.num_items, dtype=int) * user_idx)
                items = (torch.from_numpy(np.arange(self.num_items))).to(self.device)
            else:
                users = torch.tensor(user_idx).unsqueeze(0)
                items = torch.tensor(item_idx).unsqueeze(0)
            output = self.model(users.to(self.device), items.to(self.device))
        return output.squeeze().cpu().numpy()
