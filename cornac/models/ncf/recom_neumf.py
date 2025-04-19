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
from ...exception import ScoreException


class NeuMF(NCFBase):
    """Neural Matrix Factorization.

    Parameters
    ----------
    num_factors: int, optional, default: 8
        Embedding size of MF model.

    layers: list, optional, default: [64, 32, 16, 8]
        MLP layers. Note that the first layer is the concatenation of
        user and item embeddings. So layers[0]/2 is the embedding size.

    act_fn: str, default: 'relu'
        Name of the activation function used for the MLP layers.
        Supported functions: ['sigmoid', 'tanh', 'elu', 'relu', 'selu, 'relu6', 'leaky_relu']

    reg: float, optional, default: 0.
        Regularization (weight_decay).

    reg_layers: list, optional, default: [0., 0., 0., 0.]
        Regularization for each MLP layer,
        reg_layers[0] is the regularization for embeddings.

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

    name: string, optional, default: 'NeuMF'
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
        name="NeuMF",
        num_factors=8,
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
        self.num_factors = num_factors
        self.layers = layers
        self.act_fn = act_fn
        self.reg = reg
        self.pretrained = False
        self.ignored_attrs.extend(
            [
                "gmf_user_id",
                "mlp_user_id",
                "pretrained_gmf",
                "pretrained_mlp",
                "alpha",
            ]
        )

    def from_pretrained(self, pretrained_gmf, pretrained_mlp, alpha=0.5):
        """Provide pre-trained GMF and MLP models. Section 3.4.1 of the paper.

        Parameters
        ----------
        pretrained_gmf: object of type GMF, required
            Reference to trained/fitted GMF model.

        pretrained_mlp: object of type MLP, required
            Reference to trained/fitted MLP model.

        alpha: float, optional, default: 0.5
            Hyper-parameter determining the trade-off between the two pre-trained models.
            Details are described in the section 3.4.1 of the paper.
        """
        self.pretrained = True
        self.pretrained_gmf = pretrained_gmf
        self.pretrained_mlp = pretrained_mlp
        self.alpha = alpha
        return self

    ########################
    ## TensorFlow backend ##
    ########################
    def _build_model_tf(self):
        import tensorflow as tf
        from .backend_tf import GMFLayer, MLPLayer
        
        # Define inputs
        user_input = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name="user_input")
        item_input = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name="item_input")
        
        # GMF layer
        gmf_layer = GMFLayer(
            num_users=self.num_users,
            num_items=self.num_items,
            emb_size=self.num_factors,
            reg_user=self.reg,
            reg_item=self.reg,
            seed=self.seed,
            name="gmf_layer"
        )
        
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
        
        # Get embeddings and element-wise product
        gmf_vector = gmf_layer([user_input, item_input])
        mlp_vector = mlp_layer([user_input, item_input])
        
        # Concatenate GMF and MLP vectors
        concat_vector = tf.keras.layers.Concatenate(axis=-1)([gmf_vector, mlp_vector])
        
        # Output layer
        logits = tf.keras.layers.Dense(
            1,
            kernel_initializer=tf.keras.initializers.LecunUniform(seed=self.seed),
            name="logits"
        )(concat_vector)
        
        prediction = tf.keras.layers.Activation('sigmoid', name="prediction")(logits)
        
        # Create model
        model = tf.keras.Model(
            inputs=[user_input, item_input],
            outputs=prediction,
            name="NeuMF"
        )
        
        # Handle pretrained models
        if self.pretrained:
            # Get GMF and MLP models
            gmf_model = self.pretrained_gmf.model
            mlp_model = self.pretrained_mlp.model
            
            # Copy GMF embeddings
            model.get_layer('gmf_layer').user_embedding.set_weights(
                gmf_model.get_layer('gmf_layer').user_embedding.get_weights()
            )
            model.get_layer('gmf_layer').item_embedding.set_weights(
                gmf_model.get_layer('gmf_layer').item_embedding.get_weights()
            )
            
            # Copy MLP embeddings and layers
            model.get_layer('mlp_layer').user_embedding.set_weights(
                mlp_model.get_layer('mlp_layer').user_embedding.get_weights()
            )
            model.get_layer('mlp_layer').item_embedding.set_weights(
                mlp_model.get_layer('mlp_layer').item_embedding.get_weights()
            )
            
            # Copy dense layers in MLP
            for i, layer in enumerate(model.get_layer('mlp_layer').dense_layers):
                layer.set_weights(mlp_model.get_layer('mlp_layer').dense_layers[i].get_weights())
            
            # Combine weights for output layer
            gmf_logits_weights = gmf_model.get_layer('logits').get_weights()
            mlp_logits_weights = mlp_model.get_layer('logits').get_weights()
            
            # Combine kernel weights
            combined_kernel = np.concatenate([
                self.alpha * gmf_logits_weights[0],
                (1.0 - self.alpha) * mlp_logits_weights[0]
            ], axis=0)
            
            # Combine bias weights
            combined_bias = self.alpha * gmf_logits_weights[1] + (1.0 - self.alpha) * mlp_logits_weights[1]
            
            # Set combined weights to output layer
            model.get_layer('logits').set_weights([combined_kernel, combined_bias])
        
        return model

    #####################
    ## PyTorch backend ##
    #####################
    def _build_model_pt(self):
        from .backend_pt import NeuMF

        model = NeuMF(
            num_users=self.num_users,
            num_items=self.num_items,
            layers=self.layers,
            act_fn=self.act_fn,
        )
        if self.pretrained:
            model.from_pretrained(
                self.pretrained_gmf.model, self.pretrained_mlp.model, self.alpha
            )
        return model

    def _score_pt(self, user_idx, item_idx):
        import torch

        with torch.no_grad():
            if item_idx is None:
                users = torch.from_numpy(np.ones(self.num_items, dtype=int) * user_idx)
                items = (torch.from_numpy(np.arange(self.num_items))).to(self.device)
            else:
                users = torch.tensor(user_idx).unsqueeze(0)
                items = torch.tensor(item_idx).unsqueeze(0)
            gmf_users = torch.tensor(user_idx).unsqueeze(0).to(self.device)
            output = self.model(
                users.to(self.device), items.to(self.device), gmf_users.to(self.device)
            )
        return output.squeeze().cpu().numpy()
