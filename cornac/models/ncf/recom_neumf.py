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
    def _build_graph_tf(self):
        import tensorflow.compat.v1 as tf
        from .backend_tf import gmf, mlp, loss_fn, train_fn

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.seed)

            self.gmf_user_id = tf.placeholder(
                shape=[None], dtype=tf.int32, name="gmf_user_id"
            )
            self.mlp_user_id = tf.placeholder(
                shape=[None], dtype=tf.int32, name="mlp_user_id"
            )
            self.item_id = tf.placeholder(shape=[None], dtype=tf.int32, name="item_id")
            self.labels = tf.placeholder(
                shape=[None, 1], dtype=tf.float32, name="labels"
            )

            gmf_feat = gmf(
                uid=self.gmf_user_id,
                iid=self.item_id,
                num_users=self.num_users,
                num_items=self.num_items,
                emb_size=self.num_factors,
                reg_user=self.reg,
                reg_item=self.reg,
                seed=self.seed,
            )
            mlp_feat = mlp(
                uid=self.mlp_user_id,
                iid=self.item_id,
                num_users=self.num_users,
                num_items=self.num_items,
                layers=self.layers,
                reg_layers=[self.reg] * len(self.layers),
                act_fn=self.act_fn,
                seed=self.seed,
            )

            self.interaction = tf.concat([gmf_feat, mlp_feat], axis=-1)
            logits = tf.layers.dense(
                self.interaction,
                units=1,
                name="logits",
                kernel_initializer=tf.initializers.lecun_uniform(self.seed),
            )
            self.prediction = tf.nn.sigmoid(logits)

            self.loss = loss_fn(labels=self.labels, logits=logits)
            self.train_op = train_fn(
                self.loss, learning_rate=self.lr, learner=self.learner
            )

            self.initializer = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

        self._sess_init_tf()

        if self.pretrained:
            gmf_kernel = self.pretrained_gmf.sess.run(
                self.pretrained_gmf.sess.graph.get_tensor_by_name("logits/kernel:0")
            )
            gmf_bias = self.pretrained_gmf.sess.run(
                self.pretrained_gmf.sess.graph.get_tensor_by_name("logits/bias:0")
            )
            mlp_kernel = self.pretrained_mlp.sess.run(
                self.pretrained_mlp.sess.graph.get_tensor_by_name("logits/kernel:0")
            )
            mlp_bias = self.pretrained_mlp.sess.run(
                self.pretrained_mlp.sess.graph.get_tensor_by_name("logits/bias:0")
            )
            logits_kernel = np.concatenate(
                [self.alpha * gmf_kernel, (1 - self.alpha) * mlp_kernel]
            )
            logits_bias = self.alpha * gmf_bias + (1 - self.alpha) * mlp_bias

            for v in self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                if v.name.startswith("GMF"):
                    sess = self.pretrained_gmf.sess
                    self.sess.run(
                        tf.assign(v, sess.run(sess.graph.get_tensor_by_name(v.name)))
                    )
                elif v.name.startswith("MLP"):
                    sess = self.pretrained_mlp.sess
                    self.sess.run(
                        tf.assign(v, sess.run(sess.graph.get_tensor_by_name(v.name)))
                    )
                elif v.name.startswith("logits/kernel"):
                    self.sess.run(tf.assign(v, logits_kernel))
                elif v.name.startswith("logits/bias"):
                    self.sess.run(tf.assign(v, logits_bias))

    def _get_feed_dict(self, batch_users, batch_items, batch_ratings):
        return {
            self.gmf_user_id: batch_users,
            self.mlp_user_id: batch_users,
            self.item_id: batch_items,
            self.labels: batch_ratings.reshape(-1, 1),
        }

    def _score_tf(self, user_idx, item_idx):
        if item_idx is None:
            feed_dict = {
                self.gmf_user_id: [user_idx],
                self.mlp_user_id: np.ones(self.num_items) * user_idx,
                self.item_id: np.arange(self.num_items),
            }
        else:
            feed_dict = {
                self.gmf_user_id: [user_idx],
                self.mlp_user_id: [user_idx],
                self.item_id: [item_idx],
            }
        return self.sess.run(self.prediction, feed_dict=feed_dict)

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
