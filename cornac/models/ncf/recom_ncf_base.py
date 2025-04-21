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
from tqdm.auto import trange

from ..recommender import Recommender
from ...utils import get_rng
from ...exception import ScoreException


class NCFBase(Recommender):
    """Base class of NCF family.

    Parameters
    ----------
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

    name: string, optional, default: 'NCF'
        Name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already \
        pre-trained.

    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    References
    ----------
    * He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017, April). Neural collaborative filtering. \
    In Proceedings of the 26th international conference on world wide web (pp. 173-182).
    """

    def __init__(
        self,
        name="NCF",
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
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.lr = lr
        self.learner = learner
        self.backend = backend
        self.early_stopping = early_stopping
        self.seed = seed
        self.rng = get_rng(seed)
        self.ignored_attrs.extend(
            [
                "graph",
                "user_id",
                "item_id",
                "labels",
                "interaction",
                "prediction",
                "loss",
                "train_op",
                "initializer",
                "saver",
                "sess",
            ]
        )

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """
        Recommender.fit(self, train_set, val_set)

        if self.trainable:
            self.num_users = self.num_users
            self.num_items = self.num_items

            if self.backend == "tensorflow":
                self._fit_tf(train_set, val_set)
            elif self.backend == "pytorch":
                self._fit_pt(train_set, val_set)
            else:
                raise ValueError(f"{self.backend} is not supported")

        return self

    ########################
    ## TensorFlow backend ##
    ########################
    def _build_model_tf(self):
        raise NotImplementedError()

    def _fit_tf(self, train_set, val_set):
        import tensorflow as tf
        
        # Set random seed for reproducibility
        if self.seed is not None:
            tf.random.set_seed(self.seed)
            np.random.seed(self.seed)
        
        # Configure GPU memory growth to avoid OOM errors
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        
        # Build the model
        self.model = self._build_model_tf()
        
        # Get optimizer
        from .backend_tf import get_optimizer
        optimizer = get_optimizer(learning_rate=self.lr, learner=self.learner)
        
        # Training loop
        loop = trange(self.num_epochs, disable=not self.verbose)
        for _ in loop:
            count = 0
            sum_loss = 0
            for i, (batch_users, batch_items, batch_ratings) in enumerate(
                train_set.uir_iter(
                    self.batch_size, shuffle=True, binary=True, num_zeros=self.num_neg
                )
            ):
                batch_ratings = batch_ratings.reshape(-1, 1, 1)
                
                # Convert to tensors
                batch_users = tf.convert_to_tensor(batch_users, dtype=tf.int32)
                batch_items = tf.convert_to_tensor(batch_items, dtype=tf.int32)
                batch_ratings = tf.convert_to_tensor(batch_ratings, dtype=tf.float32)
                
                # Training step
                with tf.GradientTape() as tape:
                    predictions = self.model([batch_users, batch_items], training=True)
                    cross_entropy = tf.keras.losses.binary_crossentropy(
                        y_true=batch_ratings,
                        y_pred=predictions,
                        from_logits=False  # predictions are already probabilities
                    )
                    cross_entropy = tf.reduce_mean(cross_entropy)
                    loss_value = cross_entropy + tf.reduce_sum(self.model.losses)
                    
                # Apply gradients
                grads = tape.gradient(loss_value, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                
                count += len(batch_users)
                sum_loss += len(batch_users) * loss_value.numpy()
                if i % 10 == 0:
                    loop.set_postfix(loss=(sum_loss / count))
            
            if self.early_stopping is not None and self.early_stop(
                train_set, val_set, **self.early_stopping
            ):
                break
        loop.close()

    def _score_tf(self, user_idx, item_idx):
        """Score function for TensorFlow models."""
        import tensorflow as tf
        
        if item_idx is None:
            # Score all items for a given user
            user_tensor = tf.convert_to_tensor([user_idx], dtype=tf.int32)
            item_tensor = tf.convert_to_tensor(np.arange(self.num_items), dtype=tf.int32)
            
            # Broadcast user_idx to match the shape of item_tensor
            user_tensor = tf.broadcast_to(user_tensor, shape=item_tensor.shape)
        else:
            # Score a specific item for a given user
            user_tensor = tf.convert_to_tensor([user_idx], dtype=tf.int32)
            item_tensor = tf.convert_to_tensor([item_idx], dtype=tf.int32)
        
        # Get predictions
        predictions = self.model([user_tensor, item_tensor], training=False)
        return predictions.numpy().squeeze()

    #####################
    ## PyTorch backend ##
    #####################
    def _build_model_pt(self):
        raise NotImplementedError()

    def _fit_pt(self, train_set, val_set):
        import torch
        import torch.nn as nn
        from .backend_pt import optimizer_dict

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)

        self.model = self._build_model_pt().to(self.device)

        optimizer = optimizer_dict[self.learner](
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.reg,
        )
        criteria = nn.BCELoss()

        loop = trange(self.num_epochs, disable=not self.verbose)
        for _ in loop:
            count = 0
            sum_loss = 0
            for batch_id, (batch_users, batch_items, batch_ratings) in enumerate(
                train_set.uir_iter(
                    self.batch_size, shuffle=True, binary=True, num_zeros=self.num_neg
                )
            ):
                batch_users = torch.from_numpy(batch_users).to(self.device)
                batch_items = torch.from_numpy(batch_items).to(self.device)
                batch_ratings = torch.tensor(batch_ratings, dtype=torch.float).to(
                    self.device
                )

                optimizer.zero_grad()
                outputs = self.model(batch_users, batch_items)
                loss = criteria(outputs, batch_ratings)
                loss.backward()
                optimizer.step()

                count += len(batch_users)
                sum_loss += len(batch_users) * loss.data.item()

                if batch_id % 10 == 0:
                    loop.set_postfix(loss=(sum_loss / count))

            if self.early_stopping is not None and self.early_stop(
                train_set, val_set, **self.early_stopping
            ):
                break
        loop.close()

    def _score_pt(self, user_idx, item_idx):
        raise NotImplementedError()

    def save(self, save_dir=None):
        """Save a recommender model to the filesystem.

        Parameters
        ----------
        save_dir: str, default: None
            Path to a directory for the model to be stored.

        """
        if save_dir is None:
            return

        model_file = Recommender.save(self, save_dir)

        if self.backend == "tensorflow":
            # Save the TensorFlow model
            if hasattr(self, "model"):
                self.model.save_weights(model_file.replace(".pkl", ".h5"))
        elif self.backend == "pytorch":
            # TODO: implement model saving for PyTorch
            raise NotImplementedError()

        return model_file

    @staticmethod
    def load(model_path, trainable=False):
        """Load a recommender model from the filesystem.

        Parameters
        ----------
        model_path: str, required
            Path to a file or directory where the model is stored. If a directory is
            provided, the latest model will be loaded.

        trainable: boolean, optional, default: False
            Set it to True if you would like to finetune the model. By default,
            the model parameters are assumed to be fixed after being loaded.

        Returns
        -------
        self : object
        """
        model = Recommender.load(model_path, trainable)
        if hasattr(model, "pretrained"):  # NeuMF
            model.pretrained = False

        if model.backend == "tensorflow":
            # Build the model
            model.model = model._build_model_tf()
            # Load weights
            model.model.load_weights(model.load_from.replace(".pkl", ".h5"))
        elif model.backend == "pytorch":
            # TODO: implement model loading for PyTorch
            raise NotImplementedError()

        return model

    def monitor_value(self, train_set, val_set):
        """Calculating monitored value used for early stopping on validation set (`val_set`).
        This function will be called by `early_stop()` function.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        res : float
            Monitored value on validation set.
            Return `None` if `val_set` is `None`.
        """
        if val_set is None:
            return None

        from ...metrics import NDCG
        from ...eval_methods import ranking_eval

        ndcg_100 = ranking_eval(
            model=self,
            metrics=[NDCG(k=100)],
            train_set=train_set,
            test_set=val_set,
        )[0][0]

        return ndcg_100

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for which to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items
        """
        if self.is_unknown_user(user_idx):
            raise ScoreException("Can't make score prediction for user %d" % user_idx)

        if item_idx is not None and self.is_unknown_item(item_idx):
            raise ScoreException("Can't make score prediction for item %d" % item_idx)

        if self.backend == "tensorflow":
            pred_scores = self._score_tf(user_idx, item_idx)
        elif self.backend == "pytorch":
            pred_scores = self._score_pt(user_idx, item_idx)

        return pred_scores.ravel()
