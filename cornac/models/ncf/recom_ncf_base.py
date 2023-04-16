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


from tqdm.auto import trange

from ..recommender import Recommender
from ...utils import get_rng


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
            if not hasattr(self, "graph"):
                self.num_users = self.train_set.num_users
                self.num_items = self.train_set.num_items
                self._build_graph()
            self._fit_tf()

        return self

    def _build_graph(self):
        import tensorflow.compat.v1 as tf

        self.graph = tf.Graph()

    def _sess_init(self):
        import tensorflow.compat.v1 as tf

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        self.sess.run(self.initializer)

    def _step_update(self, batch_users, batch_items, batch_ratings):
        _, _loss = self.sess.run(
            [self.train_op, self.loss],
            feed_dict={
                self.user_id: batch_users,
                self.item_id: batch_items,
                self.labels: batch_ratings.reshape(-1, 1),
            },
        )
        return _loss

    def _fit_tf(self):
        loop = trange(self.num_epochs, disable=not self.verbose)
        for _ in loop:
            count = 0
            sum_loss = 0
            for i, (batch_users, batch_items, batch_ratings) in enumerate(
                self.train_set.uir_iter(
                    self.batch_size, shuffle=True, binary=True, num_zeros=self.num_neg
                )
            ):
                _loss = self._step_update(batch_users, batch_items, batch_ratings)
                count += len(batch_ratings)
                sum_loss += _loss * len(batch_ratings)
                if i % 10 == 0:
                    loop.set_postfix(loss=(sum_loss / count))

            if self.early_stopping is not None and self.early_stop(
                **self.early_stopping
            ):
                break
        loop.close()

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
        # save TF weights
        self.saver.save(self.sess, model_file.replace(".pkl", ".cpt"))

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

        model._build_graph()
        model.saver.restore(model.sess, model.load_from.replace(".pkl", ".cpt"))

        return model

    def monitor_value(self):
        """Calculating monitored value used for early stopping on validation set (`val_set`).
        This function will be called by `early_stop()` function.

        Returns
        -------
        res : float
            Monitored value on validation set.
            Return `None` if `val_set` is `None`.
        """
        if self.val_set is None:
            return None

        from ...metrics import NDCG
        from ...eval_methods import ranking_eval

        ndcg_100 = ranking_eval(
            model=self,
            metrics=[NDCG(k=100)],
            train_set=self.train_set,
            test_set=self.val_set,
        )[0][0]

        return ndcg_100
