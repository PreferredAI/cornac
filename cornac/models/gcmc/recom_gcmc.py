from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import dgl
import time
import logging
import scipy.sparse as sp
from tqdm.auto import trange

from ..recommender import Recommender
from ...utils import get_rng
from ...utils.init_utils import xavier_uniform
from ...exception import ScoreException

from .gcmc import NeuralNetwork
from .utils import get_optimizer, torch_net_info, torch_total_param_num


class GCMC(Recommender):
    def __init__(
        self,
        name="GCMC",
        max_iter=2000,
        learning_rate=0.01,
        optimizer="adam",
        activation_model="leaky",
        gcn_agg_units=500,
        gcn_out_units=75,
        gcn_dropout=0.7,
        gcn_agg_accum="stack",
        share_param=False,
        gen_r_num_basis_func=2,
        train_grad_clip=1.0,
        train_valid_interval=1,
        train_early_stopping_patience=100,
        train_min_learning_rate=0.001,
        train_decay_patience=50,
        train_lr_decay_factor=0.5,
        init_params=None,
        trainable=True,
        verbose=True,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)

        self.name = name
        self.max_iter = max_iter
        self.verbose = verbose
        self.seed = seed
        self.activation_model = activation_model
        self.gcn_agg_units = gcn_agg_units
        self.gcn_out_units = gcn_out_units
        self.gcn_dropout = gcn_dropout
        self.gcn_agg_accum = gcn_agg_accum
        self.share_param = share_param
        self.gen_r_num_basis_func = gen_r_num_basis_func
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.train_grad_clip = train_grad_clip
        self.train_valid_interval = train_valid_interval
        self.train_early_stopping_patience = train_early_stopping_patience
        self.train_min_learning_rate = train_min_learning_rate
        self.train_decay_patience = train_decay_patience
        self.train_lr_decay_factor = train_lr_decay_factor

        # Init params if provided
        self.init_params = {} if init_params is None else init_params

    def _init(self):
        # rng = get_rng(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        rating_values = self.train_set.uir_tuple[2]  # rating array
        self.rating_values = np.unique(rating_values)

    def fit(self, train_set, val_set=None):
        Recommender.fit(self, train_set, val_set)

        self._init()

        if self.trainable:
            self._fit_torch()

        return self

    def _apply_support(self, graph, rating_values, n_users, n_items, symm=True):
        def _calc_norm(x):
            x = x.numpy().astype("float32")
            x[x == 0.0] = np.inf
            x = torch.FloatTensor(1.0 / np.sqrt(x))
            return x.unsqueeze(1)

        user_ci = []
        user_cj = []
        item_ci = []
        item_cj = []

        for r in rating_values:
            r = str(r).replace(".", "_")
            user_ci.append(graph["rev-%s" % r].in_degrees())
            item_ci.append(graph[r].in_degrees())

            if symm:
                user_cj.append(graph[r].out_degrees())
                item_cj.append(graph["rev-%s" % r].out_degrees())
            else:
                user_cj.append(torch.zeros((n_users,)))
                item_cj.append(torch.zeros((n_items,)))
        user_ci = _calc_norm(sum(user_ci))
        item_ci = _calc_norm(sum(item_ci))
        if symm:
            user_cj = _calc_norm(sum(user_cj))
            item_cj = _calc_norm(sum(item_cj))
        else:
            user_cj = torch.ones(
                n_users,
            )
            item_cj = torch.ones(
                n_items,
            )
        graph.nodes["user"].data.update({"ci": user_ci, "cj": user_cj})
        graph.nodes["item"].data.update({"ci": item_ci, "cj": item_cj})

        return graph

    def _uir_tuple_to_r_data_dict(self, uir_tuple):
        """
        Convert uir tuple to a dictionary where keys are ratings,
        values are tuples of two lists (users, items) for that rating.
        """
        r_data = defaultdict()

        for u, i, r in zip(*uir_tuple):
            if r not in r_data:
                r_data.setdefault(r, ([], []))
            r_data[r][0].append(u)
            r_data[r][1].append(i)

        return r_data

    def _generate_enc_graph(self, data_set, add_support=False):
        data_dict = dict()
        num_nodes_dict = {"user": data_set.total_users, "item": data_set.total_items}
        rating_row, rating_col, rating_values = data_set.uir_tuple
        for rating in set(rating_values):
            ridx = np.where(rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating = str(rating).replace(".", "_")
            data_dict.update(
                {
                    ("user", str(rating), "item"): (rrow, rcol),
                    ("item", "rev-%s" % str(rating), "user"): (rcol, rrow),
                }
            )
            
        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

        # sanity check
        assert (
            len(data_set.uir_tuple[2])
            == sum([graph.num_edges(et) for et in graph.etypes]) // 2
        )

        if add_support:
            graph = self._apply_support(
                graph=graph,
                rating_values=self.rating_values,
                n_users=data_set.total_users,
                n_items=data_set.total_items,
            )

        return graph

    def _generate_dec_graph(self, data_set):
        rating_pairs = data_set.uir_tuple[:2]
        ones = np.ones_like(rating_pairs[0])
        user_item_ratings_coo = sp.coo_matrix(
            (ones, rating_pairs),
            shape=(data_set.total_users, data_set.total_items),
            dtype=np.float32,
        )

        g = dgl.bipartite_from_scipy(
            user_item_ratings_coo, utype="_U", etype="_E", vtype="_V"
        )

        return dgl.heterograph(
            {("user", "rate", "item"): g.edges()},
            num_nodes_dict={"user": data_set.total_users, "item": data_set.total_items},
        )

    def _fit_torch(self):
        # Prepare Data
        def _generate_labels(ratings):
            labels = torch.LongTensor(np.searchsorted(self.rating_values, ratings)).to(
                self.device
            )
            return labels

        self.train_enc_graph = self._generate_enc_graph(
            self.train_set, add_support=True
        )
        self.train_dec_graph = self._generate_dec_graph(self.train_set)

        train_labels = _generate_labels(self.train_set.uir_tuple[2])
        train_truths = torch.FloatTensor(self.train_set.uir_tuple[2]).to(self.device)

        # TODO: possibility of validation_set to be None. Check if it works!

        self.valid_enc_graph = self.train_enc_graph
        self.valid_dec_graph = self._generate_dec_graph(self.val_set)
        # valid_labels = _generate_labels(self.val_set.uir_tuple[2])
        valid_truths = torch.FloatTensor(self.val_set.uir_tuple[2]).to(self.device)

        def _count_pairs(graph):
            pair_count = 0
            for r_val in self.rating_values:
                r_val = str(r_val).replace(".", "_")
                pair_count += graph.num_edges(str(r_val))
            print("pair count:", pair_count)
            return pair_count

        print(
            "Train enc graph: {} users, {} items, {} pairs".format(
                self.train_enc_graph.num_nodes("user"),
                self.train_enc_graph.num_nodes("item"),
                _count_pairs(self.train_enc_graph),
            )
        )

        print(
            "Train dec graph: {} users, {} items, {} pairs".format(
                self.train_dec_graph.num_nodes("user"),
                self.train_dec_graph.num_nodes("item"),
                self.train_dec_graph.num_edges(),
            )
        )

        print(
            "Valid enc graph: {} users, {} items, {} pairs".format(
                self.valid_enc_graph.num_nodes("user"),
                self.valid_enc_graph.num_nodes("item"),
                _count_pairs(self.valid_enc_graph),
            )
        )

        print(
            "Valid dec graph: {} users, {} items, {} pairs".format(
                self.valid_dec_graph.num_nodes("user"),
                self.valid_dec_graph.num_nodes("item"),
                self.valid_dec_graph.num_edges(),
            )
        )

        # Build Net
        self.net = NeuralNetwork(
            self.activation_model,
            self.rating_values,
            self.train_set.total_users,
            self.train_set.total_items,
            self.gcn_agg_units,
            self.gcn_out_units,
            self.gcn_dropout,
            self.gcn_agg_accum,
            self.gen_r_num_basis_func,
            self.share_param,
            self.device,
        )
        self.net = self.net.to(self.device)
        nd_positive_rating_values = torch.FloatTensor(self.rating_values).to(
            self.device
        )
        rating_loss_net = nn.CrossEntropyLoss()
        learning_rate = self.learning_rate
        optimizer = get_optimizer(self.optimizer)(
            self.net.parameters(), lr=learning_rate
        )
        print("NN Loading Complete!")

        # Prepare training data

        # declare the loss information
        best_valid_rmse = np.inf
        no_better_valid = 0
        best_iter = -1
        count_rmse = 0
        count_num = 0
        count_loss = 0

        self.train_enc_graph = self.train_enc_graph.int().to(self.device)
        self.train_dec_graph = self.train_dec_graph.int().to(self.device)
        # valid_enc_graph = valid_enc_graph.int().to(self.device)
        self.valid_enc_graph = self.train_enc_graph
        self.valid_dec_graph = self.valid_dec_graph.int().to(self.device)

        print(self.train_enc_graph.device)

        print("Training Started!")
        dur = []
        for iter_idx in range(1, self.max_iter):
            if iter_idx > 3:
                t0 = time.time()
            self.net.train()
            pred_ratings = self.net(
                self.train_enc_graph,
                self.train_dec_graph,
                None,
                None,
            )
            loss = rating_loss_net(pred_ratings, train_labels).mean()
            count_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), self.train_grad_clip)
            optimizer.step()

            if iter_idx > 3:
                dur.append(time.time() - t0)

            if iter_idx == 1:
                print(
                    "Total # params of net: {}".format(torch_total_param_num(self.net))
                )
                print(torch_net_info(self.net))

            real_pred_ratings = (
                torch.softmax(pred_ratings, dim=1)
                * nd_positive_rating_values.view(1, -1)
            ).sum(dim=1)

            rmse = ((real_pred_ratings - train_truths) ** 2).sum()
            count_rmse += rmse.item()
            count_num += len(train_truths)

            if self.verbose == True:
                logging_str = (
                    "Epoch: {}, loss: {:.4f}, rmse:{:.4f}, time:{:.4f}".format(
                        iter_idx,
                        count_loss / iter_idx,
                        count_rmse / count_num,
                        np.average(dur),
                    )
                )
                count_rmse = 0
                count_num = 0

            if iter_idx & self.train_valid_interval == 0:
            # if False:
                nd_positive_rating_values = torch.FloatTensor(self.rating_values).to(
                    self.device
                )

                self.net.eval()
                with torch.no_grad():
                    pred_ratings = self.net(
                        self.valid_enc_graph,
                        self.valid_dec_graph,
                        None,
                        None,
                    )
                real_pred_ratings = (
                    torch.softmax(pred_ratings, dim=1)
                    * nd_positive_rating_values.view(1, -1)
                ).sum(dim=1)
                valid_rmse = ((real_pred_ratings - valid_truths) ** 2.0).mean().item()
                valid_rmse = np.sqrt(valid_rmse)
                logging_str += ",\t Val rmse={:.4f}".format(valid_rmse)

                if valid_rmse < best_valid_rmse:
                    best_valid_rmse = valid_rmse
                    no_better_valid = 0
                    best_iter = iter_idx
                    self.best_model_state_dict = self.net.state_dict()
                else:
                    no_better_valid += 1
                    if (
                        no_better_valid > self.train_early_stopping_patience
                        and learning_rate <= self.train_min_learning_rate
                    ):
                        logging.info(
                            "Early stopping threshold reached. Training stopped."
                        )
                        break

                    if no_better_valid > self.train_decay_patience:
                        new_learning_rate = max(
                            learning_rate * self.train_lr_decay_factor,
                            self.train_min_learning_rate,
                        )
                        if new_learning_rate < learning_rate:
                            learning_rate = new_learning_rate
                            logging.info("Changing LR to %g" % new_learning_rate)
                            for p in optimizer.param_groups:
                                p["lr"] = learning_rate
                            no_better_valid = 0
            if self.verbose == True:
                print(logging_str)

        print(
            "Best iter idx={}, Best valid rmse={:.4f}".format(
                best_iter, best_valid_rmse
            )
        )

        # train_enc_graph = train_enc_graph.to(self.device)
        # train_dec_graph = train_dec_graph")

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

        # (i, r) = self.train_set.user_data[1]

        self.net.load_state_dict(self.best_model_state_dict)
        self.net.eval()

        if self.train_set.is_unk_user(user_idx):
            raise ScoreException(
                "Can't make score prediction for (user_id=%d)" % user_idx
            )

        if item_idx is not None:
            if self.train_set.is_unk_item(item_idx):
                raise ScoreException(
                    "Can't make score prediction for (item_id=%d)" % item_idx
                )

        # user_id = (self.train_set.uir_tuple[0])[user_idx]

        if item_idx is None:
            if user_idx not in self.train_set.user_data.keys():
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )

            # i_list = list(self.train_set.item_ids)

            rating_pairs = (
                np.array(
                    [user_idx for _ in range(self.train_set.total_items)],
                    dtype=np.int32,
                ),
                np.array(range(self.train_set.total_items)),
            )

        else:
            # item_id = (list(self.train_set.item_ids))[item_idx]
            rating_pairs = (
                np.array([user_idx]),
                np.array([item_idx]),
            )

        ones = np.ones_like(rating_pairs[0])
        user_item_ratings_coo = sp.coo_matrix(
            (ones, rating_pairs),
            shape=(self.train_set.total_users, self.train_set.total_items),
            dtype=np.float32,
        )

        g = dgl.bipartite_from_scipy(
            user_item_ratings_coo, utype="_U", etype="_E", vtype="_V"
        )

        score_dec_graph = dgl.heterograph(
            {("user", "rate", "item"): g.edges()},
            num_nodes_dict={
                "user": self.train_set.total_users,
                "item": self.train_set.total_items,
            },
        )

        score_dec_graph = score_dec_graph.to(self.device)

        nd_positive_rating_values = torch.FloatTensor(self.rating_values).to(
            self.device
        )

        with torch.no_grad():
            pred_ratings = self.net(
                self.train_enc_graph,
                score_dec_graph,
                None,
                None,
            )
        real_pred_ratings = (
            torch.softmax(pred_ratings, dim=1) * nd_positive_rating_values.view(1, -1)
        ).sum(dim=1)

        return real_pred_ratings.cpu().numpy()
