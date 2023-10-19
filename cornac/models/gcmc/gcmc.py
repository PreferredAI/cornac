""" Fitting and scoring modules for GCMC """
import logging
import time
from collections import defaultdict
import torch
from torch import nn
import dgl
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from .nn_modules import NeuralNetwork
from .utils import get_optimizer, torch_net_info, torch_total_param_num


def _apply_support(graph, rating_values, data_set, symm=True):
    """Adds graph support. Returns DGLGraph."""

    def _calc_norm(val):
        val = val.numpy().astype("float32")
        val[val == 0.0] = np.inf
        val = torch.FloatTensor(1.0 / np.sqrt(val))
        return val.unsqueeze(1)

    n_users, n_items = data_set.total_users, data_set.total_items

    user_ci = []
    user_cj = []
    item_ci = []
    item_cj = []

    for rating in rating_values:
        rating = str(rating).replace(".", "_")
        user_ci.append(graph[f"rev-{rating}"].in_degrees())
        item_ci.append(graph[rating].in_degrees())

        if symm:
            user_cj.append(graph[rating].out_degrees())
            item_cj.append(graph[f"rev-{rating}"].out_degrees())
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


def _generate_enc_graph(data_set, add_support=False):
    """
    Generates encoding graph given a cornac data set

    Parameters
    ----------
    data_set : cornac.data.dataset.Dataset
        The data set as provided by cornac
    add_support : bool, optional
    """
    data_dict = {}
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
                ("item", f"rev-{str(rating)}", "user"): (rcol, rrow),
            }
        )

    graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

    # sanity check
    assert (
        len(data_set.uir_tuple[2])
        == sum([graph.num_edges(et) for et in graph.etypes]) // 2
    )

    if add_support:
        graph = _apply_support(
            graph=graph,
            rating_values=np.unique(rating_values),
            data_set=data_set,
        )

    return graph


def _generate_dec_graph(data_set):
    """
    Generates decoding graph given a cornac data set

    Parameters
    ----------
    data_set : cornac.data.dataset.Dataset
        The data set as provided by cornac

    Returns
    -------
    graph : dgl.heterograph
        Heterograph containing user-item edges and nodes
    """
    rating_pairs = data_set.uir_tuple[:2]
    ones = np.ones_like(rating_pairs[0])
    user_item_ratings_coo = sp.coo_matrix(
        (ones, rating_pairs),
        shape=(data_set.total_users, data_set.total_items),
        dtype=np.float32,
    )

    graph = dgl.bipartite_from_scipy(
        user_item_ratings_coo, utype="_U", etype="_E", vtype="_V"
    )

    return dgl.heterograph(
        {("user", "rate", "item"): graph.edges()},
        num_nodes_dict={"user": data_set.total_users, "item": data_set.total_items},
    )


def _generate_test_user_graph(user_idx, total_users, total_items):
    """
    Generates decoding graph given a cornac data set

    Parameters
    ----------
    data_set : cornac.data.dataset.Dataset
        The data set as provided by cornac

    Returns
    -------
    graph : dgl.heterograph
        Heterograph containing user-item edges and nodes
    """
    u_list = np.array([user_idx for _ in range(total_items)])
    i_list = np.array([item_idx for item_idx in range(total_items)])

    rating_pairs = (u_list, i_list)
    ones = np.ones_like(rating_pairs[0])
    user_item_ratings_coo = sp.coo_matrix(
        (ones, rating_pairs),
        shape=(total_users, total_items),
        dtype=np.float32,
    )

    graph = dgl.bipartite_from_scipy(
        user_item_ratings_coo, utype="_U", etype="_E", vtype="_V"
    )

    return dgl.heterograph(
        {("user", "rate", "item"): graph.edges()},
        num_nodes_dict={"user": total_users, "item": total_items},
    )


class Model:
    def __init__(
        self,
        activation_model,
        gcn_agg_units,
        gcn_out_units,
        gcn_dropout,
        gcn_agg_accum,
        share_param,
        gen_r_num_basis_func,
        verbose,
        seed,
    ):
        self.activation_model = activation_model
        self.gcn_agg_units = gcn_agg_units
        self.gcn_out_units = gcn_out_units
        self.gcn_dropout = gcn_dropout
        self.gcn_agg_accum = gcn_agg_accum
        self.share_param = share_param
        self.gen_r_num_basis_func = gen_r_num_basis_func

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.verbose = verbose
        if verbose:
            logging.basicConfig(level=logging.INFO)

        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def train(
        self,
        train_set,
        val_set,
        max_iter,
        learning_rate,
        optimizer,
        train_grad_clip,
        train_valid_interval,
        train_early_stopping_patience,
        train_min_learning_rate,
        train_decay_patience,
        train_lr_decay_factor,
    ):
        # Prepare data for training
        (
            rating_values,
            nd_positive_rating_values,
            train_dec_graph,
            valid_enc_graph,
            valid_dec_graph,
            train_labels,
            train_truths,
            valid_truths,
        ) = self._prepare_data(train_set, val_set)

        # Build Net
        self.net = NeuralNetwork(
            self.activation_model,
            rating_values,
            train_set.total_users,
            train_set.total_items,
            self.gcn_agg_units,
            self.gcn_out_units,
            self.gcn_dropout,
            self.gcn_agg_accum,
            self.gen_r_num_basis_func,
            self.share_param,
            self.device,
        ).to(self.device)

        optimizer = get_optimizer(optimizer)(self.net.parameters(), lr=learning_rate)
        rating_loss_net = nn.CrossEntropyLoss()

        self._train_model(
            rating_values,
            train_dec_graph,
            valid_enc_graph,
            valid_dec_graph,
            train_labels,
            train_truths,
            valid_truths,
            nd_positive_rating_values,
            rating_loss_net,
            max_iter,
            optimizer,
            learning_rate,
            train_grad_clip,
            train_valid_interval,
            train_early_stopping_patience,
            train_min_learning_rate,
            train_decay_patience,
            train_lr_decay_factor,
        )

    def _prepare_data(self, train_set, val_set):
        rating_values = train_set.uir_tuple[2]  # rating list
        rating_values = np.unique(rating_values)
        nd_positive_rating_values = torch.FloatTensor(rating_values).to(self.device)

        # Prepare Data
        def _generate_labels(ratings):
            labels = torch.LongTensor(np.searchsorted(rating_values, ratings)).to(
                self.device
            )
            return labels

        self.train_enc_graph = _generate_enc_graph(train_set, add_support=True)
        train_dec_graph = _generate_dec_graph(train_set)

        train_labels = _generate_labels(train_set.uir_tuple[2])
        train_truths = torch.FloatTensor(train_set.uir_tuple[2]).to(self.device)

        def _count_pairs(graph):
            pair_count = 0
            for r_val in rating_values:
                r_val = str(r_val).replace(".", "_")
                pair_count += graph.num_edges(str(r_val))
            return pair_count

        logging.info(
            "Train enc graph: %s users, %s items, %s pairs",
            self.train_enc_graph.num_nodes("user"),
            self.train_enc_graph.num_nodes("item"),
            _count_pairs(self.train_enc_graph),
        )

        logging.info(
            "Train dec graph: %s users, %s items, %s pairs",
            train_dec_graph.num_nodes("user"),
            train_dec_graph.num_nodes("item"),
            train_dec_graph.num_edges(),
        )

        valid_enc_graph = self.train_enc_graph
        if val_set:
            valid_dec_graph = _generate_dec_graph(val_set)
            valid_truths = torch.FloatTensor(val_set.uir_tuple[2]).to(self.device)
            logging.info(
                "Valid enc graph: %s users, %s items, %s pairs",
                valid_enc_graph.num_nodes("user"),
                valid_enc_graph.num_nodes("item"),
                _count_pairs(valid_enc_graph),
            )
            logging.info(
                "Valid dec graph: %s users, %s items, %s pairs",
                valid_dec_graph.num_nodes("user"),
                valid_dec_graph.num_nodes("item"),
                valid_dec_graph.num_edges(),
            )
        else:
            valid_dec_graph = None

        return (
            rating_values,
            nd_positive_rating_values,
            train_dec_graph,
            valid_enc_graph,
            valid_dec_graph,
            train_labels,
            train_truths,
            valid_truths,
        )

    def _train_model(
        self,
        rating_values,
        train_dec_graph,
        valid_enc_graph,
        valid_dec_graph,
        train_labels,
        train_truths,
        valid_truths,
        nd_positive_rating_values,
        rating_loss_net,
        max_iter,
        optimizer,
        learning_rate,
        train_grad_clip,
        train_valid_interval,
        train_early_stopping_patience,
        train_min_learning_rate,
        train_decay_patience,
        train_lr_decay_factor,
    ):
        # initialize loss variables
        best_valid_rmse = np.inf
        no_better_valid = 0
        best_iter = -1
        count_rmse = 0
        count_num = 0
        count_loss = 0

        self.train_enc_graph = self.train_enc_graph.int().to(self.device)
        train_dec_graph = train_dec_graph.int().to(self.device)
        if valid_dec_graph:
            valid_enc_graph = self.train_enc_graph
            valid_dec_graph = valid_dec_graph.int().to(self.device)

        logging.info("Training Started!")
        dur = []
        for iter_idx in tqdm(
            range(1, max_iter),
            desc="Training",
            unit="iter",
            disable=self.verbose,
        ):
            if iter_idx > 3:
                time_start = time.time()

            self.net.train()
            pred_ratings = self.net(self.train_enc_graph, train_dec_graph)
            loss = rating_loss_net(pred_ratings, train_labels).mean()
            count_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), train_grad_clip)
            optimizer.step()

            if iter_idx > 3:
                dur.append(time.time() - time_start)

            if iter_idx == 1:
                logging.info(
                    "Total # params of net: %s", torch_total_param_num(self.net)
                )
                logging.info(torch_net_info(self.net))

            real_pred_ratings = (
                torch.softmax(pred_ratings, dim=1)
                * nd_positive_rating_values.view(1, -1)
            ).sum(dim=1)

            rmse = ((real_pred_ratings - train_truths) ** 2).sum()
            count_rmse += rmse.item()
            count_num += len(train_truths)

            logging_str = (
                f"Epoch: {iter_idx}\t"
                + f"loss: {count_loss / iter_idx:.4f}\t"
                + f"rmse:{count_rmse / count_num:.4f}\t"
                + f"time:{np.average(dur):.4f}\t"
            )
            count_rmse = 0
            count_num = 0

            if valid_dec_graph and iter_idx & train_valid_interval == 0:
                nd_positive_rating_values = torch.FloatTensor(rating_values).to(
                    self.device
                )

                self.net.eval()
                with torch.no_grad():
                    pred_ratings = self.net(
                        valid_enc_graph,
                        valid_dec_graph,
                        None,
                        None,
                    )
                real_pred_ratings = (
                    torch.softmax(pred_ratings, dim=1)
                    * nd_positive_rating_values.view(1, -1)
                ).sum(dim=1)
                valid_rmse = ((real_pred_ratings - valid_truths) ** 2.0).mean().item()
                valid_rmse = np.sqrt(valid_rmse)
                logging_str += f"Val rmse={valid_rmse:.4f}"

                if valid_rmse < best_valid_rmse:
                    best_valid_rmse = valid_rmse
                    no_better_valid = 0
                    best_iter = iter_idx
                    best_model_state_dict = self.net.state_dict()
                else:
                    no_better_valid += 1
                    if (
                        no_better_valid > train_early_stopping_patience
                        and learning_rate <= train_min_learning_rate
                    ):
                        logging.info(
                            "Early stopping threshold reached." + "\nTraining stopped."
                        )
                        break

                    if no_better_valid > train_decay_patience:
                        new_learning_rate = max(
                            learning_rate * train_lr_decay_factor,
                            train_min_learning_rate,
                        )
                        if new_learning_rate < learning_rate:
                            learning_rate = new_learning_rate
                            logging.info("Changing LR to %s", new_learning_rate)
                            for param in optimizer.param_groups:
                                param["lr"] = learning_rate
                            no_better_valid = 0

            logging.info(logging_str)

        if valid_dec_graph:
            logging.info(
                "Best iter idx=%s, Best valid rmse=%.4f", best_iter, best_valid_rmse
            )

            # load best model
            self.net.load_state_dict(best_model_state_dict)

    def predict(self, test_set):
        """
        Processes test set and returns dictionary containing
        user-item idx as key and score as value.

        Parameters
        ----------
        test_set : cornac.data.dataset.Dataset
            The data set as provided by cornac
            Returns

        Returns
        -------
        u_i_rating_dict : dict
            Dictionary containing '{user_idx}-{item_idx}' as key
            and {score} as value.
        """
        test_dec_graph = _generate_dec_graph(test_set)
        test_dec_graph = test_dec_graph.int().to(self.device)

        self.net.eval()

        with torch.no_grad():
            pred_ratings = self.net(self.train_enc_graph, test_dec_graph)

        test_rating_values = test_set.uir_tuple[2]
        test_rating_values = np.unique(test_rating_values)

        nd_positive_rating_values = torch.FloatTensor(test_rating_values).to(
            self.device
        )

        test_pred_ratings = (
            torch.softmax(pred_ratings, dim=1) * nd_positive_rating_values.view(1, -1)
        ).sum(dim=1)

        test_pred_ratings = test_pred_ratings.cpu().numpy()

        uid_list = test_set.uir_tuple[0]
        uid_list = np.unique(uid_list)

        u_list = np.array([user_idx for _ in range(test_set.total_items) for user_idx in uid_list])
        i_list = np.array([item_idx for item_idx in range(test_set.total_items) for _ in uid_list])

        u_list = u_list.tolist()
        i_list = i_list.tolist()

        u_i_rating_dict = {
            f"{u_list[idx]}-{i_list[idx]}": rating
            for idx, rating in enumerate(test_pred_ratings)
        }
        return u_i_rating_dict

    def predict_one(self, train_set, user_idx):
        """
        Processes single user_idx from test set and returns numpy list of scores
        for all items.

        Parameters
        ----------
        train_set : cornac.data.dataset.Dataset
            The train set as provided by cornac

        Returns
        -------
        test_pred_ratings : numpy.array
            Numpy array containing all ratings for the given user_idx.
        """
        test_dec_graph = _generate_test_user_graph(user_idx, train_set.total_users, train_set.total_items)
        test_dec_graph = test_dec_graph.int().to(self.device)

        self.net.eval()

        with torch.no_grad():
            pred_ratings = self.net(self.train_enc_graph, test_dec_graph)

        test_rating_values = train_set.uir_tuple[2]
        test_rating_values = np.unique(test_rating_values)

        nd_positive_rating_values = torch.FloatTensor(test_rating_values).to(
            self.device
        )

        test_pred_ratings = (
            torch.softmax(pred_ratings, dim=1) * nd_positive_rating_values.view(1, -1)
        ).sum(dim=1)

        return test_pred_ratings.cpu().numpy()
