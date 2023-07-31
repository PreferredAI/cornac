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


def fit_torch(self, train_set, val_set):
    """
    Converts training + validation sets into graphs and fits into model

    Parameters
    ----------
    train_set : cornac.data.dataset.Dataset
        The training set as provided by cornac
    val_set : cornac.data.dataset.Dataset
        The validation set as provided by cornac. Could be None.
    """
    self.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if self.seed is not None:
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    if self.verbose:
        logging.basicConfig(level=logging.INFO)

    rating_values = train_set.uir_tuple[2]  # rating list
    rating_values = np.unique(rating_values)

    # Prepare Data
    def _generate_labels(ratings):
        labels = torch.LongTensor(
            np.searchsorted(rating_values, ratings)
        ).to(self.device)
        return labels

    self.train_enc_graph = generate_enc_graph(
        train_set, add_support=True
    )
    train_dec_graph = generate_dec_graph(train_set)

    train_labels = _generate_labels(train_set.uir_tuple[2])
    train_truths = torch.FloatTensor(
        train_set.uir_tuple[2]
    ).to(self.device)

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

    if val_set is not None:
        valid_enc_graph = self.train_enc_graph
        valid_dec_graph = generate_dec_graph(val_set)

        valid_truths = torch.FloatTensor(
            val_set.uir_tuple[2]
        ).to(self.device)

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
    )
    self.net = self.net.to(self.device)
    nd_positive_rating_values = torch.FloatTensor(rating_values).to(
        self.device
    )
    rating_loss_net = nn.CrossEntropyLoss()
    learning_rate = self.learning_rate
    optimizer = get_optimizer(self.optimizer)(
        self.net.parameters(), lr=learning_rate
    )
    print("NN Loading Complete!")

    # declare the loss information
    best_valid_rmse = np.inf
    no_better_valid = 0
    best_iter = -1
    count_rmse = 0
    count_num = 0
    count_loss = 0

    self.train_enc_graph = self.train_enc_graph.int().to(self.device)
    train_dec_graph = train_dec_graph.int().to(self.device)

    if self.val_set is not None:
        valid_enc_graph = valid_enc_graph.int().to(self.device)
        valid_dec_graph = valid_dec_graph.int().to(self.device)

    print("Training Started!")
    dur = []
    for iter_idx in tqdm(
        range(1, self.max_iter),
        desc="Training",
        unit="iter",
        disable=self.verbose,
    ):
        if iter_idx > 3:
            time_start = time.time()
        self.net.train()
        pred_ratings = self.net(
            self.train_enc_graph,
            train_dec_graph,
            None,
            None,
        )
        loss = rating_loss_net(pred_ratings, train_labels).mean()
        count_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.net.parameters(),
            self.train_grad_clip
        )
        optimizer.step()

        if iter_idx > 3:
            dur.append(time.time() - time_start)

        if iter_idx == 1:
            logging.info(
                "Total # params of net: %s",
                torch_total_param_num(self.net)
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
            f"Epoch: {iter_idx}\t" +
            f"loss: {count_loss / iter_idx:.4f}\t" +
            f"rmse:{count_rmse / count_num:.4f}\t" +
            f"time:{np.average(dur):.4f}\t"
        )
        count_rmse = 0
        count_num = 0

        if (
            self.val_set is not None and
            iter_idx & self.train_valid_interval == 0
        ):
            nd_positive_rating_values = torch.FloatTensor(
                rating_values
            ).to(self.device)

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
            valid_rmse = (
                (real_pred_ratings - valid_truths) ** 2.0
            ).mean().item()
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
                    no_better_valid > self.train_early_stopping_patience
                    and learning_rate <= self.train_min_learning_rate
                ):
                    logging.info(
                        "Early stopping threshold reached."
                        + "\nTraining stopped."
                    )
                    break

                if no_better_valid > self.train_decay_patience:
                    new_learning_rate = max(
                        learning_rate * self.train_lr_decay_factor,
                        self.train_min_learning_rate,
                    )
                    if new_learning_rate < learning_rate:
                        learning_rate = new_learning_rate
                        logging.info(
                            "Changing LR to %s",
                            new_learning_rate
                        )
                        for param in optimizer.param_groups:
                            param["lr"] = learning_rate
                        no_better_valid = 0

        logging.info(logging_str)

    if self.val_set is not None:
        logging.info(
            "Best iter idx=%s, Best valid rmse=%.4f",
            best_iter,
            best_valid_rmse
        )

        # load best model
        self.net.load_state_dict(best_model_state_dict)


def generate_enc_graph(data_set, add_support=False):
    """
    Generates encoding graph given a cornac data set

    Parameters
    ----------
    data_set : cornac.data.dataset.Dataset
        The data set as provided by cornac
    add_support : bool, optional
    """
    data_dict = {}
    num_nodes_dict = {
        "user": data_set.total_users,
        "item": data_set.total_items
    }
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
        graph = apply_support(
            graph=graph,
            rating_values=np.unique(rating_values),
            data_set=data_set,
        )

    return graph


def generate_dec_graph(data_set):
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
        num_nodes_dict={
            "user": data_set.total_users,
            "item": data_set.total_items
        },
    )


def apply_support(
        graph,
        rating_values,
        data_set,
        symm=True
):
    """ Adds graph support. Returns DGLGraph. """
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


def process_test_set(self, test_set):
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
    test_dec_graph = generate_dec_graph(test_set)
    test_dec_graph = test_dec_graph.int().to(self.device)

    self.net.eval()

    with torch.no_grad():
        pred_ratings = self.net(
            self.train_enc_graph,
            test_dec_graph,
            None,
            None,
        )

    test_rating_values = test_set.uir_tuple[2]
    test_rating_values = np.unique(test_rating_values)

    nd_positive_rating_values = torch.FloatTensor(test_rating_values).to(
        self.device
    )

    test_pred_ratings = (
        torch.softmax(pred_ratings, dim=1)
        * nd_positive_rating_values.view(1, -1)
    ).sum(dim=1)

    test_pred_ratings = test_pred_ratings.cpu().numpy()

    (u_list, i_list, _) = test_set.uir_tuple

    u_list = u_list.tolist()
    i_list = i_list.tolist()

    u_i_rating_dict = defaultdict(self.default_score)

    for idx, rating in enumerate(test_pred_ratings):
        u_i_rating_dict[
            str(u_list[idx]) + "-" + str(i_list[idx])
        ] = rating
    return u_i_rating_dict


def get_score(self, user_idx, item_idx):
    """
    Obtains score given a user_idx and item_idx
    (item_idx may be None for ranking evaluations)

    Dictionary containing {user_idx}-{item_idx} keys
    are generated as in process_test_set function

    Access dictionary to obtain score for a given user_idx and item_idx.

    Parameters
    ----------
    user_idx : int
        Index of user to obtain score for
    item_idx : int
        Index of item to obtain score for

    Returns
    -------
    score_list : list
        List containing a single float value of score
        if both user_idx and item_idx is provided.
        List containing corresponding float values of
        all item scores if only user_idx is provided.
    """
    if item_idx is None:
        # Return scores of all items for a given user
        # - If item does not exist in test_set, we provide a default score
        #   (as set in default_dict initialisation)
        return [
            self.u_i_rating_dict[
                str(user_idx) + "-" + str(idx)
            ] for idx in range(self.train_set.total_items)
        ]
    # Return score of known user/item
    return [self.u_i_rating_dict[str(user_idx) + "-" + str(item_idx)]]
