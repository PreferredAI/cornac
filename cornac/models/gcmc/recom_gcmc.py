import numpy as np
import torch
import torch.nn as nn
import dgl
import scipy.sparse as sp
from tqdm.auto import trange

from ..recommender import Recommender
from ...utils import get_rng
from ...utils.init_utils import xavier_uniform

from gcmc import NeuralNetwork
from utils import get_optimizer

class GCMC(Recommender):
    def __init__(
        self,
        name="GCMC",
        k=10,
        max_iter=50,
        grad_iter=50,
        lambda_text=0.1,
        l2_reg=0.001,
        vocab_size=8000,
        learning_rate=0.01,
        optimizer="adam",
        activation_model="leaky",
        gcn_agg_units=500,
        gcn_out_units=75,
        gcn_dropout=0.7,
        gcn_agg_accum="sum",
        share_param=False,
        gen_r_num_basis_func=2,
        init_params=None,
        trainable=True,
        verbose=True,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)

        self.k = k
        self.lambda_text = lambda_text
        self.l2_reg = l2_reg
        self.grad_iter = grad_iter
        self.name = name
        self.max_iter = max_iter
        self.verbose = verbose
        self.seed = seed
        self.vocab_size = vocab_size
        self.activation_model = activation_model
        self.gcn_agg_units = gcn_agg_units
        self.gcn_out_units = gcn_out_units
        self.gcn_dropout = gcn_dropout
        self.gcn_agg_accum = gcn_agg_accum
        self.share_param = share_param
        self.gen_r_num_basis_func = gen_r_num_basis_func
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        
        # Init params if provided
        self.init_params = {} if init_params is None else init_params

    def _init(self):
        # rng = get_rng(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.device = torch.device("cuda" if torch.cuda_is_available() else "cpu")

        rating_values = self.train_set.uir_tuple[2] # rating array
        self.rating_values = np.unique(rating_values)
        print("rating values: ", rating_values)



    def fit(self, train_set, val_set=None):
        Recommender.fit(self, train_set, val_set)

        self._init()

        if self.trainable:
            self._fit_torch()

        return self
    
    def _apply_support(graph, rating_values, n_users, n_items, symm=True):
        def _calc_norm(x):
            x = x.numpy().astype("float32")
            x[x == 0.0] = np.inf
            x = torch.FloatTensor(1.0 / np.sqrt(x))
            return x.unsqueeze(1)

        user_ci = []
        user_cj = []
        item_ci = []
        item_cj = []
        print(graph)
        print(type(graph))
        print(rating_values)
        print(type(rating_values))
        print("===")
        for r in rating_values:
            print("Here")
            r = str(r).replace(".", "_")
            user_ci.append(graph["rev-%s" % r].in_degrees())
            item_ci.append(graph[r].in_degrees())
            print("reached here 1")
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
    
    def _generate_enc_graph(self, train_set, n_users, n_items, add_support=True):
        print(train_set)

        data_dict = dict()
        num_nodes_dict = {"user": n_users, "item": n_items}

        print(num_nodes_dict)

        
        for batch_u, batch_i, batch_r in self.train_set.uir_iter(
            shuffle=True,
        ):
            data_dict.update(
                {
                    ("user", str(batch_r), "item"): (batch_u, batch_i),
                    ("item", "rev-%s" % str(batch_r), "user"): (batch_i, batch_u),
                }
            )
        
        graph = dgl.heterograph(data_dict, num_nodes_dict)

        if add_support:
            print(self.rating_values)
            graph = self._apply_support(graph=graph, rating_values=self.rating_values, n_users=n_users, n_items=n_items)

        return graph      

    def _generate_dec_graph(self, train_set, n_users, n_items):
        csr_matrix = train_set.csr_matrix
        user_item_ratings_coo = csr_matrix.tocoo()
        
        g = dgl.bipartite_from_scipy(
            user_item_ratings_coo, utype="_U", etype="_E", vtype="_V"
        )

        return dgl.heterograph(
            {("user", "rate", "item"): g.edges()},
            num_nodes_dict={"user": n_users, "item": n_items},
        )

    def _fit_torch(self):
        self.n_users, self.n_items = self.train_set.num_users, self.train_set.num_items

        train_enc_graph = self._generate_enc_graph(self.train_set, self.n_users, self.n_items)
        train_dec_graph = self._generate_dec_graph(self.train_set, self.n_users, self.n_items)

        def _count_pairs(graph):
            pair_count = 0
            for r_val in self.rating_values:
                r_val = str(r_val).replace(".", "_")
                pair_count += graph.num_edges(str(r_val))
            return pair_count

        print("Train enc graph: {} users, {} items, {} pairs".format(
            train_enc_graph.num_nodes("user"),
            train_enc_graph.num_nodes("item"),
            _count_pairs(train_enc_graph),
        )) 

        print("Train dec graph: {} users, {} items, {} pairs".format(
            train_dec_graph.num_nodes("user"),
            train_dec_graph.num_nodes("item"),
            train_dec_graph.num_edges(),
        ))

        # Build Net
        net = NeuralNetwork()
        net = net.to(self.device)
        nd_positive_rating_values = torch.FloatTensor(
            self.rating_values
        ).to(self.device)
        rating_loss_net = nn.CrossEntropyLoss()
        learning_rate = self.learning_rate
        optimizer = get_optimizer(self.optimizer)(
            net.parameters(), lr=learning_rate
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
        return np.array([0.5])
        


        



              