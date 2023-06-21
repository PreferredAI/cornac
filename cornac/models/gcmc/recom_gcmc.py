import numpy as np
import torch
import dgl
from tqdm.auto import trange

from ..recommender import Recommender
from ...utils import get_rng
from ...utils.init_utils import xavier_uniform

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

        # Init params if provided
        self.init_params = {} if init_params is None else init_params

    def _init(self):
        rng = get_rng(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def fit(self, train_set, val_set=None):
        Recommender.fit(self, train_set, val_set)

        self._init()

        if self.trainable:
            self._fit_torch()

        return self
    
    def _generate_graph(self, train_set, n_users, n_items):
        print(train_set)

        data_dict = dict()
        num_nodes_dict = {"user": n_users, "item": n_items}

        
        for batch_u, batch_i, batch_r in self.train_set.uir_iter(
            shuffle=True,
        ):
            data_dict.update(
                {
                    ("user", str(batch_r), "movie"): (batch_u, batch_i),
                    ("item", "rev-%s" % str(batch_r), "user"): (batch_i, batch_u),
                }
            )
        
        return dgl.heterograph(data_dict, num_nodes_dict)


        
        

    def _fit_torch(self):
        n_users, n_items = self.train_set.num_users, self.train_set.num_items

        graph = self._generate_graph(self.train_set, n_users, n_items)

        print(graph)
        


        



              