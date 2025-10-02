import collections
import os
import pickle
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy

from ..recommender import Recommender
from ...data import Dataset


class HypAR(Recommender):
    """
        HypAR: Hypergraph with Attention on Review. This model is from the paper "Hypergraph with Attention on Reviews
        for explainable recommendation", by Theis E. Jendal, Trung-Hoang Le, Hady W. Lauw, Matteo Lissandrini,
        Peter Dolog, and Katja Hose.
        ECIR 2024: https://doi.org/10.1007/978-3-031-56027-9_14

        Parameters
        ----------
        name: str, default: 'HypAR'
            Name of the model.
        use_cuda: bool, default: False
            Whether to use cuda.
        stemming: bool, default: True
            Whether to use stemming.
        batch_size: int, default: 128
            Batch size.
        num_workers: int, default: 0
            Number of workers for dataloader.
        num_epochs: int, default: 10
            Number of epochs.
        early_stopping: int, default: 10
            Early stopping.
        eval_interval: int, default: 1
            Evaluation interval, i.e., how often to evaluate on the validation set.
        learning_rate: float, default: 0.1
            Learning rate.
        weight_decay: float, default: 0
            Weight decay.
        node_dim: int, default: 64
            Dimension of learned and hidden layers.
        num_heads: int, default: 3
            Number of attention heads.
        fanout: int, default: 5
            Fanout for sampling.
        non_linear: bool, default: True
            Whether to use non-linear activation function.
        model_selection: str, default: 'best'
            Model selection method, i.e., whether to use the best model or the last model.
        objective: str, default: 'ranking'
            Objective, i.e., whether to use ranking or rating.
        review_aggregator: str, default: 'narre'
            Review aggregator, i.e., how to aggregate reviews.
        predictor: str, default: 'narre'
            Predictor, i.e., how to predict ratings.
        preference_module: str, default: 'lightgcn'
            Preference module, i.e., how to model preferences.
        combiner: str, default: 'add'
            Combiner, i.e., how to combine embeddings.
        graph_type: str, default: 'aos'
            Graph type, i.e., which nodes to include in hypergraph. Aspects, opinions and sentiment.
        num_neg_samples: int, default: 50
            Number of negative samples to use for ranking.
        layer_dropout: float, default: None
            Dropout for node and review embeddings.
        attention_dropout: float, default: .2
            Dropout for attention.
        user_based: bool, default: True
            Whether to use user-based or item-based.
        verbose: bool, default: True
            Whether to print information.
        index: int, default: 0
            Index for saving results, i.e., if hyparparameter tuning.
        out_path: str, default: None
            Path to save graphs, embeddings and similar.
        learn_explainability: bool, default: False
            Whether to learn explainability.
        learn_method: str, default: 'transr'
            Learning method, i.e., which method to use explainability learning.
        learn_weight: float, default: 1.
            Weight for explainability learning loss.
        embedding_type: str, default: 'ao_embeddings'
            Type of embeddings to use, i.e., whether to use prelearned embeddings or not.
        debug: bool, default: False
            Whether to use debug mode as errors might be thrown by dataloaders when debugging.
        """
    def __init__(self,
                 name='HypAR',
                 use_cuda=False,
                 stemming=True,
                 batch_size=128,
                 num_workers=0,
                 num_epochs=10,
                 early_stopping=10,
                 eval_interval=1,
                 learning_rate=0.1,
                 weight_decay=0,
                 node_dim=64,
                 num_heads=3,
                 fanout=5,
                 non_linear=True,
                 model_selection='best',
                 objective='ranking',
                 review_aggregator='narre',
                 predictor='narre',
                 preference_module='lightgcn',
                 combiner='add',
                 graph_type='aos',
                 num_neg_samples=50,
                 layer_dropout=None,
                 attention_dropout=.2,
                 user_based=True,
                 verbose=True,
                 index=0,
                 out_path=None,
                 learn_explainability=False,
                 learn_method='transr',
                 learn_weight=1.,
                 embedding_type='ao_embeddings',
                 debug=False,
                 ):
        super().__init__(name)
        # Default values
        if layer_dropout is None:
            layer_dropout = 0.  # node embedding dropout, review embedding dropout

        # CUDA
        self.use_cuda = use_cuda
        self.device = 'cuda' if use_cuda else 'cpu'

        # Parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.early_stopping = early_stopping
        self.eval_interval = eval_interval
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.node_dim = node_dim
        self.num_heads = num_heads
        self.fanout = fanout
        self.non_linear = non_linear
        self.model_selection = model_selection
        self.objective = objective
        self.review_aggregator = review_aggregator
        self.predictor = predictor
        self.preference_module = preference_module
        self.combiner = combiner
        self.graph_type = graph_type
        self.num_neg_samples = num_neg_samples
        self.layer_dropout = layer_dropout
        self.attention_dropout = attention_dropout
        self.stemming = stemming
        self.learn_explainability = learn_explainability
        self.learn_method = learn_method
        self.learn_weight = learn_weight
        self.embedding_type = embedding_type

        # Method
        self.node_review_graph = None
        self.review_graphs = {}
        self.train_graph = None
        self.ui_graph = None
        self.model = None
        self.n_items = 0
        self.n_relations = 0
        self.ntype_ranges = None
        self.node_filter = None
        self.sid_aos = None
        self.aos_tensor = None

        # Misc
        self.user_based = user_based
        self.verbose = verbose
        self.debug = debug
        self.index = index
        self.out_path = out_path

        # assertions
        assert objective == 'ranking' or objective == 'rating', f'This method only supports ranking or rating, ' \
                                                                f'not {objective}.'
        if early_stopping is not None:
            assert early_stopping % eval_interval == 0, 'interval should be a divisor of early stopping value.'

    def _create_graphs(self, train_set: Dataset, graph_type='aos'):
        """
        Create graphs required for training and returns all relevant data for future computations.
        Parameters
        ----------
        train_set: Dataset
        graph_type: str, which can contain a,o and s, where a is aspect, o is opinion and s is sentiment. E.g., if
        a or o, then aspect and opinion are included, if s, then splitting on sentiment is included.

        Returns
        -------
        num nodes, num node types, num items, train graph, hyper edges, node review graph, type ranges, sid to aos, and
        aos triple list.
        """
        import dgl
        import torch
        from tqdm import tqdm
        from .dgl_utils import generate_mappings

        sentiment_modality = train_set.sentiment
        n_users = len(train_set.uid_map)
        n_items = len(train_set.iid_map)

        # Group and prune aspects and opinions
        _, _, _, _, _, _, a2a, o2o = generate_mappings(train_set.sentiment, 'a', get_ao_mappings=True)

        # Get num and depending on graphtype, calculate tot num of embeddings
        n_aspects = max(a2a.values()) + 1 if self.stemming else len(sentiment_modality.aspect_id_map)
        n_opinions = max(o2o.values()) + 1 if self.stemming else len(sentiment_modality.opinion_id_map)
        n_nodes = n_users + n_items
        n_types = 4
        if 'a' in graph_type:
            n_nodes += n_aspects
            n_types += 1
        if 'o' in graph_type:
            n_nodes += n_opinions
            n_types += 1

        # Map users to review ids.
        user_item_review_map = {(uid + n_items, iid): rid for uid, irid in sentiment_modality.user_sentiment.items()
                                for iid, rid in irid.items()}

        # Initialize relevant lists
        review_edges = []
        ratings = []
        if 's' in graph_type:
            hyper_edges = {'p': [], 'n': []}
        else:
            hyper_edges = {'n': []}
        sent_mapping = {-1: 'n', 1: 'p'}

        # Create review edges and ratings
        sid_map = {sid: i for i, sid in enumerate(train_set.sentiment.sentiment)}
        for uid, isid in tqdm(sentiment_modality.user_sentiment.items(), desc='Creating review graphs',
                              total=len(sentiment_modality.user_sentiment), disable=not self.verbose):
            uid += n_items  # Shift to user node id.

            for iid, sid in isid.items():
                # Sid is used as edge id, i.e., each sid represent a single review.
                first_sentiment = {'p': True, 'n': True}  # special handling for initial sentiment.
                review_edges.extend([[sid, uid], [sid, iid]])  # Add u/i to review aggregation
                ratings.extend([train_set.matrix[uid - n_items, iid]] * 2)  # Add to rating list
                aos = sentiment_modality.sentiment[sid]  # get aspects, opinions and sentiments for review.
                for aid, oid, s in aos:
                    # Map sentiments if using else, use default (n).
                    if 's' in graph_type:
                        sent = sent_mapping[s]
                    else:
                        sent = 'n'

                    # If stemming and pruning data, use simplified id (i.e., mapping)
                    if self.stemming:
                        aid = a2a[aid]
                        oid = o2o[oid]

                    # Add to hyper edges, i.e., connect user, item, aspect and opinion to sentiment.
                    if first_sentiment[sent]:
                        hyper_edges[sent].extend([(iid, sid), (uid, sid)])
                        first_sentiment[sent] = False

                    # Shift aspect and opinion ids to node id.
                    aid += n_items + n_users
                    oid += n_items + n_users

                    # If using aspect/opinion, add to hyper edges.
                    if 'a' in graph_type:
                        hyper_edges[sent].append((aid, sid))
                        oid += n_aspects  # Shift opinion id to correct node id if using both aspect and opinion.
                    if 'o' in graph_type:
                        hyper_edges[sent].append((oid, sid))

        # Convert to tensor
        for k, v in hyper_edges.items():
            hyper_edges[k] = torch.LongTensor(v).T

        # Create training graph, i.e. user to item graph.
        edges = [(uid + n_items, iid, train_set.matrix[uid, iid]) for uid, iid in zip(*train_set.matrix.nonzero())]
        t_edges = torch.LongTensor(edges).T
        train_graph = dgl.graph((t_edges[0], t_edges[1]))
        train_graph.edata['sid'] = torch.LongTensor([user_item_review_map[(u, i)] for (u, i, r) in edges])
        train_graph.edata['label'] = t_edges[2].to(torch.float)

        # Create user/item to review graph.
        edges = torch.LongTensor(review_edges).T
        node_review_graph = dgl.heterograph({('review', 'part_of', 'node'): (edges[0], edges[1])})

        # Assign edges node_ids s.t. an edge from user to review has the item nid its about and reversely.
        node_review_graph.edata['nid'] = torch.LongTensor(node_review_graph.num_edges())
        _, v, eids = node_review_graph.edges(form='all')
        node_review_graph.edata['nid'][eids % 2 == 0] = v[eids % 2 == 1]
        node_review_graph.edata['nid'][eids % 2 == 1] = v[eids % 2 == 0]

        # Scale ratings with denominator if not integers. I.e., if .25 multiply by 4.
        # A mapping from frac to int. Thus if scale is from 1-5 and in .5 increments, will be converted to 1-10.
        denominators = [e.as_integer_ratio()[1] for e in ratings]
        i = 0
        while any(d != 1 for d in denominators):
            ratings = ratings * max(denominators)
            denominators = [e.as_integer_ratio()[1] for e in ratings]
            i += 1
            assert i < 100, 'Tried to convert ratings to integers but took to long.'

        node_review_graph.edata['r_type'] = torch.LongTensor(ratings) - 1

        # Define ntype ranges
        ntype_ranges = {'item': (0, n_items), 'user': (n_items, n_items + n_users)}
        start = n_items + n_users
        if 'a' in graph_type:
            ntype_ranges['aspect'] = (start, start + n_aspects)
            start += n_aspects
        if 'o' in graph_type:
            ntype_ranges['opinion'] = (start, start + n_opinions)

        # Get all aos triples
        sid_aos = []
        for sid in range(max(train_set.sentiment.sentiment) + 1):
            aoss = train_set.sentiment.sentiment.get(sid, [])
            sid_aos.append([(a2a[a] + n_items + n_users, o2o[o] + n_users + n_items + n_aspects, 0 if s == -1 else 1)
                            for a, o, s in aoss])


        aos_list = sorted({aos for aoss in sid_aos for aos in aoss})
        aos_id = {aos: i for i, aos in enumerate(aos_list)}
        sid_aos = [torch.LongTensor([aos_id[aos] for aos in aoss]) for aoss in sid_aos]

        return n_nodes, n_types, n_items, train_graph, hyper_edges, node_review_graph, ntype_ranges, sid_aos, aos_list

    def _flock_wrapper(self, func, fname, *args, rerun=False, **kwargs):
        """
        Wrapper for loading and saving data without accidental overrides and dual computation when running in parallel.
        If file exists, load, else run function and save.
        Parameters
        ----------
        func: function
            Function to run.
        fname: str
            File name to save/load.
        args: list
            Arguments to function.
        rerun: bool, default: False
            If true, rerun function.
        kwargs: dict
            Keyword arguments to function.

        Returns
        -------
        Data from function.
        """
        from filelock import FileLock

        fpath = os.path.join(self.out_path, fname)
        lock_fpath = os.path.join(self.out_path, fname + '.lock')

        with FileLock(lock_fpath):
            if not rerun and os.path.exists(fpath):
                with open(fpath, 'rb') as f:
                    data = pickle.load(f)
            else:
                data = func(*args, **kwargs)
                with open(fpath, 'wb') as f:
                    pickle.dump(data, f)

        return data

    def _graph_wrapper(self, train_set, graph_type, *args):
        """
        Wrapper for creating graphs and converting to correct format.
        Assigns values to self, such as train graph, review graphs, node review graph, and ntype ranges.
        Define self.node_filter based on type ranges.
        Parameters
        ----------
        train_set: Dataset
            Dataset to use for graph construction
        graph_type: str
            Which graph to create. Can contain a, o and s, where a is aspect, o is opinion and s is sentiment.
        args: list
            Additional arguments to graph creation function.

        Returns
        -------
        Num nodes, num types, sid to aos mapping, list of aos triples.
        """
        import dgl.sparse as dglsp
        import torch

        # Load graph data
        fname = f'graph_{graph_type}_data.pickle'
        data = self._flock_wrapper(self._create_graphs, fname, train_set, graph_type, *args, rerun=False)

        # Expland data and assign to self
        n_nodes, n_types, self.n_items, self.train_graph, self.review_graphs, self.node_review_graph, \
            self.ntype_ranges, sid_aos, aos_list = data

        # Convert data to sparse matrices and assign to self.
        # Review graphs is dict with positive/negative sentiment (possibly).
        shape = torch.cat(list(self.review_graphs.values()), dim=-1).max(-1)[0] + 1
        for k, edges in self.review_graphs.items():
            H = dglsp.spmatrix(
                torch.unique(edges, dim=1), shape=shape.tolist()
            ).coalesce()
            assert (H.val == 1).all()
            self.review_graphs[k] = H.to(self.device)

        self.node_filter = lambda t, nids: (nids >= self.ntype_ranges[t][0]) * (nids < self.ntype_ranges[t][1])
        return n_nodes, n_types, sid_aos, aos_list

    def _ao_embeddings(self, train_set):
        """
        Learn aspect and opinion embeddings using sentence-transformers.
        Parameters
        ----------
        train_set: dataset
            Dataset to use for learning embeddings.
        Returns
        -------
            Aspect and opinion embeddings, and the sentence-transformers model.
        """
        from .dgl_utils import generate_mappings, stem_fn
        from nltk.tokenize import word_tokenize
        from tqdm import tqdm
        import numpy as np
        from sentence_transformers import SentenceTransformer

        sentiment = train_set.sentiment
        preprocess_fn = stem_fn

        # Prepare aspect and opinion terms
        a_old_new_map = {a: preprocess_fn(a) for a in sentiment.aspect_id_map}
        o_old_new_map = {o: preprocess_fn(o) for o in sentiment.opinion_id_map}
        _, _, _, _, _, _, a2a, o2o = generate_mappings(train_set.sentiment, 'a', get_ao_mappings=True)

        # Load sentence-transformers model (use a small, fast model by default)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding_dim = model.get_sentence_embedding_dimension()

        # Encode all unique aspect and opinion terms
        aspect_terms = [a_old_new_map[a] for a in sentiment.aspect_id_map]
        opinion_terms = [o_old_new_map[o] for o in sentiment.opinion_id_map]
        aspect_vecs = model.encode(aspect_terms, show_progress_bar=self.verbose)
        opinion_vecs = model.encode(opinion_terms, show_progress_bar=self.verbose)

        # Initialize embeddings
        a_embeddings = np.zeros((len(set(a2a.values())), embedding_dim))
        o_embeddings = np.zeros((len(set(o2o.values())), embedding_dim))

        # Assign embeddings to correct aspect and opinion
        for idx, a in enumerate(sentiment.aspect_id_map):
            nid = a2a[sentiment.aspect_id_map[a]]
            a_embeddings[nid] = aspect_vecs[idx]
        for idx, o in enumerate(sentiment.opinion_id_map):
            nid = o2o[sentiment.opinion_id_map[o]]
            o_embeddings[nid] = opinion_vecs[idx]

        return a_embeddings, o_embeddings, model

    def _normalize_embedding(self, embedding):
        """
        Normalize embeddings using standard scaler.
        Parameters
        ----------
        embedding: np.array
            Embedding to normalize.

        Returns
        -------
        Normalized embedding and scaler.
        """
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(embedding)
        return scaler.transform(embedding), scaler

    def _learn_initial_ao_embeddings(self, train_set):
        """
        Learn initial aspect and opinion embeddings.
        Parameters
        ----------
        train_set: Dataset
            Dataset to use for learning embeddings.

        Returns
        -------
        Aspect and opinion embeddings as torch tensors.
        """

        import torch

        ao_fname = 'ao_embeddingsv2.pickle'
        a_fname = 'aspect_embeddingsv2.pickle'
        o_fname = 'opinion_embeddingsv2.pickle'

        # Get embeddings and store result
        a_embeddings, o_embeddings, _ = self._flock_wrapper(self._ao_embeddings, ao_fname, train_set)

        # Scale embeddings and store results. Function returns scaler, which is not needed, but required if new data is
        # added.
        a_embeddings, _ = self._flock_wrapper(self._normalize_embedding, a_fname, a_embeddings)
        o_embeddings, _ = self._flock_wrapper(self._normalize_embedding, o_fname, o_embeddings)

        return torch.tensor(a_embeddings), torch.tensor(o_embeddings)

    def fit(self, train_set: Dataset, val_set=None):
        import os
        import torch
        from .lightgcn import construct_graph
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        # Initialize self variables
        super().fit(train_set, val_set)

        # Create graphs and assigns to self (e.g., see self.review_graphs).
        n_nodes, self.n_relations, self.sid_aos, self.aos_list = self._graph_wrapper(train_set,
                                                                                     self.graph_type)

        # If using learned ao embeddings, learn and assign to kwargs
        kwargs = {}
        if self.embedding_type == 'ao_embeddings':
            a_embs, o_embs = self._learn_initial_ao_embeddings(train_set)

            emb = []
            if 'a' in self.graph_type:
                emb.append(a_embs)
            if 'o' in self.graph_type:
                emb.append(o_embs)

            if len(emb):
                kwargs['ao_embeddings'] = torch.cat(emb).to(self.device).to(torch.float32)
                n_nodes -= kwargs['ao_embeddings'].size(0)
            else:
                kwargs['ao_embeddings'] = torch.zeros((0, 0))

        self.n_relations = 0

        # Construct user-item graph used by lightgcn
        self.ui_graph = construct_graph(train_set, self.num_users, self.num_items)

        # create model
        from .hypar import Model

        self.model = Model(self.ui_graph, n_nodes, self.review_aggregator,
                           self.predictor, self.node_dim, self.review_graphs, self.num_heads, [self.layer_dropout] * 2,
                           self.attention_dropout, self.preference_module, self.use_cuda, combiner=self.combiner,
                           aos_predictor=self.learn_method, non_linear=self.non_linear,
                           embedding_type=self.embedding_type,
                           **kwargs)

        self.model.reset_parameters()

        if self.verbose:
            print(f'Number of trainable parameters: {sum(p.numel() for p in self.model.parameters())}')

        if self.use_cuda:
            self.model = self.model.cuda()
            prefetch = ['label']
        else:
            prefetch = []

        # Train model
        if self.trainable:
            self._fit(prefetch, val_set)

        return self

    def _fit(self, prefetch, val_set=None):
        import dgl
        import torch
        from torch import optim
        from . import dgl_utils
        import cornac
        from tqdm import tqdm

        # Get graph and edges
        g = self.train_graph
        u, v = g.edges()
        _, i, c = torch.unique(u, sorted=False, return_inverse=True, return_counts=True)
        mask = c[i] > 1
        _, i, c = torch.unique(v, sorted=False, return_inverse=True, return_counts=True)
        mask *= (c[i] > 1)
        eids = g.edges(form='eid')[mask]
        num_workers = self.num_workers

        if self.debug:
            num_workers = 0

        thread = False  # Memory saving and does not increase speed.

        # Create sampler
        sampler = dgl_utils.HypARBlockSampler(self.node_review_graph, self.review_graphs, self.review_aggregator,
                                              self.sid_aos, self.aos_list, 5,
                                              self.ui_graph, fanout=self.fanout)

        # If trained for ranking, define negative sampler only sampling items as negative samples.
        if self.objective == 'ranking':
            ic = collections.Counter(self.train_set.matrix.nonzero()[1])
            neg_sampler = dgl_utils.GlobalUniformItemSampler(self.num_neg_samples, self.train_set.num_items,)
        else:
            neg_sampler = None

        # Initialize sampler and dataloader
        sampler = dgl_utils.HypAREdgeSampler(sampler, prefetch_labels=prefetch, negative_sampler=neg_sampler,
                                             exclude='self')
        dataloader = dgl.dataloading.DataLoader(g, eids, sampler, batch_size=self.batch_size, shuffle=True,
                                                drop_last=True, device=self.device,
                                                num_workers=num_workers, use_prefetch_thread=thread)

        # Initialize training params.
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # Define metrics
        if self.objective == 'ranking':
            metrics = [cornac.metrics.NDCG(), cornac.metrics.AUC(), cornac.metrics.MAP(), cornac.metrics.MRR()]
        else:
            metrics = [cornac.metrics.MSE()]

        # Initialize variables for training
        best_state = None
        best_score = 0 if metrics[0].higher_better else float('inf')
        best_epoch = 0
        epoch_length = len(dataloader)
        all_nodes = torch.arange(next(iter(self.review_graphs.values())).shape[0]).to(self.device)

        # Train model
        for e in range(self.num_epochs):
            # Initialize for logging
            tot_losses = defaultdict(int)
            cur_losses = {}
            self.model.train()
            with tqdm(dataloader, disable=not self.verbose) as progress:
                for i, batch in enumerate(progress, 1):
                    # Batch depends on objective
                    if self.objective == 'ranking':
                        input_nodes, edge_subgraph, neg_subgraph, blocks = batch
                    else:
                        input_nodes, edge_subgraph, blocks = batch

                    # Get node representations and review representations
                    node_representation, e_star = self.model(blocks, self.model.get_initial_embedings(all_nodes), input_nodes)

                    # Get preiction based on graph structure (edges represents ratings, thus predictions)
                    pred = self.model.graph_predict(edge_subgraph, e_star)
                    loss = 0
                    if self.objective == 'ranking':
                        # Calculate predictions for negative subgraph and calculate ranking loss.
                        pred_j = self.model.graph_predict(neg_subgraph, e_star)
                        pred_j = pred_j.reshape(-1, self.num_neg_samples)
                        loss = self.model.ranking_loss(pred, pred_j)

                        # Calculate accuracy
                        acc = (pred > pred_j).sum() / pred_j.shape.numel()
                        cur_losses['acc'] = acc.detach()
                    else:
                        # Calculate rating loss, if using prediction instead of ranking.
                        loss = self.model.rating_loss(pred, edge_subgraph.edata['label'])

                    cur_losses['lloss'] = loss.clone().detach()  # Learning loss

                    # If using explainability, calculate loss and accuracy for explainability.
                    if self.learn_explainability:
                        aos_loss, aos_acc = self.model.aos_graph_predict(edge_subgraph, node_representation, e_star)
                        aos_loss = aos_loss.mean()
                        cur_losses['aos_loss'] = aos_loss.detach()
                        cur_losses['aos_acc'] = (aos_acc.sum() / aos_acc.shape.numel()).detach()
                        loss += self.learn_weight * aos_loss  # Add to loss with weight.

                    cur_losses['totloss'] = loss.detach()
                    loss.backward()

                    # Update batch losses
                    for k, v in cur_losses.items():
                        tot_losses[k] += v.cpu()

                    # Update model
                    optimizer.step()
                    optimizer.zero_grad()

                    # Define printing
                    loss_str = ','.join([f'{k}:{v / i:.3f}' for k, v in tot_losses.items()])

                    # If not validating, else
                    if i != epoch_length or val_set is None:
                        progress.set_description(f'Epoch {e}, ' + loss_str)
                    elif (e + 1) % self.eval_interval == 0:
                        # If validating, validate and print results.
                        results = self._validate(val_set, metrics)
                        res_str = 'Val: ' + ','.join([f'{m.name}:{r:.4f}' for m, r in zip(metrics, results)])
                        progress.set_description(f'Epoch {e}, ' + f'{loss_str}, ' + res_str)

                        # If use best state and new best score, save state.
                        if self.model_selection == 'best' and \
                                (results[0] > best_score if metrics[0].higher_better else results[0] < best_score):
                            best_state = deepcopy(self.model.state_dict())
                            best_score = results[0]
                            best_epoch = e

            # Stop if no improvement.
            if self.early_stopping is not None and (e - best_epoch) >= self.early_stopping:
                break

        # Space efficiency
        del self.node_filter
        del g, eids
        del dataloader
        del sampler

        # Load best state if using best state.
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Do inference calculation
        self.model.eval()
        with torch.no_grad():
            self.model.inference(self.node_review_graph, self.ui_graph, self.device,
                                 self.batch_size)

        # Set self values
        self.best_epoch = best_epoch
        self.best_value = best_score

    def _validate(self, val_set, metrics):
        from ...eval_methods.base_method import rating_eval, ranking_eval
        import torch

        # Do inference calculation
        self.model.eval()
        with torch.no_grad():
            self.model.inference(self.node_review_graph, self.ui_graph, self.device,
                                 self.batch_size)

            # Evaluate model
            if self.objective == 'ranking':
                (result, _) = ranking_eval(self, metrics, self.train_set, val_set)
            else:
                (result, _) = rating_eval(self, metrics, val_set, user_based=self.user_based)

        # Return best validation score
        return result

    def score(self, user_idx, item_idx=None):
        import torch

        # Ensure model is in evaluation mode and not calculating gradient.
        self.model.eval()
        with torch.no_grad():
            # Shift user ids
            user_idx = torch.tensor(user_idx + self.n_items, dtype=torch.int64).to(self.device)

            # If item_idx is None, predict all items, else predict only item_idx.
            if item_idx is None:
                item_idx = torch.arange(self.n_items, dtype=torch.int64).to(self.device)
                pred = self.model.predict(user_idx, item_idx).reshape(-1).cpu().numpy()
            else:
                item_idx = torch.tensor(item_idx, dtype=torch.int64).to(self.device)
                pred = self.model.predict(user_idx, item_idx).cpu()

            # Return predictions
            return pred

    def monitor_value(self, train_set, val_set=None):
        pass

    def save(self, save_dir=None, save_trainset=False):
        import torch

        if save_dir is None:
            return

        # Unset matrices to avoid pickling errors. Convert for review graphs due to same issues.
        self.model.review_conv.unset_matrices()
        self.review_graphs = {k: (v.row, v.col, v.shape) for k, v in self.review_graphs.items()}

        # Save model
        path = super().save(save_dir, save_trainset)
        name = path.rsplit('/', 1)[-1].replace('pkl', 'pt')

        # Save state dict, only necessary if state should be used outside of class. Thus, not part of load.
        state = self.model.state_dict()
        torch.save(state, os.path.join(save_dir, str(self.index), name))

        return path

    def load(self, model_path, trainable=False):
        import dgl.sparse as dglsp
        import torch

        # Load model
        model = super().load(model_path, trainable)

        # Convert review graphs to sparse matrices
        for k, v in model.review_graphs.items():
            model.review_graphs[k] = dglsp.spmatrix(torch.stack([v[0], v[1]]), shape=v[2]).coalesce().to(model.device)

        # Set matrices.
        model.model.review_conv.set_matrices(model.review_graphs)

        return model
