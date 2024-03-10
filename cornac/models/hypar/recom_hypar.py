import collections
import os
import pickle
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy

from ..recommender import Recommender
from ...data import Dataset


class HypAR(Recommender):
    def __init__(self, name='HypAR', use_cuda=False, use_uva=False, stemming=True,
                 batch_size=128,
                 num_workers=0,
                 num_epochs=10,
                 early_stopping=10,
                 eval_interval=1,
                 learning_rate=0.1,
                 weight_decay=0,
                 l2_weight=0.,
                 node_dim=64,
                 num_heads=3,
                 fanout=5,
                 use_relation=False,
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
                 hypergraph_attention=False,
                 user_based=True,
                 verbose=True,
                 index=0,
                 out_path=None,
                 popularity_biased_sampling=False,
                 self_enhance_loss=True,
                 learn_explainability=False,
                 learn_method='transr',
                 learn_weight=1.,
                 learn_pop_sampling=False,
                 embedding_type='ao_embeddings',
                 debug=False
                 ):

        super().__init__(name)
        # Default values
        if layer_dropout is None:
            layer_dropout = 0.  # node embedding dropout, review embedding dropout

        # CUDA
        self.use_cuda = use_cuda
        self.device = 'cuda' if use_cuda else 'cpu'
        self.use_uva = use_uva

        # Parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.early_stopping = early_stopping
        self.eval_interval = eval_interval
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.node_dim = node_dim
        self.l2_weight = l2_weight
        self.num_heads = num_heads
        self.fanout = fanout
        self.use_relation = use_relation
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
        self.hypergraph_attention = hypergraph_attention
        self.stemming = stemming
        self.self_enhance_loss = self_enhance_loss
        self.learn_explainability = learn_explainability
        self.learn_method = learn_method
        self.learn_weight = learn_weight
        self.learn_pop_sampling = learn_pop_sampling
        self.embedding_type = embedding_type
        self.popularity_biased_sampling = popularity_biased_sampling
        parameter_list = ['batch_size', 'learning_rate', 'weight_decay', 'node_dim', 'num_heads',
                          'fanout', 'use_relation', 'model_selection', 'review_aggregator', 'objective',
                          'predictor', 'preference_module', 'layer_dropout', 'attention_dropout', 'stemming',
                          'learn_explainability', 'learn_method', 'learn_weight', 'learn_pop_sampling',
                          'popularity_biased_sampling', 'combiner', 'self_enhance_loss']
        self.parameters = collections.OrderedDict({k: self.__getattribute__(k) for k in parameter_list})

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
        assert use_uva == use_cuda or not use_uva, 'use_cuda must be true when using uva.'
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
        Learn aspect and opinion embeddings using word2vec.
        Parameters
        ----------
        train_set: dataset
            Dataset to use for learning embeddings.
        Returns
        -------
            Aspect and opinion embeddings, and word2vec model.
        """
        from .dgl_utils import generate_mappings, stem_fn
        from gensim.models import Word2Vec
        from gensim.parsing import remove_stopwords, preprocess_string, stem_text
        from nltk.tokenize import word_tokenize
        from tqdm import tqdm
        import numpy as np

        sentiment = train_set.sentiment

        # Define preprocess functions for text, aspects and opinions.
        preprocess_fn = stem_fn

        # Process corpus, getting all sentences and words.
        corpus = []
        for review in tqdm(train_set.review_text.corpus, desc='Processing text', disable=not self.verbose):
            for sentence in review.split('.'):
                words = word_tokenize(sentence.replace(' n\'t ', 'n ').replace('/', ' '))
                corpus.append(' '.join(preprocess_fn(word) for word in words))

        # Process words to match with aos extraction methodology used in SEER.
        a_old_new_map = {a: preprocess_fn(a) for a in sentiment.aspect_id_map}
        o_old_new_map = {o: preprocess_fn(o) for o in sentiment.opinion_id_map}

        # Generate mappings for aspect and opinion ids.
        _, _, _, _, _, _, a2a, o2o = generate_mappings(train_set.sentiment, 'a', get_ao_mappings=True)

        # Define a progressbar for training word2vec as no information is displayed without.
        class CallbackProgressBar:
            def __init__(self, verbose):
                self.verbose = verbose
                self.progress = None

            def on_train_begin(self, method):
                if self.progress is None:
                    self.progress = tqdm(desc='Training Word2Vec', total=method.epochs, disable=not self.verbose)

            def on_train_end(self, method):
                pass

            def on_epoch_begin(self, method):
                pass

            def on_epoch_end(self, method):
                self.progress.update(1)

        # Split words on space and get all unique words
        wc = [s.split(' ') for s in corpus]
        all_words = set(s for se in wc for s in se)

        # Assert all aspects and opinions in dataset are in corpus. If not, print missing words.
        # New datasets may require more preprocessing.
        assert all([a in all_words for a in a_old_new_map.values()]), [a for a in a_old_new_map.values() if
                                                                       a not in all_words]
        assert all([o in all_words for o in o_old_new_map.values()]), [o for o in o_old_new_map.values() if
                                                                       o not in all_words]

        # Train word2vec model using callbacks for progressbar.
        l = CallbackProgressBar(self.verbose)
        embedding_dim = 100
        w2v_model = Word2Vec(wc, vector_size=embedding_dim, min_count=1, window=5, callbacks=[l], epochs=100)

        # Keyvector model
        kv = w2v_model.wv

        # Initialize embeddings
        a_embeddings = np.zeros((len(set(a2a.values())), embedding_dim))
        o_embeddings = np.zeros((len(set(o2o.values())), embedding_dim))

        # Define function for assigning embeddings to correct aspect.
        def get_info(old_new_pairs, mapping, embedding):
            for old, new in old_new_pairs:
                nid = mapping(old)
                vector = np.array(kv.get_vector(new))
                embedding[nid] = vector

            return embedding

        # Assign embeddings to correct aspect and opinion.
        a_embeddings = get_info(a_old_new_map.items(), lambda x: a2a[sentiment.aspect_id_map[x]], a_embeddings)
        o_embeddings = get_info(o_old_new_map.items(), lambda x: o2o[sentiment.opinion_id_map[x]], o_embeddings)

        return a_embeddings, o_embeddings, kv

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
        import torch
        from ..lightgcn.lightgcn import construct_graph

        super().fit(train_set, val_set)
        n_nodes, self.n_relations, self.sid_aos, self.aos_list = self._graph_wrapper(train_set,
                                                                                     self.graph_type)  # graphs are as attributes of model.

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

        if not self.use_relation:
            self.n_relations = 0

        self.ui_graph = construct_graph(train_set, self.num_users, self.num_items)
        n_r_types = max(self.node_review_graph.edata['r_type']) + 1

        # create model
        from .hypar import Model

        self.model = Model(self.ui_graph, n_nodes, self.n_relations, n_r_types, self.review_aggregator,
                           self.predictor, self.node_dim, self.review_graphs, self.num_heads, [self.layer_dropout] * 2,
                           self.attention_dropout, self.preference_module, self.use_cuda, combiner=self.combiner,
                           aos_predictor=self.learn_method, non_linear=self.non_linear,
                           embedding_type=self.embedding_type, hypergraph_attention=self.hypergraph_attention,
                           **kwargs)

        self.model.reset_parameters()

        if self.verbose:
            print(f'Number of trainable parameters: {sum(p.numel() for p in self.model.parameters())}')

        if self.use_cuda:
            self.model = self.model.cuda()
            prefetch = ['label']
        else:
            prefetch = []

        # x = self.model.get_initial_embedings(torch.arange(n_nodes + kwargs['ao_embeddings'].size(0), device=self.device))
        # self.model.review_conv(x)
        if self.trainable:
            self._fit(prefetch, val_set)

        if self.summary_writer is not None:
            self.summary_writer.close()

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

        if self.use_uva:
            # g = g.to(self.device)
            eids = eids.to(self.device)
            self.node_review_graph = self.node_review_graph.to(self.device)
            self.ui_graph = self.ui_graph.to(self.device)
            num_workers = 0

        if self.debug:
            num_workers = 0

        thread = False  # Memory saving and does not increase speed.

        # Create sampler
        sampler = dgl_utils.HearBlockSampler(self.node_review_graph, self.review_graphs, self.review_aggregator,
                                             self.sid_aos, self.aos_list, 5,
                                             self.ui_graph, fanout=self.fanout, hard_negatives=self.learn_pop_sampling)
        if self.objective == 'ranking':
            ic = collections.Counter(self.train_set.matrix.nonzero()[1])
            probabilities = torch.FloatTensor([ic.get(i) for i in sorted(ic)]) if self.popularity_biased_sampling \
                else None
            neg_sampler = dgl_utils.GlobalUniformItemSampler(self.num_neg_samples, self.train_set.num_items,
                                                             probabilities)
        else:
            neg_sampler = None

        sampler = dgl_utils.HEAREdgeSampler(sampler, prefetch_labels=prefetch, negative_sampler=neg_sampler,
                                            exclude='self')
        dataloader = dgl.dataloading.DataLoader(g, eids, sampler, batch_size=self.batch_size, shuffle=True,
                                                drop_last=True, device=self.device, use_uva=self.use_uva,
                                                num_workers=num_workers, use_prefetch_thread=thread)

        # Initialize training params.
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.objective == 'ranking':
            metrics = [cornac.metrics.NDCG(), cornac.metrics.AUC(), cornac.metrics.MAP(), cornac.metrics.MRR()]
        else:
            metrics = [cornac.metrics.MSE()]

        best_state = None
        best_score = 0 if metrics[0].higher_better else float('inf')
        best_epoch = 0
        epoch_length = len(dataloader)
        all_nodes = torch.arange(next(iter(self.review_graphs.values())).shape[0]).to(self.device)
        for e in range(self.num_epochs):
            tot_losses = defaultdict(int)
            cur_losses = {}
            self.model.train()

            with (dataloader.enable_cpu_affinity() if False else nullcontext()):
                with tqdm(dataloader, disable=not self.verbose) as progress:
                    for i, batch in enumerate(progress, 1):
                        if self.objective == 'ranking':
                            input_nodes, edge_subgraph, neg_subgraph, blocks = batch
                        else:
                            input_nodes, edge_subgraph, blocks = batch

                        node_rep, e_star, node_rep_subset = self.model(blocks,
                                                                       self.model.get_initial_embedings(all_nodes),
                                                                       input_nodes)

                        rp, pred = self.model.graph_predict(edge_subgraph, [node_rep_subset, e_star])
                        rp = rp.unsqueeze(-1)
                        loss = 0
                        if self.objective == 'ranking':
                            rp_j, pred_j = self.model.graph_predict(neg_subgraph, [node_rep_subset, e_star])
                            pred_j = pred_j.reshape(-1, self.num_neg_samples)
                            rp_j = rp_j.reshape(-1, self.num_neg_samples)
                            acc = (pred > pred_j).sum() / pred_j.shape.numel()
                            # loss += self.model.ranking_loss(pred, pred_j)
                            # cur_losses['loss'] = loss.detach()
                            # cur_losses['acc'] = acc.detach()

                            acc = (rp > rp_j).sum() / rp_j.shape.numel()
                            rl = self.model.ranking_loss(rp, rp_j)

                            if self.self_enhance_loss:
                                cur_losses['rl'] = rl.detach()
                                cur_losses['racc'] = acc.detach()

                                loss += rl

                            if self.l2_weight:
                                l2 = self.l2_weight * self.model.l2_loss(edge_subgraph, neg_subgraph, e_star)
                                loss += l2
                                cur_losses['l2'] = l2.detach()
                        else:
                            loss = self.model.rating_loss(pred, edge_subgraph.edata['label'])
                            cur_losses['loss'] = loss.detach()

                        if self.learn_explainability:
                            aos_loss, aos_acc = self.model.aos_graph_predict(edge_subgraph, node_rep, e_star)
                            aos_loss = aos_loss.mean()
                            cur_losses['aos_loss'] = aos_loss.detach()
                            cur_losses['aos_acc'] = (aos_acc.sum() / aos_acc.shape.numel()).detach()
                            loss += self.learn_weight * aos_loss

                        loss.backward()

                        for k, v in cur_losses.items():
                            tot_losses[k] += v.cpu()

                        optimizer.step()
                        optimizer.zero_grad()
                        loss_str = ','.join([f'{k}:{v / i:.3f}' for k, v in tot_losses.items()])
                        if i != epoch_length or val_set is None:
                            progress.set_description(f'Epoch {e}, ' + loss_str)
                        elif (e + 1) % self.eval_interval == 0:
                            results = self._validate(val_set, metrics)
                            res_str = 'Val: ' + ', '.join([f'{m.name}:{r:.4f}' for m, r in zip(metrics, results)])
                            progress.set_description(f'Epoch {e}, ' + f'{loss_str}, ' + res_str)

                            if self.model_selection == 'best' and (results[0] > best_score if metrics[0].higher_better
                            else results[0] < best_score):
                                best_state = deepcopy(self.model.state_dict())
                                best_score = results[0]
                                best_epoch = e

            if self.early_stopping is not None and (e - best_epoch) >= self.early_stopping:
                break

        del self.node_filter
        del g, eids
        del dataloader
        del sampler

        if best_state is not None:
            self.model.load_state_dict(best_state)

        if val_set is not None and self.summary_writer is not None:
            results = self._validate(val_set, metrics)
            self.summary_writer.add_hparams(dict(self.parameters), dict(zip([m.name for m in metrics], results)))

        self.model.eval()
        with torch.no_grad():
            self.model.inference(self.review_graphs, self.node_review_graph, self.ui_graph, self.device,
                                 self.batch_size)

        self.best_epoch = best_epoch
        self.best_value = best_score

    def _validate(self, val_set, metrics):
        from ...eval_methods.base_method import rating_eval, ranking_eval
        import torch

        self.model.eval()
        with torch.no_grad():
            self.model.inference(self.review_graphs, self.node_review_graph, self.ui_graph, self.device,
                                 self.batch_size)
            if self.objective == 'ranking':
                (result, _) = ranking_eval(self, metrics, self.train_set, val_set)
            else:
                (result, _) = rating_eval(self, metrics, val_set, user_based=self.user_based)
        return result

    def score(self, user_idx, item_idx=None):
        import torch

        self.model.eval()
        with torch.no_grad():
            user_idx = torch.tensor(user_idx + self.n_items, dtype=torch.int64).to(self.device)
            if item_idx is None:
                item_idx = torch.arange(self.n_items, dtype=torch.int64).to(self.device)
                pred = self.model.predict(user_idx, item_idx).reshape(-1).cpu().numpy()
            else:
                item_idx = torch.tensor(item_idx, dtype=torch.int64).to(self.device)
                pred = self.model.predict(user_idx, item_idx).cpu()

            return pred

    def monitor_value(self):
        pass

    def save(self, save_dir=None):
        import torch
        import pandas as pd

        if save_dir is None:
            return

        self.model.review_conv.unset_matrices()
        self.review_graphs = {k: (v.row, v.col, v.shape) for k, v in self.review_graphs.items()}
        path = super().save(save_dir)
        name = path.rsplit('/', 1)[-1].replace('pkl', 'pt')

        state = self.model.state_dict()
        torch.save(state, os.path.join(save_dir, str(self.index), name))

        # results_path = os.path.join(path.rsplit('/', 1)[0], 'results.csv')
        # header = not os.path.exists(results_path)
        # self.parameters['score'] = self.best_value
        # self.parameters['epoch'] = self.best_epoch
        # self.parameters['file'] = path.rsplit('/')[-1]
        # self.parameters['id'] = self.index
        # df = pd.DataFrame({k: [v] for k, v in self.parameters.items()})
        # df.to_csv(results_path, header=header, mode='a', index=False)

        return path

    def load(self, model_path, trainable=False):
        import dgl.sparse as dglsp
        import torch

        model = super().load(model_path, trainable)
        for k, v in model.review_graphs.items():
            model.review_graphs[k] = dglsp.spmatrix(torch.stack([v[0], v[1]]), shape=v[2]).coalesce().to(model.device)

        model.model.review_conv.set_matrices(model.review_graphs)
        return model
