import re
from collections import OrderedDict, Counter, defaultdict
from typing import Mapping
from functools import lru_cache

import dgl.dataloading
import torch
from dgl.dataloading.negative_sampler import _BaseNegativeSampler
import dgl.backend as F

class HypAREdgeSampler(dgl.dataloading.EdgePredictionSampler):
    def __init__(self, sampler, exclude=None, reverse_eids=None,
                 reverse_etypes=None, negative_sampler=None, prefetch_labels=None):
        super().__init__(sampler, exclude, reverse_eids, reverse_etypes, negative_sampler,
                         prefetch_labels)

    def sample(self, g, seed_edges):    # pylint: disable=arguments-differ
        """Samples a list of blocks, as well as a subgraph containing the sampled
        edges from the original graph.

        If :attr:`negative_sampler` is given, also returns another graph containing the
        negative pairs as edges.
        """
        if isinstance(seed_edges, Mapping):
            seed_edges = {g.to_canonical_etype(k): v for k, v in seed_edges.items()}
        exclude = self.exclude
        pair_graph = g.edge_subgraph(
            seed_edges, relabel_nodes=False, output_device=self.output_device)
        eids = pair_graph.edata[dgl.EID]

        if self.negative_sampler is not None:
            neg_graph = self._build_neg_graph(g, seed_edges)
            pair_graph, neg_graph = dgl.compact_graphs([pair_graph, neg_graph])
        else:
            pair_graph = dgl.compact_graphs(pair_graph)

        pair_graph.edata[dgl.EID] = eids
        seed_nodes = pair_graph.ndata[dgl.NID]

        exclude_eids = dgl.dataloading.find_exclude_eids(
            g, seed_edges, exclude, self.reverse_eids, self.reverse_etypes,
            self.output_device)

        input_nodes, _, (pos_aos, neg_aos), blocks = self.sampler.sample(g, seed_nodes, exclude_eids, seed_edges)
        pair_graph.edata['pos'] = pos_aos.to(pair_graph.device)
        pair_graph.edata['neg'] = neg_aos.to(pair_graph.device)

        if self.negative_sampler is None:
            return self.assign_lazy_features((input_nodes, pair_graph, blocks))
        else:
            return self.assign_lazy_features((input_nodes, pair_graph, neg_graph, blocks))


class HypARBlockSampler(dgl.dataloading.NeighborSampler):
    """
        Given nodes, samples reviews and creates a batched review-graph of all sampled reviews.
        Parameters
        ----------
        node_review_graph: DGLHeteroGraph
            A heterogeneous graph with edges from reviews to nodes (users/items) with relation-type part_of.
        review_graphs: dict[DGLGraph]
            A dictionary with sid to a graph representing a review based on sentiment.
        fanouts: int
            Number of reviews to sample per node.
        kwargs: dict
            Arguments to pass to NeighborSampler. Read DGL docs for options.
"""

    def __init__(self, node_review_graph, review_graphs, aggregator, sid_aos, aos_list, n_neg, ui_graph,
                 compact=True, fanout=5, **kwargs):
        fanouts = [fanout]

        super().__init__(fanouts, **kwargs)
        self.node_review_graph = node_review_graph
        self.review_graphs = review_graphs
        self.aggregator = aggregator
        self.sid_aos = sid_aos
        self.aos_list = torch.LongTensor(aos_list)
        ac = Counter([a for aos in sid_aos for a in aos.numpy()])
        self.aos_probabilities = torch.log(torch.FloatTensor([ac.get(a) for a in sorted(ac)]) + 1)
        self.n_neg = n_neg
        self.ui_graph = ui_graph
        self.compact = compact
        self.n_ui_graph = self._nu_graph()
        self.exclude_sids = self._create_exclude_sids(self.node_review_graph)

    def _create_exclude_sids(self, n_r_graph):
        """
        Create a list of sids to exclude based on the node_review_graph.
        Parameters
        ----------
        n_r_graph: node_review graph

        Returns
        -------
        list
        """

        exclude_sids = []
        for sid in sorted(n_r_graph.nodes('review')):
            neighbors = n_r_graph.successors(sid)
            es = []
            for n in neighbors:
                es.append(n_r_graph.predecessors(n))

            if len(es) > 0:
                exclude_sids.append(torch.cat(es))
            else:
                exclude_sids.append(torch.LongTensor([]))
        return exclude_sids

    def _nu_graph(self):
        """
        Create graph mapping user/items to node ids. Used for preference where users and items are seperated, while
        for reviews they are combined or just seen as a node.

        Returns
        -------
        DGLHeteroGraph
        """

        # Get number of user, item and nodes
        n_nodes = self.node_review_graph.num_nodes('node')
        n_users = self.ui_graph.num_nodes('user')
        n_items = self.ui_graph.num_nodes('item')

        # Get all nodes. Nodes are user/item
        nodes = self.node_review_graph.nodes('node')
        device = nodes.device
        nodes = nodes.cpu()

        # Create mapping
        data = {
            ('user', 'un', 'node'): (torch.arange(n_users, dtype=torch.int64), nodes[nodes >= n_items]),
            ('item', 'in', 'node'): (torch.arange(n_items, dtype=torch.int64), nodes[nodes < n_items])
        }

        return dgl.heterograph(data, num_nodes_dict={'user': n_users, 'item': n_items, 'node': n_nodes}).to(device)

    def sample(self, g, seed_nodes, exclude_eids=None, seed_edges=None):
        # If exclude eids, find the equivalent eid of the node_review_graph.
        nrg_exclude_eids = None
        lgcn_exclude_eids = None

        # If exclude ids, find the equivalent.
        if exclude_eids is not None:
            # Find sid of the exclude eids.
            u, v = g.find_edges(exclude_eids)
            sid = g.edata['sid'][exclude_eids].to(u.device)

            # Find exclude eids based on sid and source nodes in g.
            nrg_exclude_eids = self.node_review_graph.edge_ids(sid, u, etype='part_of')

            # Find exclude eids based on sid and source nodes in g.
            lgcn_exclude_eids = dgl.dataloading.find_exclude_eids(
                self.ui_graph, {'user_item': seed_edges}, 'reverse_types', None, {'user_item': 'item_user', 'item_user': 'user_item'},
                self.output_device)
            mask = torch.ones((len(self.sid_aos)))
            mask[sid] = 0

        # Based on seed_nodes, find reviews to represent the nodes.
        input_nodes, output_nodes, blocks = super().sample(self.node_review_graph, {'node': seed_nodes},
                                                           nrg_exclude_eids)
        block = blocks[0]

        block = block['part_of']
        blocks[0] = block

        # If all nodes are removed, add random blocks/random reviews.
        # Will not occur during inference.
        if torch.any(block.in_degrees(block.dstnodes()) == 0):
            for index in torch.where(block.in_degrees(block.dstnodes()) == 0)[0]:
                perm = torch.randperm(block.num_src_nodes())
                block.add_edges(block.srcnodes()[perm[:self.fanouts[0]]],
                                index.repeat(min(max(1, self.fanouts[0]), block.num_src_nodes())))

        blocks2 = []
        seed_nodes = output_nodes

        # LightGCN Sampling
        for i in range(4):
            if i == 0:
                # Use node to user/item graph to sample first.
                frontier = self.n_ui_graph.sample_neighbors(
                    seed_nodes, -1, edge_dir=self.edge_dir, prob=self.prob,
                    replace=self.replace, output_device=self.output_device,
                    exclude_edges=None)
            else:
                frontier = self.ui_graph.sample_neighbors(
                    seed_nodes, -1, edge_dir=self.edge_dir, prob=self.prob,
                    replace=self.replace, output_device=self.output_device,
                    exclude_edges=lgcn_exclude_eids)

            # Sample reviews based on the user/item graph.
            eid = frontier.edata[dgl.EID]
            block = dgl.to_block(frontier, seed_nodes)
            block.edata[dgl.EID] = eid
            seed_nodes = block.srcdata[dgl.NID]
            blocks2.insert(0, block)

        pos_aos = []
        neg_aos = []
        # Find aspect/opinion sentiment based on the sampled reviews.
        for sid in g.edata['sid'][exclude_eids].cpu().numpy():
            aosid = self.sid_aos[sid]
            aosid = aosid[torch.randperm(len(aosid))[0]]
            pos_aos.append(aosid)  # Add positive sample.

            probability = torch.ones(len(self.aos_probabilities))

            # Exclude self and other aspects/opinions mentioned by the user or item.
            probability[aosid] = 0
            exclude_sids = torch.cat([self.sid_aos[i] for i in self.exclude_sids[sid]])
            probability[exclude_sids] = 0

            # Add negative samples based on probability (allow duplicates).
            neg_aos.append(torch.multinomial(probability, self.n_neg, replacement=True))

        # Transform to tensors.
        pos_aos = torch.LongTensor(pos_aos)
        neg_aos = torch.stack(neg_aos)

        # Based on sid id, get actual aos.
        pos_aos, neg_aos = self.aos_list[pos_aos], self.aos_list[neg_aos]

        return input_nodes, output_nodes, [pos_aos, neg_aos], [blocks, blocks2, mask]


class GlobalUniformItemSampler(_BaseNegativeSampler):
    def __init__(self, k, n_items, probabilities=None):
        super(_BaseNegativeSampler, self).__init__()
        self.k = k
        self.n_items = n_items
        self.probabilities = probabilities

    def _generate(self, g, eids, canonical_etype):
        _, _, vtype = canonical_etype
        shape = F.shape(eids)
        dtype = F.dtype(eids)
        ctx = F.context(eids)
        src, _ = g.find_edges(eids, etype=canonical_etype)
        src = F.repeat(src, self.k, 0)
        if self.probabilities is not None:
            dst = torch.multinomial(self.probabilities, self.k, replacement=True).reshape(1, self.k)
        else:
            dst = F.randint((1, self.k), dtype, ctx, 0, self.n_items)
        dst = F.repeat(dst, shape[0], 0).reshape(-1)
        return src, dst


def stem_fn(x):
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    # Remove special characters and numbers. Multiple dashes, single quotes, and equal signs, and similar special chars.
    cleaned = re.sub(r'--+.*|-+$|\+\+|\'.+|=+.*$|-\d.*', '', x)
    return stemmer.stem(cleaned.lower())

def stem(sentiment):
    ao_preprocess_fn = stem_fn

    # Set seed for reproducibility
    import random
    random.seed(42)

    # Map id to new word
    a_id_new = {i: ao_preprocess_fn(e) for e, i in sentiment.aspect_id_map.items()}
    o_id_new = {i: ao_preprocess_fn(e) for e, i in sentiment.opinion_id_map.items()}

    # Assign new ids to words, mapping from word to id
    a_id = {e: i for i, e in enumerate(sorted(set(a_id_new.values())))}
    o_id = {e: i for i, e in enumerate(sorted(set(o_id_new.values())))}

    # Map old id to new id
    a_o_n = {i: a_id[e] for i, e in a_id_new.items()}
    o_o_n = {i: o_id[e] for i, e in o_id_new.items()}

    # Assign new ids to sentiment
    sents = OrderedDict()
    for i, aos in sentiment.sentiment.items():
        sents[i] = [(a_o_n[a], o_o_n[o], s) for a, o, s in aos]

    return sents, a_o_n, o_o_n


@lru_cache()
def generate_mappings(sentiment, match, get_ao_mappings=False, get_sent_edge_mappings=False):
    # Initialize all variables
    aos_user = defaultdict(list)
    aos_item = defaultdict(list)
    aos_sent = defaultdict(list)
    user_aos = defaultdict(list)
    item_aos = defaultdict(list)
    sent_aos = defaultdict(list)
    user_sent_edge_map = dict()
    item_sent_edge_map = dict()

    # Get new sentiments and mappings from old to new id.
    sent, a_mapping, o_mapping = stem(sentiment)

    # Iterate over all sentiment triples and create the corresponding mapping for users and items.
    edge_id = -1
    for uid, isid in sentiment.user_sentiment.items():
        for iid, sid in isid.items():
            # Assign edge id mapping for user and item.
            user_sent_edge_map[(sid, uid)] = (edge_id := edge_id + 1)  # assign and increment
            item_sent_edge_map[(sid, iid)] = (edge_id := edge_id + 1)
            for a, o, s in sent[sid]:
                if match == 'aos':
                    element = (a, o, s)
                elif match == 'a':
                    element = a
                elif match == 'as':
                    element = (a, s)
                elif match == 'ao':
                    element = (a, o)
                else:
                    raise NotImplementedError

                aos_user[element].append(uid)
                aos_item[element].append(iid)
                aos_sent[element].append(sid)
                user_aos[uid].append(element)
                item_aos[iid].append(element)
                sent_aos[sid].append(element)

    return_data = [aos_user, aos_item, aos_sent, user_aos, item_aos, sent_aos]

    if get_ao_mappings:
        return_data.extend([a_mapping, o_mapping])

    if get_sent_edge_mappings:
        return_data.extend([user_sent_edge_map, item_sent_edge_map])

    return tuple(return_data)


def create_heterogeneous_graph(train_set, bipartite=True):
    """
    Create a graph with users, items, aspects and opinions.
    Parameters
    ----------
    train_set : Dataset
    bipartite: if false have a different edge type per rating; otherwise, only use interacted.

    Returns
    -------
    DGLGraph
        A graph with edata type, label and an initialized attention of 1/k.
    int
        Num nodes in graph.
    int
        Number of items in dataset.
    int
        Number of relations in dataset.
    """
    import dgl
    import torch

    edge_types = {
        'mentions': [],
        'described_as': [],
        'has_opinion': [],
        'co-occur': [],
    }

    rating_types = set()
    for indices in list(zip(*train_set.matrix.nonzero())):
        rating_types.add(train_set.matrix[indices])

    if not bipartite:
        train_types = []
        for rt in rating_types:
            edge_types[str(rt)] = []
            train_types.append(str(rt))
    else:
        train_types = ['interacted']
        edge_types['interacted'] = []

    sentiment_modality = train_set.sentiment
    n_users = len(train_set.uid_map)
    n_items = len(train_set.iid_map)
    n_aspects = len(sentiment_modality.aspect_id_map)
    n_opinions = len(sentiment_modality.opinion_id_map)
    n_nodes = n_users + n_items + n_aspects + n_opinions

    # Create all the edges: (item, described_as, aspect), (item, has_opinion, opinion), (user, mentions, aspect),
    # (aspect, cooccur, opinion), and (user, 'rating', item). Note rating is on a scale.
    for org_uid, isid in sentiment_modality.user_sentiment.items():
        uid = org_uid + n_items
        for iid, sid in isid.items():
            for aid, oid, _ in sentiment_modality.sentiment[sid]:
                aid += n_items + n_users
                oid += n_items + n_users + n_aspects

                edge_types['mentions'].append([uid, aid])
                edge_types['mentions'].append([uid, oid])
                edge_types['described_as'].append([iid, aid])
                edge_types['described_as'].append([iid, oid])
                edge_types['co-occur'].append([aid, oid])

            if not bipartite:
                edge_types[str(train_set.matrix[(org_uid, iid)])].append([uid, iid])
            else:
                edge_types['interacted'].append([uid, iid])

    # Create reverse edges.
    reverse = {}
    for etype, edges in edge_types.items():
        reverse['r_' + etype] = [[t, h] for h, t in edges]

    # edge_types.update(reverse)
    n_relations = len(edge_types)
    edges = [[h, t] for k in sorted(edge_types) for h, t in edge_types.get(k)]
    edges_t = torch.LongTensor(edges).unique(dim=0).T

    et_id_map = {et: i for i, et in enumerate(sorted(edge_types))}

    g = dgl.graph((torch.cat([edges_t[0], edges_t[1]]), torch.cat([edges_t[1], edges_t[0]])), num_nodes=n_nodes)
    inverse_et = {tuple(v): k for k, l in edge_types.items() for v in l}
    et = torch.LongTensor([et_id_map[inverse_et[tuple(v)]] for v in edges_t.T.tolist()])

    # Return 0 if not a rating type, else if using actual ratings return values else return 1 (bipartite).
    value_fn = lambda etype: 0 if etype not in train_types else (float(etype) if etype != 'interacted' else 1)
    labels = torch.FloatTensor([value_fn(inverse_et[tuple(v)]) for v in edges_t.T.tolist()])

    g.edata['type'] = torch.cat([et, et + n_relations])

    g.edata['label'] = torch.cat([labels, labels])
    g.edata['a'] = dgl.ops.edge_softmax(g, torch.ones_like(g.edata['label']))

    return g, n_nodes, n_items, n_relations * 2