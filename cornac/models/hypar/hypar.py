import dgl.utils
import torch
from dgl.ops import edge_softmax
from torch import nn
import dgl.function as fn
import dgl.sparse as dglsp


class AOSPredictionLayer(nn.Module):
    """
        Ranking layer for AOS prediction.

        Parameters
        ----------
        aos_predictor : str
            Type of AOS predictor. Can be 'non-linear' or 'transr'.
        in_dim1: int
            Dimension of the first input. I.e., user/item
        in_dim2:
            Dimension of the second input. I.e., aspect/opinion
        hidden_dims:
            List of hidden dimensions, for multiple MLP layers.
        n_relations:
            Number of relations, i.e. sentiments.
        loss: str
            Loss function to be used. Can be 'bpr' or 'transr'.
        """

    def __init__(self, aos_predictor, in_dim1, in_dim2, hidden_dims, n_relations, loss='bpr'):
        # Initialize variables
        super().__init__()
        self.loss = loss
        assert loss in ['bpr', 'transr'], f'Invalid loss: {loss}'
        dims = [in_dim1*2] + hidden_dims
        max_i = len(dims)
        r_dim = hidden_dims[-1]

        # Either have nonlinear mlp transformation or use tranr like similarity
        if aos_predictor == 'non-linear':
            self.mlp_ao = nn.ModuleList(nn.Sequential(
                *[nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.LeakyReLU()) for i in range(max_i - 1)]
            ) for _ in range(n_relations))
            dims = [in_dim2*2] + hidden_dims
            self.mlp_ui = nn.Sequential(
                *[nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.LeakyReLU()) for i in range(max_i - 1)]
            )
            self.r = nn.Parameter(torch.zeros((n_relations, r_dim)))
        elif aos_predictor == 'transr':
            self.w_aor = nn.Parameter(torch.zeros((n_relations, in_dim1*2, r_dim)))
            self.w_uir = nn.Parameter(torch.zeros((n_relations, in_dim2*2, r_dim)))
            self.r = nn.Parameter(torch.zeros((n_relations, r_dim)))
            nn.init.xavier_normal_(self.w_aor); nn.init.xavier_normal_(self.w_uir); nn.init.xavier_normal_(self.r)
        else:
            raise NotImplementedError

        self._aos_predictor = aos_predictor
        self._n_relations = n_relations
        self._out_dim = hidden_dims[-1]

    def forward(self, u_emb, i_emb, a_emb, o_emb, s):
        """
        Calculates the AOS prediction
        Parameters
        ----------
        u_emb: torch.Tensor
            User embedding
        i_emb: torch.Tensor
            Item embedding
        a_emb: torch.Tensor
            Aspect embedding
        o_emb: torch.Tensor
            Opinion embedding
        s: torch.Tensor
            Sentiment label

        Returns
        -------
        torch.Tensor
            Score of ui/aos ranking.
        """

        # Concatenate user and item embeddings
        ui_in = torch.cat([u_emb, i_emb], dim=-1)
        ao_in = torch.cat([a_emb, o_emb], dim=-1)

        # Get size
        if len(ao_in.size()) == 3:
            b, n, d = ao_in.size()
        else:
            b, d = ao_in.size()
            n = 1

        # Reshape
        s = s.reshape(b, n)
        ao_in = ao_in.reshape(b, n, d)

        # Transform using either non-linear mlp or transr
        if self._aos_predictor == 'non-linear':
            ui_emb = self.mlp_ui(ui_in)
            aos_emb = torch.empty((len(s), n, self._out_dim), device=ui_emb.device)
            for r in range(self._n_relations):
                mask = s == r
                aos_emb[mask] = self.mlp_ao[r](ao_in[mask])
            ui_emb = ui_emb.unsqueeze(1)
        elif self._aos_predictor == 'transr':
            ui_emb = torch.empty((b, n, self._out_dim), device=u_emb.device)
            aos_emb = torch.empty((b, n, self._out_dim), device=u_emb.device)
            for r in range(self._n_relations):
                mask = s == r
                ui_emb[mask] = torch.repeat_interleave(ui_in, mask.sum(-1), dim=0) @ self.w_uir[r] + self.r[r]
                aos_emb[mask] = ao_in[mask] @ self.w_aor[r]
        else:
            raise NotImplementedError(self._aos_predictor)

        if self.loss == 'bpr':
            pred = (ui_emb * aos_emb).sum(-1)
        else:
            pred = (ui_emb - aos_emb).pow(2).sum(-1)

        return pred


class HypergraphLayer(nn.Module):
    """
        Hypergraph layer doing propagation along edges in the hypergraph.

        Parameters
        ----------
        H: dict
            Hypergraph incidence matrix for each relation relation type. I.e., positive and negative AO pairs.
        in_dim: int
            Input dimension
        non_linear: bool
            Whether to use non-linear activation function
        num_layers: int
            Number of layers
        dropout: float
            Dropout rate
        aggregator: str
            Aggregator to use. Can be 'sum' or mean, otherwise should be implemented.
        normalize: bool, default False
            Whether to normalize the output.
        """

    def __init__(self, H, in_dim, non_linear=True, num_layers=1, dropout=0, aggregator='mean',
                 normalize=False):
        super().__init__()
        self.aggregator = aggregator
        self.non_linear = non_linear
        self.normalize = normalize

        # Initialize matrices
        self.H = None
        self.D_e_inv = None
        self.L_left = None
        self.L_right = None
        self.L = None
        self.O = None
        self.D_v_invsqrt = None
        self.heads = None
        self.tails = None
        self.edges = None
        self.uniques = None

        # Set matrices
        self.set_matrices(H)

        # Define layers
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.W = nn.ModuleList([
                nn.ModuleDict({
                    k: nn.Linear(in_dim, in_dim) for k in H
                }) for _ in range(num_layers)
            ])

        # Set dropout and activation
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

    def set_matrices(self, H):
        """
        Initialize matrices for hypergraph layer for faster computation in forward and backward pass.
        Parameters
        ----------
        H: dict
            Hypergraph incidence matrix for each relation relation type. I.e., positive and negative AO pairs.

        Returns
        -------
        None
        """

        # Set hypergraph
        self.H = H

        # Compute degree matrices, node and edge-wise
        d_V = {k: v.sum(1) for k, v in H.items()}
        d_E = {k: v.sum(0) for k, v in H.items()}

        self.D_v_invsqrt = {k: dglsp.diag(v ** -.5) for k, v in d_V.items()}
        self.D_e_inv = {k: dglsp.diag(v ** -1) for k, v in d_E.items()}

        # Compute Laplacian from the equation above.
        self.L_left = {k: self.D_v_invsqrt[k] @ H[k] for k in H}
        self.L_right = {k: H[k].T @ self.D_v_invsqrt[k] for k in H}
        self.L = {k: self.L_left[k] @ self.D_e_inv[k] @ self.L_right[k] for k in H}

        # Out representation
        self.O = {k: self.D_e_inv[k] @ H[k].T for k in H}

    def unset_matrices(self):
        self.H = None
        self.D_e_inv = None
        self.L_left = None
        self.L_right = None
        self.L = None
        self.O = None
        self.D_v_invsqrt = None

    def forward(self, x, mask=None):
        D_e = self.D_e_inv

        # Mask if in train
        if mask is not None:
            # Compute laplacian matrix
            D_e = {k: dglsp.diag(D_e[k].val * mask) for k in D_e}
            L = {k: self.L_left[k]  @ D_e[k] @ self.L_right[k] for k in D_e}
        else:
            L = self.L

        node_out = [x]
        review_out = []
        # Iterate over layers
        for i, layer in enumerate(self.W):

            # Initialize in and out layers
            inner_x = []
            inner_o = []

            # Iterate over relation types (i.e., positive and negative AO pairs)
            # k is type and l linear layer.
            for k, l in layer.items():
                # Compute next layer
                e = L[k] @ l(self.dropout(x))

                # Apply non-linear activation
                if self.non_linear:
                    e = self.activation(e)

                # Get node representation
                o = self.O[k] @ e  # average of nodes participating in review edge

                inner_x.append(e)
                inner_o.append(o)

            # Combine sentiments
            x = torch.stack(inner_x)
            inner_o = torch.stack(inner_o)

            # Aggregate over sentiments
            if self.aggregator == 'sum':
                x = x.sum(0)
                inner_o = inner_o.sum(0)
            elif self.aggregator == 'mean':
                x = x.mean(0)
                inner_o = inner_o.mean(0)
            else:
                raise NotImplementedError(self.aggregator)

            # If using layer normalization, normalize using l2 norm.
            if self.normalize:
                x = x / (x.norm(2, dim=-1, keepdim=True) + 1e-5)  # add epsilon to avoid division by zero.
                inner_o = inner_o / (inner_o.norm(2, dim=-1, keepdim=True) + 1e-5)

            # Append representations
            node_out.append(x)
            review_out.append(inner_o)

        # Return aggregated representation using mean.
        return torch.stack(node_out).mean(0), torch.stack(review_out).mean(0)


class ReviewConv(nn.Module):
    """
        Review attention aggregation layer
        Parameters
        ----------
        aggregator: str
            Aggregator to use. Can be 'gatv2' and 'narre'.
        n_nodes: int
            Number of nodes
        in_feats: int
            Input dimension
        attention_feats: int
            Attention dimension
        num_heads: int
            Number of heads
        feat_drop: float, default 0.
            Dropout rate for feature
        attn_drop: float, default 0.
            Dropout rate for attention
        negative_slope: float, default 0.2
            Negative slope for LeakyReLU
        activation: callable, default None
            Activation function
        allow_zero_in_degree: bool, default False
            Whether to allow zero in degree
        bias: bool, default True
            Whether to include bias in linear transformations
        """

    def __init__(self,
                 aggregator,
                 n_nodes,
                 in_feats,
                 attention_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True):
        super(ReviewConv, self).__init__()

        # Set parameters
        self.aggregator = aggregator
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = dgl.utils.expand_as_pair(in_feats)
        self._out_feats = attention_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.fc_src = nn.Linear(
            self._in_src_feats, attention_feats * num_heads, bias=bias)

        # Initialize embeddings and layers used for other methods
        if self.aggregator == 'narre':
            self.node_quality = nn.Embedding(n_nodes, self._in_dst_feats)
            self.fc_qual = nn.Linear(self._in_dst_feats, attention_feats * num_heads, bias=bias)
        elif self.aggregator == 'gatv2':
            pass
        else:
            raise NotImplementedError(f'Not implemented any aggregator named {self.aggregator}.')

        self.attn = nn.Parameter(torch.FloatTensor(size=(1, num_heads, attention_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.activation = activation
        self.bias = bias

    def rel_attention(self, lhs_field, rhs_field, out, w, b, source=True):
        def func(edges):
            idx = edges.data[rhs_field]
            data = edges.src[lhs_field] if source else edges.data[lhs_field]
            return {out: dgl.ops.gather_mm(data, w, idx_b=idx) + b[idx]}
        return func

    def forward(self, graph, feat, get_attention=False):
        """
        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            # Check if any 0-in-degree nodes
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise dgl.DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            # Drop features
            h_src = self.feat_drop(feat)

            # Transform src node features to attention space
            feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
            graph.srcdata.update({'el': feat_src}) # (num_src_edge, num_heads, out_dim)

            # Move messages to edges
            if self.aggregator == 'narre':
                # Get quality representation for user/item
                h_qual = self.feat_drop(self.node_quality(graph.edata['nid']))

                # Transform to attention space and add to edge data
                feat_qual = self.fc_qual(h_qual).view(-1, self._num_heads, self._out_feats)
                graph.edata.update({'qual': feat_qual})

                # Add node and quality represenation on edges.
                graph.apply_edges(fn.u_add_e('el', 'qual', 'e'))
            else:
                graph.apply_edges(fn.copy_u('el', 'e'))

            # Get attention representation
            e = self.leaky_relu(graph.edata.pop('e'))# (num_src_edge, num_heads, out_dim)

            # Compute attention score
            e = (e * self.attn).sum(dim=-1).unsqueeze(dim=2)# (num_edge, num_heads, 1)

            # Normalize attention using softmax on edges
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e)) # (num_edge, num_heads)

            # If using narre set node representation to original input instead of attention representation
            if self.aggregator == 'narre':
                graph.srcdata.update({'el': h_src})

            # Aggregate reviews to nodes.
            graph.update_all(fn.u_mul_e('el', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']

            # If using activation, apply
            if self.activation:
                rst = self.activation(rst)

            # In inference, we may want to get the attention. If so return both review representation and attention.
            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst


class Model(nn.Module):
    """
        HypAR model based on DGL and Torch.
        Parameters
        ----------
        g: dgl.DGLGraph
            Heterogeneous graph with user and item nodes.
        n_nodes: int
            Number of nodes
        aggregator: str
            Aggregator to use. Can be 'gatv2' and 'narre'.
        predictor: str
            Predictor to use. Can be 'narre' and 'dot'.
        node_dim: int
            Dimension of node embeddings
        incidence_dict:
            Incidence matrix for each relation relation type. I.e., positive and negative AO pairs.
        num_heads: int
            Number of heads to use for review aggregation.
        layer_dropout: list
            Dropout rate for hypergraph and for review attention layer.
        attention_dropout: float
            Dropout rate for attention.
        preference_module: str
            Preference module to use. Can be 'lightgcn' and 'mf'.
        use_cuda: bool
            Whether we are using cuda.
        combiner: str
            Combiner to use. Can be 'add', 'mul', 'bi-interaction', 'concat', 'review-only', 'self', 'self-only'.
        aos_predictor: str
            AOS predictor to use. Can be 'non-linear' and 'transr'.
        non_linear: bool
            Whether to use non-linear activation function.
        embedding_type: str
            Type of embedding to use. Can be 'learned' and 'ao_embeddings'.
        kwargs: dict
            Additional arguments, such the learned embeddings.
        """

    def __init__(self, g, n_nodes, aggregator, predictor, node_dim,
                 incidence_dict,
                 num_heads, layer_dropout, attention_dropout, preference_module='lightgcn', use_cuda=True,
                 combiner='add', aos_predictor='non-linear', non_linear=False, embedding_type='learned',
                 **kwargs):
        super().__init__()
        from .lightgcn import Model as lightgcn
        self.aggregator = aggregator
        self.embedding_type = embedding_type
        self.predictor = predictor
        self.preference_module = preference_module
        self.node_dim = node_dim
        self.num_heads = num_heads
        self.combiner = combiner

        if embedding_type == 'learned':
            self.node_embedding = nn.Embedding(n_nodes, node_dim)
        elif embedding_type == 'ao_embeddings':
            self.node_embedding = nn.Embedding(n_nodes, node_dim)
            self.learned_embeddings = kwargs['ao_embeddings']

            # Layer to convert learned embeddings to node embeddings
            dims = [self.learned_embeddings.size(-1), 256, 128, self.node_dim]
            self.node_embedding_mlp = nn.Sequential(
                *[nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.Tanh()) for i in range(len(dims)-1)]
            )
        else:
            raise ValueError(f'Invalid embedding type {embedding_type}')

        # Define review aggregation layer
        n_layers = 3
        self.review_conv = HypergraphLayer(incidence_dict, node_dim, non_linear=non_linear, num_layers=n_layers,
                                           dropout=layer_dropout[0])
        # Define review attention layer
        self.review_agg = ReviewConv(aggregator, n_nodes, node_dim, node_dim, num_heads,
                                     feat_drop=layer_dropout[1], attn_drop=attention_dropout)
        # Define dropout
        self.node_dropout = nn.Dropout(layer_dropout[0])

        # Define preference module
        self.lightgcn = lightgcn(g, node_dim, 3, 0)

        # Define out layers
        self.W_s = nn.Linear(node_dim, node_dim, bias=False)
        if aggregator == 'narre':
            self.w_0 = nn.Linear(node_dim, node_dim)

        # Define combiner
        final_dim = node_dim
        assert combiner in ['add', 'mul', 'bi-interaction', 'concat', 'review-only', 'self', 'self-only']
        if combiner in ['concat', 'self']:
            final_dim *= 2  # Increases out embeddings
        elif combiner == 'bi-interaction':
            # Add and multiply MLPs
            self.add_mlp = nn.Sequential(
                nn.Linear(node_dim, node_dim),
                nn.Tanh()
            )
            self.mul_mlp = nn.Sequential(
                nn.Linear(node_dim, node_dim),
                nn.Tanh()
            )

        # Define predictor
        if self.predictor == 'narre':
            self.edge_predictor = dgl.nn.EdgePredictor('ele', final_dim, 1, bias=True)
            self.bias = nn.Parameter(torch.zeros((n_nodes, 1)))

        # Define aos predictor
        self.aos_predictor = AOSPredictionLayer(aos_predictor, node_dim, final_dim, [node_dim, 64, 32], 2,
                                                loss='transr')

        # Define loss functions
        self.rating_loss_fn = nn.MSELoss(reduction='mean')
        self.bpr_loss_fn = nn.Softplus()

        # Define embeddings used on inference.
        self.review_embs = None
        self.inf_emb = None
        self.lemb = None
        self.first = True
        self.review_attention = None
        self.ui_emb = None
        self.aos_emb = None

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        for name, parameter in self.named_parameters():
            if name.endswith('bias'):
                nn.init.constant_(parameter, 0)
            else:
                nn.init.xavier_normal_(parameter)

    def get_initial_embedings(self, nodes=None):
        """
        Get initial embeddings for nodes.
        Parameters
        ----------
        nodes: torch.Tensor, optional
            Nodes to get embeddings for, if none return all.

        Returns
        -------
        torch.Tensor
            Embeddings for nodes.
        """

        if self.embedding_type == 'learned':
            # If all nodes are learned, only use node embeddings
            if nodes is not None:
                return self.node_embedding(nodes)
            else:
                return self.node_embedding.weight
        elif self.embedding_type == 'ao_embeddings':
            # If AO embeddings are prelearned, use them and filter rest.

            # If nodes are given select those, else use all embeddings
            if nodes is not None:
                filter_val = self.node_embedding.weight.size(0)
                mask = nodes >= filter_val
                emb = torch.empty((*nodes.size(), self.node_dim), device=nodes.device)
                emb[~mask] = self.node_embedding(nodes[~mask]) # Get node embeddings from learned embeddings for UI

                # If any nodes are prelearned, get features from these.
                if torch.any(mask):
                    emb[mask] = self.node_embedding_mlp(self.learned_embeddings[nodes[mask]-filter_val])
                return emb
            else:
                # Return all embeddings
                return torch.cat([self.node_embedding.weight,
                                  self.node_embedding_mlp(self.learned_embeddings)], dim=0)
        else:
            raise ValueError(f'Does not support {self.embedding_type}')

    def review_representation(self, x, mask=None):
        """
        Compute review representation.
        Parameters
        ----------
        x: torch.Tensor
            Input features
        mask: torch.Tensor, optional
            Mask to use for training.

        Returns
        -------
        torch.Tensor
            Review representation
        """

        return self.review_conv(x, mask)

    def review_aggregation(self, g, x, attention=False):
        """
        Aggregate reviews.
        Parameters
        ----------
        g: dgl.DGLGraph
            Graph used for aggregation
        x: torch.Tensor
            Input features
        attention: bool, default False
            Whether to return attention.

        Returns
        -------
        torch.Tensor, optional attention
            user or item representation based on reviews. If attention is True, return attention as well.
        """

        # Aggregate reviews
        x = self.review_agg(g, x, attention)

        # Expand if using attention
        if attention:
            x, a = x

        # Sum over heads
        x = x.sum(1)

        # Return attention if needed
        if attention:
            return x, a
        else:
            return x

    def forward(self, blocks, x, input_nodes):
        """
        Forward pass for HypAR model.
        Parameters
        ----------
        blocks: list
            List of blocks for preference module, review module and mask.
        x: torch.Tensor
            Input features
        input_nodes: torch.Tensor
            Nodes to use for input.

        Returns
        -------
        torch.Tensor, torch.Tensor
            Node representations used for AOS and node representations for prediction.
        """

        # Compute preference embeddings
        blocks, lgcn_blocks, mask = blocks

        # First L-1 blocks for LightGCN are used for convolutions. Last maps review and preference blocks.
        if self.preference_module == 'lightgcn':
            u, i, _ = self.lightgcn(lgcn_blocks[:-1])
            e = {'user': u, 'item': i}
        elif self.preference_module == 'mf':
            # Get user/item representation without any graph convolutions.
            # Use srcdata from last block to get user/item embeddings.
            e = {ntype: self.lightgcn.features[ntype](nids) for ntype, nids in
                  lgcn_blocks[-1].srcdata[dgl.NID].items() if ntype != 'node'}
        else:
            raise NotImplementedError(f'{self.preference_module} is not supported')

        # Move all nodes into same sorting (non-typed) as reviews does not divide user/item by type.
        g = lgcn_blocks[-1]
        with g.local_scope():
            g.srcdata['h'] = e
            funcs = {etype: (fn.copy_u('h', 'm'), fn.sum('m', 'h')) for etype in g.etypes}
            g.multi_update_all(funcs, 'sum')
            e = g.dstdata['h']['node']

        # Compute review embeddings
        x = self.node_dropout(x)
        node_representation, r_ui = self.review_representation(x, mask)

        # Aggregate reviews
        b, = blocks
        r_ui = r_ui[b.srcdata[dgl.NID]]
        r_n = self.review_aggregation(b, r_ui)  # Node representation from reviews

        # Dropout
        r_n, e = self.node_dropout(r_n), self.node_dropout(e)

        # Combine preference and explainability
        if self.combiner == 'concat':
            e_star = torch.cat([r_n, e], dim=-1)
        elif self.combiner == 'add':
            e_star = r_n + e
        elif self.combiner == 'bi-interaction':
            a = self.add_mlp(r_n + e)
            m = self.mul_mlp(r_n * e)
            e_star = a + m
        elif self.combiner == 'mul':
            e_star = r_n * e
        elif self.combiner == 'review-only':
            e_star = r_n
        elif self.combiner == 'self':
            e_star = torch.cat([r_n, node_representation[b.dstdata[dgl.NID]]], dim=-1)
        elif self.combiner == 'self-only':
            e_star = node_representation[b.dstdata[dgl.NID]]

        return node_representation, e_star

    def _graph_predict_dot(self, g: dgl.DGLGraph, x):
        # Dot product prediction
        with g.local_scope():
            g.ndata['h'] = x
            g.apply_edges(fn.u_dot_v('h', 'h', 'm'))

            return g.edata['m'].reshape(-1, 1)

    def _graph_predict_narre(self, g: dgl.DGLGraph, x):
        # Narre prediction methodology
        with g.local_scope():
            g.ndata['b'] = self.bias[g.ndata[dgl.NID]]
            g.apply_edges(fn.u_add_v('b', 'b', 'b'))  # user/item bias

            u, v = g.edges()
            x = self.edge_predictor(x[u], x[v])
            out = x + g.edata['b']

            return out

    def graph_predict(self, g: dgl.DGLGraph, x):
        # Predict using graph
        if self.predictor == 'dot':
            return self._graph_predict_dot(g, x)
        elif self.predictor == 'narre':
            return self._graph_predict_narre(g, x)
        else:
            raise ValueError(f'Predictor not implemented for "{self.predictor}".')

    def aos_graph_predict(self, g: dgl.DGLGraph, node_rep, e_star):
        """
        AOS graph prediction.
        Parameters
        ----------
        g: dgl.DGLGraph
            Graph to use for prediction. Should have edata['pos'] and edata['neg'] representing positive and negative
            aspect and opinion pairs.
        node_rep: torch.Tensor
            Node representation for AO representation.
        e_star: torch.Tensor
            Node representation for user/item.

        Returns
        -------
        torch.Tensor
            Loss of prediction.
        """
        with g.local_scope():
            # Get user/item embeddings
            u, v = g.edges()
            u_emb, i_emb = e_star[u], e_star[v]

            # Get positive a/o embeddings.
            a, o, s = g.edata['pos'].T
            a_emb, o_emb = node_rep[a], node_rep[o]

            # Predict using AOS predictor
            preds_i = self.aos_predictor(u_emb, i_emb, a_emb, o_emb, s)

            # Get negative a/o embeddings
            a, o, s = g.edata['neg'].permute(2, 0, 1)
            a_emb, o_emb = node_rep[a], node_rep[o]

            # Predict using AOS predictor
            preds_j = self.aos_predictor(u_emb, i_emb, a_emb, o_emb, s)

            # Calculate loss using bpr or transr loss (order differs).
            if self.aos_predictor.loss == 'bpr':
                return self.bpr_loss_fn(- (preds_i - preds_j)), preds_i > preds_j
            else:
                return self.bpr_loss_fn(- (preds_j - preds_i)), preds_i < preds_j

    def _predict_dot(self, u_emb, i_emb):
        # Predict using dot
        return (u_emb * i_emb).sum(-1)

    def _predict_narre(self, user, item, u_emb, i_emb):
        # Predict using narre
        h = self.edge_predictor(u_emb, i_emb)
        h += (self.bias[user] + self.bias[item])

        return h.reshape(-1, 1)

    def _combine(self, user, item):
        # Use embeddings computed using self.inference
        u_emb, i_emb = self.inf_emb[user], self.inf_emb[item]  # review user/item embedding
        lu_emb, li_emb = self.lemb[user], self.lemb[item]  # preference user/item embedding, e.g., lightgcn

        # Depending on combiner, combine embeddings
        # if using self or self-only, then lu/li_emb are based on explainability module only, not preference module.
        if self.combiner in ['concat', 'self']:
            u_emb = torch.cat([u_emb, lu_emb], dim=-1)
            i_emb = torch.cat([i_emb, li_emb], dim=-1)
        elif self.combiner == 'add':
            u_emb += lu_emb
            i_emb += li_emb
        elif self.combiner == 'bi-interaction':
            a = self.add_mlp(u_emb + lu_emb)
            m = self.mul_mlp(u_emb * lu_emb)
            u_emb = a + m
            a = self.add_mlp(i_emb + li_emb)
            m = self.mul_mlp(i_emb * li_emb)
            i_emb = a + m
        elif self.combiner == 'mul':
            u_emb *= lu_emb
            i_emb *= li_emb
        elif self.combiner == 'review-only':
            pass
        elif self.combiner == 'self-only':
            u_emb, i_emb = lu_emb, li_emb

        # Return user item embeddings
        return u_emb, i_emb

    def predict(self, user, item):
        """
        Predict using model.
        Parameters
        ----------
        user: torch.Tensor
            User ids
        item: torch.Tensor
            Item ids

        Returns
        -------
        torch.Tensor
            Predicted ranking/rating.
        """
        u_emb, i_emb = self._combine(user, item)

        if self.predictor == 'dot':
            pred = self._predict_dot(u_emb, i_emb)
        elif self.predictor == 'narre':
            pred = self._predict_narre(user, item, u_emb, i_emb)
        else:
            raise ValueError(f'Predictor not implemented for "{self.predictor}".')

        return pred

    def rating_loss(self, preds, target):
        return self.rating_loss_fn(preds, target.unsqueeze(-1))

    def ranking_loss(self, preds_i, preds_j, loss_fn='bpr'):
        if loss_fn == 'bpr':
            loss = self.bpr_loss_fn(- (preds_i - preds_j))
        else:
            raise NotImplementedError

        return loss.mean()

    def inference(self, node_review_graph, ui_graph, device, batch_size):
        """
        Inference for HypAR model.
        Parameters
        ----------
        node_review_graph: dgl.DGLGraph
            Graph mapping reviews to nodes.
        ui_graph: dgl.DGLGraph
            Graph with user/item mappings
        device: str
            Device to use for inference.
        batch_size: int
            Batch size to use for inference.

        Returns
        -------
        None
        """

        # Review inference. nx is the node representation.
        x = self.get_initial_embedings()
        nx, self.review_embs = self.review_representation(x)

        # Node inference setup
        indices = {'node': node_review_graph.nodes('node')}
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader = dgl.dataloading.DataLoader(node_review_graph, indices, sampler, batch_size=batch_size, shuffle=False,
                                                drop_last=False, device=device)

        # Initialize embeddings
        self.inf_emb = torch.zeros((torch.max(indices['node'])+1, self.node_dim)).to(device)
        self.review_attention = torch.zeros((node_review_graph.num_edges(), self.review_agg._num_heads, 1)).to(device)

        # Aggregate reviews using attention
        for input_nodes, output_nodes, blocks in dataloader:
            x, a = self.review_aggregation(blocks[0]['part_of'], self.review_embs[input_nodes['review']], True)
            self.inf_emb[output_nodes['node']] = x
            self.review_attention[blocks[0]['part_of'].edata[dgl.EID]] = a

        # Node preference embedding
        if self.preference_module == 'lightgcn':
            # Move ui_graph to the same device as the model to avoid device mismatch
            ui_graph = ui_graph.to(device)
            u, i, _ = self.lightgcn(ui_graph)
            x = {'user': u, 'item': i}
        else:
            x = {nt: e.weight for nt, e in self.lightgcn.features.items()}

        # Combine/stack useritem embeddings
        if self.combiner.startswith('self'):
            x = nx
        else:
            x = torch.cat([x['item'], x['user']], dim=0)

        # Set embeddings for prediction
        self.lemb = x




