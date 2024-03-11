import dgl.utils
import torch
from dgl.ops import edge_softmax
from torch import nn
import dgl.function as fn
import dgl.sparse as dglsp


class AOSPredictionLayer(nn.Module):
    def __init__(self, aos_predictor, in_dim1, in_dim2, hidden_dims, n_relations, loss='bpr'):
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
        u_emb
        i_emb
        a_emb
        o_emb
        s

        Returns
        -------

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
    def __init__(self, H, in_dim, non_linear=True, num_layers=1, dropout=0, aggregator='sum',
                 normalize=False):
        super().__init__()
        self.aggregator = aggregator
        self.non_linear = non_linear
        self.normalize = normalize

        self.H = None
        self.D_e_inv = None
        self.L_left = None
        self.L_right = None
        self.L = None
        self.O = None
        self.N = None
        self.D_v_invsqrt = None
        self.heads = None
        self.tails = None
        self.edges = None
        self.uniques = None

        self.set_matrices(H)

        self.num_layers = num_layers
        self.in_dim = in_dim
        self.W = nn.ModuleList([
                nn.ModuleDict({
                    k: nn.Linear(in_dim, in_dim) for k in H
                }) for _ in range(num_layers)
            ])

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

    def set_matrices(self, H):
        self.H = H
        d_V = {k: v.sum(1) for k, v in H.items()}
        d_E = {k: v.sum(0) for k, v in H.items()}
        self.N = {}
        for k in H:
            nz = d_V[k].nonzero()
            d_V[k][nz] = d_V[k][nz] ** -.5
            nz = d_E[k].nonzero()
            d_E[k][nz] = d_E[k][nz] ** -1
            self.N[k] = H[k] @ H[k].T

        self.D_v_invsqrt = {k: dglsp.diag(v) for k, v in d_V.items()}
        self.D_e_inv = {k: dglsp.diag(v) for k, v in d_E.items()}

        # Compute Laplacian from the equation above.
        self.L_left = {k: self.D_v_invsqrt[k] @ H[k] for k in H}
        self.L_right = {k: H[k].T @ self.D_v_invsqrt[k] for k in H}
        self.L = {k: self.L_left[k] @ self.D_e_inv[k] @ self.L_right[k] for k in H}

        # Out representation
        self.O = {k: self.D_e_inv[k] @ H[k].T for k in H}

        # ## Attention
        # # Get head to tail interactions
        # n = {k: self.H[k] @ self.H[k].T for k in self.H}
        # self.heads = {k: torch.repeat_interleave(n[k].row, n[k].val.to(torch.int64)) for k in n}
        # self.tails = {k: torch.repeat_interleave(n[k].col, n[k].val.to(torch.int64)) for k in n}
        # self.edges = {k: torch.repeat_interleave(self.H[k].col, self.H[k].sum(0)[self.H[k].col].to(torch.int64))
        #               for k in self.H}
        #
        # # Find unique head-edge interactions.
        # self.uniques = {k: torch.unique(torch.stack([self.heads[k], self.edges[k]]).T, dim=0, return_inverse=True)
        #                 for k in n}

    def unset_matrices(self):
        self.H = None
        self.D_e_inv = None
        self.L_left = None
        self.L_right = None
        self.L = None
        self.O = None
        self.N = None
        self.D_v_invsqrt = None

    def forward(self, x, mask=None):
        node_out = [x]
        review_out = []
        D_e = self.D_e_inv

        # Mask if in train
        if mask is not None:
            D_e = {k: dglsp.diag(D_e[k].val * mask) for k in D_e}
            L = {k: self.L_left[k]  @ D_e[k] @ self.L_right[k] for k in D_e}
            att_mask = {k: torch.repeat_interleave(mask, self.H[k].sum(0).to(torch.int64)) for k in D_e}
        else:
            L = self.L
            att_mask = None

        for i, layer in enumerate(self.W):
            inner_x = []
            inner_o = []
            for k, l in layer.items():
                e = L[k] @ l(self.dropout(x))

                if self.non_linear:
                    e = self.activation(e)

                o = self.O[k] @ e  # average of nodes participating in review edge

                # if mask is not None:
                #     o = mask * o

                inner_x.append(e)
                inner_o.append(o)

            x = torch.stack(inner_x)
            if self.aggregator == 'sum':
                x = x.sum(0)
                inner_o = torch.stack(inner_o).sum(0)

            else:
                raise NotImplementedError(self.aggregator)

            if self.normalize:
                x = x / (x.norm(2, dim=-1, keepdim=True) + 1e-5)
                inner_o = inner_o / (inner_o.norm(2, dim=-1, keepdim=True) + 1e-5)

            node_out.append(x)
            review_out.append(inner_o)

        return torch.stack(node_out).mean(0), torch.stack(review_out).mean(0)


class HEARConv(nn.Module):
    def __init__(self,
                 aggregator,
                 n_nodes,
                 n_relations,
                 in_feats,
                 attention_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True):
        super(HEARConv, self).__init__()
        self.aggregator = aggregator
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = dgl.utils.expand_as_pair(in_feats)
        self._out_feats = attention_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if self.aggregator != 'narre-rel':
            self.fc_src = nn.Linear(
                self._in_src_feats, attention_feats * num_heads, bias=bias)
        else:
            self.fc_src = nn.Parameter(torch.Tensor(
                n_relations, self._in_src_feats, attention_feats * num_heads
            ))
            self.fc_src_bias = nn.Parameter(torch.Tensor(
                n_relations, attention_feats * num_heads
            ))

        if self.aggregator.startswith('narre'):
            self.node_quality = nn.Embedding(n_nodes, self._in_dst_feats)
            if self.aggregator == 'narre':
                self.fc_qual = nn.Linear(self._in_dst_feats, attention_feats * num_heads, bias=bias)
            elif self.aggregator == 'narre-rel':
                self.fc_qual = nn.Parameter(torch.Tensor(
                    n_relations, self._in_dst_feats, attention_feats * num_heads
                ))
                self.fc_qual_bias = nn.Parameter(torch.Tensor(
                    n_relations, attention_feats * num_heads
                ))
            else:
                raise NotImplementedError(f'Not implemented any aggregator named {self.aggregator}.')
        elif self.aggregator == 'gatv2':
            pass
        else:
            raise NotImplementedError(f'Not implemented any aggregator named {self.aggregator}.')
        if self.aggregator != 'narre-rel':
            self.attn = nn.Parameter(torch.FloatTensor(size=(1, num_heads, attention_feats)))
        else:
            self.attn = nn.Parameter(torch.FloatTensor(size=(n_relations, num_heads, attention_feats)))
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
        r"""
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

            h_src = self.feat_drop(feat)
            if self.aggregator != 'narre-rel':
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                graph.srcdata.update({'el': feat_src})# (num_src_edge, num_heads, out_dim)
            else:
                graph.srcdata.update({'el': h_src})
                graph.apply_edges(self.rel_attention('el', 'r_type', 'el', self.fc_src, self.fc_src_bias))
                graph.edata.update({'el': graph.edata['el'].view(-1, self._num_heads, self._out_feats)})

            if self.aggregator.startswith('narre'):
                h_qual = self.feat_drop(self.node_quality(graph.edata['nid']))
                if self.aggregator != 'narre-rel':
                    feat_qual = self.fc_qual(h_qual).view(-1, self._num_heads, self._out_feats)
                    graph.edata.update({'qual': feat_qual})
                    graph.apply_edges(fn.u_add_e('el', 'qual', 'e'))
                else:
                    graph.edata.update({'qual': h_qual})
                    graph.apply_edges(self.rel_attention('qual', 'r_type', 'qual', self.fc_qual, self.fc_qual_bias,
                                                         False))
                    graph.edata.update({'qual': graph.edata['qual'].view(-1, self._num_heads, self._out_feats)})
                    graph.edata.update({'e': graph.edata.pop('el') + graph.edata.pop('qual')})
            else:
                graph.apply_edges(fn.copy_u('el', 'e'))

            e = self.leaky_relu(graph.edata.pop('e'))# (num_src_edge, num_heads, out_dim)

            if self.aggregator != 'narre-rel':
                e = (e * self.attn).sum(dim=-1).unsqueeze(dim=2)# (num_edge, num_heads, 1)
            else:
                e = (e * self.attn[graph.edata['r_type']]).sum(dim=-1).unsqueeze(dim=2)

            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e)) # (num_edge, num_heads)

            if self.aggregator.startswith('narre'):
                graph.srcdata.update({'el': h_src})

            # message passing
            graph.update_all(fn.u_mul_e('el', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']

            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst


class Model(nn.Module):
    def __init__(self, g, n_nodes, n_lgcn_relations, aggregator, predictor, node_dim,
                 incidence_dict,
                 num_heads, layer_dropout, attention_dropout, preference_module='lightgcn', use_cuda=True,
                 combiner='add', aos_predictor='non-linear', non_linear=False, embedding_type='learned',
                 **kwargs):
        super().__init__()
        from cornac.models.lightgcn.lightgcn import Model as lightgcn
        self.aggregator = aggregator
        self.embedding_type = embedding_type
        self.predictor = predictor
        self.preference_module = preference_module
        self.node_dim = node_dim
        self.num_heads = num_heads

        if embedding_type == 'learned':
            self.node_embedding = nn.Embedding(n_nodes, node_dim)
        elif embedding_type == 'ao_embeddings':
            self.node_embedding = nn.Embedding(n_nodes, node_dim)
            self.learned_embeddings = kwargs['ao_embeddings']
            dims = [self.learned_embeddings.size(-1), 256, 128, self.node_dim]
            self.node_embedding_mlp = nn.Sequential(
                *[nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.Tanh()) for i in range(len(dims)-1)]
            )
        else:
            raise ValueError(f'Invalid embedding type {embedding_type}')

        n_layers = 3
        self.review_conv = HypergraphLayer(incidence_dict, node_dim, non_linear=non_linear, num_layers=n_layers,
                                           dropout=layer_dropout[0])
        self.review_agg = HEARConv(aggregator, n_nodes, n_lgcn_relations, node_dim, node_dim, num_heads,
                                   feat_drop=layer_dropout[1], attn_drop=attention_dropout)

        self.node_dropout = nn.Dropout(layer_dropout[0])


        self.lightgcn = lightgcn(g, node_dim, 3, 0)

        self.W_s = nn.Linear(node_dim, node_dim, bias=False)
        if aggregator.startswith('narre'):
            self.w_0 = nn.Linear(node_dim, node_dim)

        final_dim = node_dim
        self.combiner = combiner
        assert combiner in ['add', 'mul', 'bi-interaction', 'concat', 'review-only', 'self', 'self-only']
        if combiner in ['concat', 'self']:
            final_dim *= 2
        elif combiner == 'bi-interaction':
            self.add_mlp = nn.Sequential(
                nn.Linear(node_dim, node_dim),
                nn.Tanh()
            )
            self.mul_mlp = nn.Sequential(
                nn.Linear(node_dim, node_dim),
                nn.Tanh()
            )

        if self.predictor == 'narre':
            self.edge_predictor = dgl.nn.EdgePredictor('ele', final_dim, 1, bias=True)
            self.bias = nn.Parameter(torch.zeros((n_nodes, 1)))

        self.aos_predictor = AOSPredictionLayer(aos_predictor, node_dim, final_dim, [node_dim, 64, 32], 2,
                                                loss='transr')
        self.rating_loss_fn = nn.MSELoss(reduction='mean')
        self.bpr_loss_fn = nn.Softplus()
        self.review_embs = None
        self.inf_emb = None
        self.lemb = None
        self.first = True
        self.review_attention = None
        self.ui_emb = None
        self.aos_emb = None

        self.reset_parameters()

    def reset_parameters(self):
        for name, parameter in self.named_parameters():
            if name.endswith('bias'):
                nn.init.constant_(parameter, 0)
            else:
                nn.init.xavier_normal_(parameter)

    def get_initial_embedings(self, nodes=None):
        if self.embedding_type == 'learned':
            if nodes is not None:
                return self.node_embedding(nodes)
            else:
                return self.node_embedding.weight
        elif self.embedding_type == 'ao_embeddings':
            if nodes is not None:
                filter_val = self.node_embedding.weight.size(0)
                mask = nodes >= filter_val
                emb = torch.empty((*nodes.size(), self.node_dim), device=nodes.device)
                emb[~mask] = self.node_embedding(nodes[~mask])
                if torch.any(mask):
                    emb[mask] = self.node_embedding_mlp(self.learned_embeddings[nodes[mask]-filter_val])
                return emb
            else:
                if len(self.learned_embeddings):
                    return torch.cat([self.node_embedding.weight, self.node_embedding_mlp(self.learned_embeddings)], dim=0)
                else:
                    return self.node_embedding.weight
        else:
            raise ValueError(f'Does not support {self.embedding_type}')

    def l2_loss(self, pos, neg, emb):
        if isinstance(emb, list):
            emb = torch.cat(emb, dim=-1)

        loss = 0
        src, dst_i = pos.edges()
        _, dst_j = neg.edges()

        s_emb, i_emb, j_emb = emb[src], emb[dst_i], emb[dst_j]

        loss += s_emb.norm(2).pow(2) + i_emb.norm(2).pow(2) + j_emb.norm(2).pow(2)

        loss = 0.5 * loss / pos.num_src_nodes()

        return loss

    def review_representation(self, x, mask=None):
        return self.review_conv(x, mask)

    def review_aggregation(self, g, x, attention=False):
        x = self.review_agg(g, x, attention)

        if attention:
            x, a = x

        x = x.sum(1)

        if attention:
            return x, a
        else:
            return x

    def forward(self, blocks, x, input_nodes):
        # Compute preference embeddings
        blocks, lgcn_blocks, mask = blocks
        if self.preference_module == 'lightgcn':
            u, i, _ = self.lightgcn(lgcn_blocks[:-1])
            e = {'user': u, 'item': i}
        elif self.preference_module == 'mf':
            # Get user/item representation without any graph convolutions.
            e = {ntype: self.lightgcn.features[ntype](nids) for ntype, nids in
                  lgcn_blocks[-1].srcdata[dgl.NID].items() if ntype != 'node'}
        else:
            raise NotImplementedError(f'{self.preference_module} is not supported')

        # Move all nodes into same sorting (non-typed)
        g = lgcn_blocks[-1]
        with g.local_scope():
            g.srcdata['h'] = e
            funcs = {etype: (fn.copy_u('h', 'm'), fn.sum('m', 'h')) for etype in g.etypes}
            g.multi_update_all(funcs, 'sum')
            e = g.dstdata['h']['node']

        # Compute review embeddings
        x = self.node_dropout(x)
        node_representation, r_ui = self.review_representation(x, mask)

        b, = blocks
        r_ui = r_ui[b.srcdata[dgl.NID]]
        r_n = self.review_aggregation(b, r_ui)  # Node representation from reviews

        r_n, e = self.node_dropout(r_n), self.node_dropout(e)

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
        with g.local_scope():
            g.ndata['h'] = x
            g.apply_edges(fn.u_dot_v('h', 'h', 'm'))

            return g.edata['m'].reshape(-1, 1)

    def _graph_predict_narre(self, g: dgl.DGLGraph, x):
        with g.local_scope():
            g.ndata['b'] = self.bias[g.ndata[dgl.NID]]
            g.apply_edges(fn.u_add_v('b', 'b', 'b'))  # user/item bias

            u, v = g.edges()
            x = self.edge_predictor(x[u], x[v])
            out = x + g.edata['b']

            return out

    def graph_predict(self, g: dgl.DGLGraph, x):
        if self.predictor == 'dot':
            return self._graph_predict_dot(g, x)
        elif self.predictor == 'narre':
            return self._graph_predict_narre(g, x)
        else:
            raise ValueError(f'Predictor not implemented for "{self.predictor}".')

    def aos_graph_predict(self, g: dgl.DGLGraph, node_rep, e_star):
        with g.local_scope():
            u, v = g.edges()
            u_emb, i_emb = e_star[u], e_star[v]
            a, o, s = g.edata['pos'].T  # todo reindex a and o to be proper values.
            a_emb, o_emb = node_rep[a], node_rep[o]
            preds_i = self.aos_predictor(u_emb, i_emb, a_emb, o_emb, s)
            a, o, s = g.edata['neg'].permute(2, 0, 1)
            a_emb, o_emb = node_rep[a], node_rep[o]
            preds_j = self.aos_predictor(u_emb, i_emb, a_emb, o_emb, s)
            if self.aos_predictor.loss == 'bpr':
                return self.bpr_loss_fn(- (preds_i - preds_j)), preds_i > preds_j
            else:
                return self.bpr_loss_fn(- (preds_j - preds_i)), preds_i < preds_j

    def _predict_dot(self, u_emb, i_emb):
        return (u_emb * i_emb).sum(-1)

    def _predict_narre(self, user, item, u_emb, i_emb):
        h = self.edge_predictor(u_emb, i_emb)
        h += (self.bias[user] + self.bias[item])

        return h.reshape(-1, 1)

    def aos_predict(self, user, item, aspect, opinion, sentiment):
        u_emb, i_emb = self._combine(user, item)

        a_emb, o_emb = self.get_initial_embedings(aspect), self.get_initial_embedings(opinion)

        preds = self.aos_predictor(u_emb, i_emb, a_emb, o_emb, sentiment)

        return preds

    def _combine(self, user, item):
        u_emb, i_emb = self.inf_emb[user], self.inf_emb[item]
        lu_emb, li_emb = self.lemb[user], self.lemb[item]

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

        return u_emb, i_emb

    def predict(self, user, item):
        # u_emb, i_emb = self.inf_emb[user], self.inf_emb[item]
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

    def _bpr_loss(self, preds_i, preds_j):
        return

    def ranking_loss(self, preds_i, preds_j, loss_fn='bpr'):
        if loss_fn == 'bpr':
            loss = self.bpr_loss_fn(- (preds_i - preds_j))
        else:
            raise NotImplementedError

        return loss.mean()

    def inference(self, review_graphs, node_review_graph, ui_graph, device, batch_size):

        # Review inference
        x = self.get_initial_embedings()
        nx, self.review_embs = self.review_representation(x)

        # Node inference setup
        indices = {'node': node_review_graph.nodes('node')}
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader = dgl.dataloading.DataLoader(node_review_graph, indices, sampler, batch_size=batch_size, shuffle=False,
                                                drop_last=False, device=device)

        self.inf_emb = torch.zeros((torch.max(indices['node'])+1, self.node_dim)).to(device)
        self.review_attention = torch.zeros((node_review_graph.num_edges(), self.review_agg._num_heads, 1)).to(device)

        # Node inference
        for input_nodes, output_nodes, blocks in dataloader:
            x, a = self.review_aggregation(blocks[0]['part_of'], self.review_embs[input_nodes['review']], True)
            self.inf_emb[output_nodes['node']] = x
            self.review_attention[blocks[0]['part_of'].edata[dgl.EID]] = a

        # Node preference embedding
        if self.preference_module == 'lightgcn':
            u, i, _ = self.lightgcn(ui_graph)
            x = {'user': u, 'item': i}
        else:
            x = {nt: e.weight for nt, e in self.lightgcn.features.items()}

        if self.combiner.startswith('self'):
            x = nx
        else:
            x = torch.cat([x['item'], x['user']], dim=0)
        self.lemb = x




