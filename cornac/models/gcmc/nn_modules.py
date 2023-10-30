"""Neural Network modules"""
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from dgl import DGLError
import torch
import torch.nn as nn
from torch.nn import init

from .utils import get_activation


class NeuralNetwork(nn.Module):
    """
    Base class for all neural network modules.
    """

    def __init__(
        self,
        activation_func,
        rating_values,
        n_users,
        n_items,
        gcn_agg_units,
        gcn_out_units,
        gcn_dropout,
        gcn_agg_accum,
        gen_r_num_basis_func,
        share_param,
        device,
    ):
        super(NeuralNetwork, self).__init__()
        self._act = get_activation(activation_func)
        self.encoder = GCMCLayer(
            rating_values,
            n_users,
            n_items,
            gcn_agg_units,
            gcn_out_units,
            gcn_dropout,
            gcn_agg_accum,
            agg_act=self._act,
            share_user_item_param=share_param,
            device=device,
        )
        self.decoder = BiDecoder(
            in_units=gcn_out_units,
            num_classes=len(rating_values),
            num_basis=gen_r_num_basis_func,
        )

    def forward(self, enc_graph, dec_graph, ufeat=None, ifeat=None):
        """Forward computation

        Parameters
        ----------
        enc_graph : DGLGraph
            The graph for encoding
        dec_graph : DGLGraph
            The graph for decoding
        ufeat : torch.Tensor
            The input user feature
        ifeat : torch.Tensor
            The input item feature
        """
        user_out, item_out = self.encoder(enc_graph, ufeat, ifeat)
        pred_ratings = self.decoder(dec_graph, user_out, item_out)
        return pred_ratings


class GCMCGraphConv(nn.Module):
    """Graph convolution module used in the GCMC model.

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    weight : bool, optional
        If True, apply a linear layer. Otherwise, aggregating the messages
        without a weight matrix or with an shared weight provided by caller.
    device: str, optional
        Which device to put data in. Useful in mix_cpu_gpu training and
        multi-gpu training
    """

    def __init__(self, in_feats, out_feats, weight=True, device=None, dropout_rate=0.0):
        super(GCMCGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)

        if weight:
            self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            init.xavier_uniform_(self.weight)

    def forward(self, graph, feat, weight=None):
        """Compute graph convolution.

        Normalizer constant :math:`c_{ij}` is stored as two node data "ci"
        and "cj".

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature
        weight : torch.Tensor, optional
            Optional external weight tensor.
        dropout : torch.nn.Dropout, optional
            Optional external dropout layer.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat, _ = feat  # dst feature not used
            c_j = graph.srcdata["cj"]
            c_i = graph.dstdata["ci"]
            if self.device is not None:
                c_j = c_j.to(self.device)
                c_i = c_i.to(self.device)
            if weight is not None:
                if self.weight is not None:
                    raise DGLError(
                        "External weight is provided while at the same time"
                        "the module has defined its own weight parameter."
                        "Please create the module with flag weight=False."
                    )
            else:
                weight = self.weight

            if weight is not None:
                feat = dot_or_identity(feat, weight, self.device)

            feat = feat * self.dropout(c_j)
            graph.srcdata["h"] = feat
            graph.update_all(fn.copy_u(u="h", out="m"), fn.sum(msg="m", out="h"))
            rst = graph.dstdata["h"]
            rst = rst * c_i

        return rst


class GCMCLayer(nn.Module):
    r"""GCMC layer

    .. math::
        z_j^{(l+1)} = \sigma_{agg}\left[\mathrm{agg}\left(
        \sum_{j\in\mathcal{N}_1}\frac{1}{c_{ij}}W_1h_j, \ldots,
        \sum_{j\in\mathcal{N}_R}\frac{1}{c_{ij}}W_Rh_j
        \right)\right]

    After that, apply an extra output projection:

    .. math::
        h_j^{(l+1)} = \sigma_{out}W_oz_j^{(l+1)}

    The equation is applied to both user nodes and item nodes and the
    parameters are not shared unless ``share_user_item_param`` is true.

    Parameters
    ----------
    rating_vals : list of int or float
        Possible rating values.
    user_in_units : int
        Size of user input feature
    item_in_units : int
        Size of item input feature
    msg_units : int
        Size of message :math:`W_rh_j`
    out_units : int
        Size of of final output user and item features
    dropout_rate : float, optional
        Dropout rate (Default: 0.0)
    agg : str, optional
        Function to aggregate messages of different ratings.
        Could be any of the supported cross type reducers:
        "sum", "max", "min", "mean", "stack".
        (Default: "stack")
    agg_act : callable, str, optional
        Activation function :math:`sigma_{agg}`. (Default: None)
    out_act : callable, str, optional
        Activation function :math:`sigma_{agg}`. (Default: None)
    share_user_item_param : bool, optional
        If true, user node and item node share the same set of parameters.
        Require ``user_in_units`` and ``move_in_units`` to be the same.
        (Default: False)
    device: str, optional
        Which device to put data in. Useful in mix_cpu_gpu training and
        multi-gpu training
    """

    def __init__(
        self,
        rating_vals,
        user_in_units,
        item_in_units,
        msg_units,
        out_units,
        dropout_rate=0.0,
        agg="stack",  # or 'sum'
        agg_act=None,
        out_act=None,
        share_user_item_param=False,
        device=None,
    ):
        super(GCMCLayer, self).__init__()
        self.rating_vals = rating_vals
        self.agg = agg
        self.share_user_item_param = share_user_item_param
        self.ufc = nn.Linear(msg_units, out_units)
        if share_user_item_param:
            self.ifc = self.ufc
        else:
            self.ifc = nn.Linear(msg_units, out_units)
        if agg == "stack":
            # divide the original msg unit size by number of ratings to keep
            # the dimensionality
            assert msg_units % len(rating_vals) == 0
            msg_units = msg_units // len(rating_vals)
        self.dropout = nn.Dropout(dropout_rate)
        self.w_r = nn.ParameterDict()
        sub_conv = {}
        for rating in rating_vals:
            # PyTorch parameter name can't contain "."
            rating = str(rating).replace(".", "_")
            rev_rating = f"rev-{rating}"
            if share_user_item_param and user_in_units == item_in_units:
                self.w_r[rating] = nn.Parameter(torch.randn(user_in_units, msg_units))
                self.w_r[f"rev-{rating}"] = self.w_r[rating]
                sub_conv[rating] = GCMCGraphConv(
                    user_in_units,
                    msg_units,
                    weight=False,
                    device=device,
                    dropout_rate=dropout_rate,
                )
                sub_conv[rev_rating] = GCMCGraphConv(
                    user_in_units,
                    msg_units,
                    weight=False,
                    device=device,
                    dropout_rate=dropout_rate,
                )
            else:
                self.w_r = None
                sub_conv[rating] = GCMCGraphConv(
                    user_in_units,
                    msg_units,
                    weight=True,
                    device=device,
                    dropout_rate=dropout_rate,
                )
                sub_conv[rev_rating] = GCMCGraphConv(
                    item_in_units,
                    msg_units,
                    weight=True,
                    device=device,
                    dropout_rate=dropout_rate,
                )
        self.conv = dglnn.HeteroGraphConv(sub_conv, aggregate=agg)
        self.agg_act = get_activation(agg_act)
        self.out_act = get_activation(out_act)
        self.device = device
        self.reset_parameters()

    def partial_to(self, device):
        """
        Put parameters into device except W_r

        Parameters
        ----------
        device : torch device
            Which device the parameters are put in.
        """
        assert device == self.device
        if device is not None:
            self.ufc.cuda(device)
            if self.share_user_item_param is False:
                self.ifc.cuda(device)
            self.dropout.cuda(device)

    def reset_parameters(self):
        """Reset parameters to uniform distribution"""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, graph, ufeat=None, ifeat=None):
        """
        Forward function

        Parameters
        ----------
        graph : DGLGraph
            User-item rating graph. It should contain two node types: "user"
            and "item" and many edge types each for one rating value.
        ufeat : torch.Tensor, optional
            User features. If None, using an identity matrix.
        ifeat : torch.Tensor, optional
            Item features. If None, using an identity matrix.

        Returns
        -------
        new_ufeat : torch.Tensor
            New user features
        new_ifeat : torch.Tensor
            New item features
        """
        in_feats = {"user": ufeat, "item": ifeat}
        mod_args = {}
        for rating in self.rating_vals:
            rating = str(rating).replace(".", "_")
            rev_rating = f"rev-{rating}"
            mod_args[rating] = (self.w_r[rating] if self.w_r is not None else None,)
            mod_args[rev_rating] = (
                self.w_r[rev_rating] if self.w_r is not None else None,
            )
        out_feats = self.conv(graph, in_feats, mod_args=mod_args)
        ufeat = out_feats["user"]
        ifeat = out_feats["item"]
        ufeat = ufeat.view(ufeat.shape[0], -1)
        ifeat = ifeat.view(ifeat.shape[0], -1)

        # fc and non-linear
        ufeat = self.agg_act(ufeat)
        ifeat = self.agg_act(ifeat)
        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        ufeat = self.ufc(ufeat)
        ifeat = self.ifc(ifeat)
        return self.out_act(ufeat), self.out_act(ifeat)


class BiDecoder(nn.Module):
    r"""
    Bi-linear decoder.

    Given a bipartite graph G, for each edge (i, j) ~ G, compute the likelihood
    of it being class r by:

    .. math::
        p(M_{ij}=r) = \text{softmax}(u_i^TQ_rv_j)

    The trainable parameter :math:`Q_r` is further decomposed to a linear
    combination of basis weight matrices :math:`P_s`:

    .. math::
        Q_r = \sum_{s=1}^{b} a_{rs}P_s

    Parameters
    ----------
    in_units : int
        Size of input user and item features
    num_classes : int
        Number of classes.
    num_basis : int, optional
        Number of basis. (Default: 2)
    dropout_rate : float, optional
        Dropout raite (Default: 0.0)
    """

    def __init__(self, in_units, num_classes, num_basis=2, dropout_rate=0.0):
        super(BiDecoder, self).__init__()
        self._num_basis = num_basis
        self.dropout = nn.Dropout(dropout_rate)
        self.params = nn.ParameterList(
            nn.Parameter(torch.randn(in_units, in_units)) for _ in range(num_basis)
        )
        self.combine_basis = nn.Linear(self._num_basis, num_classes, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters to uniform distribution"""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, graph, ufeat, ifeat):
        """
        Forward function.

        Parameters
        ----------
        graph : DGLGraph
            "Flattened" user-item graph with only one edge type.
        ufeat : th.Tensor
            User embeddings. Shape: (|V_u|, D)
        ifeat : th.Tensor
            Item embeddings. Shape: (|V_m|, D)

        Returns
        -------
        torch.Tensor
            Predicting scores for each user-item edge.
        """
        with graph.local_scope():
            ufeat = self.dropout(ufeat)
            ifeat = self.dropout(ifeat)
            graph.nodes["item"].data["h"] = ifeat
            basis_out = []
            for i in range(self._num_basis):
                graph.nodes["user"].data["h"] = ufeat @ self.params[i]
                graph.apply_edges(fn.u_dot_v("h", "h", "sr"))
                basis_out.append(graph.edata["sr"])
            out = torch.cat(basis_out, dim=1)
            out = self.combine_basis(out)
        return out


class DenseBiDecoder(nn.Module):
    r"""
    Dense bi-linear decoder.

    Dense implementation of the bi-linear decoder used in GCMC. Suitable when
    the graph can be efficiently represented by a pair of arrays (one for
    source nodes; one for destination nodes).

    Parameters
    ----------
    in_units : int
        Size of input user and item features
    num_classes : int
        Number of classes.
    num_basis : int, optional
        Number of basis. (Default: 2)
    dropout_rate : float, optional
        Dropout raite (Default: 0.0)
    """

    def __init__(self, in_units, num_classes, num_basis=2, dropout_rate=0.0):
        super().__init__()
        self._num_basis = num_basis
        self.dropout = nn.Dropout(dropout_rate)
        self.P = nn.Parameter(torch.randn(num_basis, in_units, in_units))
        self.combine_basis = nn.Linear(self._num_basis, num_classes, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters to uniform distribution"""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, ufeat, ifeat):
        """
        Forward function.

        Compute logits for each pair ``(ufeat[i], ifeat[i])``.

        Parameters
        ----------
        ufeat : th.Tensor
            User embeddings. Shape: (B, D)
        ifeat : th.Tensor
            Item embeddings. Shape: (B, D)

        Returns
        -------
        torch.Tensor
            Predicting scores for each user-item edge. Shape: (B, num_classes)
        """
        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        out = torch.einsum("ai,bij,aj->ab", ufeat, self.P, ifeat)
        out = self.combine_basis(out)
        return out


def dot_or_identity(A, B, device=None):
    """
    Return as identity matrix if A is none.
    If A exists, return as dot product.
    """
    if A is None:
        return B
    if len(A.shape) == 1:
        if device is None:
            return B[A]
        return B[A].to(device)
    return A @ B
