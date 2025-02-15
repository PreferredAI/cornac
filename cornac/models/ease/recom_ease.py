import numpy as np

from ..recommender import Recommender
from ..recommender import ANNMixin, MEASURE_DOT
from ...exception import ScoreException


class EASE(Recommender, ANNMixin):
    """Embarrassingly Shallow Autoencoders for Sparse Data.

    Parameters
    ----------
    name: string, optional, default: 'EASEᴿ'
        The name of the recommender model.

    lamb: float, optional, default: 500
        L2-norm regularization-parameter λ ∈ R+.
    
    posB: boolean, optional, default: False
        Remove Negative Weights

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already \
        trained. 
        
    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    seed: int, optional, default: None
        Random seed for parameters initialization.

    References
    ----------
    * Steck, H. (2019, May). "Embarrassingly shallow autoencoders for sparse data." \
    In The World Wide Web Conference (pp. 3251-3257).
    """

    def __init__(
        self,
        name="EASEᴿ",
        lamb=500,
        posB=True,
        trainable=True,
        verbose=True,
        seed=None,
        B=None,
        U=None,
    ):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.lamb = lamb
        self.posB = posB
        self.verbose = verbose
        self.seed = seed
        self.B = B
        self.U = U

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """
        Recommender.fit(self, train_set, val_set)

        # A rating matrix
        self.U = train_set.matrix

        # Gram matrix is X^t X, compute dot product
        G = self.U.T.dot(self.U).toarray()

        diag_indices = np.diag_indices(G.shape[0])

        G[diag_indices] = G.diagonal() + self.lamb

        P = np.linalg.inv(G)

        B = P / (-np.diag(P))

        B[diag_indices] = 0.0

        # if self.posB remove -ve values
        if self.posB:
            B[B < 0] = 0

        # save B for predictions
        self.B = B

        return self

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
        if self.is_unknown_user(user_idx):
            raise ScoreException("Can't make score prediction for user %d" % user_idx)

        if item_idx is not None and self.is_unknown_item(item_idx):
            raise ScoreException("Can't make score prediction for item %d" % item_idx)

        if item_idx is None:
            return self.U[user_idx, :].dot(self.B)

        return self.U[user_idx, :].dot(self.B[:, item_idx])

    def get_vector_measure(self):
        """Getting a valid choice of vector measurement in ANNMixin._measures.

        Returns
        -------
        measure: MEASURE_DOT
            Dot product aka. inner product
        """
        return MEASURE_DOT

    def get_user_vectors(self):
        """Getting a matrix of user vectors serving as query for ANN search.

        Returns
        -------
        out: numpy.array
            Matrix of user vectors for all users available in the model.
        """
        return self.U

    def get_item_vectors(self):
        """Getting a matrix of item vectors used for building the index for ANN search.

        Returns
        -------
        out: numpy.array
            Matrix of item vectors for all items available in the model.
        """
        return self.B
