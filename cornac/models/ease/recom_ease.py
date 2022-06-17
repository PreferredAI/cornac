import numpy as np

from cornac.models.recommender import Recommender
from cornac.exception import ScoreException

class EASE(Recommender):
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
        self.U = self.train_set.matrix

        # Gram matrix is X^t X, compute dot product
        G = self.U.T.dot(self.U).toarray()

        diag_indices = np.diag_indices(G.shape[0])

        G[diag_indices] = G.diagonal() + self.lamb

        P = np.linalg.inv(G)

        B = P / (-np.diag(P))
        
        B[diag_indices] = 0.0

        # if self.posB remove -ve values
        if self.posB:
            B[B<0]=0            

        # save B for predictions
        self.B=B
      
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
        if item_idx is None:
            if self.train_set.is_unk_user(user_idx):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d)" % user_idx
                )

            known_item_scores = self.U[user_idx, :].dot(self.B)
            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(
                item_idx
            ):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )

            user_pred = self.B[item_idx, :].dot(self.U[user_idx, :])

            return user_pred