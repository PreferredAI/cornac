import numpy as np

from cornac.models.recommender import Recommender
from cornac.exception import ScoreException

import scipy.sparse as sps
import scipy.stats as stats
import numpy as np

class zscoREC(Recommender):
    """ImposeSVD: Incrementing PureSVD For Top-N Recommendations for Cold-Start Problems and Sparse Datasets.

    Parameters
    ----------
    name: string, optional, default: 'zscoREC'
        The name of the recommender model.

    lamb: float, optional, default: .2
        Shifting parameter λ ∈ R
    
    posZ: boolean, optional, default: False
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
    * Hakan Yilmazer, Selma Ayşe Özel, ImposeSVD: Incrementing PureSVD For Top-N Recommendations for Cold-Start Problems and Sparse Datasets, 
    The Computer Journal, 2022;, bxac106, https://doi.org/10.1093/comjnl/bxac106.
    """

    def __init__(
            self,
            name="zscoREC",
            lamb=.2,
            posZ=True,
            trainable=True,
            verbose=True,
            seed=None,
            Z=None,
            U=None,
    ):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.lamb = lamb
        self.posZ = posZ
        self.verbose = verbose
        self.seed = seed
        self.Z = Z
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

        # Gram matrix is X^t X, Eq.(2) in paper
        G = sps.csr_matrix(self.U.T.dot(self.U), dtype="uint64")
        
        # Shifting Gram matrix Eq.(7) in paper
        """ In a structure view, the shifting operation on the second
            matrix (which is the same as the matrix for the Gram-matrix
            estimation) is performed as row-based degree shifting on
            Gram-matrix.
        """    
        W = G - np.tile(G.diagonal()*self.lam, (G.shape[0],1)).transpose()

        # Column-based z-score normalization
        # fit each item's column (could be parallel for big streams)
        Z = stats.mstats.zscore(np.array(W), axis=0, ddof=1, nan_policy='omit')

        # if self.posZ remove -ve values
        if self.posZ:
            Z[Z<0]=0            

        # save Z for predictions
        self.Z = Z
      
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

            known_item_scores = self.U[user_idx, :].dot(self.Z)
            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(
                item_idx
            ):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )

            user_pred = self.Z[item_idx, :].dot(self.U[user_idx, :])

            return user_pred