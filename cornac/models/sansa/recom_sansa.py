import numpy as np
import scipy.sparse as sp

from ..recommender import Recommender
from ..recommender import ANNMixin, MEASURE_DOT
from ...exception import ScoreException


class SANSA(Recommender, ANNMixin):
    """Scalable Approximate NonSymmetric Autoencoder for Collaborative Filtering.

    Parameters
    ----------
    name: string, optional, default: 'SANSA'
        The name of the recommender model.

    l2: float, optional, default: 1.0
        L2-norm regularization-parameter λ ∈ R+.
    
    weight_matrix_density: float, optional, default: 1e-3
        Density of weight matrices. 

    compute_gramian: boolean, optional, default: True
        Indicates whether training input X is a user-item matrix (represents a bipartite graph) or \
        or an item-item matrix (e.g, co-occurrence matrix; not a bipartite graph).

    factorizer_class: string, optional, default: 'ICF'
        Class of Cholesky factorizer. Supported values:
        - 'CHOLMOD' - exact Cholesky factorization using CHOLMOD algorithm, followed by pruning.
        - 'ICF' - Incomplete Cholesky factorization (i.e., pruning on-the-fly)
        CHOLMOD provides higher-quality approximate factorization for increased price. \
        ICF is less accurate but more scalable (recommended method when num_items >= ~50K-100K).
        Note that ICF uses additional matrix preprocessing and hence different (smaller) l2 regularization.
        
    factorizer_shift_step: float, optional, default: 1e-3
        Used with ICF factorizer.
        Incomplete factorization may break (zero division), indicating need for increased l2 regularization.
        'factorizer_shift_step' is the initial increase in l2 regularization (after first breakdown).

    factorizer_shift_multiplier: float, optional, default: 2.0
        Used with ICF factorizer.
        Multiplier for factorizer shift. After k-th breakdown, additional l2 regularization is \
        'factorizer_shift_step' * 'factorizer_shift_multiplier'^(k-1)

    inverter_scans: integer, optional, default: 3
        Number of scans repairing the approximate inverse factor. Scans repair all columns with residual below \
        a certain threshold, and this threshold goes to 0 in later scans. More scans give more accurate results \
        but take longer. We recommend values between 0 and 5, use lower values if scans take too long.

    inverter_finetune_steps: integer, optional, default: 10
        Repairs a small portion of columns with highest residuals. All finetune steps take (roughly) the same amount of time.
        We recommend values between 0 and 30.

    use_absolute_value_scores: boolean, optional, default: False
        Following https://dl.acm.org/doi/abs/10.1145/3640457.3688179, it is recommended for EASE-like models to consider \
        the absolute value of scores in situations when X^TX is sparse.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already \
        trained. 
        
    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    seed: int, optional, default: None
        Random seed for parameters initialization.

    References
    ----------
    * Martin Spišák, Radek Bartyzal, Antonín Hoskovec, Ladislav Peska, and Miroslav Tůma. 2023. \
    Scalable Approximate NonSymmetric Autoencoder for Collaborative Filtering. \
    In Proceedings of the 17th ACM Conference on Recommender Systems (RecSys '23). \
    Association for Computing Machinery, New York, NY, USA, 763–770. https://doi.org/10.1145/3604915.3608827

    * SANSA GitHub Repository: https://github.com/glami/sansa
    """

    def __init__(
        self,
        name="SANSA",
        l2=1.0,
        weight_matrix_density=1e-3,
        compute_gramian=True,
        factorizer_class="ICF",
        factorizer_shift_step=1e-3,
        factorizer_shift_multiplier=2.0,
        inverter_scans=3,
        inverter_finetune_steps=10,
        use_absolute_value_scores=False,
        trainable=True,
        verbose=True,
        seed=None,
        W1=None,  # "weights[0] (sp.csr_matrix)"
        W2=None,  # "weights[1] (sp.csr_matrix)"
        X=None,  # user-item interaction matrix (sp.csr_matrix)
    ):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.l2 = l2
        self.weight_matrix_density = weight_matrix_density
        self.compute_gramian = compute_gramian
        self.factorizer_class = factorizer_class
        self.factorizer_shift_step = factorizer_shift_step
        self.factorizer_shift_multiplier = factorizer_shift_multiplier
        self.inverter_scans = inverter_scans
        self.inverter_finetune_steps = inverter_finetune_steps
        self.use_absolute_value_scores = use_absolute_value_scores
        self.verbose = verbose
        self.seed = seed
        self.X = X.astype(np.float32) if X is not None and X.dtype != np.float32 else X
        self.weights = (W1, W2)

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

        from sansa.core import (
            FactorizationMethod,
            GramianFactorizer,
            CHOLMODGramianFactorizerConfig,
            ICFGramianFactorizerConfig,
            UnitLowerTriangleInverter,
            UMRUnitLowerTriangleInverterConfig,
        )
        from sansa.utils import get_squared_norms_along_compressed_axis, inplace_scale_along_compressed_axis, inplace_scale_along_uncompressed_axis

        # User-item interaction matrix (sp.csr_matrix)
        self.X = train_set.matrix.astype(np.float32)

        if self.factorizer_class == "CHOLMOD":
            self.factorizer_config = CHOLMODGramianFactorizerConfig()
        else:
            self.factorizer_config = ICFGramianFactorizerConfig(
                factorization_shift_step=self.factorizer_shift_step,  # initial diagonal shift if incomplete factorization fails
                factorization_shift_multiplier=self.factorizer_shift_multiplier,  # multiplier for the shift for subsequent attempts
            )
        self.factorizer = GramianFactorizer.from_config(self.factorizer_config)
        self.factorization_method = self.factorizer_config.factorization_method

        self.inverter_config = UMRUnitLowerTriangleInverterConfig(
            scans=self.inverter_scans,  # number of scans through all columns of the matrix
            finetune_steps=self.inverter_finetune_steps,  # number of finetuning steps, targeting worst columns
        )
        self.inverter = UnitLowerTriangleInverter.from_config(self.inverter_config)

        # create a working copy of user_item_matrix
        X = self.X.copy()

        if self.factorization_method == FactorizationMethod.ICF:
            # scale matrix X
            if self.compute_gramian:
                # Inplace scale columns of X by square roots of column norms of X^TX.
                da = np.sqrt(np.sqrt(get_squared_norms_along_compressed_axis(X.T @ X)))
                # Divide columns of X by the computed square roots of row norms of X^TX
                da[da == 0] = 1  # ignore zero elements
                inplace_scale_along_uncompressed_axis(X, 1 / da)  # CSR column scaling
                del da
            else:
                # Inplace scale rows and columns of X by square roots of row norms of X.
                da = np.sqrt(np.sqrt(get_squared_norms_along_compressed_axis(X)))
                # Divide rows and columns of X by the computed square roots of row norms of X
                da[da == 0] = 1  # ignore zero elements
                inplace_scale_along_uncompressed_axis(X, 1 / da)  # CSR column scaling
                inplace_scale_along_compressed_axis(X, 1 / da)  # CSR row scaling
                del da

        # Compute LDL^T decomposition of
        # - P(X^TX + self.l2 * I)P^T if compute_gramian=True
        # - P(X + self.l2 * I)P^T if compute_gramian=False
        if self.verbose:
            print("Computing LDL^T decomposition of permuted item-item matrix...")
        L, D, p = self.factorizer.approximate_ldlt(
            X,
            self.l2,
            self.weight_matrix_density,
            compute_gramian=self.compute_gramian,
        )
        del X

        # Compute approximate inverse of L using selected method
        if self.verbose:
            print("Computing approximate inverse of L...")
        L_inv = self.inverter.invert(L)
        del L

        # Construct W = L_inv @ P
        inv_p = np.argsort(p)
        W = L_inv[:, inv_p]
        del L_inv

        # Construct W_r (A^{-1} = W.T @ W_r)
        W_r = W.copy()
        inplace_scale_along_uncompressed_axis(W_r, 1 / D.diagonal())

        # Extract diagonal entries
        diag = W.copy()
        diag.data = diag.data**2
        inplace_scale_along_uncompressed_axis(diag, 1 / D.diagonal())
        diagsum = diag.sum(axis=0)  # original
        del diag
        diag = np.asarray(diagsum)[0]

        # Divide columns of the inverse by negative diagonal entries
        # equivalent to dividing the columns of W by negative diagonal entries
        inplace_scale_along_compressed_axis(W_r, -1 / diag)
        self.weights = (W.T.tocsr(), W_r.tocsr())

        return self

    def forward(self, X: sp.csr_matrix) -> sp.csr_matrix:
        """
        Forward pass.
        """
        latent = X @ self.weights[0]
        out = latent @ self.weights[1]
        return out

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

        scores = self.forward(self.X[user_idx]).toarray().reshape(-1)
        if self.use_absolute_value_scores:
            scores = np.abs(scores)
        if item_idx is None:
            return scores
        return scores[item_idx]

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
        return self.X @ self.weights[0]

    def get_item_vectors(self):
        """Getting a matrix of item vectors used for building the index for ANN search.

        Returns
        -------
        out: numpy.array
            Matrix of item vectors for all items available in the model.
        """
        return self.self.weights[1]
