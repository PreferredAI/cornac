# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# cython: language_level=3

import multiprocessing

cimport cython
from cython.operator cimport dereference
from cython.parallel import parallel, prange
from libc.stdlib cimport abort, malloc, free
from libcpp cimport bool
from libcpp.string cimport string

import numpy as np
cimport numpy as np
from tqdm.auto import trange

from ..recommender import Recommender
from ...exception import ScoreException
from ...utils import get_rng
from ...utils import fast_dot
from ...utils.common import scale
from ...utils.init_utils import zeros, normal

from .recom_fm cimport Data
from .recom_fm cimport DataMetaInfo
from .recom_fm cimport DVectorDouble
from .recom_fm cimport sparse_entry
from .recom_fm cimport sparse_row
from .recom_fm cimport LargeSparseMatrixMemory

from .recom_fm cimport fm_model
from .recom_fm cimport fm_learn
from .recom_fm cimport fm_learn_sgd
from .recom_fm cimport fm_learn_sgd_element
from .recom_fm cimport fm_learn_sgd_element_adapt_reg
from .recom_fm cimport fm_learn_mcmc
from .recom_fm cimport fm_learn_mcmc_simultaneous



cdef Data* _prepare_data(long[:] uid, long[:] iid, float[:] val, long num_feature, bool has_t, bool verbose):
        cdef num_values = 2 * len(val)
        cdef num_rows = len(val)

        cdef Data *data = new Data(0, False, has_t)
        data.verbose = verbose

        cdef LargeSparseMatrixMemory[float] *X = new LargeSparseMatrixMemory[float]()
        cdef DVector[float] *target = new DVector[float]()

        X.data.setSize(num_rows)
        target.setSize(num_rows)

        cdef sparse_entry[float] *cache = <sparse_entry[float] *> malloc(num_values * sizeof(sparse_entry[float]))
        if cache is NULL:
            abort()

        cdef unsigned int row_id
        cdef unsigned int cache_id = 0

        for row_id in range(num_rows):
            target.set(row_id, val[row_id])
            X.data.value[row_id].data = &cache[cache_id]
            X.data.value[row_id].size = 0

            # user feature
            cache[cache_id].id = uid[row_id]
            cache[cache_id].value = 1
            cache_id += 1
            X.data.value[row_id].size += 1

            # item feature
            cache[cache_id].id = iid[row_id]
            cache[cache_id].value = 1
            cache_id += 1
            X.data.value[row_id].size += 1 

        assert num_values == cache_id

        X.num_values = num_values
        X.num_cols = num_feature

        data.load(dereference(X), dereference(target))

        return data



class FM(Recommender):
    """Factorization Machines.

    Parameters
    ----------
    k0: int, optional, default: 1
        Using bias 'w0'.
    
    k1: int, optional, default: 1
        Using first-order weights 'w'.

    k2: int, optional, default: 8
        Dimension of second-order weights 'v'.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD, SGDA.

    learning_rate: float, optional, default: 0.01
        The learning rate for SGD.

    reg0: float, optional, default: 0.0
        Bias regularization. 

    reg1: float, optional, default: 0.0
        First-order weights regularization. 

    reg2: float, optional, default: 0.0
        Second-order weights regularization. 

    reg_all: float, optional, default: 0.0
        Regularization for all parameters.
        If 'reg_all' != 0.0, this will be used for all regularization.

    init_stdev: float, optional, default: 0.1
        Standard deviation for initialization of second-order weights 'v'.

    method, str, optional, default: 'mcmc'
        Learning method (SGD, SGDA, ALS, MCMC)

    num_threads: int, optional, default: 0
        Number of parallel threads for training. If num_threads=0, all CPU cores will be utilized.
        If seed is not None, num_threads=1 to remove randomness from parallelization.

    trainable: boolean, optional, default: True
        When False, the model will not be re-trained, and input of pre-trained parameters are required.

    verbose: boolean, optional, default: True
        When True, some running logs are displayed.

    init_params: dictionary, optional, default: None
        Initial parameters, e.g., init_params = {'w0': bias, 'w': first-order, 'v': second-order}

    seed: int, optional, default: None
        Random seed for weight initialization.
        If specified, training will take longer because of single-thread (no parallelization).

    References
    ----------
    * Rendle, S. (2010, December). Factorization machines. \
    In 2010 IEEE International Conference on Data Mining (pp. 995-1000). IEEE.
    
    * Rendle, S. (2012). Factorization machines with libfm. \
    ACM Transactions on Intelligent Systems and Technology (TIST), 3(3), 1-22.
    """

    def __init__(
        self, 
        name="FM",
        k0=1,
        k1=1,
        k2=8,
        max_iter=100,
        learning_rate=0.01,
        reg0=0.0,
        reg1=0.0,
        reg2=0.0,
        reg_all=0.0,
        init_stdev=0.1,
        method="mcmc",
        num_threads=0, 
        trainable=True, 
        verbose=True, 
        init_params=None, 
        seed=None
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.k0 = k0
        self.k1 = k1
        self.k2 = k2
        self.max_iter = max_iter
        self.learning_rate = learning_rate

        self.reg0 = reg_all if reg_all != 0.0 else reg0
        self.reg1 = reg_all if reg_all != 0.0 else reg1
        self.reg2 = reg_all if reg_all != 0.0 else reg2
        
        self.init_stdev = init_stdev
        self.method = method.lower()
        self.seed = seed
        self.rng = get_rng(seed)

        if seed is not None:
            self.num_threads = 1
        elif num_threads > 0 and num_threads < multiprocessing.cpu_count():
            self.num_threads = num_threads
        else:
            self.num_threads = multiprocessing.cpu_count()

        # Init params if provided
        self.init_params = {} if init_params is None else init_params
        self.w0 = self.init_params.get('w0', None)
        self.w = self.init_params.get('w', None)
        self.v = self.init_params.get('v', None)
    
    def _init(self):
        num_features = self.train_set.total_users + self.train_set.total_items

        if self.w0 is None:
            self.w0 = 0.0
        if self.w is None:
            self.w = zeros(num_features, dtype=np.double)
        if self.v is None:
            d = self.k2 if self.k2 else 1  # dummy if self.k2 == 0
            self.v = normal((d, num_features), std=self.init_stdev, random_state=self.seed, dtype=np.double)


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

        self._init()

        if not self.trainable:
            return self

        self._fit_libfm(train_set, val_set, self.w, self.v)

        if self.verbose:
            print('Optimization finished!')

        return self


    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit_libfm(self, train_set, val_set, double[:] w, double[:, :] v):
        cdef unsigned int num_feature = self.train_set.total_users + self.train_set.total_items

        (uid, iid, val) = self.train_set.uir_tuple
        cdef Data *train = _prepare_data(
            uid, 
            iid + self.train_set.total_users, 
            val.astype(np.float32), 
            num_feature, 
            self.method in ["als", "mcmc"],
            self.verbose,
        )
        if self.verbose:
            print("Training data:")
            train.debug()

        cdef Data *validation = NULL
        if val_set is not None:
            (uid, iid, val) = val_set.uir_tuple
            validation = _prepare_data(
                uid, 
                iid + self.train_set.total_users,
                val.astype(np.float32), 
                num_feature, 
                self.method in ["als", "mcmc"],
                self.verbose,
            )
            if self.verbose:
                print("Validation data:")
                validation.debug()
            
        # (1.3) Load meta data
        # (main table)
        cdef DataMetaInfo *meta = new DataMetaInfo(num_feature)
        
        # meta.num_attr_groups = meta_main.num_attr_groups
        meta.num_attr_per_group.setSize(meta.num_attr_groups)
        meta.num_attr_per_group.init(0)
        cdef unsigned int i, j
        for i in range(meta.attr_group.dim):
            j = meta.attr_group.get(i)
            meta.num_attr_per_group.set(j, meta.num_attr_per_group.get(j) + 1)
        
        if self.verbose:
            meta.debug()

        # (2) Setup the factorization machine
        cdef fm_model fm

        fm.num_attribute = num_feature
        fm.init_stdev = self.init_stdev
        # set the number of dimensions in the factorization
        fm.k0 = self.k0 != 0
        fm.k1 = self.k1 != 0
        fm.num_factor = self.k2
        fm.init()
        # reset the weights of v
        if self.k2:    
            for i in range(self.k2):
                for j in range(num_feature):
                    fm.v.set(i, j, v[i, j]) 

        # (3) Setup the learning method:
        cdef fm_learn *fml

        if self.method == "sgd":
            fml = new fm_learn_sgd_element()
            (<fm_learn_sgd*>fml).num_iter = self.max_iter
        elif self.method == "sgda":
            if val_set is None:
                raise ValueError("'sgda' method requires validation set but None, other methods: ['sgd', 'als', 'mcmc']" )
            
            fml = new fm_learn_sgd_element_adapt_reg()
            (<fm_learn_sgd*>fml).num_iter = self.max_iter
            (<fm_learn_sgd_element_adapt_reg*>fml).validation = validation
        else: # als or mcmc
            fm.w.init_normal(fm.init_mean, fm.init_stdev)
            fml = new fm_learn_mcmc_simultaneous()
            (<fm_learn_mcmc*>fml).num_iter = self.max_iter
            (<fm_learn_mcmc*>fml).do_sample = self.method == "mcmc"
            (<fm_learn_mcmc*>fml).do_multilevel = self.method == "mcmc"
            if validation != NULL:
                fml.validation = validation
                (<fm_learn_mcmc*>fml).num_eval_cases = validation.num_cases
        
        fml.fm = &fm
        fml.max_target = self.train_set.max_rating
        fml.min_target = self.train_set.min_rating
        fml.meta = meta
        fml.task = 0  # regression

        fml.init()

        # regularization
        fm.reg0 = self.reg0
        fm.regw = self.reg1
        fm.regv = self.reg2
        if self.method in ["als", "mcmc"]:
            (<fm_learn_mcmc*>fml).w_lambda.init(fm.regw)
            (<fm_learn_mcmc*>fml).v_lambda.init(fm.regv)

        # learning rate
        (<fm_learn_sgd*>fml).learn_rate = self.learning_rate
        (<fm_learn_sgd*>fml).learn_rates.init(self.learning_rate)

        if self.verbose:
            fm.debug()
            fml.debug()

        # () learn
        fml.learn(dereference(train), dereference(validation))

        # store learned weights for prediction
        if self.k0:
            self.w0 = fm.w0
        if self.k1:
            for i in range(num_feature):
                w[i] = fm.w.get(i)
        if self.k2:
            for i in range(self.k2):
                for j in range(num_feature):
                    v[i, j] = <double>fm.v.get(i, j)


    def _fm_predict(self, user_idx, item_idx):
        uid = user_idx
        iid = item_idx + self.train_set.total_users
        score = 0.0
        if self.k0:
            score += self.w0
        if self.k1:
            score += self.w[uid] + self.w[iid]
        if self.k2:
            sum_ = self.v[:, uid] + self.v[:, iid]
            sum_sqr_ = self.v[:, uid] ** 2 + self.v[:, iid] ** 2
            score += 0.5 * (sum_ ** 2 - sum_sqr_).sum()
        return score


    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for that to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items

        """
        if item_idx is None:
            known_item_scores = np.fromiter(
                (self._fm_predict(user_idx, i) for i in range(self.train_set.total_items)), 
                dtype=np.double, 
                count=self.train_set.total_items
            )
            return known_item_scores
        else:
            return self._fm_predict(user_idx, item_idx)