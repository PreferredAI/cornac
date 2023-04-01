import time
from collections import defaultdict
from collections import OrderedDict

import powerlaw
import numpy as np
import tqdm.auto as tqdm

from ..utils.common import safe_indexing
from ..data import Dataset
from .base_method import BaseMethod, rating_eval
from .ratio_split import RatioSplit
from ..experiment.result import Result, PSTResult


def ranking_eval(
    model,
    metrics,
    train_set,
    test_set,
    val_set=None,
    rating_threshold=1.0,
    exclude_unknowns=True,
    verbose=False,
    props=None,
):
    """Evaluate model on provided ranking metrics.
    
    Parameters
    ----------
    model: :obj:`cornac.models.Recommender`, required
        Recommender model to be evaluated.
        
    metrics: :obj:`iterable`, required
        List of rating metrics :obj:`cornac.metrics.RankingMetric`.
    
    train_set: :obj:`cornac.data.Dataset`, required
        Dataset to be used for model training. This will be used to exclude
        observations already appeared during training.
    
    test_set: :obj:`cornac.data.Dataset`, required
        Dataset to be used for evaluation.
    
    val_set: :obj:`cornac.data.Dataset`, optional, default: None
        Dataset to be used for model selection. This will be used to exclude
        observations already appeared during validation.
        
    rating_threshold: float, optional, default: 1.0
        The threshold to convert ratings into positive or negative feedback.
        
    exclude_unknowns: bool, optional, default: True
        Ignore unknown users and items during evaluation.
        
    verbose: bool, optional, default: False
        Output evaluation progress.
        
    props: dictionary, optional, default: None
        items propensity scores
        
    Returns
    -------
    res: (List, List)
        Tuple of two lists:
         - average result for each of the metrics
         - average result per user for each of the metrics
    """

    if len(metrics) == 0:
        return [], []

    avg_results = []
    user_results = [{} for _ in enumerate(metrics)]

    gt_mat = test_set.csr_matrix
    train_mat = train_set.csr_matrix
    val_mat = None if val_set is None else val_set.csr_matrix

    def pos_items(csr_row):
        return [
            item_idx
            for (item_idx, rating) in zip(csr_row.indices, csr_row.data)
            if rating >= rating_threshold
        ]

    for user_idx in tqdm.tqdm(test_set.user_indices, disable=not verbose, miniters=100):
        test_pos_items = pos_items(gt_mat.getrow(user_idx))
        if len(test_pos_items) == 0:
            continue

        u_gt_pos = np.zeros(test_set.num_items, dtype='float')
        u_gt_pos[test_pos_items] = 1

        val_pos_items = [] if val_mat is None else pos_items(val_mat.getrow(user_idx))
        train_pos_items = (
            []
            if train_set.is_unk_user(user_idx)
            else pos_items(train_mat.getrow(user_idx))
        )

        u_gt_neg = np.ones(test_set.num_items, dtype='int')
        u_gt_neg[test_pos_items + val_pos_items + train_pos_items] = 0

        item_indices = None if exclude_unknowns else np.arange(test_set.num_items)
        item_rank, item_scores = model.rank(user_idx, item_indices)

        total_pi = 0.0
        if props is not None:
            for idx, e in enumerate(u_gt_pos):
                if e > 0 and props[str(idx)] > 0:
                    u_gt_pos[idx] /= props[str(idx)]
                    total_pi += 1 / props[str(idx)]

        for i, mt in enumerate(metrics):
            mt_score = mt.compute(
                gt_pos=u_gt_pos,
                gt_neg=u_gt_neg,
                pd_rank=item_rank,
                pd_scores=item_scores,
            )

            user_results[i][user_idx] = mt_score

    # avg results of ranking metrics
    for i, mt in enumerate(metrics):
        avg_results.append(sum(user_results[i].values()) / len(user_results[i]))

    return avg_results, user_results


class PropensityStratifiedEvaluation(BaseMethod):
    """Propensity-based Stratified Evaluation Method proposed by Jadidinejad et al. (2021)

    Parameters
    ----------
    data: array-like, required
        Raw preference data in the triplet format [(user_id, item_id, rating_value)].

    test_size: float, optional, default: 0.2
        The proportion of the test set, 
        if > 1 then it is treated as the size of the test set.

    val_size: float, optional, default: 0.0
        The proportion of the validation set, \
        if > 1 then it is treated as the size of the validation set.

    n_strata: int, optional, default: 2
        The number of strata for propensity-based stratification.

    rating_threshold: float, optional, default: 1.0
        Threshold used to binarize rating values into positive or negative feedback for
        model evaluation using ranking metrics (rating metrics are not affected).

    seed: int, optional, default: None
        Random seed for reproducibility.

    exclude_unknowns: bool, optional, default: True
        If `True`, unknown users and items will be ignored during model evaluation.

    verbose: bool, optional, default: False
        Output running log.

    References
    ----------
    Amir H. Jadidinejad, Craig Macdonald and Iadh Ounis, 
    The Simpson's Paradox in the Offline Evaluation of Recommendation Systems, 
    ACM Transactions on Information Systems (to appear)
    https://arxiv.org/abs/2104.08912
    """

    def __init__(
        self,
        data,
        test_size=0.2,
        val_size=0.0,
        n_strata=2,
        rating_threshold=1.0,
        seed=None,
        exclude_unknowns=True,
        verbose=False,
        **kwargs,
    ):
        BaseMethod.__init__(
            self,
            data=data,
            rating_threshold=rating_threshold,
            seed=seed,
            exclude_unknowns=exclude_unknowns,
            verbose=verbose,
            **kwargs,
        )

        self.n_strata = n_strata

        # estimate propensities
        self.props = self._estimate_propensities()

        # split the data into train/valid/test sets
        self.train_size, self.val_size, self.test_size = RatioSplit.validate_size(
            val_size, test_size, len(self._data)
        )
        self._split()

    def _eval(self, model, test_set, val_set, user_based, props=None):
        metric_avg_results = OrderedDict()
        metric_user_results = OrderedDict()

        avg_results, user_results = rating_eval(
            model=model,
            metrics=self.rating_metrics,
            test_set=test_set,
            user_based=user_based,
        )
        for i, mt in enumerate(self.rating_metrics):
            metric_avg_results[mt.name] = avg_results[i]
            metric_user_results[mt.name] = user_results[i]

        avg_results, user_results = ranking_eval(
            model=model,
            metrics=self.ranking_metrics,
            train_set=self.train_set,
            test_set=test_set,
            val_set=val_set,
            rating_threshold=self.rating_threshold,
            exclude_unknowns=self.exclude_unknowns,
            verbose=self.verbose,
            props=props,
        )
        for i, mt in enumerate(self.ranking_metrics):
            metric_avg_results[mt.name] = avg_results[i]
            metric_user_results[mt.name] = user_results[i]

        return Result(model.name, metric_avg_results, metric_user_results)

    def _split(self):
        data_idx = self.rng.permutation(len(self._data))
        train_idx = data_idx[: self.train_size]
        test_idx = data_idx[-self.test_size :]
        val_idx = data_idx[self.train_size : -self.test_size]

        train_data = safe_indexing(self._data, train_idx)
        test_data = safe_indexing(self._data, test_idx)
        val_data = safe_indexing(self._data, val_idx) if len(val_idx) > 0 else None

        # build train/test/valid datasets
        self._build_datasets(
            train_data=train_data, test_data=test_data, val_data=val_data
        )

        # build stratified dataset
        self._build_stratified_dataset(test_data=test_data)

    def _estimate_propensities(self):
        # find the item's frequencies
        item_freq = defaultdict(int)
        for u, i, r in self._data:
            item_freq[i] += 1

        # fit the exponential param
        data = np.array([e for e in item_freq.values()], dtype='float')
        results = powerlaw.Fit(data, discrete=True, fit_method="Likelihood")
        alpha = results.power_law.alpha
        fmin = results.power_law.xmin

        if self.verbose:
            print("Powerlaw exponential estimates: %f, min=%d" % (alpha, fmin))

        # replace raw frequencies with the estimated propensities
        for k, v in item_freq.items():
            if v > fmin:
                item_freq[k] = pow(v, alpha)

        return item_freq  # user-independent propensity estimations

    def _build_stratified_dataset(self, test_data):
        # build stratified datasets
        self.stratified_sets = {}

        # match the corresponding propensity score for each feedback
        test_props = np.array(
            [self.props[i] for u, i, r in test_data], dtype='float'
        )

        # stratify
        minp = min(test_props) - 0.01 * min(test_props)
        maxp = max(test_props) + 0.01 * max(test_props)
        slice = (maxp - minp) / self.n_strata
        strata = [
            f"Q{idx}"
            for idx in np.digitize(x=test_props, bins=np.arange(minp, maxp, slice))
        ]

        for stratum in sorted(np.unique(strata)):
            # sample the corresponding sub-population
            qtest_data = []
            for (u, i, r), q in zip(test_data, strata):
                if q == stratum:
                    qtest_data.append((u, i, r))

            # build a dataset
            qtest_set = Dataset.build(
                data=qtest_data,
                fmt=self.fmt,
                global_uid_map=self.global_uid_map,
                global_iid_map=self.global_iid_map,
                seed=self.seed,
                exclude_unknowns=self.exclude_unknowns,
            )
            if self.verbose:
                print("---")
                print("Test data ({}):".format(stratum))
                print("Number of users = {}".format(len(qtest_set.uid_map)))
                print("Number of items = {}".format(len(qtest_set.iid_map)))
                print("Number of ratings = {}".format(qtest_set.num_ratings))
                print("Max rating = {:.1f}".format(qtest_set.max_rating))
                print("Min rating = {:.1f}".format(qtest_set.min_rating))
                print("Global mean = {:.1f}".format(qtest_set.global_mean))
                print(
                    "Number of unknown users = {}".format(
                        qtest_set.num_users - self.train_set.num_users
                    )
                )
                print(
                    "Number of unknown items = {}".format(
                        self.test_set.num_items - self.train_set.num_items
                    )
                )

            self.stratified_sets[stratum] = qtest_set

    def evaluate(self, model, metrics, user_based, show_validation=True):
        """Evaluate given models according to given metrics

        Parameters
        ----------
        model: :obj:`cornac.models.Recommender`
            Recommender model to be evaluated.

        metrics: :obj:`iterable`
            List of metrics.

        user_based: bool, required 
            Evaluation strategy for the rating metrics. Whether results 
            are averaging based on number of users or number of ratings.

        show_validation: bool, optional, default: True 
            Whether to show the results on validation set (if exists).

        Returns
        -------
        res: :obj:`cornac.experiment.Result`
        """
        result = PSTResult(model.name)

        if self.train_set is None:
            raise ValueError("train_set is required but None!")
        if self.test_set is None:
            raise ValueError("test_set is required but None!")

        self._reset()
        self._organize_metrics(metrics)

        ###########
        # FITTING #
        ###########
        if self.verbose:
            print("\n[{}] Training started!".format(model.name))

        start = time.time()
        model.fit(self.train_set, self.val_set)
        train_time = time.time() - start

        ##############
        # EVALUATION #
        ##############
        if self.verbose:
            print("\n[{}] Evaluation started!".format(model.name))

        # evaluate on the sampled test set (closed-loop)
        test_result = self._eval(
            model=model,
            test_set=self.test_set,
            val_set=self.val_set,
            user_based=user_based,
        )
        test_result.metric_avg_results["SIZE"] = self.test_set.num_ratings
        result.append(test_result)

        if self.verbose:
            print("\n[{}] IPS Evaluation started!".format(model.name))

        # evaluate based on Inverse Propensity Scoring
        ips_result = self._eval(
            model=model,
            test_set=self.test_set,
            val_set=self.val_set,
            user_based=user_based,
            props=self.props,
        )
        ips_result.metric_avg_results["SIZE"] = self.test_set.num_ratings
        result.append(ips_result)

        if self.verbose:
            print("\n[{}] Stratified Evaluation started!".format(model.name))

        # evaluate on different strata
        start = time.time()

        for _, qtest_set in self.stratified_sets.items():
            qtest_result = self._eval(
                model=model,
                test_set=qtest_set,
                val_set=self.val_set,
                user_based=user_based,
            )

            test_time = time.time() - start
            qtest_result.metric_avg_results["SIZE"] = qtest_set.num_ratings

            result.append(qtest_result)

        result.organize()

        val_result = None
        if show_validation and self.val_set is not None:
            start = time.time()
            val_result = self._eval(
                model=model, test_set=self.val_set, val_set=None, user_based=user_based
            )
            val_time = time.time() - start

        return result, val_result
