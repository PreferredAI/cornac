"""Example SANSA (Scalable Approximate NonSymmetric Autoencoder for Collaborative Filtering) on MovieLens data"""

import cornac
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit


# Load user-item feedback
data = movielens.load_feedback(variant="1M")

# Instantiate an evaluation method to split data into train and test sets.
ratio_split = RatioSplit(
    data=data,
    test_size=0.2,
    exclude_unknowns=True,
    verbose=True,
    seed=123,
)

sansa_cholmod = cornac.models.SANSA(
    name="SANSA (CHOLMOD)",
    l2=500.0,
    weight_matrix_density=1e-2,
    compute_gramian=True,
    factorizer_class="CHOLMOD",
    factorizer_shift_step=1e-3,
    factorizer_shift_multiplier=2.0,
    inverter_scans=5,
    inverter_finetune_steps=20,
    use_absolute_value_scores=False,
)

sansa_icf = cornac.models.SANSA(
    name="SANSA (ICF)",
    l2=10.0,
    weight_matrix_density=1e-2,
    compute_gramian=True,
    factorizer_class="ICF",
    factorizer_shift_step=1e-3,
    factorizer_shift_multiplier=2.0,
    inverter_scans=5,
    inverter_finetune_steps=20,
    use_absolute_value_scores=False,
)


# Instantiate evaluation measures
rec_20 = cornac.metrics.Recall(k=20)
rec_50 = cornac.metrics.Recall(k=50)
ndcg_100 = cornac.metrics.NDCG(k=100)


# Put everything together into an experiment and run it
cornac.Experiment(
    eval_method=ratio_split,
    models=[sansa_cholmod, sansa_icf],
    metrics=[rec_20, rec_50, ndcg_100],
    user_based=True,  # If `False`, results will be averaged over the number of ratings.
    save_dir=None,
).run()
