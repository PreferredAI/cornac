"""
Example SANSA (Scalable Approximate NonSymmetric Autoencoder for Collaborative Filtering) on Tradesy data
Original data: http://jmcauley.ucsd.edu/data/tradesy/
"""

import cornac
from cornac.datasets import tradesy
from cornac.eval_methods import RatioSplit

feedback = tradesy.load_feedback()

# Define an evaluation method to split feedback into train and test sets
ratio_split = RatioSplit(
    data=feedback,
    test_size=0.1,
    rating_threshold=0.5,
    exclude_unknowns=True,
    verbose=True,
)

sansa_icf = cornac.models.SANSA(
    name="SANSA (ICF)",
    l2=20.0,
    weight_matrix_density=1e-3,
    compute_gramian=True,
    factorizer_class="ICF",
    factorizer_shift_step=1e-3,
    factorizer_shift_multiplier=2.0,
    inverter_scans=0,
    inverter_finetune_steps=5,
    use_absolute_value_scores=True,  # see https://dl.acm.org/doi/abs/10.1145/3640457.3688179 why this helps on sparse data
)

# Instantiate evaluation measures
auc = cornac.metrics.AUC()
rec_50 = cornac.metrics.Recall(k=50)

# Put everything together into an experiment and run it
cornac.Experiment(eval_method=ratio_split, models=[sansa_icf], metrics=[auc, rec_50]).run()
