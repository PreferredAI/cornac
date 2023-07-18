import cornac
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit

# Load user-item feedback
data_100k = movielens.load_feedback(variant="100K")
data_1m = movielens.load_feedback(variant="1M")

# Instantiate an evaluation method to split data into train and test sets.
ratio_split = RatioSplit(
    data=data_100k,
    test_size=0.1,
    val_size=0.1,
    exclude_unknowns=True,
    verbose=True,
    seed=123,
)

hpf = cornac.models.HPF(
    k=5,
    seed=123
)

pf = cornac.models.HPF(
    k=5,
    seed=123,
    hierarchical=False,
    name="PF"
)

bpr = cornac.models.BPR(
   k=5,
   max_iter=200,
   learning_rate=0.001,
   lambda_reg=0.01,
   seed=123)

gcmc = cornac.models.GCMC(
    seed=123,
)


# Instantiate evaluation measures
rec_20 = cornac.metrics.Recall(k=20)
ndcg_20 = cornac.metrics.NDCG(k=20)
auc = cornac.metrics.AUC()



# Put everything together into an experiment and run it
cornac.Experiment(
    eval_method=ratio_split,
    models=[pf, gcmc],
    metrics=[cornac.metrics.RMSE()],
    user_based=False,
).run()