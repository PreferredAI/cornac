# Propensity-based Stratified Evaluation Method

![results](closed_loop_diagram.jpg)
Recommendation systems are often evaluated based on user’s interactions that were collected from an existing deployed system (see the above figure/[source](https://doi.org/10.1145/3397271.3401230)). Users only provide feedback (r) on those items they were exposed to by the deployed system (e). Hence, the collected feedback dataset used to evaluate a new model is influenced by the deployed system (RecSys), as a form of closed loop feedback. In this situation, [Jadidinejad et al.](https://arxiv.org/abs/2104.08912) revealed that the typical offline evaluation of recommenders suffers from the so-called Simpson’s paradox, which is a phenomenon observed when a significant trend appears in several different sub-populations of observational data but that disappears or reverses when these sub-populations are combined together. In addition, they proposed a novel evaluation methodology (Propensity-based Stratified Evaluation) that takes into account the confounder, i.e. the deployed system’s characteristics.

See [the original paper](https://arxiv.org/abs/2104.08912) for more details.

Using the proposed propensity-based stratified evaluation method is as simple as using the classic evaluation in Cornac:

```python
import cornac
from cornac.models import WMF, BPR
from cornac.metrics import MAE, RMSE, Precision, Recall, NDCG, AUC, MAP

from cornac.eval_methods import PropensityStratifiedEvaluation
from cornac.experiment import Experiment


ml_dataset = cornac.datasets.movielens.load_feedback(variant="1M")

stra_eval_method = PropensityStratifiedEvaluation(data=ml_dataset,
                                                  n_strata=2,
                                                  rating_threshold=4.0,
                                                  verbose=True)


models = [
    WMF(k=10, seed=123),
    BPR(k=10, seed=123),
]

metrics = [MAE(), RMSE(), Precision(k=10),
           Recall(k=10), NDCG(), AUC(), MAP()]

exp_stra = Experiment(eval_method=stra_eval_method,
                      models=models, metrics=metrics)

exp_stra.run()
```

Compared to [the classic evaluation](https://github.com/PreferredAI/cornac#getting-started-your-first-cornac-experiment), you can simply use `PropensityStratifiedEvaluation` instead of `RatioSplit`. The output is based on the defined strata (Q1, Q2,...) and the "Unbiased" row represents the performance prediction based on Propensity Stratified Evaluation method:

```
[WMF]
         |    MAE |   RMSE |    AUC |    MAP | NDCG@-1 | Precision@10 | Recall@10 |        SIZE
-------- + ------ + ------ + ------ + ------ + ------- + ------------ + --------- + -----------
Closed   | 1.0864 | 1.2743 | 0.9009 | 0.0658 |  0.3791 |       0.0799 |    0.0625 | 200008.0000
-------- + ------ + ------ + ------ + ------ + ------- + ------------ + --------- + -----------
IPS      | 1.0864 | 1.2743 | 0.9009 | 0.0658 |  0.2003 |       0.0049 |    0.0459 | 200008.0000
-------- + ------ + ------ + ------ + ------ + ------- + ------------ + --------- + -----------
Q1       | 1.0911 | 1.2785 | 0.8987 | 0.0572 |  0.3637 |       0.0612 |    0.0479 | 197556.0000
Q2       | 0.7544 | 0.7658 | 0.9935 | 0.1985 |  0.3680 |       0.0653 |    0.5584 |   2452.0000
-------- + ------ + ------ + ------ + ------ + ------- + ------------ + --------- + -----------
Unbiased | 1.0869 | 1.2722 | 0.8999 | 0.0589 |  0.3638 |       0.0612 |    0.0542 | 200008.0000

[BPR]
         |    MAE |   RMSE |    AUC |    MAP | NDCG@-1 | Precision@10 | Recall@10 |        SIZE
-------- + ------ + ------ + ------ + ------ + ------- + ------------ + --------- + -----------
Closed   | 2.0692 | 2.2798 | 0.8758 | 0.0623 |  0.3720 |       0.0723 |    0.0547 | 200008.0000
-------- + ------ + ------ + ------ + ------ + ------- + ------------ + --------- + -----------
IPS      | 2.0692 | 2.2798 | 0.8758 | 0.0623 |  0.2026 |       0.0077 |    0.0547 | 200008.0000
-------- + ------ + ------ + ------ + ------ + ------- + ------------ + --------- + -----------
Q1       | 2.0840 | 2.2929 | 0.8730 | 0.0428 |  0.3390 |       0.0387 |    0.0292 | 197556.0000
Q2       | 1.3782 | 1.3908 | 0.9998 | 0.5807 |  0.6892 |       0.1170 |    1.0000 |   2452.0000
-------- + ------ + ------ + ------ + ------ + ------- + ------------ + --------- + -----------
Unbiased | 2.0754 | 2.2818 | 0.8746 | 0.0494 |  0.3432 |       0.0396 |    0.0411 | 200008.0000
```

`SIZE` column represents the number of feedback in each stratum (`Q1` or `Q2`). `Unbiased` row represents the estimated propensity-based evaluation per each metric while `Closed` row represents the classical evaluation. It reproduces Table 1-b (or Table 2) in [the original paper](https://arxiv.org/abs/2104.08912). Due to the random splitting, the above numbers are slightly different with the paper but the insight is the same!

## How to cite?
Use the corresponding bibtex entry to cite the paper:

```
@InProceedings{simpson_recsys20,
  author    = {Amir H. Jadidinejad and Craig Macdonald and Iadh Ounis},
  title     = {The Simpson's Paradox in the Offline Evaluation of Recommendation Systems},
  journal   = {ACM Transactions on Information Systems (to appear)},
  year      = {2021},
}
```