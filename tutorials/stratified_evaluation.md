# Stratified Evaluation Method

![results](closed_loop_flow.png)
Recommendation systems are often evaluated based on user’s interactions that were collected from an existing deployed system (see the above figure [Jadidinejad et al.](https://doi.org/10.1145/3397271.3401230)). Users only provide feedback (r) on those items they were exposed to by the deployed system (e). Hence, the collected feedback dataset used to evaluate a new model is influenced by the deployed system (RecSys), as a form of closed loop feedback. In this situation, [Jadidinejad et al.](https://arxiv.org/abs/2104.08912) revealed that the typical offline evaluation of recommenders suffers from the so-called Simpson’s paradox, which is a phenomenon observed when a significant trend appears in several different sub-populations of observational data but that disappears or reverses when these sub-populations are combined together. In addition, they proposed a novel evaluation methodology (Stratified Evaluation) that takes into account the confounder, i.e. the deployed system’s characteristics.

See [the paper](https://arxiv.org/abs/2104.08912) for more details.

Using the proposed stratified evaluation method is as simple as using the classic evaluation in Cornac:

```python
import cornac
from cornac.models import MF, PMF, BPR
from cornac.metrics import MAE, RMSE, Precision, Recall, NDCG, AUC, MAP

from cornac.eval_methods import StratifiedEvaluation
from cornac.experiment import STExperiment


ml_100k = cornac.datasets.movielens.load_feedback()

stra_eval_method = StratifiedEvaluation(data=ml_100k,
                                        n_strata=2,
                                        rating_threshold=4.0,
                                        verbose=True)

models = [
    MF(k=10, max_iter=25, learning_rate=0.01,
       lambda_reg=0.02, use_bias=True, seed=123),
    PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123),
    BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123),
]

metrics = [MAE(), RMSE(), Precision(k=10),
           Recall(k=10), NDCG(k=10), AUC(), MAP()]

exp_stra = STExperiment(eval_method=stra_eval_method,
                        models=models, metrics=metrics, verbose=True)

exp_stra.run()

```

Compared to [the classic evaluation](https://github.com/PreferredAI/cornac#getting-started-your-first-cornac-experiment), you can simply use `StratifiedEvaluation` and `STExperiment` instead of `RatioSplit` and `Experiment` respectively.

## How to cite?
Use the corresponding bibtex entry to cite the paper:

```
@InProceedings{simpson_recsys20,
  author    = {Amir H. Jadidinejad and Craig Macdonald and Iadh Ounis},
  title     = {The Simpson's Paradox in the Offline Evaluation of Recommendation Systems},
  journal   = {ACM Transactions on Information Systems},
  year      = {2021},
}
```