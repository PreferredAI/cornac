"""
Fit to and evaluate PCRL [1] on the Office Amazon dataset.
[1] Salah, Aghiles, and Hady W. Lauw. Probabilistic Collaborative Representation Learning\
    for Personalized Item Recommendation. In UAI 2018.

@author: Aghiles Salah <asalah@smu.edu.sg>
"""

from cornac.data import GraphModule
from cornac.eval_methods import RatioSplit
from cornac.experiment import Experiment
from cornac import metrics
from cornac.models import PCRL
from cornac.datasets import amazon_office as office

# Load office ratings and item contexts
ratings = office.load_rating()
contexts = office.load_context()

item_graph_module = GraphModule(data=contexts)

ratio_split = RatioSplit(data=ratings,
                         test_size=0.2, rating_threshold=3.5,
                         shuffle=True, exclude_unknowns=True,
                         verbose=True, item_graph=item_graph_module)

pcrl = PCRL(k=100, z_dims=[300],
            max_iter=300, 
            learning_rate=0.001)


# Evaluation metrics
nDgc = metrics.NDCG(k=-1)
rec = metrics.Recall(k=20)
pre = metrics.Precision(k=20)

# Instantiate and run your experiment
exp = Experiment(eval_method=ratio_split,
                 models=[pcrl],
                 metrics=[nDgc, rec, pre])
exp.run()


"""
Output:
     | NDCG@-1 | Recall@20 | Precision@20 | Train (s) | Test (s)
---- + ------- + --------- + ------------ + --------- + --------
pcrl |  0.1922 |    0.0862 |       0.0148 | 2591.4878 |   4.0957

*Results may change slightly from one run to another due to different random initial parameters
"""