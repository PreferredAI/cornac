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
"""Example for hyper-parameter searching with Optuna, with pruning of
unpromising trials (requires `pip install optuna`)

SASRec on 5-core Amazon Beauty reviews with per-user leave-last-out
splitting. SASRec gives its validation score to `on_epoch_end()` every
`val_eval_every` epochs, so the pruner can stop a trial before it has trained
for all epochs.
"""

import optuna
import torch

import cornac
from cornac.datasets import amazon_review
from cornac.eval_methods import NextItemEvaluation
from cornac.hyperopt import Discrete, OptunaSearch
from cornac.metrics import NDCG, Recall
from cornac.models import SASRec

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load Amazon Beauty reviews
data = amazon_review.load_feedback(category="beauty")

# For each user, hold out the last item for testing and the one before for validation
next_item_eval = NextItemEvaluation.leave_last_out(
    data=data, exclude_unknowns=True, verbose=True
)

# Define a base SASRec model with fixed hyper-parameters
sasrec = SASRec(
    num_blocks=2,
    num_heads=2,
    batch_size=128,
    n_epochs=200,
    model_selection="best",
    val_k=10,
    val_eval_every=5,
    patience=20,
    device=DEVICE,
    seed=123,
)

# Instantiate Recall@10 to score the trials on the validation set
rec_10 = Recall(k=10)

# Wrap SASRec model inside OptunaSearch along with the searching space, try 30 times.
# A trial is pruned when its validation score is below the median of the earlier
# trials at the same epoch, after 5 trials and 20 epochs.
os_sasrec = OptunaSearch(
    model=sasrec,
    space=[
        Discrete("loss", ["bce", "ce", "bpr", "bpr-max", "softmax"]),
        Discrete("learning_rate", [0.0005, 0.001, 0.005]),
        Discrete("dropout", [0.2, 0.5]),
        Discrete("embedding_dim", [64, 128]),
        Discrete("max_len", [20, 50]),
    ],
    metric=rec_10,
    eval_method=next_item_eval,
    n_trials=30,
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20),
)

# Put everything together into an experiment and run it
cornac.Experiment(
    eval_method=next_item_eval,
    models=[os_sasrec],
    metrics=[Recall(k=10), Recall(k=20), NDCG(k=10), NDCG(k=20)],
    user_based=False,
).run()

# Best parameter settings, and how each trial ended
print("Best parameters:", os_sasrec.best_params)
for trial in os_sasrec.study.trials:
    print(trial.number, trial.state.name, trial.value, trial.params)
