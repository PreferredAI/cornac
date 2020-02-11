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
"""Example on how to train and evaluate a model with provided train and test sets"""

from cornac.data import Reader
from cornac.eval_methods import BaseMethod
from cornac.models import MF
from cornac.metrics import MAE, RMSE
from cornac.utils import cache


# Download MovieLens 100K provided train and test sets
reader = Reader()
train_data = reader.read(
    cache(url="http://files.grouplens.org/datasets/movielens/ml-100k/u1.base")
)
test_data = reader.read(
    cache(url="http://files.grouplens.org/datasets/movielens/ml-100k/u1.test")
)

# Instantiate a Base evaluation method using the provided train and test sets
eval_method = BaseMethod.from_splits(
    train_data=train_data, test_data=test_data, exclude_unknowns=False, verbose=True
)

# Instantiate the MF model
mf = MF(
    k=10,
    max_iter=25,
    learning_rate=0.01,
    lambda_reg=0.02,
    use_bias=True,
    early_stop=True,
    verbose=True,
)

# Evaluation
test_result, val_result = eval_method.evaluate(
    model=mf, metrics=[MAE(), RMSE()], user_based=True
)
print(test_result)
