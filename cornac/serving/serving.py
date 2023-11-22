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
"""CLI entry point for model serving.
"""

import os
import pickle
from flask import Flask, request


def _import_model_class(model_class):
    components = model_class.split(".")
    mod = __import__(".".join(components[:-1]), fromlist=[components[-1]])
    klass = getattr(mod, components[-1])
    return klass


def _load_model():
    model_dir = os.environ.get("MODEL_DIR")
    model_class = os.environ.get("MODEL_CLASS")
    train_set_dir = os.environ.get("TRAIN_SET")

    if model_dir is None:
        raise ValueError("MODEL_DIR environment variable is not set.")

    if model_class is None:
        raise ValueError("MODEL_CLASS environment variable is not set.")

    print(f"Loading the '{model_class}' model from '{model_dir}'...")

    global model, train_set
    model = _import_model_class(model_class).load(model_dir)

    train_set = None
    if train_set_dir is not None:
        with open(train_set_dir, "rb") as f:
            train_set = pickle.load(f)

    print("Model loaded" if train_set is None else
          "Model and train set loaded. Remove seen items by adding \
            'remove_seen=true' query param to the recommend endpoint."
          )


def create_app():
    app = Flask(__name__)
    _load_model()
    return app


app = create_app()


@app.route("/recommend", methods=["GET"])
def recommend():
    global model, train_set

    params = request.args
    uid = params.get("uid")
    k = params.get("k")
    remove_seen = params.get("remove_seen")
    
    if uid is None:
        return "uid is required", 400
    if k is None:
        k = -1
    if remove_seen is None:
        remove_seen = False
    elif remove_seen == "true":
        if train_set is None:
            return "Unable to remove seen items. 'train_set' is not provided", 400
        remove_seen = True

    response = model.recommend(
        user_id=uid,
        k=k,
        remove_seen=remove_seen,
        train_set=train_set,
    )

    return response


if __name__ == "__main__":
    app.run(host='0.0.0.0')
