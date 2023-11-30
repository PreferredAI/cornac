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
from datetime import datetime, timezone
from csv import writer, reader
from cornac.data import Dataset
from cornac.metrics import Precision, Recall
from cornac.eval_methods import BaseMethod

try:
    from flask import Flask, jsonify, request
except ImportError:
    exit(
        "Flask is required in order to serve models.\n"
        + "Run: pip3 install Flask"
    )


def _import_model_class(model_class):
    components = model_class.split(".")
    mod = __import__(".".join(components[:-1]), fromlist=[components[-1]])
    klass = getattr(mod, components[-1])
    return klass


def _load_model(instance_path):
    model_path = os.environ.get("MODEL_PATH")
    model_class = os.environ.get("MODEL_CLASS")
    train_set_path = os.environ.get("TRAIN_SET")

    if model_path is None:
        raise ValueError("MODEL_PATH environment variable is not set.")
    elif not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(instance_path), model_path)

    if model_class is None:
        raise ValueError("MODEL_CLASS environment variable is not set.")

    print(f"Loading the '{model_class}' model from '{model_path}'...")

    global model, train_set

    try:
        model = _import_model_class(model_class).load(model_path)
    except:  # fallback to Recommender as our last resort
        from ..models import Recommender

        model = Recommender.load(model_path)

    train_set = None
    if train_set_path is not None:
        if not os.path.isabs(train_set_path):
            train_set_path = os.path.join(
                os.path.dirname(instance_path), train_set_path
            )
        with open(train_set_path, "rb") as f:
            train_set = pickle.load(f)

    print(
        "Model loaded"
        if train_set is None
        else "Model and train set loaded. Remove seen items by adding \
            'remove_seen=true' query param to the recommend endpoint."
    )


def create_app():
    app = Flask(__name__)
    _load_model(app.instance_path)
    return app


app = create_app()


@app.route("/recommend", methods=["GET"])
def recommend():
    global model, train_set

    if model is None:
        return "Model is not yet loaded. Please try again later.", 400

    params = request.args
    uid = params.get("uid")
    k = int(params.get("k", -1))
    remove_seen = params.get("remove_seen", "false").lower() == "true"

    if uid is None:
        return "uid is required", 400

    if remove_seen and train_set is None:
        return "Unable to remove seen items. 'train_set' is not provided", 400

    response = model.recommend(
        user_id=uid,
        k=k,
        remove_seen=remove_seen,
        train_set=train_set,
    )

    data = {
        "recommendations": response,
        "query": {"uid": uid, "k": k, "remove_seen": remove_seen},
    }

    return jsonify(data), 200


@app.route("/feedback", methods=["POST"])
def add_feedback():
    params = request.args
    uid = params.get("uid")
    iid = params.get("iid")
    rating = params.get("rating", 1)
    time = datetime.now(timezone.utc)
    directory = "data/"

    if uid is None:
        return "uid is required", 400

    if iid is None:
        return "iid is required", 400

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(directory + "feedback.csv", "a+", newline="") as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow([uid, iid, rating, time])
        write_obj.close()

    data = {
        "message": "Feedback added",
        "data": {
            "uid": uid,
            "iid": iid,
            "rating": rating,
            "time": str(time),
        },
    }

    return jsonify(data), 200

@app.route("/evaluate", methods=["GET"])
def evaluate():
    global model, train_set

    # read from csv
    directory = "data/"
    file_name = "feedback.csv"
    if os.path.exists(directory + file_name):
        with open(directory + file_name, "r") as read_obj:
            csv_reader = reader(read_obj)
            list_of_rows = list(csv_reader)
            read_obj.close()
    else:
        list_of_rows = []
    
    test_data = [(row[0], row[1], row[2]) for row in list_of_rows]
    test_set = Dataset.from_uir(test_data)

    if model is None:
        return "Model is not yet loaded. Please try again later.", 400

    if train_set is None:
        return "Unable to evaluate. 'train_set' is not provided", 400
    
    metrics = [Precision(k=10), Recall(k=10)]

    response = BaseMethod.from_splits(
        train_data=train_set,
        test_data=test_set,
        exclude_unknowns=True,
        verbose=True,
    ).evaluate(model, metrics)

    data = {
        "evaluation": response,
        "query": {"metrics": metrics},
    }

    return jsonify(data), 200


if __name__ == "__main__":
    app.run()
