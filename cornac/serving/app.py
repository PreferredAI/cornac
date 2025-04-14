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
import sys
import inspect
from datetime import datetime, timezone
from csv import writer

from cornac.data import Dataset, Reader
from cornac.eval_methods import BaseMethod
from cornac.metrics import *

try:
    from flask import Flask, jsonify, request, abort, make_response
except ImportError:
    exit("Flask is required in order to serve models.\n" + "Run: pip3 install Flask")


ALLOWED_METRIC_NAMES = {
    name: obj
    for name, obj in inspect.getmembers(sys.modules[__name__])
    if inspect.isclass(obj) and obj.__module__.startswith("cornac.metrics")
}


def _safe_eval(metric_str):
    code = compile(metric_str, "<string>", "eval")
    for name in code.co_names:
        if name not in ALLOWED_METRIC_NAMES:
            raise NameError(f"Use of {name} not allowed")
    return eval(code, {"__builtins__": {}}, ALLOWED_METRIC_NAMES)


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
        train_set = Dataset.load(train_set_path)
    elif os.path.exists(train_set_path := model.load_from + ".trainset"):
        train_set = Dataset.load(train_set_path)

    print(
        "Model loaded"
        if train_set is None
        else """
        Model and train set loaded. Remove seen items by adding 
        remove_seen=true' query param to the recommend endpoint.
        """
    )


def _get_cornac_metric_classnames():
    """For security checking in the evaluate API"""
    global metric_classnames

    metric_classnames = set()
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj) and obj.__module__.startswith("cornac.metrics"):
            metric_classnames.add(name)


def create_app():
    app = Flask(__name__)
    _load_model(app.instance_path)
    _get_cornac_metric_classnames()
    return app


app = create_app()


@app.route("/recommend", methods=["GET"])
def recommend():
    global model, train_set # noqa: F824

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
    data_fpath = "data/feedback.csv"

    if uid is None:
        return "uid is required", 400

    if iid is None:
        return "iid is required", 400

    os.makedirs(os.path.dirname(data_fpath), exist_ok=True)

    with open(data_fpath, "a+", newline="") as write_obj:
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


@app.route("/evaluate", methods=["POST"])
def evaluate():
    global model, train_set, metric_classnames # noqa: F824

    if model is None:
        return "Model is not yet loaded. Please try again later.", 400

    if train_set is None:
        return "Unable to evaluate. 'train_set' is not provided", 400

    query = request.json
    validate_query(query)

    exclude_unknowns = (
        query.get("exclude_unknowns", "true").lower() == "true"
    )  # exclude unknown users/items by default, otherwise specified

    if "data" in query:
        data = query.get("data")
    else:
        data = []
        data_fpath = "data/feedback.csv"
        if os.path.exists(data_fpath):
            reader = Reader()
            data = reader.read(data_fpath, fmt="UIR", sep=",")

    if not data:
        response = make_response("No feedback has been provided so far. No data available to evaluate the model.")
        response.status_code = 400
        abort(response)

    test_set = Dataset.build(
        data,
        fmt="UIR",
        global_uid_map=train_set.uid_map,
        global_iid_map=train_set.iid_map,
        exclude_unknowns=exclude_unknowns,
    )

    return process_evaluation(test_set, query, exclude_unknowns)


def validate_query(query):
    query_metrics = query.get("metrics")

    if not query_metrics:
        response = make_response("metrics is required")
        response.status_code = 400
        abort(response)
    elif not isinstance(query_metrics, list):
        response = make_response("metrics must be an array of metrics")
        response.status_code = 400
        abort(response)


def process_evaluation(test_set, query, exclude_unknowns):
    global model, train_set # noqa: F824
    
    rating_threshold = query.get("rating_threshold", 1.0)
    user_based = (
        query.get("user_based", "true").lower() == "true"
    )  # user_based evaluation by default, otherwise specified

    query_metrics = query.get("metrics")

    # organize metrics
    metrics = []
    for metric in query_metrics:
        try:
            metrics.append(_safe_eval(metric))
        except:
            return (
                f"Invalid metric initiation: {metric}.\n"
                + "Please input correct metrics (e.g., 'RMSE()', 'Recall(k=10)')",
                400,
            )

    rating_metrics, ranking_metrics = BaseMethod.organize_metrics(metrics)

    # evaluation
    result = BaseMethod.eval(
        model=model,
        train_set=train_set,
        test_set=test_set,
        val_set=None,
        rating_threshold=rating_threshold,
        exclude_unknowns=exclude_unknowns,
        rating_metrics=rating_metrics,
        ranking_metrics=ranking_metrics,
        user_based=user_based,
        verbose=False,
    )

    # map user index back into the original user ID
    metric_user_results = {}
    for metric, user_results in result.metric_user_results.items():
        metric_user_results[metric] = {
            train_set.user_ids[int(k)]: v for k, v in user_results.items()
        }

    # response
    response = {
        "result": result.metric_avg_results,
        "user_result": metric_user_results,
    }

    return jsonify(response), 200


if __name__ == "__main__":
    app.run()
