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

import os
import pytest

from cornac.data import Reader, Dataset
from cornac.models import BPR


@pytest.fixture()
def load_model():
    triplet_data = Reader().read("tests/data.txt")
    train_set = Dataset.from_uir(triplet_data)
    model = BPR(k=10, max_iter=100, learning_rate=0.01, lambda_reg=0.01, seed=123)
    model.fit(train_set)
    return model.save(save_dir="saved_models", save_trainset=True)  # returns directory path


@pytest.fixture()
def app(load_model):
    current_path = os.getcwd()
    os.environ["MODEL_PATH"] = os.path.join(current_path, load_model)
    os.environ["MODEL_CLASS"] = "cornac.models.BPR"

    from cornac.serving.app import app

    app.config.update({
        "TESTING": True,
    })

    yield app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def runner(app):
    return app.test_cli_runner()


def test_create_app(app):
    assert app.name == 'cornac.serving.app'


def test_recommend(client):
    response = client.get('/recommend?uid=930&k=5')
    assert response.status_code == 200
    assert len(response.json['recommendations']) == 5
    assert response.json['query']['uid'] == '930'
    assert response.json['query']['k'] == 5
    assert response.json['query']['remove_seen'] == False


def test_feedback(client):
    response = client.post('/feedback?uid=930&iid=795&rating=5')
    assert response.status_code == 200


def test_feedback_missing_uid(client):
    response = client.post('/feedback?iid=795&rating=5')
    assert response.status_code == 400
    assert response.data == b'uid is required'


def test_feedback_missing_iid(client):
    response = client.post('/feedback?uid=930&rating=5')
    assert response.status_code == 400
    assert response.data == b'iid is required'


def test_feedback_missing_rating(client):
    response = client.post('/feedback?uid=195&iid=795')
    assert response.status_code == 200


def test_evaluate_json(client):
    json_data = {
        'metrics': ['RMSE()', 'Recall(k=5)']
    }
    response = client.post('/evaluate', json=json_data)
    # assert response.content_type == 'application/json'
    assert response.status_code == 200
    assert 'RMSE' in response.json['result']
    assert 'Recall@5' in response.json['result']
    assert 'RMSE' in response.json['user_result']
    assert 'Recall@5' in response.json['user_result']


def test_evalulate_incorrect_get(client):
    response = client.get('/evaluate')
    assert response.status_code == 405  # method not allowed


def test_evalulate_incorrect_post(client):
    response = client.post('/evaluate')
    assert response.status_code == 415  # bad request, expect json


def test_evaluate_missing_metrics(client):
    json_data = {
        'metrics': []
    }
    response = client.post('/evaluate', json=json_data)
    assert response.status_code == 400
    assert response.data == b'metrics is required'


def test_evaluate_not_list_metrics(client):
    json_data = {
        'metrics': 'RMSE()'
    }
    response = client.post('/evaluate', json=json_data)
    assert response.status_code == 400
    assert response.data == b'metrics must be an array of metrics'


def test_recommend_missing_uid(client):
    response = client.get('/recommend?k=5')
    assert response.status_code == 400
    assert response.data == b'uid is required'


def test_evaluate_use_data(client):
    json_data = {
        'metrics': ['RMSE()', 'Recall(k=5)'],
        'use_data': [['930', '795', 5], ['195', '795', 3]]
    }
    response = client.post('/evaluate', json=json_data)
    # assert response.content_type == 'application/json'
    assert response.status_code == 200
    assert 'RMSE' in response.json['result']
    assert 'Recall@5' in response.json['result']
    assert 'RMSE' in response.json['user_result']
    assert 'Recall@5' in response.json['user_result']


def test_evaluate_use_data_empty(client):
    json_data = {
        'metrics': ['RMSE()', 'Recall(k=5)'],
        'use_data': []
    }
    response = client.post('/evaluate', json=json_data)
    assert response.status_code == 400
    assert response.data == b"'use_data' is empty. No data available to evaluate the model."


