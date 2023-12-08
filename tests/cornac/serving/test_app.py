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


@pytest.fixture()
def app():

    relative_path = os.path.dirname(__file__)
    print(relative_path)
    os.environ["MODEL_PATH"] = os.path.join(relative_path, "saved_model.pkl")
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
    response = client.get('/recommend?uid=123&k=10')
    assert response.status_code == 200
    assert len(response.json['recommendations']) == 10
    assert response.json['query']['uid'] == '123'
    assert response.json['query']['k'] == 10
    assert response.json['query']['remove_seen'] == False


def test_feedback(client):
    response = client.post('/feedback?uid=1&iid=1&rating=5')
    assert response.status_code == 200


def test_feedback_missing_uid(client):
    response = client.post('/feedback?iid=1&rating=5')
    assert response.status_code == 400
    assert response.data == b'uid is required'


def test_feedback_missing_iid(client):
    response = client.post('/feedback?uid=1&rating=5')
    assert response.status_code == 400
    assert response.data == b'iid is required'


def test_feedback_missing_rating(client):
    response = client.post('/feedback?uid=1&iid=1')
    assert response.status_code == 200
