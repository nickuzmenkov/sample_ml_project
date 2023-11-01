from unittest import mock

import numpy as np
import pandas as pd
import pytest
from iris.backend.app import app, IrisType
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_healthcheck(client: TestClient):
    response = client.request("GET", "/healthcheck")
    assert response.status_code == 200


def test_predict(client: TestClient):
    data = pd.DataFrame(
        columns=["petal_width", "petal_height", "sepal_width", "sepal_height"],
        data=np.random.uniform(low=0.0, high=10.0, size=(10, 4)),
    )
    data["target"] = np.random.randint(low=0, high=3, size=(len(data),))

    with mock.patch("iris.backend.app.fetch_data", return_value=data):
        response = client.request(
            "GET",
            "/predict",
            json={
                "petal_height": 10,
                "petal_width": 10,
                "sepal_height": 10,
                "sepal_width": 10,
            },
        )
    assert response.status_code == 200
    IrisType(response.json()["predict"])
