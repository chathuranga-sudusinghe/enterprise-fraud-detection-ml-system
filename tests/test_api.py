from fastapi.testclient import TestClient
import pandas as pd

from api import main as api_main
from api.main import app

client = TestClient(app)

def test_predict_endpoint():

    sample_request = {
     "data": {
        "TransactionDT": 86400,
        "TransactionAmt": 100.0,
        "card1": 1234,
        "card2": 111,
        "card3": 150,
        "card4": "visa",
        "addr1": 100
      }
    }

    response = client.post("/predict", json=sample_request)

    assert response.status_code == 200
    assert "fraud_prediction" in response.json()
    assert "fraud_probability" in response.json()


def test_predict_endpoint_response_contract_remains_v1_only():

    sample_request = {
     "data": {
        "TransactionDT": 86400,
        "TransactionAmt": 100.0,
        "card1": 1234,
        "card2": 111,
        "card3": 150,
        "card4": "visa",
        "addr1": 100
      }
    }

    response = client.post("/predict", json=sample_request)

    assert response.status_code == 200
    assert set(response.json()) == {"fraud_probability", "fraud_prediction"}


def test_predict_v2_endpoint_returns_model_v2_contract(monkeypatch):
    monkeypatch.setattr(api_main, "predictor_v2", FakePredictorV2())

    response = client.post("/predict/v2", json={"data": {"TransactionAmt": 100.0}})

    assert response.status_code == 200
    assert response.json() == {
        "fraud_probability": 0.91,
        "fraud_prediction": 1,
        "threshold": 0.1,
        "model_version": "v2",
        "model_family": "catboost",
        "feature_count": 831,
    }


def test_predict_v2_endpoint_returns_controlled_error(monkeypatch):
    monkeypatch.setattr(api_main, "predictor_v2", FailingPredictorV2())

    response = client.post("/predict/v2", json={"data": {"TransactionAmt": 100.0}})

    assert response.status_code == 500
    assert response.json() == {"detail": "Model v2 prediction error"}


class FakePredictorV2:
    def predict(self, input_df):
        return pd.DataFrame(
            {
                "fraud_probability": [0.91],
                "fraud_prediction": [1],
                "threshold": [0.10],
                "model_version": ["v2"],
                "model_family": ["catboost"],
                "feature_count": [831],
            }
        )


class FailingPredictorV2:
    def predict(self, input_df):
        raise RuntimeError("simulated v2 failure")
