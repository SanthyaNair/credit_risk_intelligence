# tests/test_api.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint():
    payload = {
        "AMT_CREDIT": 500000,
        "AMT_INCOME_TOTAL": 150000,
        "AMT_ANNUITY": 24000,
        "DAYS_EMPLOYED": -1500,
        "DAYS_BIRTH": -12000,
        "EXT_SOURCE_2": 0.65,
        "EXT_SOURCE_3": 0.55,
        "CNT_FAM_MEMBERS": 2.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "decision" in response.json()
    assert "default_probability" in response.json()