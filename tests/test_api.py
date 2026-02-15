from fastapi.testclient import TestClient
from api.index import app

client = TestClient(app)

def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_endpoint_no_file():
    """Test prediction without file upload."""
    response = client.post("/api/predict")
    assert response.status_code == 422  # Validation Error
