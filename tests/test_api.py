from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from dispatcher import InferenceError, Overloaded, ServiceUnavailable
from server import app


class StubDispatcher:
    enable_batching = True

    def __init__(self, workers=2, error=None):
        self.workers = workers
        self.error = error

    def stats(self):
        return {
            "workers": self.workers,
            "configured_workers": 2,
            "pending_requests": 0,
            "max_pending_requests": 256,
        }

    async def predict_async(self, array, timeout_s):
        if self.error:
            raise self.error
        return {"prediction": 0}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        app, "state", SimpleNamespace(dispatcher=StubDispatcher(), request_timeout_s=5)
    )
    return TestClient(app)


@pytest.mark.parametrize(
    "pixels", [[], [0] * 783, [0] * 785, [2] * 784, [-0.1] * 784, ["NaN"] * 784, ["Infinity"] * 784]
)
def test_invalid_pixels(client, pixels):
    assert client.post("/predict", json={"pixels": pixels}).status_code == 422


def test_valid_pixels(client):
    response = client.post("/predict", json={"pixels": [0.5] * 784})
    assert response.status_code == 200
    assert response.json()["prediction"] == 0
    assert response.json()["total_latency_ms"] >= 0


@pytest.mark.parametrize(
    "error,code",
    [
        (Overloaded("full"), 503),
        (ServiceUnavailable("recovering"), 503),
        (TimeoutError(), 504),
        (InferenceError("private traceback"), 500),
    ],
)
def test_error_mapping(client, error, code):
    app.state.dispatcher.error = error
    response = client.post("/predict", json={"pixels": [0] * 784})
    assert response.status_code == code
    assert "private traceback" not in response.text
    if code == 503:
        assert response.headers["Retry-After"] == "1"


@pytest.mark.parametrize(
    "workers,status,code", [(2, "ok", 200), (1, "degraded", 200), (0, "unavailable", 503)]
)
def test_readiness(client, workers, status, code):
    app.state.dispatcher.workers = workers
    response = client.get("/health")
    assert response.status_code == code
    assert response.json()["status"] == status
    assert client.get("/live").status_code == 200


def test_service_without_dispatcher(client):
    app.state.dispatcher = None
    assert client.get("/health").status_code == 503
    assert client.post("/predict", json={"pixels": [0] * 784}).status_code == 503


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_raw_json(client, value):
    content = '{"pixels":[' + value + ",0" * 783 + "]}"
    assert (
        client.post(
            "/predict", content=content, headers={"Content-Type": "application/json"}
        ).status_code
        == 422
    )
