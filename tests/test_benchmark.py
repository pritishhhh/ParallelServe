from types import SimpleNamespace

import pytest
import requests

from benchmarks import load_test


def test_failed_responses_do_not_count_as_throughput(monkeypatch):
    class Response:
        status_code = 503

        def raise_for_status(self):
            raise requests.HTTPError(response=self)

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr(requests, "Session", Session)
    latencies, batch_sizes, correct, errors = load_test.client_run("http://unused", 3, 1, 0)
    assert latencies == batch_sizes == []
    assert correct == 0
    assert errors == {"http_503": 3}


def test_invalid_success_payload_is_counted_as_failure(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"error": "Model inference failed"}

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr(requests, "Session", Session)
    latencies, _, _, errors = load_test.client_run("http://unused", 2, 1, 0)
    assert not latencies
    assert errors == {"ValueError": 2}


def test_measure_separates_warmup_and_failures(monkeypatch):
    def client_run(url, count, timeout, seed):
        if seed >= 1000:
            return [1000], [1], 1, {}
        return [10, 20], [2, 2], 1, {"http_503": 1}

    monkeypatch.setattr(load_test, "client_run", client_run)
    args = SimpleNamespace(concurrency=2, warmup=1, timeout=1, requests_per_client=3)
    result = load_test.measure("unused", args)
    assert result["attempted"] == 6
    assert result["successful"] == 4
    assert result["failed"] == 2
    assert result["errors"] == {"http_503": 2}
    assert result["accuracy"] == 0.5
    assert result["p50_ms"] == 15


@pytest.mark.parametrize(
    "argv",
    [
        ["--concurrency", "0"],
        ["--window-ms", "nan"],
        ["--warmup", "0"],
        ["--workers", "-1"],
        ["--timeout", "inf"],
    ],
)
def test_invalid_benchmark_settings(argv):
    with pytest.raises(SystemExit):
        load_test.parse_args(argv)
