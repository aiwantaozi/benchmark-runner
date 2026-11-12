"""Exercise the real scheduler, HTTP transport, progress and output without a GPU."""

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from benchmark_runner.execution import benchmark_generative_text
from benchmark_runner.auto_tune import _normalize
from benchmark_runner.output_dual_json import ITL_METRIC_KEY
from benchmark_runner.progress import ServerBenchmarkerProgress
from benchmark_runner.scenario_builder import build_scenario_args
from guidellm.settings import settings


@pytest.mark.parametrize("with_errors,concurrency", [(True, 1), (False, 16)])
def test_real_http_streaming_goodput_errors_and_progress(
    tmp_path, monkeypatch, with_errors, concurrency
):
    monkeypatch.setattr(settings, "max_worker_processes", 1)
    updates = []

    class Handler(BaseHTTPRequestHandler):
        requests = 0
        requests_lock = threading.Lock()

        def log_message(self, *args):
            pass

        def handle(self):
            try:
                super().handle()
            except (BrokenPipeError, ConnectionResetError):
                # The completed-count boundary intentionally cancels streams.
                pass

        def do_GET(self):
            body = json.dumps({"data": [{"id": "test-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/progress":
                updates.append(payload)
                self.send_response(200)
                self.end_headers()
                return
            with Handler.requests_lock:
                Handler.requests += 1
                request_number = Handler.requests
            if with_errors and request_number % 5 == 0:
                body = (
                    b'{"error":{"message":"synthetic overload",'
                    b'"type":"CapacityError","code":429}}'
                )
                self.send_response(429)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for word in ["one", " two", " three"]:
                chunk = {"choices": [{"index": 0, "delta": {"content": word}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.005)
            self.wfile.write(
                b'data: {"choices":[],"usage":{"prompt_tokens":2,'
                b'"completion_tokens":3,"total_tokens":5}}\n\ndata: [DONE]\n\n'
            )
            self.wfile.flush()

        def do_PATCH(self):
            updates.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            self.send_response(200)
            self.end_headers()

    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer_path = tmp_path / "tokenizer"
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]"
    ).save_pretrained(tokenizer_path)
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps({"prompt": "hello world", "output_tokens": 3})
            for _ in range(100)
        )
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    target = f"http://127.0.0.1:{server.server_port}"
    scenario = build_scenario_args(
        {
            "backend": "openai_http_error_detail",
            "backend_kwargs": {"target": target, "model": "test-model"},
            "processor": str(tokenizer_path),
            "data": [str(dataset)],
            "profile": "concurrent",
            "rate": [concurrency],
            "min_completed_requests": 50,
            "max_seconds": 30,
            "metrics": {"slo": {"ttft_ms": 10000, "tpot_ms": 10000}, "warnings": []},
            "outputs": [str(tmp_path / "result.dual_json")],
        }
    )
    tracker = ServerBenchmarkerProgress(progress_url=target + "/progress")
    try:
        report, _ = asyncio.run(
            benchmark_generative_text(args=scenario, progress=tracker)
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert updates, "custom HTTP progress callback was lost"
    metrics = report.benchmarks[0].metrics
    assert metrics.request_totals.successful > 0
    if with_errors:
        assert metrics.request_totals.errored > 0
        assert 0 < metrics.slo_attainment < 1
    else:
        assert metrics.request_totals.errored == 0
        assert metrics.slo_attainment == 1
        assert _normalize(report.benchmarks[0], concurrency, 0).success >= 0.95
    assert metrics.request_goodput.successful.mean > 0
    summary = json.loads((tmp_path / "result.json").read_text())
    full = json.loads((tmp_path / "result.full.json").read_text())
    assert (
        summary["benchmarks"][0]["metrics"]["slo_attainment"] == metrics.slo_attainment
    )
    if with_errors:
        assert "synthetic overload" in json.dumps(full)
        assert "CapacityError" in json.dumps(full)
    assert summary["benchmarks"][0]["metrics"][ITL_METRIC_KEY]["successful"]["max"] > 0
