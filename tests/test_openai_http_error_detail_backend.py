import asyncio
from types import SimpleNamespace

import httpx
import pytest
from guidellm.schemas import (
    GenerationRequest,
    GenerationRequestArguments,
    GenerationResponse,
    RequestInfo,
)

from benchmark_runner.openai_http_error_detail_backend import (
    ITL_TIMINGS_FIELD,
    OpenAIHTTPErrorDetailBackend,
    OpenAIHTTPErrorDetailBackendArgs,
    format_http_status_error,
    format_http_status_error_async,
)


def _build_http_status_error(
    status_code: int,
    *,
    json_body=None,
    text_body: str | None = None,
) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
    response_kwargs = {"status_code": status_code, "request": request}
    if json_body is not None:
        response_kwargs["json"] = json_body
    elif text_body is not None:
        response_kwargs["content"] = text_body.encode("utf-8")
    response = httpx.Response(**response_kwargs)
    return httpx.HTTPStatusError("request failed", request=request, response=response)


def test_format_http_status_error_with_openai_error_body():
    exc = _build_http_status_error(
        400,
        json_body={
            "error": {
                "message": "The input is longer than the model's context length.",
                "type": "BadRequestError",
                "code": 400,
            }
        },
    )

    message = format_http_status_error(exc)

    assert message.startswith("HTTP 400:")
    assert "The input is longer than the model's context length." in message
    assert "type=BadRequestError" in message
    assert "code=400" in message


def test_format_http_status_error_with_non_json_fallback():
    exc = _build_http_status_error(
        502,
        text_body="upstream gateway timeout",
    )

    message = format_http_status_error(exc)

    assert message.startswith("HTTP 502:")
    assert "upstream gateway timeout" in message


def test_format_http_status_error_without_response():
    request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
    exc = httpx.HTTPStatusError("request failed", request=request, response=None)

    message = format_http_status_error(exc)

    assert message.startswith("HTTP request failed:")


def test_format_http_status_error_async_with_unread_stream_response():
    request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
    response = httpx.Response(
        400,
        request=request,
        stream=httpx.ByteStream(
            b'{"error":{"message":"too long","type":"BadRequestError","code":400}}'
        ),
    )
    exc = httpx.HTTPStatusError("request failed", request=request, response=response)

    message = asyncio.run(format_http_status_error_async(exc))

    assert message.startswith("HTTP 400:")
    assert "too long" in message
    assert "type=BadRequestError" in message
    assert "code=400" in message


class _RecordingHandler:
    """Minimal request handler that records whether ``post_validation`` ran.

    ``post_validation`` is the last step of the handler lifecycle (added in
    guidellm 0.7.3): it rejects a compiled response with no text, no tool calls
    and no output tokens. Our backend mirrors upstream's ``_resolve_*`` bodies, so
    it has to call it at the same point — otherwise an unusable response is
    silently counted as a successful request with zero output.
    """

    def __init__(self, reject: bool = False):
        self.reject = reject
        self.validated: list[GenerationResponse] = []
        self.last_iteration_had_content = True

    def _response(self) -> GenerationResponse:
        return GenerationResponse(request_id="req-1", request_args="{}")

    def compile_non_streaming(self, request, arguments, data) -> GenerationResponse:
        return self._response()

    def compile_streaming(self, request, arguments) -> GenerationResponse:
        return self._response()

    def add_streaming_line(self, line: str) -> int | None:
        return None if line.strip() == "data: [DONE]" else 1

    def post_validation(self, response: GenerationResponse) -> None:
        self.validated.append(response)
        if self.reject:
            raise ValueError(
                "[UNUSABLE_BACKEND_RESPONSE] backend resolved with empty "
                "response payload"
            )


def _started_backend(handler: httpx.MockTransport) -> OpenAIHTTPErrorDetailBackend:
    backend = OpenAIHTTPErrorDetailBackend(
        OpenAIHTTPErrorDetailBackendArgs(target="http://localhost:8000", model="m")
    )
    backend._async_client = httpx.AsyncClient(transport=handler)
    return backend


async def _drain(agen) -> list:
    return [item async for item in agen]


@pytest.mark.parametrize("streaming", [False, True])
def test_request_uses_header_write_timestamp_trace(streaming, monkeypatch):
    seen = []
    request_info = RequestInfo(request_id="req-1")
    clock = SimpleNamespace(now=123.0)
    monkeypatch.setattr(
        "guidellm.backends.openai.http.time",
        SimpleNamespace(time=lambda: clock.now),
    )

    async def transport(request):
        trace = request.extensions["trace"]
        seen.append(trace)
        initial_start = request_info.timings.request_start
        await trace("http11.receive_response_headers.started", {})
        assert request_info.timings.request_start == initial_start
        await trace("http11.send_request_headers.started", {})
        assert (
            request_info.timings.request_start == 123.0
        ), "header-write event must timestamp request start"
        clock.now = 456.0
        await trace("http11.send_request_headers.started", {})
        assert request_info.timings.request_start == 123.0
        return httpx.Response(200, content=b"data: [DONE]\n\n" if streaming else b"{}")

    backend = _started_backend(httpx.MockTransport(transport))

    async def run():
        method = (
            backend._resolve_streaming if streaming else backend._resolve_non_streaming
        )
        try:
            await _drain(
                method(
                    GenerationRequest(),
                    request_info,
                    _RecordingHandler(),
                    GenerationRequestArguments(),
                    {
                        "method": "POST",
                        "url": "http://localhost:8000/v1/chat/completions",
                    },
                )
            )
        finally:
            await backend._async_client.aclose()

    asyncio.run(run())
    assert len(seen) == 1
    assert request_info.timings.request_start == 123.0


@pytest.mark.parametrize("reject", [False, True])
def test_resolve_non_streaming_runs_post_validation(reject):
    handler = _RecordingHandler(reject=reject)
    transport = httpx.MockTransport(lambda _req: httpx.Response(200, json={"id": "1"}))
    backend = _started_backend(transport)
    request_info = RequestInfo(request_id="req-1")

    async def run():
        return await _drain(
            backend._resolve_non_streaming(
                GenerationRequest(),
                request_info,
                handler,
                GenerationRequestArguments(),
                {"method": "POST", "url": "http://localhost:8000/v1/chat/completions"},
            )
        )

    if reject:
        # The ValueError must escape so the scheduler records the request as
        # errored instead of banking a zero-output success.
        with pytest.raises(ValueError, match="UNUSABLE_BACKEND_RESPONSE"):
            asyncio.run(run())
    else:
        yielded = asyncio.run(run())
        assert [response for response, _ in yielded] == handler.validated

    assert len(handler.validated) == 1


@pytest.mark.parametrize("reject", [False, True])
def test_resolve_streaming_runs_post_validation(reject):
    handler = _RecordingHandler(reject=reject)
    body = b'data: {"choices":[]}\n\ndata: [DONE]\n\n'
    transport = httpx.MockTransport(lambda _req: httpx.Response(200, content=body))
    backend = _started_backend(transport)
    request_info = RequestInfo(request_id="req-1")

    async def run():
        return await _drain(
            backend._resolve_streaming(
                GenerationRequest(),
                request_info,
                handler,
                GenerationRequestArguments(),
                {"method": "POST", "url": "http://localhost:8000/v1/chat/completions"},
            )
        )

    if reject:
        with pytest.raises(ValueError, match="UNUSABLE_BACKEND_RESPONSE"):
            asyncio.run(run())
    else:
        yielded = asyncio.run(run())
        # First yield is the TTFT marker (None), the last is the compiled response.
        assert yielded[0][0] is None
        assert [yielded[-1][0]] == handler.validated

    assert len(handler.validated) == 1


def _run_streaming_with_clock(monkeypatch, lines, timestamps, handler=None):
    """Drive ``_resolve_streaming`` over fixed lines and a fixed clock.

    Both are faked because the metric under test is a DURATION between chunks:
    a real MockTransport delivers the whole body at once, so every gap would
    measure as ~0 and the assertions could not tell a correct implementation
    from one that records garbage.
    """
    handler = handler or _RecordingHandler()

    async def fake_lines(self, stream):
        for line in lines:
            yield line

    monkeypatch.setattr(OpenAIHTTPErrorDetailBackend, "_aiter_lines", fake_lines)

    transport = httpx.MockTransport(lambda _req: httpx.Response(200, content=b""))
    backend = _started_backend(transport)
    request_info = RequestInfo(request_id="req-1")

    # Replace the MODULE's ``time`` name, not ``time.time`` itself: httpx reads
    # the real clock too (its cookie jar does, between request_start and the
    # first chunk), and patching the shared function would let those calls eat
    # the scripted values and shift every timestamp.
    class _Clock:
        def __init__(self, values: list[float]):
            self.pending = list(values)
            self.last = values[-1]

        def time(self) -> float:
            # Falls back to the last value once exhausted so an extra call
            # cannot turn into an IndexError.
            return self.pending.pop(0) if self.pending else self.last

    monkeypatch.setattr(
        "benchmark_runner.openai_http_error_detail_backend.time",
        _Clock(timestamps),
    )

    asyncio.run(
        _drain(
            backend._resolve_streaming(
                GenerationRequest(),
                request_info,
                handler,
                GenerationRequestArguments(),
                {"method": "POST", "url": "http://localhost:8000/v1/chat/completions"},
            )
        )
    )
    return request_info


def test_resolve_streaming_records_itl_gaps(monkeypatch):
    """One sample per gap between streamed outputs; the first chunk has none.

    The first interval is TTFT, so three token chunks must yield exactly two
    ITL samples — same convention as vLLM.
    """
    request_info = _run_streaming_with_clock(
        monkeypatch,
        lines=[
            'data: {"a":1}',
            'data: {"a":2}',
            'data: {"a":3}',
            "data: [DONE]",
        ],
        # request_start, chunk1, chunk2, chunk3, [DONE], request_end
        timestamps=[0.0, 0.100, 0.150, 0.950, 0.960, 1.000],
    )

    gaps = getattr(request_info.timings, ITL_TIMINGS_FIELD)
    # 0.150-0.100 = 50ms, then 0.950-0.150 = 800ms: the stall survives as its
    # own sample rather than being averaged into the request's other gaps.
    assert gaps == pytest.approx([50.0, 800.0])
    # TTFT keeps the first interval, and it is NOT one of the ITL samples.
    assert request_info.timings.first_token_iteration == pytest.approx(0.100)


def test_resolve_streaming_single_chunk_records_no_itl(monkeypatch):
    """A response arriving as one chunk has no inter-token gap to report.

    It must come back as an empty list, not as ``[0.0]`` — a zero-millisecond
    gap would enter the distribution and drag the whole tail down.
    """
    request_info = _run_streaming_with_clock(
        monkeypatch,
        lines=['data: {"a":1}', "data: [DONE]"],
        timestamps=[0.0, 0.100, 0.110, 0.120],
    )

    assert getattr(request_info.timings, ITL_TIMINGS_FIELD) == []


def test_resolve_streaming_itl_survives_model_roundtrip(monkeypatch):
    """The samples must reach the aggregation process intact.

    ``RequestInfo`` crosses the scheduler's process boundary through
    ``MessageEncoding``, which dict-dumps and re-validates the model. The
    samples ride on ``timings`` (``extra="allow"``) precisely because
    ``RequestInfo`` itself is ``extra="ignore"`` and would drop them.
    """
    request_info = _run_streaming_with_clock(
        monkeypatch,
        lines=['data: {"a":1}', 'data: {"a":2}', "data: [DONE]"],
        timestamps=[0.0, 0.100, 0.300, 0.310, 0.320],
    )

    restored = RequestInfo.model_validate(request_info.model_dump())

    assert getattr(restored.timings, ITL_TIMINGS_FIELD) == pytest.approx([200.0])


def _guidellm_stats(request_info: RequestInfo):
    """The same request as guidellm's own stats object.

    Built through `compile_stats`, so `inter_token_latency_ms` below is
    guidellm's computation, not a formula restated in the test.
    """
    response = GenerationResponse(request_id="req-1", request_args="{}")
    return response.compile_stats(
        request=GenerationRequest(request_id="req-1"),
        info=request_info,
        prefer_response=False,
    )


def test_our_itl_mean_equals_guidellms_own_per_token_latency(monkeypatch):
    """Cross-check against a value we did not compute.

    The mean of our recorded gaps and guidellm's `inter_token_latency_ms` are
    the same quantity reached two independent ways:

        ours:     sum(gaps) / len(gaps)
        guidellm: (last_token - first_token) / (output_tokens - 1)

    `sum(gaps)` telescopes to `last_token - first_token`, and `len(gaps)` is
    `output_tokens - 1`, so they must agree exactly. Any drift means a chunk
    was dropped, counted twice, or the first one was not excluded — none of
    which the fixed-sample tests above can see, because they assert against
    gaps the test itself supplied.
    """
    request_info = _run_streaming_with_clock(
        monkeypatch,
        lines=[
            'data: {"a":1}',
            'data: {"a":2}',
            'data: {"a":3}',
            'data: {"a":4}',
            "data: [DONE]",
        ],
        # Deliberately uneven, including one long stall, so an implementation
        # that averaged wrongly could not coincidentally match.
        timestamps=[0.0, 0.100, 0.150, 0.950, 1.100, 1.110, 1.120],
    )

    gaps = getattr(request_info.timings, ITL_TIMINGS_FIELD)
    stats = _guidellm_stats(request_info)

    assert len(gaps) == 3
    assert sum(gaps) / len(gaps) == pytest.approx(stats.inter_token_latency_ms)


def test_gap_sum_telescopes_to_the_decode_window(monkeypatch):
    """The gaps must cover exactly first-token → last-token, no more, no less.

    Catches both directions of error at once: including the TTFT interval would
    overshoot, dropping a chunk would undershoot.
    """
    request_info = _run_streaming_with_clock(
        monkeypatch,
        lines=['data: {"a":1}', 'data: {"a":2}', 'data: {"a":3}', "data: [DONE]"],
        timestamps=[0.0, 0.100, 0.150, 0.950, 0.960, 1.000],
    )

    gaps = getattr(request_info.timings, ITL_TIMINGS_FIELD)
    timings = request_info.timings
    decode_window_ms = 1000 * (
        timings.last_token_iteration - timings.first_token_iteration
    )

    assert sum(gaps) == pytest.approx(decode_window_ms)
