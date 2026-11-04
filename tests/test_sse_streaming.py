import asyncio
import json
from contextlib import aclosing

import httpx
import pytest
from guidellm.schemas import (
    GenerationRequest,
    GenerationRequestArguments,
    RequestInfo,
)

from benchmark_runner.custom_response_handler import (
    ChatCompletionsWithReasoningResponseHandler,
)
from benchmark_runner.openai_http_error_detail_backend import (
    OpenAIHTTPErrorDetailBackend,
    OpenAIHTTPErrorDetailBackendArgs,
)


class _ChunkedStream(httpx.AsyncByteStream):
    def __init__(self, body: bytes, chunk_size: int):
        self.body = body
        self.chunk_size = chunk_size

    async def __aiter__(self):
        for offset in range(0, len(self.body), self.chunk_size):
            yield self.body[offset : offset + self.chunk_size]


def _event(data: dict, ending: str) -> str:
    return "data: " + json.dumps(data, ensure_ascii=False) + ending * 2


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
@pytest.mark.parametrize("field", ["content", "reasoning_content"])
@pytest.mark.parametrize("ending", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize("chunk_size", [1, 7, None])
def test_resolve_streaming_preserves_unicode(separator, field, ending, chunk_size):
    text = "左" + separator + "右\nnext"
    body = (
        _event({"id": "response-1", "choices": [{"delta": {field: text}}]}, ending)
        + _event({"choices": [{"delta": {"content": "尾"}}]}, ending)
        + _event(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 5,
                    "total_tokens": 14,
                },
            },
            ending,
        )
        + "data: [DONE]"
        + ending * 2
    ).encode("utf-8")

    async def run():
        backend = OpenAIHTTPErrorDetailBackend(
            OpenAIHTTPErrorDetailBackendArgs(target="http://test", model="m")
        )
        transport = httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                headers={"content-type": "text/event-stream; charset=utf-8"},
                stream=_ChunkedStream(body, chunk_size or len(body)),
            )
        )
        async with httpx.AsyncClient(transport=transport) as client:
            backend._async_client = client
            return [
                item
                async for item in backend._resolve_streaming(
                    GenerationRequest(request_id="req-1"),
                    RequestInfo(request_id="req-1"),
                    ChatCompletionsWithReasoningResponseHandler(),
                    GenerationRequestArguments(stream=True),
                    {"method": "POST", "url": "http://test/v1/chat/completions"},
                )
            ]

    outputs = asyncio.run(run())
    response, info = outputs[-1]
    assert response.text == text + "尾"
    assert response.reasoning_text == (text if field == "reasoning_content" else None)
    assert response.response_id == "response-1"
    assert response.input_metrics.text_tokens == 9
    assert response.output_metrics.text_tokens == 5
    assert info.timings.token_iterations == 2
    assert info.timings.request_iterations == 4


@pytest.mark.parametrize("ending", [b"\n", b"\r\n", b"\r"])
def test_aiter_lines_yields_before_reading_the_next_chunk(ending):
    class FirstLineStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"data: ready" + ending
            raise AssertionError("Read ahead of a complete line")

    async def run():
        backend = OpenAIHTTPErrorDetailBackend(
            OpenAIHTTPErrorDetailBackendArgs(target="http://test", model="m")
        )
        async with aclosing(httpx.Response(200, stream=FirstLineStream())) as response:
            lines = backend._aiter_lines(response)
            try:
                assert await anext(lines) == "data: ready"
            finally:
                await lines.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("ending", [b"\n", b"\r\n", b"\r"])
def test_aiter_lines_preserves_the_final_unterminated_line(ending):
    body = ending + b" \t" + ending + b"data: first" + ending + b"data: last"

    async def run():
        backend = OpenAIHTTPErrorDetailBackend(
            OpenAIHTTPErrorDetailBackendArgs(target="http://test", model="m")
        )
        async with aclosing(
            httpx.Response(200, stream=_ChunkedStream(body, 1))
        ) as response:
            return [line async for line in backend._aiter_lines(response)]

    assert asyncio.run(run()) == ["data: first", "data: last"]


def test_streaming_malformed_json_is_not_ignored():
    body = b'data: {"choices":[{"delta":{"content":"unfinished'

    async def run():
        backend = OpenAIHTTPErrorDetailBackend(
            OpenAIHTTPErrorDetailBackendArgs(target="http://test", model="m")
        )
        handler = ChatCompletionsWithReasoningResponseHandler()
        async with aclosing(
            httpx.Response(200, stream=_ChunkedStream(body, 1))
        ) as response:
            async for line in backend._aiter_lines(response):
                handler.add_streaming_line(line)

    with pytest.raises(json.JSONDecodeError):
        asyncio.run(run())


def test_aiter_lines_propagates_transport_errors():
    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"data: partial"
            raise httpx.ReadError("Connection lost")

    async def run():
        backend = OpenAIHTTPErrorDetailBackend(
            OpenAIHTTPErrorDetailBackendArgs(target="http://test", model="m")
        )
        async with aclosing(httpx.Response(200, stream=BrokenStream())) as response:
            return [line async for line in backend._aiter_lines(response)]

    with pytest.raises(httpx.ReadError, match="Connection lost"):
        asyncio.run(run())
