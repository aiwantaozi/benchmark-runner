"""
The core flow in this module is adapted from GuideLLM's OpenAI backend:
- Source: guidellm v0.8.1, ``src/guidellm/backends/openai/http.py``
  (https://github.com/vllm-project/guidellm/tree/v0.8.1)
- Reference implementation: `guidellm/backends/openai/http.py::OpenAIHTTPBackend`

The pinned parent class supplies ``_trace_request_start`` for both HTTP paths.

Adjustments in this module:
1) Register a new backend type `openai_http_error_detail` (args + implementation).
2) Keep the request/stream processing flow aligned with GuideLLM 0.8.1 (reuse the
   base `_prepare_resolve_request`, which formats the request, filters ``None`` via
   ``deep_filter`` and builds the httpx kwargs).
3) For HTTP error responses, read the body early (`aread()`) before the stream
   closes, extract OpenAI-style error fields (`error.message/type/code`) with a
   safe fallback, and re-raise as a concise `RuntimeError` so the scheduler's
   `RequestInfo.error` keeps the detail. This replaces the plain
   ``response.raise_for_status()`` used by the stock backend.
4) Support selecting a custom request handler (e.g. the reasoning-aware chat
   handler) via a ``request_handlers`` field keyed by API path -> registered
   handler NAME. The stock ``OpenAIHTTPBackend`` in 0.7.3 does NOT surface handler
   overrides: ``_prepare_resolve_request`` calls
   ``OpenAIRequestHandlerFactory.create(request_format)`` with no override
   argument. Rather than fork that whole method, we let the base build its handler
   and then SWAP IN the configured one (see ``_prepare_resolve_request`` below).
   That is sound only because ``format()`` on these handlers writes nothing to
   ``self`` — it returns a fresh ``GenerationRequestArguments`` — so the arguments
   the base already built stay valid for a different instance of a sibling class.
   A handler that accumulated per-request state in ``format()`` would need the
   override applied BEFORE formatting instead.
5) Record per-chunk inter-token latency (ITL) samples, which the stock loop
   discards: it keeps only the first and last token timestamps, so the individual
   gaps are unrecoverable afterwards (``iter_tokens_timings`` reconstructs them
   with ``np.linspace``, i.e. assumes they were evenly spaced — which is what
   hides a decode stall). The samples ride back to the main process on
   ``RequestInfo.timings`` under ``ITL_TIMINGS_FIELD``; ``output_dual_json``
   aggregates them into a distribution. See that module for the aggregation and
   for why this needs no guidellm changes.
6) Split streaming responses only on SSE line endings (CR, LF, CRLF), preserving
   Unicode line separators inside JSON strings.

guidellm 0.7.x vs 0.6.0:
- Backend config is stored on ``self._args`` (an ``OpenAIHTTPBackendArgs``), not as
  plain attributes. The request format lives at ``self._args.request_format`` (an
  API path like ``/v1/chat/completions``), and routes at ``self._args.api_routes``.
- The handler lifecycle is ``format(...)`` -> ``add_streaming_line(line)`` ->
  ``compile_streaming(request, arguments)`` / ``compile_non_streaming(request,
  arguments, data)`` (``arguments`` is now a required positional arg on compile).

guidellm 0.7.3 note:
- The lifecycle gained a final ``post_validation(response)`` step, which the text
  handlers use to reject a compiled response carrying no text, no tool calls and no
  output tokens. Both resolve paths below call it in the same position as the stock
  backend, so an empty payload lands in ``RequestInfo.error`` instead of being
  counted as a successful request with zero output.

Backend kwargs shape gpustack must send (spec.backend):
    {
        "kind": "openai_http_error_detail",
        "target": "http://host:port",
        "request_handlers": {
            "/v1/chat/completions": "chat_completions_with_reasoning"
        }
    }
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any, Literal

import httpx
from pydantic import Field

from guidellm.backends.backend import Backend, BackendArgs
from guidellm.backends.openai.http import OpenAIHTTPBackend, OpenAIHTTPBackendArgs
from guidellm.backends.openai.request_handlers import (
    OpenAIRequestHandler,
    OpenAIRequestHandlerFactory,
)
from guidellm.schemas import (
    GenerationRequest,
    GenerationRequestArguments,
    GenerationResponse,
    RequestInfo,
)

ERROR_DETAIL_BACKEND_TYPE = "openai_http_error_detail"
MAX_ERROR_DETAIL_LENGTH = 2048

# Extra field stamped onto ``RequestInfo.timings`` carrying this request's ITL
# samples in milliseconds. NOT a declared guidellm field -- ``RequestTimings``
# extends ``StandardBaseDict`` (``extra="allow"``), which is what makes this
# possible without forking guidellm. ``RequestInfo`` itself is ``extra="ignore"``,
# so the samples MUST hang off ``timings``, not off the info object.
#
# The name is a constant rather than a literal because the aggregation side
# (``output_dual_json``) reads it back; a typo on either side would silently
# produce an empty distribution instead of failing.
ITL_TIMINGS_FIELD = "inter_token_latencies_ms"
ITL_TIMESTAMPS_FIELD = "inter_token_latency_timestamps"

__all__ = [
    "ERROR_DETAIL_BACKEND_TYPE",
    "ITL_TIMESTAMPS_FIELD",
    "ITL_TIMINGS_FIELD",
    "MAX_ERROR_DETAIL_LENGTH",
    "OpenAIHTTPErrorDetailBackend",
    "OpenAIHTTPErrorDetailBackendArgs",
    "format_http_error_response",
    "format_http_status_error_async",
    "format_http_status_error",
]


def _truncate(text: str, max_length: int = MAX_ERROR_DETAIL_LENGTH) -> str:
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 3]}..."


def _stringify(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def format_http_status_error(exc: httpx.HTTPStatusError) -> str:
    response = exc.response
    if response is None:
        return _truncate(f"HTTP request failed: {exc}")

    status_code = response.status_code
    message: str | None = None
    error_type: str | None = None
    error_code: str | None = None
    body_fallback: str | None = None

    try:
        payload = response.json()
        if isinstance(payload, dict):
            error_obj = payload.get("error")
            if isinstance(error_obj, dict):
                message = _stringify(error_obj.get("message"))
                error_type = _stringify(error_obj.get("type"))
                error_code = _stringify(error_obj.get("code"))
            else:
                message = _stringify(payload.get("message"))
                error_type = _stringify(payload.get("type"))
                error_code = _stringify(payload.get("code"))

        if message is None:
            body_fallback = _truncate(json.dumps(payload, ensure_ascii=False))
    except Exception:
        try:
            body_fallback = _truncate(response.text)
        except Exception:
            body_fallback = _stringify(exc)

    detail = message or body_fallback or _stringify(exc) or "unknown error"
    suffix_parts = []
    if error_type:
        suffix_parts.append(f"type={error_type}")
    if error_code:
        suffix_parts.append(f"code={error_code}")
    suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""
    return _truncate(f"HTTP {status_code}: {detail}{suffix}")


async def format_http_error_response(
    response: httpx.Response, fallback: str | None = None
) -> str:
    """
    Format HTTP error detail directly from a response object.

    This function is safe for streaming responses and attempts to read the body
    before parsing OpenAI-style `error.message/type/code`.
    """
    try:
        await response.aread()
    except Exception:
        pass

    payload: Any = None
    message: str | None = None
    error_type: str | None = None
    error_code: str | None = None
    body_fallback: str | None = None

    try:
        payload = response.json()
        if isinstance(payload, dict):
            error_obj = payload.get("error")
            if isinstance(error_obj, dict):
                message = _stringify(error_obj.get("message"))
                error_type = _stringify(error_obj.get("type"))
                error_code = _stringify(error_obj.get("code"))
            else:
                message = _stringify(payload.get("message"))
                error_type = _stringify(payload.get("type"))
                error_code = _stringify(payload.get("code"))

        if message is None and payload is not None:
            body_fallback = _truncate(json.dumps(payload, ensure_ascii=False))
    except Exception:
        try:
            body_fallback = _truncate(response.text)
        except Exception:
            body_fallback = None

    detail = message or body_fallback or _stringify(fallback) or "unknown error"
    suffix_parts = []
    if error_type:
        suffix_parts.append(f"type={error_type}")
    if error_code:
        suffix_parts.append(f"code={error_code}")
    suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""
    return _truncate(f"HTTP {response.status_code}: {detail}{suffix}")


async def format_http_status_error_async(exc: httpx.HTTPStatusError) -> str:
    """
    Async-safe formatter for streaming responses.

    When httpx raises from a stream context, response content may not be read yet.
    We call `aread()` first so JSON/text extraction is available.
    """
    response = exc.response
    if response is None:
        return format_http_status_error(exc)
    return await format_http_error_response(response, fallback=str(exc))


@BackendArgs.register(ERROR_DETAIL_BACKEND_TYPE)
class OpenAIHTTPErrorDetailBackendArgs(OpenAIHTTPBackendArgs):
    """Args for the error-detail backend.

    Extends the stock OpenAI HTTP backend args with a ``request_handlers`` mapping
    (API path -> registered handler NAME) since 0.7.3's base args do not expose a
    handler-override field.
    """

    kind: Literal[ERROR_DETAIL_BACKEND_TYPE] = Field(  # type: ignore[assignment]
        default=ERROR_DETAIL_BACKEND_TYPE,
        description="Type identifier for the error-detail backend configuration.",
    )
    request_handlers: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional override of request handlers, keyed by API path "
            "(e.g. '/v1/chat/completions') mapped to a handler NAME registered on "
            "OpenAIRequestHandlerFactory (e.g. 'chat_completions_with_reasoning')."
        ),
        examples=[{"/v1/chat/completions": "chat_completions_with_reasoning"}],
    )


@Backend.register(ERROR_DETAIL_BACKEND_TYPE)
class OpenAIHTTPErrorDetailBackend(OpenAIHTTPBackend):
    """
    OpenAI HTTP backend that enriches HTTP errors with response-body details and
    supports selecting a custom request handler via ``request_handlers``.
    """

    def _resolve_handler_overrides(
        self,
    ) -> dict[str, type[OpenAIRequestHandler]] | None:
        """Resolve the configured ``request_handlers`` names into handler classes.

        :return: Mapping of API path -> handler class, or None when unset.
        """
        mapping: dict[str, str] = getattr(self._args, "request_handlers", None) or {}
        if not mapping:
            return None
        overrides: dict[str, type[OpenAIRequestHandler]] = {}
        for path, name in mapping.items():
            handler_cls = OpenAIRequestHandlerFactory.get_registered_object(name)
            if handler_cls is None:
                available = list(OpenAIRequestHandlerFactory.registry or {})
                raise ValueError(
                    f"Unknown request handler '{name}' for path '{path}'. "
                    f"Available handlers: {available}"
                )
            overrides[path] = handler_cls
        return overrides

    async def _prepare_resolve_request(
        self,
        request: GenerationRequest,
        history: (
            list[tuple[GenerationRequest, GenerationResponse | None]] | None
        ) = None,
    ) -> tuple[OpenAIRequestHandler, GenerationRequestArguments, dict[str, Any]]:
        """Reuse the base preparation, then swap in the overridden handler.

        The base method formats the request, applies ``deep_filter`` and builds the
        httpx kwargs. Replacing the handler AFTERWARDS is safe here because
        ``format()`` on the text/chat handlers is effectively pure: it builds and
        returns a ``GenerationRequestArguments`` and writes no instance state (the
        streaming accumulators are only touched by ``add_streaming_line`` and the
        ``compile_*`` methods). So the ``arguments`` the base produced remain valid
        for the substitute instance, which then processes and compiles the response.

        This does NOT hold for a handler that stashes per-request state during
        ``format()``; such an override has to be installed before formatting.
        """
        request_handler, arguments, request_kwargs = (
            await super()._prepare_resolve_request(request, history)
        )
        overrides = self._resolve_handler_overrides()
        if overrides and self._args.request_format in overrides:
            request_handler = overrides[self._args.request_format]()
        return request_handler, arguments, request_kwargs

    async def _resolve_non_streaming(
        self,
        request: GenerationRequest,
        request_info: RequestInfo,
        request_handler: OpenAIRequestHandler,
        arguments: GenerationRequestArguments,
        request_kwargs: dict[str, Any],
    ) -> AsyncIterator[tuple[GenerationResponse | None, RequestInfo]]:
        """Non-streaming path: same as the base, but with error-body detail.

        Replaces ``response.raise_for_status()`` with an early body read + concise
        RuntimeError so the scheduler keeps the server's error detail.
        """
        if self._async_client is None:
            raise RuntimeError("Backend not started up for process.")

        request_info.timings.request_start = time.time()
        response = await self._async_client.request(
            **request_kwargs,
            extensions={"trace": self._trace_request_start(request_info)},
        )
        request_info.timings.request_end = time.time()
        if response.is_error:
            raise RuntimeError(
                await format_http_error_response(response, fallback="request failed")
            )
        data = response.json()
        gen_response = request_handler.compile_non_streaming(request, arguments, data)
        request_handler.post_validation(gen_response)
        yield gen_response, request_info
        self._check_tool_call_expectations(request, gen_response)

    async def _aiter_lines(self, stream: httpx.Response) -> AsyncIterator[str]:
        """Read SSE lines without treating Unicode text as protocol delimiters.

        HTTPX's generic line iterator also splits on U+0085, U+2028 and U+2029,
        which are valid inside JSON strings. Decode UTF-8 incrementally through
        ``aiter_text`` and keep fragments until a CR or LF ends the line.
        """
        fragments: list[str] = []
        skip_lf = False
        async for text in stream.aiter_text():
            if skip_lf:
                text = text.removeprefix("\n")
            skip_lf = text.endswith("\r")
            segments = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            fragments.append(segments[0])
            for segment in segments[1:]:
                line = "".join(fragments)
                if line.strip():
                    yield line
                fragments = [segment]

        line = "".join(fragments)
        if line.strip():
            yield line

    async def _resolve_streaming(
        self,
        request: GenerationRequest,
        request_info: RequestInfo,
        request_handler: OpenAIRequestHandler,
        arguments: GenerationRequestArguments,
        request_kwargs: dict[str, Any],
    ) -> AsyncIterator[tuple[GenerationResponse | None, RequestInfo]]:
        """Streaming path: mirrors the base loop (incl. TTFOT tracking) but reads
        the error body before raising so the detail is preserved, and records the
        per-chunk ITL samples the base loop throws away."""
        if self._async_client is None:
            raise RuntimeError("Backend not started up for process.")

        # ITL samples for this request, in milliseconds.
        #
        # guidellm keeps only the FIRST and LAST token timestamps per request
        # (``first_token_iteration`` / ``last_token_iteration``); the individual
        # gaps never survive collection -- ``iter_tokens_timings`` reconstructs
        # them after the fact with ``np.linspace``, i.e. it assumes they were
        # evenly spaced. That assumption is exactly what hides a decode stall, so
        # the gaps are recorded here, where the timestamps actually exist.
        #
        # Same measurement point and formula as vLLM's ITL: one sample per gap
        # between consecutive streamed outputs. A chunk carrying several tokens
        # (speculative decoding, reasoning deltas) still yields ONE sample -- no
        # zero-duration gaps are invented for its bundled tokens, matching
        # vLLM's "the tokens in the same output do not create additional ITL
        # samples". The first chunk has no predecessor and contributes nothing:
        # that interval is TTFT.
        itl_gaps_ms: list[float] = []
        itl_timestamps: list[float] = []

        try:
            request_info.timings.request_start = time.time()

            async with self._async_client.stream(
                **request_kwargs,
                extensions={"trace": self._trace_request_start(request_info)},
            ) as stream:
                if stream.is_error:
                    request_info.timings.request_end = time.time()
                    raise RuntimeError(
                        await format_http_error_response(
                            stream, fallback="request failed"
                        )
                    )
                end_reached = False

                async for chunk in self._aiter_lines(stream):
                    if stream.is_error:
                        request_info.timings.request_end = time.time()
                        raise RuntimeError(
                            await format_http_error_response(
                                stream, fallback="request failed"
                            )
                        )
                    iter_time = time.time()

                    if request_info.timings.first_request_iteration is None:
                        request_info.timings.first_request_iteration = iter_time
                    request_info.timings.last_request_iteration = iter_time
                    request_info.timings.request_iterations += 1

                    iterations = request_handler.add_streaming_line(chunk)
                    if iterations is None or iterations <= 0 or end_reached:
                        end_reached = end_reached or iterations is None
                        if end_reached:
                            break
                        continue

                    if request_info.timings.first_token_iteration is None:
                        request_info.timings.first_token_iteration = iter_time
                        request_info.timings.token_iterations = 0
                        yield None, request_info

                    if (
                        request_info.timings.first_output_token_iteration is None
                        and request_handler.last_iteration_had_content
                    ):
                        request_info.timings.first_output_token_iteration = iter_time

                    # Read the PREVIOUS token chunk's timestamp before it is
                    # overwritten below: None means this is the first one.
                    prev_iter = request_info.timings.last_token_iteration
                    if prev_iter is not None:
                        itl_gaps_ms.append((iter_time - prev_iter) * 1000.0)
                        itl_timestamps.append(iter_time)

                    request_info.timings.last_token_iteration = iter_time
                    request_info.timings.token_iterations += iterations

            request_info.timings.request_end = time.time()
            setattr(request_info.timings, ITL_TIMINGS_FIELD, itl_gaps_ms)
            setattr(request_info.timings, ITL_TIMESTAMPS_FIELD, itl_timestamps)
            gen_response = request_handler.compile_streaming(request, arguments)
            request_handler.post_validation(gen_response)
            self._check_tool_call_expectations(request, gen_response)
            yield gen_response, request_info
        except asyncio.CancelledError as err:
            # Stamp what was measured before the cancellation too. A cancelled
            # request is incomplete and does not feed the successful aggregate,
            # but dropping the samples here would make the partial record lie
            # about having none.
            setattr(request_info.timings, ITL_TIMINGS_FIELD, itl_gaps_ms)
            setattr(request_info.timings, ITL_TIMESTAMPS_FIELD, itl_timestamps)
            yield request_handler.compile_streaming(request, arguments), request_info
            raise err
