"""Request ID + timing middleware (pure ASGI, so it also wraps error responses)."""

import time
import uuid

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = dict(scope.get("headers", [])).get(b"x-request-id", b"").decode("latin-1")
        # Accept a caller-provided ID only if it looks sane (no header injection / huge values).
        request_id = (
            incoming if incoming and len(incoming) <= 128 and incoming.isprintable() else ""
        )
        request_id = request_id or uuid.uuid4().hex

        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        state["start_time"] = time.perf_counter()

        async def send_with_header(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers[REQUEST_ID_HEADER] = request_id
            await send(message)

        await self.app(scope, receive, send_with_header)
