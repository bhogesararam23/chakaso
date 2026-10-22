"""The transport: a stdlib HTTP server in front of :class:`~chakaso.serve.app.ServingApplication`.

No framework, deliberately. The brief suggested a minimal well-supported one, and the honest measurement
against it is this: the surface is four endpoints, one model and a request schema small enough to validate
by hand, so a dependency would buy a WSGI/ASGI layer this project does not use, cost a fourth install tier in
CI, and add a supply chain to a service whose whole purpose is running a model on the user's own machine
without a third party. :mod:`http.server` plus :mod:`socketserver` is in the standard library, is threaded,
and is enough.

What the transport adds on top of the application layer, and why each is here rather than in the app:

* **Loopback by default.** ``127.0.0.1`` unless an operator names an address, and a startup line that prints
  what was bound. Authentication is not offered, so a non-loopback bind is a decision the operator states.
* **Bounded concurrency.** A counter caps simultaneous generations, because each one is CPU-bound on this
  project's hardware and unbounded means the first request starves the rest. Over the limit is a 429 with
  ``Retry-After``, not a queue that grows.
* **Disconnect detection.** A streaming response writes chunk by chunk; a broken pipe aborts the
  generation through the application's own signal.
* **Clean shutdown.** ``stop()`` joins the serving thread, and a Ctrl-C at the terminal goes through the same
  path.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from chakaso.core.text import safe_for_terminal
from chakaso.serve.app import ServiceResponse, ServingApplication

__all__ = ["DEFAULT_HOST", "DEFAULT_PORT", "MAX_BODY_BYTES", "ServingService", "serve"]

#: Loopback, not ``0.0.0.0``: a local inference endpoint on a shared machine is an open door to someone
#: else's prompts and someone else's CPU.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
#: A hard read cap, so a hostile Content-Length cannot make the server buffer without limit.
MAX_BODY_BYTES = 1_048_576
#: Generations served at once. Beyond this a caller is told to come back, rather than joining a queue that
#: makes every answer slower with no explanation.
DEFAULT_MAX_CONCURRENT = 2


class _Handler(BaseHTTPRequestHandler):
    """The thinnest possible adapter from a request line to the application and back."""

    application: ServingApplication
    limiter: threading.Semaphore
    max_body_bytes: int = MAX_BODY_BYTES

    protocol_version = "HTTP/1.1"
    server_version = "chakaso/0.1"

    def do_GET(self) -> None:
        self._respond("GET", self.path, None)

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length") or 0)
        if length > self.max_body_bytes:
            self._send(
                ServiceResponse.json(
                    {
                        "error": {
                            "message": "request body is too large",
                            "type": "invalid_request_error",
                            "param": "body",
                        }
                    },
                    status=413,
                )
            )
            return
        body = self.rfile.read(length) if length > 0 else b""
        self._respond("POST", self.path, body)

    def do_OPTIONS(self) -> None:
        self._respond("OPTIONS", self.path, None)

    def _respond(self, method: str, path: str, body: bytes | None) -> None:
        if method == "POST" and not self.limiter.acquire(blocking=False):
            busy = ServiceResponse.json(
                {
                    "error": {
                        "message": "this service is serving its maximum number of generations at once",
                        "type": "service_unavailable",
                        "param": None,
                    }
                },
                status=429,
            )
            self._send(busy)
            return
        try:
            self._send(self.application.handle(method, path, body))
        finally:
            if method == "POST":
                self.limiter.release()

    def _send(self, response: ServiceResponse) -> None:
        if response.stream is not None:
            self.send_response(response.status)
            for key, value in response.headers:
                self.send_header(key, value)
            self.send_header("transfer-encoding", "chunked")
            self.end_headers()
            try:
                for event in response.stream:
                    payload = event.encode("utf-8")
                    self.wfile.write(f"{len(payload):x}\r\n".encode("ascii"))
                    self.wfile.write(payload)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            except (ConnectionError, BrokenPipeError):
                # The caller hung up. Cancelling the generation is the point: without it the model keeps
                # producing an answer nobody is reading, on a machine with one CPU pool. Nothing is logged
                # as an error, because from the server's side nothing failed.
                #
                # The family rather than one class because the platform does not commit to one: measured on
                # this machine, the same hang-up produced WSAECONNABORTED (``ConnectionAbortedError``) on one
                # run and WSAECONNRESET (``ConnectionResetError``) on another, and a POSIX pipe gives
                # ``BrokenPipeError``. Catching only the one that happened to appear is how this became a
                # traceback in the server log and an uncancelled generation.
                if response.abort is not None:
                    response.abort()
                return
            return
        body = response.body or b""
        self.send_response(response.status)
        for key, value in response.headers:
            self.send_header(key, value)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        # Route the access line through the terminal-safety layer: a request path can carry hostile bytes,
        # and a server that prints them to a terminal unescaped is a control-sequence injection into whoever
        # is watching the logs.
        line = f"{self.address_string()} {format % args}"
        print(safe_for_terminal(line))  # noqa: T201 - the operator's own terminal is this server's log


class ServingService:
    """A startable, stoppable local server around one application."""

    def __init__(
        self,
        application: ServingApplication,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    ) -> None:
        if max_concurrent < 1:
            message = f"max_concurrent must be >= 1, got {max_concurrent}"
            raise ValueError(message)
        handler = type(
            "BoundHandler",
            (_Handler,),
            {
                "application": application,
                "limiter": threading.Semaphore(max_concurrent),
            },
        )
        self._server = ThreadingHTTPServer((host, port), handler)
        self._server.daemon_threads = True
        self._thread: threading.Thread | None = None
        # ``server_address`` is typed as an address of either flavour, and a TCP server's is a host string;
        # the annotation says so, because the URL below interpolates it and a bytes host would render as
        # "b'127.0.0.1'" instead of failing.
        self.host = str(self._server.server_address[0])
        self.port = int(self._server.server_address[1])

    @property
    def base_url(self) -> str:
        """What a client is pointed at, with the port actually bound."""
        return f"http://{self.host}:{self.port}"

    @property
    def loopback_only(self) -> bool:
        """Whether this instance is reachable only from its own machine."""
        return self.host in ("127.0.0.1", "::1", "localhost")

    def start(self) -> ServingService:
        """Serve in a background thread and return self, for tests and for a script that keeps working."""
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop accepting, finish the current request, and join the serving thread.

        Safe on a service that never started, and safe to call twice. The guard is not decoration:
        :meth:`http.server.HTTPServer.shutdown` waits for the serve loop to acknowledge the stop, and that
        loop only exists while ``serve_forever()`` is running, so asking an unstarted or already-stopped
        server to shut down waits on an acknowledgement that can never arrive.
        """
        thread = self._thread
        if thread is not None:
            self._thread = None
            if thread.is_alive():
                self._server.shutdown()
            thread.join(timeout=5.0)
        self._server.server_close()

    def serve_forever(self) -> None:
        """Block until interrupted, then shut down cleanly."""
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            # The loop has already returned by the time this runs, so there is nothing to shut down; the
            # listening socket is the only resource left held.
            self._server.server_close()


def serve(
    path: str,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    max_input_characters: int | None = None,
) -> ServingService:
    """Load ``path`` and build a server for it, without starting it.

    Raises:
        ServingError: the model could not be loaded or is not a trained model.
    """
    from chakaso.serve.app import DEFAULT_MAX_INPUT_CHARACTERS, ServingApplication

    application = ServingApplication.for_path(
        path,
        max_input_characters=max_input_characters or DEFAULT_MAX_INPUT_CHARACTERS,
    )
    return ServingService(application, host=host, port=port, max_concurrent=max_concurrent)
