"""Small JSON-over-HTTP client for benchmark adapters."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Optional, Sequence, Tuple


class HttpError(RuntimeError):
    pass


class JsonClient:
    def __init__(self, base_url: str, timeout_secs: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_secs = timeout_secs

    def request(
        self,
        method: str,
        path: str,
        payload: Optional[dict] = None,
        expected: Sequence[int] = (200,),
        decode_json: bool = True,
    ) -> dict:
        body = None
        headers = {}

        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        url = f"{self.base_url}{path}"
        request = urllib.request.Request(url, data=body, headers=headers, method=method)

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_secs) as response:
                data = response.read()
                status = response.status
        except urllib.error.HTTPError as error:
            status = error.code
            data = error.read()

        if status not in expected:
            text = data.decode("utf-8", errors="replace")
            raise HttpError(f"{method} {path} returned {status}: {text}")

        if not data or not decode_json:
            return {}

        return json.loads(data.decode("utf-8"))

    def wait_until_ready(
        self,
        path: str,
        attempts: int,
        sleep_secs: float,
        expected: Sequence[int] = (200,),
    ) -> None:
        last_error = None

        for _ in range(attempts):
            try:
                self.request("GET", path, expected=expected, decode_json=False)
                return
            except Exception as error:
                last_error = error
                time.sleep(sleep_secs)

        raise HttpError(f"{self.base_url}{path} did not become ready: {last_error}")


def timed_call(callable_obj) -> Tuple[dict, float]:
    start = time.perf_counter_ns()
    result = callable_obj()
    latency_micros = (time.perf_counter_ns() - start) / 1000.0
    return result, latency_micros
