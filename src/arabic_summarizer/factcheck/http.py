"""Fixed-endpoint JSON transport: bounded cache, retries and process-wide throttling."""
import json
import threading
import time
from collections import OrderedDict

ENDPOINTS = {
    "google": "https://factchecktools.googleapis.com/v1alpha1/claims:search",
    "gdelt": "https://api.gdeltproject.org/api/v2/doc/doc",
}


class SourceError(RuntimeError):
    """Safe diagnostic code; never retains URLs, credentials or response bodies."""


class JsonClient:
    _lock = threading.Lock()
    _last = {}

    def __init__(self, source, session=None, clock=time.monotonic, sleep=time.sleep):
        self.source = source
        self.session = session
        self.clock, self.sleep = clock, sleep
        self.cache = OrderedDict()

    def get(self, params, deadline):
        # Keys never contain the API credential.
        key = tuple(sorted((k, str(v)) for k, v in params.items() if k != "key"))
        cached = self.cache.get(key)
        if cached and self.clock() - cached[0] < 900:
            return cached[1]
        if self.session is None:
            import requests
            self.session = requests.Session()
        for attempt in range(3):
            remaining = deadline - self.clock()
            if remaining < 1:
                raise SourceError("time_budget")
            with self._lock:
                now = self.clock()
                interval = 6 if self.source == "gdelt" else 2
                if now - self._last.get(self.source, -1000) < interval:
                    raise SourceError("local_rate_limit")
                self._last[self.source] = now
            response = None
            try:
                response = self.session.get(
                    ENDPOINTS[self.source], params=params, timeout=min(6, remaining),
                    allow_redirects=False, stream=True,
                )
                status = response.status_code
                if status in (401, 403):
                    raise SourceError("authentication_or_access")
                if status == 429 or 500 <= status < 600:
                    if attempt == 2:
                        raise SourceError("quota" if status == 429 else "service_unavailable")
                    retry = response.headers.get("Retry-After", "0")
                    delay = max(6 if self.source == "gdelt" else 2, 2 ** attempt,
                                float(retry) if retry.isdigit() else 0)
                    if delay > 10 or self.clock() + delay + 1 >= deadline:
                        raise SourceError("quota" if status == 429 else "time_budget")
                    self.sleep(delay)
                    continue
                if status != 200:
                    raise SourceError("http_error")
                if "json" not in response.headers.get("Content-Type", "").lower():
                    raise SourceError("non_json_response")
                body = bytearray()
                for chunk in response.iter_content(16_384):
                    body.extend(chunk)
                    if len(body) > 1_000_000 or self.clock() >= deadline:
                        raise SourceError("response_limit")
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise SourceError("malformed_payload")
                self.cache[key] = (self.clock(), payload)
                if len(self.cache) > 128:
                    self.cache.popitem(last=False)
                return payload
            except SourceError:
                raise
            except Exception:
                # Avoid leaking requests exceptions containing the API key.
                if attempt == 2 or self.clock() + 7 >= deadline:
                    raise SourceError("network_or_payload_error") from None
                self.sleep(6 if self.source == "gdelt" else 2)
            finally:
                if response is not None:
                    response.close()
        raise SourceError("service_unavailable")
