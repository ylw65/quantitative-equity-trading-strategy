"""Polite, cached access to public SEC reference and filing endpoints."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DEFAULT_CONTACT = "DSA5205-Proj2 research data audit"


class RateLimiter:
    """Thread-safe fixed-rate limiter shared by all worker threads."""

    def __init__(self, requests_per_second: float) -> None:
        self.minimum_interval = 1.0 / max(requests_per_second, 0.1)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self.minimum_interval
        if delay:
            time.sleep(delay)


class SecClient:
    """SEC client with a descriptive user agent, retries, and rate limiting."""

    def __init__(
        self,
        requests_per_second: float = 5.0,
        timeout_seconds: float = 30.0,
        retries: int = 3,
    ) -> None:
        self.user_agent = os.environ.get("SEC_USER_AGENT", DEFAULT_CONTACT)
        if "@" not in self.user_agent:
            self.user_agent = f"{self.user_agent} (academic project; no credential)"
        self.timeout_seconds = timeout_seconds
        self.limiter = RateLimiter(requests_per_second)
        self.retries = retries
        self._local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            retry = Retry(
                total=self.retries,
                connect=self.retries,
                read=self.retries,
                backoff_factor=1.0,
                status_forcelist=(403, 429, 500, 502, 503, 504),
                allowed_methods=frozenset({"GET", "HEAD"}),
            )
            session.mount("https://", HTTPAdapter(max_retries=retry))
            session.headers.update(
                {
                    "User-Agent": self.user_agent,
                    "Accept-Encoding": "gzip, deflate",
                }
            )
            self._local.session = session
        return session

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        destination: str | Path | None = None,
    ) -> requests.Response:
        """GET a URL and optionally atomically write the response body."""
        self.limiter.wait()
        response = self._session().get(
            url,
            headers=headers,
            timeout=self.timeout_seconds,
            allow_redirects=True,
        )
        response.raise_for_status()
        if destination is None:
            return response

        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        with temporary.open("wb") as handle:
            handle.write(response.content)
        os.replace(temporary, target)
        return response

    def get_filing_probe(self, url: str, byte_count: int) -> str:
        """Read only the beginning of a compressed SEC filing submission."""
        response = self.get(
            url,
            headers={
                "Range": f"bytes=0-{max(byte_count - 1, 0)}",
                "Accept-Encoding": "gzip",
            },
        )
        return response.content.decode("utf-8", errors="replace")
