"""Minimal GitHub API client built on the standard library only.

Supports the issue search endpoint used for Phase 1 ingestion. Network access
is encapsulated here so the rest of the toolkit can be unit-tested with a fake
client (see ``tests/``).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterator, Protocol


API_ROOT = "https://api.github.com"
USER_AGENT = "intel-cpu-triage/0.1"


class SearchClient(Protocol):
    """Protocol implemented by both the real and fake GitHub clients."""

    def search_issues(self, query: str, per_page: int = 100) -> Iterator[dict[str, Any]]:
        ...


class GitHubClient:
    """Tiny REST client. Uses a token from the caller for authentication."""

    def __init__(
        self,
        token: str | None = None,
        api_root: str = API_ROOT,
        max_retries: int = 3,
        sleep: float = 2.0,
    ) -> None:
        self.token = token
        self.api_root = api_root.rstrip("/")
        self.max_retries = max_retries
        self.sleep = sleep

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": USER_AGENT,
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        return headers

    def _get(self, url: str) -> dict[str, Any]:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(url, headers=self._headers())
            try:
                with urllib.request.urlopen(req) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                # Respect secondary rate limits / transient 5xx with backoff.
                if exc.code in (403, 429, 500, 502, 503) and attempt + 1 < self.max_retries:
                    time.sleep(self.sleep * (attempt + 1))
                    last_err = exc
                    continue
                raise
            except urllib.error.URLError as exc:
                last_err = exc
                time.sleep(self.sleep * (attempt + 1))
        assert last_err is not None
        raise last_err

    def search_issues(
        self, query: str, per_page: int = 100
    ) -> Iterator[dict[str, Any]]:
        """Yield issue items matching a GitHub search query, paginating fully.

        The search API caps results at 1000 items per query; callers should
        keep queries narrow (label + updated watermark) to stay under it.
        """
        page = 1
        while True:
            params = urllib.parse.urlencode(
                {"q": query, "per_page": per_page, "page": page}
            )
            url = f"{self.api_root}/search/issues?{params}"
            data = self._get(url)
            items = data.get("items", [])
            if not items:
                break
            for item in items:
                # Exclude PRs; the search/issues endpoint returns both.
                if "pull_request" in item:
                    continue
                yield item
            if len(items) < per_page:
                break
            page += 1
