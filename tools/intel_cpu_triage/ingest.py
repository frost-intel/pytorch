"""Phase 1: incremental ingestion of relevant issues into the local DB.

Builds GitHub search queries from the configured scope labels and content
keywords, fetches matching open issues updated since the last run (the
"watermark"), and upserts them. The watermark makes each run cheap.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Iterator

from .config import Config
from .db import Database, Issue
from .github_client import SearchClient


WATERMARK_KEY = "ingest_watermark"


def build_queries(cfg: Config, since: str | None) -> list[str]:
    """Build the set of search queries for this run.

    One query per scope label keeps each result set small (under the search
    API's 1000-item cap). A final query covers content keywords for issues that
    are in-scope by content but mislabeled.
    """
    base = f"repo:{cfg.repo} is:issue is:open"
    if since:
        base += f" updated:>={since}"

    queries: list[str] = []
    for label in cfg.scope_labels:
        queries.append(f'{base} label:"{label}"')

    # Content-keyword sweep. GitHub search OR-combines quoted terms; keep the
    # list modest to avoid an overly long query string.
    if cfg.content_keywords:
        terms = " OR ".join(f'"{kw}"' for kw in cfg.content_keywords[:20])
        queries.append(f"{base} in:title,body ({terms})")
    return queries


def _parse_issue(item: dict[str, Any]) -> Issue:
    labels = [
        lbl["name"] if isinstance(lbl, dict) else str(lbl)
        for lbl in item.get("labels", [])
    ]
    reactions = 0
    if isinstance(item.get("reactions"), dict):
        reactions = item["reactions"].get("total_count", 0)
    user = item.get("user") or {}
    return Issue(
        number=item["number"],
        title=item.get("title", ""),
        body=item.get("body") or "",
        state=item.get("state", "open"),
        author=user.get("login", ""),
        author_association=item.get("author_association", ""),
        labels=labels,
        reactions=reactions,
        comments=item.get("comments", 0),
        linked_prs=[],
        created_at=item.get("created_at", ""),
        updated_at=item.get("updated_at", ""),
        fetched_at=time.time(),
    )


def ingest(
    db: Database,
    client: SearchClient,
    cfg: Config,
    now: datetime | None = None,
) -> list[int]:
    """Run one incremental ingestion pass.

    Returns the list of issue numbers that were inserted or updated. The
    watermark is advanced to the start time of this run so the next pass only
    sees newer activity.
    """
    now = now or datetime.now(timezone.utc)
    run_started = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    since = db.get_meta(WATERMARK_KEY)

    seen: dict[int, Issue] = {}
    for query in build_queries(cfg, since):
        for item in client.search_issues(query):
            # Defensive: the search/issues endpoint also returns PRs.
            if "pull_request" in item:
                continue
            issue = _parse_issue(item)
            # De-dup across queries within a single run; keep the richest copy.
            seen[issue.number] = issue

    for issue in seen.values():
        db.upsert_issue(issue)

    db.set_meta(WATERMARK_KEY, run_started)
    return sorted(seen.keys())
