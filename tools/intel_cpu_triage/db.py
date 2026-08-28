"""SQLite persistence: the toolkit's single source of truth.

The database keeps issue records, triage results, board state, and tracked
pull requests. It is intentionally a thin layer over :mod:`sqlite3` so it has
no third-party dependencies and is trivial to inspect with the ``sqlite3`` CLI.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


# Kanban columns (Phase 3). ``BACKLOG`` is the entry point for triaged,
# ready-to-work issues; the rest mirror engineering progress.
BOARD_COLUMNS: tuple[str, ...] = (
    "backlog",
    "investigating",
    "repro_confirmed",
    "fix_in_progress",
    "pr_open",
    "merged",
    "closed",
)

# Triage buckets (Phase 2).
BUCKET_READY = "ready_to_work"
BUCKET_NEEDS_INFO = "needs_info_repro"
BUCKET_NEEDS_DECISION = "needs_maintainer_decision"
BUCKETS: tuple[str, ...] = (BUCKET_READY, BUCKET_NEEDS_INFO, BUCKET_NEEDS_DECISION)

SCHEMA = """
CREATE TABLE IF NOT EXISTS issues (
    number INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    body TEXT,
    state TEXT,
    author TEXT,
    author_association TEXT,
    labels TEXT,              -- JSON array
    reactions INTEGER DEFAULT 0,
    comments INTEGER DEFAULT 0,
    linked_prs TEXT,          -- JSON array of PR numbers
    created_at TEXT,
    updated_at TEXT,
    fetched_at REAL,
    -- triage (Phase 2)
    bucket TEXT,
    actionability INTEGER,
    confidence REAL,
    triage TEXT,              -- JSON blob of full triage record
    triaged_at REAL,
    -- board (Phase 3)
    board_column TEXT DEFAULT 'backlog',
    assignee TEXT,
    board_updated_at REAL
);

CREATE TABLE IF NOT EXISTS pull_requests (
    number INTEGER PRIMARY KEY,
    issue_number INTEGER,
    reviewer TEXT,
    state TEXT,               -- open | merged | closed
    changes_requested INTEGER DEFAULT 0,
    opened_at TEXT,
    closed_at TEXT,
    merged_at TEXT,
    url TEXT
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_issues_bucket ON issues(bucket);
CREATE INDEX IF NOT EXISTS idx_issues_board ON issues(board_column);
CREATE INDEX IF NOT EXISTS idx_pr_state ON pull_requests(state);
"""


@dataclass
class Issue:
    number: int
    title: str
    body: str = ""
    state: str = "open"
    author: str = ""
    author_association: str = ""
    labels: list[str] = field(default_factory=list)
    reactions: int = 0
    comments: int = 0
    linked_prs: list[int] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    fetched_at: float = 0.0
    bucket: str | None = None
    actionability: int | None = None
    confidence: float | None = None
    triage: dict[str, Any] | None = None
    triaged_at: float | None = None
    board_column: str = "backlog"
    assignee: str | None = None
    board_updated_at: float | None = None

    @property
    def text(self) -> str:
        """Combined title + body for keyword/semantic matching."""
        return f"{self.title}\n\n{self.body or ''}"


def _row_to_issue(row: sqlite3.Row) -> Issue:
    return Issue(
        number=row["number"],
        title=row["title"],
        body=row["body"] or "",
        state=row["state"] or "open",
        author=row["author"] or "",
        author_association=row["author_association"] or "",
        labels=json.loads(row["labels"]) if row["labels"] else [],
        reactions=row["reactions"] or 0,
        comments=row["comments"] or 0,
        linked_prs=json.loads(row["linked_prs"]) if row["linked_prs"] else [],
        created_at=row["created_at"] or "",
        updated_at=row["updated_at"] or "",
        fetched_at=row["fetched_at"] or 0.0,
        bucket=row["bucket"],
        actionability=row["actionability"],
        confidence=row["confidence"],
        triage=json.loads(row["triage"]) if row["triage"] else None,
        triaged_at=row["triaged_at"],
        board_column=row["board_column"] or "backlog",
        assignee=row["assignee"],
        board_updated_at=row["board_updated_at"],
    )


class Database:
    """Thin SQLite wrapper for the triage toolkit."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # -- issues -----------------------------------------------------------
    def upsert_issue(self, issue: Issue) -> None:
        """Insert or update an issue, preserving triage/board fields on update.

        Ingestion (Phase 1) only owns the raw GitHub fields; it must not clobber
        triage or board state set by later phases. We therefore merge: ingest
        fields are overwritten, but triage/board columns are kept unless this
        record explicitly carries new values.
        """
        existing = self.get_issue(issue.number)
        if existing is not None:
            # Preserve downstream state unless the incoming issue sets it.
            if issue.bucket is None:
                issue.bucket = existing.bucket
                issue.actionability = existing.actionability
                issue.confidence = existing.confidence
                issue.triage = existing.triage
                issue.triaged_at = existing.triaged_at
            if issue.board_updated_at is None:
                issue.board_column = existing.board_column
                issue.assignee = existing.assignee
                issue.board_updated_at = existing.board_updated_at
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO issues (
                    number, title, body, state, author, author_association,
                    labels, reactions, comments, linked_prs, created_at,
                    updated_at, fetched_at, bucket, actionability, confidence,
                    triage, triaged_at, board_column, assignee, board_updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(number) DO UPDATE SET
                    title=excluded.title,
                    body=excluded.body,
                    state=excluded.state,
                    author=excluded.author,
                    author_association=excluded.author_association,
                    labels=excluded.labels,
                    reactions=excluded.reactions,
                    comments=excluded.comments,
                    linked_prs=excluded.linked_prs,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    fetched_at=excluded.fetched_at,
                    bucket=excluded.bucket,
                    actionability=excluded.actionability,
                    confidence=excluded.confidence,
                    triage=excluded.triage,
                    triaged_at=excluded.triaged_at,
                    board_column=excluded.board_column,
                    assignee=excluded.assignee,
                    board_updated_at=excluded.board_updated_at
                """,
                (
                    issue.number,
                    issue.title,
                    issue.body,
                    issue.state,
                    issue.author,
                    issue.author_association,
                    json.dumps(issue.labels),
                    issue.reactions,
                    issue.comments,
                    json.dumps(issue.linked_prs),
                    issue.created_at,
                    issue.updated_at,
                    issue.fetched_at,
                    issue.bucket,
                    issue.actionability,
                    issue.confidence,
                    json.dumps(issue.triage) if issue.triage is not None else None,
                    issue.triaged_at,
                    issue.board_column,
                    issue.assignee,
                    issue.board_updated_at,
                ),
            )

    def get_issue(self, number: int) -> Issue | None:
        row = self.conn.execute(
            "SELECT * FROM issues WHERE number = ?", (number,)
        ).fetchone()
        return _row_to_issue(row) if row else None

    def all_issues(self) -> list[Issue]:
        rows = self.conn.execute(
            "SELECT * FROM issues ORDER BY number"
        ).fetchall()
        return [_row_to_issue(r) for r in rows]

    def issues_in_bucket(self, bucket: str) -> list[Issue]:
        rows = self.conn.execute(
            "SELECT * FROM issues WHERE bucket = ? ORDER BY "
            "actionability DESC, reactions DESC",
            (bucket,),
        ).fetchall()
        return [_row_to_issue(r) for r in rows]

    def issues_in_column(self, column: str) -> list[Issue]:
        rows = self.conn.execute(
            "SELECT * FROM issues WHERE board_column = ? ORDER BY number",
            (column,),
        ).fetchall()
        return [_row_to_issue(r) for r in rows]

    def save_triage(
        self,
        number: int,
        bucket: str,
        actionability: int,
        confidence: float,
        triage: dict[str, Any],
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE issues SET bucket=?, actionability=?, confidence=?, "
                "triage=?, triaged_at=? WHERE number=?",
                (
                    bucket,
                    actionability,
                    confidence,
                    json.dumps(triage),
                    time.time(),
                    number,
                ),
            )

    def set_board(
        self, number: int, column: str, assignee: str | None = None
    ) -> None:
        if column not in BOARD_COLUMNS:
            raise ValueError(f"unknown board column: {column}")
        with self._tx() as conn:
            conn.execute(
                "UPDATE issues SET board_column=?, assignee=COALESCE(?, assignee), "
                "board_updated_at=? WHERE number=?",
                (column, assignee, time.time(), number),
            )

    # -- pull requests ----------------------------------------------------
    def upsert_pr(
        self,
        number: int,
        issue_number: int | None = None,
        reviewer: str | None = None,
        state: str = "open",
        changes_requested: int = 0,
        opened_at: str = "",
        closed_at: str = "",
        merged_at: str = "",
        url: str = "",
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO pull_requests (
                    number, issue_number, reviewer, state, changes_requested,
                    opened_at, closed_at, merged_at, url
                ) VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(number) DO UPDATE SET
                    issue_number=COALESCE(excluded.issue_number, issue_number),
                    reviewer=COALESCE(excluded.reviewer, reviewer),
                    state=excluded.state,
                    changes_requested=excluded.changes_requested,
                    closed_at=excluded.closed_at,
                    merged_at=excluded.merged_at,
                    url=excluded.url
                """,
                (
                    number,
                    issue_number,
                    reviewer,
                    state,
                    changes_requested,
                    opened_at,
                    closed_at,
                    merged_at,
                    url,
                ),
            )

    def all_prs(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM pull_requests ORDER BY number"
        ).fetchall()
        return [dict(r) for r in rows]

    def open_prs(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM pull_requests WHERE state = 'open' ORDER BY number"
        ).fetchall()
        return [dict(r) for r in rows]

    # -- meta / watermark -------------------------------------------------
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
