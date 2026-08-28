"""Phase 3: tracking board state machine.

The board mirrors engineering progress for issues a human has *pulled* into the
sprint. AI proposes the ready-to-work list (Phase 2); only an explicit
``pull_to_board`` call (a human action in practice) moves work forward. This
keeps a human accountable for every issue picked up.
"""

from __future__ import annotations

from .db import BOARD_COLUMNS, BUCKET_READY, Database


# Allowed forward/backward transitions. Backwards moves (e.g. back to
# investigating) are permitted because real work is non-linear, but skipping
# ahead past repro confirmation into a PR is not.
_ALLOWED = {
    "backlog": {"investigating", "closed"},
    "investigating": {"repro_confirmed", "needs_info", "backlog", "closed"},
    "repro_confirmed": {"fix_in_progress", "investigating", "closed"},
    "fix_in_progress": {"pr_open", "repro_confirmed", "closed"},
    "pr_open": {"merged", "fix_in_progress", "closed"},
    "merged": set(),
    "closed": set(),
}


class BoardError(ValueError):
    pass


def ready_list(db: Database, limit: int | None = None) -> list:
    """Return AI-proposed ready-to-work issues, highest actionability first."""
    issues = db.issues_in_bucket(BUCKET_READY)
    return issues[:limit] if limit else issues


def pull_to_board(db: Database, number: int, assignee: str) -> None:
    """Human gate #1: assign an engineer and start investigation.

    Only issues that triage marked ready-to-work may be pulled. This is the
    enforcement point for "AI proposes, humans dispose".
    """
    issue = db.get_issue(number)
    if issue is None:
        raise BoardError(f"unknown issue #{number}")
    if issue.bucket != BUCKET_READY:
        raise BoardError(
            f"issue #{number} is in bucket {issue.bucket!r}, not ready-to-work; "
            "it must be re-triaged or handled via the appropriate gate"
        )
    db.set_board(number, "investigating", assignee=assignee)


def advance(db: Database, number: int, to_column: str) -> None:
    """Move an issue to a new board column, validating the transition."""
    if to_column not in BOARD_COLUMNS and to_column != "needs_info":
        raise BoardError(f"unknown column {to_column!r}")
    issue = db.get_issue(number)
    if issue is None:
        raise BoardError(f"unknown issue #{number}")
    current = issue.board_column
    allowed = _ALLOWED.get(current, set())
    if to_column not in allowed:
        raise BoardError(
            f"illegal transition for #{number}: {current} -> {to_column}; "
            f"allowed: {sorted(allowed)}"
        )
    target = "backlog" if to_column == "needs_info" else to_column
    db.set_board(number, target)


def board_snapshot(db: Database) -> dict[str, list[int]]:
    """Return issue numbers grouped by board column (for display/export)."""
    return {col: [i.number for i in db.issues_in_column(col)] for col in BOARD_COLUMNS}
