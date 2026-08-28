"""Phase 6: rate-limiting to protect reviewer trust.

Enforces a team-wide cap on outstanding external PRs and a per-reviewer budget
so the team never floods maintainers. ``can_open_pr`` is the gate that the
Phase 5 internal-review step should consult before a PR is opened upstream.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .config import Config
from .db import Database


@dataclass
class RateDecision:
    allowed: bool
    reasons: list[str]

    def __bool__(self) -> bool:
        return self.allowed


def open_pr_count(db: Database) -> int:
    return len(db.open_prs())


def reviewer_load(db: Database) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for pr in db.open_prs():
        reviewer = pr.get("reviewer")
        if reviewer:
            counts[reviewer] += 1
    return dict(counts)


def can_open_pr(db: Database, cfg: Config, reviewer: str | None = None) -> RateDecision:
    """Decide whether opening one more upstream PR is within policy."""
    reasons: list[str] = []
    n_open = open_pr_count(db)
    if n_open >= cfg.max_open_external_prs:
        reasons.append(
            f"team-wide open-PR cap reached ({n_open}/{cfg.max_open_external_prs})"
        )
    if reviewer:
        load = reviewer_load(db).get(reviewer, 0)
        if load >= cfg.max_prs_per_reviewer:
            reasons.append(
                f"reviewer {reviewer!r} budget reached "
                f"({load}/{cfg.max_prs_per_reviewer})"
            )
    return RateDecision(allowed=not reasons, reasons=reasons)
