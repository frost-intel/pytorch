"""Phase 7: health metrics and the feedback loop.

Computes metrics that reflect workflow *health* rather than vanity volume:
merge/acceptance rate, change-request (churn) rate, and triage precision. Also
exports labelled examples of accepted vs. rejected PRs so they can be fed back
into the Phase 2 scoring prompt as few-shot guidance.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .db import BUCKET_READY, Database


@dataclass
class Metrics:
    total_issues: int
    triaged_issues: int
    ready_to_work: int
    open_prs: int
    merged_prs: int
    closed_unmerged_prs: int
    merge_rate: float  # merged / (merged + closed_unmerged)
    change_request_rate: float  # PRs with changes requested / total reviewed
    triage_precision: float  # ready issues that progressed past investigating

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_metrics(db: Database) -> Metrics:
    issues = db.all_issues()
    prs = db.all_prs()

    triaged = [i for i in issues if i.bucket is not None]
    ready = [i for i in issues if i.bucket == BUCKET_READY]

    merged = [p for p in prs if p["state"] == "merged"]
    closed_unmerged = [p for p in prs if p["state"] == "closed"]
    open_prs = [p for p in prs if p["state"] == "open"]

    decided = len(merged) + len(closed_unmerged)
    merge_rate = len(merged) / decided if decided else 0.0

    reviewed = [p for p in prs if p["state"] in ("merged", "closed")]
    churned = [p for p in reviewed if p.get("changes_requested")]
    change_request_rate = len(churned) / len(reviewed) if reviewed else 0.0

    # Triage precision proxy: of issues triaged ready-to-work, how many actually
    # advanced beyond the backlog (i.e. a human agreed they were workable)?
    progressed = [
        i
        for i in ready
        if i.board_column not in ("backlog", "closed")
    ]
    triage_precision = len(progressed) / len(ready) if ready else 0.0

    return Metrics(
        total_issues=len(issues),
        triaged_issues=len(triaged),
        ready_to_work=len(ready),
        open_prs=len(open_prs),
        merged_prs=len(merged),
        closed_unmerged_prs=len(closed_unmerged),
        merge_rate=round(merge_rate, 3),
        change_request_rate=round(change_request_rate, 3),
        triage_precision=round(triage_precision, 3),
    )


def feedback_examples(db: Database, limit: int = 10) -> dict[str, list[dict[str, Any]]]:
    """Export accepted/rejected PR outcomes for few-shot scoring feedback.

    Returns ``{"good": [...], "bad": [...]}`` where each entry pairs an issue's
    triage record with the eventual PR outcome. Feeding these back into the
    Phase 2 prompt improves triage precision over time.
    """
    prs = db.all_prs()
    good: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []
    for pr in prs:
        issue_number = pr.get("issue_number")
        if issue_number is None:
            continue
        issue = db.get_issue(issue_number)
        if issue is None or issue.triage is None:
            continue
        example = {
            "issue": issue.number,
            "title": issue.title,
            "triage": issue.triage,
            "outcome": pr["state"],
            "changes_requested": bool(pr.get("changes_requested")),
        }
        if pr["state"] == "merged" and not pr.get("changes_requested"):
            good.append(example)
        elif pr["state"] == "closed":
            bad.append(example)
    return {"good": good[:limit], "bad": bad[:limit]}
