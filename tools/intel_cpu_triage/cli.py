"""Command-line interface for the Intel CPU triage toolkit.

Subcommands map onto workflow phases:

    ingest    Phase 1   pull relevant issues into the local DB
    triage    Phase 2   score & bucket issues (+ duplicate detection)
    ready     Phase 3   show the AI-proposed ready-to-work list
    pull      Phase 3   assign an engineer (human gate #1)
    advance   Phase 3   move an issue along the board
    board     Phase 3   print the board snapshot
    pr-check  Phase 6   check whether a new upstream PR is within policy
    metrics   Phase 7   print health metrics
    feedback  Phase 7   export accepted/rejected examples for prompt tuning

The triage subcommand uses the heuristic triager by default. Wire an LLM by
importing :class:`triage.LLMTriager` from your own driver script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

from . import board, metrics, ratelimit
from .config import load_config
from .db import Database
from .dedup import SimilarityIndex
from .github_client import GitHubClient
from .ingest import ingest
from .triage import HeuristicTriager


def _db(args: argparse.Namespace):
    cfg = load_config(args.config)
    if args.db:
        cfg.db_path = args.db
    return Database(cfg.db_path), cfg


def cmd_ingest(args: argparse.Namespace) -> int:
    db, cfg = _db(args)
    token = os.environ.get("GITHUB_TOKEN")
    client = GitHubClient(token=token)
    changed = ingest(db, client, cfg)
    print(f"ingested/updated {len(changed)} issues")
    return 0


def cmd_triage(args: argparse.Namespace) -> int:
    db, cfg = _db(args)
    issues = db.all_issues()
    index = SimilarityIndex()
    index.build(issues)
    triager = HeuristicTriager(cfg)
    for issue in issues:
        if not args.all and issue.bucket is not None:
            continue
        related = [n for n, _ in index.most_similar(issue, top_k=3, threshold=0.2)]
        result = triager.triage(issue, related=related)
        db.save_triage(
            issue.number,
            result.bucket,
            result.actionability,
            result.confidence,
            result.to_dict(),
        )
    counts: dict[str, int] = {}
    for issue in db.all_issues():
        if issue.bucket:
            counts[issue.bucket] = counts.get(issue.bucket, 0) + 1
    print(json.dumps(counts, indent=2))
    return 0


def cmd_ready(args: argparse.Namespace) -> int:
    db, _ = _db(args)
    for issue in board.ready_list(db, limit=args.limit):
        subsystems = ""
        if issue.triage:
            subsystems = ", ".join(issue.triage.get("suspected_subsystems", []))
        print(
            f"#{issue.number:<7} score={issue.actionability:<3} "
            f"conf={issue.confidence}  {issue.title[:70]}"
            + (f"  [{subsystems}]" if subsystems else "")
        )
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    db, _ = _db(args)
    try:
        board.pull_to_board(db, args.number, args.assignee)
    except board.BoardError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"#{args.number} -> investigating (assignee: {args.assignee})")
    return 0


def cmd_advance(args: argparse.Namespace) -> int:
    db, _ = _db(args)
    try:
        board.advance(db, args.number, args.to)
    except board.BoardError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"#{args.number} -> {args.to}")
    return 0


def cmd_board(args: argparse.Namespace) -> int:
    db, _ = _db(args)
    print(json.dumps(board.board_snapshot(db), indent=2))
    return 0


def cmd_pr_check(args: argparse.Namespace) -> int:
    db, cfg = _db(args)
    decision = ratelimit.can_open_pr(db, cfg, reviewer=args.reviewer)
    if decision:
        print("OK: opening a PR is within policy")
        return 0
    print("BLOCKED: " + "; ".join(decision.reasons), file=sys.stderr)
    return 2


def cmd_metrics(args: argparse.Namespace) -> int:
    db, _ = _db(args)
    print(json.dumps(metrics.compute_metrics(db).to_dict(), indent=2))
    return 0


def cmd_feedback(args: argparse.Namespace) -> int:
    db, _ = _db(args)
    print(json.dumps(metrics.feedback_examples(db), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="intel_cpu_triage", description=__doc__)
    p.add_argument("--config", help="path to config JSON", default=None)
    p.add_argument("--db", help="path to SQLite DB (overrides config)", default=None)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="Phase 1: pull relevant issues").set_defaults(
        func=cmd_ingest
    )

    t = sub.add_parser("triage", help="Phase 2: score & bucket issues")
    t.add_argument("--all", action="store_true", help="re-triage already-triaged issues")
    t.set_defaults(func=cmd_triage)

    r = sub.add_parser("ready", help="Phase 3: list ready-to-work issues")
    r.add_argument("--limit", type=int, default=20)
    r.set_defaults(func=cmd_ready)

    pl = sub.add_parser("pull", help="Phase 3: assign an engineer (human gate #1)")
    pl.add_argument("number", type=int)
    pl.add_argument("assignee")
    pl.set_defaults(func=cmd_pull)

    av = sub.add_parser("advance", help="Phase 3: move an issue along the board")
    av.add_argument("number", type=int)
    av.add_argument("to", help="target column")
    av.set_defaults(func=cmd_advance)

    sub.add_parser("board", help="Phase 3: print board snapshot").set_defaults(
        func=cmd_board
    )

    pc = sub.add_parser("pr-check", help="Phase 6: is a new upstream PR allowed?")
    pc.add_argument("--reviewer", default=None)
    pc.set_defaults(func=cmd_pr_check)

    sub.add_parser("metrics", help="Phase 7: health metrics").set_defaults(
        func=cmd_metrics
    )
    sub.add_parser("feedback", help="Phase 7: export scoring feedback").set_defaults(
        func=cmd_feedback
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
