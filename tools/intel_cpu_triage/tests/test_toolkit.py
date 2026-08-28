"""Unit tests for the Intel CPU triage toolkit.

These use stdlib ``unittest`` (no PyTorch build required) since the toolkit is
dependency-free and runs independently of the framework.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from tools.intel_cpu_triage.config import Config
from tools.intel_cpu_triage.db import (
    BUCKET_NEEDS_DECISION,
    BUCKET_NEEDS_INFO,
    BUCKET_READY,
    Database,
    Issue,
)
from tools.intel_cpu_triage import board, ingest, metrics, ratelimit
from tools.intel_cpu_triage.dedup import SimilarityIndex
from tools.intel_cpu_triage.triage import (
    HeuristicTriager,
    LLMTriager,
    _extract_json,
)


def make_issue(number=1, title="t", body="", labels=None, **kw) -> Issue:
    return Issue(
        number=number,
        title=title,
        body=body,
        labels=labels or [],
        **kw,
    )


class FakeSearchClient:
    """In-memory stand-in for GitHubClient used by ingestion tests."""

    def __init__(self, items_by_query=None, items=None):
        self._items_by_query = items_by_query or {}
        self._items = items or []

    def search_issues(self, query, per_page=100):
        if query in self._items_by_query:
            yield from self._items_by_query[query]
        else:
            yield from self._items


class ConfigTest(unittest.TestCase):
    def test_defaults_present(self):
        cfg = Config()
        self.assertIn("module: cpu", cfg.scope_labels)
        self.assertIn("mkldnn", cfg.content_keywords)
        self.assertEqual(cfg.max_open_external_prs, 5)


class DatabaseTest(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")

    def test_upsert_and_get(self):
        self.db.upsert_issue(make_issue(1, "hello", labels=["module: cpu"]))
        got = self.db.get_issue(1)
        self.assertIsNotNone(got)
        self.assertEqual(got.title, "hello")
        self.assertEqual(got.labels, ["module: cpu"])

    def test_upsert_preserves_triage_on_reingest(self):
        self.db.upsert_issue(make_issue(1, "hello"))
        self.db.save_triage(1, BUCKET_READY, 80, 0.9, {"x": 1})
        # Re-ingest (Phase 1) with a fresh Issue that has no triage fields.
        self.db.upsert_issue(make_issue(1, "hello updated"))
        got = self.db.get_issue(1)
        self.assertEqual(got.title, "hello updated")
        self.assertEqual(got.bucket, BUCKET_READY)
        self.assertEqual(got.actionability, 80)

    def test_board_preserved_on_reingest(self):
        self.db.upsert_issue(make_issue(1))
        self.db.save_triage(1, BUCKET_READY, 80, 0.9, {})
        self.db.set_board(1, "investigating", assignee="alice")
        self.db.upsert_issue(make_issue(1, "new title"))
        got = self.db.get_issue(1)
        self.assertEqual(got.board_column, "investigating")
        self.assertEqual(got.assignee, "alice")

    def test_meta_watermark(self):
        self.assertIsNone(self.db.get_meta("k"))
        self.db.set_meta("k", "v")
        self.assertEqual(self.db.get_meta("k"), "v")


class IngestTest(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.cfg = Config()

    def test_build_queries_includes_labels_and_keywords(self):
        qs = ingest.build_queries(self.cfg, since=None)
        self.assertTrue(any('label:"module: cpu"' in q for q in qs))
        self.assertTrue(any("in:title,body" in q for q in qs))

    def test_build_queries_watermark(self):
        qs = ingest.build_queries(self.cfg, since="2024-01-01T00:00:00Z")
        self.assertTrue(all("updated:>=2024-01-01T00:00:00Z" in q for q in qs))

    def test_ingest_dedup_and_watermark(self):
        item = {
            "number": 42,
            "title": "AMX brgemm segfault",
            "body": "import torch\nrepro here",
            "state": "open",
            "labels": [{"name": "module: cpu"}],
            "user": {"login": "bob"},
            "author_association": "NONE",
            "comments": 3,
            "reactions": {"total_count": 5},
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-02-01T00:00:00Z",
        }
        client = FakeSearchClient(items=[item])
        now = datetime(2024, 3, 1, tzinfo=timezone.utc)
        changed = ingest.ingest(self.db, client, self.cfg, now=now)
        self.assertEqual(changed, [42])
        got = self.db.get_issue(42)
        self.assertEqual(got.reactions, 5)
        self.assertEqual(got.author, "bob")
        self.assertEqual(self.db.get_meta(ingest.WATERMARK_KEY), "2024-03-01T00:00:00Z")

    def test_ingest_skips_pull_requests(self):
        item = {"number": 1, "title": "pr", "pull_request": {}, "labels": []}
        client = FakeSearchClient(items=[item])
        changed = ingest.ingest(self.db, client, self.cfg)
        self.assertEqual(changed, [])


class HeuristicTriageTest(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.triager = HeuristicTriager(self.cfg)

    def test_ready_when_scope_and_repro(self):
        issue = make_issue(
            1,
            "mkldnn conv wrong result",
            body="import torch\nSteps to reproduce:\n```python\n...\n```",
            labels=["module: mkldnn", "triaged", "high priority"],
            reactions=4,
        )
        res = self.triager.triage(issue)
        self.assertEqual(res.bucket, BUCKET_READY)
        self.assertGreaterEqual(res.actionability, self.cfg.ready_score_threshold)
        self.assertIn("aten/src/ATen/native/mkldnn/", res.suspected_subsystems)

    def test_needs_info_when_no_repro(self):
        issue = make_issue(
            2, "cpu slow", body="it is slow", labels=["module: cpu"]
        )
        res = self.triager.triage(issue)
        self.assertEqual(res.bucket, BUCKET_NEEDS_INFO)

    def test_quarantine_feature_request(self):
        issue = make_issue(
            3,
            "Add new API for bf16",
            body="import torch\nrepro\nplease add a flag",
            labels=["module: cpu", "feature"],
        )
        res = self.triager.triage(issue)
        self.assertEqual(res.bucket, BUCKET_NEEDS_DECISION)
        self.assertTrue(res.quarantine_reasons)

    def test_quarantine_existing_pr(self):
        issue = make_issue(
            4,
            "mkldnn bug",
            body="import torch\n```python\nrepro\n```",
            labels=["module: cpu"],
            linked_prs=[999],
        )
        res = self.triager.triage(issue)
        self.assertEqual(res.bucket, BUCKET_NEEDS_DECISION)
        self.assertIn("already has a linked/open PR", res.quarantine_reasons)

    def test_content_keyword_in_scope_when_mislabeled(self):
        issue = make_issue(
            5,
            "Segfault on Sapphire Rapids",
            body="import torch\n```python\nAVX512 brgemm crash\n```",
            labels=[],  # mislabeled
        )
        res = self.triager.triage(issue)
        self.assertTrue(res.in_scope)


class LLMTriageTest(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def test_llm_result_merged(self):
        class FakeLLM:
            def complete(self, prompt):
                return (
                    '{"in_scope": true, "has_repro": true, "fix_type": "kernel",'
                    ' "difficulty": "small", "suspected_subsystems": ["x/"],'
                    ' "root_cause_hypothesis": "race", "actionability": 88,'
                    ' "confidence": 0.8, "quarantine_reasons": []}'
                )

        triager = LLMTriager(self.cfg, FakeLLM())
        issue = make_issue(1, "mkldnn", body="import torch", labels=["module: cpu"])
        res = triager.triage(issue)
        self.assertEqual(res.source, "llm")
        self.assertEqual(res.actionability, 88)
        self.assertEqual(res.bucket, BUCKET_READY)
        self.assertEqual(res.root_cause_hypothesis, "race")

    def test_llm_failure_falls_back(self):
        class BrokenLLM:
            def complete(self, prompt):
                raise RuntimeError("no model")

        triager = LLMTriager(self.cfg, BrokenLLM())
        issue = make_issue(1, "mkldnn", body="x", labels=["module: cpu"])
        res = triager.triage(issue)
        self.assertEqual(res.source, "heuristic")

    def test_extract_json_with_fences(self):
        raw = "```json\n{\"a\": 1}\n```"
        self.assertEqual(_extract_json(raw), {"a": 1})


class DedupTest(unittest.TestCase):
    def test_similar_issues_rank_high(self):
        issues = [
            make_issue(1, "mkldnn conv segfault on avx512", body="brgemm crash amx"),
            make_issue(2, "mkldnn convolution crash avx512", body="brgemm segfault amx"),
            make_issue(3, "docs typo in readme", body="fix spelling"),
        ]
        index = SimilarityIndex()
        index.build(issues)
        ranked = index.most_similar(issues[0], top_k=2)
        self.assertEqual(ranked[0][0], 2)
        self.assertGreater(ranked[0][1], 0.0)


class BoardTest(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.cfg = Config()

    def _ready_issue(self, number=1):
        self.db.upsert_issue(make_issue(number))
        self.db.save_triage(number, BUCKET_READY, 80, 0.9, {})

    def test_pull_requires_ready(self):
        self.db.upsert_issue(make_issue(1))
        self.db.save_triage(1, BUCKET_NEEDS_INFO, 30, 0.5, {})
        with self.assertRaises(board.BoardError):
            board.pull_to_board(self.db, 1, "alice")

    def test_pull_and_advance_flow(self):
        self._ready_issue(1)
        board.pull_to_board(self.db, 1, "alice")
        self.assertEqual(self.db.get_issue(1).board_column, "investigating")
        board.advance(self.db, 1, "repro_confirmed")
        board.advance(self.db, 1, "fix_in_progress")
        board.advance(self.db, 1, "pr_open")
        board.advance(self.db, 1, "merged")
        self.assertEqual(self.db.get_issue(1).board_column, "merged")

    def test_illegal_transition_rejected(self):
        self._ready_issue(1)
        board.pull_to_board(self.db, 1, "alice")
        with self.assertRaises(board.BoardError):
            board.advance(self.db, 1, "merged")  # skipped steps


class RateLimitTest(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.cfg = Config(max_open_external_prs=2, max_prs_per_reviewer=1)

    def test_team_cap(self):
        self.db.upsert_pr(1, state="open")
        self.db.upsert_pr(2, state="open")
        decision = ratelimit.can_open_pr(self.db, self.cfg)
        self.assertFalse(decision)
        self.assertTrue(any("cap reached" in r for r in decision.reasons))

    def test_reviewer_budget(self):
        self.db.upsert_pr(1, reviewer="maint", state="open")
        decision = ratelimit.can_open_pr(self.db, self.cfg, reviewer="maint")
        self.assertFalse(decision)

    def test_allowed_when_under_caps(self):
        decision = ratelimit.can_open_pr(self.db, self.cfg, reviewer="maint")
        self.assertTrue(decision)


class MetricsTest(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")

    def test_metrics_compute(self):
        self.db.upsert_issue(make_issue(1))
        self.db.save_triage(1, BUCKET_READY, 80, 0.9, {})
        self.db.set_board(1, "investigating", assignee="a")
        self.db.upsert_pr(1, issue_number=1, state="merged", changes_requested=0)
        self.db.upsert_pr(2, issue_number=1, state="closed", changes_requested=1)
        m = metrics.compute_metrics(self.db)
        self.assertEqual(m.total_issues, 1)
        self.assertEqual(m.merged_prs, 1)
        self.assertEqual(m.merge_rate, 0.5)
        self.assertEqual(m.change_request_rate, 0.5)
        self.assertEqual(m.triage_precision, 1.0)

    def test_feedback_examples(self):
        self.db.upsert_issue(make_issue(1, "good one"))
        self.db.save_triage(1, BUCKET_READY, 80, 0.9, {"k": "v"})
        self.db.upsert_pr(1, issue_number=1, state="merged", changes_requested=0)
        fb = metrics.feedback_examples(self.db)
        self.assertEqual(len(fb["good"]), 1)
        self.assertEqual(fb["good"][0]["issue"], 1)


if __name__ == "__main__":
    unittest.main()
