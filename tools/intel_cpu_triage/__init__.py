"""AI-assisted Intel CPU issue resolution toolkit for pytorch/pytorch.

This package implements the workflow described in ``README.md``: incremental
ingestion of relevant issues from GitHub, AI-assisted triage and scoring,
duplicate detection, a tracking board state machine, rate-limiting to protect
reviewer trust, and health metrics.

The toolkit is intentionally dependency-free (standard library only) so it can
run in restricted environments and CI without a PyTorch build.
"""

from __future__ import annotations

__all__ = [
    "config",
    "db",
    "github_client",
    "ingest",
    "triage",
    "dedup",
    "board",
    "ratelimit",
    "metrics",
]

__version__ = "0.1.0"
