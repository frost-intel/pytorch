"""Configuration for the Intel CPU triage toolkit.

Defaults encode the labels and content keywords that are relevant to an Intel
CPU / oneDNN team working on ``pytorch/pytorch``. They can be overridden by a
JSON file (see ``config.example.json``) loaded via :func:`load_config`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


# Labels that, on their own, make an issue in-scope for an Intel CPU team.
DEFAULT_SCOPE_LABELS: list[str] = [
    "module: cpu",
    "module: mkldnn",
    "module: amp",
    "module: half",
    "module: performance",
    "oncall: quantization",
]

# Labels that maintainers use to signal an issue is worth working on. Presence
# of these raises an issue's priority during scoring.
DEFAULT_SIGNAL_LABELS: list[str] = [
    "triaged",
    "triage review",
    "high priority",
    "small",
    "good first issue",
]

# Labels that should lower priority or quarantine an issue.
DEFAULT_NEGATIVE_LABELS: list[str] = [
    "needs reproduction",
    "module: dependency bug",
    "feature",
    "enhancement",
    "proposal accepted",
]

# Content keywords implying Intel hardware even when an issue is mislabeled.
# Matched case-insensitively against title + body.
DEFAULT_CONTENT_KEYWORDS: list[str] = [
    "amx",
    "onednn",
    "mkldnn",
    "mkl-dnn",
    "ideep",
    "avx512",
    "avx-512",
    "avx2",
    "vnni",
    "bf16",
    "bfloat16",
    "fp16",
    "xeon",
    "sapphire rapids",
    "granite rapids",
    "emerald rapids",
    "intel",
    "cpublas",
    "brgemm",
]

# Subsystem hints: keyword -> suspected source path. Used by the heuristic
# triager to populate "suspected files/subsystem".
DEFAULT_SUBSYSTEM_HINTS: dict[str, str] = {
    "mkldnn": "aten/src/ATen/native/mkldnn/",
    "onednn": "aten/src/ATen/native/mkldnn/",
    "ideep": "third_party/ideep/",
    "brgemm": "aten/src/ATen/native/CPUBlas.cpp",
    "cpublas": "aten/src/ATen/native/CPUBlas.cpp",
    "quant": "aten/src/ATen/native/quantized/cpu/",
    "avx512": "aten/src/ATen/cpu/vec/",
    "avx2": "aten/src/ATen/cpu/vec/",
    "vec256": "aten/src/ATen/cpu/vec/",
    "bf16": "aten/src/ATen/cpu/vec/",
    "openmp": "aten/src/ATen/ParallelOpenMP.h",
}


@dataclass
class Config:
    """Toolkit configuration."""

    repo: str = "pytorch/pytorch"
    scope_labels: list[str] = field(default_factory=lambda: list(DEFAULT_SCOPE_LABELS))
    signal_labels: list[str] = field(
        default_factory=lambda: list(DEFAULT_SIGNAL_LABELS)
    )
    negative_labels: list[str] = field(
        default_factory=lambda: list(DEFAULT_NEGATIVE_LABELS)
    )
    content_keywords: list[str] = field(
        default_factory=lambda: list(DEFAULT_CONTENT_KEYWORDS)
    )
    subsystem_hints: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_SUBSYSTEM_HINTS)
    )

    # Phase 6 rate-limiting policy.
    max_open_external_prs: int = 5
    max_prs_per_reviewer: int = 2

    # Phase 2 scoring: minimum actionability score (0-100) to land in
    # "ready-to-work".
    ready_score_threshold: int = 60

    db_path: str = "intel_cpu_triage.db"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from a JSON file, falling back to defaults.

    Unknown keys in the file are ignored so that newer config files remain
    loadable by older code.
    """
    cfg = Config()
    if path is None:
        return cfg
    p = Path(path)
    if not p.exists():
        return cfg
    data = json.loads(p.read_text())
    known = set(asdict(cfg).keys())
    for key, value in data.items():
        if key in known:
            setattr(cfg, key, value)
    return cfg
