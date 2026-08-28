"""Phase 2: AI triage & scoring.

Each issue is converted into a structured :class:`TriageResult` (serialised to
JSON in the DB). The :class:`LLMTriager` delegates to a user-supplied language
model that must return JSON; if no model is configured or it fails, the
:class:`HeuristicTriager` produces a deterministic, explainable result from
labels and content keywords so the pipeline always makes progress.

Scoring deliberately *down-ranks or quarantines*:
  * issues that already have an open PR (avoid reviewer flooding),
  * feature requests / design discussions (need maintainer buy-in),
  * anything implying API/semantics changes (need an RFC first).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Protocol

from .config import Config
from .db import BUCKET_NEEDS_DECISION, BUCKET_NEEDS_INFO, BUCKET_READY, Issue


# Phrases that suggest an API/semantics change needing maintainer agreement.
_API_CHANGE_HINTS = (
    "new api",
    "add a flag",
    "add an option",
    "change the default",
    "change default",
    "deprecate",
    "rename",
    "signature",
    "bc-breaking",
    "backward incompat",
    "rfc",
)

_REPRO_HINTS = (
    "import torch",
    "```python",
    "```py",
    "traceback",
    "to reproduce",
    "steps to reproduce",
    "minimal example",
    "minimal repro",
    "reproduce:",
)


@dataclass
class TriageResult:
    """Structured triage record (JSON-serialisable)."""

    number: int
    bucket: str
    actionability: int  # 0-100
    confidence: float  # 0-1
    in_scope: bool
    has_repro: bool
    fix_type: str  # kernel | numerical | perf | build | docs | unknown
    difficulty: str  # small | medium | large
    suspected_subsystems: list[str] = field(default_factory=list)
    root_cause_hypothesis: str = ""
    quarantine_reasons: list[str] = field(default_factory=list)
    related_issues: list[int] = field(default_factory=list)
    rationale: str = ""
    source: str = "heuristic"  # heuristic | llm

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLM(Protocol):
    """A language model that returns a JSON string for a given prompt."""

    def complete(self, prompt: str) -> str:
        ...


def _matches(text: str, needles) -> list[str]:
    low = text.lower()
    return [n for n in needles if n.lower() in low]


def _detect_quarantine(issue: Issue, cfg: Config) -> list[str]:
    reasons: list[str] = []
    if issue.linked_prs:
        reasons.append("already has a linked/open PR")
    labelset = {label.lower() for label in issue.labels}
    if {"feature", "enhancement", "proposal accepted"} & labelset:
        reasons.append("feature request / design discussion")
    if _matches(issue.text, _API_CHANGE_HINTS):
        reasons.append("implies API/semantics change (needs RFC/maintainer)")
    return reasons


def _detect_subsystems(issue: Issue, cfg: Config) -> list[str]:
    found: list[str] = []
    low = issue.text.lower()
    for keyword, path in cfg.subsystem_hints.items():
        if keyword in low and path not in found:
            found.append(path)
    return found


def _fix_type(issue: Issue) -> str:
    low = issue.text.lower()
    labelset = {label.lower() for label in issue.labels}
    if "module: docs" in labelset or "documentation" in low:
        return "docs"
    if "module: build" in labelset or "compil" in low or "cmake" in low:
        return "build"
    if "module: performance" in labelset or "slow" in low or "regression" in low:
        return "perf"
    if "nan" in low or "incorrect result" in low or "numerical" in low:
        return "numerical"
    if any(k in low for k in ("segfault", "crash", "kernel", "brgemm", "assert")):
        return "kernel"
    return "unknown"


class HeuristicTriager:
    """Deterministic, explainable triage from labels + content.

    Used as a fallback and as a sane default in offline/CI environments.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def triage(self, issue: Issue, related: list[int] | None = None) -> TriageResult:
        cfg = self.cfg
        labelset = {label.lower() for label in issue.labels}
        scope_hit = bool({s.lower() for s in cfg.scope_labels} & labelset)
        content_hits = _matches(issue.text, cfg.content_keywords)
        in_scope = scope_hit or bool(content_hits)

        has_repro = bool(_matches(issue.text, _REPRO_HINTS))
        quarantine = _detect_quarantine(issue, cfg)
        subsystems = _detect_subsystems(issue, cfg)

        # Score: start from a base, add maintainer signal, repro, scope,
        # engagement; subtract for quarantine and missing repro.
        score = 30
        if scope_hit:
            score += 20
        if content_hits:
            score += min(10, 2 * len(content_hits))
        signal_hit = {s.lower() for s in cfg.signal_labels} & labelset
        score += 10 * len(signal_hit)
        if "high priority" in labelset:
            score += 10
        if has_repro:
            score += 15
        score += min(10, issue.reactions)
        if "needs reproduction" in labelset or not has_repro:
            score -= 15
        if quarantine:
            score -= 40
        score = max(0, min(100, score))

        # Bucket assignment.
        if quarantine:
            bucket = BUCKET_NEEDS_DECISION
        elif not has_repro or "needs reproduction" in labelset:
            bucket = BUCKET_NEEDS_INFO
        elif in_scope and score >= cfg.ready_score_threshold:
            bucket = BUCKET_READY
        else:
            bucket = BUCKET_NEEDS_INFO

        # Confidence is higher when labels and content agree.
        agreement = sum([scope_hit, bool(content_hits), has_repro])
        confidence = round(0.4 + 0.2 * agreement, 2)
        confidence = min(confidence, 0.95)

        return TriageResult(
            number=issue.number,
            bucket=bucket,
            actionability=score,
            confidence=confidence,
            in_scope=in_scope,
            has_repro=has_repro,
            fix_type=_fix_type(issue),
            difficulty="small" if "small" in labelset else "medium",
            suspected_subsystems=subsystems,
            root_cause_hypothesis="",
            quarantine_reasons=quarantine,
            related_issues=related or [],
            rationale=(
                f"scope_label={scope_hit}, content_hits={content_hits}, "
                f"signals={sorted(signal_hit)}, repro={has_repro}"
            ),
            source="heuristic",
        )


TRIAGE_PROMPT = """\
You are triaging a GitHub issue for an Intel CPU / oneDNN team working on
pytorch/pytorch. Respond with a single JSON object and nothing else, using
exactly these keys:
  in_scope (bool), has_repro (bool), fix_type (one of
  kernel|numerical|perf|build|docs|unknown), difficulty (small|medium|large),
  suspected_subsystems (list of repo paths), root_cause_hypothesis (string),
  actionability (integer 0-100), confidence (number 0-1),
  quarantine_reasons (list of strings).

Quarantine (low actionability) if the issue already has an open PR, is a
feature request / design discussion, or implies an API/semantics change that
needs an RFC.

ISSUE #{number}
Labels: {labels}
Title: {title}

Body:
{body}
"""


class LLMTriager:
    """Triage via a language model, falling back to heuristics on any failure.

    The model must return JSON. We merge its fields onto a heuristic baseline so
    that even a partial/garbled response yields a complete, valid record. The
    bucket is *derived* from the merged fields here (not taken from the model)
    to keep bucketing policy in one place.
    """

    def __init__(self, cfg: Config, llm: LLM) -> None:
        self.cfg = cfg
        self.llm = llm
        self._fallback = HeuristicTriager(cfg)

    def triage(self, issue: Issue, related: list[int] | None = None) -> TriageResult:
        baseline = self._fallback.triage(issue, related)
        prompt = TRIAGE_PROMPT.format(
            number=issue.number,
            labels=", ".join(issue.labels),
            title=issue.title,
            body=(issue.body or "")[:6000],
        )
        try:
            raw = self.llm.complete(prompt)
            data = _extract_json(raw)
        except Exception:
            return baseline
        if not isinstance(data, dict):
            return baseline

        merged = baseline
        merged.source = "llm"
        for key in (
            "in_scope",
            "has_repro",
            "fix_type",
            "difficulty",
            "suspected_subsystems",
            "root_cause_hypothesis",
            "actionability",
            "confidence",
            "quarantine_reasons",
        ):
            if key in data and data[key] is not None:
                setattr(merged, key, data[key])
        merged.actionability = max(0, min(100, int(merged.actionability)))
        merged.confidence = max(0.0, min(1.0, float(merged.confidence)))
        merged.bucket = _bucket_from_result(merged, self.cfg)
        return merged


def _bucket_from_result(result: TriageResult, cfg: Config) -> str:
    if result.quarantine_reasons:
        return BUCKET_NEEDS_DECISION
    if not result.has_repro:
        return BUCKET_NEEDS_INFO
    if result.in_scope and result.actionability >= cfg.ready_score_threshold:
        return BUCKET_READY
    return BUCKET_NEEDS_INFO


def _extract_json(raw: str) -> Any:
    """Parse JSON from a model response, tolerating code fences/extra prose."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        # Drop an optional language tag on the first line.
        if "\n" in raw:
            first, rest = raw.split("\n", 1)
            if first.strip().lower() in ("json", ""):
                raw = rest
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start : end + 1]
    return json.loads(raw)
