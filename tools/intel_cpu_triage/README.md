# Intel CPU Issue Triage Toolkit

An AI-assisted workflow for a small (3–5 engineer) team resolving Intel
CPU / oneDNN issues in [`pytorch/pytorch`](https://github.com/pytorch/pytorch),
**without** flooding maintainers with low-quality PRs.

This toolkit implements the plan as runnable code. It is **standard-library
only** (no third-party dependencies, no PyTorch build required) so it can run
anywhere, including a scheduled GitHub Action.

## Guiding principles

1. **Reviewer trust is the scarcest resource.** Optimize for PR *acceptance
   rate* and reviewer goodwill, not PR count.
2. **AI is an accelerator, not an autonomous submitter.** A human engineer owns
   every issue and every PR. AI does triage, reproduction, drafting, and
   tracking; humans decide what ships.

Two **mandatory human gates** are baked into the flow:
- **Gate #1 (grooming):** an engineer must `pull` an issue onto the board.
- **Gate #2 (internal review):** a second engineer signs off before any PR is
  opened upstream, subject to the rate-limit policy.

## Phases → modules

| Phase | Purpose | Module |
|------:|---------|--------|
| 1 | Ingest relevant issues into a local SQLite DB (incremental, watermark-based) | `ingest.py`, `github_client.py` |
| 2 | AI triage & scoring into 3 buckets + duplicate detection | `triage.py`, `dedup.py` |
| 3 | Tracking board state machine (Kanban) | `board.py` |
| 5/6 | Rate-limiting (open-PR cap + per-reviewer budget) | `ratelimit.py` |
| 7 | Health metrics + feedback examples for prompt tuning | `metrics.py` |
| — | Single source of truth | `db.py` |
| — | Orchestration | `cli.py` |

> The DB is the single source of truth. Upstream GitHub labels are never
> modified by this toolkit.

## The three triage buckets (Phase 2)

- **`ready_to_work`** — in-scope, has a repro, scores above threshold.
- **`needs_info_repro`** — missing a reproduction or below threshold.
- **`needs_maintainer_decision`** — *quarantined*: already has an open PR, is a
  feature request / design discussion, or implies an API/semantics change that
  needs an RFC.

Quarantining is what keeps the team from chasing work that would generate
unwelcome PRs.

## Usage

```bash
# Phase 1 — incremental ingestion (needs a GITHUB_TOKEN env var for higher rate limits)
python -m tools.intel_cpu_triage.cli --config tools/intel_cpu_triage/config.example.json ingest

# Phase 2 — score & bucket (heuristic by default; see "Wiring an LLM" below)
python -m tools.intel_cpu_triage.cli triage

# Phase 3 — review the AI-proposed list, then a human pulls work onto the board
python -m tools.intel_cpu_triage.cli ready
python -m tools.intel_cpu_triage.cli pull 12345 alice           # human gate #1
python -m tools.intel_cpu_triage.cli advance 12345 repro_confirmed
python -m tools.intel_cpu_triage.cli board

# Phase 6 — before opening an upstream PR, check the policy (human gate #2)
python -m tools.intel_cpu_triage.cli pr-check --reviewer some-maintainer

# Phase 7 — health metrics and feedback examples
python -m tools.intel_cpu_triage.cli metrics
python -m tools.intel_cpu_triage.cli feedback
```

The board enforces legal transitions
(`backlog → investigating → repro_confirmed → fix_in_progress → pr_open →
merged`); you cannot skip repro confirmation to jump straight to a PR.

## Wiring an LLM (optional)

`triage` uses the deterministic `HeuristicTriager` by default so it works
offline. To use a real model, write a tiny driver that constructs
`triage.LLMTriager(cfg, llm)` where `llm` has a `complete(prompt) -> str`
method returning JSON. The toolkit:

- merges the model's fields onto the heuristic baseline (so a partial or garbled
  response still yields a complete, valid record), and
- derives the bucket itself from the merged fields, keeping bucketing policy in
  one place.

Feed `feedback` output back into your prompt as few-shot "good vs. bad
candidate" examples to improve triage precision over time.

## Scheduling

Copy `intel_cpu_triage.yml.example` into your **private** tracking repo at
`.github/workflows/intel_cpu_triage.yml`. It runs ingestion + triage daily and
caches the SQLite DB between runs to keep ingestion incremental. (Shipped with a
`.example` suffix so it never auto-runs from a PyTorch fork.)

## Configuration

See `config.example.json`. Key knobs: scope/signal/negative labels, content
keywords, `max_open_external_prs`, `max_prs_per_reviewer`, and
`ready_score_threshold`.

## Tests

```bash
python -m unittest tools.intel_cpu_triage.tests.test_toolkit
```

## What this toolkit deliberately does NOT do

- It never auto-posts comments or auto-opens PRs on `pytorch/pytorch`.
- It never modifies upstream labels.
- It does not advance API-change / feature work without a maintainer decision.
- It does not optimize for PR volume.
