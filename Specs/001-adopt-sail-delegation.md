# Adopt sail-subs/sail-swarm for unattended delegation; deprecate Sandcastle

Priority: high
State: done

## Problem

This repo was bootstrapped into the shared night-shift baseline
(`Specs/`, `AGENT_LOOP.md`, the ground-loop guard), which assumes
unattended work runs through Sandcastle — `.sandcastle/main.mts`,
spawning `codex exec` as the agent. Two problems with that:

1. **The documentation and the implementation had already diverged.**
   `~/repositories/.night-shift/SANDCASTLE.md` and `SANDBOX_PROFILES.md`
   describe a Docker-sandboxed runner that reads a per-repo
   `.sandcastle/sandbox.json` profile. The actual `main.mts` (read in
   full before this decision) is a "Native Host Runner" — its own
   startup banner says so — that spawns `codex exec` directly on the
   host machine. It never reads `sandbox.json`. There is a `Dockerfile`
   in `.sandcastle/` but nothing in `main.mts` invokes it.
2. **The intended agent (Codex) isn't the one actually in use.** This
   repo's unattended work is done through Claude Code, not Codex, and
   the operator's stated practice is Sail-based delegation
   (`sail-subs`/`sail-swarm`), not Sandcastle at all.

## Desired Behavior

`trans-go-former` delegates unattended, spec-driven work via the `sail`
Claude Code plugin (`sail-subs` for bounded leaf work, `sail-swarm` for
multi-worker campaigns needing scouting first), keeping the rest of the
night-shift discipline (`Specs/`, preflight gating, ground-loop guard,
morning-style review before merging) unchanged. No Sandcastle-specific
files remain in the repo.

## Non-Goals

- Do not modify `.sandcastle/main.mts` itself (shared across every repo
  under `~/repositories/individual/`) to add a Sail execution path. A
  design for that was considered and explicitly rejected — see Decision
  Log #1.
- Do not remove or alter `~/repositories/.night-shift/` (the shared
  framework) beyond the one bug fix in `bootstrap_agent_repo.py`
  unrelated repos also benefit from (Decision Log #3).
- Do not adopt Beads (`bd`) queue tracking — see Decision Log #3.

## Likely Files

- `.sandcastle/` (deleted entirely — `implement-night-shift.md`,
  `preflight-specs.md`, `doctor.md`, `sandbox.json`)
- `.guardrails/` (deleted — empty, Databricks-specific, unused)
- `AGENTS.md` (rewritten: deduplicated, dropped Sandcastle/Beads/
  Databricks references, added a one-line note on the Sail backend)
- `AGENT_LOOP.md` (trimmed: dropped the `<promise>NIGHT_SHIFT_COMPLETE
  </promise>` sentinel, which only `main.mts` consumed)
- `~/repositories/work/scripts/bootstrap_agent_repo.py` (one-line fix,
  shared script, see Decision Log #3)

## Environment

Requires: `sail` CLI authenticated (`sail auth login`, already done
before this spec), the `sail` Claude Code plugin installed
(`/plugin marketplace add sailresearchco/sail-skills` then
`/plugin install sail@sail` — done interactively by the repo owner,
confirmed live via the `sail-delegate` MCP tools and `sail:*` skills
appearing in a subsequent session), and `uv` installed locally (already
present, required by the `sail-delegate` MCP server's launch command).

## Dependencies

None — this is the second spec for this repo (numbered after
`002-bootstrap-knowledge-docs.md` was drafted, but decided and
implemented first; see `Specs/README.md`'s numbering note).

## Edge Cases

- If a future `bootstrap_agent_repo.py --refresh` run re-injects the
  managed baseline block this spec removed from `AGENTS.md`, that's a
  known tradeoff of not keeping the marker-block system (Decision Log
  #4) — resolve by re-applying this spec's cleanup, not by reverting it
  silently.
- `sail-delegate`'s workers run in an isolated project copy, not this
  machine's live `.venv` — any future delegation needs explicit
  `setup_commands` to restore the environment (see
  `go-style-engine/README.md`'s setup instructions), not an assumption
  that dependencies are already installed.

## Test Expectations

- `cd go-style-engine && .venv/bin/python -m pytest -q` — unaffected by
  this spec (docs/config only, no code paths touched); confirmed
  132 passed, 7 xfailed after each step below.

## Acceptance Criteria

- [x] `.sandcastle/` and `.guardrails/` removed.
- [x] `AGENTS.md`/`AGENT_LOOP.md` cleaned of Sandcastle/Beads/Databricks
      references, deduplicated.
- [x] `sail` plugin installed and confirmed live in a session
      (`sail-delegate` MCP tools + `sail:*` skills visible).
- [x] The `bootstrap_agent_repo.py` bug (`--skip-bd-init` not gating
      `docs/beads.md`) fixed at the source, not worked around locally.
- [x] `go-style-engine` test suite unaffected (132 passed, 7 xfailed).
- [x] `CHANGELOG.md` updated.

## Known Risks

Low. This was a documentation/config/tooling change with no code paths
touched. The main residual risk is the one named in Edge Cases: a
future shared-script refresh could reintroduce content this spec
deliberately removed, since the managed-baseline-marker mechanism that
would normally prevent that was itself part of what got dropped.

## Decision Log

| # | Proposed | Status | Why |
|---|---|---|---|
| 1 | Extend `.sandcastle/main.mts` with an opt-in `"type": "sailbox"` execution path — keep Codex as the agent, run it on a Sailbox instead of the host, reusing `sandbox.json`. | **Rejected** | Repo owner: "I no longer use sandcastle" — not a Docker-vs-Sail question, Sandcastle itself is deprecated in practice. Also would have meant risking shared infra (`main.mts` is used by every repo under `~/repositories/individual/`) for a mechanism about to be unused here anyway. |
| 2 | Adopt `sail-skills` (`sail-subs`/`sail-swarm`, Claude Code plugin + `sail-delegate` MCP server) as the delegation mechanism, entirely separate from `main.mts`. | **Accepted** | Matches actual current practice; zero shared-infra risk; every prerequisite (`sail` CLI auth, `uv`) was already in place. |
| 3 | Adopt Beads (`bd`) as the operational queue, per `bootstrap_agent_repo.py`'s default behavior (it ran `bd init` and templated `docs/beads.md` unconditionally, even under `--skip-bd-init`, which was itself a bug — see the script's `bootstrap_repo()`). | **Rejected** | Repo owner: "delete beads, we are not using that." Fixed the underlying bug (`--skip-bd-init` now also gates the `docs/beads.md` template copy) rather than restoring the missing template or working around it locally — a real, narrowly-scoped fix that also benefits any other repo bootstrapped with that flag. |
| 4 | Keep `.sandcastle/*` prompt files and the `AGENTS.md`/`AGENT_LOOP.md` managed-baseline marker block as inert reference material, in case Sandcastle is reintroduced later. | **Rejected** | Repo owner: "delete what's unnecessary." Dead weight with no near-term use given #1/#2; also `AGENTS.md` was literally duplicated (a hand-edited-looking section followed by an almost-identical managed block), which is worse than having nothing — a stale, misleading doc actively contradicts `HARNESS_PRINCIPLES.md`'s legibility principle, an empty/missing one doesn't. |
