# Night Shift Agent Loop

## Operating Logic

Use the Tractatus-style proposition logic from `~/repositories/.night-shift/HARNESS_PRINCIPLES.md`.

- Treat ready specs as the mission boundary.
- Treat repository docs, tests, templates, and guardrails as functional leverage.
- Report output against both mission fit and functional quality.
- Surface missing environment context instead of guessing strategy or product intent.
- Convert repeated corrections into docs, checks, specs, skills, or guardrails.

## Operating Rules

- Read `AGENTS.md` first.
- Do not work on specs whose filename starts with `draft-`.
- Work on one task at a time.
- Before implementation, write a short testing plan.
- Add or update tests before or alongside implementation.
- Run relevant tests.
- Run typecheck, lint, and other mechanical checks when available.
- Commit each completed task separately.
- Use detailed commit messages for human review.
- Update `CHANGELOG.md` after each completed behavior or configuration change.
- Record unrelated observations in `TODO.md`; do not opportunistically fix unrelated issues.
- If blocked, write a concise note in `TODO.md`, commit useful diagnostic work, and move to the next ready task.

## Task Selection

1. Inspect `Specs/`.
2. Ignore files starting with `draft-`.
3. Pick the highest-priority spec that declares `State: ready`, or the first ready spec if no priority is stated.
4. The spec is the contract: behavior, acceptance criteria, and test expectations live in the spec file itself.
5. Complete it fully before starting another.
6. When a spec is complete, set `State: done` in the spec and commit it with the work.

## Completion Report

When there are no ready tasks left, produce a concise final report containing:

- completed specs
- commits created
- tests run
- unresolved blockers
- follow-up TODOs
