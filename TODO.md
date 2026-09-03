# TODO

Use this for follow-up work, blockers, and unrelated observations found during Night Shift runs.

- Keep actionable items specific.
- Promote ready work into `Specs/`.

## Needs Clarification

Questions only a human can answer. Night Shift marks these `needs-clarification`
and moves on; morning review answers them.

## Follow-ups

Work items and observations. Promote ready ones into `Specs/` using
`docs/templates/spec-template.md`.

- **`sail-delegate` MCP: stale `CLAUDE_PROJECT_DIR` after a repo move**
  (2026-09-03). This repo was moved mid-session from
  `~/repositories/individual/deep-learning/trans-go-former` to
  `~/repositories/individual/trans-go-former` (see
  `001-adopt-sail-delegation.md`'s history) — the `sail` plugin was
  installed *after* that move, but its `sail-delegate` MCP server
  still resolved the old, now-nonexistent path when
  `sail_delegate`/`sail_fanout` were called (error: `could not launch
  delegation runner: [Errno 2] No such file or directory:
  '.../deep-learning/trans-go-former'`). Passing an explicit
  `project_path` to work around it is rejected outright ("project_path
  conflicts with the host-provided CLAUDE_PROJECT_DIR"). Fix: `/plugin
  reload` (or a fresh Claude Code session) so the MCP server re-reads
  the current project root. If `sail_delegate` fails with that exact
  error again after a reload, this is still open — check whether
  `sail-update`'s "reload plugins or start a new session" guidance
  covers MCP server env refresh specifically, not just skill/version
  updates.
  - Status: **resolved** — `/plugin reload` (2026-09-03) fixed it. The
    retried `sail_delegate` call (dispatching
    `002-bootstrap-knowledge-docs.md`) launched successfully (ran past
    120s doing real work, moved to background, instead of the
    immediate stale-path error from before the reload).
