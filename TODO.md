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

- **`sail-subs` reliability on this repo: two failed attempts, possible
  `papers/*.pdf` context-bloat cause** (2026-09-03). Both real attempts
  to delegate `002-bootstrap-knowledge-docs.md` via `sail-subs` failed
  (see that spec's Decision Log entries #2-#4 for full detail) — not
  from any pipeline/environment problem (setup commands passed, repo
  checkout worked both times), but from the worker itself: attempt 1
  never engaged with the task at all (generic "I don't see a question"
  response, 0 edits); attempt 2 explored the repo properly (`ls`,
  `git log`, read `papers/README.md`) but consumed **271,074 input
  tokens** doing it and then produced incoherent output, 0 edits.
  **Hypothesis, not confirmed:** `papers/` holds several multi-MB PDFs,
  git-tracked, so present in any full-repo checkout — if the worker's
  file-reading swept PDF content into context as raw text (rather than
  skipping binaries or extracting only what's needed), that would
  plausibly explain both the token blowout and a fast/cheap default
  model's coherence collapse under that much junk context.
  - To test: try a `sail-subs` delegation on a task that explicitly
    avoids touching `papers/` (or on a smaller test repo with no large
    binaries) and see if it behaves normally. If confirmed, either
    keep `papers/` out of future delegation checkouts somehow (unclear
    if `sail_delegate`'s `paths` parameter can *exclude* rather than
    only declare ownership) or reconsider whether `sail-subs` is
    reliable for this repo at all until that's fixed upstream.
  - Not investigated further as of this note — `002` was executed
    directly instead once two failures established a real, not
    infra-level, problem.
