# Specs

Specs are the source of truth for Night Shift work.

- `draft-*` files are ignored.
- A spec is selectable only when it is non-draft **and** declares `State: ready`.
  Non-draft alone is not enough — `needs-clarification` and `blocked` specs must
  not be picked up.
- States are declared as `State: <value>` in the spec front matter:
  `draft` | `needs-clarification` | `blocked` | `ready` | `done`.
- Non-draft specs with no `State:` are legacy specs; review and assign a state
  before handing them to Night Shift.
- `State: done` means the implementation exists in the current tree, relevant
  checks passed, and the result has survived human review. A done spec is a
  record, not an active task.
- If a done behavior is later removed, pruned, or found incomplete, change the
  spec back to `State: ready` or create a new ready follow-up spec.
- Keep specs bounded enough for one reviewable commit.
- Prefer one to three ready specs per night.

Use the root template at `~/repositories/.night-shift/templates/spec-template.md` when drafting a new spec, or copy the structure manually if this repository is used outside the parent workspace.

## Local conventions for this repo

Two additions on top of the shared template, specific to `trans-go-former`:

- **Sequential numbering.** Filenames are prefixed `NNN-` (`001-`, `002-`,
  ...), assigned in the order the spec was drafted, so the queue can be
  read top-to-bottom as a history, not just a set of independent files.
  Numbering doesn't imply execution order — `State`/`Priority` still
  govern what Night Shift picks up next — it's for humans tracing how
  the repo's specs accumulated. Renumbering an existing spec is fine;
  update this file's own references if you do.
- **Decision Log section.** Every spec ends with a `## Decision Log`
  table recording what was proposed, rejected, or accepted while
  defining the spec's scope — including decisions made *before* the
  spec was written, if the spec is retroactively documenting work that
  already happened. This is the mechanism that keeps
  `HARNESS_PRINCIPLES.md`'s "repository knowledge is the system of
  record" rule honest: a decision that only exists in chat history
  doesn't count as decided. Format:

  ```markdown
  ## Decision Log

  | # | Proposed | Status | Why |
  |---|---|---|---|
  | 1 | <option> | Accepted / Rejected / Pending | <reason> |
  ```
