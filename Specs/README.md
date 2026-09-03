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
