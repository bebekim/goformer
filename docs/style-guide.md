# Style Guide

This is a research codebase. Two conventions matter more here than in
a typical application repo:

## Findings live in docs, not just code

Prefer clear, well-commented empirical findings in `.md` design docs
over terse code comments. Every new positional-info mechanism,
training lever, or CLI flag should get:

1. A docstring/comment explaining *why*, not just what (a hidden
   constraint, a workaround, a subtle invariant — not a restatement of
   the code).
2. A test.
3. If it's a genuine empirical finding (not just plumbing) — a section
   in `go-style-engine/docs/board-specification.md` with real numbers,
   following the existing evidence-first pattern (§7, §8a-§8e): a
   claim is backed by an actual run's output, not asserted from
   intuition.

## No enforced formatter/linter

Code follows plain, PEP 8-ish conventions already visible throughout
`engine/` — but nothing currently enforces this mechanically (no
configured `black`/`ruff`/`flake8` or equivalent). Don't invent or
claim a stricter convention than what's actually enforced; if this
changes, update this doc to say so.

## Commit message convention

Explain what was found and why, not just what changed. Recent commits
in this repo (`git log` in the repo root) demonstrate the actual
pattern in practice — e.g. commits documenting an empirical result
lead with the finding, not the diff summary. Follow that pattern for
any commit that represents a real decision or discovery, not just a
mechanical change.
