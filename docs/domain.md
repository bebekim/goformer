# Domain

## Baduk/Go vocabulary

For an agent with no prior Go knowledge:

- **Board** — a grid of intersections (9x9, 13x13, or 19x19 here;
  parameterized, see `go-style-engine/docs/board-specification.md`).
- **Intersection** — one point on the grid where a stone can be
  placed.
- **Stone** — a piece, black or white, placed on an intersection.
- **Group** — a set of same-color stones connected by adjacency.
- **Liberty** — an empty intersection adjacent to a group; a group
  with zero liberties is captured.
- **Capture** — removing an opponent's group when it reaches zero
  liberties.
- **Ko** — a rule preventing immediately recapturing in a way that
  would repeat the prior board position.
- **Pass** — a legal move that plays no stone.
- **Komi** — a score bonus given to White to compensate for Black's
  first-move advantage.
- **Self-play** — the engine playing games against itself (or variants
  of itself) to generate training data.
- **MCTS** — Monte Carlo Tree Search, the move-selection algorithm
  `engine/mcts.py` implements.
- **Policy head / value head** — a network's two outputs: which move
  looks good (policy), and who's winning (value).
- **Checkpoint** — saved trained network weights (a `.pt` file).
- **Epoch** — one full pass through a training dataset.

## What this project is actually for

This is architecture research toward eventually supporting
personality-conditioned play: idols/characters with distinct playing
styles at equal skill, and coaching that visibly, measurably changes
behavior over time — the actual game mechanic this is all in service
of. **As of this doc, none of the personality/coaching work has been
built.** Everything in `go-style-engine/` so far is board-
representation and training-architecture research (does a token/
Transformer encoding work at all, does it avoid known failure modes) —
upstream of, not yet connected to, that goal. See
`Specs/007-style-without-strength-loss-validation.md` for the
highest-priority unexecuted work that actually tests the core
mechanic.

## Two style-conditioning mechanisms — don't conflate them

- **`StyleKnobs`** (`engine/mcts.py`) — search-time re-ranking among
  near-value-equal branches (complexity/variance preference, breadth,
  risk tolerance). **Net-agnostic** — works identically with
  `GoZeroNet` or `TokenTransformerNet`, since both honor the same
  `ZeroAgent` contract (see `docs/architecture.md`). This is the
  actual coaching/personality mechanism the project's core mechanic
  depends on — see `go-style-engine/docs/style-without-strength-loss.md`.
- **`TokenTransformerNet`'s `pos_mode`** (`absolute`/`relative`/
  `both`/`gab`/`gab_absolute`) — an **architecture-level** concern:
  how the token-Transformer represents board position internally. It
  is not a style-conditioning mechanism and has nothing to do with
  personality — see `go-style-engine/docs/board-specification.md` §6.

These are two unrelated axes that happen to both use the word "style"
loosely in conversation. `StyleKnobs` conditions *behavior*;
`pos_mode` conditions *architecture*.
