# Architecture

## Top-level layout

Three sibling areas at the repo root:

- `go-style-engine/` — the actual code: Python, PyTorch, pytest.
- `papers/` — a local PDF mirror of reference papers (Chessformer,
  Maia/Maia-2/Maia-3), with its own `papers/README.md` explaining why
  each one is kept and what it's cited for.
- `convo-style-engine/` — currently empty. Placeholder for a future
  dialogue/coaching-text counterpart to the Go-playing work; not yet
  started.

## Inside `go-style-engine/`

- `engine/` — the library: `goboard.py`/`gotypes.py`/`zobrist.py`/
  `scoring.py` (the rules engine, ported from `dlgo/`), `mcts.py`
  (AlphaZero-style search + the `StyleKnobs` coaching-knob layer),
  `network.py` (the CNN net), `encoder.py`/`token_encoder.py` (board
  encoders), `token_transformer.py` (the Transformer/GAB net),
  `experience.py` (self-play data collection).
- `selfplay.py`, `train.py`, `trace_move.py`, `run_generations.sh` —
  CLI entry points.
- `tests/` — pytest suite; see `docs/testing.md` for the index.
- `docs/` — **this repo has two `docs/` directories.** This one
  (`go-style-engine/docs/`) holds the real technical design
  documents. The one this file lives in (root `docs/`, alongside
  `AGENTS.md`) holds the night-shift agent-knowledge stubs — this
  file, `testing.md`, `domain.md`, `style-guide.md`,
  `common-pitfalls.md`. Don't confuse the two when routing.

## Two network architectures, both intentional

`engine/network.py` (`GoZeroNet`) and `engine/token_transformer.py`
(`TokenTransformerNet`) both exist on purpose — not leftover cruft
from an abandoned rewrite. `GoZeroNet` is the older, simpler
AlphaZero-style CNN baseline. `TokenTransformerNet` is the newer
Transformer/GAB line of research.

**Both implement the same external contract**, which is the single
most important architectural fact for extending either one:

- Encoder: `.encode(game_state)`, `.decode_move_index(idx)`,
  `.num_moves()`.
- Model: `.predict(state_tensor, device) -> (priors, value)`.

Because of this shared contract, `engine/mcts.py`'s `ZeroAgent` (the
search/coaching-knob layer) works with either net interchangeably, no
modification needed — swapping architectures is just swapping which
encoder/model pair gets passed in.

## Where the real design history lives

`go-style-engine/docs/board-specification.md` is the primary design
document for the token/Transformer line — long (10 sections plus
subsections `8a`-`8e`), and it already contains the full empirical
history (three staged positional-info mechanisms, each built and
validated, not assumed). Don't duplicate its content here; read it
directly for anything about `BoardSpec`, `TokenEncoder`,
`TokenTransformerNet`, or GAB.

Three more docs under `go-style-engine/docs/`:

- `from-transformer-to-trans-go-former.md` — a from-zero explanation
  of the original Transformer architecture and what changed to get to
  this repo's board-token model.
- `mcts-trace-walkthrough.md` — a from-zero explanation of how the
  CNN engine picks a move.
- `style-without-strength-loss.md` — the design doc for the
  `StyleKnobs` coaching-knob mechanism (see `docs/domain.md` for how
  this relates to the Transformer's `pos_mode` variants — they are
  not the same kind of thing).

## `run_generations.sh`

Lives at `go-style-engine/`'s root. Runs an iterative self-play/train
loop: generation N's self-play is guided by generation (N-1)'s trained
checkpoint instead of one-shot random-net generation. Meant to be run
unattended, typically off-laptop (a remote Sail devbox, or delegated) —
see `Specs/003-run-multigeneration-selfplay-loop.md`.
