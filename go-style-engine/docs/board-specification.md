# Board Specification — intersection tokens for the trans-go-former encoder

**Date:** 2026-09-02
**Question:** what should a Chessformer-style (square-token + attention-bias)
encoder look like for Baduk, and how do we keep the board size a config
value rather than a rewrite?
**Answer:** one token per intersection, a minimal 3-state occupancy
alphabet (`empty/self/opponent`) times a short move-history window, plus
a 2D relative-position attention bias. Every shape in the pipeline is
`N = board_size`, one constructor argument. Default `N = 13` to match the
existing engine; `N = 19` and `N = 9` must construct and run with no code
changes, only slower/faster.

This is a **new, alternative encoder** living next to `engine/encoder.py`
(`ZeroEncoder`, 11-plane CNN input), not a replacement. It targets a
future Transformer body (`engine/token_encoder.py` + a GAB-style bias
module, not built yet); the existing `GoZeroNet` CNN and MCTS/style-knob
machinery are untouched. Both encoders can read the same `GameState`
from `engine/goboard.py`.

## 1. Why intersection tokens, and why they're simpler than Chessformer's

Chessformer tokenizes each of the 64 squares by *which of 12 piece types*
occupies it, because chess geometry is piece-dependent (rook lines vs.
knight jumps vs. pawn direction) — see `papers/chessformer-2605.19091.pdf`.
Baduk stones carry no such distinction: every stone obeys the same
4-neighbour adjacency rule, and the interesting structure (groups,
liberties, life/death) is *derived* from connectivity, not stamped on
each token by piece identity. So the per-token alphabet collapses from
12 piece types to 3 occupancy states:

```
{ empty, self, opponent }
```

"self" / "opponent" (not "black" / "white") because the encoder always
works from the perspective of the player to move — same convention
`ZeroEncoder` already uses (`board_tensor[8]`/`[9]` komi-holder flags,
`docs/mcts-trace-walkthrough.md:51`). This lets one network play both
colors and keeps the token alphabet, not the color, doing the work.

No liberty-count planes, no group/eye features, no influence maps in the
baseline. `ZeroEncoder` hand-computes liberty buckets (planes 0-7); the
open question for the token model is whether attention can recover group
topology from raw adjacency + history on its own. Baking liberties in
now would foreclose that experiment. The one exception is the
**ko-illegal marker**, which is not a spatial pattern the model could
learn from a single board snapshot — the existing `ZeroEncoder` computes
it in plane 10 by asking `game_state.does_move_violate_ko(...)`; the
token encoder does the same per-token instead of adding it as a whole
extra channel category.

## 2. Token layout

```
num_tokens(N) = N * N
```

For board_size N, intersections are enumerated row-major, `Point(row=r,
col=c)` for `r, c in 1..N`, matching `ZeroEncoder.decode_move_index` /
`encode_move` exactly — token index `i = N * (r - 1) + (c - 1)`. This
means the token encoder's move-index convention is drop-in compatible
with the existing policy decode/encode helpers; a future policy head
does not need a new index scheme.

Per-token feature vector, before history:

```
x_i = [ is_empty, is_self, is_opponent, is_ko_illegal ]   # 4 dims
```

`is_ko_illegal` is only ever 1 on an empty point, so it's cheap
(matches `ZeroEncoder` plane 10's semantics, just folded into the same
token instead of a separate 13x13 plane). It is only meaningful for the
*current* position (history_depth frame 0) — "is this move illegal for
the player to move right now" isn't well-defined for a hypothetical
replay against an older board, so history frames > 0 leave this flag at
0 rather than compute something that reads like a signal but isn't one.

## 3. History window

Chessformer concatenates the current position with 7 previous ones
(`n=7`) so the model can infer "what just moved" without an explicit
move-history token stream. Same idea here, same default, but exposed as
a parameter (`history_depth`) rather than hard-coded, since it is a
separate knob from board size and 13x13 games are shorter than 19x19
ones — this may warrant retuning later, but that is a training-time
experiment, not a shape constraint.

```
per_token_dim(history_depth) = 4 * (history_depth + 1)
token_features.shape = (N*N, 4 * (history_depth + 1))
```

Default `history_depth = 7` → 32 dims/token at any N.

Boards before game start are encoded as all-empty (matches
`ZeroEncoder`'s implicit behavior — there is no "no history yet" plane;
early-game history frames are just empty boards, which is a legal and
distinguishable state from a stone having been captured).

## 4. Positional information

Two independent things are usually called "position" here — don't
conflate them:

1. **Which intersection is this token** — needed by the policy head to
   turn a token back into a move, and by the model to know it's near an
   edge/corner. Represented the ordinary Transformer way: a learned or
   sinusoidal embedding added to each token, indexed by `(row, col)`.
2. **How do two tokens relate geometrically** — this is the GAB-shaped
   question deferred to a follow-up step (see §6). It is *not* required
   to define the board tokenization itself; the tokens above are a
   valid Transformer input with plain per-token positional embeddings
   alone. Relative-position attention bias is an architecture choice on
   top of this spec, not part of it.

## 5. Parameterization contract

```python
@dataclass
class BoardSpec:
    board_size: int = 13
    history_depth: int = 7

    @property
    def num_tokens(self) -> int:
        return self.board_size * self.board_size

    @property
    def token_dim(self) -> int:
        return 4 * (self.history_depth + 1)

    @property
    def num_moves(self) -> int:
        return self.num_tokens + 1  # + pass
```

Everything downstream is a pure function of `board_size` and
`history_depth`. Changing `board_size` from 13 to 19:

- automatically changes `num_tokens` (169 → 361) and `num_moves`
  (170 → 362) — no code touches these as literals anywhere;
- does **not** carry checkpoints across sizes — a 13x13-trained model's
  token-position embeddings and policy head are shaped for 169 tokens
  and are not meaningfully reinterpretable at 361; retraining from
  scratch is expected, same as the existing engine's stance on 9x9 vs
  13x13 checkpoints;
- multiplies self-attention cost by `(19/13)^2 ≈ 2.1x` per layer
  (`361² / 169² ≈ 4.55x` in the pairwise attention matrix itself, before
  the softmax/matmul cost which scales with token count squared: 169² =
  28,561 vs. 361² = 130,321). This is the concrete number behind
  "different cost regime" in the README — now falsifiable by just
  running the shape at N=19.

## 6. Positional information: three options, staged rollout

Having a per-token feature vector (§2-3) is not yet enough for a
Transformer -- plain multi-head attention is permutation-invariant, so
something has to tell the model which token is which intersection.
Reading the actual `maia3`/`chessformer` reference implementations
(not just the paper) surfaced three real options, each with genuinely
different Baduk-relevant tradeoffs -- this isn't a single obvious
choice, so it's recorded here rather than only in chat history.

| | **Absolute PE** | **Static relative bias** | **Dynamic GAB** |
|---|---|---|---|
| What it is | Learned per-square embedding, added to the token | Learned bias keyed only by (Δrow, Δcol) between query/key | Bias generated fresh per forward pass from the actual board content |
| Content-dependent? | No | No | Yes (rank-bounded by `gen_size`) |
| Sole PE in production Maia3? | No (`use_absolute_pe: False` in every shipped config) | No (coded as `RelativeBias`, shipped nowhere) | **Yes** -- all 4 shipped configs use GAB alone, no separate PE |
| Learned params at N=13 (d_model=64, 8 heads) | 2·13·64 = 1,664 (row+col split) | ~8×625 = 5,000 | ~1.83M (`169²×64`) |
| Learned params at N=19 | 2·19·64 = 2,432 | ~8×1,369 ≈ 11,000 | ~8.3M (`gen_size=64`) to ~16.7M (`gen_size=128`) |
| Interpretability | Low | High -- each head is a plottable Δrow/Δcol heatmap | Medium -- needs transcoder-style analysis |
| Can express Go connectivity (two stones are in the same group)? | No | No -- connectivity isn't a function of geometric offset alone | In principle yes, bounded by rank |

The connectivity point is the Baduk-specific argument, not just an
imported chess one: relative bias and absolute PE are both pure
functions of geometry, and group connectivity in Go is *not* a function
of geometry alone -- two stones five spaces apart are connected only if
an actual chain exists between them. GAB is the only option that is a
function of board content, so it's the only one that could in principle
learn that. It's also the most expensive by a wide margin: at N=13 the
GAB shared weight alone (~1.83M params, `gen_size=64` minimum) is
**~4x larger than the entire existing `GoZeroNet`** (474,557 params,
`docs/mcts-trace-walkthrough.md:42`) -- a real sizing mismatch for a
small CPU-only self-play net, on top of the fact that every published
result for GAB comes from supervised imitation on a 169M-game dataset,
not AlphaZero-style self-play from scratch with far less data. One
concrete implementation risk on the relative-bias side: `maia3`'s
`RelativeBias` is a dense one-hot factorizer matmul (chess: 3.7MB
buffer, negligible); the same trick at N=13 is ~71MB and at N=19 is
~714MB -- fine to reuse the *idea*, not the literal matmul; a gather/
index implementation (`(N², N²)` int buffer, ~114KB at N=13) is the
right port.

Given all that, positional info is being rolled out in stages rather
than committing to one:

1. **Stage 1 (done) -- decomposed row/col absolute PE.**
   Cheapest possible way to get encoder -> transformer -> policy/value
   heads shape-correct and trainable at any `board_size`, before
   spending complexity budget on bias. `engine/token_transformer.py`
   (`RowColPositionalEmbedding`, `TokenTransformerNet(pos_mode='absolute')`);
   see §8.
2. **Stage 2 (done, this branch) -- static relative-position bias**,
   gather-based, not `maia3`'s dense matmul. `RelativePositionBias` +
   `TokenTransformerNet(pos_mode='relative')`, injected via
   `nn.TransformerEncoder`'s `mask` argument -- the same mechanism
   `maia3`'s `MHA.forward` uses for `attn_mask`. Cheap (learned params
   = `nheads*(2N-1)²`, e.g. 8×625=5,000 at N=13) and interpretable;
   tests whether a pure adjacency prior helps over pos-only-by-
   embedding at all. See §8.
3. **Stage 3 -- dynamic GAB**, mean-pooled/cheap variant
   (`gab_per_square_dim=0`, per the two smallest shipped Maia3 models)
   first, only once the self-play data volume justifies training an
   extra ~1.8M-param module. This is the one that can actually test the
   group-topology question above.

Each stage replaces only the positional-info module; `TokenEncoder`
(§2-3) and the policy/value heads are untouched across all three.

## 7. Explicitly deferred (not part of this spec)

- **Stages 2 and 3 above.** Tracked as follow-up work on top of this
  same `BoardSpec`/`TokenEncoder` foundation, not a re-derivation of it.
- **Group/liberty/eye features.** Deliberately left out so the "does
  attention learn group topology from raw adjacency" question stays
  open and testable, per §1 -- this is exactly what stage 3 (GAB) is
  meant to probe.

## 8. What's built alongside this doc

- `engine/token_encoder.py` — `BoardSpec` dataclass + `TokenEncoder`,
  producing `(num_tokens, token_dim)` arrays from a `GameState`,
  parameterized exactly as above.
- `tests/test_token_encoder.py` — parametrized over `board_size in
  (9, 13, 19)` to make "readily changeable" a checked property, not a
  claim; also checks the move-index convention matches `ZeroEncoder`'s
  at 13x13 (index compatibility, not import coupling — the two encoders
  don't share code, only the row-major index convention).
- `engine/token_transformer.py` — stages 1 and 2 from §6, selected via
  `TokenTransformerNet(pos_mode=...)`: `RowColPositionalEmbedding`
  (`'absolute'`) and `RelativePositionBias` (`'relative'`), both feeding
  the same input projection → `nn.TransformerEncoder` → spatial policy
  head + pooled pass/value heads. Matches `GoZeroNet.predict`'s external
  contract (`predict(state_tensor, device) -> (priors, value)`) on
  purpose, so it drops into `ZeroAgent` (`engine/mcts.py`) unmodified —
  same encoder/model contract, different internals.
- `tests/test_token_transformer.py` — stage 1 shape/gradient checks
  parametrized over board size, the `predict()` contract, and an
  end-to-end integration smoke test that plugs `TokenEncoder` +
  `TokenTransformerNet` into the real `ZeroAgent` MCTS harness and plays
  moves on a 5x5 board.
- `tests/test_relative_position_bias.py` — stage 2: standalone
  `RelativePositionBias` checks (shape, the quadratic-not-quartic
  learned-param count, the defining relative-bias property that
  identical (Δrow,Δcol) offsets get identical bias regardless of where
  on the board they occur), then the same shape/gradient/ZeroAgent
  integration coverage as stage 1 with `pos_mode='relative'`.

---
*Key files:* `engine/encoder.py` (prior art, CNN plane encoding),
`engine/goboard.py` (board-size-agnostic rules, reused as-is),
`engine/token_transformer.py`, `docs/style-without-strength-loss.md`
(the MCTS/style-knob layer this does not touch),
`papers/chessformer-2605.19091.pdf`, `papers/README.md`. Local reference
implementations read while writing §6:
`~/repositories/individual/deep-learning/maia3/maia3/models.py`
(`RelativeBias`, `MHA._sq_bias`, `model_registry.py`'s shipped configs)
and `~/repositories/individual/deep-learning/chessformer/chessformer.py`
(the earlier, simpler integer-embedding-per-square model, predating GAB).
