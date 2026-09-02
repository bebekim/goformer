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
   see §10.
2. **Stage 2 (done -- built and empirically validated) -- static
   relative-position bias**, gather-based, not `maia3`'s dense matmul.
   `RelativePositionBias` + `TokenTransformerNet(pos_mode='relative')`,
   injected via `nn.TransformerEncoder`'s `mask` argument -- the same
   mechanism `maia3`'s `MHA.forward` uses for `attn_mask`. Cheap
   (learned params = `nheads*(2N-1)²`, e.g. 8×625=5,000 at N=13) and
   interpretable; **§7's validation found it does NOT help the value
   head learn at this scale, and has a working (untested) hypothesis
   why** -- read §7 before building on this stage further. See §10 for
   the code.
3. **Stage 3 (done -- built and empirically validated) -- dynamic
   GAB**, mean-pooled/cheap variant (`gab_per_square_dim=0`, per the
   two smallest shipped Maia3 models), shipped as two options from day
   one per §7's recommendation: `pos_mode='gab'` (alone) and
   `'gab_absolute'` (GAB + companion absolute PE). **§8's validation
   found GAB's content-dependence really does avoid stage 2's
   value-pooling failure (both variants fit the value target better
   than every earlier stage) -- but both show real policy overfitting
   at this project's small self-play data scale**, which its ~420K-582K
   param cost (dominated by `gab_weight`) makes more likely, not less.
   Read §8 before training this further. See §10 for the code.

Each stage replaces only the positional-info module; `TokenEncoder`
(§2-3) and the policy/value heads are untouched across all three.

## 7. Stage 2 empirical validation

Before building stage 3 (the expensive one), stage 2 was checked
against real data rather than assumed to help. This needed real
infrastructure first: `selfplay.py`/`train.py` were hardcoded to
`ZeroEncoder`/`GoZeroNet` and had no way to run the token-transformer at
all. Both scripts gained a `--net-type {cnn,token}` / `--pos-mode`
switch (`build_encoder_and_model` in `selfplay.py`, `build_model` in
`train.py`), and `train.py` gained a seeded `--val-fraction` split with
held-out loss reporting (`evaluate()`), so two runs sharing `--seed`
get the *identical* split — necessary for `absolute` vs. `relative` to
be compared on the same held-out examples, not different ones.

**Two real bugs surfaced by this exercise, before the comparison was
even possible to run**, both fixed and covered by regression tests:

1. `TokenEncoder` was missing a `num_moves()` method entirely (only
   `BoardSpec.num_moves` existed as a property). `engine/mcts.py`'s
   `ZeroAgent.select_move` only calls `self.encoder.num_moves()` when a
   collector is attached — a path none of stage 1/2's `ZeroAgent`
   smoke tests exercised (they never attach one), but one
   `selfplay.py` always exercises (to save experience shards). Fixed;
   see `TokenEncoder.num_moves()` and
   `tests/test_token_transformer.py::test_select_move_with_collector_attached`.
2. A real numerical bug in PyTorch's `nn.MultiheadAttention` fused
   "fastpath" kernel: in `.eval()` mode, with a non-causal float
   `attn_mask` (exactly what `RelativePositionBias` — and GAB later —
   inject), it can silently return NaN for certain *trained* weight
   values. Reproduced directly: does not happen in `.train()` mode, and
   does not happen with a freshly initialized model — only after real
   gradient steps, which is why none of stage 2's original unit tests
   (all forward-pass-only) caught it. Fixed with
   `torch.backends.mha.set_fastpath_enabled(False)` (the documented
   PyTorch workaround) at the top of `token_transformer.py`, applied
   globally since the bug lives in a shared backend and stage 3 will
   hit the same code path. Covered by
   `tests/test_relative_position_bias.py::test_eval_mode_after_training_step_produces_no_nan`,
   verified to actually fail without the fix before being accepted.

**The experiment.** One shared self-play dataset, 9x9 (cheaper than
13x13 for a same-day check), `TokenEncoder`/`TokenTransformerNet`
(`pos_mode='absolute'`, random-init — the data-generating net's mode
doesn't bias the comparison since final game scores come from real
Go scoring regardless of policy quality, only the MCTS visit-count
*targets* reflect a weak/noisy search from a random net, same caveat
`mcts-trace-walkthrough.md` already notes for the CNN case), root
Dirichlet noise + temperature 1.0 for real game diversity (default
knobs are fully deterministic — a first probe at default knobs
produced 3 byte-identical games):

```
selfplay.py --board-size 9 --games 30 --rounds-per-move 50 \
  --net-type token --pos-mode absolute \
  --dirichlet-epsilon 0.25 --temperature 1.0 --seed 0 \
  --out runs/stage2_validation/selfplay \
  --save-experience runs/stage2_validation/shared_exp.npz
# -> 30 games, 1,506 positions, 141.8s (4.7s/game)
```

Then `train.py --net-type token --pos-mode {absolute,relative}` on
that *identical* file, `--seed 0 --val-fraction 0.2` (same split both
runs), otherwise-default hyperparameters:

| epoch | absolute: train / val value_loss | relative: train / val value_loss |
|---|---|---|
| 1 | -- | 0.845 / 0.810 |
| 5 | 0.653 / 0.583 | 0.754 / 0.761 |
| 8 | 0.293 / 0.185 | -- |
| 12 | 0.092 / 0.110 | 0.770 / 0.776 |
| 20 (relative only, extended run) | -- | 0.716 / 0.753 |
| 40 (relative only, extended run) | -- | 0.711 / 0.760 |

(Policy loss for both hovers in a similar 3.38-3.46 range, well above
converged, for both modes — neither variant shows a clear policy
advantage at this budget; the value loss is where they diverge sharply.)

**Result: `absolute` clearly learns the value target; `relative`
plateaus and does not, even given more than 3x the training (40 vs 12
epochs).** This isn't a bug — checked directly: `relative_bias.gate`'s
values move substantially away from their zero initialization (mean
|value| 0.089 after training), so gradients are real and flowing, it
just isn't converging to a useful value estimate.

**Working hypothesis:** `TokenTransformerNet`'s value head pools with
`h.mean(dim=1)` — content-based mean pooling, which is itself
permutation-invariant. `pos_mode='absolute'` gives every token an
explicit "where am I" signal added directly into its own content vector
before any attention happens, so that signal survives pooling trivially.
`pos_mode='relative'` never touches token *content* with position at
all — it only reweights how tokens attend to each other — so a
location-differentiated global summary has to be *built* by attention
layers moving positional information into content over several layers,
which 4 shallow layers and 1,506 positions may simply not be enough to
accomplish.

**Confirmed directly**, via a fourth diagnostic option added for exactly
this purpose — `pos_mode='both'` (`TokenTransformerNet`, not a real
stage, see its module docstring): run absolute PE and relative bias
*simultaneously*, same shared dataset, same seed/split, same 40-epoch
budget as the `relative`-alone extended run:

| epoch | absolute (12ep) val_value | relative (40ep) val_value | **both (40ep) val_value** |
|---|---|---|---|
| 1 | -- | 0.810 | 0.827 |
| 5 | 0.583 | 0.761 | 0.522 |
| 10 | -- | -- | 0.104 |
| 12 | 0.110 | 0.776 | 0.087 |
| 20 | -- | 0.753 | 0.094 |
| 40 | -- | 0.760 | **0.049** |

`both` doesn't just fix the plateau — it ends up *better* than
`absolute` alone (0.049 vs. `absolute`'s best ~0.076-0.11 around epoch
11), with `relative`'s adjacency prior apparently adding something on
top once content actually carries a location signal. Policy loss shows
the same pattern (train drops 3.99→3.27 under `both`, vs. staying
~3.38-3.46 the whole time under `relative`-alone), though with a mild
val-loss uptick after ~epoch 20 worth watching, not chasing yet.

**What this settles and what it doesn't.** It settles the mechanism:
content never carrying a position signal, not the relative-bias idea
itself, was the actual bottleneck — `RelativePositionBias` is fine, it
just isn't sufficient alone for this pooled value-head design. It does
**not** settle whether stage 3 (GAB) has the same problem. Production
Maia3 ships GAB *alone* (`use_absolute_pe: False`, §6's table) and its
value head pools the same way (`x = self.last_ln(x.mean(dim=1))`,
`models.py:396`) — yet works, at scale, on 169M human games. The
difference our finding turns on: `RelativePositionBias`'s bias is
*board-content-independent* — literally identical every forward pass
regardless of what's on the board — so with `pos_mode='relative'`
alone, the network's only per-example signal is raw token content run
through ordinary QK attention, with no location grounding anywhere.
GAB's bias, by contrast, is generated *from* the board content each
forward pass (§6), so even without a separate absolute PE, the
attention pattern itself already varies per-position in a
content-coupled way — it may not hit the same wall. That's a real,
open, untested distinction, not a reason to assume GAB is safe.

**Scope caveats, stated plainly:** one seed, one board size, one small
architecture size, no hyperparameter sweep. Real, reproducible evidence
at the scale it was run, not a definitive verdict on either mechanism.

**Recommendation for stage 3, given this:** build GAB with `pos_mode`
supporting a `'gab'` + companion-absolute-PE combination available from
day one (the same `'both'`-style pattern that fixed this), rather than
assuming pure-GAB will "just work" here the way it does for Maia3 —
we're in a materially different regime (small CPU-only AlphaZero
self-play from scratch, not supervised imitation on 169M games), and
this section is direct, cheap-to-obtain evidence that pooled heads in
this codebase are sensitive to exactly that assumption. Validate
pure-GAB vs. GAB+absolute-PE the same way this section validated
`relative` vs. `both`, before trusting either.

## 8. Stage 3 empirical validation

Same protocol as §7, and the same recommendation it made, followed
through: `pos_mode='gab'` and `'gab_absolute'` trained on the
*identical* `runs/stage2_validation/shared_exp.npz` from §7 (no new
self-play needed -- same 9x9 dataset, same `--seed 0 --val-fraction
0.2` split, same 40-epoch budget as the `relative`/`both` runs, so all
five variants are directly comparable):

```
train.py --experience runs/stage2_validation/shared_exp.npz --board-size 9 \
  --net-type token --pos-mode {gab,gab_absolute} --seed 0 --val-fraction 0.2 --epochs 40 \
  --out-checkpoint checkpoints/stage3_val_{gab,gab_absolute}.pt
```

| epoch | absolute (12ep) val_value / val_policy | both (40ep) val_value / val_policy | gab (40ep) val_value / val_policy | gab_absolute (40ep) val_value / val_policy |
|---|---|---|---|---|
| 1 | -- | 0.827 / -- | 0.721 / 3.480 | 0.860 / 3.613 |
| 5 | 0.583 / -- | 0.522 / -- | 0.053 / 3.460 | 0.181 / 3.416 |
| 10 | -- / -- | 0.104 / -- | 0.029 / 3.526 | 0.055 / 3.438 |
| 12 | 0.110 / 3.376 | 0.087 / -- | 0.027 / 3.597 | 0.034 / 3.459 |
| 20 | -- | 0.094 / -- | 0.023 / 3.730 | 0.029 / 3.564 |
| 40 | -- | **0.049** / 3.428 | 0.019 / **4.159** | **0.014** / 3.797 |

Model sizes at these defaults (`d_model=64`, `gab_gen_size=64`,
`gab_intermediate_dim=64`): `gab` 581,699 params, `gab_absolute`
582,851 -- versus `absolute`/`relative`/`both`'s roughly 140-150K.
`gab_weight` alone (419,904 of that) is the dominant term, exactly the
O(N²·N²) cost flagged in §6.

**Finding 1 (confirms the open question §7 left): GAB's content-
dependence does let it avoid stage 2's value-pooling failure.** Both
GAB variants fit the value target better than every earlier
stage/diagnostic, including `both` -- `gab_absolute` reaches
val_value_loss 0.014, roughly 3.5x better than `both`'s 0.049. Even
`gab` *alone*, with no separate absolute PE at all, comfortably beats
`both` (0.019). This is the real, structural difference §7 predicted
but left untested: unlike `RelativePositionBias` (identical bias every
forward pass regardless of board content), GAB's bias is generated
*from* the board each time, so it carries a location-coupled signal
into the attention pattern even without ever touching token content
directly -- confirmed directly here, not merely by analogy to
Maia3's chess results.

**Finding 2 (a new problem, not predicted): both GAB variants overfit
on the policy objective, which none of the earlier stages did at this
budget.** `val_policy_loss` for `gab` climbs from 3.46 (epoch 5) to
4.16 (epoch 40) while its *train* policy loss keeps falling (3.27→2.68)
-- classic overfitting, and worse than `both`'s much milder policy
drift (3.37→3.43 over the same 40 epochs). The likely cause: 1,205
training examples is very little data for a ~582K-param net whose
capacity is dominated by one O(N²·N²) matrix specifically built to be
expressive per-position -- exactly the capacity/data mismatch §6 and
the `GeometricAttentionBias` docstring warned about in the abstract,
now observed concretely. `gab_absolute` overfits somewhat less than
`gab` alone (val_policy 3.80 vs. 4.16 at epoch 40) but still clearly
more than `both` or `absolute`.

**Net assessment:** GAB is a real, validated improvement for the value
objective and a real, validated regression for the policy objective, at
this data scale. Neither earlier stage showed this split -- `absolute`/
`relative`/`both` were all small enough that policy and value moved
together. This means GAB isn't simply "better" or "worse" than the
earlier stages; it trades one failure mode (§7's value-pooling problem)
for a different one (policy overfitting from excess capacity), and
which trade is worth it depends on how much self-play data is actually
available before this gets used for real. `gab_absolute` is the safer
of the two GAB variants -- equal-or-better on both metrics than `gab`
alone -- so it's the better default if GAB is used at all right now.

**Scope caveats, same as §7:** one seed, one board size, one small
dataset, no hyperparameter sweep, no regularization attempted (weight
decay, a smaller `gab_gen_size`, or more self-play data are the obvious
next levers, in that rough order of cheapness, if GAB's policy
overfitting needs fixing before it's trusted for real training).

### 8a. Regularization follow-up

`--dropout` (0.0 → `TokenTransformerNet`'s existing but previously
unwired `dropout` constructor arg) and `--weight-decay` (new; switches
`train.py`'s optimizer from `Adam` to `AdamW` — identical to `Adam` at
`weight_decay=0.0`, but the *correct* decoupled implementation once
it's nonzero, since plain `Adam(weight_decay=...)` applies L2 through
the gradient, which interacts badly with Adam's adaptive per-parameter
rates) were added to `selfplay.py`/`train.py` specifically to test
whether cheap regularization narrows finding 2's gap. Same shared
dataset/split/budget as the rest of §8, `pos_mode='gab_absolute'`:

| config | val_policy_loss @ epoch 40 | val_value_loss @ epoch 40 |
|---|---|---|
| baseline (no regularization) | 3.797 | 0.014 |
| `--weight-decay 1e-4` only | 3.811 (no improvement — within noise) | 0.014 |
| `--dropout 0.1` only | 3.624 | 0.017 |
| `--dropout 0.1 --weight-decay 1e-4` | 3.615 (best) | 0.016 |

**Dropout is doing essentially all of the work here; `weight_decay=1e-4`
is inert at this scale** (barely distinguishable from the unregularized
baseline, possibly noise). Combining both barely improves on dropout
alone. Dropout meaningfully narrows finding 2's gap (3.80 → 3.62) at
negligible value-loss cost (0.014 → 0.016) -- worth turning on as a
default when training `gab`/`gab_absolute` -- but at a fixed epoch 40
does **not** close it: 3.62 is still clearly worse than `both`'s 3.428
or plain `absolute`'s 3.376. §8b below revisits this "fixed epoch 40"
framing itself -- it turns out to be part of the problem, not just the
measurement.

### 8b. Two more levers: weight_decay magnitude, and gab_gen_size

**Weight_decay magnitude sweep**, `dropout=0.1` held fixed, same
protocol, `1e-4` (§8a) through `3e-2` -- nearly three orders of
magnitude:

| weight_decay | val_policy_loss @ 40ep | val_value_loss @ 40ep |
|---|---|---|
| 1e-4 (§8a) | 3.615 | 0.016 |
| 1e-3 | 3.616 | 0.029 |
| 3e-3 | 3.609 | 0.019 |
| 1e-2 | 3.645 | 0.029 |
| 3e-2 | 3.592 | 0.029 |

All five sit in a tight, noise-level band (3.59-3.65) -- no trend
across nearly 300x in magnitude. **Confirmed, not just suspected:
weight_decay is not an effective lever here, at any reasonable
magnitude.** Not worth the added complexity of tuning it further.

**`gab_gen_size` sweep** (the template-library size behind
`gab_weight`'s dominant O(N²·N²) cost), `dropout=0.1` held fixed,
default `gab_gen_size=64` (582,851 params) vs. 32 (358,083) vs. 16
(248,771):

| gab_gen_size | val_policy_loss @ 40ep | val_value_loss @ 40ep | **best-epoch val_policy** | **best-epoch val_value** |
|---|---|---|---|---|
| 64 (default) | 3.615-3.797 (§8/§8a) | 0.014-0.016 | ~3.41 (epoch ~10) | ~0.045 (epoch ~10-12) |
| 32 | 3.654 | 0.023 | ~3.39 (epoch ~10) | ~0.067 (epoch ~10) |
| 16 | 3.566 | 0.032 | **3.370 (epoch 13)** | 0.055 (epoch 13) |

This is a real, not noise-level, effect: smaller `gab_gen_size`
consistently narrows the val_policy_loss gap. But the more important
finding came from pulling `gab_gen_size=16`'s **full** per-epoch trace,
not just the epoch-40 snapshot every other comparison in this doc uses:

```
epoch  9: val_policy=3.3768  val_value=0.1284
epoch 11: val_policy=3.3702  val_value=0.0889
epoch 13: val_policy=3.3704  val_value=0.0546   <- best joint point
epoch 17: val_policy=3.3974  val_value=0.0377
epoch 20: val_policy=3.4275  val_value=0.0410
   ...                                             (climbs from here)
epoch 40: val_policy=3.5662  val_value=0.0317
```

**val_policy_loss bottoms out at epoch 13 (3.370) — better than `both`
mode's own reported number (3.428) — before climbing back up exactly
the way every other GAB configuration in this doc does.** Every
epoch-40 comparison so far in §8/§8a, including the ones judging GAB
against `both`/`absolute`, compared numbers *after* each config had
already started overfitting to different degrees -- a fixed epoch
isn't a fair comparison when different configs overfit at different
rates. The fair comparison is each config's own best epoch, i.e. what
early stopping on the val set would actually select in a real training
pipeline (something none of this doc's runs have used until now).

**Revised conclusion:** at its own best checkpoint,
`gab_gen_size=16`+`dropout=0.1` `gab_absolute` is competitive with
`both` on policy (3.370 vs. ~3.43) *while still ahead of it* on value
in that same epoch region (0.055-0.09 vs. `both`'s own ~0.08-0.10 in
the comparable epoch range) -- not a clean sweep on every axis
simultaneously (the value-loss optimum and the policy-loss optimum
don't land on exactly the same epoch), but no longer the "GAB wins
value, loses policy" trade §8's Finding 2 first framed. §8's Finding 1
(GAB's content-dependence avoiding stage 2's failure) and the mechanism
behind Finding 2 (excess capacity relative to 1,205 training examples)
both still hold -- reducing `gab_gen_size` directly targets that
mechanism, which is why it works better than either regularization
lever tried in §8a.

**Recommendation, updated:** `gab_absolute` with `dropout≈0.1` and
`gab_gen_size` reduced to 16-32 (not the default 64), trained with
early stopping on a val split (not a fixed epoch count), is the
practical default this doc now points to -- not `weight_decay`, and not
training gen_size=64 to a fixed epoch. **Still untried**, in order of
what's likely to matter most: more self-play data (the actual
underlying constraint -- 1,205 examples is small regardless of
architecture tuning), and applying this same "best-epoch, not fixed-
epoch" re-comparison retroactively to `relative`/`both`/`absolute`
too, which this section did not do and which could change §7's
conclusions somewhat, though probably not their direction (`relative`
alone plateaus rather than overfits, so an epoch-40 read was fair for
that specific comparison; `both`/`absolute` showed much milder epoch-40
drift than GAB to begin with, so the correction is likely smaller for
them than it was for GAB).

### 8c. Real early stopping in `train.py`

§8b's headline result -- `gab_gen_size=16`'s true minimum at epoch 13,
found only by reading a full per-epoch trace by hand -- wasn't
reproducible as a normal training run: nothing in `train.py` actually
selected or kept that checkpoint. `--early-stopping-patience N` (0 =
disabled, the old behavior) fixes this: whenever `--val-fraction > 0`,
every epoch's checkpoint is compared against the best combined val loss
(`val_policy_loss + val_value_loss`) seen so far, the best one is saved
to `<out-stem>.best.pt`, and `--out-checkpoint` itself becomes a copy
of that best checkpoint at the end of training -- **not** the last
epoch trained, which is the point: this doc's whole finding was that
the last epoch is usually worse. Tracking-and-keeping-the-best is
always on with a val split; `--early-stopping-patience` only adds
stopping *early* (compute savings) once that many epochs pass with no
improvement. `train.py`'s `main()` was refactored to take an optional
`argv` list so this could be tested in-process
(`tests/test_train_early_stopping.py`) instead of only by shelling out.

Re-running §8b's exact `gab_gen_size=16` config through the new flag
reproduced the hand-found result exactly: best epoch 13, `val_policy_loss=3.3704`,
`val_value_loss=0.0546` -- and, given a generous 40-epoch budget with
`--early-stopping-patience 3`, training actually stopped at epoch 16
(3 non-improving epochs after epoch 13), saving 24 of the 40 epochs'
compute. This is the piece that makes §8b's finding something you get
by default from running `train.py` normally, not something that
required hand-reading logs after the fact.

### 8d. More self-play data

§8b/§9 flagged more self-play data as "likely the largest remaining
lever," untested. Tested now: `selfplay.py`'s resumability extended the
identical 9x9 run from 30 to 90 games (same board, same
`--dirichlet-epsilon 0.25 --temperature 1.0 --seed 0` diversity
settings, just `--games 90` on the same `--out` directory -- it skipped
the 30 already-completed games and generated 60 more), yielding 4,417
positions, **3.7x** the 1,205 used throughout §7/§8. Same
`gab_absolute`, `dropout=0.1`, `gab_gen_size=16` config, now with real
early stopping (`--early-stopping-patience 5`, `--epochs 60`):

| dataset | best epoch | val_policy_loss | val_value_loss | combined |
|---|---|---|---|---|
| 1,205 positions (§8b/§8c) | 13 | 3.3704 | 0.0546 | 3.4251 |
| 4,417 positions (this section) | 14 | 3.3939 | 0.0504 | **3.4444** |

**3.7x more data did not improve the combined loss -- if anything it's
marginally worse** (3.4444 vs. 3.4251), though the two numbers are
close enough that this could be within run-to-run noise rather than a
real effect (different games went into each dataset; this wasn't
re-run at multiple seeds to check). Value loss improved slightly
(0.0546 → 0.0504); policy loss got slightly worse (3.3704 → 3.3939).
Training did stop earlier relative to the larger epoch budget offered
(epoch 19 of 60, vs. epoch 16 of 40 for the smaller dataset) -- more
data did *not* buy more useful training time here.

**This complicates, rather than confirms, §9's "more data is the
biggest remaining lever" expectation.** A plausible explanation, not
verified here: §8b's fix already matched capacity down to roughly what
1,205 examples could support (`gab_gen_size` 64 → 16); having done
that, the bottleneck may no longer be data volume at all, so more data
at the *same* now-reduced capacity doesn't reveal further headroom --
data and capacity likely need to scale together, not data alone. A
second plausible factor, already flagged as a caveat back in §7: every
self-play game in this doc is generated by a randomly-initialized,
never-trained net, so the MCTS visit-count targets it produces are
inherently weak/noisy (per `mcts-trace-walkthrough.md`'s own framing)
-- more *games* from that same weak generator may just be more of the
same limited signal, not new signal, regardless of `gab_gen_size`.
Neither explanation is distinguished from the other here; doing so
would need a `gab_gen_size` sweep repeated at both data volumes -- §8e.

### 8e. Resolving §8d: sweeping `gab_gen_size` at both dataset sizes

`gab_gen_size ∈ {16, 32, 64}` × both datasets (1,205 and 4,417
examples), same `dropout=0.1`, same `--early-stopping-patience 5`,
same seed:

| `gab_gen_size` | 1,205 examples: best epoch / combined val_loss | 4,417 examples: best epoch / combined val_loss |
|---|---|---|
| 16 | 13 / **3.4251** | 14 / **3.4444** |
| 32 | 11 / 3.4450 | 10 / 3.4537 |
| 64 | 8 / 3.4423 | 7 / 3.4637 |

**This decides it: the capacity/data co-scaling explanation is
refuted.** If data and capacity needed to scale together, `gen_size=64`
(or 32) should have closed the gap with 3.7x more data -- instead
`gen_size=16` wins at *both* volumes, and every `gen_size` got slightly
*worse* with more data, not better (16: 3.4251→3.4444; 32:
3.4450→3.4537; 64: 3.4423→3.4637 -- the same direction, three times, a
real pattern rather than one noisy run). **The weak-generator-signal
explanation is what the data actually supports**: since every game in
this doc comes from an MCTS search guided by a randomly-initialized,
never-trained net, more games from that same generator add more of the
same limited signal, not new signal -- so more data doesn't help, and
mildly hurts by giving each config more low-quality examples to overfit
to before early stopping catches it.

A second, independent pattern in the same table corroborates this:
best-epoch gets *earlier* as `gen_size` grows, at both data volumes
(13→11→8, and 14→10→7). Bigger nets overfit *faster* regardless of how
much data they're given -- exactly what capacity mismatch predicts, and
not what you'd expect if data volume were the binding constraint.

**Updated recommendation:** stop chasing more data from this
random-net self-play generator -- it isn't the lever. `gab_gen_size=16`
+ `dropout=0.1` + early stopping is the settled practical default from
this whole §8 investigation. The one lever that could plausibly still
move this number is generating self-play data from an *actually
improving* net (a real iterative AlphaZero-style gen0→train→gen1→train
loop, per `style-without-strength-loss.md`'s own §4a-4b, never run in
this doc) rather than one-shot random-net self-play -- untested here,
and a materially bigger undertaking than anything else in §8.

## 9. Explicitly deferred (not part of this spec)

- **The follow-ups §8a-§8e surfaced**: `weight_decay` is settled
  (inert, don't pursue further); `gab_gen_size=16` is settled as the
  practical default, confirmed best at *both* data volumes tested
  (§8e), not just 16 vs. 32/64 -- still not swept below 16 or checked
  at N=13/19; early stopping on the val split is now real (`train.py
  --early-stopping-patience`, §8c); more self-play data is settled too
  (§8d/§8e) -- it doesn't help, because the self-play generator itself
  (a randomly-initialized, never-trained net) is the actual bottleneck,
  not data volume or capacity mismatch. The one lever that could still
  move this number -- a real iterative self-play/train loop instead of
  one-shot random-net generation (`style-without-strength-loss.md`
  §4a-4b) -- now has real tooling (`run_generations.sh`, verified
  end-to-end by its own smoke test, including two real bugs it
  surfaced and fixed) but has not actually been run for a real
  multi-generation experiment yet, so its result is still unknown.
  §8b's own suggestion of re-reading `relative`/`both`/
  `absolute`'s §7 numbers at their best epoch rather than a fixed one
  still hasn't been done -- now cheap to do, since
  `--early-stopping-patience` exists. The full per-square GAB variant
  (`gab_per_square_dim>0`, maia3-23m/79m's setting), more expressive
  and more expensive still, remains untried too. And §7's own
  still-open follow-up (value-head pooling design more broadly, beyond
  the specific GAB-vs-relative comparison §7/§8 already ran). Tracked
  as follow-up work on top of this same `BoardSpec`/`TokenEncoder`/
  `TokenTransformerNet` foundation, not a re-derivation of it.
- **Group/liberty/eye features.** Deliberately left out so the "does
  attention learn group topology from raw adjacency" question stays
  open and testable, per §1 -- this is exactly what GAB (stage 3) was
  meant to probe, and remains open: §8's validation measured value/
  policy loss, not whether the model's attention actually organized
  around groups/liberties -- that would need the kind of interpretability
  analysis Chessformer's own paper does, not attempted here.

## 10. What's built alongside this doc

- `engine/token_encoder.py` — `BoardSpec` dataclass + `TokenEncoder`,
  producing `(num_tokens, token_dim)` arrays from a `GameState`,
  parameterized exactly as above.
- `tests/test_token_encoder.py` — parametrized over `board_size in
  (9, 13, 19)` to make "readily changeable" a checked property, not a
  claim; also checks the move-index convention matches `ZeroEncoder`'s
  at 13x13 (index compatibility, not import coupling — the two encoders
  don't share code, only the row-major index convention).
- `engine/token_transformer.py` — all three stages from §6, selected via
  `TokenTransformerNet(pos_mode=...)`: `RowColPositionalEmbedding`
  (`'absolute'`), `RelativePositionBias` (`'relative'`), and
  `GeometricAttentionBias` (`'gab'`, or `'gab_absolute'` for GAB +
  companion absolute PE — the combination §7/§8 recommend using), all
  feeding the same input projection → `nn.TransformerEncoder` → spatial
  policy head + pooled pass/value heads. Matches `GoZeroNet.predict`'s
  external contract (`predict(state_tensor, device) -> (priors, value)`)
  on purpose, so it drops into `ZeroAgent` (`engine/mcts.py`)
  unmodified — same encoder/model contract, different internals.
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
  integration coverage as stage 1 with `pos_mode='relative'`, plus the
  eval-mode-after-training NaN regression test from §7, plus
  `TestTokenTransformerNetBothMode` for the `pos_mode='both'`
  diagnostic that confirmed §7's pooling hypothesis.
- `tests/test_geometric_attention_bias.py` — stage 3: standalone
  `GeometricAttentionBias` checks (shape, that `gab_weight` dominates
  param count as expected, and the defining property that makes it
  *dynamic* — different board content gives different bias, unlike
  `RelativePositionBias`), then shape/gradient/ZeroAgent integration
  coverage for both `'gab'` and `'gab_absolute'`, plus the same
  eval-after-training NaN regression check.
- `selfplay.py` / `train.py` — both gained `--net-type {cnn,token}` and
  `--pos-mode {absolute,relative,both,gab,gab_absolute}`
  (`build_encoder_and_model` / `build_model`), plus `--gab-gen-size` /
  `--gab-intermediate-dim` and `--dropout` (wires `TokenTransformerNet`'s
  previously-unwired `dropout` constructor arg to a flag), so the
  token-transformer path can actually be self-played and trained, not
  just unit-tested in isolation. `train.py` additionally gained a
  seeded `--val-fraction` split with held-out loss reporting
  (`evaluate()`) — the infrastructure §7/§8's validations needed and
  used — plus `--weight-decay` (switches the optimizer from `Adam` to
  `AdamW`, identical at the 0.0 default; see §8a) and
  `--early-stopping-patience` with real best-checkpoint tracking/saving
  (see §8c). `train.py`'s `main()` now takes an optional `argv` so it
  can be called in-process from tests.
- `tests/test_train_early_stopping.py` — best-checkpoint tracking (with
  and without a val split), that `--out-checkpoint` really does match
  the epoch with the lowest combined val loss (not just that a file
  exists), the `--early-stopping-patience`-without-`--val-fraction`
  error guard, and that patience actually stops training before
  `--epochs` completes.

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
