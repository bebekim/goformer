# MCTS Trace Walkthrough — what actually happens when the engine picks a move

**Audience:** someone who has never read search/ML code.
**Prerequisites:** none. Every term is defined when first used, and again
in the glossary at the bottom.
**Source material:** one instrumented run of `trace_move.py` (2026-09-01),
13x13 board, 30 search rounds, randomly initialized network.
**Code referenced:** `engine/mcts.py` (line numbers as of commit `b1133f4`).

---

## 1. The 30-second version

The engine decides a move by **playing out 30 imaginary futures** and
keeping score. Each future ("round") does four things:

1. **Descend** — walk down the tree of futures explored so far, always
   picking the branch that looks most promising *right now*.
2. **Expand** — when the walk reaches the edge of what's been explored,
   add one new position (one hypothetical move deeper).
3. **Evaluate** — ask the neural network to judge that new position:
   "who's winning, and what moves look good here?"
4. **Back up** — carry that judgment back up the tree, flipping the sign
   at each level (my gain is your loss), so every branch on the path
   updates its track record.

After 30 rounds, the engine looks at the *statistics accumulated at the
root* and commits to a move. That final choice is where the **style
knobs** act: among moves that are statistically near-tied, pick by
personality (complexity-seeking, risk tolerance) instead of by raw score.

Everything below is this same story told slowly, with real numbers.

---

## 2. The cast — five pieces of code, one decision

| Piece | File | Job in one sentence |
|---|---|---|
| Rules | `engine/goboard.py` | What moves are legal; what a position *is*. |
| Encoder | `engine/encoder.py` | Translates a position into an 11×13×13 grid of 0s and 1s. |
| Network | `engine/network.py` | A function with 474,557 tunable numbers; grid in, judgments out. |
| Search | `engine/mcts.py` | Plays out futures and accumulates statistics (this document). |
| Knobs | `engine/mcts.py:48-57` | Per-trainee personality: how to choose among near-equal moves. |

The network's two judgments have names used throughout:

- **policy** — a score for every legal move: "how promising does this
  look?" (first impressions, no thinking)
- **value** — one number in [-1, +1]: "how good is this position for
  the side to move?" (+1 = winning, -1 = losing)

---

## 3. The experiment setup

`trace_move.py` plays a fixed 4-move opening in the corner, then traces
the engine's 5th move decision:

```
black (4,4) → white (3,3) → black (4,3) → white (3,4) → black to move
```

The network is **randomly initialized** — it has never seen a game. This
is deliberate: with a brain whose opinions are meaningless, anything the
search produces is the *machinery* talking, not the brain. It makes the
machinery visible.

`TracedAgent` subclasses `ZeroAgent` and only adds print statements to
the two decision points. No behavior is changed.

---

## 4. Step zero: the root gets its first judgment

Before any searching, the engine evaluates the current position once
(`select_move` → `create_node`, `mcts.py:149,191-208`):

```
[setup] root position evaluated by net: value=-0.057 for black
```

The net's first impression: this position is slightly bad for black
(-0.057, barely below even). With a random net this number is noise —
remember it anyway, because **-0.057 shows up everywhere below**.

The root node now holds 170 branches (one per legal move: 169
intersections + pass), each storing:

- `prior` — the policy score the net gave that move (~0.006 each here)
- `visit_count` (`n`) — how many futures have gone down this branch (0)
- `total_value` / `total_value_sq` — running sum of backed-up verdicts,
  and their squares (the squares let us compute **variance** later)

---

## 5. One round, in slow motion (round 1)

### 5a. Selection — which branch deserves a look? (`mcts.py:210-222`)

Every branch is scored with the **PUCT formula** (`mcts.py:220`):

```
score(move) = Q  +  c · prior · √(total visits at this node) / (n + 1)
              │      └────────── "explore" ──────────┘
              └ "exploit"
```

Three ingredients:

- **Q** (`mean_value`) — the branch's track record: average of all
  verdicts backed up through it so far. Unvisited branch → Q = 0.
- **prior** — the net's first impression of this move.
- **explore** — a bonus that is *large for unvisited branches* and
  *shrinks as the branch gets visited* (the `n+1` denominator), while
  *growing slowly over time* (the `√N` numerator). It is the engine's
  curiosity: "we haven't tried this one yet, and it's getting embarrassing."
- **c** (knob, = 2.0) — how loudly curiosity speaks relative to evidence.

Round 1, every branch: Q=0, n=0, prior≈0.006:

```
round 1:
  pick (r 4, c 6)  Q=+0.000 prior=0.006 n=0 explore=+0.013
```

All branches tie at Q=0, so the winner is whoever has the biggest
`prior · √1 / 1` — i.e. **the net's favorite first impression**. That was
(4,6). Note what this means: *the very first playout is just the policy
talking; search hasn't added anything yet.*

### 5b. Expansion + evaluation (`mcts.py:163-164, 191-193`)

(4,6) has no child yet, so the engine creates one: apply the move,
encode the new position, ask the net:

```
      expand (r 4, c 6): net values this leaf -0.057; backed up to parent as +0.057
```

The net says: the position *after black plays (4,6)* is worth **-0.057
for white** (the side to move there).

### 5c. Backup — the zero-sum relay (`mcts.py:166-173`)

The leaf verdict must be translated into each ancestor's perspective.
The rule is one line: **flip the sign at every level.**

- White moving into a -0.057 position = good news for black, so the
  (4,6) branch at the root records **+0.057**.
- Had the descent gone deeper, the next level up would flip it again.

After round 1, branch (4,6) has: n=1, Q=+0.057.

---

## 6. Round 2 — the seesaw, and why it's the whole point

```
round 2:
  pick (r 4, c 6)  Q=+0.057 n=1 explore=+0.009     ← root level: black's choice
    pick (r 12, c 5)  Q=+0.000 n=0 explore=+0.013  ← one level down: white's reply
      expand (r 12, c 5): net values this leaf -0.058; backed up to parent as +0.058
```

Three things happened:

1. **(4,6) won the root selection again.** Its explore bonus shrank
   (0.013 → 0.009 — the `n+1` at work), but Q=+0.057 carried it.
   This is *exploitation*: the search returns to what worked.
2. **The descent continued one level deeper.** Inside the (4,6) child
   node, the same PUCT game plays out — but now it's **white** choosing,
   maximizing *white's* score. White's favorite first impression was
   (12,5).
3. **Backup flipped twice.** The new leaf is -0.058 for black (the side
   to move there) → +0.058 recorded on white's (12,5) branch → flipped
   again to **-0.058** recorded on black's (4,6) branch at the root.

Now look at (4,6)'s track record: (+0.057 - 0.058) / 2 ≈ **-0.001**.

Read that slowly. After round 1, (4,6) looked like a small win for
black. After round 2 — after the search simulated **white's best
response** — it looked like nothing. That is the entire value of search
over first impressions: *a move isn't good or bad; a move plus the
opponent's best reply is.* MCTS is the machine that prices in the reply.

---

## 7. The metronome — why the trace alternates so perfectly

From round 3 onward, the trace falls into a strict rhythm: **odd rounds
open a brand-new root branch; even rounds deepen the branch opened last
round.** Round 3 opens (11,1); round 4 descends into it; round 5 opens
(12,5); round 6 descends... all 30 rounds.

Why? Follow the numbers:

- After a branch's first expansion it has Q≈+0.056, n=1 — the best score
  at the root, so the *next* round descends into it (even round).
- The reply at level 2 flips another ≈-0.057 into it, so its Q collapses
  to ≈0. Meanwhile every *unvisited* branch's explore bonus has been
  growing with √N (round 3: +0.022, round 5: +0.028, round 9: +0.037...).
  Q≈0 + a small explore loses to 0 + a big explore → the next round
  (odd) opens a fresh branch.
- Repeat forever: two steps forward, the evidence cancels, curiosity
  moves on.

And *why does every leaf say -0.057?* Because the random net has
exactly one opinion: "every position is slightly bad for whoever has to
move." Every backup flips it (+0.057 for the parent), every reply flips
it back (-0.057), and each branch averages to ≈0 after two visits.
The perfectly regular rhythm is the fingerprint of a brain with no
knowledge: it has no reason to prefer any branch, so the explore bonus
is the only force acting, and it acts uniformly.

**When the net is trained, this pattern breaks** — and that's how you'll
know the brain is real: some leaves will evaluate to +0.4, branches
won't collapse after the reply, Q values will spread apart, and the
search will keep returning to a few strong branches instead of
metronoming through all of them.

One more random-brain fingerprint: white's reply is **(12,5) in nearly
every child node**. A net with no knowledge has a fixed location
favorite regardless of the board — like someone who answers "Tuesday"
to every question.

---

## 8. The decision — where the style knobs act

After 30 rounds, `choose_from_root` (`mcts.py:224-273`) looks at the
root statistics. Here is the actual final table (top 10 of 15 visited
branches):

```
      move  prior   n       Q   var  cplx  viable style_score
(r 1, c 2)  0.006   2  -0.000 0.003 0.976     yes      +0.586
(r 1, c 8)  0.006   2  -0.000 0.003 0.976     yes      +0.586
(r 5, c 12) 0.006   2  +0.000 0.003 0.976     yes      +0.586
(r 4, c 6)  0.006   2  -0.001 0.003 0.976     yes      +0.585
...
```

Read it in three steps, mirroring the code:

### 8a. The viable band (`mcts.py:233-235`)

```
viable = moves whose Q is within 0.05 of the best Q
```

The best Q here is +0.000, the worst is -0.001 — the whole field is 50×
*inside* the 0.05 tolerance, so **all 15 visited branches are viable**
(`viable_count: 15` in the diagnostics). The band is doing its job; the
search just found no real differences to work with.

### 8b. The style score (`mcts.py:244-253`)

```
style_score = Q + complexity_weight · cplx + safety_lambda · var
            = Q +       0.6          · cplx +     0.08      · var
```

- **cplx** (complexity) — how "open" the position after this move is,
  measured as the entropy of the child node's policy: if the net sees
  many live options there, it's complex; if one forced line, it's quiet
  (`mcts.py:237-242`). Here every child is ≈0.976 — near-maximally
  "complex," because the random net sees every position as wide open.
- **var** — the variance of the verdicts backed up along this branch:
  did the futures down this line swing (a fight, a gamble) or agree
  (a quiet, predictable line)?

Since every branch has Q≈0, var≈0.003, cplx≈0.976, every style score
lands at ≈0 + 0.6·0.976 + 0.08·0.003 ≈ **+0.586**. Uniform input,
uniform output.

### 8c. The pick (`mcts.py:255-263`)

`temperature=0` means deterministic argmax, so the engine picked the
top style score by a third-decimal margin:

```
chosen move: (r 13, c 2)
diagnostics: viable_count=15, complexity=0.976, variance=0.003,
             think_time_s=0.129, root_value=-0.057
```

**The mechanism executed perfectly and meant nothing.** Every stage of
the knob pipeline — viable band, complexity measurement, style scoring,
selection — ran exactly as designed. It had no signal to amplify. This
is the empirical argument for training before tuning: the knob pipeline
is provably wired up; what it lacks is a brain with real opinions
underneath it.

---

## 9. What changes with a trained brain

Same trace, same code, checkpoint loaded. Predicted differences:

| Number in this trace | Random net (what we saw) | Trained net (what to expect) |
|---|---|---|
| leaf values | always ≈ -0.057 | spread across [-1, +1] |
| Q after 2 visits | ≈ 0 everywhere | real separation between branches |
| root metronome | strict alternation | search camps on a few strong branches |
| `viable_count` | 15 of 15 | typically 1-5 genuinely close moves |
| `cplx` differences | ±0.001 (noise) | quiet moves vs fight-starters separate |
| white's reply | (12,5) everywhere | position-dependent |
| knob decision | third-decimal coin flip | a real stylistic choice with real cost |

Re-run then and compare: `.venv/bin/python trace_move.py` (load the
checkpoint by extending the script's `GoZeroNet` construction with
`load_state_dict`, same as `selfplay.py:160-161`).

---

## 10. Glossary

- **MCTS (Monte Carlo Tree Search):** deciding a move by simulating many
  what-if futures and aggregating their outcomes. "Monte Carlo" = uses
  randomized/sampled evaluation; here the sampling is replaced by the
  network's judgment.
- **round / playout:** one descent–expand–evaluate–backup cycle.
- **node:** a position in the tree of futures. The **root** is the
  current real position. A **leaf** is a node at the unexplored edge.
- **branch:** one legal move out of a node, with its statistics.
- **policy / prior:** the net's per-move first impressions (before any
  search). "Prior" = belief before evidence.
- **value:** the net's who-is-winning estimate, [-1, +1], always from
  the perspective of the side to move in that position.
- **Q (`mean_value`):** a branch's track record — average of all
  verdicts backed up through it. The search's *evidence*, vs the prior's
  *first impression*.
- **n (`visit_count`):** how many rounds have passed through a branch.
- **PUCT:** the selection formula, `Q + c·prior·√N/(n+1)`. Balances
  exploiting evidence (Q) against exploring the untried (the bonus).
- **c:** knob scaling the explore bonus. High c = broad, shallow reading;
  low c = narrow, deep reading.
- **backup:** carrying a leaf's verdict up the tree, flipping sign per
  level (zero-sum).
- **viable set:** root branches whose Q is within
  `equal_value_tolerance` (0.05) of the best Q. The stage on which style
  performs — style never overrides a genuinely better move.
- **complexity (cplx):** entropy of the child position's policy,
  normalized to [0,1]. High = many live options = messy/open game.
- **variance (var):** spread of backed-up verdicts along a branch.
  High = swingy, risky continuation.
- **style knobs (`StyleKnobs`):** per-trainee parameters —
  `rounds_per_move` (think time), `c` (reading breadth),
  `complexity_weight` (mess preference), `safety_lambda` (risk
  preference), `equal_value_tolerance` (what counts as "near-equal"),
  `temperature` (0 = deterministic, >0 = sampled/unpredictable),
  `dirichlet_*` (training-time exploration noise).
- **checkpoint:** a saved file of the network's 474,557 numbers, the
  product of training. "Random init" = a checkpoint that has never
  learned anything.
- **epoch / generation:** one pass of training over the experience
  buffer / one round of self-play-then-train producing `genN.pt`.

---

## 11. Reproduce this document's numbers

```bash
cd research/go-style-engine
.venv/bin/python trace_move.py     # 13x13, 30 rounds, fighter knobs, seed 0
```

Deterministic: same seeds, same trace. Tweak `KNOBS` at the top of the
script to watch a different personality choose; tweak `ROUNDS` to watch
the explore/exploit balance shift.
