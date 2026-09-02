# From "Attention Is All You Need" to trans-go-former — what we built and why

**Audience:** someone who knows a Transformer is "the thing behind ChatGPT"
but hasn't traced through what actually changed to make it read a board
instead of a sentence.
**Prerequisites:** none. Every term is defined where it's first used, and
again in the glossary at the bottom.
**Code referenced:** `engine/token_encoder.py`, `engine/token_transformer.py`
(this repo); `chessformer.py` in
`~/repositories/individual/deep-learning/chessformer` (the older, simpler
reference model); `maia3/models.py` in
`~/repositories/individual/deep-learning/maia3` (the production
Chessformer/GAB implementation).

---

## 1. The 30-second version

The original Transformer (2017) reads a *sentence* — a 1D sequence of
word tokens — and was built to translate it into another sentence.
Chessformer (2026) reads a *chessboard* — the 64 squares, all at once,
no sequence order — and predicts a move. Our `TokenTransformerNet` reads
a *Baduk board* — up to 361 intersections — and predicts a move. Same
core mechanism (self-attention) all three times; what changes at each
step is **what counts as a token**, **how a token knows where it is**,
and **what shape the output needs to be**. This doc walks through each
change and says exactly why we made it.

---

## 2. What the original Transformer actually does

Introduced in *Attention Is All You Need* (Vaswani et al., 2017) for
machine translation. Two pieces matter for everything downstream:

### 2a. Self-attention

Every token looks at every other token and asks "how relevant are you
to me right now?", then blends their information weighted by that
relevance:

```
Attention(Q, K, V) = softmax( Q Kᵀ / sqrt(d_k) ) V
```

Each token produces a **query** (what am I looking for), a **key**
(what do I offer), and a **value** (what information do I actually
carry) — three learned linear projections of the token's embedding.
`softmax(QKᵀ/√d_k)` turns "how well does my query match your key" into
a weight distribution; that distribution mixes the values.

**Multi-head** attention just runs several of these in parallel with
smaller dimensions each, so different heads can specialize (one head
tracks local structure, another tracks something else), then
concatenates and projects the results back to full size.

This mechanism does not care what a "token" is. That's the whole reason
it transplants to chess and Go at all — nothing about `softmax(QKᵀ/√d)`
assumes English words or board squares. Every net discussed in this doc
uses this exact formula, unmodified, for the QK/V part. Everything
below is about what gets fed in and read out around it.

### 2b. Why it needs a positional encoding at all

Attention alone is **permutation-invariant** — if you shuffle the input
tokens, you get the same set of outputs, just shuffled the same way.
For a sentence, word order is meaning ("dog bites man" ≠ "man bites
dog"), so the original paper adds a fixed sinusoidal signal to each
token's embedding before the first attention layer:

```
PE(pos, 2i)   = sin( pos / 10000^(2i/d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )
```

This is a fixed (not learned) function of a token's position in the
sequence, added directly to its embedding. It's literally implemented,
unchanged, in the older reference model at
`chessformer.py:6-21` (`class PositionalEncoding`) — a straight port,
even though chess squares aren't a sentence.

### 2c. Encoder-decoder, because translation needs to *generate* a sequence

The original Transformer has two stacks: an **encoder** that reads the
source sentence, and a **decoder** that generates the target sentence
one word at a time, attending back to the encoder's output at each
step. That two-stack design exists because translation's output is
itself a variable-length sequence that has to be produced token by
token.

None of the models in this doc need that. A move prediction isn't a
sequence to generate — it's one classification over a fixed set of
possible moves, computed from a single fixed-size board. So Chessformer,
Maia-3, and our `TokenTransformerNet` are all **encoder-only**: one
stack that reads the whole board, then a small head on top reads out an
answer directly. This isn't a Go/chess-specific
trick — it's the same move Vision Transformer (ViT, 2020) made for
images, and it shows up in the code we read: `maia3/models.py`
literally names its config fields `dim_vit`, `mlp_ratio`
(`model_registry.py:37-38`) — vocabulary borrowed straight from ViT,
not from the original NLP Transformer. **This whole model family is
architecturally closer to "ViT for board games" than to "the
translation Transformer."**

---

## 3. Step one: text tokens → board-square tokens (Chessformer)

The first real adaptation, made by Chessformer, is what counts as a
"token."

| | Original Transformer | Chessformer |
|---|---|---|
| Token | One word/subword | One board square (64 total) |
| Token content | A learned embedding, looked up by word ID | Which of 12 piece types occupies that square (one-hot), across several past board states |
| Sequence order | Meaningful (word order = meaning) | Arbitrary (squares aren't ordered) — but *identity* still matters |

Because token identity (which square) matters even though sequence
*order* doesn't, Chessformer still needs positional information — just
not the sinusoidal kind, because squares form a 2D grid, not a 1D
sequence. The older reference model (`chessformer.py:36-38`) handles
this with two small learned embedding tables instead of the sinusoidal
formula:

```python
self.x_embedding = nn.Embedding(8, d_model)   # which file (column)
self.y_embedding = nn.Embedding(8, d_model)   # which rank (row)
...
combined_emb = board_emb + x_emb + y_emb      # chessformer.py:67
combined_emb = combined_emb * math.sqrt(self.d_model)  # chessformer.py:70
```

That `sqrt(d_model)` scaling on the last line is, itself, a direct
carryover from the original paper (section 3.4) — one more piece of
"attention is all you need" surviving unchanged all the way through.

Chessformer's more advanced contribution — **GAB** (Geometric Attention
Bias) — replaces this fixed positional-embedding approach with a
*dynamic* one: instead of adding a position signal to each token once,
generate an entire attention bias matrix fresh from the current board
content, every forward pass, and inject it directly into the attention
computation rather than into the input. We read the real implementation
in `maia3/models.py:95-118` (`MHA._sq_bias`) — the mechanism the
`board-specification.md`'s §6 pros/cons table calls "Dynamic GAB." We
have **not** built this yet; see §5 below for exactly what we did build
instead, and why in that order.

Chessformer also changes the *output* shape: instead of a distribution
over a word vocabulary, it needs "move piece from square i to square
j," so its policy head is a `64×64` outer-product score between a
per-square "from" projection and a per-square "to" projection
(`maia3/models.py:373-376`), plus extra logic for pawn promotion. This
detail will matter in §4 — it's the one place Go and chess genuinely
diverge in what the *output* needs to look like, not just the input.

---

## 4. Step two: chess squares → Baduk intersections (this repo)

Same move (tokenize the board), different board, and a few things
about Baduk make the adaptation simpler in some ways and different in
others.

### 4a. The token alphabet gets much smaller

Chess has 12 distinct token identities per square (6 piece types × 2
colors) because different pieces move differently — a rook and a bishop
occupying the same square mean structurally different things to the
model. Go stones don't have that distinction: every stone obeys the
same adjacency rule, and the interesting structure (which stones are
connected into a group, how many liberties a group has) is *derived*
from occupancy, not stamped onto the token by piece identity. So our
token alphabet collapses to three states:

```
{ empty, self, opponent }
```

(plus one `is_ko_illegal` flag — see `docs/board-specification.md` §1-2
for the full reasoning, including why we deliberately did **not** bake
liberty counts or group features into the token, unlike the CNN
`ZeroEncoder` which does compute those by hand.)

`engine/token_encoder.py`'s `TokenEncoder.encode()` is the direct
analogue of `tokenize_board()` in `maia3/dataset.py:16-29` — same idea
(one-hot occupancy per square, flipped to the current player's
perspective, concatenated across a short history window), just a
3-state alphabet instead of chess's 12-state one, and parameterized by
`board_size` instead of hardcoded to 8.

### 4b. The board gets much bigger, and has to stay a parameter

Chess is always 8×8 (64 tokens); every model we read hardcodes `64`
directly in the source (`self.bias = nn.Parameter(torch.zeros(64,
d_model))`, `gab_shared_weight = ... (64*64, gen_size)`, etc.) — that's
a legitimate choice for them, since chess only ever has one board size.
Baduk doesn't: this project's whole premise requires 13x13 today and
possibly 19x19 later, without a rewrite. So everywhere Chessformer/Maia3
write a literal `64`, our code reads `spec.num_tokens` off a `BoardSpec`
dataclass instead (`engine/token_encoder.py`). This is the single
biggest structural difference from the reference code, and it's why
`tests/test_token_encoder.py` and `tests/test_token_transformer.py`
both run every check at N=9, 13, **and** 19 — proving the parameterization
actually holds, not just asserting it does.

### 4c. The policy head gets *simpler*, not harder

This is the one place Go is architecturally easier than chess. A chess
move is "piece moves from square i to square j" — inherently a *pair*
of squares, hence Chessformer's `64×64` from/to score matrix. A Go move
is just "place a stone at intersection i" — one square, no pair. So our
policy head (`engine/token_transformer.py`, `self.policy_head =
nn.Linear(d_model, 1)`) is a single scalar per token, no outer product,
no promotion-logic special case. The only addition Go needs that chess
doesn't is a **pass** move, which isn't tied to any square — handled by
reading one extra logit off a separately pooled (mean-over-tokens)
representation, alongside the value head, rather than adding a 362nd
sequence position (which would have broken the clean `(num_tokens,
num_tokens)` shape that stage 2/3 attention-bias work needs later).

### 4d. The value head is an AlphaZero addition, not a Transformer one

Translation has no concept of "who's winning." Chess/Go engines do, and
need it for search (MCTS backs up exactly this number). Both Maia-3 and
our net add a small head that pools the token sequence into one vector
and reads out a scalar — this piece has nothing to do with the original
Transformer paper at all; it's bolted on from the AlphaZero lineage
(same role as `GoZeroNet`'s value head in `engine/network.py`). One
implementation difference worth flagging: Maia-3 predicts a 3-way
win/draw/loss classification (`fc_value`, 3 output logits;
`models.py:326`), because chess has draws. Go effectively doesn't (no
draws under standard scoring), so we kept `GoZeroNet`'s convention
instead — a single `tanh`-bounded scalar in `[-1, 1]` — so the new net's
`predict()` output is a drop-in match for what `engine/mcts.py`'s
`ZeroAgent` already expects, not a reimplementation of Maia-3's head.

---

## 5. What stage 1 actually built, in that context

`engine/token_transformer.py`'s `TokenTransformerNet` is deliberately
the *cheapest* point in this whole design space that still qualifies as
"a Transformer reading a board":

```
TokenEncoder.encode(game_state)
        │
        ▼
  (num_tokens, token_dim) array      -- e.g. (169, 32) at 13x13
        │
        ▼
  input_proj: Linear(token_dim, d_model)      -- per-token projection
        │
        ▼
  + RowColPositionalEmbedding                 -- learned row/col tables,
        │                                        the chessformer.py-style
        │                                        approach, generalized to N
        ▼
  nn.TransformerEncoder (stock PyTorch)       -- unmodified self-attention,
        │                                        exactly section 2a's formula
        ▼
  per-token policy_head ──► spatial logits (num_tokens)
  pooled (mean) ──► pass_head ──► pass logit
  pooled (mean) ──► value head ──► tanh value
```

Put in terms of §2-4 above: we took the *original* Transformer's
attention mechanism completely unmodified (stock
`nn.TransformerEncoder`), used Chessformer's older, simpler positional
scheme (learned row/col embeddings, not GAB) generalized from a fixed 8
to a parameter N, used our own smaller occupancy-only token alphabet
suited to Go's uniform stones, and used a Go-native policy head
(one logit per square, no from/to pairing) plus an AlphaZero-style
pooled value head matched to `GoZeroNet`'s existing scalar convention.

Every piece of this was a deliberate choice to stay at the *cheapest*
point on each axis first (§6 of `board-specification.md` has the full
pros/cons table behind that choice), specifically so that stage
1 could answer one question only — "is the pipeline shape-correct and
trainable, at any board size?" — before spending any complexity budget
on the parts that might actually make it *play well* (relative-position
bias next, then GAB). That's also why stage 1's tests
(`tests/test_token_transformer.py`) check shapes, gradients, and
integration with the existing MCTS harness, and deliberately do **not**
check playing strength — a randomly-initialized net has none to check.

---

## 6. Summary table — what changed, at each step

| | Original Transformer | Chessformer | trans-go-former (this repo, stage 1) |
|---|---|---|---|
| Token = | word/subword | 1 of 64 chess squares | 1 of N² Baduk intersections (N parameterized) |
| Token content | embedding lookup by word ID | 1-of-12 piece type, one-hot, × history | 1-of-3 occupancy (empty/self/opp) + ko flag, × history |
| Positional info | fixed sinusoidal, added to input | learned row/col embeddings (older) → dynamic GAB (production) | learned row/col embeddings (stage 1); relative bias and GAB are staged next |
| Stack | encoder **+ decoder** (autoregressive generation) | encoder only | encoder only |
| Core attention math | `softmax(QKᵀ/√d)V` | unchanged | unchanged (stock `nn.TransformerEncoder`) |
| Output head | softmax over vocabulary, one word at a time | 64×64 from/to score + promotion logits | 1 logit per intersection + 1 pooled pass logit |
| Value head | none (not applicable) | WDL 3-class (chess has draws) | tanh scalar (matches `GoZeroNet`, no draws in Go) |
| Board-size flexibility | n/a | hardcoded 64 everywhere | parameterized (`BoardSpec`), tested at 9/13/19 |

---

## Glossary

- **Token** — one unit of input to a Transformer; a word for the
  original model, a board square/intersection here.
- **Self-attention** — the mechanism where every token computes a
  weighted blend of every other token's information, weights learned
  from how well their "query" and "key" projections match.
- **Multi-head attention** — running several smaller self-attention
  computations in parallel ("heads"), letting each specialize.
- **Positional encoding (PE)** — any signal that breaks attention's
  blindness to token order/identity, whether fixed (sinusoidal),
  learned-and-static (embedding table), or dynamic (GAB).
- **GAB (Geometric Attention Bias)** — Chessformer's dynamic
  positional mechanism: generates the attention bias fresh from board
  content each forward pass, instead of adding a fixed signal to the
  input. Not yet built here — planned as stage 3, see
  `board-specification.md` §6.
- **Encoder-only** — a Transformer using just the "read the input"
  stack, no "generate output token by token" decoder stack. Used
  because move prediction is one classification, not sequence
  generation.
- **ViT (Vision Transformer)** — the 2020 adaptation of the same
  encoder-only idea to images (image patches as tokens instead of
  words). The architectural template Chessformer/Maia-3/our net all
  actually follow, more than the original NLP Transformer.
- **Value head** — a small network on top of the Transformer's pooled
  output that predicts who's winning; an AlphaZero-lineage addition,
  unrelated to the original Transformer paper.
- **BoardSpec** — this repo's dataclass (`engine/token_encoder.py`)
  holding `board_size`/`history_depth`, from which every shape in the
  pipeline is derived — the mechanism behind "readily changeable"
  board size.

---
*Key files:* `engine/token_encoder.py`, `engine/token_transformer.py`,
`docs/board-specification.md` (the design decisions this doc explains
the lineage behind), `chessformer.py` and `maia3/models.py` (external
reference implementations read to write this).
