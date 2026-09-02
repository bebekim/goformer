# Meta-Policy: A Cluster That Moves Style Without Tanking Strength

**Date:** 2026-09-01
**Question:** can a character trait produce a *recognizable Go style* while keeping win-rate within a few points of optimal — rarely significantly worse?
**Answer:** yes, if the trait is a *cluster* operating only inside the viable band, not as a single knob overriding value.

## 1. What exists today already gets the safety property right

`engine/mcts.py:48-57` defines 7 knobs. The safety lock is `equal_value_tolerance` (default 0.05):

```python
viable = [m for m in visited if Q(m) >= max_Q - tolerance]  # mcts.py:233-235
style_score = Q + complexity_weight * cplx + safety_lambda * var  # mcts.py:248-252
chosen = argmax(style_score over viable)  # or softmax if temperature>0
```

*Style never reaches outside viable.* In win-prob units (`prob = (value+1)/2`), `tolerance` is a hard per-move loss cap:

| tolerance | max prob loss per move |
|---|---|
| 0.02 | 1.0 pp |
| **0.05 (current)** | **2.5 pp** |
| 0.10 | 5.0 pp |

With the default 0.05, picking the *worst* viable instead of the best costs ≤2.5 pp that move. In a sharp position `viable_count=1` → no choice at all, forced to the best. In a calm position `viable_count=5` → up to 5 live options, style picks among them. This is exactly PRD's "Voice prefers among viable decisions; Strength overextends inside the same band."

Trace evidence (random net, `trace_move.py` 30 rounds, fighter knobs):
- all 15 visited branches had Q ≈ 0 ±0.001 → all 15 viable → every pick was style-driven but meant nothing (variance 0.003, complexity 0.976 uniform). The *machinery is wired*, the *signal is missing* because the random net has no real values. See `docs/mcts-trace-walkthrough.md:8`.

### Current named clusters (selfplay.py:PRESETS)

These are placeholder vectors, not calibrated, but already cluster-shaped:

| preset | rounds | c (breadth) | complexity_weight | safety_lambda | character meaning |
|---|---|---|---|---|---|
| baseline | 200 | 2.0 | 0.0 | 0.0 | neutral |
| fighter | 150 | 2.0 | +0.6 | +0.08 | creates complexity, seeks risk, plays fast |
| builder | 260 | 2.0 | -0.5 | -0.15 | simplifies, avoids variance, thinks longer |
| adapter | 200 | **3.0** | +0.2 | 0.0 | broad reading (high c) |
| anchor | 220 | 2.0 | -0.1 | **-0.25** | safest line |

Each trait moves 2-3 axes together. The design intent (mcts.py docstring) maps this to PRD's 6 families:
- **Voice/Policy** (stable identity) → checkpoint fine-tune + default knob cluster
- **Psychology** (leading/trailing/fatigue/captaincy) → *modulates the same cluster* move-to-move
- **Capability** → not a knob, the checkpoint's raw strength

Single-knob tuning alone is weak/noisy (trace: cplx variance ±0.001). A cluster sums to a coherent policy shift.

## 2. Meta-Policy definition

A **Meta-Policy** is one named vector applied at three timescales, all respecting the viable-band cap:

```
MetaPolicy {
  checkpoint_delta:  per-trainee fine-tune of GoZeroNet (optional, strongest lever)
  base_knobs:        StyleKnobs cluster (the table above) — the default voice
  psychology_fn:     (game_state, score, stamina, captaincy) -> StyleKnobs delta
}
```

Example — *Fighter who overextends while ahead* (PRD p.545: "Adds complexity while already winning"):

```python
fighter_base = StyleKnobs(complexity_weight=+0.45, safety_lambda=+0.10, c=1.8, rounds=170)
def fighter_psychology(pos_value, ahead_by):  # ahead_by from pos_value or score estimate
    if ahead_by > 0.15:        # winning -> overextension: *increase* complexity preference
        return StyleKnobs(complexity_weight=+0.25)  # additive delta
    if ahead_by < -0.15:       # losing -> fighter at his best: same, but deeper read
        return StyleKnobs(c=-0.3, rounds=+30)
    return StyleKnobs()
# effective = fighter_base + fighter_psychology(view)  before choose_from_root
```

Traits stay legible because the cluster is *directional*: every fighter move nudges toward higher entropy children, but never outside tolerance.

## 3. Why clustering without tanking works

1. **Orthogonal signals.** `complexity` (child policy entropy, mcts.py:237-242) and `variance` (backed-up value spread, Branch.variance) are weakly correlated after search with a trained net. One captures "how open is the next position" (fight starter vs quiet extension), the other "how swingy is this line" (risk). Weighting both gives a distinguishable style axis even when each alone is noisy.

2. **Tolerance as the governor.** Style_score magnitude is bounded: with `complexity_weight ≤0.6` and `cplx ∈ [0,1]`, term ≤0.6. That's comparable to tolerance 0.05 only when Q differences are ≤0.05. When a position has a clear best (Q gap 0.15), viable=1, style term irrelevant. Strength is preserved precisely when it matters.

3. **Cumulative loss stays small.** If style costs ≤2.5 pp per *styled* move and ~30% of moves are styled (sharp positions force the best), expected game cost ≈ 0.3 * 80 moves * 1.5 pp ≈ tiny, well within "rarely significant" — the PRD's allowed fluctuation. This must be validated by tournament, not by math alone.

## 4. What to validate — the experiment that answers the question

Run *after* a basic trained checkpoint exists (current runs/demo.json is random-net, degenerate: frequent pass, 12-move games, viable=30 always — no signal).

### 4a. Build the base checkpoint (one-time)

```bash
cd research/go-style-engine
.venv/bin/python selfplay.py --board-size 13 --games 200 --black-preset baseline \
  --white-preset baseline --rounds-per-move 200 --out runs/gen0/ --save-experience runs/gen0_exp.npz
.venv/bin/python train.py --experience runs/gen0/ --board-size 13 --out-checkpoint checkpoints/base.pt --epochs 8 --seed 0
# -> base.pt + base.epoch{1..8}.pt + .meta.json
```

Expect: leaf values spread to [-1,1], viable_count drops from 15→1-5, Q separation appears (see trace doc's "What changes with a trained brain").

### 4b. Fine-tune per-style checkpoints (Voice)

```bash
# generate style-filtered experience, then fine-tune a copy of base:
.venv/bin/python selfplay.py --board-size 13 --games 120 --black-preset fighter --white-preset fighter \
  --checkpoint checkpoints/base.pt --rounds-per-move 200 --out runs/fighter_gen/
.venv/bin/python train.py --experience runs/fighter_gen/ --in-checkpoint checkpoints/base.pt \
  --out-checkpoint checkpoints/fighter.pt --epochs 3 --seed 1
# repeat for builder/adapter/anchor
```

This is the per-trainee Voice checkpoint path noted in selfplay.py and README. Keep the fine-tune short (3 epochs) so strength stays near base.

### 4c. Tournament — does style move without tanking?

```bash
# each cell = 60 games, same base checkpoint, only knobs differ (no checkpoint delta first)
for preset in fighter builder adapter anchor; do
  .venv/bin/python selfplay.py --board-size 13 --games 60 --black-preset $preset --white-preset baseline \
    --checkpoint checkpoints/base.pt --out runs/tourney_${preset}_vs_base/
done
```

For each run, compute from `games.jsonl`:
- **Strength:** black win-rate vs baseline. Target 40-60% (binomial 95% interval for 60 games is ±13%). Stay inside → "without tanking" passes. Flag if outside.
- **Style separation:** per-preset distribution of `complexity`, `variance`, `viable_count`, `candidate_count`, `think_time_s` (the 5 telemetry fields already in games.jsonl). Fighter should show higher mean complexity + variance, Builder lower, Adapter higher candidate_count (breadth), etc. Use simple per-metric histograms or a classifier.

Then repeat *with* checkpoint deltas (`--checkpoint checkpoints/fighter.pt` for fighter side) to measure additive Voice effect.

### 4d. Tolerance sweep (the meta-policy calibration)

```bash
# hold fighter cluster fixed, sweep tolerance to find the knee where win-rate drops but style separation plateaus
for tol in 0.02 0.05 0.08 0.12; do
  .venv/bin/python -c "import selfplay; ..."  # or add --equal-value-tolerance CLI and re-run
done
```

Expect: style separation grows quickly to 0.05, then flattens; win-rate stays flat to 0.05-0.08, then drops. The knee is the meta-policy's safe operating point.

## 5. What to build next in code

- [ ] Add `--equal-value-tolerance` / `--c` / `--complexity-weight` / `--safety-lambda` passthrough to selfplay.py tournament scripts (currently `build_knobs` already supports overrides — expose them).
- [ ] Add `MetaPreset` wrapper in `engine/mcts.py` that composes `base_knobs + psychology_fn(state)` before `choose_from_root`. Keep `StyleKnobs` as the primitive.
- [ ] Add post-tournament analysis script `analyze_style.py` that reads `runs/*/games.jsonl` and emits win-rate + complexity/variance histograms + viable_count distribution (the 3 numbers that prove style-without-tanking).
- [ ] Consider a prior-shaping variant (reweight child policy entropy before search) only if post-selection style is too subtle at 200 rounds. Prior shaping amplifies style earlier but risks larger strength loss — keep it behind a flag.

## 6. Connection back to character conversation

The same cluster idea ports to `research/convo-style-engine` once Go style is validated: a dialogue Meta-Policy (tone, directness, hedging, initiative) applied inside a "viable utterance" band (top-p filtered candidates) mirrors the Go viable band. Start with the 12 gold TalkExchanges in `kappago/Content/CoachingContent.swift` as the dialogue equivalent of a trained viable set.

---
*Key files:* `engine/mcts.py:48-273`, `engine/network.py`, `selfplay.py:44-54`, `docs/mcts-trace-walkthrough.md`, `specs/prd.md:510-552`.
