# Testing

## Canonical test command

```sh
cd go-style-engine && .venv/bin/python -m pytest -q
```

Real observed output as of this doc's writing (2026-09-03, re-run
directly, not copied from an earlier source):

```
132 passed, 7 xfailed
```

## Environment setup

```sh
cd go-style-engine
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**Known pin issue:** `requirements.txt` pins `torch==2.13.0` /
`numpy==2.5.2`, which need Python 3.12+. On a host with only Python
3.11 (e.g. Debian 12's default apt repos, encountered setting up a
remote Sail devbox — see `TODO.md`), that install fails. Fallback:
install unpinned-but-compatible versions instead —

```sh
.venv/bin/pip install -q numpy torch pytest
```

— confirmed to work and produce the same test results at 3.11.

## The `xfail(strict=True)` convention

`tests/test_rules.py` marks some tests `@pytest.mark.xfail(strict=True,
...)`. These are **intentional documentation of not-yet-implemented
behavior**, describing what a future Swift port should do — not
failing tests that need fixing. `test_rules.py`'s own module docstring
states this directly: it's an executable specification for that future
port. Do not "fix" an xfail without understanding whether it's marking
a real gap or a deliberate future-behavior placeholder.

## Test file index

One line each — see the file itself for real detail:

- `test_rules.py` — the Go rules engine (`goboard.py`, `scoring.py`,
  `gotypes.py`, `zobrist.py`); doubles as the Swift-port executable
  spec (see xfail note above).
- `test_resume.py` — self-play run resumability (`selfplay.py`'s
  `games.jsonl`-based skip-completed-indices logic).
- `test_token_encoder.py` — `BoardSpec`/`TokenEncoder`, parametrized
  over board size (9/13/19) to check the "readily changeable" claim.
- `test_token_transformer.py` — `TokenTransformerNet` stage 1
  (`RowColPositionalEmbedding`, `pos_mode='absolute'`), shape/gradient
  checks, the `predict()` contract, `ZeroAgent` integration.
- `test_relative_position_bias.py` — `RelativePositionBias`
  (`pos_mode='relative'`) and the diagnostic `pos_mode='both'`.
- `test_geometric_attention_bias.py` — `GeometricAttentionBias`
  (`pos_mode='gab'`/`'gab_absolute'`), including the content-dependent-
  vs-content-independent distinction from `RelativePositionBias`.
- `test_train_early_stopping.py` — `train.py`'s best-checkpoint
  tracking, `--early-stopping-patience`, and small-dataset val-split
  edge cases (see `docs/common-pitfalls.md`).
