# data/

Real Go game records, used as training/reference data — not generated
by this project's own self-play. This directory holds only this
project's own *derived* training-format artifacts (`.npz` shards, via
`kifu_to_experience.py` per `../../Specs/009-sgf-kifu-supervised-training-pipeline.md`)
— none of it is committed, all of it is reproducible.

Fetching the raw archives and parsing SGF into normalized game records
is no longer done here — that now lives in a separate, sibling repo,
[`go-gibo-ingestion`](../../../go-gibo-ingestion), decoupled from this
model-training repo the same way `nem-forecast-orchestration` is split
from `nem-forecast`. Run it first:

```sh
cd ~/repositories/individual/go-gibo-ingestion
.venv/bin/python fetch.py
.venv/bin/python ingest.py --input data/raw/9x9 --archive-name 9x9 \
  --output data/parsed/9x9.jsonl
```

Then, from here, `kifu_to_experience.py` reads that JSONL and produces
`.npz` shards `train.py` can consume — see `Specs/009` and that
script's own `--help` for details, including the human-only game
filter (`go-gibo-ingestion`'s README documents why: most of the 9x9
archive is AI self-play, not human kifu, and that distinction matters
for what this data is used to validate — see
`../docs/board-specification.md` §8d/§8e).
