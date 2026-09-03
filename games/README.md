# Game records — local mirror

Real Go game records (SGF format), downloaded for potential use as
supervised training data — a direct response to `docs/board-
specification.md` §8d/§8e's finding that self-play from a randomly-
initialized, never-trained net is a weak, noisy signal source
regardless of volume. See `Specs/009` (once drafted) for the design
discussion.

## `9x9/`

**Source:** [homepages.cwi.nl/~aeb/go/games/games/other_sizes/9x9/](https://homepages.cwi.nl/~aeb/go/games/games/other_sizes/9x9/)
— a long-standing public Go game record archive maintained at CWI
(Centrum Wiskunde & Informatica, the Dutch national research institute
for mathematics and computer science).

**Downloaded:** 2026-09-03, directly from the live site (not from an
earlier local snapshot — an initial local `.tgz` had 502 files across
the same categories; the live site had 503, one additional game in
`Go_Seigen/`, so this mirror was re-fetched fresh rather than patched).

**Verified, not assumed:** every file confirmed `SZ[9]` (9x9 board,
no size contamination), 503 games total, 22,884 total half-move
positions, average 45.5 moves/game.

| Subdirectory | Games | What it is |
|---|---|---|
| `Minigo/` | 419 | Despite the folder name, these are real professional human games from a "Mini-Go" 9x9 tournament series — not related to Google's Minigo AI project. Spot-checked: e.g. `1.sgf` is Hoshikawa Takumi (1p) vs. Yuki Satoshi (9p). |
| `ProPairgo/` | 28 | Professional pair-go (two-player-per-side) tournament games, Ricoh Pairgo 2003-2009. |
| `computer/` | 18 | **Human-vs-AI games** — real professional players (Ichiriki Ryo, Cho Riyu, Zhou Junxun, Catalin Taranu, and others) against historical Go-playing programs (Zen, Mogo, Fuego, circa 2008-2013). Both sides' moves are real (the AI's moves are real AI output, not human play) — see `Specs/009`'s design discussion for whether/how these are used differently from purely-human games. |
| `Go_Seigen/` | 3 | Games by Go Seigen, one of the most significant professional players of the 20th century. |
| `NHK/` | 33 (across 6 year subdirectories) | Games broadcast on NHK (Japanese public broadcaster), various years 1989-2009. |
| `Misc/` | 2 | One-off event games (an igo festival, a title-celebration game). |

Each SGF carries real metadata: event, player names and ranks, komi,
result, date — see any file directly for the exact fields available.

## Status

Not yet consumed by any training pipeline — `go-style-engine/` has no
SGF parsing capability as of this mirror's creation. See `Specs/009`
for the design of converting these into `train.py`-compatible
`(state, one-hot move, game-result reward)` triples.
