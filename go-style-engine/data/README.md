# data/

Real Go game records, used as training/reference data — not generated
by this project's own self-play. Not committed to git: both source
archives come from a stable public site
([homepages.cwi.nl/~aeb/go/games/](https://homepages.cwi.nl/~aeb/go/games/)),
so the thing worth versioning is the fetch logic, not a copy of
someone else's data.

```sh
./fetch_go_games.sh          # 9x9 archive only (small, ~2MB)
./fetch_go_games.sh --full   # also the full corpus (~46MB, mostly 19x19)
```

See `fetch_go_games.sh`'s header comment for exactly what's in each
archive, current counts, and why this data matters (a direct response
to `../docs/board-specification.md` §8d/§8e's finding that self-play
from a random, never-trained net is a weak signal source regardless of
volume — see `Specs/009`).
