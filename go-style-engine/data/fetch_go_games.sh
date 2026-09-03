#!/usr/bin/env bash
set -euo pipefail

# Fetches Go game record archives from CWI's public collection
# (homepages.cwi.nl/~aeb/go/games/) directly into this directory.
#
# Deliberately NOT committed to git as data -- both archives come from
# a stable, public, always-current source, so the right artifact to
# version is this script, not a multi-megabyte copy of someone else's
# data. Re-run this whenever a training/experiment run needs the games
# (locally, or as a setup_command on a Sail delegation / devbox) rather
# than syncing files by hand. See docs/board-specification.md §8d/§8e
# for why real human game data matters here (self-play from a random,
# never-trained net was found to be a weak, noisy signal source
# regardless of volume -- see also Specs/009).
#
# Usage:
#   ./fetch_go_games.sh            # fetch/refresh the 9x9 archive only
#                                     (cheap; the full corpus is always
#                                     skipped by default, see below)
#   ./fetch_go_games.sh --full     # also fetch the full 46MB corpus,
#                                     but only if not already present
#   ./fetch_go_games.sh --force    # re-fetch everything, even if present
#
# Two archives:
#   9x9/          -- 517 games (per the pre-built tarball; an earlier
#                    manual per-directory crawl of the live site found
#                    only 503, missing Misc/'s own nested subdirectories
#                    -- the tarball is the more complete, correct
#                    source, use it, not a hand-rolled crawler).
#                    Specifically 9x9, verified SZ[9] throughout. Small
#                    (~2MB extracted). Always (re-)fetched fresh --
#                    cheap.
#   go-games.tgz  -- the FULL professional archive (~96,000 games,
#                    mostly 19x19 -- SGF's SZ[] tag defaults to 19 when
#                    omitted, which most files here do), ~46MB
#                    compressed, organized by tournament (Honinbo,
#                    Kisei, Fujitsu, AlphaGo, and dozens more). Left
#                    compressed by default -- 96K small files is a lot
#                    of inodes to extract for consumers that only need
#                    a subset; extract with:
#                      tar xzf go-games.tgz
#                    Only fetched if missing, or with --full/--force,
#                    since it's the expensive one.

cd "$(dirname "$0")"

BASE="https://homepages.cwi.nl/~aeb/go/games"
FETCH_FULL=0
FORCE=0

for arg in "$@"; do
  case "$arg" in
    --full) FETCH_FULL=1 ;;
    --force) FETCH_FULL=1; FORCE=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

echo "=== 9x9 archive ==="
if [ -d "9x9" ] && [ "$FORCE" -eq 0 ]; then
  echo "9x9/ already present; skipping (use --force to refetch)."
else
  rm -rf 9x9 9x9.tgz.tmp
  curl -sL -o 9x9.tgz.tmp "${BASE}/9x9.tgz"
  tar xzf 9x9.tgz.tmp
  rm 9x9.tgz.tmp
  n=$(find 9x9 -name '*.sgf' | wc -l | tr -d ' ')
  echo "Fetched 9x9/: $n games."
fi

echo "=== full games archive ==="
if [ "$FETCH_FULL" -eq 0 ]; then
  echo "Skipped by default (46MB) -- pass --full or --force to fetch it."
elif [ -f "go-games.tgz" ] && [ "$FORCE" -eq 0 ]; then
  echo "go-games.tgz already present; skipping (use --force to refetch)."
else
  curl -sL -o go-games.tgz.tmp "${BASE}/games.tgz"
  mv go-games.tgz.tmp go-games.tgz
  echo "Fetched go-games.tgz ($(du -h go-games.tgz | cut -f1)). Not extracted -- see this script's header."
fi

echo "Done."
