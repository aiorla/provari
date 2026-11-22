#!/bin/sh
# Fetches the network and the model that this version of provari carries, as
# data/network.nnue and data/model.json, from the networks release of
# provari's repository, where each file is named after its digest, and checks
# them against the SHA-256 below. With --legacy, it also fetches
# Atomic-Stockfish's network, run3b, which provari carried before varifish's, as
# data/legacy.nnue for the tests of that format. Files already in place and
# right are kept. PROVARI_NETWORKS names another place to fetch them from.
# Usage: data/fetch.sh [--legacy]
set -e
network=3d76258c556ce216d088f89ff094dadab4310e3838110c64f66049b6ab5c40b1
model=51be8f472304ce535a385ad88906c75a976e5c8d2b267162bd9818a59231882b
legacy=99dc67eabf26a64faeeca3a88b4c38597a840b8d4a874b9f2cf658c6f92a04a6
base=${PROVARI_NETWORKS:-https://github.com/aiorla/provari/releases/download/networks}
case $1 in
  "" | --legacy) ;;
  *)
    echo "usage: data/fetch.sh [--legacy]" >&2
    exit 2
    ;;
esac
cd "$(dirname "$0")"
if command -v sha256sum >/dev/null 2>&1; then
  sha256() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
  sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
  echo "needs sha256sum or shasum" >&2
  exit 1
fi
# Fetches a file unless it is in place with the digest given, from the
# release's file of that kind (network or model) and digest.
fetch() {
  if [ -f "$1" ] && [ "$(sha256 "$1")" = "$3" ]; then
    echo "$1 is in place"
    return
  fi
  name=$2-$(printf %s "$3" | cut -c1-12).${1##*.}
  if ! curl -fsSL -o "$1.part" "$base/$name"; then
    rm -f "$1.part"
    exit 1
  fi
  if [ "$(sha256 "$1.part")" != "$3" ]; then
    rm -f "$1.part"
    echo "$name has the wrong SHA-256" >&2
    exit 1
  fi
  mv "$1.part" "$1"
  echo "fetched $name as $1"
}
fetch network.nnue network "$network"
fetch model.json model "$model"
if [ "$1" = --legacy ]; then
  fetch legacy.nnue network "$legacy"
fi
