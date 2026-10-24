#!/bin/sh
# Fetches the network and the model that this version of provari carries, as
# data/network.nnue and data/model.json, from the networks release of
# provari's repository, where each file is named after its digest, and checks
# them against the SHA-256 below. Files already in place and right are kept.
# PROVARI_NETWORKS names another place to fetch them from.
# Usage: data/fetch.sh
set -e
network=99dc67eabf26a64faeeca3a88b4c38597a840b8d4a874b9f2cf658c6f92a04a6
model=8822a85fde620b734b23e99bf48263b3c82bc6fbe706fc8a836f710560f7f987
base=${PROVARI_NETWORKS:-https://github.com/aiorla/provari/releases/download/networks}
cd "$(dirname "$0")"
if command -v sha256sum >/dev/null 2>&1; then
  sha256() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
  sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
  echo "needs sha256sum or shasum" >&2
  exit 1
fi
# Fetches a file unless it is in place with the digest given.
fetch() {
  if [ -f "$1" ] && [ "$(sha256 "$1")" = "$2" ]; then
    echo "$1 is in place"
    return
  fi
  name=${1%.*}-$(printf %s "$2" | cut -c1-12).${1##*.}
  if ! curl -fsSL -o "$1.part" "$base/$name"; then
    rm -f "$1.part"
    exit 1
  fi
  if [ "$(sha256 "$1.part")" != "$2" ]; then
    rm -f "$1.part"
    echo "$name has the wrong SHA-256" >&2
    exit 1
  fi
  mv "$1.part" "$1"
  echo "fetched $name as $1"
}
fetch network.nnue "$network"
fetch model.json "$model"
