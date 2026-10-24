#!/bin/sh
# Builds provari for the browser: the WebAssembly module, the worker that runs
# it, and the data the worker loads: the network and the model, from data/ as
# bundled builds take them, and the 3-4 piece tables of wins and losses from
# the tablebases given, if any (1.2 MB, where the 5-piece ones take 0.4 GB).
# Into web/dist, or into the directory given. The module, the worker, the
# network and the model are named after their contents, so that browsers never
# mix two builds, each with a gzipped copy beside it (the network shrinks from
# 48 to 27 MB); manifest.json names them, and a page fetches it fresh.
# Usage: web/build.sh [--tablebases DIR] [OUT]
set -e
usage() {
  echo "usage: web/build.sh [--tablebases DIR] [OUT]" >&2
  exit 2
}
# Paths given are taken from where the script is called.
absolute() {
  case $1 in
    /*) echo "$1" ;;
    *) echo "$PWD/$1" ;;
  esac
}
tablebases=
out=
while [ $# -gt 0 ]; do
  case $1 in
    --tablebases)
      [ $# -gt 1 ] || usage
      tablebases=$(absolute "$2")
      shift
      ;;
    -*) usage ;;
    *) out=$(absolute "$1") ;;
  esac
  shift
done
cd "$(dirname "$0")/.."
out=${out:-$PWD/web/dist}
if [ ! -f data/network.nnue ] || [ ! -f data/model.json ]; then
  echo "needs the network and the model in data/: data/fetch.sh fetches them" >&2
  exit 1
fi
if [ -n "$tablebases" ] && [ ! -d "$tablebases" ]; then
  echo "no such directory: $tablebases" >&2
  exit 1
fi
if command -v sha256sum >/dev/null 2>&1; then
  sha256() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
  sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
  echo "needs sha256sum or shasum" >&2
  exit 1
fi
# clang builds zstd's C code for WebAssembly, and LLVM's archiver indexes it.
host=$(rustc -vV | sed -n 's/host: //p')
AR_wasm32_unknown_unknown="$(rustc --print sysroot)/lib/rustlib/$host/bin/llvm-ar"
if [ ! -x "$AR_wasm32_unknown_unknown" ]; then
  echo "needs LLVM's archiver: rustup component add llvm-tools" >&2
  exit 1
fi
export AR_wasm32_unknown_unknown
# A 64 MB stack, as deep lines recurse deeply; and WebAssembly's SIMD. A second
# build adds relaxed SIMD's dot product of bytes, for browsers that have it.
stack="-C link-arg=-zstack-size=67108864"
RUSTFLAGS="$stack -C target-feature=+simd128" \
  cargo rustc --lib --release --target wasm32-unknown-unknown --crate-type cdylib
RUSTFLAGS="$stack -C target-feature=+simd128,+relaxed-simd" \
  cargo rustc --lib --release --target wasm32-unknown-unknown --crate-type cdylib \
  --target-dir target/relaxed

mkdir -p "$out/tables"
rm -f "$out"/provari-*.wasm* "$out"/relaxed-*.wasm* "$out"/worker-*.js* "$out"/network-*.nnue* \
  "$out"/model-*.json* "$out"/manifest.json "$out"/tables/*
# Copies a file under its name with a digest of its contents, with a gzipped
# copy beside it for servers that send it compressed, and prints the name.
named() {
  digest=$(sha256 "$1" | cut -c1-12)
  cp "$1" "$out/$2-$digest.$3"
  gzip -9 -k -n "$out/$2-$digest.$3"
  echo "$2-$digest.$3"
}
wasm=$(named target/wasm32-unknown-unknown/release/provari.wasm provari wasm)
relaxed=$(named target/relaxed/wasm32-unknown-unknown/release/provari.wasm relaxed wasm)
worker=$(named web/worker.js worker js)
network=$(named data/network.nnue network nnue)
model=$(named data/model.json model json)
tables=
if [ -n "$tablebases" ]; then
  for table in "$tablebases"/*.atbw; do
    name=$(basename "$table")
    pieces=$(printf %s "${name%.*}" | tr -d v | wc -c)
    if [ -f "$table" ] && [ "$pieces" -le 4 ]; then
      cp "$table" "$out/tables/"
      tables="$tables${tables:+, }\"$name\""
    fi
  done
  if [ -z "$tables" ]; then
    echo "no tables of 3 or 4 pieces (.atbw) in $tablebases" >&2
    exit 1
  fi
fi
# The network's size as a page receives it, decompressed, for its progress.
network_bytes=$(wc -c < "$out/$network" | tr -d ' ')
cat > "$out/manifest.json" <<JSON
{"wasm": "$wasm", "relaxed": "$relaxed", "worker": "$worker", "network": "$network", "network_bytes": $network_bytes, "model": "$model", "tables": [$tables]}
JSON
ls -la "$out/$wasm"
