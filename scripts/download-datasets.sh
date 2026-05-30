#!/usr/bin/env bash
# Downloads benchmark datasets into ./datasets (which is gitignored).
# siftsmall is tiny and good for a quick check. sift1m is the real one.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p datasets
cd datasets

base_url="ftp://ftp.irisa.fr/local/texmex/corpus"

fetch() {
  local name="$1"
  if [ -d "$name" ]; then
    echo "$name already present, skipping"
    return
  fi
  echo "downloading $name ..."
  curl -L -O "$base_url/$name.tar.gz"
  tar xzf "$name.tar.gz"
  rm -f "$name.tar.gz"
}

target="${1:-siftsmall}"
fetch "$target"

echo
echo "run the benchmark with:"
echo "  cargo run --release -p loupe-bench -- \\"
echo "    datasets/$target/${target}_base.fvecs \\"
echo "    datasets/$target/${target}_query.fvecs \\"
echo "    datasets/$target/${target}_groundtruth.ivecs"
