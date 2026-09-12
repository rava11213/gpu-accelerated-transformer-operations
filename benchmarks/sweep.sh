#!/usr/bin/env bash
set -euo pipefail

binary="${1:-./build/transformer_ops}"
output="${2:-results/sweep.csv}"
mkdir -p "$(dirname "$output")"
echo 'op,variant,m,n,k,rows,cols,gpu_ms,cpu_ms,max_error' > "$output"

# Non-multiple dimensions exercise boundary handling; 768/1024/4096 mirror
# common transformer hidden sizes and token-row counts.
for shape in '127 193 61' '512 512 512' '1024 1024 1024'; do
  read -r m n k <<< "$shape"
  for variant in naive optimized; do
    "$binary" --op matmul --variant "$variant" --m "$m" --n "$n" --k "$k" \
      --warmup 5 --iterations 50 --check --csv | tail -n +2 >> "$output"
  done
done

for shape in '17 31' '1024 768' '4096 1024'; do
  read -r rows cols <<< "$shape"
  for op in softmax layernorm residual-layernorm; do
    "$binary" --op "$op" --rows "$rows" --cols "$cols" \
      --warmup 5 --iterations 50 --check --csv | tail -n +2 >> "$output"
  done
done

echo "Wrote $output"

