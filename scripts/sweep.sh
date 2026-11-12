#!/usr/bin/env bash
# Sweep the headless physics benchmark across body counts and seeds.
# Prints one line per run, or an aggregate table with --agg.
#
# Usage:
#   scripts/sweep.sh                                          # n = 100..5000, seed 42
#   scripts/sweep.sh --seeds 42,1,2 --secs 10                 # multi-seed, faster runs
#   scripts/sweep.sh --ns 500,1000,2000 --agg                 # per-n mean/min/max
#   scripts/sweep.sh --ns 100,500 --seeds 42 --out /tmp/x.tsv # save TSV

set -euo pipefail

SECS=20
SEEDS=(42)
NS=(100 200 500 1000 2000 5000)
SPEED=1.0
BIN="./target/release/headless"
OUT=""
AGG=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --secs)  SECS=$2;         shift 2 ;;
        --speed) SPEED=$2;        shift 2 ;;
        --seeds) IFS=, read -r -a SEEDS <<< "$2"; shift 2 ;;
        --ns)    IFS=, read -r -a NS    <<< "$2"; shift 2 ;;
        --out)   OUT=$2;          shift 2 ;;
        --agg)   AGG=1;           shift 1 ;;
        -h|--help)
            # Header comment is lines 2-9. `14p` leaked the `set -euo
            # pipefail` line and the `SECS=…` assignments.
            sed -n '2,9p' "$0" | sed 's/^# //'
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -x "$BIN" ]]; then
    echo "building $BIN..." >&2
    cargo build --release --bin headless >&2
fi

run_one() {
    local n=$1 seed=$2
    "$BIN" --n "$n" --seed "$seed" --speed "$SPEED" --secs "$SECS" --quiet --json \
        | tail -1
}

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

total=$(( ${#NS[@]} * ${#SEEDS[@]} ))
i=0
for n in "${NS[@]}"; do
    for s in "${SEEDS[@]}"; do
        i=$(( i + 1 ))
        printf '\r[%d/%d] n=%-5s seed=%s  ' "$i" "$total" "$n" "$s" >&2
        run_one "$n" "$s" >> "$TMP"
    done
done
echo "" >&2

# Pretty fixed-width table for the terminal.
emit_table() {
    printf '%-6s %-6s %-10s %-14s %-8s %-10s\n' \
        n seed steps steps/sec ratio avg_us
    jq -r '[.n, .seed, .steps, .steps_per_sec, .realtime_ratio, .avg_advance_us]
           | @tsv' "$TMP" \
        | sort -n -k1,1 -k2,2 \
        | awk -F'\t' '{printf "%-6d %-6d %-10d %-14.2f %-8.3f %-10.2f\n",
                            $1, $2, $3, $4, $5, $6}'
}

# Machine-readable TSV for --out (header + one row per run).
emit_tsv() {
    printf 'n\tseed\tsteps\tsteps_per_sec\trealtime_ratio\tavg_advance_us\n'
    jq -r '[.n, .seed, .steps, .steps_per_sec, .realtime_ratio, .avg_advance_us]
           | @tsv' "$TMP" \
        | sort -n -k1,1 -k2,2
}

# Aggregate per-n table. Groups by `n0` (requested count), not `n` (final).
emit_agg() {
    printf '%-6s %-4s %-13s %-13s %-13s %-13s %-11s %-11s %-11s\n' \
        n0 runs n_final_mean sps_mean sps_min sps_max ratio_mean ratio_min ratio_max
    jq -r '[.n0, .n, .steps_per_sec, .realtime_ratio] | @tsv' "$TMP" \
        | awk -F'\t' '
            {
                n0  = $1;   n   = $2;
                sps = $3+0; r   = $4+0;
                if (!(n0 in cnt)) {
                    cnt[n0] = 0;
                    n_sum[n0] = 0;
                    sps_sum[n0] = 0; sps_min[n0] = 1e18; sps_max[n0] = -1;
                    r_sum[n0] = 0;   r_min[n0] = 1e18;   r_max[n0] = -1;
                }
                cnt[n0]++;
                n_sum[n0] += n;
                sps_sum[n0] += sps;
                if (sps < sps_min[n0]) sps_min[n0] = sps;
                if (sps > sps_max[n0]) sps_max[n0] = sps;
                r_sum[n0] += r;
                if (r < r_min[n0]) r_min[n0] = r;
                if (r > r_max[n0]) r_max[n0] = r;
            }
            END {
                for (n0 in cnt) {
                    printf "%-6d %-4d %-13.1f %-13.2f %-13.2f %-13.2f %-11.4f %-11.4f %-11.4f\n",
                        n0, cnt[n0],
                        n_sum[n0]/cnt[n0],
                        sps_sum[n0]/cnt[n0], sps_min[n0], sps_max[n0],
                        r_sum[n0]/cnt[n0],   r_min[n0],   r_max[n0];
                }
            }' \
        | sort -n -k1,1
}

if ! command -v jq >/dev/null 2>&1; then
    echo "note: install jq for pretty output; showing raw JSON" >&2
    if [[ -n "$OUT" ]]; then
        cat "$TMP" > "$OUT"
        echo "wrote JSON lines to $OUT" >&2
    fi
    cat "$TMP"
    exit 0
fi

if [[ -n "$OUT" ]]; then
    emit_tsv > "$OUT"
    echo "wrote TSV to $OUT" >&2
fi
if [[ "$AGG" -eq 1 ]]; then
    emit_agg
else
    emit_table
fi