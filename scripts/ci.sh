#!/usr/bin/env bash
# Continuous-integration gate for the n-body physics project.
#
# Runs, in order:
#   1. clippy (all targets, warnings = errors)
#   2. unit tests
#   3. release build of headless
#   4. headless CLI contract (--n 0 → exit 2; --min-ratio fail/pass; --json;
#      --quiet --json → exactly 1 line; --solar smoke run)
#   5. every example compiles (they are the project's ad-hoc diagnostics and
#      are the first thing to rot when a signature changes)
#   6. sweep.sh output format (header + rows, all 6-tab TSV fields) and
#      --help does not leak shell code
#
# Exits 0 only if every gate passes. Any failure leaves the last command's
# stdout/stderr visible for diagnosis.
#
# Thresholds are deliberately loose (>= 100× headroom on the project
# maintainer's machine) so a slower CI box does not spuriously fail. The
# gate is meant to catch *regressions* (5× slowdown, wrong exit code),
# not absolute performance.
#
# The `--min-ratio` fail gate uses a value far above any plausible
# free-running ratio so it works on fast boxes too: it is a test of
# "the flag propagates failure", not of "n = 100 is slow".

set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v jq >/dev/null; then
    echo "error: jq is required (sweep.sh --out parses JSON)" >&2
    exit 1
fi

capture_exit() {
    local expected="$1"; shift
    set +e
    "$@" >/dev/null 2>&1
    local code=$?
    set -e
    if [ "$code" -ne "$expected" ]; then
        echo "FAIL: '$*' exited $code, expected $expected" >&2
        return 1
    fi
    printf '  ok: %s → exit %d\n' "$*" "$code"
}

echo "== 1. clippy (all targets, warnings as errors) =="
cargo clippy --all-targets -- -D warnings

echo "== 2. unit tests (incl. fuzz smoke) =="
cargo test --lib --quiet

echo "== 3. release build (headless) =="
cargo build --release --bin headless --quiet

echo "== 4. headless CLI contract =="
capture_exit 2 ./target/release/headless --n 0
# Fail gate: --min-ratio is far above any plausible free-running ratio, so
# the flag *must* trigger. This is a test of the flag, not of n = 100 being
# slow (which it isn't on fast machines).
capture_exit 1 ./target/release/headless --n 100 --secs 1 --min-ratio 100000 --quiet
capture_exit 0 ./target/release/headless --n 100 --secs 1 --min-ratio 0.001 --quiet
# Solar is ~214 bodies and free-running throughput is nowhere near 100×
# 480 steps/s. Use a loose threshold that only catches a real regression
# (e.g. a hang or a 100× slowdown). The gate is: exit 0.
capture_exit 0 ./target/release/headless --solar --secs 2 --min-ratio 0.001 --quiet
./target/release/headless --n 100 --secs 1 --json | jq -c . >/dev/null
echo "  ok: --json produced valid JSON"

# `--quiet --json` must print exactly one line (the final summary). The
# output is captured *before* counting so a non-zero exit from `headless`
# produces a clear FAIL instead of tripping `set -e` mid-assignment.
QUIET_OUT=$(./target/release/headless --n 100 --secs 2 --quiet --json) || {
    echo "FAIL: --quiet --json exited non-zero" >&2
    exit 1
}
QUIET_LINES=$(printf '%s\n' "$QUIET_OUT" | wc -l | tr -d ' ')
if [ "$QUIET_LINES" -ne 1 ]; then
    echo "FAIL: --quiet --json printed $QUIET_LINES lines, expected 1" >&2
    exit 1
fi
echo "  ok: --quiet --json suppresses progress (1 line)"

echo "== 5. examples compile =="
cargo build --examples --quiet
echo "  ok: all examples build"

echo "== 6. sweep.sh format =="
TSV=$(mktemp)
trap 'rm -f "$TSV"' EXIT
scripts/sweep.sh --ns 100 --seeds 42 --secs 1 --out "$TSV" >/dev/null
if awk -F'\t' 'NF != 6 { exit 1 }' "$TSV"; then
    echo "  ok: TSV has 6 tab-separated fields on every line"
else
    echo "FAIL: TSV output malformed" >&2
    exit 1
fi

# `--help` must not leak shell code (`set -euo pipefail`, `SECS=20`, ...).
# Capture the output first so a non-zero exit from `sweep.sh` produces a
# clear FAIL, not a silent "grep found nothing → we say ok".
HELP_OUT=$(scripts/sweep.sh --help) || {
    echo "FAIL: scripts/sweep.sh --help exited non-zero" >&2
    exit 1
}
if printf '%s\n' "$HELP_OUT" | grep -q "set -euo pipefail"; then
    echo "FAIL: scripts/sweep.sh --help leaks shell code" >&2
    exit 1
fi
echo "  ok: sweep.sh --help runs and contains only comments"

echo ""
echo "ALL CI GATES PASSED"