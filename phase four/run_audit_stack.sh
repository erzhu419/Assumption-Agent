#!/usr/bin/env bash
# Reproduce the six-layer audit stack on the three KEEP candidates
# (W076, W077, W078) from the paper. Requires ruoli.dev API keys in
# the environment (see phase zero/scripts/model_router.py).
#
# Each layer writes a self-contained JSON log to phase four/autonomous/.
# Given cached answers in phase two/analysis/cache/answers/, the judge-
# side of every layer is deterministic modulo the hosted LLM's own
# non-determinism (see Reproducibility appendix).

set -e
PY=${PY:-python3}
cd "$(dirname "$0")/.."

echo "=== L1: cross-family re-judgment (Opus on W076/W077/W078) ==="
$PY "phase four/exp1_cross_judge.py" || true

echo "=== L2: side-randomization reseed ==="
$PY "phase four/exp5_side_shuffle.py" || true

echo "=== L3: sample extension n=50 -> n=100 ==="
$PY "phase four/exp8_extend_n100.py" || true

echo "=== L4: cross-solver replication (gemini/haiku/gpt-mini) ==="
$PY "phase four/exp26_solver_ablation.py" || true

echo "=== L5a: fresh-domain cross-family (own math pool) ==="
$PY "phase four/exp31_benchmark_portability.py" || true

echo "=== L5b: GSM8K standard-benchmark port ==="
$PY "phase four/exp32_gsm8k_portability.py" || true

echo "=== L6: expensive-tier judge sanity check ==="
$PY "phase four/exp34_expensive_judges.py" || true

echo ""
echo "All layers complete. Logs under phase four/autonomous/exp*_log.json"
echo "Binomial CI tabulation: see phase four/compute_binomial_cis.py"
