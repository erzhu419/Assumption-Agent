#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

BENCHMARK_ROOT="${BENCHMARK_ROOT:-reference/self_evo_continual_20260707/repos/SkillLearnBench}"
ENV_FILE="${ENV_FILE:-../.env}"
PROTOCOL="${PROTOCOL:-manifests/skilllearn_paper_protocol_v3_10_ruoli_gpt54mini.json}"
MANIFEST="${MANIFEST:-manifests/skilllearnbench_instance_holdout_offline_ready_v1.json}"
RUN_ROOT="${RUN_ROOT:-artifacts/paper_primary_v3_10_offline86_ruoli_gpt54mini_outer6_model1_diverse01}"
LOCK="${RUN_ROOT}/protocol_lock.json"
RECEIPT="${RUN_ROOT}/freeze_receipt.json"
PREWARM_RECEIPT="${RUN_ROOT}/development_prewarm.json"
PARALLEL_WORKERS="$(python3 -c 'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); m=json.load(open(sys.argv[2], encoding="utf-8")); phase="family_out_development" if m["protocol"] == "family_out" else "development"; print(p["phases"][phase]["parallel_workers"])' "${PROTOCOL}" "${MANIFEST}")"
MODEL="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["model"])' "${PROTOCOL}")"
TRIAL_PROVIDER_MODE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["trial_provider_mode"])' "${PROTOCOL}")"
PROPOSAL_PROVIDER_CHAIN="$(python3 -c 'import json,sys; print(",".join(json.load(open(sys.argv[1], encoding="utf-8"))["proposal_provider_chain"]))' "${PROTOCOL}")"
PROVIDER_ENDPOINT_ORIGIN="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["provider_endpoint_origin"])' "${PROTOCOL}")"
PROVIDER_ENDPOINT_IPV4S="$(python3 -c 'import json,sys; print(",".join(json.load(open(sys.argv[1], encoding="utf-8"))["provider_endpoint_ipv4s"]))' "${PROTOCOL}")"
TRIAL_NETWORK_BYTE_LIMIT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["execution"]["trial_network_byte_limit"])' "${PROTOCOL}")"
export ASSUMPTION_V2_MODEL="${MODEL}"
export ASSUMPTION_V2_SKILLLEARN_PROVIDER_MODE="${TRIAL_PROVIDER_MODE}"
export ASSUMPTION_V2_PROVIDER_CHAIN="${PROPOSAL_PROVIDER_CHAIN}"
export ASSUMPTION_V2_API_BASE="${PROVIDER_ENDPOINT_ORIGIN}"
export ASSUMPTION_V2_API_ALLOWED_IPV4S="${PROVIDER_ENDPOINT_IPV4S}"
export ASSUMPTION_V2_SKILLLEARN_CACHE_ONLY=1
export ASSUMPTION_V2_TRIAL_NETWORK_BYTE_LIMIT="${TRIAL_NETWORK_BYTE_LIMIT}"

run_docker_group() {
  if docker info >/dev/null 2>&1; then
    "$@"
    return
  fi
  local command
  printf -v command '%q ' "$@"
  sg docker -c "${command}"
}

preflight() {
  run_docker_group python3 -m assumption_agent.benchmarks.docker_egress --ensure
  run_docker_group python3 -m assumption_agent.benchmarks.preflight \
    --env-file "${ENV_FILE}" \
    --manifest "${MANIFEST}" \
    --trial-provider-mode "${TRIAL_PROVIDER_MODE}" \
    --root "${BENCHMARK_ROOT}"
}

lock_protocol() {
  mkdir -p "${RUN_ROOT}"
  run_docker_group python3 -m assumption_agent.benchmarks.paper_protocol \
    --protocol "${PROTOCOL}" \
    --project-root . \
    --benchmark-root "${BENCHMARK_ROOT}" \
    --env-file "${ENV_FILE}" \
    --out "${LOCK}" \
    --require-claim-eligible
}

prewarm() {
  mkdir -p "${RUN_ROOT}"
  run_docker_group python3 -m assumption_agent.benchmarks.prewarm \
    --root "${BENCHMARK_ROOT}" \
    --manifest "${MANIFEST}" \
    --env-file "${ENV_FILE}" \
    --events "${RUN_ROOT}/development_prewarm.events.jsonl" \
    --out "${PREWARM_RECEIPT}" \
    --parallel-workers "${PARALLEL_WORKERS}" \
    --attempts 3 \
    --trial-provider-mode "${TRIAL_PROVIDER_MODE}" \
    --require-passed
}

run_generation() {
  local name="$1"
  shift
  run_docker_group python3 -m assumption_agent.benchmarks.docker_egress --ensure
  run_docker_group python3 -m assumption_agent.benchmarks.skilllearn_experiment \
    --root "${BENCHMARK_ROOT}" \
    --protocol "${PROTOCOL}" \
    --protocol-lock "${LOCK}" \
    --manifest "${MANIFEST}" \
    --env-file "${ENV_FILE}" \
    --out "${RUN_ROOT}/${name}.report.json" \
    --events "${RUN_ROOT}/${name}.events.jsonl" \
    --work-dir "${RUN_ROOT}/${name}" \
    --archive-out "${RUN_ROOT}/${name}.archive.json" \
    --prewarm-receipt "${PREWARM_RECEIPT}" \
    --execute \
    "$@"
}

smoke() {
  mkdir -p "${RUN_ROOT}"
  run_generation smoke_recursive \
    --train-limit 4 --validation-limit 2 \
    --paired-no-recursive-out "${RUN_ROOT}/smoke_no_recursive.report.json" \
    --paired-no-recursive-archive-out "${RUN_ROOT}/smoke_no_recursive.archive.json"
}

develop() {
  test -s "${LOCK}" || { echo "Missing claim-eligible protocol lock: ${LOCK}" >&2; exit 2; }
  run_generation development_recursive \
    --paired-no-recursive-out "${RUN_ROOT}/development_no_recursive.report.json" \
    --paired-no-recursive-archive-out "${RUN_ROOT}/development_no_recursive.archive.json"
}

freeze() {
  python3 -m assumption_agent.benchmarks.paper_freeze \
    --protocol "${PROTOCOL}" \
    --protocol-lock "${LOCK}" \
    --manifest "${MANIFEST}" \
    --benchmark-root "${BENCHMARK_ROOT}" \
    --project-root . \
    --recursive-report "${RUN_ROOT}/development_recursive.report.json" \
    --recursive-archive "${RUN_ROOT}/development_recursive.archive.json" \
    --no-recursive-report "${RUN_ROOT}/development_no_recursive.report.json" \
    --no-recursive-archive "${RUN_ROOT}/development_no_recursive.archive.json" \
    --controls-out "${RUN_ROOT}/frozen_controls" \
    --out "${RECEIPT}"
}

run_controls() {
  local split="$1"
  local records="${RUN_ROOT}/${split}.records.jsonl"
  local journal_args=()
  if [[ "${split}" == "test" ]]; then
    journal_args=(--sealed-journal "${RUN_ROOT}/sealed_test.journal.json")
  fi
  run_docker_group python3 -m assumption_agent.benchmarks.docker_egress --ensure
  run_docker_group python3 -m assumption_agent.benchmarks.paper_controls \
    --project-root . \
    --benchmark-root "${BENCHMARK_ROOT}" \
    --manifest "${MANIFEST}" \
    --protocol "${PROTOCOL}" \
    --protocol-lock "${LOCK}" \
    --freeze-receipt "${RECEIPT}" \
    --env-file "${ENV_FILE}" \
    --events "${RUN_ROOT}/${split}.events.jsonl" \
    --records "${records}" \
    --trials-dir "${RUN_ROOT}/${split}_trials" \
    --split "${split}" \
    "${journal_args[@]}"
}

report() {
  local split="$1"
  python3 -m assumption_agent.benchmarks.paper_report \
    --records "${RUN_ROOT}/${split}.records.jsonl" \
    --protocol "${PROTOCOL}" \
    --protocol-lock "${LOCK}" \
    --manifest "${MANIFEST}" \
    --split "${split}" \
    --out-json "${RUN_ROOT}/${split}.paper_report.json" \
    --out-md "${RUN_ROOT}/${split}.paper_report.md"
}

case "${1:-}" in
  preflight) preflight ;;
  lock) lock_protocol ;;
  prewarm) prewarm ;;
  smoke) smoke ;;
  develop) develop ;;
  freeze) freeze ;;
  validation-controls) run_controls validation; report validation ;;
  sealed-test) run_controls test; report test ;;
  all-development)
    preflight
    lock_protocol
    prewarm
    smoke
    develop
    freeze
    run_controls validation
    report validation
    ;;
  *)
    echo "Usage: $0 {preflight|lock|prewarm|smoke|develop|freeze|validation-controls|sealed-test|all-development}" >&2
    exit 2
    ;;
esac
