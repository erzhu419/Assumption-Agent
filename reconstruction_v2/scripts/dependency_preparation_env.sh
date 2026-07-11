#!/usr/bin/env bash

# Source this file only for an explicit, supervised dependency preparation run.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file; do not execute it as a benchmark step." >&2
  exit 2
fi

export ASSUMPTION_V2_PREP_PIP_INDEX_URL="${ASSUMPTION_V2_PREP_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_INDEX_URL="${ASSUMPTION_V2_PREP_PIP_INDEX_URL}"
export PIP_EXTRA_INDEX_URL=""
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-15}"
