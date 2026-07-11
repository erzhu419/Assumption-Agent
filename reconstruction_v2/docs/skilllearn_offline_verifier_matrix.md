# SkillLearnBench Offline Verifier Matrix

## Policy

The benchmark has 100 local tasks across 20 families. Ninety-five upstream
`test.sh` files contain runtime network setup such as `apt`, `curl`, `pip`, or
`uvx`. Local task data alone therefore does not make evaluation offline.

The paper path uses three states:

- **ready**: no runtime network command is needed, or a pinned local profile has
  passed known-pass/known-fail contract probes plus an original-test collection
  probe under Docker `--network none`;
- **blocked**: the task is rejected before model execution;
- **excluded**: a preregistered benchmark rule excludes the family for a reason
  independent of model performance.

## Coverage

| Family | Tasks | Local status | Required local runtime or action |
|---|---:|---|---|
| anthropic-poster-design | 5 | ready | `anthropic-poster-py312-v1` |
| dbscan-parameter-tuning | 5 | ready | pytest/CTRF already pinned in the task image |
| chinese-poem-generator | 5 | ready | `chinese-poem-py312-v1`; includes pinned pypinyin |
| court-form-filling | 6 | ready | `common-pytest-ctrf-py312-v1`; PDF audit copies preserved |
| dependency-vulnerability-check | 5 | ready | `common-pytest-ctrf-py310-v1`; CSV audit copy preserved |
| earthquake-plate-calculation | 6 | ready | `common-pytest-ctrf-py310-v1` |
| enterprise-information-search | 6 | ready | `common-pytest-ctrf-py312-v1` |
| financial-analysis | 6 | ready | `common-pytest-ctrf-py312-v1`; pandas is in the task image |
| offer-letter-generator | 6 | ready | `common-pytest-ctrf-py312-v1`; python-docx is in the task image |
| organize-messy-files | 6 | ready | `common-pytest-ctrf-py312-v1`; PyPDF2 is in the task image |
| schedule-planning | 5 | ready | `common-pytest-ctrf-py312-v1`; `RESULTS_PATH` preserved |
| stock-data-visualization | 5 | ready | `common-pytest-ctrf-py312-v1` |
| temperature-simulation | 5 | ready | `common-pytest-ctrf-py38-v1`; adds a real CTRF receipt |
| travel-planning | 5 | ready | `common-pytest-ctrf-py311-v1`; itinerary audit copy preserved |
| video-object-counting | 5 | ready | `common-pytest-ctrf-py312-v1`; OpenCV is in the task image |
| weighted-gdp-calculation | 6 | blocked | openpyxl plus local `ssconvert` prelude |
| fix-security-bug | 3 | blocked | requests plus local Druid restart/readiness prelude |
| nlp-paper-reproduction | 3 | blocked | large pinned Torch/Transformers/DeepSpeed stack |
| python-scala-translation | 2 | blocked | prebuilt Scala/coursier toolchain; no GitHub/Maven fetch |
| github-repo-analytics | 5 | excluded | external `GH_TOKEN` credential rule |

Current credential-independent offline coverage is 81/95 tasks. The full local
inventory is 81 ready, 14 blocked, and 5 credential-excluded tasks. This is an
infrastructure coverage figure, not an accuracy result.

## Validation Evidence

The 2026-07-11 v2 matrix used only train-split representatives. It made no model
call and did not access sealed-test content. All six profiles produced reward 1
for a one-test known-pass fixture and reward 0 for a one-test known-fail fixture.
A train representative from each of the 14 declared families then collected and
executed its original local `test_outputs.py` under an empty workspace,
producing a non-empty CTRF and the expected reward 0. The family probes
collected between 2 and 11 tests each.

This proves dependency availability, Python ABI compatibility, wrapper reward
semantics, and original-test executability. It does not prove task accuracy or
that an arbitrary model output will pass. Real benchmark trials still require a
valid CTRF/reward receipt and are scored by the unchanged original assertions.

## Remaining Work

1. Add a local `ssconvert` semantic prelude for weighted GDP.
2. Add a local Druid restart/readiness prelude for fix-security.
3. Keep Scala and NLP blocked until their larger toolchains are deliberately
   prefetched, hashed, and justified for the selected experiment.

Every new profile must pass the same pass/fail contract and original-family
execution matrix before it enters a benchmark manifest.
