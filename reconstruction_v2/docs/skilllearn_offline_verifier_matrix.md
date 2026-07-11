# SkillLearnBench Offline Verifier Matrix

## Policy

The benchmark has 100 local tasks across 20 families. Ninety-five upstream
`test.sh` files contain runtime network setup such as `apt`, `curl`, `pip`, or
`uvx`. Local task data alone therefore does not make evaluation offline.

The paper path uses three states:

- **ready**: no runtime network command is needed, or a pinned local profile has
  passed an original-verifier probe under Docker `--network none`;
- **blocked**: the task is rejected before model execution;
- **excluded**: a preregistered benchmark rule excludes the family for a reason
  independent of model performance.

## Coverage

| Family | Tasks | Local status | Required local runtime or action |
|---|---:|---|---|
| anthropic-poster-design | 5 | ready | `anthropic-poster-py312-v1`; pass/fail probe verified |
| dbscan-parameter-tuning | 5 | ready | pytest/CTRF already pinned in the task image |
| chinese-poem-generator | 5 | blocked | pytest, CTRF, pypinyin |
| court-form-filling | 6 | blocked | pytest/CTRF; preserve local PDF artifact copy |
| dependency-vulnerability-check | 5 | blocked | common pytest/CTRF profile |
| earthquake-plate-calculation | 6 | blocked | common pytest/CTRF profile |
| enterprise-information-search | 6 | blocked | common pytest/CTRF profile |
| financial-analysis | 6 | blocked | pytest, CTRF, pandas |
| offer-letter-generator | 6 | blocked | pytest, CTRF, python-docx |
| organize-messy-files | 6 | blocked | pytest, CTRF, PyPDF2 |
| schedule-planning | 5 | blocked | common pytest/CTRF profile; preserve `RESULTS_PATH` |
| stock-data-visualization | 5 | blocked | common pytest/CTRF profile |
| temperature-simulation | 5 | blocked | pytest plus local CTRF receipt wrapper |
| travel-planning | 5 | blocked | common pytest/CTRF profile |
| video-object-counting | 5 | blocked | pytest, CTRF, pandas, OpenCV |
| weighted-gdp-calculation | 6 | blocked | openpyxl plus local `ssconvert` prelude |
| fix-security-bug | 3 | blocked | requests plus local Druid restart/readiness prelude |
| nlp-paper-reproduction | 3 | blocked | large pinned Torch/Transformers/DeepSpeed stack |
| python-scala-translation | 2 | blocked | prebuilt Scala/coursier toolchain; no GitHub/Maven fetch |
| github-repo-analytics | 5 | excluded | external `GH_TOKEN` credential rule |

Current credential-independent offline coverage is 10/95 tasks. This is an
infrastructure coverage figure, not an accuracy result.

## Expansion Order

1. Build and probe the common pytest/CTRF profile families.
2. Add small Python extras: pypinyin, pandas, python-docx, PyPDF2, openpyxl.
3. Preserve each upstream verifier's local semantic prelude instead of merely
   replacing its installation commands.
4. Handle OpenCV separately and measure its one-time wheel size.
5. Keep Scala and NLP blocked until their larger toolchains are deliberately
   prefetched, hashed, and justified for the selected experiment.

Every new profile must pass both a known-pass and known-fail fixture against the
original local tests before it enters a benchmark manifest.
