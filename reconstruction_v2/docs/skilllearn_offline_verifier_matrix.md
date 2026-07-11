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
- **cataloged experimental**: a semantic runtime exists for diagnostics, but an
  explicit activation blocker keeps it out of paper trials;
- **upstream-blocked**: the local upstream checkout lacks an authoritative
  verifier payload, so no score can be produced without modifying the benchmark;
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
| weighted-gdp-calculation | 6 | 5 ready, 1 upstream-blocked | `weighted-gdp-ssconvert-py312-v1`; instance 2 lacks `test_outputs.py` in upstream Git HEAD |
| fix-security-bug | 3 | cataloged experimental, blocked | `druid-security-py312-v1`; `druid_maven_cache_incomplete` |
| nlp-paper-reproduction | 3 | blocked | large pinned Torch/Transformers/DeepSpeed stack |
| python-scala-translation | 2 | blocked | prebuilt Scala/coursier toolchain; no GitHub/Maven fetch |
| github-repo-analytics | 5 | excluded | external `GH_TOKEN` credential rule |

Current credential-independent runnable coverage is 86/95 tasks. The full local
inventory is 86 ready, 9 blocked, and 5 credential-excluded tasks. The nine
blocked tasks are one incomplete upstream GDP verifier, three Druid tasks,
three NLP tasks, and two Scala tasks. This is an infrastructure coverage
figure, not an accuracy result.

## Validation Evidence

The 2026-07-11 v4 matrix used only train-split representatives. It made no model
call and did not access sealed-test content. All seven active profiles produced reward 1
for a one-test known-pass fixture and reward 0 for a one-test known-fail fixture.
A complete train representative from each of the 15 active families then
executed its original local `test_outputs.py` in a disposable controlled
workspace, producing a non-empty CTRF and the expected reward 0. The family
probes collected between 2 and 27 tests each. Profiles with semantic input state
use the immutable image root; other profiles use an empty `/root` mount.

The weighted-GDP prelude ran `ssconvert` with exit code 0, produced two sheet
CSVs, and collected 27 original tests. A separate reference-solution diagnostic
on the matching item-3 image completed with solution exit 0, prelude exit 0,
reward 1, and 27/27 original tests passing under Docker `--network none`.
Provider credentials are removed before every verifier or child service starts.

The active capability matrix is 7/7 profiles and 15/15 families. Its overall
`passed` field remains false by design: the selected train manifest has one
incomplete GDP verifier payload, one task backed only by an inactive profile,
and two tasks with no local profile. The v4 report records all three blocker
classes and does not substitute another instance's assertions.

The Druid diagnostic is deliberately not active. With a local hostname repair,
the immutable image can start Druid and collect all four security tests without
network access. However, the upstream reference solution's Maven build fails
because the image-local `.m2` cache lacks plugin dependencies. The semantic
receipt records `deployed_jar_count=0`, yet the original verifier still reports
4/4 passing against the base service plus source-tree patch. That is a verifier
false-positive path, not proof that the patched binary was built and deployed.
Offline Maven probes also fail on missing clean/checkstyle plugin dependencies.
The catalog therefore retains the diagnostic profile but the active profile
lookup returns no Druid runtime and rejects the task before model execution.

This proves dependency availability, Python ABI compatibility, wrapper reward
semantics, and original-test executability. It does not prove task accuracy or
that an arbitrary model output will pass. Real benchmark trials still require a
valid CTRF/reward receipt and are scored by the unchanged original assertions.

## Remaining Work

1. Obtain an authoritative verifier for `weighted-gdp-calculation-2` from the
   benchmark maintainers or a pinned upstream release; do not synthesize it.
2. Build and hash a complete Druid Maven corpus from an explicitly approved
   mirror, then require a non-zero deployed-JAR receipt and a negative vulnerable
   control before removing `druid_maven_cache_incomplete`.
3. Keep Scala and NLP blocked until their larger toolchains are deliberately
   prefetched, hashed, and justified for the selected experiment.

Every new profile must pass the same pass/fail contract and original-family
execution matrix before it enters a benchmark manifest.
