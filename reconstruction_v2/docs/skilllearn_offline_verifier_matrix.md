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

The paper protocol now freezes exactly those 86 ready items before any new
model call. It filters the earlier credential-independent manifests without
reassigning any item to another split:

- instance holdout: 38 train / 16 validation / 32 sealed test;
- family out: 48 train / 11 validation / 27 sealed test;
- 16 eligible families; the family-out split contains 9/2/5 train/validation/test
  families.

This is a preregistered infrastructure subset, not an outcome-based exclusion.
The nine blocked items remain a separate coverage-extension workstream and are
not evaluated by an online substitute.

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

The original 95-item capability matrix is 7/7 profiles and 15/15 active
families, but its overall `passed` field remains false because the selected
train manifest contains all three blocker classes. The new offline-ready matrix
at `artifacts/offline_verifier_matrix_offline86_20260711_v1/matrix.json` uses
manifest hash `9c7eb39a...` and reports 7/7 profiles, 15/15 train-family probes,
`blockers=[]`, `manifest_execution_ready=true`, and `passed=true`. It made no
model call and did not access sealed-test content. A full manifest-scoped
preflight also passed all required checks for 86 selected items.
The compact, version-controlled summary is
[`skilllearn_offline_readiness_receipt_v1.json`](../manifests/skilllearn_offline_readiness_receipt_v1.json);
the paper protocol and execution lock bind its content hash rather than relying
on an ignored local artifact path.

This readiness receipt is not a claim that all 86 task verifiers were executed.
It combines the 7/7 profile contracts and 15/15 train-family dynamic probes with
an all-selected-item static preflight. Runtime availability is checked
separately: after a bounded cache-preparation pass built 14 missing images, the
cache-only all-manifest prewarm passed 86/86 items, covering 47 unique images and
seven offline-verifier runtimes without an agent call or sealed scoring.

The Druid diagnostic is deliberately not active. A newer zero-download
direct-`javac` reference probe compiled four changed classes against
`/opt/druid/lib/*`, overlaid them into one indexing-service JAR, deployed that
JAR, started Druid offline, and passed the original 4/4 tests. This proves a
complete Maven corpus is not the only possible build route. It is still not an
activation proof: the upstream behavior check accepts any exploit response
`>=400`, the legitimate path need not return 2xx, a source-only/no-deploy run
has also produced 4/4, no vulnerable negative control is established, and only
one reference patch/module path was tested. Direct `javac` is therefore the
preferred future route; Maven prefetch is only a fallback.

This proves dependency availability, Python ABI compatibility, wrapper reward
semantics, and original-test executability. It does not prove task accuracy or
that an arbitrary model output will pass. Real benchmark trials still require a
valid CTRF/reward receipt and are scored by the unchanged original assertions.

## Remaining Work

1. Obtain an authoritative verifier for `weighted-gdp-calculation-2` from the
   benchmark maintainers or a pinned upstream release; do not synthesize it.
2. Turn the Druid direct-`javac` route into a general candidate-build adapter,
   and require a vulnerable negative control, legitimate 2xx path, deployed-JAR
   hash, and source-only/no-deploy failure before activation.
3. For Scala, pin a compatible SBT/compiler-bridge and Circe/ScalaTest closure
   (estimated 100--300 MB) and add a CLI verifier adapter; downloading JARs alone
   is insufficient because the upstream verifier is not pytest-shaped.
4. Treat NLP last: it needs a Python 3.10 runtime shared by agent and verifier
   plus a minimal CPU-only Torch/Transformers/TRL stack (estimated 0.5--1.2 GB
   compressed). Do not pull a default CUDA closure merely to raise coverage.

Every new profile must pass the same pass/fail contract and original-family
execution matrix before it enters a benchmark manifest.
