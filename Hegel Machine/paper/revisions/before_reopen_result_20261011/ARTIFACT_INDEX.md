# Experimental materials

Open [manuscript.md](manuscript.md) for the main argument and [supplementary.md](supplementary.md) for Appendices A–H. Links preserve the extracted project's directory layout. Existing results, numerical replay, and new language-model authorship are different operations; the package records which is available.

## Main evidence

| Study | Protocol and results | Source and execution status |
|---|---|---|
| Retention after a measured failure, §5.3 | [Protocol](../artifacts/paper_scoped_retention_20261005/protocol.json), [30-case summary](../artifacts/paper_scoped_retention_20261005/summary.json) | [Original runner](../artifacts/paper_scoped_retention_20261005/run_cohort.py) and [saved runtime](../artifacts/paper_scoped_retention_20261005/runtime/). This is a retained source snapshot. The 90 shared-prefix tasks and 90 successors remain separate in the summary. |
| Measurement/model revision and procedure availability, §§5.1–5.2 | [Protocol](../artifacts/growth_paper_20261006_main_30/protocol.json), [summary](../artifacts/growth_paper_20261006_main_30/summary.json), [derived joint outcomes and procedure intervention](../artifacts/growth_paper_20261006_main_30/derived_analysis.json) | [Saved source bundle](../artifacts/growth_paper_20261006_frozen_source/), [analysis script](../scripts/analyze_growth_paper_study.py). The large per-case raw records needed to recompute this analysis are outside the selective review package. |
| Frozen method revisions, §5.2/Table 5 | Separate [18-target protocol](../artifacts/temporal_counterexample_learning_luna_20261006/protocol.json)/[results](../artifacts/temporal_counterexample_learning_luna_20261006/summary.json), [32-target protocol](../artifacts/temporal_evidence_scope_learning_luna_20261006/protocol.json)/[results](../artifacts/temporal_evidence_scope_learning_luna_20261006/summary.json), and [56-target protocol](../artifacts/temporal_selective_preservation_luna_20261006/protocol.json)/[results](../artifacts/temporal_selective_preservation_luna_20261006/summary.json) | Frozen published procedures: [18-target source](../artifacts/temporal_counterexample_learning_luna_20261006/learning/shared/consolidated_method.py), [32-target source](../artifacts/temporal_evidence_scope_learning_luna_20261006/study/learning/shared/consolidated_method.py), [56-target source](../artifacts/temporal_selective_preservation_luna_20261006/study/learning/shared/consolidated_method.py). [18-target source snapshot](../artifacts/temporal_counterexample_learning_luna_20261006/source_at_run/) and [56-target source snapshot](../artifacts/temporal_selective_preservation_luna_20261006/source_at_run/) are partial execution snapshots; the [32-target saved runner](../artifacts/temporal_evidence_scope_learning_luna_20261006/runner_at_start.py) is not a complete runtime freeze. |
| Online supporting research and strong flat controls, §5.1 | [Finite-language protocol](../artifacts/online_inquiry_20261005_main/protocol.json)/[results](../artifacts/online_inquiry_20261005_main/summary.json); separate [temporal protocol](../artifacts/temporal_matched_repertoire_20261006/plan.json)/[results](../artifacts/temporal_matched_repertoire_20261006/summary.json) | [Finite-language saved source](../artifacts/online_inquiry_20261005_main/source/). The temporal [comparison runner](../scripts/compare_temporal_research.py) and [flat implementation](../src/hegel_machine/cognition/temporal_flat_research.py) are current source; no complete immutable temporal runtime bundle is identified here. |
| Complete common-model agents, §5.1/Table 4 | [Protocol](../artifacts/online_inquiry_luna_subscription_20261005_main_30/protocol.json), [results](../artifacts/online_inquiry_luna_subscription_20261005_main_30/summary.json), [run interpretation](../artifacts/online_inquiry_luna_subscription_20261005_main_30/RUN_STATUS.md) | [Saved source files](../artifacts/online_inquiry_luna_subscription_20261005_main_30/frozen_source/). New execution of [the agent runner](../scripts/run_online_inquiry_llm.py) calls a model service; it is new agent research, not deterministic result reconstruction. |
| Autonomous recall from one regressed method, §5.2 and Appendix F.11 | [Protocol](../artifacts/python_autonomous_recall_luna_20261007/protocol.json), [20-author summary](../artifacts/python_autonomous_recall_luna_20261007/summary.json), [cost/error extraction](../artifacts/python_autonomous_recall_luna_20261007/derived_analysis.json) | [Saved author runner](../artifacts/python_autonomous_recall_luna_20261007/run_python_autonomous_recall_study_at_start.py), [saved method interface](../artifacts/python_autonomous_recall_luna_20261007/python_method_learning_at_start.py), [saved provider-context adapter](../artifacts/python_autonomous_recall_luna_20261007/code_provider_context_at_start.py). These are source files, not a complete immutable execution environment. Failed authors remain in the registered denominator. |
| New transfer of distinct recalled programs, §5.2 | [Protocol](../artifacts/python_recall_frozen_transfer_20261007/protocol.json), [full results](../artifacts/python_recall_frozen_transfer_20261007/summary.json), [compact results](../artifacts/python_recall_frozen_transfer_20261007/brief.json), [all source acquisition outcomes](../artifacts/python_recall_frozen_transfer_20261007/source_outcomes.json) | [Saved runner](../artifacts/python_recall_frozen_transfer_20261007/runner_at_start.py); four frozen sources [P0](../artifacts/python_recall_frozen_transfer_20261007/published_0.py), [P1](../artifacts/python_recall_frozen_transfer_20261007/published_1.py), [P2](../artifacts/python_recall_frozen_transfer_20261007/published_2.py), [P3](../artifacts/python_recall_frozen_transfer_20261007/published_3.py), and [fixed repair](../artifacts/python_recall_frozen_transfer_20261007/fixed_repair.py). Published source and runner are saved; their shared executor dependencies are current source. This replay uses zero target-time model calls. |
| Executable hypothesis assignment, §5.2 and Appendix F.13 | [Fixed protocol](../artifacts/response_assignment_comparison_20261010/protocol.json), [six-run totals](../artifacts/response_assignment_comparison_20261010/summary.json), [per-run declarations and physical traces](../artifacts/response_assignment_comparison_20261010/brief.json) | [Executed runner](../artifacts/response_assignment_comparison_20261010/run_response_assignment_comparison.py) and [saved local runtime](../artifacts/response_assignment_comparison_20261010/frozen_runtime/). Six operator-requested continuations of one development failure; two Hegel assignments, not a complete-agent or recursion comparison. |
| Historical common-prefix SC-KG, §5.4 | [Protocol](../artifacts/sc_kg_structural_prior_comparison_20261007/protocol.json), [results](../artifacts/sc_kg_structural_prior_comparison_20261007/summary.json), [compact comparison](../artifacts/sc_kg_structural_prior_comparison_20261007/brief.json), [historical evidence extraction](../artifacts/kg_op_paper_integration_20261007/evidence.json) | [Extracted historical KG backend](../artifacts/sc_kg_structural_prior_comparison_20261007/kg_source/), [historical Hegel method overrides](../artifacts/sc_kg_goal_completion_20260911/method_source/), [historical baseline overrides](../artifacts/sc_kg_unknown_revision_20260911/baseline_source_v052/), and [current comparison runner](../scripts/run_sc_kg_structural_prior_comparison.py). The original prefix/evaluation records and external KG checkout are dependencies of replay; the comparison does not freeze an entire machine environment. |

The native zero-observation extension in §5.2 and Appendix F.14 has completed all twelve runs: [protocol](../artifacts/response_revision_native_20261010/protocol.json), [totals](../artifacts/response_revision_native_20261010/summary.json), [per-run traces](../artifacts/response_revision_native_20261010/brief.json), and [C5-specific analysis and cache reconciliation](../artifacts/response_revision_native_20261010/analysis.json). The [runner](../artifacts/response_revision_native_20261010/run_response_revision_replay.py), [prepared archive](../artifacts/response_revision_native_20261010/frozen_bundle.json), and [launch-time source snapshot](../artifacts/response_revision_native_20261010/frozen_runtime/) are retained. This source snapshot covers modules loaded at launch, not all dynamic dependencies. All unsuccessful runs, including one fixture assertion, remain in the denominator.

The return intervention in Appendix F.14 uses [all 24 recorded decisions and original limits](../artifacts/response_native_parent_return_20261010/protocol.json), [paired results](../artifacts/response_native_parent_return_20261010/summary.json), and the [four-file runtime patch](../artifacts/response_native_parent_return_20261010/reference_return.patch). Its initial assembly failure and completed second attempt remain in the [execution record](../artifacts/response_native_parent_return_20261010/RUN_STATUS.md). Missing dynamic dependencies are shared by both replay arms and retained separately from the launch snapshot. A [separate two-decision diagnostic](../artifacts/response_native_two_decision_replay_20261010/RUN_STATUS.md) records the rejected direct-support candidate. These are zero-new-model-call execution comparisons, not new authoring attempts.

Appendix F.15 includes the completed [premature-return diagnosis and fixed continuation](../artifacts/response_refuted_author_return_20261010/RUN_STATUS.md), [protocol](../artifacts/response_refuted_author_return_20261010/live/protocol.json), [six outcomes](../artifacts/response_refuted_author_return_20261010/live/summary.json), and [reproducible aggregation](../artifacts/response_refuted_author_return_20261010/analyze_live.py). The original and repaired policies both use Hegel, replay the same three decisions, and then permit 21 new calls. Before/after completion and requested-point confirmation are 0/3 and 1/3; all failed outcomes remain. The common runtime snapshot and saved inquiry variants distinguish replay from new model authoring.

The same appendix retains the subsequent [integrated-connection comparison](../artifacts/response_research_connections_live_20261010/RUN_STATUS.md), [protocol](../artifacts/response_research_connections_live_20261010/protocol.json), [six outcomes](../artifacts/response_research_connections_live_20261010/live/summary.json) and [aggregation](../artifacts/response_research_connections_live_20261010/analyze.py). Both arms are Hegel: task completion is 3/3 before versus 1/3 after integration, requested-point confirmation 0/3 versus 1/3, and fresh calls 27 versus 43. Full costs include the diverging recorded-action prefix. Derived analyses of both cohorts use raw provider completions for cache-call counts; original run summaries are preserved.

The replay retains its [shared dynamic dependencies](../artifacts/response_native_parent_return_20261010/runtime_dependencies/), [changed runtime files](../artifacts/response_native_parent_return_20261010/runtime_changes/), [original archive](../artifacts/response_native_parent_return_20261010/original_frozen_bundle.json), and actual prepared programs for [before](../artifacts/response_native_parent_return_20261010/attempt_02/before/frozen_bundle.json) and [after](../artifacts/response_native_parent_return_20261010/attempt_02/parent_return/frozen_bundle.json). These source records complement the original launch-time snapshot; they do not freeze Python packages or a complete machine environment.

## Framework and implementation

Main §§3.2–3.3 develop the functional-analytic perspective: observations induce distinguishability on prediction spaces, and research can revise observations, representations and explanatory scope. Supplement A.3 gives the elementary derivation; B.2 maps it to frozen online and growth sources, including the executed probe change in [chain 79101013](../artifacts/online_inquiry_20261005_main/case_79101013/case.json). These are mathematical exposition and existing evidence, with no new algorithm or experiment.

## Revision analyses

The [stopping-trace audit](../artifacts/paper_retention_stopping_audit_20261007/) recovers first-change stopping behavior through the [original-controller replay script](../scripts/audit_scoped_retention_stopping.py). Its [cost contract](../artifacts/paper_retention_stopping_audit_20261007/cost_contract.json) distinguishes charged operations from elapsed time. The audit preserves original results and reports exact-match checks for the rerun cases.

The [archive-control extraction](../artifacts/executable_archive_controls_20261007/evidence.json) and [analysis script](../scripts/summarize_executable_archive_controls.py) separate the conditional-marker ablation, matched flat control and ordinary Python frozen execution. This is a read-only aggregation, not a newly executed generic-archive comparison.

The assignment comparison in Appendix F.13 restores the same complete 184-observation
failure state in all six runs. Its compact record retains the authored declarations,
actual physical trials, new costs and outcomes for every run. The full per-run result
files and restored provider state remain in the research checkout; they are not
included in this selective package. The [first successful source before](../artifacts/response_assignment_comparison_20261010/repeat_01/historical_assignment/source_before.py)
and [after](../artifacts/response_assignment_comparison_20261010/repeat_01/historical_assignment/source_after.py)
show the added declaration. The subsequent [paid-exclusion replay result](../artifacts/response_assignment_comparison_20261010/paid_exclusion_replay/result.json),
[compact verification](../artifacts/response_assignment_comparison_20261010/paid_exclusion_replay/verified.json)
and [changed executor](../artifacts/response_assignment_comparison_20261010/paid_exclusion_replay/world_reference_hypothesis.py)
record a separate zero-model-call replay of that proposal. Its [replay entry point](../artifacts/response_assignment_comparison_20261010/paid_exclusion_replay/replay.py)
requires the original failure state and complete checkout. Replaying an already
authored proposal is distinct from resampling the model's adoption decision.

## Reading and execution

Run commands from the project or extracted package root. The included [table generator](../scripts/regenerate_manuscript_tables.py) regenerates main Tables 4–7 directly from saved result and protocol JSON. It uses only the Python standard library and makes no simulator or model calls. The main-document builder uses the same table bodies before rendering, so editing narrative text cannot silently become the numerical data source:

```bash
python3 scripts/regenerate_manuscript_tables.py --output /tmp/hegel-tables.md --json-output /tmp/hegel-tables.json
```

The JSON output also gives input paths and recomputed retention and selective-revision resource differences. Table 5 keeps the failed [initial 44-target selective cohort](../artifacts/temporal_selective_revision_luna_20261006/summary.json) distinct from the later 56-target cohort. Tables 1–3 organize concepts and experimental roles rather than numerical results.

Python's JSON reader is sufficient to inspect saved numbers, for example:

```bash
python3 -m json.tool artifacts/python_recall_frozen_transfer_20261007/brief.json
python3 -m json.tool artifacts/paper_scoped_retention_20261005/summary.json
```

One representative numerical experiment runs directly from the extracted bundle, with no model service, additional dataset, or nonstandard Python dependency. This command reruns the original six-chain development panel through the frozen online source (78 source/continuation episodes); choose a new output directory. It verifies executable reproduction and does not add independent evidence to the paper's main cohort:

```bash
env -u PYTHONPATH python3 artifacts/online_inquiry_20261005_main/source/scripts/run_online_inquiry_study.py --profile development --output /tmp/hegel-online-development-replay --workers 1
```

The clean Python path keeps the frozen source separate from a checkout supplied through `PYTHONPATH`. Its existing `--profile main` option reruns the original 30-chain panel. No author decisions or model outputs are resampled by either option.

Local bundle verification on 2026-10-07 regenerated Tables 4–7 from the extracted result files and completed all six development chains with zero execution failures and zero model calls. Each of the recursive, flat and frozen-method continuations completed all 24 targets correctly. This verifies the supplied replay route; it is not an additional scientific replication cohort.

For F.12, `artifacts/python_recall_frozen_transfer_20261007/` contains all four `published_0.py`–`published_3.py` sources, `fixed_repair.py`, the 24 target specifications in `protocol.json.transfer_cases`, and the 120 program/target outcomes in `summary.json.results`. Each result's `program_id` and `target_id` identify its source and target; the targets are paired, not 120 independent tasks. These files support source inspection and outcome reconstruction within the bundle. The original transfer runner additionally reads historical author records and donor predictor snapshots through saved absolute paths; its full numerical replay still requires those archives in the complete checkout. Those paths are provenance, not portable bundle links.

In a complete research checkout, the following commands recompute analyses from existing local records, without simulator or model calls. They write new outputs:

```bash
python3 scripts/analyze_growth_paper_study.py --input artifacts/growth_paper_20261006_main_30 --output /tmp/growth-analysis.json
python3 scripts/summarize_executable_archive_controls.py --output /tmp/archive-control-analysis.json
```

Numerical replay executes experiments again. Use a new output directory and the dependencies stated above. These entry points make no LLM calls:

```bash
python3 artifacts/online_inquiry_20261005_main/source/scripts/run_online_inquiry_study.py --profile main --output /tmp/hegel-online-replay --workers 1
python3 scripts/compare_temporal_research.py --pairs 30 --seed 86106007 --output /tmp/hegel-flat-replay
python3 scripts/run_python_recall_frozen_transfer.py --output /tmp/hegel-recall-transfer-replay
python3 scripts/audit_scoped_retention_stopping.py --apparatus 701045 --output /tmp/hegel-stopping-replay
```

The temporal comparison and recall-transfer commands use current shared dependencies. They do not by themselves reproduce an immutable historical software environment. The retention cohort's saved `run_cohort.py` writes beside itself; replay it only in a fresh copy containing its `protocol.json` and `runtime/`, without copying old case outputs. The growth snapshot's [runtime record](../artifacts/growth_paper_20261006_frozen_source/runtime.json) gives its original command.

The [method-revision runner](../scripts/run_temporal_counterexample_learning.py), [scope-revision runner](../scripts/run_temporal_evidence_scope_learning.py), [selective-preservation runner](../scripts/run_temporal_selective_preservation.py) and [autonomous-recall runner](../scripts/run_python_autonomous_recall_study.py) include new LLM authorship. Running them acquires new research behavior and cost; it does not reconstruct the recorded author decisions. Existing authored source remains the input for frozen replay.

## Package and limitations

Preview or create the selective bundle after both PDFs have been compiled:

```bash
python3 scripts/package_manuscript_evidence.py --dry-run
python3 scripts/package_manuscript_evidence.py
```

The output is `paper/hegel_machine_review_materials.zip`. Extract it as one directory and open `paper/ARTIFACT_INDEX.md`; relative Markdown links retain the project layout. The bundle contains the main and supplementary MD/TeX/PDF files, figures, linked local evidence, saved source files and the local Python dependencies of linked analysis scripts. `paper/README.md` explains that the MD files are the editing sources and gives commands for the included [document builder](../scripts/build_manuscript.py). `paper/BUNDLE_CONTENTS.json` lists the included paths and distinguishes current source files.

This is a selective review bundle. Large unlinked case records, raw provider logs, environments, credentials, CSV datasets and checkpoints are excluded. Original result JSON may contain absolute provenance paths from the research machine; these are preserved, not presented as portable links. Analysis or replay requiring those omitted archives needs the complete checkout and path rebinding. Partial snapshots and current dependencies are identified above. The package supplies inspectable evidence and the available replay entry points; it is not an immutable full-environment reproduction or a permanent external repository.
