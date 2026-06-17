# HLE Anti-Pollution And Verified-Abstain Follow-up

Date: 2026-06-17

## Scope

This note records the HLE work after the anti-pollution review:

1. Run a fresh anti-pollution diagnostic line.
2. Add a verified-or-abstain hard gate.
3. Keep the remaining executable verifier / answer-bearing retrieval / routing-only morphism / module ablation work explicit.

Raw HLE questions, answers, rationales, canaries, and prediction text were not persisted.

## Implemented

- Added `verified_or_abstain` selection gate in `assumption_os/hle_smoke_eval.py`.
- Added HLE failure diagnostics in `assumption_os/hle_parallel_shard_runner.py`.
- Added sanitized endpoint error labels, including `RemoteDisconnected`, to top-level and jsonl error stratification.
- Added answer-bearing evidence filtering:
  - evidence must overlap question terms and a candidate/option;
  - multiple-choice evidence must be discriminative for one option family;
  - MC evidence bridge is off by default until option-specific retrieval is implemented.
- Applied the same answer-bearing gate to the Assumption Agent internal HippoRAG child context.
- Adjusted morphism credit assignment:
  - weak morphism only counts as unhelpful if it was actually injected into answer context;
  - routing-only weak morphism is tracked as `weak_morphism_routing_only_not_credited`.

## Validation

Unit validation:

- `python3 -m unittest tests.test_hle_smoke_eval tests.test_hle_parallel_shard_runner`
- Result: 65 tests OK.

Fresh dry-run:

- Eval: `hle_parallel_diag_dryrun_n12_verified_abstain_seed2300_stride17_20260617`
- Result: pass=true, paper_clean_pass=true, pollution_pass=true.
- Sample: 12 distinct problems, 0 duplicates.

Fresh live diagnostic:

- Eval: `hle_parallel_diag_live_n12_verified_abstain_seed2300_stride17_gpt54mini_soft9000_global3_20260617`
- Result: pass=true, pollution_pass=true, paper_clean_pass=false.
- Reason paper-clean failed: 5 top-level endpoint errors, all separated as RuntimeError / RemoteDisconnected.
- Clean shared subset: agent 0.125, raw 0.125, HippoRAG 0.125.
- Verified-or-abstain status: 11 abstained, 1 allowed.
- Main diagnosis: verified-abstain prevented unverified overrides, but fallback was raw-like and gave no accuracy gain.

Focused evidence-gate validation:

- Eval before final MC block: `hle_evidence_gate_focused_rerun_n1_seed1843_gpt54mini_20260617`
- Result: paper_clean_pass=true, pollution_pass=true.
- Evidence bridge correctly blocked ambiguous multi-option HippoRAG child context, but one MC evidence bridge still activated and was wrong.

Focused MC context block validation:

- Eval: `hle_mc_context_block_focused_n1_seed1843_gpt54mini_20260617`
- Result: paper_clean_pass=true, pollution_pass=true.
- Failure buckets no longer included `evidence_invalid_or_unhelpful` or `hipporag_context_invalid_or_unhelpful`.

Focused morphism credit validation:

- Eval: `hle_context_credit_focused_n1_seed1843_gpt54mini_20260617`
- Result: paper_clean_pass=true, pollution_pass=true.
- Failure bucket changed from weak morphism as answer failure to `weak_morphism_routing_only_not_credited`.

## Current Interpretation

The new gates improve measurement quality and prevent polluted context from being credited as cognition. They do not yet improve HLE answer accuracy: on the clean shared HLE slices, the agent is currently tied with raw rather than better.

This is the right failure mode for the next step: the system now abstains instead of injecting weak context, so the remaining work must create genuinely verified candidates.

## Remaining 3-6

3. Executable verifier:
   - Generate candidate -> formal/executable claim -> verifier check.
   - Verified candidates can override raw/majority.
   - Prioritize math exact and structured MC constraints.

4. Answer-bearing retrieval:
   - Replace direct MC evidence injection with option-specific retrieval.
   - Retrieve evidence per option, compute discriminative margins, and only emit a candidate when one option has strong support.

5. Morphism/graph routing:
   - Keep graph/morphism as routing and hypothesis generation by default.
   - Only inject morphism context when a strong certificate exists.

6. Automatic module ablation:
   - Run same fresh problems under raw, full agent, no graph, no evidence, no morphism, verified-only.
   - Report problem-level clean shared accuracy and pollution table for each module.

## Next Best Step

Implement the executable verifier and option-specific MC evidence scorer as first-class candidate generators, then run a fresh n=12/n=30 clean shared validation. The goal is no longer "more context"; it is "verified candidate or abstain."
