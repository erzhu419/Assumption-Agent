We are debugging an HLE multiple-choice assumption-agent system. Current accepted baseline for this branch is still the prior stable baseline; do not accept new changes unless they improve unseen HLE performance or tie accuracy with clear stability/fidelity gains.

Recent implementation:
1. Wired real source verifier / candidate span bundles into the option matrix source lane.
2. Added a candidate-span-bundle source lane that consumes deterministic option-specific witness bundles rather than only aggregate source verifier audit counters.
3. Kept behavior conservative: source lane can only change selection when it yields a direct source override and there are no existing directness candidates or unresolved span-directness conflicts.
4. Added detailed hash/count logs for:
   - candidate span bundle lane status/reason
   - source-audit fallback lane status/reason
   - source lane behavior block reason
   - witness counts, direct witness counts, rejection counts, bundle hashes, row hashes
5. Added term-hash-aware multi-witness required-term completion:
   - It no longer uses count-only required-term aggregation by default.
   - It only synthesizes an aggregate direct witness when covered required-term hash union completes the option's required-term hash set.
   - Count-only aggregation is behind an explicit env flag and is disabled by default.

Validation:
Unit/focused tests pass:
- `tests/test_hle_option_matrix_router.py`: 11 passed
- `tests/test_hle_smoke_eval.py -k candidate_span_bundle`: 9 passed
- focused source-quality/click tests: 92 passed

Cache-only HLE mini results:
1. Debug replay, same two seeds as previous source-lane probe:
   - `hle_optionmatrix_realbundle_same2_cacheonly_mini_20260706`: 0/2
   - Candidate span bundle lane ran, but both were `ambiguous/direct_pair_bound_margin_too_small`.
   - Existing direct witnesses were not discriminative; margin was ~0.
2. Fresh unseen n=6:
   - `hle_unseen_realbundle_source_lane_n6_cacheonly_mini_20260706`: 1/6
   - No process timeouts or top-level errors.
   - Candidate span bundle source lane was 5/6 `no_candidate/no_direct_pair_bound_span`; one row had no lane detail due disabled/skipped path.
   - Behavior block was `no_source_direct_override`.
   - Main failure buckets: `candidate_generation_missed_gold=5`, `gold_option_source_verifier_unaccepted=5`, `source_quality_promotion_no_direct_span=4`, `verified_or_abstain no_fallback=6`.
3. Debug replay seed1079 after term-hash-aware aggregation:
   - `hle_optionmatrix_termhash_seed1079_cacheonly_mini_20260706`: 0/1
   - Candidate bundle still had no direct witness.
   - Term-hash trace showed the issue clearly: multiple candidate/source-cache witnesses covered the same required term hash, not complementary required terms.

Key diagnostic:
The source lane/router is now connected. The bottleneck is upstream source coverage:

`source/preferred/shared docs -> candidate-specific required-term-complete direct span -> comparator accepted direct candidate -> final selection`

The current failure is not that the router ignores good evidence. It is that local cache/source backfill usually produces:
- relation-ish but incomplete witnesses
- shared/generic witnesses
- candidate-specific rows missing required relation terms
- source-cache rows that cover only the same required term hash repeatedly

Concrete seed1079 term-hash observation:
Required term hashes were like `[h1, h2, h3]`, but top witnesses for candidate options repeatedly covered only `h1`; missing required terms remained `h2/h3`. The new aggregator correctly refused to synthesize a direct witness.

Question:
Given this, what is the highest-leverage next architecture change?

Candidate directions:
A. Build a term-identity-aware targeted source prefetch/backfill loop:
   - For each option, compute missing required term hashes from the current best witness bundle.
   - Map hashes back to internal raw required terms only inside decision/runtime, not persisted.
   - Query local/full-text/S2/OpenAlex/PubMed/PubChem sources for option + missing required term + relation anchor.
   - Cache snippets locally, then replay cache-only.
B. Improve candidate generation so gold/sweep-only options enter source verifier earlier:
   - The source lane cannot help if the gold option is not in candidate summaries or only appears as weak sweep coverage.
C. Add a stronger semantic comparator over the existing incomplete witnesses:
   - But this risks hallucinating relation completion when source coverage is genuinely missing.
D. Add consensus/repeated verifier calls:
   - This seems less useful because term-hash evidence shows source coverage is objectively incomplete, not merely noisy.

What minimal experiment would prove the next change?
Proposed proof:
1. Use a fresh unseen HLE operator/source-bearing n=6 or n=12 cohort, not previous debug seeds.
2. Run cache-only baseline with current code and record:
   - option_with_direct_witness_count
   - covered required-term hash union per option
   - source lane activation rate
   - accuracy
3. Run targeted term-missing source prefetch once, then lock back to cache-only.
4. Replay the same cohort.
5. Accept only if:
   - direct witness coverage increases,
   - source lane activation increases without false-positive ambiguity,
   - accuracy improves or ties with lower no_fallback/source_generic rate,
   - no gold answer is read during decision.

Should we implement A as the next cut, or is there a better way to turn incomplete relation-ish witnesses into reliable answer-bearing evidence without overfitting HLE debug seeds?
