# Orthogonal next-generation promotion failures - 2026-06-09

This log records the attempted next-generation descendants after the accepted
`cand_f8ca2582dbc4` live descendant.  None of these attempts was pushed to the
main graph because every real live gate rejected promotion.

## Business/action nextgen

Parent: `cand_f8ca2582dbc4`

1. Scope-only retained child: `prop_d44aae0f9127`
   - Trigger winners against `phase2_v20_claude_opus_execution_baseline`: candidate 1, baseline 3.
   - Controls: 8 route-scoped no-op ties.
   - Result: `reject_benefit`.
   - Residual: simply narrowing to the previously winning rows removed useful specificity.  The baseline was better on city/channel posterior updating, secrecy contract structure, and staged education-transition details.

2. Residual-specific business repair: `prop_99b7c2f9b052`
   - Trigger winners against `phase2_v20_claude_opus_execution_baseline`: candidate 1, baseline 2, tie 1.
   - Controls: 8 route-scoped no-op ties.
   - Result: `reject_benefit`.
   - Residual: adding the missing bridge fields improved one production-line row to tie, but did not reliably beat the strong same-model baseline.

## Technical sibling branch

Parent: `cand_f8ca2582dbc4`

1. Five-trigger technical child: `prop_584773b088ff`
   - Trigger winners against `phase2_v20_claude_opus_technical_baseline`: candidate 3, baseline 1, tie 1.
   - Controls: 8 route-scoped no-op ties.
   - Result: positive point estimate, but `reject_benefit` because the LCB90 did not clear the acceptance gate.
   - Residual: API-DX should become a separate DX/prototype child; release pipeline was only a tie.

2. Three-trigger selective-retention child: `prop_412034c92b89`
   - Trigger winners against `phase2_v20_claude_opus_technical_baseline`: candidate 1, baseline 1, tie 1.
   - Controls: 8 route-scoped no-op ties.
   - Result: `reject_benefit`.
   - Residual: the previous three wins were not stable under fresh generation.

3. Residual-specific technical repair: `prop_6c22137d982d`
   - Against `phase2_v20_claude_opus_technical_baseline`: candidate 1, ties 2.
   - Against parent `proposal_d7abf65010d2_technical_parent`: parent 1, ties 2.
   - Controls: 8 route-scoped no-op ties.
   - Result: `reject_benefit` in both absolute-baseline and parent-comparison modes.
   - Residual: the child mostly matched parent/baseline quality but did not add enough marginal value for promotion.

## Interpretation

The recursive loop is functioning correctly: failure generated child proposals,
the children were scoped and repaired, live answer/judge validation ran, and
the acceptance gate rejected proposals whose benefit was not strong enough.

The next productive direction is not to keep narrowing these same prompts.  It
is to improve the proposal generator/world model so it predicts when a child is
only likely to tie the parent, and to create separate DX/prototype and pipeline
bottleneck children with enough independent trigger rows before spending live
calls.
