# HLE Strong-Child Recursive Runner Findings

Date: 2026-06-18

## What changed

The HLE runner now supports fixed problem-hash manifests, shard-level retry/reaggregation, and paired run comparison. This lets us run fresh HLE evaluations without persisting raw HLE questions, answers, rationales, canaries, or prediction text.

The effective HLE improvement came from making the recursive child/critic/planner path stronger, while keeping the base answer model fixed:

- base answer model: `gpt-5.4-mini`
- recursive child model: `gpt-5.5`
- critic model: `gpt-5.5`
- candidate claim planner model: `gpt-5.5`

This is a real recursive-assumption effect: the base raw model did not change, and the lift appears when the agent gets stronger multi-candidate recursive verification.

## Same-hash n=30 result

Run: `hle_ablation_n30_samehash_full_gpt55_child_nohardtimeout_20260617`

- sample: 30 fixed HLE multiple-choice problem hashes
- agent: 16/30 = 0.5333
- raw: 7/30 = 0.2333
- HippoRAG baseline: 6/30 = 0.2000
- original full agent: 8/30 = 0.2667
- endpoint errors: 0
- duplicate samples: 0

Paired deltas:

- agent vs raw: +0.3000, wins 9, losses 0, sign-test p = 0.00390625, CI95 [0.1333, 0.4667]
- agent vs HippoRAG: +0.3333, wins 11, losses 1, sign-test p = 0.00634765625, CI95 [0.1333, 0.5333]
- agent vs original full agent: +0.2667, wins 9, losses 1, sign-test p = 0.021484375, CI95 [0.1000, 0.4667]

## Fresh-hash n=60 result

Run: `hle_ablation_n60_freshhash_strongchild_triad_nohardtimeout_20260618`

The manifest excluded 206 previously seen HLE problem hashes and selected 60 fresh text-only multiple-choice hashes. It stores hashes only.

- sample: 60 distinct fresh HLE multiple-choice problem hashes
- agent: 24/60 = 0.4000
- raw: 8/60 = 0.1333
- HippoRAG baseline: 10/60 = 0.1667
- planned live rows: 180
- resolved live rows: 180
- underlying model calls: 477
- endpoint errors: 0
- process timeouts: 0
- duplicate samples: 0
- paper-clean pass: true
- pollution pass: true

Paired deltas:

- agent vs raw: +0.2667, wins 18, losses 2, sign-test p = 0.0004024505615234375, CI95 [0.1333, 0.4000]
- agent vs HippoRAG: +0.2333, wins 17, losses 3, sign-test p = 0.0025768280029296875, CI95 [0.1000, 0.3667]

## Module interpretation

The HLE gain is not coming from generic graph context alone.

- recursive child validation activated on 60/60 and reached 0.4000 accuracy
- diverse recursive candidates activated on 60/60 and reached 0.4000 accuracy
- candidate-claim verifier priority fired on 4/60 and reached 0.7500 accuracy
- verified-or-abstain direct fallback handled 56/60 and reached 0.3750 accuracy
- morphism hits appeared on 32/60, accuracy 0.3438
- morphism context injection appeared on 6/60, accuracy 0.5000
- world-model context appeared on 11/60, accuracy 0.4545

The strongest remaining bottleneck is selection and verification, not endpoint reliability:

- agent wrong/error: 36/60
- all three wrong: 33/60
- verified-or-abstain fallback wrong: 35/60
- raw-correct agent regression: 2/60

## Boundary

This supports the claim that stronger recursive self-verification materially improves HLE multiple-choice performance over raw and HippoRAG on fresh problem hashes. It does not yet prove that the current world model is a production simulator or that morphism alone drives the gain.

Next useful optimization is selective strong-child routing: call `gpt-5.5` children only when uncertainty or candidate disagreement predicts positive value, then validate cost-adjusted accuracy against this n=60 line.
