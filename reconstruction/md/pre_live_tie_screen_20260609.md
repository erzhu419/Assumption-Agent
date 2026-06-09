# Pre-Live Tie / Low-Benefit Screen Log

Date: 2026-06-09

## Motivation

The orthogonal descendant line proved the recursive runner can generate and
validate real descendants, but the next generation exposed a cost/productivity
gap: several descendants passed preflight and then failed live acceptance as
low-benefit or underpowered ties.

This patch adds a pre-live screen before expensive answer/judge calls.  It
uses only prior sibling/descendant evidence in the same family, not gold
answers or API secrets.

## Result

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/pre_live_tie_screen_20260609.json`

Chronological replay over seven orthogonal descendant validations:

- no screen: 7 live calls, 1 accepted, 6 failed
- with screen: 3 live calls, 1 accepted, 2 failed
- failed live calls saved: 4
- live call reduction: 0.5714
- accepted positive blocked: 0
- accepted rate among run calls: 0.1429 -> 0.3333

Saved failed calls:

- `prop_99b7c2f9b052`
- `prop_412034c92b89`
- `prop_6c22137d982d`
- `prop_6c22137d982d_vs_parent`

Still allowed exploratory failures:

- `prop_d44aae0f9127`
- `prop_584773b088ff`

These were first-in-family or first-in-cluster probes.  Blocking them would be
too aggressive without prior evidence.

## Interpretation

This is a recursive self-evolution productivity improvement, not a direct QA
quality improvement.  The system now has a concrete budget gate between:

1. proposal generation,
2. preflight readiness,
3. live answer/judge spend,
4. acceptance/readback.

The gate preserves the live-positive seed while avoiding repeated descendants
that reuse low-utility scopes or narrow underpowered siblings.

## Validation

Commands:

```bash
python3 -m py_compile assumption_os/pre_live_tie_screen.py
python3 -m unittest tests.test_assumption_os.AssumptionOSTest.test_pre_live_tie_screen_preserves_positive_and_saves_failed_descendant_calls
python3 -m assumption_os.pre_live_tie_screen --root . --eval-id pre_live_tie_screen_20260609 --out 'phase four/assumption_graph/paper_readiness_20260604/pre_live_tie_screen_20260609.json'
```

All gates passed in the generated artifact.
