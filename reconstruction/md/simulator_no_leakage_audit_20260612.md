# Simulator No-Leakage Audit

- pass: `True`
- rows: `2160/2160`
- state feature leak count: `0`
- provenance leak count: `0`
- prediction/outcome exact identity count: `0`
- prediction/outcome near-identity rate: `0.0134`
- mean prediction/outcome gap: `0.0508`
- best-arm agreement: `0.9611`
- source direct alias count: `0`

## Claim Boundary

The audit supports a leakage-audited triage/router simulator claim only.  It does not permit
replacing live ablation, judge evidence, or external validation.
