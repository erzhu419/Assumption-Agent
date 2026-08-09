# Hegel Machine Phase-3 shrink-2 dual complete-enumeration diagnostic result v1

Status: **DUAL DSL_TOO_LARGE — HOST REPLAY PASS — DIAGNOSTIC ONLY**

Implementation basis: f94cf1fb27c6734f24d4510efba0ca3726132706

Evidence record:
artifacts/phase3_shrink2_dual_complete_enumeration_diagnostic_v1.json

## Result

Python and Rust independently produced byte-identical program, chunk, and
bucket-accounting streams for hegel-old-dsl-v1.2.0. A separate target-free host
process reconstructed the frozen typed traversal from an empty state before
consulting either endpoint witness and accepted the result. This host is
independent of endpoint-reported witness selection, but it reuses the committed
Python generator internals and is not a third independent implementation.

| field | observed value |
|---|---|
| diagnostic terminal | DSL_TOO_LARGE |
| archived canonical programs | 50,000 |
| raw operator applications through the fully closed boundary bucket | 3,259,343 |
| rank-50,001 witness hash | a694a8b2d69741d705a50cdd9474fa1dddcd7658cff3ce979bfd2c30eb277574 |
| residual canonical programs in the closed boundary bucket | 673 |
| canonical program archive root | fa3c55cf09f0b370e27a5e1b457e45fe1b48a12b55ebe7d98b76a40e43ff1aff |
| program chunk manifest root | 9823965291c08483a67c1f9c9a97816cfc86d6e849c71ae5958edcb12765a88a |
| bucket accounting root | 8565cb86c3ba98ea642be373de6f7b43ccea95fff0ce7652a2d88bd964fd1d76 |

The host verified exact archive prefix, AST/hash/MDL metadata, program indices,
three embedded diagnostic bindings, chunk framing and hashes, all 175 bucket
rows, untouched post-witness traversal buckets, the complete residual count,
and the witness rank derived without using either endpoint-reported witness.

## Isolation and reproducibility

- Python endpoint: pinned CPython 3.10 image, flags -I -S -B, exact
  eleven-module allowlist.
- Rust endpoint: pinned Rust image, cargo --release --locked --offline, binary
  SHA-256 5874aa9a712b33b1608dc2392c714c39f86daf479e2cc6a1dca7b1b96f73ab76.
- Host replay: pinned CPython image, direct-only entrypoint, exact twelve-module
  allowlist.
- Every container used network none, pull never, a read-only root and source
  mount, dropped all capabilities, and mounted no user secrets.
- Full streams and logs remain in the local external archive whose complete
  file manifest is committed in the evidence record.

The first Rust publication attempt exited before creating its output directory
because the dedicated parent mount was not writable to the container mapping.
The empty failure output and stderr were retained. After changing only that
dedicated parent permission, the same Commit H, image, roots, source snapshot,
and initially fresh target volume (now containing only its first-attempt build
cache) were rerun successfully. No prior evidence file was overwritten.

## Claim boundary

This result does not create child formal roots or execute formal M3. It does not
establish the full closure cardinality beyond the frozen 50,001 boundary,
evaluate odd/sink roles, issue an `OUTSIDE_FROZEN_CLOSURE(...)` or MDL
certificate, sign a formal object, or authorize ACTIVE governance.

The state remains:

    execution_state = NOT_RUN
    formal_roots_generated = false
    formal_roots = null
    formal_state_transition_allowed = false

The preregistered routing consequence is engineering-only: begin shrink step 3,
remove add; retain difference, without promoting any formal status.
