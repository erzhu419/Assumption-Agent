# Hegel Machine Phase-3 shrink-3 dual complete-enumeration diagnostic result v1

Status: **DUAL DSL_TOO_LARGE — HOST REPLAY PASS — DIAGNOSTIC ONLY**

Implementation basis: `d17b03e14f3f3e8a63c924706086f17367fbc0d6`

Evidence record:
`artifacts/phase3_shrink3_dual_complete_enumeration_diagnostic_v1.json`

## Result

The isolated Python and Rust endpoints independently produced byte-identical
program, chunk-manifest, and bucket-accounting streams for
`hegel-old-dsl-v1.3.0`. A separate target-free host process reconstructed the
frozen typed traversal from an empty state before consulting either endpoint
witness and accepted the result. The host is independent of endpoint-reported
witness selection, but it reuses the committed Python generator internals and
is not a third independent enumerator.

| field | observed value |
|---|---|
| diagnostic terminal | `DSL_TOO_LARGE` |
| archived canonical programs | 50,000 |
| raw operator applications through the fully closed boundary bucket | 3,120,739 |
| rank-50,001 witness hash | `040eb443d8e3e422702e9f2b9fc984fb0bccee353a44f32b96e6f8265d4a5975` |
| rank-50,001 witness CBOR | `820183010384020183000001860003050101828201f48202f4` |
| residual canonical programs in the closed boundary bucket | 2,257 |
| canonical program archive root | `67a247db14868b57c4ff4fa5432f98a9d7de26b50ee3f816c82097e187109a4e` |
| program chunk manifest root | `001403eaf4bd9bed257316bdaeb2676b35f0341ba17ef83daed5cec35e17f559` |
| bucket accounting root | `085c0f481820b479809715bdfef710b15d7ce2bd31c2bb9f75062542746bc015` |

The host verified the exact archive prefix, strict AST/hash/MDL metadata,
program indices, all three synthetic diagnostic bindings, chunk framing and
blob hashes, all 175 bucket rows, an untouched post-witness traversal, the
complete residual count in the closed boundary bucket, the binary-operator
registry, and the rank-50,001 witness derived without using either endpoint's
reported witness. Removed operator `add` is absent from the archive; retained
operator `difference` remains ordered and active.

This result agrees with the earlier quarantined Python-only probe, but the
probe was not used as an expected-value oracle and carried no routing
authority. Routing authority comes from this dual result plus the host replay
and Evidence Commit N.

## Isolation and reproducibility

- Both endpoints ran concurrently in separate Docker containers as UID/GID
  `65534:65534`, with `--network none`, `--pull never`, read-only roots and
  commit-bound source snapshots, dropped capabilities, no-new-privileges, and
  purpose-specific seccomp profiles.
- Python used the already cached image digest
  `e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3`.
- Rust used the already cached image digest
  `38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89`
  and `cargo --release --locked --offline`.
- The Rust dependency seed contained only reverified `cache` and `index`
  subtrees. All 21 locked crates were checksum-verified, copied to a fresh
  snapshot, and unpacked again in a fresh tmpfs Cargo home. No pre-unpacked
  `src` tree was mounted.
- The run-specific target volume `hegel-shrink3-enum-d17b03e14f3f` was fresh
  before the run and removed afterward. No container or volume bearing this
  run's Commit-M identity remained after the supervisor exited. Two older,
  unlabeled shrink-3 cache volumes were outside this run and were not removed.
- The 44,952,687-byte external evidence set contains 13 files and is bound by
  root `db8e2b3b3c1879ee3d4f69dcbfae9b66e5f74cbe9c3ad897d0857e7850cb92ed`.
  The 22 MB streams remain outside Git; the committed evidence record contains
  their complete path/size/hash manifest and replays them when present.

Within the supervisor qualification scope, no image pull, dependency download,
or container network access was performed.

## Qualification note

The Source Commit M focused suites passed 39 Python supervisor/evidence tests,
50 shrink-3 strict tests, 25 inherited base/shrink-2 tests, and 12 Rust tests
under a fresh tmpfs/cache-index-only offline Cargo environment. The new
evidence replay contributes four focused tests.

A repository-wide serial run was interrupted after encountering the known
historical M2.5 currentness assertion: that legacy test compares an artifact
correctly bound to its original implementation basis against a later change to
`src/hegel_machine/__init__.py`. Source Commit M did not introduce that
failure. Therefore this record does not claim a green completed repo-wide
suite.

## Claim boundary and next route

This result does not create child formal roots or execute formal M3. It does
not establish full closure cardinality beyond the bounded rank-50,001 result,
evaluate odd/sink roles, issue an
`OUTSIDE_FROZEN_CLOSURE(dsl_version, universe_root, target_root,
exact_extensional)` or MDL certificate, sign a formal object, or authorize
ACTIVE governance.

The state remains:

```text
execution_state = NOT_RUN
formal_roots_generated = false
formal_roots = null
formal_state_transition_allowed = false
```

The only routing consequence is engineering admission to the already
preregistered shrink-order step 4: reduce `max_top_level_clauses` from `3` to
`2`. This transition must receive its own freeze, dual strict qualification,
and complete-enumeration evidence before any formal status can change.
