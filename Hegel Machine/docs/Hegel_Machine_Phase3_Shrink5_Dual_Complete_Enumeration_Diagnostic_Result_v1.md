# Hegel Machine Phase-3 shrink-5 dual complete-enumeration diagnostic result v1

Status: **DUAL DSL_TOO_LARGE — HOST REPLAY PASS — DIAGNOSTIC ONLY**

Source U: `a3c384b4cb0f95583af6a1eb1c1d256ef6e9128a`

Evidence record:
`artifacts/phase3_shrink5_dual_complete_enumeration_diagnostic_v1.json`

## Result

The isolated Python and Rust endpoints independently produced byte-identical
program, chunk-manifest, and bucket-accounting streams for
`hegel-old-dsl-v1.5.0`. A separate target-free host process reconstructed the
frozen typed traversal from an empty state without treating either endpoint's
reported boundary values or witness as authoritative. The host is independent
of endpoint-reported witness selection, but it reuses the committed Python
generator and is not a third independent enumerator.

| field | observed value |
|---|---|
| diagnostic terminal | `DSL_TOO_LARGE` |
| archived canonical programs | 50,000 |
| raw operator applications through the fully closed boundary bucket | 3,120,719 |
| rank-50,001 witness hash | `31320fc9f8926792aaf1416a4963df46a2300d87db8096f42e574a62272a68ee` |
| rank-50,001 witness CBOR | `820183010384020183000001860003050200818203f5` |
| residual canonical programs in the closed boundary bucket | 2,237 |
| canonical program archive root | `9fc165f1e0a7fcf239422045daefabc5597a32c5aa87f486585b051ef701a629` |
| program chunk manifest root | `b02ff6d766ce010880bbccb1fb89845afedc7c7171d21cd28931d560e893c172` |
| bucket accounting root | `9e07bc1f63a47e9b77de495b242f431daa6c11cf0a9d128d6aa62b066a8629e1` |
| exact endpoint report fields | Python 78; Rust 75 |

The host verified the exact archive prefix, strict AST/hash/Q32-MDL metadata,
program indices, all three synthetic diagnostic bindings, chunk framing and
blob hashes, all 150 bucket rows, an untouched post-witness traversal, the
complete residual count in the closed boundary bucket, the sparse binary
operator registry, and the rank-50,001 witness. The shrink-5 generator created
only normalized AND2 candidates: `maximum_top_level_clauses=2`,
`and3_generator_attempts_allowed=false`, and
`and3_raw_operator_application_count=0`.

This establishes a bounded `DSL_TOO_LARGE` diagnostic boundary, not complete
closure cardinality. The 2,237 residual programs belong only to the fully
closed boundary bucket; later traversal buckets were not visited.

## Isolation and reproducibility

- Both endpoints ran concurrently in separate digest-pinned Docker containers
  as UID/GID `65534:65534`, with `--network=none`, `--pull=never`, read-only
  roots and Source-U snapshots, dropped capabilities, no-new-privileges, and
  purpose-specific seccomp profiles.
- Python used image digest
  `e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3`.
- Rust used image digest
  `38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89`
  and `cargo --release --locked --offline`.
- The Rust dependency seed contained only the exact `cache` and `index`
  subtrees. All 21 locked crates were checksum-verified; every mounted regular
  file was frozen to mode `0444`; the pre/post-build manifest root remained
  `55e28914deab8d03b769f9a749bd1f73e540345fdf62a841622043c0a1f575d0`.
  Cargo unpacked the crates into a fresh tmpfs home, and no pre-unpacked `src`
  tree was mounted.
- The run-specific target volume `hegel-shrink5-enum-a3c384b4cb0f` was fresh
  before the run and removed afterward.
- The 45,366,664-byte external evidence set contains exactly 13 files and is
  bound by root
  `d24989f7be76cc2369f104f3b817ffcad4baffd60e90b203ebe1382d41e825cb`.
  The large streams and tar archives remain outside Git. The compact repository
  artifact records their complete path/size/SHA-256 manifest and replays them
  when the external directory is present.

Within the supervisor qualification scope, no image pull, dependency download,
or container network access was performed. These are technically isolated
roles under one administrative controller, not organizationally independent
human custodians or signers.

## Evidence replay

The four focused Evidence V tests independently check:

1. the self-bound evidence record ID, exact Source U/Evidence T ancestry, strict
   qualification binding, canonical/type-strict JSON, and all non-formal
   authority guards;
2. every external file hash, the external artifact-set root, all 81 Git source
   rows, the Source-U source-set root, and the safe commit-bound source tar;
3. the supervisor summary hash, host/report hashes, exact 78/75 report schemas,
   common-field equality, Docker/Cargo offline controls, and frozen dependency
   manifest; and
4. both byte-identical framed stream sets, exact frame counts, all three RFC6962
   roots, strict witness decode/hash identity, deterministic output tar, and the
   sole permitted next route.

No repository-wide all-green claim is made by this evidence record.

## Claim boundary and next route

This result does not create child formal roots or execute formal M3. It does
not establish full closure cardinality, evaluate odd/sink roles, issue
`OUTSIDE_FROZEN_CLOSURE(dsl_version, universe_root, target_root,
exact_extensional)` or an MDL certificate, generate or access a split seed,
sign a formal object, authorize M4, or change ACTIVE governance.

The state remains:

```text
execution_state = NOT_RUN
formal_roots_generated = false
formal_roots = null
formal_state_transition_allowed = false
```

The only routing consequence is engineering admission to the already frozen
shrink-order step 6: reduce `max_total_ast_depth` from `4` to `3`, while
keeping `maximum_ast_node_count=6` and `maximum_top_level_clauses=2`.
Step 6 must receive its own freeze, dual strict qualification, and
complete-enumeration evidence before any formal status can change.
