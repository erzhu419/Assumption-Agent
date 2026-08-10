# Hegel Machine Phase-3 shrink-6 dual complete-enumeration diagnostic result v1

Status: **DUAL DSL_TOO_LARGE — HOST REPLAY PASS — DIAGNOSTIC ONLY**

Source Y: `5217568303d5c7f902682c092750f637c64f080a`

Evidence record:
`artifacts/phase3_shrink6_dual_complete_enumeration_diagnostic_v1.json`

## Result

The isolated Python and Rust endpoints independently produced byte-identical
program, chunk-manifest, and bucket-accounting streams for
`hegel-old-dsl-v1.6.0`. A separate target-free host process reconstructed the
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
| witness bucket | 63 (`output_sort_id=3`, depth 2, 4 nodes) |
| residual canonical programs in the closed boundary bucket | 2,237 |
| canonical program archive root | `cbe332032acf527d80ec744d311fb45c99d6b06b31636667a1b2a21b2e9e2ceb` |
| program chunk manifest root | `f3c74184111d5ed56347f31ef4e903b5f38c9d88c4973e82f0aa34480b59edb4` |
| bucket accounting root | `6da3bfeef3e423456507d9ae3258df96650aa4890fcfc1422a6a5425bae424aa` |
| preregistered shrink-5 prefix preservation | verified |
| exact endpoint report fields | Python 78; Rust 75 |

The host verified the exact archive prefix, strict AST/hash/Q32-MDL metadata,
program indices, all three synthetic diagnostic bindings, chunk framing and
blob hashes, all 120 bucket rows, an untouched post-witness traversal, the
complete residual count in the closed boundary bucket, the sparse binary
operator registry, the rank-50,001 witness, and the preregistered shrink-5
prefix-preservation expectation. The shrink-6 generator created only
normalized AND2 candidates: `maximum_top_level_clauses=2`,
`and3_generator_attempts_allowed=false`, and
`and3_raw_operator_application_count=0`.

This establishes a bounded `DSL_TOO_LARGE` diagnostic boundary, not complete
closure cardinality. The 2,237 residual programs belong only to the fully
closed boundary bucket; later traversal buckets were not visited. Reducing the
maximum AST depth from 4 to 3 did not move the observed boundary: the counts
and rank-50,001 depth-2 witness remained equal to shrink-5 under the frozen
preservation check. That is a negative capacity result for this shrink step,
not evidence that the complete closures of the two DSL versions are equal.

## Isolation and reproducibility

- Both endpoints ran concurrently in separate digest-pinned Docker containers
  as UID/GID `65534:65534`, with `--network=none`, `--pull=never`, read-only
  roots and Source-Y snapshots, dropped capabilities, no-new-privileges, and
  purpose-specific seccomp profiles.
- Python used image digest
  `e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3`.
- Rust used image digest
  `38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89`
  and `cargo --release --locked --offline`.
- The Rust dependency seed contained only the exact `cache` and `index`
  subtrees. All 21 locked crates were checksum-verified; every mounted regular
  file was frozen to mode `0444`; the pre/post-build manifest root remained
  `2af526c9b0ece44c0fb67499e2eb33815299bb3cd81e49ec4a5e2e7ba823bc93`.
  Cargo unpacked the crates into a fresh tmpfs home, and no pre-unpacked `src`
  tree was mounted.
- The run-specific target volume `hegel-shrink6-enum-5217568303d5` was fresh
  before the run and removed afterward.
- The 45,554,519-byte external evidence set contains exactly 13 files and is
  bound by root
  `16685c291595b0a9cc653545d6156b47419f41885c6cbf1562f35b49b58c65f9`.
  The large streams and tar archives remain outside Git. The compact repository
  artifact records their complete path/size/SHA-256 manifest and replays them
  when the external directory is present.

Within the supervisor qualification scope, no image pull, dependency download,
or container network access was performed. These are technically isolated
roles under one administrative controller, not organizationally independent
human custodians or signers.

## Evidence replay

The four focused Evidence Z tests independently check:

1. the self-bound evidence record ID, exact Source Y/Evidence X ancestry, strict
   qualification binding, canonical/type-strict JSON, and all non-formal
   authority guards;
2. every external file hash, the external artifact-set root, all 91 Git source
   rows, the Source-Y source-set root, and the safe commit-bound source tar;
3. the supervisor summary hash, host/report hashes, exact 78/75 report schemas,
   common-field equality, Docker/Cargo offline controls, and frozen dependency
   manifest; and
4. both byte-identical framed stream sets, exact frame counts, all three RFC6962
   roots, strict witness decode/hash identity, deterministic output tar,
   shrink-5 prefix preservation, and the fail-closed exhausted-shrink route.

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

All six preregistered shrink steps are now consumed. The fail-closed route is:

```text
HALT_NO_PREREGISTERED_SHRINK_REMAINING_NEEDS_NEW_NORMATIVE_DECISION
```

This evidence authorizes no budget increase, shrink-7, or new DSL version. A
new normative decision must specify the next research route and its fresh
preregistration before engineering can resume; the diagnostic itself cannot
promote formal status.
