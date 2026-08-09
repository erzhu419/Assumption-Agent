# Hegel Machine Phase-3A M3 terminal result and shrink-step-2 transition

Status: local formal-runtime terminal evidence published; no external-authority certificate, role evaluation, Phase-3 exit, ACTIVE promotion, or remote publication is claimed.

## 1. Bound run identity

The sole admitted canonical run was executed from the local two-commit runtime topology:

```text
Commit B  78d5c77994ad9088c082c32a948b5a2b40407966
Commit C  7636aba6e07f565f673e2f3cdf39a1c5dc143d9e
Commit D  1a434ba1236ae6481ba1cb93f85b1d8886d37243
run ID    e4af9f57c38fb298462ec628c4ed8a03
attempt   attempt-1
```

The persisted start record replayed all 24 entry gates and made the only authorized transition:

```text
NOT_RUN/NONE
  -> RUNNING/CANONICAL_ENUMERATION
```

The execution used target-independent snapshots only. Python and Rust ran concurrently in digest-pinned Docker images with `--pull=never`, `--network=none`, `--restart=no`, a read-only root filesystem, all capabilities dropped, and the frozen seccomp profile. The stopped formal containers remain available as local evidence.

## 2. Terminal result

Both implementations independently produced:

```text
terminal state                 DSL_TOO_LARGE/NONE
closure status                 DSL_TOO_LARGE
canonical programs archived    50,000
raw operator applications      3,292,439
first out-of-budget ordinal    50,001
first witness AST SHA-256      96200a6a131204315ffcd1efd0aa2dcfe2ce665a2c06516461772c9812f0ec71
frontier exhausted             false
closure cardinality            null
role evaluation started        false
```

The state transition reason is `CANONICAL_PROGRAM_50001_ACCEPTED`. It is not an execution timeout, raw-budget abort, semantic disagreement, or container failure.

The two implementations agreed byte-for-byte on all three core archives:

| Archive | Bytes | SHA-256 |
|---|---:|---|
| canonical program records | 10,913,073 | `8ffdcf1e64d1d1934404c6fc98f8340662bd9194852a2b558d84e7a533d32e21` |
| program chunk manifests | 1,610 | `571b7eddeba06e555b35de9be8ed808b7369da2ff6b898ff25bbb6fbbc56885c` |
| bucket accounting records | 9,872 | `03e039144d1d3e88054e85b83141c4764916cdafe4583d4e4f9806f0c03deb62` |

Each archive contains 50,000 program records, 13 chunks, and 175 bucket rows. Both stderr streams are empty. Both containers exited zero without OOM or Docker error.

## 3. Published terminal carrier

The exact canonical start state, start publication receipt, and terminal outcome are archived at:

```text
artifacts/phase3_m3_runtime/formal_m3_start_state_v1.json
artifacts/phase3_m3_runtime/formal_m3_start_publication_receipt_v1.json
artifacts/phase3_m3_runtime/formal_m3_terminal_outcome_v1.json
```

Their exact file identities are:

```text
start-state bytes / SHA-256    1,525 / 9f07564d4f859e082288ddf971c336a03b490062c65bce7eb81ddcfa64ea4053
start-receipt bytes / SHA-256  26,879 / dede9fb1bf1febe4ec6646f00be456c94ff181fa91e23a23d8392c7596a70df3
terminal bytes / SHA-256       62,942 / 4f631224383297f6f30d70dbcefc15ed1c1296ba634a604e5a59562d11e67aed
```

The terminal carrier's internal identities are deliberately distinguished:

```text
embedded outcome self-hash  973214b278e0bd3af474fa0b095e518e1ea8323917845b856e8b5de72913c67c
dual replay agreement root  48454aef57c3b560ee2f05e46b1dae1f4cd3e9fd0de52b392083a7c8b2359d83
terminal state record root  1f54925f8187c955ae4e7b2c9bb83144ce6627468153d8e1696b27c4705d23dd
```

The file hash covers the canonical JSON including its embedded self-hash. The embedded hash covers the canonical outcome body before insertion of that field. They must not be substituted for one another.

The large duplicate archive blobs remain under the private canonical run root and are not duplicated into Git. Their paths, sizes, hashes, container evidence, and formal roots are bound by the terminal carrier. Exact-once replay revalidated the existing terminal and returned `ALREADY_TERMINAL_VERIFIED` without creating another container or changing state.

## 4. Claim boundary

This result establishes only that the frozen bounded canonical-enumeration profile reached its 50,001st syntactically canonical program under the 50,000-program limit. The embedded enumerator claim remains:

```text
FORMAL_PROFILE_CANDIDATE_NOT_AUTHORITY
authoritative_claim_allowed = false
```

The run did not read raw split seed material, split assignment rows, target-role inputs, private keys, or hidden target outputs. Therefore it does not establish any of the following:

- `COMPLETE` closure or a closure cardinality;
- an odd-target match or outside-language verdict;
- hidden-sink recognition or conservation discovery;
- an MDL invention certificate;
- Phase-3 exit, ACTIVE governance, or a final M4 signature.

## 5. Mandatory successor

The terminal old run has no in-version successor. In particular, `ROLE_EVALUATION` and M4 are closed. The only preregistered successor is a cross-version shrink transition:

```text
SHRINK_STEP_2_REDUCE_RATIONAL_PARAMETER_TO_NEG1_ZERO_POS1
```

The child language must preserve the original three-bit `RationalParameterId/v1` allocation:

```text
active IDs      1 (-1), 3 (0), 5 (1)
tombstone IDs   0 (-2), 2 (-1/2), 4 (1/2), 6 (2)
reserved ID     7
```

Surviving canonical AST bytes, AST hashes, and MDL codewords remain unchanged. Removed values must fail in source and formal admission with `REJECT_REMOVED_RATIONAL_PARAMETER`; IDs are not compacted or reused. The new DSL begins from `NOT_RUN/NONE` and requires new DSL/operator/registry bindings, formal roots, implementation qualification, execution manifest, and run identity before another formal M3 attempt.
