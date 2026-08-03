# Phase-3A Gate 17 Parent Audit — Touched-Path Engineering Amendment v1

Status: deterministic engineering freeze for Gate 17 generation and replay.
This amendment changes no formal schema, numeric tag, hash domain, legacy
source ID, audited parent, or signature rule.  It resolves only the previously
underspecified meaning of a commit's “touched path” set.

## 1. Frozen inputs

- audited parent commit: `fb3a3ee4865a140c558821017ddd3e9a6a99de48`;
- parent DSL: `hegel-old-dsl-v1.0.0`;
- parent freeze: `hegel-freeze-p2b-p3-v1.0.2`;
- Git object format: SHA-1 only;
- exact legacy source rows: the odd target and sink-control IDs already frozen
  by E7;
- absence reason mask: `0b1111`.

The real parent freeze is v1.0.2.  The v1.0.0 value used by an earlier
synthetic fixture is not an admissible real-parent value.

## 2. Strong touched-path rule

The rule ID is `git-any-parent-result-or-deletion-blob-v1`.

1. Traverse every commit reachable from the audited parent, without a path
   filter.  Read the `parent` headers in their commit-object order and preserve
   that order in `AuditedHistoryRowV1.ordered_parent_commit_ids`.
2. A root commit contributes every recursive blob leaf in its tree.
3. For a non-root commit, compare its tree independently with every parent
   tree, with rename detection disabled.  Take the union of raw paths whose
   leaf tree entry differs against at least one parent.
4. If such a path has a blob in the resulting commit, bind that resulting
   blob and mode exactly once.  Do not also bind superseded parent blobs.
5. If the path is absent in the resulting commit, bind every distinct deleted
   parent blob record.  The record identity includes path, blob digest, mode
   and byte length, so mode-distinct parent entries remain distinct.
6. Gitlinks or another non-blob leaf fail closed.  Repository paths must be
   valid, relative UTF-8 byte strings; no newline or quoting normalization is
   performed.
7. Per-commit rows and the top-level union use the complete frozen formal
   ordering tuple
   `(raw_repository_path_utf8_bytes,
   repository_path_alias_id_digest, git_blob_digest)`.  Equal formal keys keep
   the deterministic encounter order induced by history order, commit parent
   order, and raw Git diff order.
8. The top-level tree is the canonical-CBOR deduplicated union of every
   per-commit row.  Empty commits bind the RFC6962 empty root.

`repository_path_alias_id_digest` uses rule
`repo-path-sha256-raw-bytes-v1`:

```text
IdDigestV1("repo-path-sha256:" || lowercase_hex(SHA256(raw_path_bytes)))
```

`commit_generation` remains zero for roots and
`1 + max(parent_generation)` otherwise.  History rows sort by generation and
then raw commit digest.

## 3. Git replay boundary

The implementation uses only Git plumbing with NUL-delimited raw paths.  It
does not checkout or mutate a worktree.  Replay rejects shallow repositories,
non-SHA-1 repositories, missing/lazily fetched objects, replacement objects,
malformed commit headers, malformed raw diffs, and non-blob audited entries.

The only reachable merge is
`cb6bb75d60376115709bd6fb45121cfb8087cf92`.  Its parent order is preserved as
`76aad4bdfd0bf801d49232a4c583bb50adc2705e`, then
`8de51079362bbf2f7ab19383d6ea84b744818a4b`.

## 4. Path-name and unique-blob content diagnostics

Path-name checks are an additional diagnostic receipt, not a new formal
object and not a substitute for the formal history/path replay.  ASCII letters
are lowercased and `-` / `.` are normalized to `_`; matching is byte-substring
matching.  The three frozen predicate groups are:

- `typed_or_parent_binding_manifest`: `typed_binding_manifest`,
  `parent_binding_manifest`;
- `split_seed_commitment_or_allocation`: `split_seed_commitment`,
  `split_seed_allocation`, `split_assignment_manifest`,
  `split_allocation_manifest`;
- `hidden_access_ledger`: `hidden_access_ledger`.

For each group, the receipt binds the matching unique-path count, matching
formal-row count, and the RFC6962 root of matching `AuditedPathBlobRecordV1`
rows.  An empty match therefore binds the standard empty root.  These
predicates establish path-name absence only; they do not claim that arbitrary
blob contents were semantically searched.

Therefore path-name checks are paired with the stronger content profile
`legacy-parent-formal-artifact-unique-blob-content-absence-v1`.  The generator
streams every unique Git blob referenced by the top-level 7,945-row union,
including blobs under generic names such as `artifact.json` and binary blobs.
For every blob it independently recomputes
`SHA1("blob " || decimal_size || NUL || payload)` and verifies the reported
byte length.  Missing, truncated, wrong-type, size-mismatched or hash-mismatched
objects fail closed.  Because matching is on raw bytes, no structured blob is
discarded because of text decoding; the unscannable relevant-structured count
must be zero.

The negative content predicates use exact machine/schema/key byte signatures:

- typed/parent binding: `hegel-typed-binding-manifest/`,
  `typed_binding_manifest_root`, `parent_binding_manifest_root`;
- split commitment/allocation: `hegel-split-seed-commitment-manifest/`,
  `split_seed_commitment_digest`, `split_seed_commitment_manifest_root`,
  `hegel-split-assignment-row/`, `split_assignment_tree_root`,
  `split_allocation_manifest_root`;
- hidden-access ledger: `hegel-hidden-access-ledger-record/`,
  `hidden_access_ledger_genesis_root`, `hidden_access_ledger_head_root`.

Natural-language phrases are not used as content signatures: prose discussing
a future manifest is not a realized machine-readable manifest.  Conversely,
any exact-signature hit fails the absence audit for external review.  The two
frozen legacy source IDs are positive content predicates and each must occur.
The receipt publishes only counts and SHA-256 digests of sorted matching Git
blob-ID sets, never blob bodies or matching excerpts.

## 5. Replayed frozen result

The full replay completed over one root, 1,399 reachable commits and one merge:

| Item | Value |
|---|---:|
| deduplicated audited path/blob rows | 7,945 |
| history rows | 1,399 |
| legacy source rows | 2 |
| audited path tree root | `55c4670498efcfb80055f6a0ada0c3b44da2f24c82a1268701a38834a649cc3f` |
| audited history tree root | `c8b59bf44f5020656f34932c3e0394959d26e0438bf75f0040ac93f449077854` |
| legacy source tree root | `982a60f88ceee5a08f3f0ab4cb44002308ce4b288de334407e02fdc210bbf3c7` |
| `ParentAbsenceAuditBundleV1` root | `136c9eee4c616d9f55dae699cb467e56921ce4706943ae87a5ad89bf9d82ff51` |

All three path-name predicate groups have zero matching paths and rows; each
matching-row tree root is
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
The content pass inspected all 7,792 unique blobs (1,520,624,571 bytes), of
which 7,298 were referenced through a structured-file suffix.  Its inventory
digest is
`b30255e89dc306a07ae72d25c1b67c86fe6552a8d69268e6d4d8a3008ebcdd09`.
All negative content signatures have zero matches.  Each frozen legacy source
ID occurs five times across three unique blobs / three formal path-blob rows;
the matching blob-ID-set digest is
`e758773a4cb8ae93719a7f69f91c881225932df66c7764b916dbdf965f74415f`.

The generator exposes the five static
`ParentManifestAbsenceAttestationV2` fields.  A purpose-4 actor must still add
its key ID and authoritative timestamp and sign through the already-frozen
signature envelope.  This amendment and its receipt generate no key or
signature and make no authority claim.
