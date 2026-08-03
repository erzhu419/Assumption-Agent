# Phase-3A M2.5 Commit-B staged publication audit v1

This control audits the bytes that Git would actually place in publication
Commit B. It does not audit a caller-provided directory and does not treat the
working tree as publication authority.

The frozen path policy discovers bytes dynamically from the index, but path
roles themselves are exact. Prepare requires each of the following exactly
once (and accepts no aliases or additional files):

- `phase3_m25_actor_qualification_v1.json`
- `phase3_m25_errata_qualification_v1.json`
- `phase3_m3_implementation_qualification_v1.json`
- `phase3_m25_bridge_dag_rust_binary_qualification_v1.json`
- `phase3_m25_live_actor_protocol_qualification_v1.json`
- `phase3_m25_pre_genesis_execution_status_v1.json`
- `phase3_m25_pre_genesis_readiness_v1.json`
- `phase3_m25_formal_gate_evidence_v1.json`
- `phase3_m25_gate_promotion_v1.json`
- `phase3_m25_gate_promotion_v1.json.publication-receipt.json`
- `Hegel Machine/docs/phase3_m25_external_status.md`

The JSON files above live under
`Hegel Machine/artifacts/phase3_m25_external/`. Finalize requires the same set
plus exactly one
`phase3_m25_commit_b_publication_audit_receipt_v1.json`. The manifest binds
each exact path to its unique role ID and cardinality one. An unknown JSON,
even under the public prefix, fails closed.

Every staged A-to-B path is discovered from the index. Add/modify mode-`100644`
blobs are accepted; executable modes, symlinks, deletes, renames, unmerged
entries, non-allowlisted paths and unstaged drift of an audited path fail
closed. The manifest binds basis commit, repository-relative path, Git mode,
index blob ID, byte length and SHA-256. All candidate bytes are copied to a
private Linux-local `/tmp` snapshot; the repository is never mounted into the
actor.

The purpose-4 actor uses the pinned local Python image with `--pull=never`,
`--network=none`, read-only root/mounts, all capabilities dropped,
`no-new-privileges`, the frozen seccomp profile and UID/GID 65534. It
independently checks the manifest/file set, strict duplicate-free JSON,
forbidden secret field names, record-start private-key headers and raw
host/author path forms. Existing diagnostic CLIs may retain their stable
pretty-JSON framing; compact canonical bytes are mandatory for the formal
evidence, formal promotion, formal transaction receipt and publication-audit
receipt. The host independently rebuilds the index manifest and
replays the formal public gate evidence/promotion/transaction receipt.

The renderer plus two audit phases avoid self-reference:

1. `render-status` reads exactly the other ten public worktree roles through
   no-symlink dirfd walks and creates the unique frozen status path with
   `O_EXCL`. This is only deterministic generation; `prepare` remains the Git
   index authority.
2. `prepare` requires the unique audit-receipt path to be absent from the
   index, audits all other staged public files, and writes the canonical public
   receipt at that unique path without staging it.
3. After the operator stages that receipt, `finalize-index` first launches a
   fresh receipt-excluded purpose-4 replay and requires its canonical bytes to
   equal the staged prepare receipt. It then launches another fresh actor over
   the complete index including the receipt and performs another host replay.
   The second actor's receipt-excluded manifest projection must equal the first
   fresh manifest exactly, and its receipt row must still bind the already-read
   staged receipt bytes; an index change between the two actors therefore fails
   closed.
   Its final receipt is stdout or an explicitly repo-external file; it is never
   added to Commit B.
4. After Commit B exists, `verify-commit` parses the raw commit object (ignoring
   local graft metadata), requires Commit A as its encoded sole parent,
   replays the committed tree blobs, and requires the mode-`0600` repo-external
   `finalize-index` receipt. It proves that receipt's embedded fresh-actor
   manifest is byte-for-byte the committed B tree inventory. It performs no
   Git mutation; omitting the finalize receipt is not an accepted path.

Purpose-4 image selection is read from the exact actor-profile blob in the
caller's Commit-A tree. Dirty worktree profile bytes cannot select the image.
Both committed purpose-4 actor receipts must bind that same Commit-A profile
image reference; a self-consistent receipt naming another local image is
rejected.
The live `prepare` and `finalize-index` replays retain their strong local source
and persisted-binary checks. In contrast, post-commit `verify-commit` uses a
deliberate public/commit-only replay: it validates report schemas, self-hashes
and cross-report bindings, recomputes actor inputs, source sets,
toolchain-policy, seccomp and fixture bindings from the supplied repository's
Commit-A blobs, and never reads current worktree bytes or ignored persisted
binaries as evidence.

The commit-only Gate 15--24 path first validates the archived actor and errata
reports exclusively against those Commit-A blobs. It then injects the two
exact, already-validated report objects into the unchanged remainder of the
frozen gate evaluator before replaying promotion and the transaction receipt.
It therefore neither invokes live Docker/Rust qualification nor silently
substitutes the verifier's current checkout, while retaining the same gate DAG
and eligibility rules after the report-validation boundary.

Secret lint treats a private-key header at a logical record start, any complete
embedded PEM/OpenPGP private-key block, and the frozen non-PEM key magics at any
offset as findings. The archived genesis-secret-absence receipt is not trusted
as a zero-findings assertion: `verify-commit` regenerates it exactly from the
supplied repository by raw commit-parent traversal over the entire reachable
history. Shallow, promisor, alternate-object, graft and replace-ref repositories
fail closed, and the regenerated canonical receipt must byte-match the archived
one.

Example (from the repository root):

```bash
python3 'Hegel Machine/tools/phase3_m25_commit_b_publication_audit_v1.py' \
  render-status --basis-commit "$COMMIT_A"
git add -- \
  'Hegel Machine/artifacts/phase3_m25_external/' \
  'Hegel Machine/docs/phase3_m25_external_status.md'
python3 'Hegel Machine/tools/phase3_m25_commit_b_publication_audit_v1.py' \
  prepare --basis-commit "$COMMIT_A"
git add -- 'Hegel Machine/artifacts/phase3_m25_external/phase3_m25_commit_b_publication_audit_receipt_v1.json'
python3 'Hegel Machine/tools/phase3_m25_commit_b_publication_audit_v1.py' \
  finalize-index --basis-commit "$COMMIT_A" \
  --output /durable/repo-external/commit-b-final-audit.json
```

The disclosure remains exact: this is an
`OWNER_CONTROLLED_SELF_CONSISTENT_TRANSCRIPT_NOT_REMOTE_ATTESTATION` under the
same administrator, with no organizational or human independence, technical
role isolation accepted by the owner, no remote attestation and no hardware
non-exportability. It does not claim that a third party could not forge a
self-consistent transcript. Neither audit phase changes a
formal gate, creates a key/seed/signature/root/marker, or starts M3.
