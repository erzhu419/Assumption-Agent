# Hegel Machine Phase-2B Formal Wire/Covert Step-5 Authority v1

**Document ID:** `hegel-phase2b-formal-wire-covert-step5-authority-v1`

**Status:** `V0_FROZEN_CONSTRUCTION_CONTRACT_HOLD_NOT_IMPLEMENTED_NOT_EXECUTED`

**Claim level:** `PROSPECTIVE_R01_R05_AUTHORITY_DESIGN_ONLY_NOT_GATE_EVIDENCE`

## 1. Decision and current HOLD

This document freezes the construction boundary for a new Step-5 authority
attempt. It does not report that the authority implementation exists, that a
ceremony has run, or that any readiness node has passed. No Step-5 readiness
evidence overlay artifact currently exists. Authority v1 can parse and
cryptographically check a Stage-B bootstrap graph of candidate profile,
candidate registry, candidate R01, and candidate R02 objects; authority v2 can
parse a separate closed lock/profile/registry graph with five typed static
service bindings. Each derives only a non-evidence snapshot in which all
sixteen readiness nodes remain false. The readiness state therefore remains
`HOLD`; no candidate is a passed receipt.

This revision is a candidate document BYTE-STOP pending independent reviewer
GO. Its hashes and focused test counts identify construction bytes only; they
do not make the cross-module chain final and do not turn authority, R01,
readiness, live-service, runtime-execution or execution-authorization
predicates true.

Seventeen construction foundations now exist: authority/profile v1 and v2,
retention/result projection, retention-only CAS v1, generic planned CAS v2,
local logical time, a governance-signed owner-status ledger candidate, offline
static image/runtime-closure replay, and verifier-only logical-time, generic-
store, and owner-status replay cores, plus a producer-free R01 pure-replay
composer, a verifier-only R01 evidence contract, connected governance
current-cut mechanics, an initial external-witness local spool, and an
external-witness admission-transcript candidate, and a post-`b480` presented
live-runtime overlay replay candidate. Both authority foundations remain
candidate-only and the other fifteen remain mechanics-only. The
composer cross-binds presented raw mechanics, the evidence
contract replays a signed but explicitly non-admitted wire DAG, the current-cut
module observes only a connected optimistic snapshot, and the external-witness
module commits only a local one-leaf spool. The admission-transcript module
replays only a presented bytes-only Q1/A/Q2 DAG and supplied logical-time
chain. The post-`b480` overlay replays only a one-way presented DAG through a
six-object CAS session and presented A/2 transcript. None composes a deployed
Step-5 authority graph or actor, and none changes any of the sixteen all-false
readiness rows. The owner-status schemas, replay core and six R01 candidate
surfaces are frozen only for their current nonauthority mechanics scope.
Consequently R01 and R02 remain candidates, real R01/R02 do not exist, and
R03--R05 remain locked.

The retained Task-4 attempts are calibration and mechanics provenance only:

- r12 is the frozen P0 retained-pair member. Its terminal SHA-256 is
  `a6acbc45f820aa3a651fe1ddc66b6abebd07d3735a20871a3bf35ef6ad7dd249`.
- r13 is the fresh P1 live-mechanics member, campaign
  `qv2-p1-20260822-9938b66f-r13`, attempt
  `live-qualification-013-9938b66f-r13`. Its terminal SHA-256 is
  `4889a474e5b23ad547f2430d82b675124a8087d823352643caa4479bbc9461e0`
  and its disposition is `MECHANICS_VERIFIED_NOT_AUTHORITY`. Its diagnostic
  directions were `KNOWN_NULL` pass with zero material witnesses and
  `KNOWN_BINDING_LEAK` expected failure with 169 material witnesses; those
  observations remain non-authoritative.

Neither member retained the complete Step-5 evidence surface. In particular,
Task 4 did not retain all original worker bytes, the complete result matrix,
an immutable runtime image, or the full import/native closure under a
pretrusted signed authority. The pair may be bound as diagnostic provenance
for a new run, but no signature, CAS copy, profile rewrite, or later blessing
may promote r12 or r13 into R03, R04, or R05 evidence. Step 5 must use fresh
Step-5 attempt identity and rerun the qualification.

The authoritative readiness DAG remains immutable. Step 5 may publish a
separate evidence overlay; it must not edit the DAG or infer readiness from
node order.

## 2. Frozen timeout decision

The statistics-worker timeout remains exactly 3,600 seconds. The r13 live
measurements were:

| Retained measurement | Nanoseconds | Minutes | Frozen timeout |
|---|---:|---:|---:|
| statistics plus parent post-processing, measurement 1 | 1,907,379,765,357 | 31.79 | 3,600 s |
| statistics plus parent post-processing, measurement 2 | 2,348,028,873,762 | 39.13 | 3,600 s |
| invariance, measurement 1 | 2,982,997,900,472 | 49.72 | 7,200 s |
| invariance, measurement 2 | 2,946,987,187,902 | 49.12 | 7,200 s |

The slower statistics measurement used approximately 65.2% of the existing
3,600-second allowance, including the measured parent post-processing scope.
The invariance timeout was already 7,200 seconds. These observations support
retaining the existing split; they are not a performance or hardware-portable
claim.

A future 7,200- or 10,800-second statistics allowance is technically possible,
but only as a new coordinated contract version. It would require a new profile
ID, a new attempt ID, updated total-deadline and reserve arithmetic, new signed
bindings, and fresh qualification. It must never be applied retroactively to
r12, r13, or an in-flight Step-5 attempt. This v0 contract therefore freezes:

```text
statistics_worker_timeout_seconds = 3600
invariance_worker_timeout_seconds = 7200
timeout_relaxation_in_place = false
```

## 3. Disclosed threat model

This authority uses owner-accepted technical separation under one
administrative controller. It does not reinterpret container or process
separation as separate people or organizations.

The exact v0 authority-profile `threat_model` mapping is:

```yaml
same_admin_controller: true
owner_accepted_technical_actor_model: true
producer_may_be_malicious: true
producer_outside_authority_roles: true
producer_local_filesystem_claims_authoritative: false
producer_local_filesystem_is_non_authoritative: true
purpose_separated_keys_required: true
distinct_control_domains_required: true
organizational_independence: false
independent_human_actors: false
external_authority: false
remote_attestation: false
hardware_key_nonexportability: false
malicious_admin_resistance: false
malicious_host_root_resistance: false
malicious_container_runtime_resistance: false
```

Authority/profile v2 retains those values and extends its exact threat-model
mapping with:

```yaml
static_distinct_service_uids_gids_requested: true
static_oci_graphs_requested: true
service_image_existence_verified: false
service_image_pull_or_execution_verified: false
live_runtime_closure_verified: false
live_distinct_service_uids_gids_verified: false
```

The two requested-static `true` values are construction requirements, not live
observations; the corresponding live predicates remain false.

The wider construction policy additionally fixes
`cooperative_admin_required=true`, `technical_control_domain_count=8`, and
`trusted_utc_time=false`. Those are construction assumptions and nonclaims,
not extra keys that a producer may inject into the exact profile mapping.

The producer is outside the verification authority. It may lie about paths,
digests, ordering, counts, results, cleanup, or completion. Acceptance therefore
depends on independent strict parsing, signatures from purpose-bound keys,
store-observed bytes, logical-time ordering, and producer-free replay. The
same administrator and local container daemon remain trusted. Root compromise,
daemon compromise, hostile kernel behavior, and administrative theft of role
volumes are outside the accepted threat model.

The exact role order and exact technical-control-domain order are:

```text
GOVERNANCE_ANCHOR
SOURCE_RUNTIME_CUSTODIAN
WIRE_QUALIFIER_R03
AUDITOR_QUALIFIER_R04
INTEGRATION_QUALIFIER_R05
CAS_STORE
LOGICAL_TIME
PURE_REPLAYER
```

Each role has one distinct purpose ID, control-domain ID, container identity,
purpose-private state volume, Ed25519 key ID, and public key. No private key is
accepted through repository bytes, profile bytes, argv, environment, stdin,
logs, public output, or a shared writable mount. Duplicate key IDs, duplicate
public keys, duplicate control-domain IDs, wrong-purpose signatures, wrong
epoch signatures, revoked keys, or cross-attempt signatures fail closed.

The candidate registry of public keys is pinned out of band and signed by
`GOVERNANCE_ANCHOR` before any future attempt. Adding a key after seeing
qualification output,
substituting an attacker registry, or allowing the producer to choose accepted
keys is forbidden. Purpose-private key volumes are retained while recovery is
required and destroyed only after a complete terminal has been durably read
back and producer-free verified. This lifecycle is a technical-control
statement, not a hardware non-exportability claim.

The v0 purpose and control-domain boundary is exact:

| Role | Exact control-domain ID | Exclusive authority purpose |
|---|---|---|
| `GOVERNANCE_ANCHOR` | `STEP5_GOVERNANCE_CONTROL` | authority-profile and role-registry trust anchor |
| `SOURCE_RUNTIME_CUSTODIAN` | `STEP5_SOURCE_RUNTIME_CUSTODY_CONTROL` | source/runtime custody and the custodian side of R01/R02 |
| `WIRE_QUALIFIER_R03` | `STEP5_R03_WIRE_QUALIFICATION_CONTROL` | R03 qualification decision |
| `AUDITOR_QUALIFIER_R04` | `STEP5_R04_AUDITOR_QUALIFICATION_CONTROL` | R04 qualification decision |
| `INTEGRATION_QUALIFIER_R05` | `STEP5_R05_INTEGRATION_QUALIFICATION_CONTROL` | R05 integration decision |
| `CAS_STORE` | `STEP5_CAS_STORE_CONTROL` | original-byte CAS publication, read-back and checkpoint receipts |
| `LOGICAL_TIME` | `STEP5_LOGICAL_TIME_CONTROL` | attempt-bound logical-order token chain |
| `PURE_REPLAYER` | `STEP5_PURE_REPLAY_CONTROL` | producer-free replay, the replay side of R01/R02, and final pure-verification decision |

The current R01 checkpoint candidate and R02 synthetic-branch candidate require
both `SOURCE_RUNTIME_CUSTODIAN` and `PURE_REPLAYER` signatures over their exact
purpose-specific preimages. Either signature alone is insufficient, but both
still leave R01 and R02 false. These dual signatures remain technical
separation under the disclosed same-admin model; they do not create
organizational independence.

## 4. Two-stage provenance without self-reference

Step 5 uses two construction stages.

### 4.1 Stage A: source and immutable images

Stage A freezes a source commit and canonical source archive, then builds the
five `runner`, `pure_replayer`, `cas_store`, `logical_time`, and
`governance_status` runtime images. The evidence basis must retain and bind the
OCI image index, image manifest, image configuration, ordered layer digests,
build inputs, source archive, and dependency lock. Each runtime is selected by
immutable image digest with no pull or network fallback. Authority/profile v2
currently records only requested static digest identities for those five
services; it is not evidence that Stage A has built or inspected any image.

The Step-5 image is a new runtime contract. If its pinned base exposes CPython
3.10.20, it must say so explicitly; it must not describe itself as the r13 host
CPython 3.10.12 runtime. Equality of scientific output, if established, is an
executed result of the new image and not identity with the old runtime.

### 4.2 Stage B: exact-byte trust lock and candidate bootstrap graph

In the retained authority-v1 path, after all Stage-A image digests exist,
Stage B constructs the candidate authority profile and governance-signed
candidate role registry outside those images, then freezes one external
`StageBTrustLock` over their exact RFC 8785 payload bytes. The v1 lock body has
exactly these fields:

```text
stage_b_trust_lock.body = {
  schema_version, status, canonicalization,
  authority_profile_id, authority_profile_payload_sha256,
  signer_registry_id, signer_registry_payload_sha256, qv2_binding
}
```

Its schema is
`hegel-machine-phase2b-formal-wire-covert-step5-stage-b-trust-lock/1`, its
status is `FROZEN_OUT_OF_BAND_STAGE_B_TRUST_LOCK`, and its canonicalization is
`RFC8785_EXACT_BYTES_NO_TRAILING_LF`. The loader accepts only `bytes` and
requires two independently provisioned pins:
`out_of_band_expected_lock_id` and
`out_of_band_expected_payload_sha256`. Copying either pin from the candidate
payload supplies no trust and violates the contract. The profile and registry
loaders are likewise bytes-only and exact-compare their content IDs and
payload SHA-256 values against the already loaded lock; decoded mappings or
producer-selected pins are rejected.

Authority/profile v2 freezes a separate `/2` Stage-B graph and does not
reinterpret the `/1` graph. Its lock additionally exact-binds `claim_level`,
`authority_profile_schema` and `signer_registry_schema`; its out-of-band lock
ID is exact ASCII bytes and its SHA-256 pin is a raw 32-byte digest. Its profile
uses the five exact ordered typed static-service bindings in Section 9.1.2.
Successful `/2` loading still proves only a candidate static graph; it does
not prove that the Stage-A images exist or ran.

Registry and R01/R02 candidate signatures are verified over exact canonical
preimages with purpose-specific domain separation by the real
`cryptography` Ed25519 backend. There is no injected or caller-selected
signature verifier. A missing backend, altered preimage, wrong role/purpose,
or wrong ordered signature set fails closed. Successful byte parsing and
cryptographic verification establishes only candidate bootstrap mechanics:
the profile, registry, R01, and R02 objects are all candidates and do not
satisfy a readiness node.

The profile directly binds the exact image digests, source commit,
source-archive SHA-256, dependency-lock digest, role policy, and QV2 basis.
The later runtime closure, evidence manifest, and CAS receipts must add the
archive size, fixture root, timeout contract, and full closure observations
without weakening those direct bindings. Neither the profile, registry, nor
trust lock is copied back into an already hashed Stage-A image.

This A-then-B construction avoids an image/profile digest cycle. Provenance
must disclose the two commits or immutable source states when Stage B adds the
external candidate graph after Stage A. A verifier must reject an image rebuilt
after Stage B, even when its source appears semantically equivalent.

### 4.3 Full runtime and import/native closure

At entry and exit, the source/runtime custodian must bind at least:

- exact OCI index, manifest, configuration, and layer digests;
- executable bytes, interpreter identity, ABI, and standard-library files;
- every imported project or standard-library module path and SHA-256 observed
  from the live module table, with built-in and frozen modules represented by
  their separately frozen origin kind rather than a fabricated path;
- the live executable and native mappings observed from `/proc/self/maps`,
  including interpreter, loader, and mapped shared libraries;
- exact argv, allowlisted environment keys, working directory, UID/GID,
  namespace identities, cgroup membership and limits, open-file-descriptor
  allowlist, and network-disabled state; and
- entry/exit closure equality under the frozen allowlist, with no unclassified
  imported or mapped executable object.

A requested container configuration is not live evidence. Missing mappings,
anonymous executable mappings not allowed by profile, late imports, path/hash
splices, image-digest drift, or producer-supplied closure summaries fail
closed. This remains local measured provenance, not remote attestation.

## 5. R01/R02 candidate heads and real-receipt boundary

### 5.1 R01: candidate checkpoint, not current authorization status

R01 must not accept a detached historical assertion that the owner
authorization was once active. The current foundation accepts only a
sequence-zero *candidate* checkpoint with these exact values:

```yaml
sequence: 0
event: INITIAL_ACTIVE_STATUS_CANDIDATE_NOT_LATEST_STATUS_EVIDENCE
effective_status: ACTIVE_CANDIDATE_NOT_VERIFIED_CURRENT
previous_status_head_id: null
revocation_receipt_id: null
revocation_count: 0
```

The candidate binds the exact owner-authorization payload bytes and content
ID and requires the exact ordered custodian-plus-pure-replayer Ed25519
signatures under the locked governance-signed candidate registry. This proves
only that the frozen bytes and candidate signatures match. Sequence zero
cannot establish that authorization is currently active and not revoked; it
does not authenticate an original-owner signature, establish a latest status
head, or exclude a later/off-path revocation.

A real R01 receipt remains unavailable until the mechanics-only owner-status
ledger is admitted under a new pinned registry/profile/trust-lock cascade and
a generic durable CAS checkpoint, a current-cut logical-time token, external
anti-fork anchoring, and producer-free replay jointly prove the latest
active/non-revoked state. The legacy real-receipt verifier
`verify_step5_r01_authorization_status_receipt_v1` therefore fails with
`FAIL_STEP5_AUTHORITY_V1_NOT_IMPLEMENTED`. No sequence-zero candidate may be
renamed, wrapped, or countersigned into a satisfied R01.

### 5.2 R02: synthetic/development branch candidate

The R02 candidate binds the exact protocol, exact freeze, owner authorization,
three logical-wire/stage-event identifiers, readiness DAG, Step-5 candidate
profile, candidate registry, candidate R01, QV2 basis, source basis, and
runtime-image basis. Its selected branch is exactly
`SYNTHETIC_DEVELOPMENT_QUALIFICATION_ONLY`, with input class exactly
`SYNTHETIC_DEV_ONLY`.

The branch admits the two frozen 1,024-row qualification fixtures only. It may
not allocate an actual-unsealed campaign ID, generate any member of the future
960-case cohort, admit recognizer or baseline inference, or consume a hidden
actual-corpus answer. Actual, mixed, ambiguous, or producer-selected branch
labels fail closed. Its ordered custodian-plus-pure-replayer signatures prove
only the declared candidate equality under R02-specific signature domains;
they do not establish independent branch replay or a stage event. A real R02
can become true only after an admitted real R01 and an independently verified
branch event. The legacy real-receipt verifier
`verify_step5_r02_synthetic_branch_receipt_v1` therefore fails with
`FAIL_STEP5_AUTHORITY_V1_NOT_IMPLEMENTED`.

## 6. Original-worker and full-result retention

This section specifies required future authority behavior. The current runtime
module reaches only the narrower caller-sink and statistical-projection
boundaries stated in Section 9.2.

The CAS-store v1 module supplies the exact-twelve request/1 path; CAS-store v2
adds a generic retention plan and complete presented-session bytes replay.
Both supply `SOCK_SEQPACKET`/`SCM_RIGHTS`, local create-if-absent storage,
Ed25519-signed receipt and append-only local checkpoint mechanics. Neither
connects the runtime's arbitrary caller-supplied sink to a fixed authority
deployment, starts a listener, establishes a live distinct UID/control domain,
protects the signing key, or proves external durability/anti-rollback.
Sections 9.3 and 9.4 freeze the exact current CAS-store and logical-time
mechanics without promoting this future authority behavior into completed
evidence.

The existing auditor creates six transient worker artifacts per mode. The
exact Step-5 retention order is invariance first, then statistics targets zero
through four:

```text
invariant-result.json
target-0.json
target-1.json
target-2.json
target-3.json
target-4.json
```

Across `KNOWN_NULL` and `KNOWN_BINDING_LEAK`, Step 5 must retain exactly twelve
original worker artifacts: ten statistics-target files and two invariance
files. The mode/object order is exactly `KNOWN_NULL` invariance then targets
0--4, followed by `KNOWN_BINDING_LEAK` invariance then targets 0--4. Exactly
four frozen cleanup calls occur: one invariance cleanup and one statistics
cleanup per mode. Reconstructing equivalent bytes from the final receipt is
required as a cross-check, but reconstruction is not original-byte retention.

Before the frozen cleanup path may unlink a worker artifact, a source-pinned
retention interposition opens that artifact relative to its already validated
directory descriptor with no symlink following. It transfers the open file
descriptor to the separate `CAS_STORE` process over an attempt-private Unix
socket using `SCM_RIGHTS`. The producer may not substitute a pathname, claimed
digest, or later copy for that descriptor. The store independently checks
regular-file type, link and mutation invariants, expected role/name, EOF, size,
and SHA-256; performs a zero-to-one CAS publication and durability barrier;
reads the object back; signs the store receipt; and only then acknowledges the
cleanup barrier.

The complete rich result matrix is retained in canonical chunks. Each chunk is
materialized once, opened under the same descriptor and no-follow discipline,
and transferred to `CAS_STORE` with `SCM_RIGHTS`. An index binds exact chunk
order, offsets, row counts, byte sizes, digests, and the total result count.
Missing, extra, reordered, overlapping, gapped, duplicated, post-hash-mutated,
or differently decoded chunks fail closed.

After receipt finalization, the pure replayer independently reconstructs all
twelve worker serializations from the complete retained audit receipt and
exact-compares their bytes, sizes, names, mode/target roles, and SHA-256 values
to the original CAS objects. A match does not erase the distinction between
the originals and reconstructions; both evidence paths must be present.

Cleanup remains blocked until all expected originals and result chunks have
signed, durable, read-back-verified store receipts. Store rejection, socket or
descriptor truncation, producer/store crash, missing acknowledgement, or
uncertain cleanup state aborts the whole qualification. No partial R04 or R05
receipt is issued.

## 7. Receipt chain and exact scientific boundary

### 7.1 R03: formal trusted-wire qualification

R03 consumes the exact R02 receipt and qualifies the formal trusted-wire
implementation in the fresh Step-5 runtime. Its evidence includes official and
adversarial RFC 8785 vectors, duplicate-key/Unicode/number boundaries, the
exact-ten-namespace case graph, exact 65,536-byte envelopes and capacity
boundaries, key-purpose separation, global shuffle, HMAC-derived public IDs,
source/answer binding, public decoding, and all 1,024 rows in both synthetic
modes. It binds source/runtime closure, original wire artifacts, CAS receipts,
logical-time token, and the `WIRE_QUALIFIER_R03` signature.

R03 is atomic. Any failed vector, unclassified field, ambiguous namespace,
capacity error, key-purpose collision, closure drift, missing envelope, or
signature/CAS/time failure leaves R03 false and prevents R04 admission.

### 7.2 R04: namespace-aware covert-auditor qualification

R04 consumes one exact R03 receipt from the same attempt, profile, fixture,
source/runtime basis, and registry epoch. It retains and verifies the complete
result matrix plus the twelve original worker artifacts for:

- 1,024 authority rows and 81 channel leaves per mode;
- ten frozen ID namespaces;
- 32 global renamings;
- 32 global case-order permutations;
- 16 within-case observation-order permutations;
- exactly 10,000 label permutations;
- five target families, target-maxT inference, Holm-Bonferroni family-wise
  alpha 0.01, NMI limit 0.02, and balanced-accuracy-advantage limit 0.05; and
- `standard_error` exactly disabled and unsupported.

The expected calibration direction is not optional: `KNOWN_NULL` must pass
with zero material witnesses and `KNOWN_BINDING_LEAK` must fail with at least
one independently replayed material binding witness. Expected-direction labels
cannot turn a malformed, incomplete, or statistically insufficient run into a
pass. `AUDITOR_QUALIFIER_R04` signs only after full retention and exact R03
dependency replay.

### 7.3 R05: integrated synthetic/dev qualification

R05 consumes the exact R03 and R04 receipts and reruns their cross-bindings as
one atomic integration decision. It verifies that wire bytes audited by R04 are
the same wire bytes qualified by R03; that both fixture modes, source rows,
private labels, public envelopes, full results, workers, runtime closure, CAS
checkpoint, logical-time chain, and attempt context are identical by root; and
that the null/leak directions are fresh results of this attempt.

`INTEGRATION_QUALIFIER_R05` may sign only after R03 and R04 are true under the
same authority profile. An exchanged receipt from another attempt, a missing
dependency, a root alias, historical r12/r13 evidence, or a producer summary
in place of retained objects leaves R05 false.

R05 is implementation qualification on synthetic/development inputs only. A
successful R05 would make R06 generator qualification and R07 three-baseline
registration eligible to proceed in parallel. It would not satisfy either,
would not satisfy R08 or any later node, and would not authorize actual 960-case
generation, actual-corpus audit, recognizer inference, baseline inference,
commitment, reveal, or scoring.

## 8. CAS, logical time, pure replay, overlay, and terminal

### 8.1 Durable CAS

Every public evidence object is canonicalized and addressed as
`sha256:<lowercase-hex>`. `CAS_STORE` alone publishes the object with
create-if-absent semantics. It rejects an existing object whose bytes, size,
type, ownership contract, or receipt history differs. Publication requires
file fsync, atomic link/rename under the frozen implementation, parent-directory
fsync, independent read-back hashing, and an append-only signed store receipt.

The store maintains a signed checkpoint chain over the exact ordered set of
objects. Missing or extra objects, overwrite, deletion, stale checkpoints,
rollback, alternate heads, equivocation, unacknowledged temporary files, and
same-name/different-byte or same-byte/different-role substitutions fail closed.
Deduplicated content may have multiple declared semantic roles, but every role
must be explicit in the manifest; content equality is not a signature.

CAS-store v1 implements the twelve-original-worker local path. CAS-store v2
implements a generic exact-order plan and complete raw presented-session replay
under an authority-v2-only static graph. Both are described in Section 9.3.
Their signed receipts can be cryptographically bound to a candidate Stage-B
profile/registry and `CAS_STORE` role key, but same-process, same-UID, local-
filesystem or fixture-key execution is not an authority deployment and cannot
satisfy `cas_store_verified` or unlock R01--R05.

### 8.2 Logical time

`LOGICAL_TIME` issues attempt-bound, purpose-bound, monotonically sequenced
tokens in a signed hash chain. Tokens bind the request digest, prior token root,
attempt context, registry epoch, event kind, and sequence. Replay, gaps,
rollback, forks, cross-attempt use, post-terminal issuance, or event-order
violations fail closed.

Logical time proves only frozen protocol order. Wall-clock strings, filesystem
mtimes, container start times, and host clock readings are diagnostic and are
never accepted as trusted UTC, timestamp authority, or proof of external
notarization.

The current logical-time foundation implements canonical request/token/head/
checkpoint bytes, Ed25519 token verification, a contiguous local ledger, and
complete bytes-only replay against an exact caller-supplied event plan. Its
whole-chain result is privately issued and remains mechanics-only. It
intentionally does not decide whether an event purpose is allowed, prove a
fresh attempt, enforce terminal policy, attest an image/UID/control domain,
establish an external anti-fork current cut, or commit the complete ledger into
an externally durable CAS checkpoint. Those external predicates remain
mandatory before any token or replay result can enter authority evidence.

### 8.3 Producer-free pure verification

`PURE_REPLAYER` receives only the frozen profile, public registry, canonical
evidence manifest, CAS objects, and public signatures. It does not import or
call the producer, trust producer caches, read producer private state, or accept
producer-selected paths. A separately built and digest-pinned implementation
must:

1. strict-parse every schema and reject unknown, missing, duplicate, or
   noncanonical fields;
2. recompute every content root and signature preimage;
3. replay R01, R02, R03, R04, then R05 in dependency order;
4. decode all full-result chunks and recompute complete statistics and every
   retained original-worker serialization;
5. replay runtime closure, CAS store receipts/checkpoints, and logical-time
   chain; and
6. emit one signed pure-verification result without mutating evidence.

Any divergence between producer and pure replay is a whole-attempt failure,
even if both independently report a scientific pass.

### 8.4 Readiness overlay and terminal

A future readiness evidence overlay is append-only evidence, not a mutation of
the frozen DAG. It starts with all sixteen nodes false. Each transition binds
the exact prior overlay root, corresponding receipt, dependency roots,
logical-time token, and CAS checkpoint. It cannot skip, reorder, unset, or
infer a node. The current all-false candidate snapshot is not such an overlay.

At most one terminal may be published for an attempt. A success terminal is
eligible only after the final manifest, store checkpoint, R01--R05 receipt
chain, pure-verification result, overlay, runtime exit closure, and residue
audit all exact-compare and are durably read back. A failure or uncertain
terminal preserves false readiness and records the earliest classified stop;
it is never retried in place. Publication interruption, ambiguous terminal
linkage, or a second terminal leaves the attempt failed and requires a new
attempt ID.

## 9. Frozen v0 artifact graph and schema boundary

The exact artifact-graph order is:

```text
authority_profile
role_registry
technical_independence
attempt_context
authorization_status_head
r01
synthetic_branch
r02
runtime_closure
cas_store_receipt
worker_retention_batch
full_results_chunk
full_results_index
audit_receipt_projection
r03
r04
r05
logical_time
evidence_manifest
cas_checkpoint
pure_verification
readiness_overlay
terminal
```

### 9.1 Exact current candidate-foundation surfaces

#### 9.1.1 Authority v1 R01/R02 candidate surface

The authority-v1 foundation covers an out-of-band-pinned Stage-B trust lock,
candidate authority profile, governance-signed candidate registry,
dual-signed R01 candidate, dual-signed R02 candidate, and a non-evidence
candidate snapshot. `VerifiedArtifact` in this source means that exact bytes,
bindings, and real signatures passed the foundation parser; it never means
that a readiness predicate passed. All four profile/registry/R01/R02 objects
remain candidate bootstrap objects. R03--R05, real R01/R02 receipts, and the
remaining authority artifacts still require their machine schemas, durable
services, KAT admission, and live evidence.

The exact current accepted wire wrappers are:

```text
stage_b_trust_lock = {body, id}
authority_profile = {body, id}
role_registry = {body, id, signatures}
r01_candidate = {body, id, signatures}
r02_candidate = {body, id, signatures}
```

The derived candidate snapshot is not an accepted overlay wire object. It is a
privately issued foundation result with `kind=readiness_candidate`, one
content-addressed `artifact_id`, the exact body below, and no signature
backend. It cannot be serialized and reintroduced as readiness evidence merely
by wrapping it as `{body,id}`.

The exact schema versions are:

```text
stage_b_trust_lock:
  hegel-machine-phase2b-formal-wire-covert-step5-stage-b-trust-lock/1
authority_profile:
  hegel-machine-phase2b-formal-wire-covert-step5-authority-profile/1
role_registry:
  hegel-machine-phase2b-formal-wire-covert-step5-signer-registry/1
detached_signature:
  hegel-machine-phase2b-formal-wire-covert-step5-detached-signature/1
r01_candidate:
  hegel-machine-phase2b-formal-wire-covert-step5-r01-authorization-status-candidate/1
r02_candidate:
  hegel-machine-phase2b-formal-wire-covert-step5-r02-synthetic-branch-candidate/1
readiness_candidate_snapshot:
  hegel-machine-phase2b-formal-wire-covert-step5-readiness-candidate-snapshot/1
role_key:
  hegel-machine-phase2b-step5-role-key/1
authorization_status_checkpoint_candidate:
  hegel-machine-phase2b-step5-owner-authorization-status-checkpoint-candidate/1
```

Every brace-delimited line below is an exact keyset, not an illustrative
subset:

```text
stage_b_trust_lock.body = {
  schema_version, status, canonicalization,
  authority_profile_id, authority_profile_payload_sha256,
  signer_registry_id, signer_registry_payload_sha256, qv2_binding
}

authority_profile.body = {
  schema_version, status, claim_level, canonicalization, profile_epoch,
  owner_authorization_binding, readiness_dag_binding, qv2_binding,
  source_binding, runtime_image_bindings, role_policy, governance_root,
  threat_model, permanent_nonclaims
}

owner_authorization_binding = {
  authorization_schema, authorization_id, authorization_config_sha256,
  authorization_source_sha256, protocol_id, exact_freeze_version,
  exact_freeze_id, authority_class, authority_requirement_id,
  actor_profile_content_id, actor_profile_canonical_sha256,
  logical_stage_wire_schema_id, logical_stage_wire_policy_id,
  stage_event_contract_manifest_id
}

readiness_dag_binding = {schema_version, readiness_dag_id}
qv2_binding = {
  schema_version, profile_id, profile_payload_sha256, source_sha256,
  dependency_lock_sha256
}
source_binding = {
  source_commit_sha, source_archive_sha256, dependency_lock_sha256
}
runtime_image_bindings = {runner, pure_replayer, cas_store, logical_time}
role_policy = {
  roles_in_exact_order, role_count, distinct_public_keys, distinct_key_ids,
  distinct_control_domains, key_epoch
}

role_registry.body = {
  schema_version, authority_profile_id, registry_epoch, status,
  revocation_epoch, role_keys_in_exact_order
}
role_key = {
  schema_version, role, control_domain, authorized_signature_purposes,
  algorithm, key_epoch, public_key_hex, key_id
}
detached_signature = {
  schema_version, algorithm, role, purpose, key_epoch, signer_key_id,
  signature_hex
}

r01_candidate.body = {
  schema_version, authority_profile_id, signer_registry_id,
  owner_authorization_binding, owner_authorization_payload_sha256,
  status_checkpoint_candidate, claim_boundary
}
authorization_status_checkpoint_candidate = {
  schema_version, authorization_id, sequence, event, effective_status,
  previous_status_head_id, revocation_receipt_id, revocation_count
}

r02_candidate.body = {
  schema_version, authority_profile_id, signer_registry_id, r01_candidate_id,
  declared_expected_bindings,
  declared_observed_selected_branch_bindings,
  qv2_binding, source_binding, runtime_image_bindings, branch_selection,
  claim_boundary
}
declared_expected_bindings = {
  owner_authorization_id, readiness_dag_id, protocol_id,
  exact_freeze_version, exact_freeze_id, logical_stage_wire_schema_id,
  logical_stage_wire_policy_id, stage_event_contract_manifest_id
}
declared_observed_selected_branch_bindings = {
  owner_authorization_id, readiness_dag_id, protocol_id,
  exact_freeze_version, exact_freeze_id, logical_stage_wire_schema_id,
  logical_stage_wire_policy_id, stage_event_contract_manifest_id
}
branch_selection = {
  selected_branch, input_class, synthetic_input, actual_corpus_input,
  actual_input_count, actual_campaign_id, actual_attempt_id,
  actual_material_consumed
}

readiness_candidate_snapshot.body = {
  schema_version, claim_level, authority_profile_id, signer_registry_id,
  readiness_dag_id, node_states_in_canonical_topological_order,
  satisfied_node_count, highest_contiguous_satisfied_node,
  all_nodes_satisfied, actual_campaign_id, actual_attempt_id,
  actual_input_consumed, r01_satisfied, r02_satisfied,
  r03_or_later_unlocked, readiness_evidence_overlay_issued,
  cas_store_verified, logical_time_verified
}
readiness_candidate_snapshot.node_state = {
  node_id, satisfied, candidate_artifact_id
}
```

The exact current header and role-policy values are:

```yaml
authority_profile.status: FROZEN_OWNER_ACCEPTED_TECHNICAL_AUTHORITY_PROFILE
authority_profile.claim_level: SAME_ADMIN_TECHNICAL_AUTHORITY_NOT_EXTERNAL_OR_ORGANIZATIONAL_INDEPENDENCE
authority_profile.canonicalization: RFC8785_EXACT_BYTES_NO_TRAILING_LF
authority_profile.profile_epoch: 1
role_policy.role_count: 8
role_policy.distinct_public_keys: true
role_policy.distinct_key_ids: true
role_policy.distinct_control_domains: true
role_policy.key_epoch: 1
role_registry.registry_epoch: 1
role_registry.status: ACTIVE
role_registry.revocation_epoch: 0
role_key.algorithm: Ed25519
role_key.key_epoch: 1
detached_signature.algorithm: Ed25519
```

`role_registry.status=ACTIVE` is only the exact internal value of the pinned
candidate registry. It is not live-domain qualification, a latest governance
status-ledger statement, or readiness evidence.

The exact QV2 binding is the `/9` profile and includes its dependency lock:

```yaml
schema_version: hegel-machine-phase2b-formal-wire-covert-qualification/9
profile_id: phase2b_formal_wire_covert_qualification_profile_v2_ad20443a8f68a9f07492a532adb389c8e73234d343a14bc547a91d868a935d6b
profile_payload_sha256: 1c2f40f8227a8e3b7f7d56bce00db8ea410d60ac3dd006fd53d7fd7fa7b2297e
source_sha256: 921bbb771d27c339acaf2c5e1e48f35a899c56552929065c02894e590d32e2ed
dependency_lock_sha256: 0ccd291a28df11fa0e7e7399a086bf01516a20fa2aa5e8abf3a05687b54b85db
```

The owner binding includes these three exact logical-contract IDs in addition
to the authorization/protocol/freeze/actor bindings:

```yaml
logical_stage_wire_schema_id: phase2b_c1_logical_stage_wire_schema_v1_1a625fbc369b057b9fc295fec02391d0421fc071ca41905b15901d4365833b5d
logical_stage_wire_policy_id: phase2b_c1_logical_stage_wire_policy_v2_c88653f2caa92eae1dfb7456217c4ca05c339e8f23e0540a06e435552dc59573
stage_event_contract_manifest_id: phase2b_c1_stage_event_contract_manifest_v1_c0929d05bdae56205e08700ab760e88a4668c686e73fe4be7a00ce6d69f9baf7
```

Production loading is exact-byte only. `load_step5_stage_b_trust_lock_v1`
requires the independently supplied lock ID and payload-SHA pins;
`load_step5_authority_profile_v1` and `load_step5_signer_registry_v1` require
that loaded lock. The R01/R02 candidate loaders likewise accept canonical
candidate bytes, and R01 separately binds the exact owner-authorization
payload bytes. The decoded-object verifier entry points fail closed. Detached
signatures are checked only by the real `cryptography` Ed25519 backend over
purpose-domain-separated canonical preimages; an unavailable backend or
injected verifier cannot be treated as success.

The exact authority-profile permanent nonclaims are:

```yaml
actual_corpus_input: false
actual_campaign_id_allocated: false
actual_attempt_id_allocated: false
actual_execution_authorized: false
r01_satisfied: false
r02_satisfied: false
r03_satisfied: false
r04_satisfied: false
r05_satisfied: false
r06_or_later_unlocked: false
cas_store_verified: false
logical_time_verified: false
readiness_evidence_overlay_issued: false
technical_control_domains_live_verified: false
organizational_independence: false
independent_human_actors: false
external_authority: false
remote_attestation: false
hardware_key_nonexportability: false
malicious_admin_resistance: false
```

The exact R01-candidate `claim_boundary` is:

```yaml
owner_authorization_exact_bytes_verified: true
owner_signature_authenticated_by_original_authorization: false
governance_status_ledger_verified: false
latest_revocation_status_verified: false
cas_store_checkpoint_verified: false
logical_time_verified: false
r01_satisfied: false
r02_or_later_satisfied: false
execution_authorized: false
actual_input_consumed: false
organizational_independence: false
independent_human_actors: false
```

The exact R02-candidate `claim_boundary` is:

```yaml
declared_binding_equality_verified: true
independent_branch_replay_verified: false
branch_stage_event_verified: false
cas_store_checkpoint_verified: false
logical_time_verified: false
r01_satisfied: false
r02_satisfied: false
r03_satisfied: false
r04_satisfied: false
r05_satisfied: false
actual_execution_authorized: false
actual_corpus_qualified: false
campaign_or_attempt_id_allocated: false
organizational_independence: false
independent_human_actors: false
```

The current candidate role registry freezes these exact ordered
signature-purpose lists:

```text
GOVERNANCE_ANCHOR = [SIGN_SIGNER_REGISTRY]
SOURCE_RUNTIME_CUSTODIAN = [
  SIGN_R01_AUTHORIZATION_STATUS_CANDIDATE,
  SIGN_R02_SYNTHETIC_BRANCH_CANDIDATE
]
WIRE_QUALIFIER_R03 = [SIGN_R03_WIRE_QUALIFICATION]
AUDITOR_QUALIFIER_R04 = [SIGN_R04_AUDITOR_QUALIFICATION]
INTEGRATION_QUALIFIER_R05 = [SIGN_R05_INTEGRATION_QUALIFICATION]
CAS_STORE = [SIGN_CAS_IF_ABSENT_RECEIPT]
LOGICAL_TIME = [SIGN_LOGICAL_TIME_TOKEN]
PURE_REPLAYER = [
  COUNTERSIGN_R01_AUTHORIZATION_STATUS_CANDIDATE,
  COUNTERSIGN_R02_SYNTHETIC_BRANCH_CANDIDATE,
  SIGN_PURE_REPLAY_RESULT
]
```

Only the candidate registry and R01/R02-candidate signature domains are
present in the current foundation surface. Listing the
R03/R04/R05/store/time/pure-result purposes in a role key reserves and
constrains their future purpose assignments; it is not an implementation,
issuance, service-liveness, or readiness claim.

The exact R02 branch values are:

```yaml
selected_branch: SYNTHETIC_DEVELOPMENT_QUALIFICATION_ONLY
input_class: SYNTHETIC_DEV_ONLY
synthetic_input: true
actual_corpus_input: false
actual_input_count: 0
actual_campaign_id: null
actual_attempt_id: null
actual_material_consumed: false
```

The only derivable object is
`derive_step5_r01_r02_candidate_snapshot_v1`. It sets all sixteen node rows to
`satisfied=false`; only the first two rows carry their corresponding candidate
artifact IDs. It fixes `satisfied_node_count=0`,
`highest_contiguous_satisfied_node=null`, `all_nodes_satisfied=false`, both
actual IDs null, `actual_input_consumed=false`, `r01_satisfied=false`,
`r02_satisfied=false`, `r03_or_later_unlocked=false`,
`readiness_evidence_overlay_issued=false`, `cas_store_verified=false`, and
`logical_time_verified=false`. The legacy
`derive_step5_r01_r02_readiness_overlay_v1` entry point fails with
`FAIL_STEP5_AUTHORITY_V1_NOT_IMPLEMENTED`; it cannot issue an overlay or turn
R01/R02 true.

#### 9.1.2 Authority/profile v2 five-static-service candidate surface

The authority/profile v2 source SHA-256 is
`03d689327ca422f93e5a32db0a7f813fe84b1099f1bac1b31395c975d76d8e86`;
its focused test source SHA-256 is
`e11b0c2f8056ccca8534acef3628e1be6a870727441f2b199dd0039a28db2501`.
At this candidate BYTE-STOP, the focused v2 suite reported 37/37. This is a
construction-test result, not independent-review admission, image existence,
live evidence, authority, or readiness.

The exact v2 schema identifiers are:

```text
authority_profile:
  hegel-machine-phase2b-formal-wire-covert-step5-authority-profile/2
stage_b_trust_lock:
  hegel-machine-phase2b-formal-wire-covert-step5-stage-b-trust-lock/2
signer_registry:
  hegel-machine-phase2b-formal-wire-covert-step5-signer-registry/2
detached_signature:
  hegel-machine-phase2b-formal-wire-covert-step5-detached-signature/2
service_static_binding:
  hegel-machine-phase2b-formal-wire-covert-step5-service-static-binding/2
readiness_candidate_snapshot:
  hegel-machine-phase2b-formal-wire-covert-step5-readiness-candidate-snapshot/2
role_key:
  hegel-machine-phase2b-step5-role-key/1
```

V2 preserves the exact eight-role order from Section 3; it does not add a
ninth role. Its complete ordered purpose policy is:

```text
GOVERNANCE_ANCHOR = [
  SIGN_SIGNER_REGISTRY,
  SIGN_OWNER_AUTHORIZATION_STATUS_EVENT
]
SOURCE_RUNTIME_CUSTODIAN = [
  SIGN_R01_AUTHORIZATION_STATUS_CANDIDATE,
  SIGN_R02_SYNTHETIC_BRANCH_CANDIDATE,
  SIGN_RUNTIME_CLOSURE,
  SIGN_R01_AUTHORIZATION_STATUS_RECEIPT
]
WIRE_QUALIFIER_R03 = [SIGN_R03_WIRE_QUALIFICATION]
AUDITOR_QUALIFIER_R04 = [SIGN_R04_AUDITOR_QUALIFICATION]
INTEGRATION_QUALIFIER_R05 = [SIGN_R05_INTEGRATION_QUALIFICATION]
CAS_STORE = [SIGN_CAS_IF_ABSENT_RECEIPT]
LOGICAL_TIME = [SIGN_LOGICAL_TIME_TOKEN]
PURE_REPLAYER = [
  COUNTERSIGN_R01_AUTHORIZATION_STATUS_CANDIDATE,
  COUNTERSIGN_R02_SYNTHETIC_BRANCH_CANDIDATE,
  SIGN_PURE_REPLAY_RESULT,
  COUNTERSIGN_R01_AUTHORIZATION_STATUS_RECEIPT,
  SIGN_OWNER_STATUS_ANTI_FORK_ANCHOR
]
```

The new purposes are reservations in a cryptographically closed candidate
policy, not evidence that a runtime closure, R01 receipt, countersignature or
external anti-fork anchor was issued. Role-key/1 remains the role-key schema,
but its content ID is recomputed over the complete v2 purpose list. The
registry still requires exactly eight pairwise-distinct role key IDs, public
keys and control domains, key epoch 1, and real governance Ed25519 verification
over the exact purpose-domain-separated preimage.

The exact static service order, service type, role and requested UID/GID are:

| Service | Type | Existing role | UID | GID |
|---|---|---|---:|---:|
| `runner` | `RUNNER_STATIC_SERVICE` | `SOURCE_RUNTIME_CUSTODIAN` | 41001 | 42001 |
| `pure_replayer` | `PURE_REPLAYER_STATIC_SERVICE` | `PURE_REPLAYER` | 41002 | 42002 |
| `cas_store` | `CAS_STORE_STATIC_SERVICE` | `CAS_STORE` | 41003 | 42003 |
| `logical_time` | `LOGICAL_TIME_STATIC_SERVICE` | `LOGICAL_TIME` | 41004 | 42004 |
| `governance_status` | `GOVERNANCE_STATUS_STATIC_SERVICE` | `GOVERNANCE_ANCHOR` | 41005 | 42005 |

Each row has exactly this field surface:

```text
service_static_binding = {
  schema_version, service_name, service_type, binding_status, role,
  control_domain, uid, gid, role_key_id, key_epoch,
  authorized_signature_purposes, oci_index_digest, oci_manifest_digest,
  oci_config_digest, ordered_layer_digests,
  static_runtime_closure_policy_sha256, binding_id
}
```

The binding status is exactly
`STATIC_REQUESTED_BINDING_NOT_OBSERVED_LIVE`. All five binding IDs, role-key
IDs, UIDs and GIDs are pairwise distinct; all ten UID/GID values are globally
distinct. Every service binds its exact role, control domain, key epoch and
complete authorized-purpose list. Its nonzero OCI index, manifest and config
digests are mutually distinct, all fifteen graph digests are distinct across
services, and its ordered layer list is nonempty, content-bound, duplicate-
free and order-sensitive. The nonzero static runtime-closure-policy SHA-256 is
only the identity of a requested policy. It does not prove that any image,
layer or policy artifact exists, that a container ran, or that an entry/exit
runtime closure matched it.

The profile retains the exact fourteen-field owner binding and exact QV2 `/9`
binding. Its role policy fixes `role_count=8`, `purpose_policy_version=2`, the
five-service order, `service_binding_count=5`, distinct requested service
UIDs/GIDs and `static_service_identity_only=true`. The `governance_status`
service is deliberately bound to the existing `GOVERNANCE_ANCHOR` root and
`STEP5_GOVERNANCE_CONTROL`, including the exact two-purpose governance list.

Production loading is canonical-bytes-only. The Stage-B v2 loader requires
the lock ID as exact ASCII `bytes` and the payload SHA-256 as an independently
supplied raw 32-byte digest; strings, bytearrays, hex-text digests, payload-
copied pins or wrong pins fail closed. Profile and registry loading then close
the exact lock/profile/registry/service-key graph. Decoded-object verifier
entry points and injected verifier callbacks are unavailable.

`Phase2BStep5VerifiedArtifactV2` is constructor-disabled. Issuance and require
capabilities live only in a closure and are deleted as module globals after
the bytes-only public call graph captures them. The closure retains an object-
identity and immutable-field record, so `object.__new__` plus copied fields,
cross-graph token mixing, or mutation of a genuinely issued frozen object is
rejected. Capturing critical validators and Ed25519 verification before the
public entries are published also prevents ordinary module-global rebinding
from laundering invalid bytes or signatures. This is a Python caller/KAT
boundary, not resistance to arbitrary interpreter-memory, closure-cell or
loaded-code mutation; live image/runtime-closure integrity remains external.

An independent runtime-closure review found a transitive constant-capture P1
in the preceding v2 bytes: a public loader had captured its loader callable,
but that callable still resolved accepted schema and contract constants from
mutable module globals at call time. An ordinary global replacement could
therefore admit a fully self-consistent, re-content-addressed, re-signed and
out-of-band-pinned wrong-schema graph. The current source fixes that boundary
by definition-time private capture across the public loader call graph for the
accepted schemas, domains, prefixes, QV2 pins, roles, services, purposes, hard
bounds, regular expressions, canonical decoder/encoder and concrete Ed25519
backend. The focused suite now includes a self-consistent EVIL graph KAT and
rejects it. This is still an ordinary Python rebinding/KAT property, not
protection against closure-cell, interpreter-memory or loaded-code mutation,
and it does not establish any live authority predicate.

The only derivable v2 result is a non-evidence candidate snapshot. Profile,
registry and snapshot all retain the sixteen exact readiness rows as false,
with `satisfied_node_count=0`, `all_nodes_satisfied=false`, image existence and
execution false, static bindings live-verified false, runtime closure false,
distinct live UID/GID false, CAS/logical time/governance-status/latest-status/
external-anchor/R01 receipt false, R01 false, R02-or-later false, overlay false
and execution authorization false. The public runtime-closure, real-R01 and
R01-overlay entry points fail exactly with
`FAIL_STEP5_AUTHORITY_V2_NOT_IMPLEMENTED`.

Authority/profile v2, generic CAS-store v2, logical-time schema/1 and
owner-status schema/1 are still separate mechanics modules. Logical time and
owner status now accept only authority-profile/2 and signer-registry/2 ID
prefixes; owner status additionally byte-matches the v2 two-purpose governance
role-key projection, but neither loads the raw authority graph. Generic CAS-
store v2 verifies the raw authority-profile/2 and signer-registry/2 graph but
remains a local mechanics service. No current source composes those surfaces
into one admitted profile-v2 CAS/status/time chain. Thus the purpose cascade
and fifth static binding cannot be interpreted as a live fifth service, a
current owner-status head, a real R01 receipt, or a readiness overlay.

### 9.2 Exact current runtime mechanics surfaces

#### 9.2.1 Retention and statistical-result projection

The current runtime module provides prospective producer-side retention
interposition and a separate statistical-result plus exact feature-manifest
preimage projection. Its frozen source SHA-256 is
`35315054bf360a6c7148e79b018f7657a16b19d5f833563693461830422b4979`;
the focused test source SHA-256 is
`e74c4a4dcee2e3f5c9fca1207001a4826530d8c351abb7aa90928dfde27037fd`.
Its runtime schema is
`hegel-machine-phase2b-formal-wire-covert-step5-runtime/1`. It pins the
Auditor source SHA-256 to
`5d77403dc41c661cb8a8d7deb90c9b2644de7a27fda9369c37f76eb82474a9be`
and QV2 source SHA-256 to
`921bbb771d27c339acaf2c5e1e48f35a899c56552929065c02894e590d32e2ed`.
Import-time function defaults capture both source pins, both original cleanup
callables, both metric-projection callables, and the receipt-validator
callable. Synchronized rebinding of the public `_ORIGINAL_*` names, live
auditor names, source paths, and live pin constants therefore fails closed.
The bootstrap still requires a source-pinned fresh process; pre-import
monkeypatching or arbitrary same-process code/function-default mutation is
outside this in-process check and requires future fixed-image/runtime-closure
controls. These guards remain mechanics, not authority.

The exact retention API is
`RetentionSessionV1(*, attempt_context_id, retention_session_id, sink)`. Within
the one-shot context, the caller must run `begin_mode("KNOWN_NULL")`, the null
audit, and `crosscheck_and_complete_mode(receipt)`, followed by the same exact
sequence for `KNOWN_BINDING_LEAK`; successful exit requires exactly twelve
retained objects, twelve sink calls, twelve acknowledgements, and four frozen
cleanup calls. The sink API is
`ingest_open_fd(metadata, fd, sha256, size_bytes) -> RetentionIngestAckV1`.
The runtime admits only exact DTO classes, exact tuples/dicts, exact strings,
and exact integers/booleans at their corresponding boundaries; subclasses,
coercions, DTO-like alternates, and `bool` substituted for an integer fail
closed. This includes exact `RetentionIngestAckV1`,
`RetainedWorkerObjectV1`, `FormalCovertAuditReceiptV1`, runtime-evidence,
result-chunk, feature-manifest-chunk, and aggregate-projection objects.

The request and acknowledgement schemas are:

```text
retention_request:
  hegel-machine-phase2b-formal-wire-covert-step5-scm-request/1
retention_ack:
  hegel-machine-phase2b-formal-wire-covert-step5-retention-ack/1
```

Their exact field surfaces are:

```text
retention_request = {
  schema_version, attempt_context_id, retention_session_id, ordinal, mode,
  phase, target_index_or_null, source_basename, auditor_reported_sha256,
  auditor_reported_size_bytes, fd_count, request_id
}
retention_ack = {
  schema_version, attempt_context_id, retention_session_id, request_id,
  ordinal, mode, phase, source_basename, content_address, sha256, size_bytes,
  durable_before_return, readback_verified, store_receipt_ref
}
```

The request uses exactly `fd_count=1`; its `request_id` is `sha256:` plus the
SHA-256 of the canonical RFC 8785 request fields excluding `request_id`. The
acknowledgement exact-compares attempt, session, request, ordinal, mode, phase,
basename, digest, and size, and requires `content_address==sha256`,
`durable_before_return=true`, and `readback_verified=true`. The exact twelve
ordinals are the two-mode order in Section 6. A DTO-like alternate type,
cross-attempt/session/request swap, wrong ordinal, changed bytes, reentrant
sink, incomplete count, or unverified acknowledgement fails before the
corresponding frozen cleanup is admitted.

After the original cleanup returns, the runtime `fstat`s the already-held
directory descriptor, requires it still denotes a directory, and on Linux
requires `st_nlink==0`. It then requires the descriptor view to be empty and
the original pathname to be absent. Merely deleting the expected files and
renaming the still-linked directory away is therefore nonqualifying.

The retention claim boundary is exactly:

```text
ORIGINAL_TRANSIENT_WORKER_BYTES_RETAINED_BEFORE_FROZEN_CLEANUP_UNDER_CALLER_SUPPLIED_SINK_ACK_NOT_CAS_IMPLEMENTATION_NOT_SIGNATURE_AUTHORITY_NOT_RUNTIME_IMAGE_ATTESTATION_NOT_R03_R04_R05
```

Any caller-provided sink and its current ACK DTO remain non-authoritative. A
`store_receipt_ref` is only a bounded identifier; the current module neither
authenticates a CAS-store signature nor implements `SCM_RIGHTS`. Actual Step-5
execution must replace arbitrary sink selection with one fixed,
source-pinned Unix-domain `SCM_RIGHTS` client and must verify a purpose-bound,
signed, durable/read-back CAS-store acknowledgement before cleanup. Until that
client, CAS service, status ledger, logical time, and authority receipts exist,
the live gate remains `HOLD`.

The separate projector API is
`project_statistical_results_chunks_v1(receipt, *, attempt_context_id,
retention_session_id, mode)`. It invokes the original pinned receipt validator,
requires the exact 1,024-row receipt and null/pass or leak/fail direction,
checks complete ordered target-feature and target-family coverage, and emits
the statistical/target-family subprojection together with the exact replayable
`FeatureManifestV1` preimage. Its schemas are:

```text
feature_manifest_chunk:
  hegel-machine-phase2b-formal-wire-covert-step5-feature-manifest-chunk/1
statistical_results_chunk:
  hegel-machine-phase2b-formal-wire-covert-step5-statistical-results-chunk/1
statistical_results_manifest:
  hegel-machine-phase2b-formal-wire-covert-step5-statistical-results-manifest/2
```

Their exact current keysets are:

```text
feature_manifest_chunk.body = {
  schema_version, auditor_schema_version, auditor_semantics_version,
  attempt_context_id, retention_session_id, mode, receipt_id,
  feature_manifest_root, concrete_feature_count, chunk_index,
  feature_start, feature_end_exclusive, feature_count, channel_groups,
  unique_id_feature_family, uuid_leaf_count, channel_leaf_count,
  raw_semantic_feature_count, deduplicated_feature_count, entries,
  projection_scope, claim_boundary, trusted_bootstrap_requirement
}
feature_manifest_chunk.entry = {
  feature_index, concrete_feature_id, semantic_aliases
}
feature_manifest_chunk.ref = {
  chunk_index, feature_start, feature_end_exclusive, sha256, size_bytes
}
statistical_results_chunk.body = {
  schema_version, auditor_schema_version, auditor_semantics_version,
  attempt_context_id, retention_session_id, mode, receipt_id,
  audited_row_count, feature_manifest_root, concrete_feature_count,
  target_feature_result_count,
  chunk_kind, chunk_index, row_start, row_end_exclusive, row_count,
  row_projection_sha256, rows, projection_scope, claim_boundary,
  trusted_bootstrap_requirement
}
statistical_results_manifest.body = {
  schema_version, auditor_schema_version, auditor_semantics_version,
  attempt_context_id, retention_session_id, mode, receipt_id,
  audited_row_count, feature_manifest_root, concrete_feature_count,
  target_order, target_feature_result_count,
  target_family_result_count, rows_per_nonfinal_target_feature_chunk,
  maximum_chunk_bytes, feature_manifest_chunk_schema_version,
  maximum_feature_manifest_entries_per_chunk,
  maximum_feature_manifest_chunk_bytes, feature_manifest_chunks,
  target_feature_chunks, target_family_chunk,
  projection_scope, claim_boundary, trusted_bootstrap_requirement
}
statistical_results_manifest.chunk_ref = {
  chunk_index, row_start, row_end_exclusive, row_projection_sha256,
  sha256, size_bytes
}
target_feature_result_row = {
  target, feature_id, raw_nmi_hex, symmetric_permutation_mean_nmi_hex,
  bias_corrected_nmi_hex, balanced_accuracy_advantage_hex,
  observed_omnibus_hex, pairwise_exceedance_count, pairwise_raw_p,
  pairwise_holm_adjusted_p, feature_max_t_exceedance_count,
  feature_max_t_p, target_holm_critical_value,
  target_family_holm_rejected, material_effect,
  feature_localization_rejected, failure_witness, passed
}
target_family_result_row = {
  target, observed_target_max_omnibus_hex, exceedance_count, raw_p,
  holm_adjusted_p, holm_rank, holm_critical_value, holm_rejected
}
probability_projection = {numerator, denominator}
```

The exact projection DTO field surfaces are:

```text
CanonicalFeatureManifestChunkV1 = {
  attempt_context_id, retention_session_id, mode, receipt_id,
  feature_manifest_root, concrete_feature_count, chunk_index,
  feature_start, feature_end_exclusive, sha256, size_bytes, canonical_bytes
}
CanonicalResultChunkV1 = {
  attempt_context_id, retention_session_id, mode, receipt_id,
  audited_row_count, feature_manifest_root, concrete_feature_count,
  target_feature_result_count, chunk_kind, chunk_index, row_start,
  row_end_exclusive, row_projection_sha256, sha256, size_bytes,
  canonical_bytes
}
StatisticalResultsProjectionV1 = {
  attempt_context_id, retention_session_id, mode, receipt_id,
  audited_row_count, feature_manifest_root, concrete_feature_count,
  target_feature_result_count, feature_manifest_chunks,
  target_feature_chunks, target_family_chunk, manifest_sha256,
  manifest_size_bytes, manifest_canonical_bytes
}
```

Chunks are exact RFC 8785/JCS bytes with no appended line feed and at most
2 MiB each. Every non-final target-feature chunk has exactly 1,024 rows; the
number of chunks follows the exact concrete-feature count. One separate
target-family chunk covers the five targets. Every chunk and manifest binds the
attempt context, retention session, mode, validated receipt ID, exact audited
row count, feature-manifest root, concrete-feature count, and result count.
Each canonical reference binds both the projected-row SHA-256 and the complete
chunk-byte SHA-256/size; gaps, overlaps, reorderings, incomplete coverage, or
bytes/DTO divergence fail closed. There is no excluded `receipt_id` field.

Feature-manifest chunks carry at most 256 entries unless the 2 MiB byte bound
requires a smaller deterministic split. They retain ordered concrete feature
IDs, exact alias coverage, channel groups, unique-ID family, and all manifest
counts. Pure validation reconstructs an exact `FeatureManifestV1`, replays its
`manifest_root`, regenerates and exact-compares the canonical split, and binds
every target-feature row's `feature_id` to that replayed ordered manifest.
Changing a feature ID and recomputing the row, chunk and manifest hashes does
not bypass the unchanged validated feature-manifest preimage.

The projector claim boundary is separately and exactly:

```text
STATISTICAL_RESULTS_TARGET_FAMILY_AND_EXACT_FEATURE_MANIFEST_PREIMAGE_SUBPROJECTION_ONLY_NOT_WORKER_RETENTION_NOT_RECEIPT_CORE_NOT_RUNTIME_OR_RESOURCE_OR_QUOTA_AUTHORITY_NOT_R03_R04_R05
```

Its `projection_scope` is exactly
`STATISTICAL_RESULTS_TARGET_FAMILY_AND_EXACT_FEATURE_MANIFEST_PREIMAGE_ONLY_NOT_FULL_RECEIPT`,
and its bootstrap field is exactly
`IMPORT_THIS_MODULE_ONLY_FROM_A_SOURCE_PINNED_FRESH_PROCESS_BOOTSTRAP_PREIMPORT_MONKEYPATCHING_IS_OUTSIDE_THIS_IN_PROCESS_CHECK`.
Projection is not worker retention, a complete audit receipt, CAS publication,
or any authority receipt. Projected chunks must still cross the future fixed
descriptor/store boundary and receive signed durable CAS receipts.

The current `statistical_results_manifest` is not the future authority graph's
signed `full_results_index`. The latter must additionally join both modes,
twelve original-worker store receipts, store-observed chunk receipts, complete
audit-receipt core material, the exact feature-manifest chunks, exact
descriptor-transfer evidence,
profile/registry/attempt context, status-ledger evidence, and logical time. The
names may not be treated as aliases or promoted by renaming.

#### 9.2.2 Offline static image/runtime-closure mechanics v1

The offline runtime-closure source SHA-256 is
`f836b3d00f332398214d3ccf7787109abb8f2de96b7b8281748bbb0ec695f2f9`;
its focused test source SHA-256 is
`356953f7ee80d4f87815283a5dbbf996173ea65ce546d800278fa1659e19c6c8`.
At this offline-mechanics candidate BYTE-STOP, the focused suite reported
78/78. This is a construction-test result over caller-presented bytes, not
static admission,
image existence, execution, live closure, authority, R01, or readiness. The
three exact schemas are:

```text
runtime_closure_policy:
  hegel-machine-phase2b-formal-wire-covert-step5-runtime-closure-policy/1
image_closure_evidence:
  hegel-machine-phase2b-formal-wire-covert-step5-image-closure-evidence/1
static_runtime_manifest:
  hegel-machine-phase2b-formal-wire-covert-step5-static-runtime-manifest/1
```

`verify_offline_step5_image_closure_mechanics_v1` accepts only the raw
Stage-B lock, authority-profile/2 and signer-registry/2 payloads; the
independently supplied lock ID as exact ASCII bytes and lock-payload SHA-256
as an exact raw 32-byte digest; one exact service name from the frozen
five-service order; canonical policy and evidence bytes; raw OCI index,
manifest and config bytes; an exact tuple of exact byte-chunk tuples for every
ordered OCI layer; and another exact tuple of exact byte-chunk tuples for the
five ordered support artifacts. It accepts no decoded-object, digest, reader,
signature or live-observer callback.

The policy is intentionally pre-profile. Its body cannot carry the later
profile, registry or service-binding content IDs, avoiding a policy-SHA to
binding-ID to profile-ID to policy-payload cycle. The selected authority-v2
static service binding instead pins the policy payload SHA-256; the separate
content-addressed evidence object carries the verified Stage-B lock, profile,
registry and selected binding lineage. The verifier requires the exact
eight-role/five-service authority-v2 graph and the service's exact role,
control domain, UID/GID, role key, purpose, OCI descriptor graph, ordered layer
digests and static closure-policy SHA.

The replay hashes and exact-compares the raw OCI index, manifest, config and
every ordered raw layer chunk against both the OCI descriptors and the frozen
service binding. It separately consumes these five support artifacts in exact
order:

```text
source_archive
dependency_lock
sbom
provenance
python_executable
```

The exact runtime identity is CPython `3.10.12`, language version `3.10`, ABI
tag `cp310`, and cache tag `cpython-310`. The evidence exact-binds four ordered,
duplicate-free static manifests with kinds `PYTHON_STDLIB`, `IMPORT_CLOSURE`,
`NATIVE_LOADER`, and `SHARED_LIBRARIES`, along with their entry counts,
paths/module names, content digests, sizes and the frozen resource bounds.
These are static inventories of presented bytes; their `live_observation`
value is exactly false.

The result `VerifiedOfflineImageClosureMechanicsV1` is constructor-disabled
and closure-privately issued. Its validator rechecks the exact issued identity,
field identities and claim booleans, rejecting `object.__new__`-shaped values
and post-issuance field mutation. The authority loaders, canonical decoder,
policy/evidence validators, OCI/support replayers, hashing path and private
issuer are captured before the public entry is published, so ordinary later
module-global rebinding does not redirect verification. This is same-process
mutation-isolation KAT coverage only, not resistance to closure-cell,
interpreter-memory or loaded-code mutation; a pinned live runtime closure is
still required for that boundary.

Successful offline replay fixes only:

```yaml
offline_static_byte_graph_mechanics_verified: true
mechanics_only: true
```

It simultaneously and immutably fixes every live or authority predicate
below to false:

```yaml
service_image_exists: false
service_image_pulled: false
service_image_executed: false
live_runtime_closure_observed: false
entry_exit_closure_equal: false
live_uid_gid_verified: false
live_control_domain_verified: false
key_custody_verified: false
runtime_closure_signature_verified: false
remote_attestation_verified: false
external_cas_verified: false
authority_evidence: false
readiness_evidence: false
r01_satisfied: false
execution_authorized: false
```

The module starts no Docker container, systemd unit, service or process,
inspects no `/proc` state, issues no signature, and observes no entry/exit
closure. Thus it closes an offline static-byte replay mechanics gap only; all
live predicates and every readiness row remain false, and the gate remains
`HOLD`.

### 9.3 Exact current CAS-store mechanics surfaces

#### 9.3.1 Retention-only CAS-store v1

The CAS-store source SHA-256 is
`d22f003771cb7fa378299b53382e556a4d7b723a678659962db1577b2517f5f0`;
its focused test source SHA-256 is
`1dcdc222cb571a3a9a78a494c600913ad101ae7cc8a405ba61ca59496b54b17b`.
The current module freezes these canonical schema identifiers:

```text
store:
  hegel-machine-phase2b-formal-wire-covert-step5-store/1
request:
  hegel-machine-phase2b-formal-wire-covert-step5-scm-request/1
receipt_body:
  hegel-machine-phase2b-formal-wire-covert-step5-cas-store-receipt-body/1
signed_receipt:
  hegel-machine-phase2b-formal-wire-covert-step5-signed-cas-store-receipt/1
checkpoint:
  hegel-machine-phase2b-formal-wire-covert-step5-cas-store-checkpoint/1
head:
  hegel-machine-phase2b-formal-wire-covert-step5-cas-store-head/1
response:
  hegel-machine-phase2b-formal-wire-covert-step5-cas-store-response/1
acknowledgement:
  hegel-machine-phase2b-formal-wire-covert-step5-retention-ack/1
```

The server-side request contract is a literal local copy of request/1. Its
top-level import closure does not import the Auditor, QV2, or runtime module;
producer-side conversion to `RetentionIngestAckV1` is lazy and is not used by
server request handling. The transport requires an exact AF_UNIX
`SOCK_SEQPACKET` socket, one and only one `SCM_RIGHTS` descriptor,
`SO_PEERCRED`, a configured allowed peer UID, canonical request bytes, and no
truncation or extra control record.

The public `CasStoreV1` bootstrap accepts the trust graph only through these
five exact constructor inputs:

```text
stage_b_lock_payload: bytes
authority_profile_payload: bytes
signer_registry_payload: bytes
out_of_band_expected_lock_id: str
out_of_band_expected_lock_payload_sha256: str
```

It accepts no caller-supplied verified-artifact DTO. Internally, the store
first invokes the public bytes-only Stage-B lock loader with both independently
provisioned out-of-band pins, then invokes the public bytes-only profile loader
under that lock, and finally invokes the public bytes-only governance-signed
registry loader under that lock and profile. Only the locally produced
verified results reach the private role-binding helper. A caller-created
`VerifiedArtifact`, including one laundered through a private construction
helper, cannot replace those bytes or bypass either out-of-band pin through
this public constructor.

The resulting graph must have the exact eight-role policy and the exact
`CAS_STORE` registry row at its frozen role position. That row is
`hegel-machine-phase2b-step5-role-key/1`, role `CAS_STORE`, control domain
`STEP5_CAS_STORE_CONTROL`, algorithm `Ed25519`, key epoch 1, and authorized
signature-purpose list exactly `[SIGN_CAS_IF_ABSENT_RECEIPT]`; its role-key ID
is recomputed from that entire policy row, and the signing private seed must
derive the registry public key.

One `CasStoreV1` instance freezes one chain, authority profile, registry,
attempt context and retention session. New requests must arrive in exact
ordinal/checkpoint order 0 through 11 and exact mode/phase/target/basename
order; `checkpoint_index==ordinal`. Completion requires exactly twelve
records. A duplicate request ID is idempotent only when the complete request,
peer credentials, received FD identity/hash/size, stored object, signed
receipt and checkpoint exact-replay the existing record. It returns the
existing response without appending another checkpoint; any near replay or
context swap fails closed.

The store independently hashes the read-only regular FD, performs local
create-if-absent publication, fsync/readback checks, signs the purpose-bound
receipt, appends the contiguous checkpoint, and advances the local head. Its
exact claim boundary is:

```text
CAS_STORE_MECHANICS_ONLY_SAME_PROCESS_OR_SAME_UID_OR_LOCAL_FILESYSTEM_IS_NOT_AUTHORITY_ACTUAL_AUTHORITY_REQUIRES_INDEPENDENT_UID_CONTROL_DOMAIN_KEY_CUSTODY_DURABILITY_AND_LIVE_RUNTIME_CLOSURE
```

The source provides both a store primitive and a verifying
`ScmRightsRetentionSinkV1`, but starts no listener and makes no deployment
claim. The runtime API still accepts an arbitrary caller-supplied sink. Same-
process, same-UID, local-filesystem, fixture-key, source-pin without a live
runtime closure, or ordinary local `fsync` tests establish mechanics only. A
locally consistent rollback or deletion can be excluded only by an external
durable anchor, which this module does not supply. Actual Step 5 still requires
a fixed composed client/server path, separately controlled UID/control domain,
protected key custody, pinned images and live closures, independently durable
storage with an external checkpoint anchor, and authority admission of the
resulting receipts.

#### 9.3.2 Generic planned CAS-store v2

The authority-v2-lineage generic CAS-store source SHA-256 is
`6c545144dd584244d9405c92120e09c8ecb1d115c90e964a75fb3239b0b670b1`;
its focused test source SHA-256 is
`81e2056837750d37736572b990d07be95875523285c016ea5e8456f5287586ab`.
At this replacement mechanics candidate BYTE-STOP, the focused suite reported
54/54. This is a construction-test result, not static admission, service
liveness, authority evidence, or readiness. The exact schema family is:

```text
store:
  hegel-machine-phase2b-formal-wire-covert-step5-store/2
retention_plan:
  hegel-machine-phase2b-formal-wire-covert-step5-retention-plan/1
generic_request:
  hegel-machine-phase2b-formal-wire-covert-step5-generic-scm-request/1
transition_binding:
  hegel-machine-phase2b-formal-wire-covert-step5-cas-transition-binding/1
receipt_body:
  hegel-machine-phase2b-formal-wire-covert-step5-cas-store-receipt-body/2
signed_receipt:
  hegel-machine-phase2b-formal-wire-covert-step5-signed-cas-store-receipt/2
checkpoint:
  hegel-machine-phase2b-formal-wire-covert-step5-cas-store-checkpoint/2
head:
  hegel-machine-phase2b-formal-wire-covert-step5-cas-store-head/2
response:
  hegel-machine-phase2b-formal-wire-covert-step5-cas-store-response/2
```

The dependency replacement accepts only the raw authority/profile v2
Stage-B lock, authority-profile/2 and signer-registry/2 graph. A real valid
authority-v1 triple, any V1/V2 hybrid, cross-graph lock/profile/registry mix,
or exchanged CAS_STORE key/purpose fails closed. The accepted v2 graph must
carry the exact eight-role registry and five typed static-service bindings,
and the `cas_store` binding must close onto the exact `CAS_STORE` role-key/1,
`STEP5_CAS_STORE_CONTROL`, key epoch 1 and
`[SIGN_CAS_IF_ABSENT_RECEIPT]` purpose. These are static graph checks only;
they do not establish that the bound service image or distinct UID is live.

Store/2 retains its existing wire field shapes because this is a pre-freeze
authority-dependency replacement. That schema-number retention is not a V1
compatibility promise: the old authority-v1 graph is affirmatively rejected.

Unlike v1's fixed twelve-object qualification order, retention-plan/1 freezes
one generic, content-addressed, exact-order object plan with an exact positive
object count and `complete_head_required=true`. Each plan entry fixes its
ordinal, artifact kind and maximum byte count. Each generic request and every
transition then bind the plan, attempt, retention session, chain, ordinal,
artifact ID/SHA-256, size, predecessor and exact observed/expected counts.
Only the last planned transition may carry `complete=true`; gaps, forks,
post-completion appends, stale current heads, partial appends and inconsistent
duplicate requests fail closed. Equal object bytes may be deduplicated across
distinct declared ordinals, but the two semantic plan entries remain distinct.

The v2 server mechanics require one read-only regular single-link FD over an
exact AF_UNIX `SOCK_SEQPACKET`/`SCM_RIGHTS` transfer, exactly one descriptor,
kernel `SO_PEERCRED`, and an allowed peer UID. The store independently hashes
the received object, enforces create-if-absent plus readback/fsync checks,
signs the purpose-bound Ed25519 receipt, and publishes immutable checkpoint,
head and response records. The module constructs no listener and starts no
service. Its raw authority bootstrap still consumes exact Stage-B lock,
profile and registry bytes. The independent pins are the lock ID as exact
ASCII `bytes` and the lock-payload SHA-256 as an exact raw 32-byte digest;
strings, bytearrays, hex-text digests, self-copied or wrong pins fail closed.
The authority-v2 loaders and CAS_STORE role-key derivation are captured when
the critical store verifier/constructor paths are defined, so later ordinary
module-global rebinding cannot redirect them. Ed25519 receipt verification
separately captures, at verifier definition time, the concrete
`Ed25519PublicKey.from_public_bytes` loader, the concrete invalid-signature
exception type, the receipt-signature preimage helper and the fail path; the
builder is then deleted. Consequently even a fully readdressed invalid receipt
signature remains rejected after ordinary rebinding of the public backend,
exception and preimage names. The resulting local cryptographic binding is
still not deployment or authority.

`verify_cas_store_complete_session_v2` is a producer-free, bytes-only replay
surface. It accepts the raw retention-plan bytes and six exact-order tuples:
requests, object payloads, independently supplied signed receipts,
checkpoints, heads and responses. Every tuple cardinality must equal the plan
count and every member must be exact `bytes`. For every ordinal, the verifier
independently canonical-parses and exact-compares the request and signed
receipt with the embedded checkpoint/response graph, verifies the registry
Ed25519 signature, rehashes the presented object, replays the predecessor
chain, and requires one exact terminal complete head. Supplying only a
self-consistent embedded graph cannot substitute for any raw sequence.

Direct DTO integer fields require `type(value) is int`, excluding `bool` and
integral `float`. Canonical transport restores decoder-produced integral JSON
numbers to integers before exact RFC 8785 re-encoding; non-integral or
out-of-bound numeric values and type-confused direct DTOs fail closed. The
complete-session result has strong closure-private lineage: the unexported
issuance token and issuer, a locked weak-reference identity registry and an
exact immutable-field snapshot are retained inside the construction closure.
The require function accepts only the exact originally issued object identity
with every frozen field unchanged; direct construction, an `object.__new__`
clone, shallow/deep copy and post-issuance mutation fail closed. Its replay
entry also captures every canonical byte loader, authority loader, receipt and
record verifier, and hash/fail helper rather than resolving caller-rebindable
module globals. This is same-process mutation-isolation KAT coverage only; it
does not resist arbitrary closure-cell, process-memory or loaded-code mutation
and does not replace a pinned runtime closure.

The exact v2 claim boundary and rollback disclosure are:

```text
CAS_STORE_V2_MECHANICS_ONLY_SAME_PROCESS_SAME_UID_LOCAL_FILESYSTEM_DECLARATIVE_SOURCE_PINS_AND_STATIC_AUTHORITY_V2_SERVICE_BINDINGS_NOT_LIVE_AND_PRESENTED_COMPLETE_REPLAY_ARE_NOT_AUTHORITY_NOT_IMAGE_OR_RUNTIME_CLOSURE_NOT_KEY_CUSTODY_NOT_ANTI_ROLLBACK_OR_ANTI_CLONE
GAPS_FORKS_STALE_CURRENT_HEAD_AND_PARTIAL_APPENDS_FAIL_CLOSED_BUT_A_CONSISTENT_WHOLE_TREE_ROLLBACK_OR_CLONE_REQUIRES_AN_EXTERNAL_ANCHOR_TO_DETECT
```

Accordingly, a complete v2 replay leaves listener/systemd/live deployment,
distinct UID and control-domain enforcement, key custody, image and entry/exit
runtime closure, fresh authorized attempt context, independently durable
storage, external complete-head anti-rollback/anti-clone anchoring, and final
authority admission unproved. It sets no readiness row true and does not issue
R01 or any readiness overlay.

#### 9.3.3 Verifier-only store/2 replay core

The verifier-only store replay-core source SHA-256 is
`e33bb8ae1f4c70668de3bfbdbd803a953513db94cb0a9c1ceb1082e97141de02`;
its focused test source SHA-256 is
`581cdcf3934c703419a30dc3c87b809fb29fd8f7710653e37aa52e874b2c3845`.
At this replay-core candidate BYTE-STOP, the focused suite reported 52/52. Its
module schema is
`hegel-machine-phase2b-formal-wire-covert-step5-store-replay-core/2`, and it
pins the inspected store/2 source SHA-256 to the current
`6c545144dd584244d9405c92120e09c8ecb1d115c90e964a75fb3239b0b670b1`
BYTE-STOP. This result is verifier construction evidence only.

This core deliberately does not import the CAS-store producer/service module,
the retention producer or any live-service adapter. Its only project
dependencies are the authority/profile-v2 verifier and the RFC 8785
implementation; it uses the concrete Ed25519 backend directly. It has no
filesystem, file descriptor, socket, `SCM_RIGHTS`, signer, private key, clock,
path, callback or decoded-object public entry. Therefore it can verify only
the raw bytes presented to this pure process, never assert that the store
produced, retained or served them.

`verify_cas_store_complete_session_replay_core_v2` consumes the exact raw
Stage-B lock/profile/registry graph and independently provisioned raw OOB
pins, one raw retention-plan payload, and six equal-cardinality exact tuples
of exact bytes in this semantic order:

```text
request_payloads_in_exact_order
object_payloads_in_exact_order
signed_receipt_payloads_in_exact_order
checkpoint_payloads_in_exact_order
head_payloads_in_exact_order
response_payloads_in_exact_order
```

The core independently re-freezes the store/2 plan/request/transition/
receipt/checkpoint/head/response wire schemas. For every ordinal it strict-
parses and restores only exact integral JSON values, recomputes canonical
bytes and content IDs, verifies the authority-v2-only CAS_STORE role/service/
purpose graph and Ed25519 receipt signature, exact-matches every separately
supplied request, receipt, checkpoint, head and response to the embedded
signed graph, rehashes the separately supplied object bytes, and replays the
contiguous predecessor chain. It also recomputes the expected
`CREATED_IF_ABSENT` or `PRESENT_VERIFIED` disposition from the first-seen
content order and requires exactly one terminal complete head. An embedded
self-consistent graph cannot stand in for an omitted raw sequence.

The complete call graph, authority loaders, RFC 8785 functions, concrete
Ed25519 loader/exception, exact-type primitives, hashing path and private
result issuer are captured at import. The result type is constructor-disabled,
cannot be copied or deep-copied, and is accepted only through a locked weak-
reference registry of exact issued identities plus an exact immutable-field
snapshot; `object.__new__` forgeries and later mutation fail closed. This is
ordinary Python/KAT mutation isolation, not protection against arbitrary
interpreter-memory, closure-cell or loaded-code mutation.

Successful replay sets canonical-byte, content-address, Ed25519-signature,
append-chain and IF_ABSENT/idempotent-complete mechanics true. It fixes all of
the following to false: durability observed, live CAS service, service image,
runtime closure, live UID/GID and control-domain separation, private-key
custody, external anti-rollback, authority evidence, readiness evidence and
execution authorization. Signed `durable_before_return`, readback and peer-UID
fields remain replayed signed claims, not live observations.

Its exact claim and rollback boundaries are:

```text
PRESENTED_RAW_COMPLETE_SESSION_CANONICAL_CONTENT_SIGNATURE_CHAIN_AND_IF_ABSENT_IDEMPOTENCY_MECHANICS_ONLY_NOT_DURABILITY_NOT_LIVE_SERVICE_NOT_UID_SEPARATION_NOT_KEY_CUSTODY_NOT_EXTERNAL_ANTI_ROLLBACK_NOT_AUTHORITY_NOT_READINESS
GAPS_FORKS_STALE_CURRENT_HEAD_AND_PARTIAL_APPENDS_FAIL_CLOSED_BUT_A_CONSISTENT_WHOLE_TREE_ROLLBACK_OR_CLONE_REQUIRES_AN_EXTERNAL_ANCHOR_TO_DETECT
```

This verifier-only replay core does not compose a live service or promote the
store/2 service result. Listener/systemd/live state, current external durable
head, UID/control domain, key custody and final authority admission remain
false/unproved; every readiness row stays false and the gate remains `HOLD`.

### 9.4 Exact current logical-time mechanics surfaces

#### 9.4.1 Logical-time service/local-ledger mechanics

The authority-v2-lineage logical-time source SHA-256 is
`3b576720c0ec9f75727f1cd5a5ebba1d02c565fd70cb6fcd8917f644667be65f`;
its focused test source SHA-256 is
`7f71ad34f7c850e08aabc297cc9445cfc20155708a94697d0046555ff633a397`.
At this replacement mechanics candidate BYTE-STOP, the focused suite reported
154/154. This is a construction-test result, not static admission, live
evidence, authority, or readiness. The exact byte-schema family is:

```text
request:
  hegel-machine-phase2b-formal-wire-covert-step5-logical-time-request/1
signed_token:
  hegel-machine-phase2b-formal-wire-covert-step5-signed-logical-time-token/1
ledger_head:
  hegel-machine-phase2b-formal-wire-covert-step5-logical-time-ledger-head/1
ledger_checkpoint:
  hegel-machine-phase2b-formal-wire-covert-step5-logical-time-ledger-checkpoint/1
```

The dependency contract now accepts only
`phase2b_step5_authority_profile_v2_*` and
`phase2b_step5_signer_registry_v2_*` IDs. Authority-v1 IDs and every V1/V2
hybrid fail at the exact common-binding boundary. A complete chain cannot be
replayed under foreign expected V2 profile/registry IDs that differ from the
IDs frozen into its genesis, checkpoints and heads. This module still accepts
IDs rather than raw profile/registry graph bytes, so the independent bytes-only
authority-v2 graph and out-of-band pins remain external mandatory inputs to
authority admission; V2 prefix syntax alone is not graph verification.

The four logical-time schemas deliberately remain `/1` because this pre-freeze
dependency replacement changes neither their exact field sets nor their value
domains, role-key/1 projection or `SIGN_LOGICAL_TIME_TOKEN` purpose. The
module's `DEPENDENCY_REPLACEMENT_V1` explicitly records that this is an
uncommitted/unfrozen candidate replacement and makes no V1 or hybrid
compatibility promise. The BYTE-STOP above identifies only the inspected
replacement mechanics bytes; it does not promote them into authority.

The `LOGICAL_TIME` role-key content ID binds role `LOGICAL_TIME`, control domain
`STEP5_LOGICAL_TIME_CONTROL`, algorithm `Ed25519`, key epoch 1, public key, and
authorized signature-purpose list exactly `[SIGN_LOGICAL_TIME_TOKEN]`.
Request/1 binds the out-of-band profile and registry IDs, role-key ID/epoch,
attempt context, nonce, signature purpose, event purpose, bound artifact
ID/SHA-256, prior token ID and exact next sequence. Token/1 exact-copies those
bindings, carries the sequence, Ed25519 signature and content-derived token
ID. The verifier accepts no callback crypto backend.

The bytes-only transition verifier binds the same profile, registry, role key,
epoch and attempt through previous head, signed token, checkpoint and next
head. Genesis starts at sequence/count zero. Each checkpoint index and token
sequence must equal the prior head's exact next sequence; prior-token,
previous-head and previous-checkpoint links must match, and the next immutable
head advances by exactly one. Gaps, forks, replay, cross-attempt exchange and
alternate heads fail closed. The local ledger uses immutable versioned heads
and checkpoints, exclusive locking, `O_EXCL`, fsync, canonical reparse and
full-chain audit; those remain local mechanics.

The public `verify_logical_time_ledger_replay_v1` replays one complete
caller-supplied canonical chain. It accepts an exact raw genesis-head `bytes`,
exact tuples of raw checkpoint and subsequent-head `bytes`, the exact public
key and expected profile/registry/key/epoch/attempt pins, and an exact tuple of
exact-type `ExpectedLogicalEventV1` rows. For each ordinal it reparses the
checkpoint-embedded signed token and both heads, verifies Ed25519 and the full
transition, reconstructs and binds the content-addressed request, requires
contiguous sequence/index values, exact-compares all four event-plan fields,
and rejects duplicate request IDs/nonces, token IDs, checkpoint IDs, head IDs,
or event bindings. It accepts no decoded artifact or verifier callback.

The zero-event case is valid only when all three supplied tuples are exact
empty tuples and the raw initial head is the exact configured genesis for the
pinned ledger. Its result has zero verified events/signatures, only the
genesis head ID, and `zero_event_genesis_only=true`; a later head cannot be
laundered as an empty chain.

The replay implementation and its critical parser, signature, request-
projection and transition call graph are captured before the public entry is
published. Ordinary rebinding of corresponding module globals or parser
classmethods therefore neither diverts valid replay nor makes an invalid
signature acceptable. This is a mutation-isolation KAT property only: mutation
of function defaults, closure cells, arbitrary process memory or loaded code
remains outside the claim and must be controlled by the external pinned image
and runtime closure.

`VerifiedLogicalTimeChainV1` is constructor-disabled and issued only through a
closure-private lineage that is not exposed as a module-global issuance token
or issuer helper. `require_verified_logical_time_chain_v1` rechecks the exact
type, hidden lineage, ledger identity, event/signature cardinalities, unique
ID tuples, exact final-head linkage and every mechanics/authority boolean. A
same-shaped value built with `object.__new__` is not a verified result.
Successful issuance fixes `mechanics_only=true` and all of
`authority_evidence`, `external_anti_fork_verified`,
`event_authorization_verified`, `attempt_context_freshness_verified`,
`service_image_verified`, `service_uid_verified`,
`control_domain_verified`, `key_custody_verified`,
`external_cas_checkpoint_verified` and `trusted_utc_verified` to false.
Two self-consistent local forks can therefore each pass mechanics replay while
neither establishes the externally current branch.

Its exact claim boundary is:

```text
LOGICAL_TIME_CANONICAL_SIGNATURE_AND_LOCAL_APPEND_ONLY_MECHANICS_ONLY_NOT_AUTHORITY_NOT_TRUSTED_UTC_NOT_EXTERNAL_NOTARIZATION_NOT_RUNTIME_ATTESTATION_NOT_EVENT_ORDER_POLICY_NOT_TERMINAL_CLOSURE_NOT_CAS_CHECKPOINT
```

Before any token can contribute to authority, an independent verifier must
still establish the out-of-band authority-profile pin; signer-registry pin and
LOGICAL_TIME role policy; a fresh Step-5 attempt context; pinned service image
and runtime closure; the expected distinct service UID and technical control
domain; private-key custody; the frozen event-order and terminal policy; and an
external durable CAS checkpoint of the complete ledger. The module starts no
service or socket, does not provide trusted UTC, and does not itself enforce
event admissibility, freshness, terminal closure, anti-fork externality, image
identity, UID separation or key custody. Those unresolved live predicates keep
the gate at `HOLD`, with every readiness row false.

#### 9.4.2 Verifier-only logical-time replay core v1

The verifier-only logical-time replay-core source SHA-256 is
`a618291256723e2ce5ac8d77d4b77a3deec4bbf787ab3422b0b61b9c36358def`;
its focused test source SHA-256 is
`eef1e7922cc96a9304ae93b6829ae9bad76dc1f6b8b34ba396d5f75340f2798a`.
At this replay-core candidate BYTE-STOP, the focused suite reported 29/29. Its
module schema is
`hegel-machine-phase2b-formal-wire-covert-step5-logical-time-replay-core/1`.
This is a pure verifier construction result, not a service observation,
external currentness, authority or readiness.

The replay core deliberately does not import the logical-time producer/service
module, CAS store, owner-status actor or any live-service code. Its only project
dependencies are the authority/profile-v2 verifier and RFC 8785; concrete
Ed25519 verification comes directly from `cryptography`. It independently
re-freezes the exact logical-time request/token/checkpoint/head `/1` wire
schemas rather than calling the service's parser or replay routine. It has no
private key, issuer path, wall clock, trusted UTC source, filesystem, CAS
client, anti-fork anchor, runtime attestation or readiness issuer. It verifies
only the raw bytes supplied to this core.

`verify_logical_time_replay_core_v1` consumes raw Stage-B lock, profile and
registry bytes; exact raw lock-ID and 32-byte digest OOB pins; exact tuples of
raw requests, signed tokens, checkpoints and heads; one exact expected attempt
context; and an exact tuple of exact `ExpectedLogicalEventCoreV1` rows. The
request, token and checkpoint tuple counts must equal the event-plan count,
while the head tuple must contain the genesis plus exactly one next head per
event. The raw authority graph must expose exactly the authority-v2
`LOGICAL_TIME` role key, singleton `[SIGN_LOGICAL_TIME_TOKEN]` purpose and
matching typed `logical_time` static service binding.

For each event the core strict-parses and exact-recanonicalizes every raw
artifact, reconstructs the content-derived request/token/checkpoint/head IDs,
verifies the Ed25519 token signature, exact-projects the independently
presented request from the token, exact-matches the separately presented token
to the checkpoint-embedded token bytes, and exact-compares purpose, artifact
ID/SHA-256 and nonce with the caller-supplied event-plan row. Sequence,
prior-token, previous-head and previous-checkpoint links must advance
contiguously, every identity/nonce is unique, and the final supplied head must
exactly commit the final supplied checkpoint.

For zero events, the request/token/checkpoint/event tuples must all be exact
empty tuples and the sole raw head must be the exact content-derived genesis
for the pinned authority/attempt/ledger lineage. The result then has zero
requests, signatures, checkpoints and events, one verified genesis head, and
`zero_event_genesis_only=true`; a later head cannot be presented as an empty
chain. For nonzero input, success establishes supplied-chain final-head
consistency only. It does not establish that the supplied head is the current
or externally unique head, and the public `verify_external_currentness_v1`
entry fails closed unconditionally.

The verifier's parser, signature, projection, transition, canonicalization,
hashing, authority-extraction and result-issuance graph is captured before the
public entry is published. Its constructor-disabled result retains the exact
original issued identity and exact immutable scalar/tuple snapshots, rejecting
`object.__new__` forgeries and post-issuance mutation. This is Python caller/
KAT mutation isolation only, not arbitrary closure-cell, process-memory or
loaded-code protection.

Success fixes `supplied_chain_final_head_verified=true`, `verifier_only=true`,
and the canonical/signature/count identities represented by the private
result. It simultaneously fixes private-key presence, issuer-path presence,
filesystem observation, service observation, external currentness, external
anti-fork, trusted UTC, external CAS checkpoint, runtime closure, authority
evidence, readiness evidence, R01 and execution authorization to false. Its
exact claim boundary is:

```text
VERIFIER_ONLY_CANONICAL_SIGNATURE_AND_SUPPLIED_CHAIN_MECHANICS_NOT_EXTERNAL_CURRENTNESS_NOT_TRUSTED_UTC_NOT_CAS_NOT_RUNTIME_ATTESTATION_NOT_AUTHORITY_NOT_READINESS_NOT_EXECUTION_AUTHORIZATION
```

Thus the service/local-ledger replay and this independently re-frozen pure core
may cross-check the same presented chain without giving the core a producer or
service import. Agreement remains mechanics evidence over those presented raw
bytes; it does not supply UTC, current-cut/anti-fork externality, live image or
service closure, CAS durability, authority or a readiness transition. Every
readiness row remains false and the gate remains `HOLD`.

### 9.5 Exact current owner-status mechanics surfaces

#### 9.5.1 Owner-status schema/1 v2-lineage service/local-ledger mechanics

The owner-status source SHA-256 is
`477178052e3db8e7773e3832334c3a2faf2b2d04f2e93255ad7e3d59556206a2`;
its focused test source SHA-256 is
`6732c18413a16601f353984360b241f0c608a0db2b506706bbf3cc1a56571645`.
At this replacement mechanics candidate BYTE-STOP, the focused owner-status
suite reported 63/63. This is a construction-test result, not static admission,
live evidence, authority, R01, or readiness. This source freezes only a
governance-signed, local append-only mechanics-candidate ledger. Its current
schemas are:

```text
module_contract:
  hegel-machine-phase2b-formal-wire-covert-step5-owner-status/1
cas_dependency_reference:
  hegel-machine-phase2b-formal-wire-covert-step5-owner-status-cas-dependency-reference/1
logical_time_dependency_reference:
  hegel-machine-phase2b-formal-wire-covert-step5-owner-status-logical-time-dependency-reference/1
decision:
  hegel-machine-phase2b-formal-wire-covert-step5-owner-status-decision/1
signed_event:
  hegel-machine-phase2b-formal-wire-covert-step5-signed-owner-status-event/1
checkpoint:
  hegel-machine-phase2b-formal-wire-covert-step5-owner-status-checkpoint/1
head:
  hegel-machine-phase2b-formal-wire-covert-step5-owner-status-head/1
projection:
  hegel-machine-phase2b-formal-wire-covert-step5-owner-status-projection/1
complete_event_set_root_preimage:
  hegel-machine-phase2b-step5-owner-status-complete-event-set-root/1
```

Owner-status retains schema `/1` because this pre-freeze dependency replacement
does not change its exact field shapes. It now accepts only
authority-profile/2 and signer-registry/2 identifier lineages; authority-v1
IDs, V1/V2 hybrids, wrong expected V2 pairs and cross-graph/attempt exchanges
fail closed. This is not a V1 compatibility promise and not a raw authority-
graph verification claim.

The module intentionally imports neither authority v2 nor the CAS or logical-
time actor. It byte-matches the exact authority-v2 governance role-key/1
projection—role `GOVERNANCE_ANCHOR`, control domain
`STEP5_GOVERNANCE_CONTROL`, key epoch 1, Ed25519 public key and ordered purposes
`[SIGN_SIGNER_REGISTRY, SIGN_OWNER_AUTHORIZATION_STATUS_EVENT]`—and its KAT
derivation equals `authority_v2.phase2b_step5_role_key_id_v2` byte for byte. It
also requires the logical-time dependency and signed token to carry the same
profile-v2/registry-v2/attempt lineage and the exact logical-time role-key/
purpose projection. However, it does not bytes-load the Stage-B lock, profile
or registry, verify that closed raw graph, replay the complete logical-time
ledger, or verify a complete CAS-store v2 session.

Every mapping below is an exact current keyset:

```text
cas_dependency_reference = {
  schema_version, verification_status, authority_profile_id,
  signer_registry_id, attempt_context_id, retention_session_id, chain_id,
  signer_role, signer_key_id, signer_key_epoch, signature_purpose,
  signature_domain, bound_artifact_id, bound_artifact_sha256, object_id,
  object_canonical_sha256, signed_receipt_schema_version,
  signed_receipt_id, signed_receipt_canonical_sha256,
  checkpoint_schema_version, checkpoint_id, checkpoint_canonical_sha256,
  checkpoint_index, head_schema_version, head_id, head_canonical_sha256,
  head_last_checkpoint_id, head_checkpoint_count
}
logical_time_dependency_reference = {
  schema_version, verification_status, authority_profile_id,
  signer_registry_id, attempt_context_id, signer_role, signer_key_id,
  signer_key_epoch, signature_purpose, token_schema_version, token_id,
  token_canonical_sha256, token_bound_artifact_id,
  token_bound_artifact_sha256, token_sequence
}
decision = {
  schema_version, ledger_id, authority_profile_id, signer_registry_id,
  governance_key_id, governance_key_epoch, attempt_context_id,
  owner_authorization_binding, owner_authorization_payload_sha256, sequence,
  event_type, prior_effective_status, effective_status, revocation_count,
  reason_code, decision_nonce, previous_event_id, previous_head_id,
  previous_checkpoint_id, previous_complete_event_set_root, decision_id
}
signed_event = {
  schema_version, signature_purpose, signature_domain, authority_profile_id,
  signer_registry_id, governance_key_id, governance_key_epoch,
  governance_public_key_hex, attempt_context_id, decision,
  decision_canonical_sha256, cas_store_dependency, logical_time_dependency,
  algorithm, signature_hex, event_id, claim_boundary
}
checkpoint = {
  schema_version, purpose, ledger_id, authority_profile_id,
  signer_registry_id, governance_key_id, governance_key_epoch,
  attempt_context_id, checkpoint_index, previous_head_id,
  previous_checkpoint_id, previous_event_id,
  previous_complete_event_set_root, event_id,
  signed_event_canonical_sha256, signed_event, complete_event_set_root,
  checkpoint_id
}
head = {
  schema_version, purpose, ledger_id, authority_profile_id,
  signer_registry_id, governance_key_id, governance_key_epoch,
  attempt_context_id, event_count, next_sequence, last_event_id,
  last_checkpoint_id, complete_event_set_root, latest_effective_status,
  revocation_count, terminal, head_id
}
projection = {
  schema_version, claim_level, ledger_id, authority_profile_id,
  signer_registry_id, governance_key_id, governance_key_epoch,
  attempt_context_id, owner_authorization_binding,
  owner_authorization_payload_sha256, event_count, latest_sequence,
  latest_effective_status, revocation_count, last_event_id,
  last_checkpoint_id, head_id, complete_event_set_root,
  governance_signatures_verified,
  governance_status_purpose_authorized_by_pinned_registry,
  owner_signature_authenticated, logical_time_token_signatures_verified,
  store_dependency_candidate_references_exact, store_dependencies_verified,
  latest_authority_status_verified, r01_satisfied,
  readiness_node_transition_issued, execution_authorized,
  authority_evidence, projection_id
}
```

The source admits the frozen owner authorization only as exact canonical
`bytes`, with wrapper keyset `{authorization_body, authorization_id}`, no
trailing line feed, and these frozen pins:

```yaml
authorization_schema: hegel-machine-phase2b-c1-owner-actual-unsealed-960-authorization/1
authorization_id: phase2b_c1_owner_actual_unsealed_960_authorization_v1_dd2cd6a98b429fe01f172daf16ea70404c91356e40e9e42daed4cb5416a5c50d
authorization_payload_sha256: 26ff84c3fe239158b1e8db24b5634ccfe8b3f3995d54bdc1f728c49a70f8fca1
authorization_source_sha256: 4c96d50706e520f1c67b2d2c31fce9ed605feacbf4c1177ddf9a7a2a9c1d9a73
```

Every decision and projection carries the full exact fourteen-field binding,
not only the owner ID or payload digest:

```text
owner_authorization_binding = {
  authorization_schema, authorization_id, authorization_config_sha256,
  authorization_source_sha256, protocol_id, exact_freeze_version,
  exact_freeze_id, authority_class, authority_requirement_id,
  actor_profile_content_id, actor_profile_canonical_sha256,
  logical_stage_wire_schema_id, logical_stage_wire_policy_id,
  stage_event_contract_manifest_id
}
```

The event signature uses real Ed25519, the exact purpose
`SIGN_OWNER_AUTHORIZATION_STATUS_EVENT`, and the exact domain
`HEGEL/PHASE2B/STEP5/SIGN/OWNER_AUTHORIZATION_STATUS_EVENT/GOVERNANCE/V1`.
Authority/profile v2 freezes the exact ordered governance list
`[SIGN_SIGNER_REGISTRY, SIGN_OWNER_AUTHORIZATION_STATUS_EVENT]`, along with the
role-key/profile/registry content-address policy, and owner status byte-matches
that role-key derivation. This is still not authorization by a pinned registry:
the owner-status module does not load or verify raw lock/profile/registry bytes.
Consequently its projection deliberately keeps
`governance_status_purpose_authorized_by_pinned_registry=false`. Authority v2
also does not issue keys, signatures, a concrete admitted graph, or new out-of-
band pins. The exact remaining cascade is:

```text
OWNER_STATUS_SCHEMA_V1_FIELD_SHAPE_UNCHANGED
AUTHORITY_DEPENDENCY_REPLACED_BY_PROFILE_V2_AND_SIGNER_REGISTRY_V2
GOVERNANCE_TWO_PURPOSE_ROLE_KEY_POLICY_STATICALLY_FROZEN_IN_AUTHORITY_V2
OWNER_STATUS_ROLE_KEY_DERIVATION_BYTE_MATCHES_AUTHORITY_V2
RAW_STAGE_B_LOCK_PROFILE_REGISTRY_GRAPH_STILL_MUST_BE_BYTES_VERIFIED
CAS_AND_LOGICAL_TIME_DEPENDENCIES_BIND_PROFILE_V2_REGISTRY_V2_ATTEMPT
PINNED_DISTINCT_SERVICE_IMAGES_AND_RUNTIME_CLOSURES_STILL_MUST_BE_VERIFIED
RUN_COMPLETE_BYTES_ONLY_PURE_REPLAY_BEFORE_ANY_REAL_R01_VERIFIER
```

The CAS reference is intentionally fixed at
`verification_status=STORE_DEPENDENCY_CANDIDATE_NOT_VERIFIED`. The module
exact-compares its profile/registry/attempt graph, decision ID/SHA, object
address, receipt/checkpoint/head references, and contiguous cross-event CAS
context, but it neither verifies the referenced CAS signature nor proves that
the referenced object is in a current durable store. This module's CAS `/1`
dependency surface corresponds to retention-only CAS v1, which is frozen to
exactly twelve qualification-worker objects and cannot ingest an owner-status
decision. Generic CAS-store v2 now freezes a separate generic plan/session and
bytes-only complete-replay mechanics contract, but the current owner-status
module neither submits to it nor verifies its receipts. The v2 store now
bootstraps through the authority/profile v2 graph, but that graph and receipt
chain are not yet consumed by owner status. No owner-status-to-generic-CAS
integration or authority admission is therefore claimed.

Each event also carries an exact logical-time dependency with
`verification_status=LOGICAL_TIME_TOKEN_SIGNATURE_VERIFIED_MECHANICS_ONLY_NOT_AUTHORITY`.
Its token signature is verified with real Ed25519 under signature purpose
`SIGN_LOGICAL_TIME_TOKEN` and event purpose
`OWNER_AUTHORIZATION_STATUS_DECISION`. The owner-status event set uses one
dedicated attempt/profile/registry/key stream. Initial `ACTIVE` is exactly
decision sequence 0, prior status `NONE`, revocation count 0, and a logical
token at sequence 0 following the logical-time genesis token. The only later
event is optional terminal `REVOKED`: decision sequence 1, prior status
`ACTIVE`, revocation count 1, and a logical token at sequence 1 linked to the
ACTIVE token. Restoration after `REVOKED` is forbidden. The verifier
reconstructs the request/1 ID from the exact token request subject, binds the
token to the exact decision ID/SHA, and rejects reused decision nonces, token
IDs, request nonces, CAS receipt/checkpoint IDs, wrong request IDs, gaps,
reordering, prior-token splices, or cross-attempt/session/key swaps.

The complete-event-set root uses these exact preimage keysets:

```text
genesis = {schema_version, events_in_sequence}
transition = {
  schema_version, previous_complete_event_set_root, sequence, event_id,
  event_canonical_sha256
}
```

Genesis fixes an empty event list. Each checkpoint and next immutable head
exact-bind the advanced `complete_event_set_root`. The local ledger holds and
rechecks directory identities, uses exclusive
locking and `O_EXCL` publication, fsyncs and reads back each versioned token,
checkpoint, and head, requires exact filename/cardinality sets, and fully
replays the chain after append. A crash gap remains visible and poisons later
admission. These are local crash-consistency gates only: a privileged actor can
clone the whole ledger or consistently delete the terminal suffix and obtain a
locally replayable older `ACTIVE` view. The exact overall claim boundary is:

```text
OWNER_STATUS_CANONICAL_GOVERNANCE_SIGNATURE_AND_LOCAL_APPEND_ONLY_MECHANICS_ONLY_AUTHORITY_V2_PURPOSE_STATICALLY_FROZEN_NOT_RAW_LOCK_PROFILE_REGISTRY_RUNTIME_CAS_TIME_OR_EXTERNAL_ANCHOR_VERIFIED_NOT_LATEST_AUTHORITY_NOT_R01_NOT_READINESS_NOT_EXECUTION_NOT_ROLLBACK_OR_CLONE_RESISTANT
```

The exact event and projection claims are respectively:

```text
GOVERNANCE_SIGNED_STATUS_EVENT_MECHANICS_CANDIDATE_AUTHORITY_V2_PURPOSE_STATICALLY_FROZEN_NOT_RAW_REGISTRY_VERIFIED_NOT_R01_NOT_READINESS_NOT_EXECUTION
STATUS_LEDGER_REPLAY_PROJECTION_CANDIDATE_NOT_LATEST_AUTHORITY_NOT_R01
```

The current projection intentionally fixes the mechanics observations and all
authority outcomes as follows:

```yaml
governance_signatures_verified: true
logical_time_token_signatures_verified: true
store_dependency_candidate_references_exact: true
governance_status_purpose_authorized_by_pinned_registry: false
owner_signature_authenticated: false
store_dependencies_verified: false
latest_authority_status_verified: false
r01_satisfied: false
readiness_node_transition_issued: false
execution_authorized: false
authority_evidence: false
```

Authority/profile v2 now adds `governance_status` as a fifth exact *static*
service binding under the existing `GOVERNANCE_ANCHOR` role and
`STEP5_GOVERNANCE_CONTROL`; it does not add a ninth role and does not establish
an image or live closure. Real R01 integration must construct and admit that
fresh profile/registry/trust-lock graph, a profile-v2-bound generic CAS, a
profile-v2-bound current-cut logical-time token that binds the replayed current
head and complete-event-set root, an external anti-fork/anti-rollback anchor,
and producer-free pure replay. Only a new real-R01 verifier may then issue an
overlay transition that sets R01 alone true; R02 and all later rows remain
false. The live fifth image/closure, integrated generic CAS v2, current-cut
token issuer, real-R01 admission, and sequence-one R01-only overlay issuer
remain unimplemented. Section 9.6.2 freezes only their non-admitted candidate
wire shapes; it does not invent those live or authority facts.

#### 9.5.2 Verifier-only owner-status replay core v1

The verifier-only owner-status replay-core source SHA-256 is
`98bc33eecd4ef54131a8b25c2469466993154753b380c6a14c1606cd2761a2ca`;
its focused test source SHA-256 is
`493a4284e5d1931ae162810ff3cc0a4224dfe5b253a28bc570214b7a7d4814e5`.
At this replay-core candidate BYTE-STOP, the focused suite reported 30/30. Its
module schema is
`hegel-machine-phase2b-formal-wire-covert-step5-owner-status-replay-core/1`.
This is pure-verifier construction evidence only, not independent-review
admission, external currentness, authority, R01, readiness or execution.

The core imports the authority/profile-v2 verifier, the verifier-only logical-
time replay core and RFC 8785 implementation. It deliberately does not import
the owner-status service/local-ledger module, CAS-store producer or service,
runtime closure, auditor, qualification runtime, filesystem or socket paths,
and it contains no private signing key or producer callback. It independently
re-freezes the owner-authorization, decision, signed-event, checkpoint, head,
projection and legacy CAS-reference schema/1 surfaces and uses concrete
Ed25519 verification.

`verify_owner_status_replay_core_v1` consumes the raw Stage-B lock/profile/
registry bytes and independent raw out-of-band pins; the exact owner-
authorization bytes; exact ordered raw decisions, signed events, owner
checkpoints and owner heads; one separately supplied current-head byte string;
the raw projection; and exact ordered raw logical-time requests, tokens,
checkpoints and heads plus their expected event plan. It accepts exactly one
`ACTIVE` event or the exact two-event `ACTIVE` then terminal `REVOKED` chain.
It verifies the closed authority-v2 governance key/service/purpose graph,
governance signatures, all owner content IDs and predecessor links, complete-
event-set roots, and projection equality. It delegates only the separately
presented logical-time raw chain to the verifier-only logical-time core and
then exact-binds that result back to each owner decision.

The separately supplied current head must be byte-for-byte equal to the final
replayed owner head. That equality is explicitly only caller-supplied-chain
mechanics: `verify_external_currentness_v1` always fails closed because no
external current-cut or anti-fork anchor is present. The embedded legacy
CAS/1 references are checked for exact lineage, uniqueness and contiguous
candidate checkpoint order, but their receipts are not authenticated and
`verify_cas_dependencies_v1` always fails closed. A generic CAS-store/2
complete session, external durable checkpoint and external anti-rollback/
anti-clone anchor remain required outside this core.

The result is constructor-disabled, closure-privately issued and accepted only
for the exact issued identity with its immutable field snapshot unchanged.
The public verifier captures its parser, canonicalization, hashing, concrete
Ed25519, authority extraction, logical-time core and private result-issuance
graph before publication. This is ordinary Python/KAT mutation isolation, not
arbitrary closure-cell, interpreter-memory or loaded-code protection.

Successful replay fixes raw authority-graph verification, governance-
signature verification, canonical owner-chain replay, supplied-final-head
equality and logical-time supplied-chain mechanics true. It simultaneously
fixes legacy CAS receipt authentication, external CAS checkpoint, external
currentness, external anti-fork anchor, runtime closure, owner-signature
authentication, authority evidence, R01, readiness evidence and execution
authorization false. Its exact claim boundary is:

```text
PURE_OWNER_STATUS_REPLAY_CANONICAL_AUTHORITY_GRAPH_GOVERNANCE_SIGNATURE_AND_SUPPLIED_CHAIN_MECHANICS_ONLY_LEGACY_CAS_REFERENCE_CANDIDATE_NOT_CAS_VERIFIED_NOT_EXTERNAL_CURRENTNESS_NOT_ANTI_FORK_NOT_RUNTIME_NOT_OWNER_SIGNATURE_NOT_AUTHORITY_NOT_R01_NOT_READINESS_NOT_EXECUTION
```

Agreement between this pure core and the owner-status service/local-ledger
replay remains mechanics evidence over the presented bytes. It does not make
the supplied head externally current, authenticate the legacy CAS references,
establish the fifth service image or live closure, issue a real R01 receipt or
change any readiness row; the gate remains `HOLD`.

### 9.6 Exact current producer-free R01 composition surface

#### 9.6.1 Producer-free R01 pure-replay candidate composer v1

The producer-free R01 pure-replay composer source SHA-256 is
`db1a5b15ad830b3c7371ae6a913e970945e3b876e39058e6f6b9aa63b3badedc`;
its focused test source SHA-256 is
`c592bf7c92fa07820bd4cf19af4395f1fb75a9bf255c69322a069fae63262589`.
At this mechanics-candidate BYTE-STOP, its focused suite reported 12/12. The
same stop also exercised the 37 authority-v2, 29 logical-time-core, 30 owner-
status-core, 78 runtime-closure and 52 store-core upstream differential tests,
for 238/238 focused-plus-upstream checks. These counts identify construction
bytes and mutation/cross-splice coverage only; they are not live evidence or
authority admission.

The module schema is
`hegel-machine-phase2b-formal-wire-covert-step5-r01-pure-replay-candidate/1`.
Its sole accepting public composer consumes exact raw DTOs rather than decoded
producer objects: the raw out-of-band-pinned authority-v2 lock/profile/registry
graph; one raw owner-status chain and its raw logical-time chain; one raw
generic CAS-store/2 complete session; five raw offline runtime-closure graphs
with their canonical reports; and one canonical report bundle. The dependency
graph is definition-captured before publication, the result constructor is
disabled, and only the closure-private result identity with an unchanged field
snapshot is accepted. The module imports no producer, service, filesystem,
clock, socket or private signing key and exposes no decoded-mapping acceptance
path.

The composer replays the authority graph and the verifier-only owner-status,
logical-time and generic-store cores, and replays each offline closure graph.
It then exact-binds five canonical
`hegel-machine-phase2b-formal-wire-covert-step5-runtime-closure-report/1`
objects in the immutable service order `runner`, `pure_replayer`, `cas_store`,
`logical_time`, `governance_status`, followed by one canonical
`hegel-machine-phase2b-formal-wire-covert-step5-runtime-closure-report-bundle/1`.
Each report recomputes its content ID from the raw inputs and the privately
issued offline-verifier result, and the bundle binds the exact five service
names, report IDs and report-payload digests. These objects are unsigned,
offline mechanics summaries; they are not runtime attestations, durable
retention receipts, image-existence evidence or live entry/exit observations.

The generic CAS object sequence is exactly `13 + 8N`, where `N` is exactly one
`ACTIVE` event or exactly two events ordered `ACTIVE` then terminal `REVOKED`.
Thus the exact object count is 21 for the active chain and 29 for the revoked
chain. The fixed thirteen objects are, in order:

```text
stage_b_trust_lock_v2
authority_profile_v2
signer_registry_v2
owner_authorization_v1
runtime_closure_report_v1             # runner
runtime_closure_report_v1             # pure_replayer
runtime_closure_report_v1             # cas_store
runtime_closure_report_v1             # logical_time
runtime_closure_report_v1             # governance_status
runtime_closure_report_bundle_v1
owner_status_projection_v1
owner_status_genesis_head_v1
logical_time_genesis_head_v1
```

Each event contributes exactly these eight objects, in order:

```text
owner_status_decision_v1
owner_status_signed_event_v1
owner_status_checkpoint_v1
owner_status_non_genesis_head_v1
logical_time_request_v1
logical_time_signed_token_v1
logical_time_checkpoint_v1
logical_time_non_genesis_head_v1
```

The composer rejects any object-byte, ordinal, artifact-kind, content-ID,
payload-digest, size, service-order, result-identity, graph-context or cross-
module splice difference. Success fixes only presented raw authority-graph,
owner/time/store/offline-closure replay and their exact cross-bindings true.
It simultaneously fixes external currentness, external anti-fork anchoring,
CAS durability, live CAS, durable closure-input retention, live runtime
closure, live UID/GID and control-domain isolation, key custody, owner-
signature authentication, signed runtime attestations, authority evidence,
real R01, R01 satisfaction, overlay issuance, readiness evidence and execution
authorization false. Its exact claim boundary is:

```text
PRODUCER_FREE_PRESENTED_AUTHORITY_OWNER_TIME_CAS_AND_FIVE_OFFLINE_RUNTIME_CLOSURE_MECHANICS_EXACTLY_CROSS_BOUND_NOT_EXTERNAL_CURRENTNESS_NOT_ANTI_FORK_NOT_CAS_DURABILITY_NOT_LIVE_RUNTIME_NOT_UID_CONTROL_NOT_KEY_CUSTODY_NOT_OWNER_SIGNATURE_NOT_AUTHORITY_NOT_REAL_R01_NOT_READINESS_NOT_EXECUTION
```

`verify_real_r01_authorization_status_receipt_v1` and
`derive_r01_readiness_overlay_v1` remain fail-closed placeholders. Both reject
with `FAIL_STEP5_R01_PURE_REPLAY_V1_NOT_IMPLEMENTED`; the candidate composer
cannot admit a real R01 or issue even an R01-only readiness overlay. The gate
therefore remains `HOLD` and all sixteen readiness rows remain false.

#### 9.6.2 Verifier-only R01 evidence contract v1

The verifier-only R01 evidence-contract source SHA-256 is
`574e07e7ce0f19075c29fa59cc94e4e988a34c301e9e09005a8aed805d69e96d`;
its focused test source SHA-256 is
`5879e77ca4c0097a07f13e86b0d8c9c3c0d4018105c4e53c17f913983198213c`.
At this candidate-mechanics BYTE-STOP, its focused suite reported 173/173 and
both source and test passed bytecode compilation. The related regression stop
reported 384/384: 37 authority-v2, 63 owner-status producer, 30 owner-status
replay-core, 29 logical-time replay-core, 52 store replay-core and 173 R01
evidence-contract checks. Independent second review returned GO for this
candidate-only surface with no P0 or P1 finding left in the reviewed scope.
None of these checks is live, durable, currentness or authority evidence.

The contract freezes these nine canonical `/1` wire-schema kinds; the live
report kind occurs exactly five times in the authority-v2 service order:

| Ordinal | Frozen schema kind | Cardinality in the candidate DAG |
|---:|---|---:|
| 1 | `hegel-machine-phase2b-formal-wire-covert-step5-current-cut-response/1` | 1 |
| 2 | `hegel-machine-phase2b-formal-wire-covert-step5-current-cut-statement/1` | 1 |
| 3 | `hegel-machine-phase2b-formal-wire-covert-step5-live-runtime-closure-report/1` | 5 |
| 4 | `hegel-machine-phase2b-formal-wire-covert-step5-live-runtime-closure-bundle/1` | 1 |
| 5 | `hegel-machine-phase2b-formal-wire-covert-step5-signed-owner-status-anti-fork-anchor/1` | 1 |
| 6 | `hegel-machine-phase2b-formal-wire-covert-step5-external-owner-status-anchor-receipt/1` | 1 |
| 7 | `hegel-machine-phase2b-formal-wire-covert-step5-r01-receipt-body/1` | 1 |
| 8 | `hegel-machine-phase2b-formal-wire-covert-step5-dual-signed-r01-receipt/1` | 1 |
| 9 | `hegel-machine-phase2b-formal-wire-covert-step5-r01-readiness-overlay/1` | 1 |

The separately canonical
`hegel-machine-phase2b-formal-wire-covert-step5-external-witness-policy/1`
is not a tenth in-band DAG kind. It is supplied as raw canonical bytes under
independent out-of-band policy-ID and payload-SHA-256 pins. Its Ed25519 public
key, key ID and control domain must be disjoint from all eight authority-v2
roles, and its witness namespace/control domain cannot masquerade as an
authority role.

The exact candidate dependency order is:

```text
raw Stage-B lock/profile/registry + independent raw lock ID/SHA pins
  -> current-cut response
  -> current-cut statement (governance peer + owner projection + pre-cut time)

raw profile service bindings
  -> five signed report wrappers in exact order
     [runner, pure_replayer, cas_store, logical_time, governance_status]
  -> signed live-runtime-closure bundle

current-cut statement + a distinct current-cut token/checkpoint/head
  + independently raw-pinned external witness policy
  -> PURE-signed owner-status anti-fork anchor
  -> external-witness signed anchor receipt

profile owner authorization + current-cut response/statement
  + live-runtime bundle + PURE anchor/external receipt
  + main-CAS complete-session binding + five service-custody rows
  -> R01 receipt body
  -> SOURCE_RUNTIME_CUSTODIAN signature
  -> PURE_REPLAYER countersignature over body ID/body SHA/custodian-signature SHA
  -> dual-signed R01 receipt
  -> sequence-zero, no-transition, sixteen-row all-false candidate overlay
```

The only accepting public artifact boundary is
`verify_r01_evidence_contract_candidate_v1`. It requires the complete raw
authority-v2 graph and both raw lock pins; all nine candidate wire kinds; the
separately raw policy and both policy pins; the exact five-report tuple; and
the expected attempt-context ID. The nine decoded-mapping loaders used inside
that composition are removed from the module namespace after a two-stage
definition capture. Function-valued defaults are rebound onto the final
private clone graph, so replacing either local helpers or imported upstream
exports after publication does not redirect the accepting path. Public
canonicalization, content-ID and preimage helpers do not accept or admit an
artifact by themselves.

Each candidate signature is checked with concrete Ed25519 and an exact frozen
authority role/purpose. Live reports and their bundle use
`SOURCE_RUNTIME_CUSTODIAN/SIGN_RUNTIME_CLOSURE`; the owner-status anchor uses
`PURE_REPLAYER/SIGN_OWNER_STATUS_ANTI_FORK_ANCHOR`; the R01 receipt body first
uses `SOURCE_RUNTIME_CUSTODIAN/SIGN_R01_AUTHORIZATION_STATUS_RECEIPT`; and the
PURE countersignature uses
`PURE_REPLAYER/COUNTERSIGN_R01_AUTHORIZATION_STATUS_RECEIPT`. The countersign
subject includes the SHA-256 of the canonical custodian-signature object, so a
valid PURE signature cannot be spliced onto a different custodian signature.
The independent witness receipt is verified under the separately pinned
witness key, not under a ninth authority role.

The current-cut candidate accepts only an `ACTIVE`, zero-revocation,
nonterminal projection and requires `latest_sequence = event_count - 1`.
Owner ledger/head/projection and logical-time token/checkpoint/head identifiers
must carry their frozen producer prefixes; payload references use the exact
bare-SHA or `sha256:<hex>` form frozen for that field. The current-cut logical-
time token/head must differ from the pre-cut token/head. The initial PURE
anchor has sequence 1 and no predecessor, and the initial witness receipt is
exactly leaf zero of a one-leaf tree with an empty inclusion path and the
recomputed domain-separated leaf root.

The receipt body's *main* CAS binding requires exactly `21 + 8N` request IDs
and receipt IDs, where `N` is the current owner-status event count. The
canonical one-event `ACTIVE` case therefore binds 29 objects. It explicitly
rejects reuse of the pure-composer `13 + 8N` spool, which has only 21 objects
for that case. This is still only a candidate cardinality and lineage check:
the evidence contract does not replay a live CAS session or prove fsync,
readback, durability, external checkpointing or anti-rollback state.

Neither the pure-composer `13 + 8N` plan nor this evidence-contract
`21 + 8N` plan is admissible or promotable as an external-witness admission
plan. If `N` remains the owner-status event count, the prospective retention
lower bound is at least `26 + 8N`: the existing `21 + 8N` candidate set, one
admission transcript, and the four raw artifacts for one transcript-bound
post-Q2 logical-time event (request, signed token, checkpoint and new head).
That is a lower bound, not a frozen exact plan. Every concrete three-policy,
Q1/A/Q2, candidate-result, runtime/custody and durable-closure blob and its
retention receipt must be enumerated and counted separately in a future exact
plan.

The five report bodies exact-bind service name, static binding, role/key,
runtime instance, closure-policy/image evidence and typed entry/exit
observations. The bundle rejects runtime-instance, UID/GID-pair or control-
domain collisions and exact-binds all five report IDs and payload digests.
Each of the five custody rows must bind the same service/static role key and
runtime instance. Nevertheless every `service_image_executed`, entry/exit,
UID/GID, control-domain and custody-verification predicate is required to be
false in this candidate contract.

The final review also closed the earlier P1 families: RFC-8785 integral floats
are recursively restored only within the I-JSON safe-integer range, and every
integer validator rejects JSON booleans rather than accepting Python
`true == 1` equality; exact
mapping-versus-`MappingProxyType` comparison no longer loses the verified
profile/registry lineage; owner sequence/count and producer ID/SHA-address
forms are enforced; logical-time and store address forms are separated;
report nested-field types, role/purpose/runtime collisions and custody rows
are fail-closed; the first-anchor/predecessor/inclusion-root rules are exact;
the PURE countersign binds the custodian-signature SHA; main-CAS cardinality
cannot reuse the pure spool; and the privately issued immutable result now
retains the raw authority lock/profile/registry IDs and payload SHA-256 values
plus the independently pinned witness-policy ID and payload SHA-256.

A successful full raw replay sets only
`candidate_signature_mechanics_verified=true`. Its immutable result keeps
external currentness, live runtime closure, external anti-fork anchoring,
durable CAS, authority establishment, R01 satisfaction, overlay admission and
execution authorization false. Every R01-verdict field, receipt readiness
effect, sequence-zero overlay node, and live/bundle/custody claim likewise
must remain false. The exact candidate claim level is:

```text
SIGNED_WIRE_MECHANICS_CANDIDATE_NOT_LIVE_NOT_CURRENT_NOT_DURABLE_NOT_AUTHORITY_NOT_R01_NOT_READINESS_NOT_EXECUTION
```

`verify_real_r01_admission_v1` and
`issue_admitted_r01_only_overlay_v1` both fail exactly with
`FAIL_STEP5_R01_EVIDENCE_CONTRACT_V1_NOT_IMPLEMENTED`. No real currentness,
durability or authority admission and no sequence-one R01-only transition can
be inferred from the candidate signatures, external receipt or all-false
overlay. The gate remains `HOLD`, R01 remains false, and all sixteen readiness
rows remain false.

#### 9.6.3 Governance current-cut mechanics v1 focused FINAL

The governance current-cut mechanics source SHA-256 after the mandatory
evidence-contract repin is
`575feec9957681debf269f164b4deb22dd91a57ad648b1c470e2d4ec1e19fa89`;
its focused test source SHA-256 is
`00307a6c3c70eb4d7d903df683d457f699626e486bac5f322fef5e260a79b7cb`.
The rerun focused suite reported 24/24. This is the module-local focused FINAL
for the evidence-repinned current-cut mechanics surface. It is not a
live, authority or R01 FINAL. The subsequently reviewed external-witness
cascade is frozen separately in Section 9.6.4; together the three subsections
form the unified candidate-mechanics documentation cut, while every live,
authority and readiness gate remains closed.

The module freezes one canonical
`hegel-machine-phase2b-formal-wire-covert-step5-current-cut-request/1`.
The request takes a nonzero exact 32-byte challenge nonce and emits its
lowercase 64-hex representation; the request ID is recomputed under the frozen
current-cut request domain. The response and client statement use exactly the
`CURRENT_CUT_RESPONSE /1` and `CURRENT_CUT_STATEMENT /1` keysets already frozen
by the R01 evidence contract. All accepting authority inputs are the raw
Stage-B lock/profile/registry bytes, the raw out-of-band lock-ID bytes and raw
32-byte lock-payload digest, plus the expected attempt-context ID. The public
surface exposes no decoded-authority bypass, peer-credential callback, UID/GID
override, caller-supplied logical-time result or private result issuer.

The service and client primitives require an already-connected Linux
`AF_UNIX` `SOCK_SEQPACKET` socket. They do not bind, listen, accept, start a
daemon, or install a systemd unit. Packet handling uses `recvmsg`; empty,
oversize, truncated, control-truncated, ancillary-bearing or source-addressed
packets are rejected, as is a short `send`. The held socket inode and kernel
`SO_PEERCRED` tuple are read before and after the exchange and must remain
equal. The governance endpoint checks that its requester is the raw-profile
`runner` static UID/GID `41001/42001`; the client checks that its peer is the
raw-profile `governance_status` static UID/GID `41005/42005`. These checks bind
the local connected exchange to the static candidate identities. They do not
prove image execution, key custody, a distinct live control domain or the
identity of deployed service code.

For an `OwnerStatusLedgerV1` source, the governance side uses only its public
producer API in this exact optimistic order:

```text
current_head_bytes()
  -> latest_projection_bytes()       # producer performs complete-chain replay
  -> current_head_bytes()
```

The two head byte strings must be equal. The exact head and projection must
also agree on ledger ID, authority-profile ID, signer-registry ID,
attempt-context ID, governance role key and epoch, current head ID,
complete-event-set root, event count, latest sequence, status, revocation
count and terminal state. The only accepted status shape is one `ACTIVE`
event, latest sequence zero, zero revocations and `terminal=false`. An exact
`OwnerStatusServiceAppendV1` producer result is the alternative producer
input and is reparsed under its DTO invariants.

The three producer reads are not one atomic snapshot. Equal before/after head
bytes establish only stability during those three append-only replays; a
`REVOKED` append may occur immediately after the final read or after the
response is sent. Moreover, the evidence response does not record which of
the ledger-three-read or exact-append-result paths the service used. The
client result therefore keeps both `stable_during_three_replays` and
`exact_producer_append_projection_presented` false instead of accepting a
caller assertion as evidence. A real admission requires a fresh
admission-time requery or fence; this module does not provide one.

The pre-cut logical-time binding is not accepted as a decoded mapping or
caller-authored result. The client supplies the complete raw request, signed-
token, checkpoint and head byte tuples together with the raw authority graph,
raw lock pins and the exact event plan. The module invokes the producer-free
logical-time replay core and then requires that result's private issuance
lineage. Genesis-only input is rejected, and the verified logical-time event
count must equal the owner response event count, exactly one. The statement
projects only the verified time ledger ID, final-head ID and payload SHA,
last-token ID and next sequence. Because the public owner ledger API does not
export its raw decision/event chain, this count/authority/attempt cross-check
does not prove that the logical-time event plan is bound to the raw owner
decision bytes; `owner_time_raw_decision_binding_verified` remains false.

The response nonce, request hash, response hash, raw authority IDs, attempt
context and governance static binding are cross-checked through the statement.
Its socket-endpoint and runtime-instance identifiers are local content
addresses of the observed socket/peer/request/response tuple. The authority
profile contains no endpoint pin, and these local identifiers are not
cross-bound to a live runtime-closure report.

The final P1 closure correction moved the private result's exact error type,
failure code, schema, claim boundary and payload-SHA implementation into the
issuance/require closure. The accepting function graph and function-valued
defaults are cloned at the definition cut, including captured upstream raw
authority, time-replay and evidence helpers; the public wrappers hide the
captured issuer. Focused rebinding checks replace local globals, transitive
upstream entry points, implementation defaults, result constants and the real
admission error surface, while the accepting path and exact failure code stay
at the reviewed definition cut. Forged, copied or mutated private results are
rejected.

A privately issued candidate result may set only raw authority-graph replay,
kernel `SO_PEERCRED`, seqpacket request/response mechanics, presented owner
projection, one-event ACTIVE candidate, supplied pre-cut time-chain replay and
owner/time event-count cross-check true. Response-time currentness,
admission-time currentness, atomic owner/time fencing, external currentness,
external anti-fork anchoring, live runtime closure, live distinct UID/GID,
live distinct control domains, service-key custody, CAS durability, owner-
signature authentication, authority evidence, R01, readiness evidence and
execution authorization all remain exactly false. The exact claim boundary is:

```text
RAW_AUTHORITY_V2_BOUND_PRESENTED_OWNER_PRODUCER_PROJECTION_AND_SO_PEERCRED_SEQPACKET_AND_PRESENTED_PRECUT_TIME_REPLAY_MECHANICS_ONLY_NOT_ATOMIC_NOT_EXTERNAL_CURRENTNESS_NOT_ANTI_FORK_NOT_RUNTIME_CLOSURE_NOT_DURABLE_NOT_AUTHORITY_NOT_R01_NOT_READINESS_NOT_EXECUTION
```

`verify_real_current_cut_admission_v1` remains fail-closed and returns exactly
`FAIL_STEP5_R01_CURRENT_CUT_V1_NOT_IMPLEMENTED`. No successful candidate
round-trip changes R01, any readiness row or execution authorization; the gate
remains `HOLD`.

#### 9.6.4 Initial external-witness local-spool mechanics v1 focused FINAL

The evidence/current-cut/external-witness cascade is frozen at these exact
source, test and focused-suite cuts:

| Candidate-mechanics component | Source SHA-256 | Test SHA-256 | Focused stop |
|---|---|---|---:|
| R01 evidence contract v1 | `574e07e7ce0f19075c29fa59cc94e4e988a34c301e9e09005a8aed805d69e96d` | `5879e77ca4c0097a07f13e86b0d8c9c3c0d4018105c4e53c17f913983198213c` | 173/173 |
| Governance current-cut mechanics v1 | `575feec9957681debf269f164b4deb22dd91a57ad648b1c470e2d4ec1e19fa89` | `00307a6c3c70eb4d7d903df683d457f699626e486bac5f322fef5e260a79b7cb` | 24/24 |
| Initial external-witness spool v1 | `b0622d5e50e2393e116211a64622e6d3d36d76d476afdcd6e5fd98a777061a88` | `3d38655c28a9f9865a9bb8b8791268e790d8b953f9a8ca57cb763b27ed361c1d` | 46/46 |

The evidence-contract plus external-witness differential stop reported
219/219. The current-cut focused stop is reported separately rather than being
silently added to an unexecuted aggregate. These are candidate-mechanics FINAL
pins only; they do not promote the external-witness name, local filesystem
operations or valid signatures into external control, authority, durability,
R01, readiness or execution evidence.

The sole accepting production entry,
`issue_initial_external_owner_status_anchor_receipt_v1`, consumes the raw
Stage-B lock/profile/registry, raw external-witness policy and raw PURE-signed
owner-status anti-fork anchor. It also requires the independent raw Stage-B
lock ID and payload-digest pins, independent raw external-policy ID and
payload-digest pins, separately supplied expected lock/profile/registry and
attempt IDs as bytes, one concrete Ed25519 private-key object, and one existing
absolute owner-private spool-directory path. It accepts no decoded authority,
verified-object token, signer/verifier callback, key seed, relative directory
or inferred pin.

The raw authority graph is fully reopened under its pins. The policy is
RFC-8785 canonical, exact-keyset and separately content-addressed. Its witness
public key, key ID, control-domain ID and namespace cannot collide with or
masquerade as any of the eight authority-v2 roles. The witness is neither a
ninth authority role nor a sixth static Step-5 service. The concrete private
key must derive the exact public key in the raw-pinned policy, but that
private-key match proves neither custody nor an independently controlled
external service.

The anchor loader binds the exact raw authority and attempt IDs, current-cut
statement, current-cut logical-time token/checkpoint/head, owner ledger/head/
projection, external-policy ID, sequence one and null predecessor. It verifies
the concrete authority-v2 detached signature as exactly
`PURE_REPLAYER/SIGN_OWNER_STATUS_ANTI_FORK_ANCHOR`, with artifact kind
`owner_status_anti_fork_anchor`, epoch one and the raw-registry PURE public key.
No callback or locally substituted public key can satisfy this path.

Only the initial one-leaf receipt is implemented. Its exact construction order
is:

```text
receipt_id_subject = receipt fields excluding receipt_id and signature_hex
receipt_id = evidence content_id(external_owner_status_anchor_receipt,
                                 receipt_id_subject)
signature_subject = receipt_id_subject + receipt_id
signature_preimage = frozen external-receipt domain
                     + RFC8785(signature_subject)
signature_hex = external_policy_private_key.sign(signature_preimage)
```

Thus the ID/signature construction has no cycle: the ID excludes itself and
the signature, while the signature includes the already-derived ID and
excludes the signature. The first receipt is fixed to witness sequence one,
null previous receipt, leaf index zero, tree size one and an empty inclusion
path. Its tree root is exactly the lowercase SHA-256 hex digest of the frozen
one-leaf domain followed by the complete signed-anchor payload bytes, not the
anchor body or anchor ID.

The owner-private spool contains exactly five files in commit order:

```text
external-witness-initialization-v1.lock
00000000000000000001.external-anchor-receipt.<payload-sha>.json
00000000000000000001.external-witness-ledger.<payload-sha>.json
00000000000000000001.external-witness-checkpoint.<payload-sha>.json
00000000000000000001.external-witness-head.<payload-sha>.json
```

The initialization lock binds the raw authority IDs, attempt, policy and
anchor plus every later artifact ID, payload SHA and basename. Every record is
published with Linux `O_NOFOLLOW|O_CREAT|O_EXCL|O_CLOEXEC`, mode `0600`, the
effective owner UID, one hard link, bounded exact size, file `fsync`, directory
`fsync`, reopen and byte-exact readback. The already-existing directory must
be absolute, held with `O_DIRECTORY|O_NOFOLLOW`, owned by the effective UID and
have no group/other permission bits. The module never deletes a partial
publication: any gap, extra file, wrong mode/owner/link count, symlink,
content drift or unexpected basename poisons reuse.

An exact complete retry reopens and reconstructs all five artifacts and
returns the original receipt bytes without resigning or advancing sequence;
response-loss retry is therefore byte- and inode-idempotent. Concurrent empty-
spool initializers are serialized by exclusive creation, and a later exact
retry must replay the one committed graph. A partial crash prefix remains
fail-closed. A complete byte-for-byte copied spool can nevertheless replay,
and two different owner-private directories under the same policy can each
commit a different sequence-one anchor and root. Those explicit negative KATs
demonstrate that local `O_EXCL`, fsync, signatures and replay do not establish
host rollback resistance, clone resistance or an external global anti-fork
order.

The P1 repair cascade starts in the evidence contract: every exact integer
validator now rejects JSON booleans, so `true` can no longer masquerade as
epoch or sequence one in Python equality. The external-witness policy and
persisted receipt independently revalidate `witness_key_epoch`, witness
sequence, leaf index and tree size with that exact-type rule. The external
module pins the corrected evidence source and differentially reuses its exact
field constants, content-ID helper and receipt-signature preimage helper. The
current-cut module was then repinned to the same corrected evidence source.
Definition-captured call graphs and function-valued defaults keep post-import
rebinding of local helpers, upstream authority/evidence exports and Merkle
domains from redirecting the accepting paths.

The privately issued external-witness candidate may set exactly three positive
mechanics predicates: raw authority/PURE-signature mechanics, receipt-signature
mechanics and local fsync/readback mechanics. Same-UID isolation, independent
or local-filesystem identity, host rollback resistance, key custody, attempt
freshness, a live external service, external control, external currentness,
external anti-fork anchoring, durability, authority, R01, readiness and
execution authorization all remain false. Its exact claim boundary is:

```text
INITIAL_ONE_LEAF_EXTERNAL_WITNESS_SIGNATURE_AND_EXCLUSIVE_LOCAL_SPOOL_MECHANICS_ONLY_NOT_DISTINCT_UID_NOT_INDEPENDENT_OR_LOCAL_FILESYSTEM_NOT_HOST_ROLLBACK_RESISTANCE_NOT_KEY_CUSTODY_NOT_EXTERNAL_CONTROL_NOT_EXTERNAL_CURRENTNESS_NOT_AUTHORITY_NOT_R01_NOT_READINESS_NOT_EXECUTION
```

`verify_real_external_witness_admission_v1` remains fail-closed with
`FAIL_STEP5_R01_EXTERNAL_WITNESS_V1_NOT_IMPLEMENTED`. The module has no
listener, systemd unit, process observation or live admission path. Its valid
receipt and complete local spool remain inputs to later candidate replay only;
they do not change the `HOLD` gate, R01, any readiness row or execution
authorization.

#### 9.6.5 External-witness admission-transcript candidate v1 focused FINAL

The external-witness admission-transcript candidate source SHA-256 is
`b480b0d4b286e1b0cfb43f88846bbbf635b753fffc64b5e89e9073264360e285`
(3,070 lines; 133,966 bytes). Its focused test source SHA-256 is
`30d5b6b68144a052c0aaeb9da3d610c0d8687bc90d340c1268cbe37d155ecf3c`
(1,885 lines; 73,475 bytes), and the focused stop reported 94/94. Source and
test bytecode compilation passed. The full Step-5 wildcard final reported
1,010/1,010 in 387.24 seconds with exit zero. This is the module-local FINAL
for a presented transcript candidate only. It does not make the transcript
live, current, durable, atomic, external or authoritative and does not admit
R01.

The exact pre-freeze first-failure and repair chain was retained rather than
rewritten as an immediate success:

1. the original pre-freeze KAT first reproduced the missing current-map proof
   and distinct durable-policy/high-water/CAS schema surface as 2 failures;
2. extending those schemas next exposed a loader/public-wrapper argument
   misalignment, so the positive corpus stopped with `TypeError`;
3. after complete Q1/A/Q2 linkage was wired, the candidate-result projection
   still selected pre-cut logical-time ID `[0]` and failed exactly with
   `FAIL_STEP5_R01_EXTERNAL_ADMISSION_V1_CLAIM_BOUNDARY`; and
4. selecting the final post-Q2 IDs with `[-1]`, making the signature count
   dynamic as fourteen plus verified logical-token signatures, and freezing
   its exact scope closed the focused suite at 94/94.

That repair chain establishes only source/test mechanics. Real admission must
still fail at the first missing or unverified live edge in this separate
operational chain:

```text
raw/OOB-pinned Stage-B graph and three external policies
  -> live W Q1 current-map observation and proof
  -> independent A exact-high-water atomic durable compare-and-append
  -> live W Q2 current-map re-observation and confirmation
  -> complete same-ledger post-Q2 logical-time append
  -> live runtime closure, key custody and distinct UID/GID control
  -> real external-witness admission
```

This module replays only a presented signed shape for those edges. It neither
contacts W or A nor obtains an admission-time current-map observation, durable
storage barrier, process observation or kernel peer credential. Therefore the
pre-freeze chain still stops before real admission even when every candidate
byte verifies.

The sole accepting candidate entry consumes the raw canonical Stage-B
lock/profile/registry and both raw lock pins, the expected attempt ID as bytes,
and three raw canonical policy payloads. Each policy has its own independently
supplied out-of-band policy-ID bytes and raw 32-byte payload-SHA-256 digest:

1. the bootstrap
   `hegel-machine-phase2b-formal-wire-covert-step5-external-witness-policy/1`,
   content-addressed under the corrected evidence contract;
2. the
   `hegel-machine-phase2b-formal-wire-covert-step5-online-witness-policy/1`,
   signed by the bootstrap key and binding the W online key, custody-attestor
   key, static service identity and runtime/custody policy references; and
3. the
   `hegel-machine-phase2b-formal-wire-covert-step5-external-durable-anchor-policy/1`,
   signed by the separately pinned durable-anchor key A and binding both prior
   policies, A's high-water and its exact compare-and-append semantics.

No decoded policy or mapping can cross the public boundary. The bootstrap,
online-W, custody-attestor and A key IDs and public keys are pairwise distinct
and disjoint from all eight authority-v2 role keys. Their control domains,
namespaces and static service bindings are collision-checked as applicable;
the W policy UID/GID values must also avoid the authority service UID/GID set.
These are static presented-policy separations only, not evidence that the keys
are held by independent live actors or that the named UID/GID identities ran.

W's presented current-map proof shape is reconstructed from the exact raw
initial anchor, receipt, ledger, checkpoint and head. It is fixed to witness,
tree and anchor sequence one, `CURRENT_UNSUPERSEDED`, `superseded=false`,
`revoked=false`, leaf zero of a one-leaf map, a recomputed domain-separated map
root, and empty inclusion and presented-to-live consistency paths. Q1 requires
the same candidate mapping in the bootstrap-signed request and in both the
online-W-signed `observed_current_mapping` and `prepared_candidate_mapping`.
Q2 requires W to sign that same mapping again after the presented A receipt.
The empty one-leaf paths and repeated signed mapping are candidate proof
mechanics; they do not query or establish a live current map.

A is governed by a distinct raw/OOB-pinned policy. Its only frozen initial
high-water is the exact genesis mapping: checkpoint and head sequence zero,
the exact genesis checkpoint/head IDs, zero checkpoint/head payload digests,
and the zero genesis tree root. The online-W-signed A request compares that
exact high-water and proposes W's one-leaf candidate mapping under
`ATOMIC_COMPARE_EXACT_OOB_HIGH_WATER_THEN_APPEND_ONCE`. A then signs a
sequence-one checkpoint and head with the exact genesis predecessors and a
receipt binding both payloads, both nonces, previous and committed mappings,
`compare_matched=true`, `append_applied=true` and
`durable_before_return=true`. Those are internally cross-bound signed fields;
because this verifier does not operate A, they do not prove that a real atomic
CAS or durable-before-return barrier occurred.

The exact presented protocol order is acyclic:

```text
bootstrap-signed Q1 PREPARE request
  -> online-W-signed Q1 PREPARE response
  -> online-W-signed A compare-and-append request
  -> A-signed checkpoint -> A-signed head -> A-signed receipt
  -> bootstrap-signed Q2 CONFIRM request
  -> online-W-signed Q2 CONFIRM response
  -> bootstrap-signed admission transcript
  -> one transcript-bound post-Q2 logical-time event
  -> bootstrap-signed admission candidate result
```

Prepare, confirmation and post-Q2 logical-time nonces must be pairwise
distinct. Every Q1/A/Q2 edge exact-binds its predecessor IDs and payload
digests, both protocol nonces, the three policy IDs, the raw authority graph,
attempt and expected mapping/outcome. The transcript binds all Q1/A/Q2 IDs and
digests but does not refer to a not-yet-created time artifact. The final time
event instead binds the transcript ID and `sha256:<payload-sha>` under purpose
`COMMIT_EXTERNAL_WITNESS_ADMISSION_TRANSCRIPT`; only the later result binds
both. This direction removes a transcript/time content-ID cycle.

The logical-time input is four complete raw exact-order tuples: requests,
signed tokens, checkpoints and heads. There must be at least two events, equal
request/token/checkpoint counts and exactly one more head than event count.
The producer-free replay core verifies the complete same-ledger chain. The
penultimate token, checkpoint and head IDs and payload digests must equal the
PURE-signed candidate anchor's current-cut bindings, and the final event must
be the transcript-bound post-Q2 event above. A same-authority foreign prefix,
single-event genesis fork, missing ancestry, reorder or incomplete tuple is
rejected; a detached valid final event cannot replace the anchor's complete
same-ledger ancestry.

The verified signature scope is dynamically `14 + N`, where `N` is the
producer-free replay result's verified logical-token signature count. The
fixed fourteen are the PURE anchor signature, bootstrap-witness receipt
signature, bootstrap signature on the online policy, A signature on the
durable policy, and the ten Q1/A/Q2/transcript/result signatures. Authority-
graph signatures are deliberately excluded. The exact exported scope is:

```text
ADMISSION_LAYER_14_PLUS_LOGICAL_TOKEN_SIGNATURES_EXCLUDES_AUTHORITY_GRAPH_SIGNATURES
```

The fixed presented admission-layer object count is 18: three policies, five
initial external-witness candidate artifacts, and ten Q1/A/Q2/transcript/result
objects. It does not replace the retention-plan rule above. In particular,
neither the old `13 + 8N` nor `21 + 8N` owner-event plan is promotable; even
the `26 + 8N` transcript-plus-four-time-artifact figure is only a lower bound,
with every concrete closure blob counted separately.

The Q1 response, Q2 response and transcript must carry the same
runtime/custody binding and must match the online policy's static service, UID,
GID, user namespace, control domain and policy references. The signed binding
also says that the online key is nonexportable. The contract nevertheless
requires both transcript and candidate result to state
`runtime_closure_verified=false`,
`key_custody_verified=false` and `distinct_live_uid_gid_verified=false`.
Static equality and a signed nonexportable assertion cannot establish those
live facts.

A privately issued immutable result may set only raw authority-graph replay,
three OOB policy pins, pairwise key-set separation, presented Q1/A/Q2 linkage
and supplied post-Q2 logical-time-chain replay true. It requires private
issuance lineage and an unchanged field snapshot. Private-key, spool-path,
decoded-mapping, callback and caller-nonce inputs are absent. Live channel,
external currentness, real atomic CAS, CAS durability, non-equivocation,
global anti-fork, external control, runtime closure, key custody, distinct live
UID/GID control, authority evidence, R01 satisfaction, readiness evidence and
execution authorization remain false. Its exact claim boundary is:

```text
PRESENTED_SIGNED_TRANSCRIPT_MECHANICS_ONLY_NOT_LIVE_CHANNEL_NOT_CURRENT_NOT_DURABLE_NOT_NON_EQUIVOCATING_NOT_GLOBAL_ANTI_FORK_NOT_RUNTIME_CLOSURE_NOT_KEY_CUSTODY_NOT_DISTINCT_LIVE_UID_GID_NOT_AUTHORITY_NOT_R01_NOT_READINESS_NOT_EXECUTION
```

`verify_real_external_witness_admission_v1` remains fail-closed with
`FAIL_STEP5_R01_EXTERNAL_ADMISSION_V1_NOT_IMPLEMENTED`. The candidate verifier
has no channel, socket, filesystem, path, clock, process, service or private
key input and does not operate Q1, A, Q2 or a logical-time producer/service. A
94/94 focused stop therefore leaves the gate `HOLD`, R01 and all readiness rows
false, and execution unauthorized.

### 9.7 Definitive local W/A actor mechanics and integration stop

The frozen `/1` admission contract now has two concrete local actor-mechanics
implementations and one public-sequence integration KAT. They close the
presented-byte path from W through A and back through W into the producer-free
`b480` verifier. They do not close any external, live, durable, custody,
currentness or authority edge. The exact reviewed artifacts are:

| Surface | Artifact | SHA-256 | Focused stop |
|---|---|---|---:|
| local durable A | `src/hegel_machine/phase2b_formal_wire_covert_step5_r01_durable_anchor_v1.py` | `98478a96c832d48a5ffc574af8a20ed25408845c3d9918e3c391606cc34fdfff` | 35/35 |
| local durable A KAT | `tests/test_phase2b_formal_wire_covert_step5_r01_durable_anchor_v1.py` | `5831b89344a7b6ce72d80da1e8fb7f87899ca5c4f6c9f16a03ac43a71c788f49` | 35/35 |
| in-process online W | `src/hegel_machine/phase2b_formal_wire_covert_step5_r01_online_witness_v1.py` | `1560e3ed93e2caf16f40d2f31455ed6c10a2044d7a7883288d619d91d6389fcb` | 10/10 |
| in-process online W KAT | `tests/test_phase2b_formal_wire_covert_step5_r01_online_witness_v1.py` | `b23cb3c2e30e62127a6e7475b021e8a75f5205d67e8df0d3474d6c03410e912e` | 10/10 |
| public actor integration KAT | `tests/test_phase2b_formal_wire_covert_step5_r01_external_witness_actor_integration_v1.py` | `d99281b3d482043e08cb8513a1914ef6c1bc129d0c668fb757401fb21da258f4` | 9/9 |

The frozen `b480` admission contract, A, W and the cross-actor integration KAT
closed together at 148/148. This count is the arithmetic union of the
previous 94 admission-contract KATs, 35 A KATs, 10 W KATs and 9 cross-actor
KATs. It is a source/test mechanics stop, not an operational or scientific
qualification result.

#### 9.7.1 Local durable-anchor A mechanics

A is constructed from the exact raw bootstrap, online-W and durable-anchor
policy payloads, their independently supplied raw ID and SHA-256 pins, the
pinned A Ed25519 private key and one already-existing absolute owner-private
`0700` directory. It accepts no decoded mapping, caller outcome, UID override,
socket, listener, service, daemon or clock input. Construction verifies the
three-policy closure, the bootstrap and A signatures, the exact genesis OOB
high-water and the private/public A key match.

One directory is one initial slot keyed by the exact attempt context and
`anchor_sequence=1`. Under an exclusive directory lock, A replays the exact
raw Q1 request, Q1 response and A request, compares the exact genesis mapping,
and composes the contract-exact A checkpoint, head and receipt. Publication
uses an exclusive intent and no-replace operations; every record is fsynced,
reopened, metadata-checked and byte-read back, with directory fsyncs at the
publication boundaries. The head is published last as the sole commit marker.
An exact complete retry returns the original three payloads byte-for-byte and
does not re-sign. A conflicting complete-slot request is rejected while the
original exact replay remains available; an intent-only or partial/gapped,
extra, corrupt, symlinked, hardlinked or mode-invalid namespace fails closed
without recovery or repair.

The A red-team found and repaired one P1 path-replacement defect. Construction
now records the opened spool directory's exact `(st_dev, st_ino)` identity.
Every compare-and-append reopens and locks the directory and requires that
identity both before replay/publication and after append. Replacing the path
with a different directory after actor construction can no longer redirect a
valid actor into an attacker-selected empty slot.

These are local filesystem and signature mechanics only. Local flock, fsync,
reopen and readback do not establish external durability, host rollback or
clone resistance, non-equivocation, live-service execution, external
currentness, key custody, authority, R01, readiness or execution. A's
`verify_real_external_witness_admission_v1` therefore remains fail-closed with
`FAIL_STEP5_R01_DURABLE_ANCHOR_V1_NOT_IMPLEMENTED`.

#### 9.7.2 In-process online-witness W mechanics

W is constructed from the exact raw Stage-B lock/profile/registry, the three
raw external policies, the five raw initial external-witness candidate
artifacts, every required OOB ID/SHA pin, the expected attempt-ID bytes and the
pinned online-W private key. It replays the candidate mapping from those raw
bytes, creates the confirmation nonce internally, takes a concrete local
`/proc` process observation, and emits the exact online-W-signed Q1 response,
A request and Q2 response. The actor exposes no decoded mapping, caller nonce,
runtime-observation callback, UID/GID override or caller-selected outcome.

The stored namespace lifecycle has three explicit states:

```text
PREPARED -> COMPLETE
    |
    +------> POISONED on a validation/signing failure before completion
```

An absent namespace has no stored session. An exact Q1 retry in `PREPARED`
returns the original Q1 response and A request bytes. A malformed request that
fails before the occupied-session fence does not poison that session. A fully
valid but different Q1 or context for an occupied `PREPARED` namespace is a
conflict and permanently transitions that session to `POISONED`. This last
transition is the repaired W P1: the prior implementation raised only
`FAIL_FENCE` and left the original prepared session reusable after a valid
cross-request. Once `COMPLETE`, an exact terminal retry returns the cached Q2
response without re-observing the process or re-signing; a nonexact terminal
input is rejected as a conflict but does not destroy the immutable exact
terminal replay. Any A/Q2 validation or signing failure entered from
`PREPARED` poisons the session. Process restart loses this in-memory fence and
provides no recovery.

W's process and runtime/custody fields remain a local self-report. Equal
entry/Q2 observations, a generated nonce and exact signatures do not prove a
live transport, runtime closure, nonexportable key custody, restart-safe
session, distinct live UID/GID control, logical-time ancestry, external
currentness, durability, non-equivocation, global anti-fork, authority, R01,
readiness or execution. W's `verify_real_external_witness_admission_v1`
therefore remains fail-closed with
`FAIL_STEP5_R01_ONLINE_WITNESS_V1_NOT_IMPLEMENTED`.

#### 9.7.3 Public W -> A -> W -> b480 same-ledger closure

The cross-actor KAT uses only the public actor sequence for the three actor
steps:

```text
W.begin_prepare(raw Q1)
  -> A.compare_and_append(raw Q1, raw W Q1 response, raw W A request)
  -> W.confirm(raw A checkpoint, raw A head, raw A receipt, raw Q2)
  -> b480 producer-free admission-candidate replay over the complete raw graph
```

The W outputs are byte-exact differentials against the frozen `b480` contract,
and A's checkpoint/head/receipt are likewise byte-exact. The integration then
forms the bootstrap-signed transcript and candidate result and appends the
transcript-bound post-Q2 event to the same logical-time ledger whose preceding
event is bound by the original PURE anchor. The accepted chain contains two
requests, two signed tokens, two checkpoints and three heads. The producer-free
`b480` verifier accepts the complete ancestry and sets only
`presented_q1_a_q2_chain_verified` and
`supplied_post_q2_logical_time_chain_verified` for these edges.

The nine cross-actor KATs also reject a re-signed divergent authority triple
before A creates an intent, a genesis/candidate mapping swap before A creates
an intent, a valid cross-session Q2 splice with subsequent W poison, and a
same-attempt detached logical-time genesis fork. Thus the public path closes
the exact presented mechanics and same-ledger lineage; it still does not
create a live W channel, an independent durable A, runtime closure, external
custody, currentness, non-equivocation or authority.

Across the A result, both W results and the final `b480` result, every external
or elevated claim remains false: live transport/service, runtime closure,
distinct live UID/GID control, key custody, external durability, host rollback
or clone resistance, external currentness, non-equivocation, global anti-fork,
authority evidence, R01 satisfaction, readiness and execution authorization.
The integrated 148/148 stop therefore leaves the construction gate `HOLD`.

#### 9.7.4 Post-`b480` `/2` presented-overlay candidate mechanics

The former absent-surface statement is superseded by the following exact
candidate-mechanics BYTE-STOP. This is a presented, producer-free replay
surface, not the real live-runtime overlay:

| Surface | Artifact | SHA-256 | Lines | Bytes |
|---|---|---|---:|---:|
| presented overlay `/2` source | `src/hegel_machine/phase2b_formal_wire_covert_step5_r01_live_runtime_overlay_v2.py` | `5e4b23b79c04e4a613995ddec5aafe883e1aa694b945b86bda7ba67caff3be0e` | 3290 | 172787 |
| presented overlay `/2` KAT | `tests/test_phase2b_formal_wire_covert_step5_r01_live_runtime_overlay_v2.py` | `a1c694b15b3c7b7479a92590e888815f0211c65abd98370d6b17f82cbc9a44d2` | 2129 | 88117 |

The focused stop is 18/18, the relevant combined stop is 279/279, and the full
Step-5 wildcard regression on these final source/test bytes is 1082/1082 in
484.59 seconds with exit 0. These counts identify only the exact source/test
mechanics above. They are not a live-service, deployment, durability, custody,
authority, R01, readiness or execution qualification.

The accepted dependency and publication relation is strictly one-way. In
compact form, the presented DAG is
`b480 + f836 -> six live artifacts -> six-object CAS -> presented A2`:

```text
frozen b480 presented admission-candidate replay
  + frozen f836 offline image-closure replay over the complete raw graph
  -> live-runtime policy
  -> challenge
  -> response
  -> presented process-observation report
  -> presented custody-attestation object
  -> mechanics-only live-runtime result
  -> six-object CAS replay in that exact artifact order
  -> presented A/2 request -> checkpoint -> head -> receipt
  -> privately issued presented-overlay mechanics result
```

The custody-attestation policy and A/2 policy are separately raw and
out-of-band pinned prerequisites; they are not extra members of the six-object
CAS plan. The six stored artifact kinds, in exact ordinal order, are
`live_runtime_policy_v2`, `live_runtime_challenge_v2`,
`live_runtime_response_v2`, `live_runtime_report_v2`,
`live_runtime_custody_attestation_v2`, and `live_runtime_result_v2`, each with
the frozen maximum of 16,777,216 bytes. Every one of those six bodies rejects
same-session CAS fields, and the result additionally contains no A/2 request,
checkpoint, head or receipt reference. Only after all six content IDs exist
does the CAS replay derive its complete binding; only that downstream binding
enters the presented A/2 compare-and-append transcript. There is no reverse
CAS/A2-to-result edge, so the earlier content-address cycle is absent.

The public `verify_presented_live_runtime_overlay_v2` ABI has exactly 51
required positional raw inputs, in this order:

```text
stage_b_trust_lock_payload
authority_profile_payload
signer_registry_payload
bootstrap_external_witness_policy_payload
online_witness_policy_payload
durable_anchor_policy_payload
candidate_anchor_payload
candidate_receipt_payload
candidate_ledger_payload
candidate_checkpoint_payload
candidate_head_payload
q1_prepare_request_payload
q1_prepare_response_payload
a_compare_and_append_request_payload
a_durable_checkpoint_payload
a_durable_head_payload
a_compare_and_append_receipt_payload
q2_confirm_request_payload
q2_confirm_response_payload
admission_transcript_payload
post_commit_logical_time_request_payloads_in_exact_order
post_commit_signed_logical_time_token_payloads_in_exact_order
post_commit_logical_time_checkpoint_payloads_in_exact_order
post_commit_logical_time_head_payloads_in_exact_order
admission_candidate_result_payload
runtime_closure_policy_payload
runtime_closure_evidence_payload
raw_oci_index_payload
raw_oci_manifest_payload
raw_oci_config_payload
ordered_raw_layer_chunk_tuples
ordered_support_artifact_chunk_tuples
custody_attestation_policy_payload
live_runtime_policy_payload
live_runtime_anchor_policy_payload
live_runtime_challenge_payload
live_runtime_response_payload
live_runtime_report_payload
live_runtime_custody_attestation_payload
live_runtime_result_payload
live_runtime_cas_retention_plan_payload
live_runtime_cas_store_request_payloads_in_exact_order
live_runtime_cas_object_payloads_in_exact_order
live_runtime_cas_signed_receipt_payloads_in_exact_order
live_runtime_cas_checkpoint_payloads_in_exact_order
live_runtime_cas_head_payloads_in_exact_order
live_runtime_cas_response_payloads_in_exact_order
live_runtime_anchor_request_payload
live_runtime_anchor_checkpoint_payload
live_runtime_anchor_head_payload
live_runtime_anchor_receipt_payload
```

It then requires exactly 19 keyword-only raw pins, with no defaults, in this
order:

```text
out_of_band_expected_lock_id_bytes
out_of_band_expected_lock_payload_sha256_digest
out_of_band_expected_bootstrap_external_witness_policy_id_bytes
out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest
out_of_band_expected_online_witness_policy_id_bytes
out_of_band_expected_online_witness_policy_payload_sha256_digest
out_of_band_expected_durable_anchor_policy_id_bytes
out_of_band_expected_durable_anchor_policy_payload_sha256_digest
out_of_band_expected_runtime_closure_policy_id_bytes
out_of_band_expected_runtime_closure_policy_payload_sha256_digest
out_of_band_expected_runtime_closure_evidence_id_bytes
out_of_band_expected_runtime_closure_evidence_payload_sha256_digest
out_of_band_expected_custody_attestation_policy_id_bytes
out_of_band_expected_custody_attestation_policy_payload_sha256_digest
out_of_band_expected_live_runtime_policy_id_bytes
out_of_band_expected_live_runtime_policy_payload_sha256_digest
out_of_band_expected_live_runtime_anchor_policy_id_bytes
out_of_band_expected_live_runtime_anchor_policy_payload_sha256_digest
expected_attempt_context_id_bytes
```

The nineteen pins are mandatory: nine exact ID/SHA-256 pairs for the Stage-B
lock, bootstrap, online-W, durable-A, runtime-closure policy,
runtime-closure evidence, custody-attestation policy, live-runtime policy and
live-runtime A/2 policy, plus the exact attempt-context ID. The verifier also
pins the inspected dependency source bytes:

| Frozen dependency | Source SHA-256 |
|---|---|
| authority/profile v2 | `03d689327ca422f93e5a32db0a7f813fe84b1099f1bac1b31395c975d76d8e86` |
| external-witness admission contract v1 (`b480`) | `b480b0d4b286e1b0cfb43f88846bbbf635b753fffc64b5e89e9073264360e285` |
| offline runtime-closure replay v1 (`f836`) | `f836b3d00f332398214d3ccf7787109abb8f2de96b7b8281748bbb0ec695f2f9` |
| generic CAS-store replay core v2 | `e33bb8ae1f4c70668de3bfbdbd803a953513db94cb0a9c1ceb1082e97141de02` |

Definition-time capture clones the local verifier function graph into a
private namespace and freezes the `b480`, `f836` and store replay entrypoints
as defaults. The capture KAT replaces public module helpers and upstream
public entrypoints with bombs and still reaches only the captured graph and
the private-result requirements. The first 279-test combined invocation
exposed one test-only identity oracle after an earlier dependency module
reload; it was not a source/product failure. With source bytes unchanged, the
KAT now snapshots the dependency entrypoint objects at collection time. The
focused rerun is 18/18 and the same relevant combined set is 279/279, making
the capture check independent of that suite order. An independent read-only
reviewer disposition is `GO; no reproducible P0/P1`. This static GO is scoped
only to the presented candidate mechanics and does not promote any claim.

The privately issued result may mark raw authority-graph replay, frozen
`b480` candidate replay, offline static image-graph replay, all nineteen pins,
key-set disjointness, the six-artifact chain, presented report and
custody-attestation byte/signature checks, six-object CAS replay and presented
A/2 compare-and-append mechanics as verified. In particular, the named
`presented_custody_attestation_verified` predicate describes only the supplied
object's bytes and signature; it is not a custody or non-exportability claim.

Every elevated or operational predicate remains false, including service-image
existence/pull/execution, runtime-closure signature or live closure, remote
attestation, entry/exit process identity, socket/peercred/pidfd/procfs checks,
nonce freshness, high-water, one-shot and round-trip checks, distinct live
UID/GID or control domain, key custody, online-key non-exportability, live
channel, external CAS or anchor, anti-rollback, external currentness,
durability, non-equivocation, global anti-fork, malicious-root resistance,
authority evidence, R01 satisfaction, readiness evidence and execution
authorization. The claim level remains exactly
`RAW_PRESENTED_MECHANICS_ONLY_NOT_LIVE_NOT_CURRENT_NOT_DURABLE_NOT_NON_EQUIVOCATING_NOT_AUTHORITY_NOT_R01`.

`verify_real_live_runtime_overlay_v2` remains fail-closed with
`FAIL_STEP5_R01_LIVE_RUNTIME_OVERLAY_V2_NOT_IMPLEMENTED`. Thus the real overlay
remains `NOT_IMPLEMENTED`; this candidate does not set either live-overlay
readiness row, issue an R01-only readiness overlay, change the all-false
sixteen-row readiness state, or retroactively upgrade the frozen `b480`
candidate. The gate remains `HOLD` and execution remains unauthorized.

### 9.8 Remaining prospective schema boundary

For every remaining object, the machine validator must freeze an exact schema
version, exact top-level keyset, exact nested keysets, exact enum domains,
exact list order, and exact signature-preimage domain. The prose in this
document cannot add aliases, optional fields, default values, or alternate
encodings. Before static admission, this section must be checked
field-for-field against the committed validator's exported keyset constants;
any mismatch is a construction failure, not a documentation tolerance.

The minimum binding relation is:

| Object | Required exact binding purpose |
|---|---|
| `authority_profile` | candidate Stage-B profile ID; Stage-A images/source; threats; timeouts; graph and schema set |
| `role_registry` | governance-signed candidate registry with exact ordered eight roles, purposes/domains, public keys and epoch |
| `technical_independence` | live distinct process/container/namespace/key/volume controls under the disclosed same-admin model |
| `attempt_context` | fresh Step-5 identity, profile, registry epoch, fixture branch, nonce, no actual-960 identity |
| `authorization_status_head` | future authority-admitted governance status head; owner-status-head/1 and the sequence-zero R01 candidate remain mechanics/candidate objects, not latest-status evidence |
| `r01` | future real receipt over latest active/non-revoked status, CAS checkpoint and logical time; not the dual-signed, local-spool or admission-transcript candidate |
| `synthetic_branch` | candidate exact protocol/freeze/logical-contract/DAG/profile bindings and synthetic-only declaration |
| `r02` | future admitted R01 plus independently observed selected-branch event; not the declared-equality candidate |
| `runtime_closure` | image/source/import/native/argv/env/FD/namespace/cgroup entry and exit closure |
| `cas_store_receipt` | store-observed original bytes, zero-to-one publication, fsync/readback, role and logical-time token |
| `worker_retention_batch` | exact twelve original worker roles and CAS receipts, no missing/extra/reorder |
| `full_results_chunk` | canonical chunk ordinal/range/count/bytes/root transferred by descriptor |
| `full_results_index` | exact ordered chunk set, total count/bytes, receipt/result roots |
| `audit_receipt_projection` | complete canonical scientific projection needed for independent full replay |
| `r03` | exact R02, wire/vector/envelope/source/runtime/CAS/time evidence and R03 signature |
| `r04` | exact R03, full results, twelve workers, invariance/statistics and R04 signature |
| `r05` | exact R03 plus R04, integrated fresh null/leak direction and R05 signature |
| `logical_time` | attempt/purpose/request-bound monotonic signed hash-chain event |
| `evidence_manifest` | exact ordered semantic-role-to-CAS mapping with no undeclared object |
| `cas_checkpoint` | append-only store head and complete manifest object set |
| `pure_verification` | producer-free full replay inputs, recomputed roots, predicates, disposition and signature |
| `readiness_overlay` | append-only R01--R05 state transitions and exact receipt dependencies |
| `terminal` | unique final disposition, complete chain roots, exit/residue state and nonclaims |

No artifact is accepted from a generic mapping with merely sufficient keys.
All mappings use equality against their frozen keyset. Unknown fields, missing
fields, wrong ordering where order is semantic, alternate numbers, noncanonical
Unicode, duplicate JSON names, floats outside an explicitly frozen binary64
projection, non-finite values, and trailing bytes fail closed.

## 10. Fail-closed KAT matrix

Every row below requires at least one positive known-answer vector and the
listed negative family. Passing ordinary unit tests without these vectors does
not admit construction or live execution.

| Attack family | Required fail-closed rejection |
|---|---|
| Schema/encoding | unknown, missing, duplicate, aliased, reordered-semantic, noncanonical RFC 8785, trailing, Unicode, number, overflow, or non-finite input |
| Transitive loader rebinding | fully readdressed, re-signed and OOB-pinned wrong schema/domain/prefix/QV2/role/service/purpose/bound/regex/canonical/backend graph after ordinary module-global replacement |
| Stage-B trust | lock ID/SHA copied from its payload, missing or wrong out-of-band pin, changed exact profile/registry bytes, QV2 `/9`/dependency-lock drift, or rebuilt Stage-A image |
| CAS bootstrap | caller-supplied verified-artifact token, private `_issue` token laundering, non-`bytes` lock/profile/registry input, missing or wrong out-of-band lock ID/payload-SHA pin, or cross-graph/unsigned profile-registry splice |
| Registry substitution | attacker-chosen registry, untrusted governance anchor, post-output key addition, altered registry epoch or unsigned registry |
| Role collapse | duplicate public key, key ID, purpose ID, control-domain ID, container/state volume, or forbidden role combination |
| Signature context | injected/missing verifier backend, fake Ed25519, wrong role, purpose, domain separator, object type, epoch, profile, attempt, branch, or revoked key |
| Signed-object mutation | any field, nested byte, list order, digest, size, or semantic-role change after signature |
| R01 head | promote sequence-zero candidate as current, omit governance ledger/CAS/time, stale active entry, sequence gap/fork/equivocation, revoked/superseded head, or wrong authorization bytes/ID |
| Current-cut mechanics | non-raw or copied OOB authority pins, zero/wrong-size nonce, wrong request domain/ID/hash, non-UNIX/non-seqpacket/unconnected socket, ancillary/truncated/addressed packet, short send, wrong or changing kernel peer credentials/socket inode, changed head across the optimistic three reads, forged producer projection, genesis/multi-event/unrelated pre-cut time replay, or promotion of a response into atomic/current admission |
| Owner-status mechanics | drift any exact owner byte or any of fourteen bindings, use the current one-purpose governance registry, forge/alter the governance or logical-time signature, promote a CAS candidate reference, splice request ID/nonce/prior token/sequence, restore after terminal REVOKED, hide a crash gap, clone the ledger, or consistently roll back its suffix |
| External-witness mechanics | decoded/inferred authority or policy pin, witness/authority key-ID/control-domain collision, wrong private key, JSON boolean masquerading as an integer, wrong PURE role/purpose/body/attempt/policy, receipt-ID/signature cycle or wrong one-leaf root, partial/extra/gapped/symlink/hardlink/mode/owner spool, non-idempotent retry, competing initial leaf in one spool, copied/rolled-back spool, or promotion of local fsync/signatures into external control, durability or anti-fork authority |
| External-witness admission transcript | decoded or unpinned one of the three policies, bootstrap/W/custody/A/authority key or control collision, non-genesis A high-water, wrong W map root/path/state, Q1/A/Q2 ID/digest/nonce/mapping/outcome splice, transcript/time content-ID cycle, incomplete or foreign same-ledger ancestry, non-anchor penultimate token/checkpoint/head, wrong transcript-bound final event, altered `14 + N` signature scope, any true runtime/custody/live-UID/current/durable/anti-fork/authority/R01 claim, or promotion of a presented transcript into real admission |
| Local W/A actor integration | replace A's directory after construction, change its device/inode, reuse a partial or poisoned slot, re-sign a conflicting PREPARED W request without poisoning, poison W with malformed pre-fence input, destroy COMPLETE on a nonexact retry, reobserve/re-sign an exact retry, swap the common authority triple or current mapping, splice a valid cross-session Q2, detach a same-attempt logical-time genesis fork, or promote the 148/148 presented path into live/durable/custody/authority evidence |
| R02 branch | promote declared-equality candidate, omit admitted real R01 or independent branch event, actual/mixed/ambiguous branch, wrong logical-contract binding, or actual ID allocation |
| Source/runtime splice | wrong commit/archive/lock, rebuilt image, altered layer/config, late import, mapped-library drift, executable splice, or incomplete closure |
| CAS publication | overwrite, delete, missing/extra object, stale/rollback/forked checkpoint, equivocation, hash/size mismatch, temp residue, or false fsync/readback claim |
| Descriptor retention | pathname substitution, symlink, non-regular FD, reopened/mutated inode, truncation, wrong request/attempt/session/role/name, caller ACK treated as signed CAS authority, or cleanup before durable receipt |
| Worker set | mutate, reorder, drop, duplicate, exchange mode/target, inject thirteenth object, or mismatch reconstructed and original bytes |
| Full results | promote statistical subprojection as retention/full receipt, mutate/reorder/drop/duplicate chunks or rows, range overlap/gap, row-projection/chunk-ref mismatch, incomplete total, or decode ambiguity |
| Logical time | replay, rollback, gap, fork, cross-purpose, cross-attempt, wrong request digest, dependency inversion, or post-terminal issuance |
| Receipt dependency | R03 without exact R02, R04 without or exchanging R03, R05 without or exchanging R04/R03, cross-attempt/profile/fixture roots |
| Historical promotion | any attempt to use r12/r13 mechanics, reconstructed artifacts, or a new signature as the fresh Step-5 R03/R04/R05 receipt |
| Scientific direction | null false rejection, leak false acceptance, missing binding witness, altered threshold/permutation count, partial target family, or unsupported `standard_error` |
| Pure replay | producer import/call, producer cache/path trust, incomplete recomputation, output/root/statistic divergence, or evidence mutation |
| Crash/barrier | crash before object fsync, directory fsync, store ack, checkpoint, receipt, overlay, terminal, or cleanup confirmation |
| Concurrency/one-shot | two publishers, two terminal links, duplicate event sequence, retry in place, late writer, or alternate success/failure head |
| Timeout mutation | in-place 3,600-to-7,200/10,800 relaxation, changed reserve/total without new profile, or historical-attempt reinterpretation |
| Secret leakage | private key in public bytes, logs, argv, environment, stdin, repository, CAS, wrong volume, or cross-purpose mount |

Each negative must be exercised at the earliest accepting boundary. A later
generic failure is insufficient when an earlier authority accepted the bad
object. Tests must also show that a rejected producer cannot leave a reusable
success receipt or readiness transition.

## 11. Construction, static-review, and live gates

### 11.1 Construction gate

Construction is permitted while readiness remains `HOLD`. It may add pure
schemas, strict validators, source-pinned retention interposition, actor
workers, Dockerfiles, build scripts, profiles, runbooks, and KAT fixtures. It
must not create live authority keys, allocate actual-unsealed IDs, run the
1,024-row authority ceremony, or claim any readiness transition.

Construction closes only when the source tree has one authoritative schema
surface, no legacy scaffold can issue a Step-5 receipt, and every prospective
claim is paired with a verifier predicate and negative vector.

### 11.2 Static-review gate

Static admission requires, before any long live run:

1. exact schema/keyset/domain/signature-preimage review;
2. complete KAT matrix pass, including crash and concurrency barriers;
3. Stage-A image builds and immutable digest/source/archive/lock capture;
4. exact-byte Stage-B trust-lock replay from independently provisioned lock-ID
   and payload-SHA pins, followed by bytes-only candidate profile/registry
   replay with the real Ed25519 backend;
5. separate producer and pure-replayer implementations and images;
6. a fixed source-pinned `SCM_RIGHTS` client, signed CAS ACK validator, and
   proof that exactly twelve original workers and every authority-required
   result chunk cross that store boundary before cleanup;
7. runtime import/native-closure allowlist review;
8. a fresh concrete authority-v2 profile/registry/trust-lock graph under the
   frozen two-purpose governance policy, binding a fifth `governance_status`
   image under the existing governance role; integrated generic CAS v2 rather
   than the exact-twelve worker store;
   a dedicated owner-status logical-time stream plus current-cut token;
   externally anchored complete-event-set root; producer-free replay; and a
   real-R01 verifier/only-R01 overlay transition rather than either current
   candidate; independently controlled online witness W and durable anchor A
   under three raw/OOB-pinned policies; live Q1/A/Q2 current-map re-observation
   with an exact-high-water atomic durable CAS; a complete same-ledger post-Q2
   logical-time append; and a newly frozen exact retention plan whose lower
   bound is `26 + 8N` owner-event objects with every concrete closure blob
   counted separately; and
9. clean-tree/hash review that distinguishes pre-existing unrelated files and
   records the exact Step-5 construction basis.

Source-level static review first uses fixture keys and a KAT-only registry.
Only after that review passes may a pre-live ceremony generate the eight live
purpose keys and construct a new out-of-band-pinned Stage-B graph, then rerun
the exact-byte trust-lock/profile/registry static checks. The current candidate
profile/registry/R01/R02 graph cannot be promoted into that live graph by
renaming. The ceremony still may not start the 1,024-row qualification. Live
private keys and the KAT fixture keys must never share a key ID, public key,
registry epoch, or purpose volume.

A reviewer finding marked `HOLD`, any schema drift, any unclassified closure,
or any incomplete KAT keeps live execution prohibited.

### 11.3 Live gate

Live execution is unavailable in the current foundation revision. A future
live gate requires a new attempt context, fresh purpose keys under an
independently pinned Stage-B graph, qualified distinct technical actors, a
real R01 from the governance ledger/CAS/logical-time chain, a real R02 from an
independently observed synthetic branch event, and adequate
host/cgroup/deadline preflight. The run then executes R03, R04, and R05 in
order, uses the fixed source-pinned `SCM_RIGHTS` client, verifies signed CAS
ACKs, retains originals and complete results, publishes CAS and logical-time
evidence, performs producer-free full replay, verifies cleanup and residue
state, and publishes at most one terminal.

Only an independently replayed success terminal may set R01--R05 true in the
new overlay. Failure at any step preserves all affected and downstream nodes
as false. A successful Step-5 live gate stops at R05 and reports only that R06
and R07 are eligible to begin in parallel.

## 12. Permanent nonclaims

This document and any construction-only artifact permanently do not establish:

```text
R01_passed
R02_passed
R03_passed
R04_passed
R05_passed
step5_implemented
step5_executed
governance_status_ledger_completed_and_admitted
governance_status_purpose_authorized_by_current_pinned_registry
owner_signature_authenticated_for_owner_status
owner_status_store_dependencies_verified
owner_status_latest_authority_status_verified
owner_status_external_anti_rollback_or_anti_clone_anchor_verified
owner_status_fifth_image_live_closure_verified
generic_cas_v2_live_service_deployed_and_admitted
generic_cas_v2_complete_session_admitted_as_authority
owner_status_current_cut_logical_time_token_verified
owner_status_pure_replay_admitted_for_real_r01
owner_status_replay_core_admitted_as_authority
r01_evidence_contract_candidate_admitted_as_authority
r01_current_cut_mechanics_admitted_as_currentness
r01_current_cut_owner_time_atomic_fence_verified
r01_candidate_external_witness_receipt_admitted
r01_external_witness_local_spool_admitted_as_external_control
r01_external_witness_local_spool_host_rollback_or_clone_resistant
r01_external_witness_local_spool_admitted_as_durable_authority
r01_external_witness_admission_transcript_candidate_admitted
r01_external_witness_local_a_actor_admitted_as_external_durable_anchor
r01_external_witness_online_w_actor_admitted_as_live_or_restart_safe
r01_external_witness_public_w_a_w_integration_admitted_as_real_r01
r01_external_witness_live_current_map_verified
r01_external_witness_atomic_durable_anchor_cas_verified
r01_external_witness_runtime_custody_distinct_live_uid_gid_verified
r01_live_runtime_overlay_v2_schema_frozen
r01_live_runtime_overlay_v2_implemented
r01_candidate_live_runtime_closure_reports_admitted
r01_only_readiness_overlay_issued
cas_store_authority_service_deployed_and_admitted
logical_time_authority_service_deployed_and_admitted
fixed_source_pinned_scm_rights_client_integrated_into_actual_step5
signed_cas_ack_verified_for_actual_step5_attempt
readiness_evidence_overlay_issued
organizational_independence
independent_human_actors
malicious_admin_resistance
remote_attestation
trusted_utc_time
actual_unsealed_960_generated
actual_corpus_audited
recognizer_or_baseline_inference_admitted
commitment_reveal_or_scoring_admitted
```

Those facts, if ever established, require their own fresh, signed, durable,
producer-free-verifiable artifacts under the frozen readiness order.
