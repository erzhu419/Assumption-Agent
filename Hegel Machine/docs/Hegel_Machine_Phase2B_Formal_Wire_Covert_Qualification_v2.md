# Hegel Machine Phase-2B Formal Wire / Covert Qualification V2

## Current disposition

This revision is a source-and-profile candidate. Its checked status is
`CANDIDATE_SOURCE_FROZEN_FULL_1024_SHAPE_CALIBRATION_PENDING`, and
`live_qualification_enabled=false`. The ordinary candidate entry and the
live-inner entry fail closed before a 1,024-row audit. The development entry is
the only route through the pending-profile gate.

Task 4 can publish only transaction and mechanics evidence with disposition
`MECHANICS_VERIFIED_NOT_AUTHORITY`. It never sets R03, R04, or R05. Host tests,
an outer success terminal, a content hash, a CAS object, or a later signature
cannot promote a Task-4 artifact into any of those readiness nodes.

The old identity-only values 16,863 concrete features, 24,367 raw semantic
features, 7,991 partitions, and 84,315 hypotheses are not qualifications for
this fixture and are intentionally absent from the profile.

## Frozen dependencies and source bootstrap

The canonical dependency lock pins the final Wire and auditor source, config,
test, and documentation bytes, the exact-transform core, RFC-8785 source/test/
documentation/vectors/Node oracle, and the seven project modules actually
compiled by the source-only loader:

- `hashing.py`;
- `phase2b_exact_transform_core_v1.py`;
- `phase2b_formal_covert_auditor_v1.py`;
- `phase2b_formal_trusted_wire_v3.py`;
- `phase2b_freeze_v1.py`;
- `rfc8785_v1.py`; and
- `schema.py`.

The checked canonical profile ID is
`phase2b_formal_wire_covert_qualification_profile_v2_c81a345cadb799976b61ca665df7a898436254eb4dc85fa4922f9f880c1ae3ea`,
and its dependency-lock SHA-256 is
`7f524eb9c111315f010efc27d3e73d508da33ebfe52c5b838b35e11dcf25ee8c`.
The dependency-lock root schema is
`hegel-machine-phase2b-formal-wire-covert-qualification-dependency-lock/3`,
and that exact discriminator is also closed inside the profile's lock binding.
The qualification profile schema is
`hegel-machine-phase2b-formal-wire-covert-qualification/4`; the profile domain
and ID prefix remain V2 because the `/4` discriminator is part of the hashed
canonical content. The development-calibration and live-inner candidate
schemas are `/4`. Their prior `/3` forms already contained the receipt-build
elapsed field; `/4` is required because the natural STOP was followed by
changes to the nested complete-plan, observed/invariant/static-shape roots, and
their replay grammar. The prior `/3` identities are not reused.
The pinned Auditor contract is
`hegel-machine-phase2b-formal-covert-auditor/4`, its profile schema is
`hegel-machine-phase2b-formal-covert-auditor-profile/4`, semantics are
`phase2b_formal_covert_semantics_627506ef33be3134ee81c2a8bd765e4e7898abd7da9c94660bfc82176b0cbc1f`,
and policy binding
`phase2b_formal_covert_policy_binding_19c00ee7ccd65d0d11146aab1a1b6b5007a1d5e1cfd97599b26bfb7feb9f2825`.

A direct qualified CLI invocation must begin with CPython `-I -S -B`; missing
initial flags fail before any project import, business input, output, or
service launch. That small launcher no-follow descriptor-reads the current
main source, combines its measured SHA-256 with the seven fixed dependency
pins, and execs the external `-I -S -B -c <bootstrap>`. The external bootstrap
opens the main and all seven imported project modules once, verifies the exact
manifest, and compiles those same snapshots. A source-only loader consumes
only those preopened bytes. For attempt-bearing entries, the bootstrapped CLI
then requires the executed-main SHA and profile SHA to equal the frozen
attempt before dispatch; the outer runner repeats the executed-source check
before reservation. This is cooperative executed-byte provenance, not
independent source authentication. The broad `hegel_machine.__init__` is
bypassed; its recorded hash is forensic and explicitly non-TCB.
The selected interpreter path, its initial standard-library imports, and the
host loader remain a cooperative immutable-path TCB until Step 5 retains a
full authenticated runtime closure.

`standard_library_only=true` has the narrow scope
`PYTHON_THIRD_PARTY_PACKAGES_ONLY`. It says that no third-party Python package
is in the locked Python import set. It does not claim byte closure of
libpython, the standard library, extension modules, hashlib/native libraries,
or the host loader. The runtime manifest is a cooperative host fingerprint,
not an immutable runtime image or full runtime closure; therefore
`runtime_image_digest=null`.

## Locked synthetic fixture

The development measurement and any later fresh qualification use exactly
1,024 rows. The private label design is:

```text
2 case types x 2 families x 2 scales x 2 bindings
x 2 answerability labels x 2 joint-decision classes x 16 repetitions
```

Every one of the 64 label cells occurs 16 times. Repetition is crossed with
the eight typed exact-transform operations, so every cell contains each
operation twice and each operation occurs exactly 128 times:

- identity;
- unit conversion;
- coordinate affine;
- temporal aggregation;
- spatial aggregation;
- sampling resolution;
- equivalent split/merge; and
- coarse graining.

Every row is a real `FormalCaseGraphV3`, populates all ten UUID namespaces,
passes the exact-transform-core projection, and binds one shared catalog with
one UUID-bearing exact source-transform use per row across all eight
operations. This remains a fixture-specific two-class
development qualification. It does not qualify arbitrary Wire V3 graphs, the
auditor's separate 100-class resource KAT, or an actual corpus.

The qv2-local fixture identity is
`CAPACITY_BOUNDED_EXACT_10_EIGHT_OPERATION_FORMAL_CASE_GRAPH_V4`. It is a new,
strictly narrower profile, not a lossless rename of the former V3 fixture. The
coordinate-affine witness is the one-dimensional, non-identity, frame-changing
map `target_0 = 2 * source_0 + 1` on source support `[0,1]`, with exact inverse
`source_0 = (target_0 - 1) / 2` and target support `[1,3]`. The temporal
sampling witness has exactly two grid-point inputs, selects the point at zero,
discards the point at one, and copies the selected metadata exactly to its one
output. Exact-core tests outside this Task-4 fixture continue to cover the
two-dimensional off-diagonal affine case and sampling shapes with more than
one selected input; Task 4 V4 does not claim those shapes as fixture coverage.

The former V3 fixture's concrete union-path projection was 38,168 for
`KNOWN_NULL` and 37,866 for `KNOWN_BINDING_LEAK`, so it exceeded the frozen
Auditor limit of 32,768 before feature materialization. V4 retains all eight
operation kinds and the complete label crossing while reducing only those two
operation witnesses. Its exact saturated counts are 32,337 concrete / 50,257
raw semantic features for `KNOWN_NULL`, and 32,186 / 49,994 for
`KNOWN_BINDING_LEAK`. Eight rows already contain all operation path shapes;
the same union shape is repeated at 32 and 1,024 rows. The independently
decoded union of `channel_only_leaf_owners` paths is exactly 229 for
`KNOWN_NULL` and 228 for `KNOWN_BINDING_LEAK`; each mode's count is frozen in
the V4 feature-capacity contract and stored on the fixture execution. Its
canonical path-to-owner rows are ordered by the Auditor JSON-path text and
owner group, and their mode-specific union root is frozen beside the count;
conflicting owners for one path are rejected. Construction also retains the
small complete feature-cardinality projection DTO (including its projection
root) after deleting the build-local decoded tuple. The
pinned Auditor `FeatureManifestV1` has no owner-union-count field, so receipt
validation reads only its real UUID/channel/concrete/raw fields, while the
six-field receipt-capacity projection must equal both that independently
decoded execution projection and the frozen contract. The owner count and root
flow from the source-derived execution through audit evidence and the retained
static/inner projections. The complete feature-capacity projection and its
root follow the same path; pure replay reconstructs the projection DTO, checks
every closed field and recomputes its root before accepting the static shape.
It then re-derives audited rows, channel/UUID paths, concrete/raw counts, and
the pre-extraction byte bound, checks them through the frozen receipt-capacity
validator, and separately replays the mode-specific owner count/root contract;
synchronizing a forged measurement and its root cannot bypass these checks.
These are static construction/capacity
facts, not filled calibration statistics: both
`known_*_frozen_static_projection` fields remain null until a successful fresh
development calibration.

Before either formal 1,024-row mode is launched, its separate slow
resource-only gate builds and decodes the real V4 batch, replays the exact
feature-capacity projection, and constructs the invariant-phase plan from the
measured envelope/decode graph and pinned adapter bounds. It invokes the same
invariant resource evaluator at this resource-only point, but it does not
construct the invariant fork snapshot, signal/child ownership state, or
artifact directory and therefore is not final fork-boundary evidence. It then
repeats the feature-allocation preflight, materializes feature columns and
partitions under one partition-preparation deadline, drops the decoded and
column aliases, and calls the production statistics-plan constructor. That
constructor performs the sole deep traversal of the materialized partition
graph and takes a fresh statistics parent-RSS baseline. The gate validates the
resource-only invariant sample against the composite plan, invokes the same
statistics resource evaluator, and validates that resource-only sample against
the same plan. It does not construct the production label/permutation masks,
worker ownership state, or artifact directory, so this second sample is not
the statistics fork-boundary evidence either.

The gate executes no invariance batches, statistical permutations, receipt
construction, PENDING registration, pending replay, ISSUED transition, public
validation, release, or official attempt publication. It proves that the
Auditor receipt registry and reservation set are unchanged. It has no CLI
artifact or pure-verifier authority and is only a non-final
preallocation/admissibility smoke over the real 1,024-row object graph and
frozen formulas. The two modes are independently built and freshly decoded.
Operators should run them in the intended host/cgroup containment, but that
placement is separately observed rather than authenticated by the test. Only
the subsequent formal audit records the fully prepared, final fork-boundary
preflights and issuance linearization. Passing this gate is not an audit
disposition or a replacement for the full run.

Fixture construction uses the pinned Auditor decode-memory preflight before
its first row decode. Every returned row must be the exact decoded DTO, pass
the pinned semantic validator, and remain within the conservative 4 MiB
per-row decoded-object-graph bound before it can enter any namespace, owner, or
feature-capacity projection.

## Known-null, known-leak, and visible statistical replay

The known-null graph exposes every legal observation in the eligibility mask.
The known-leak graph differs only at
`public_case.graph.eligibility_mask.observation_ids`: binding A exposes the
canonical first legal observation and binding B exposes the canonical first
two. Role-candidate IDs and scale IDs remain byte-for-byte unchanged. Replacing
only those nested observation IDs reconstructs the exact paired null case.

The decision family is five target-level maxT p-values followed by exact Holm
adjustment at family-wise alpha 0.01. Retained verification recomputes from the
visible rows:

- raw target p = `(exceedance_count + 1) / 10001`;
- deterministic target order and Holm rank;
- step-down adjusted p, critical value, and rejection polarity; and
- the binding E1 witness link, including its feature maxT arithmetic and
  same-feature localization comparison.

Known leak must have exactly one visible binding E1 witness with finite
omnibus statistic greater than one. Pairwise Holm remains diagnostic only and
has `pairwise_holm_decision_eligible=false`.

The Task-4 primitive projection does not retain the full result matrix,
permutation workers, or all worker artifacts. Consequently its pure directory
verifier does not independently recompute the complete statistics. It verifies
the closed visible projection and binds it to the pinned cooperative child
code. A known-null PASS can repeat only
`NO_MATERIAL_COVERT_CHANNEL_DETECTED_AT_TARGET_FAMILY_FWER_0.01`; it must not
be paraphrased as a universal absence-of-leakage claim.

## Resource and Wire execution contract

Each audit runs one baseline plus the atomic 80-schedule invariance suite: 32
global renamings, 32 case-order permutations, and 16 observation-order
permutations. The consumer performs exactly `rows x 80` full Wire decodes and
compares five projected fields per row. The lightweight recognizer is accepted
only after exact equivalence to a trusted full decode for base and transformed
envelopes.

The composite `ResourcePlanV1` has a closed key set partitioned into frozen
static fields and recorded dynamic fields. Its dynamic side includes both
phase baselines, the invariant-plan root, the derived statistics-plan root,
phase peak estimates, CPU/worker selection, and admissibility. Retained
comparison exact-compares only the frozen fixture-dependent fields, but still
requires every dynamic field and reconstructs the complete DTO so its derived
statistics root must match.

Invariant and statistical runtime evidence each retain an exact nested
phase-preflight DTO. The invariant DTO is bound to
`invariance_resource_plan_root`; the statistical DTO is bound to
`resource_plan_root`. Receipt validation calls the pinned Auditor's two
cross-stage validators against the composite plan.

Every receipt also carries the branch-neutral
`ReceiptIssuanceRuntimeEvidenceV1`. For an executed QV2 audit its deadline kind
is `STATISTICS`: `W` is the exact sum of wait4 worker peaks,
`C = 12 GiB - W`, the recorded pre-finalization parent peak must not exceed
`C`, and the conservative charge is exactly `C + W = 12 GiB`. The evidence
records exactly one producer PENDING full replay, exactly one deadline check
under the receipt-registry lock, a successful locked deadline verdict, and a
successful locked post-replay parent-peak ceiling verdict. Auditor `/4` also
defines the insufficient branch as `PARTITION_PREPARATION`, `W=0`, and
`C=12 GiB`; QV2's two full modes require statistical evidence and therefore do
not use that branch.

The inner projection keeps the complete plan and all three complete runtime
DTOs. One shared executed-runtime-bundle evaluator is used before producer
projection and by run/measurement, pure retained, two-run, and candidate
validation. It reconstructs the 56-field plan, then the 20-field statistical
DTO against that plan, then the 11-field issuance DTO. Every invocation binds
the expected branch mode, exact 1,024 authority rows, and the mode-specific
frozen concrete-feature count; candidate issuance also binds those values to
the live Auditor receipt and manifest before release. For full run projections
it additionally cross-binds visible row/feature counts, invariant resident
bounds and adapter pin, compact/full invariant counts, roots, artifact digest,
peak, wall time and cap, statistical worker/artifact cardinalities and resource
bounds, compact/full statistical digest/peaks/wall/cap/charge/supervisor fields,
diagnostics, permutation commit/reveal and root grammar, the five-target Holm
family, and the `5 * concrete_features` pairwise family. The exact Auditor
transcript-ID grammar is enforced and the audit, issuance, run, measurement,
and candidate copies must agree. The eligibility-mask quota instead has its
own `phase2b_formal_eligibility_mask_quota_<sha256>` ID grammar; it is included
in the closed quota projection and quota root and must not equal the top-level
audit transcript by construction. This reduced QV2 projection does not retain
the Auditor's signature schema, public signature root, target cardinalities,
or private contingency roots, so detached QV2 replay does not claim to rederive
the Auditor quota content ID; it validates that ID's own grammar and fully
cross-binds the retained projection bytes to the QV2 quota root. Environmental
values remain recorded evidence, not cross-revision fixture identity.

Each Auditor receipt receives exactly one public full validation in its issuing
process. QV2 captures that validated ID, projects and pure-replays the complete
issuance evidence without another public validation or top-level receipt
`__post_init__`, and then releases the exact ISSUED receipt while checking the
returned ID. Repeated candidate success does not consume the bounded Auditor
receipt registry. No receipt DTO or live full-result object crosses the inner
process boundary. Each candidate branch retains the complete ResourcePlan
projection and root before its complete statistical and issuance projections,
and detached replay now requires the reconstructed plan's `admissible` field to
be exactly true. Because that semantic gate was added after the V4 natural
STOP, the local candidate receipt uses the V5 receipt hash domain and
`phase2b_formal_wire_covert_qualification_receipt_v5_` prefix. Complete and
observed ResourcePlan roots, complete invariant runtime roots, and static-shape
roots use V3 prefixes; the static-shape schema is `/3`. Statistical and issuance
runtime roots remain V3, while the unchanged static ResourcePlan projection
continues to use its V2 prefix.

## Pending shape and two-revision calibration

The checked `rich_positive_shape` is deliberately pending. For each of the six
retained calibration roles—basis profile, attempt manifest, inner artifact,
outer supervisor result, RFC-8785 receipt, and mechanics artifact—the SHA-256,
size, and CAS reference are null. Campaign ID, attempt ID, attempt-manifest
root, combined fixture root, and both frozen static projections are also null.
No observed feature count, target result, p-value, RSS, or wall time may be
filled before a real development campaign is retained and reviewed.

The intended two-revision transition is:

1. revision P0 runs a pending development-calibration attempt and retains its
   complete attempt directory;
2. a reviewer copies only the approved static projection into revision P1 and
   binds P0's canonical basis-profile bytes, IDs, six role objects, terminal,
   tree projection, and CAS objects;
3. P1 performs a fresh live attempt with distinct campaign and attempt IDs;
4. the live verifier recursively replays the paired P0 development directory
   at depth one, while depth zero remains current-profile-only.

The calibration and live attempt directories are a jointly retained pair.
The pair projection binds the old terminal hash, attempt root, exact retained
tree root/count/bytes, and `pair_retention_required=true`. Step 5 must bind the
pair as provenance in evidence for its new authoritative run; the pair is not
a promotion path or a substitute for that run. Moving, deleting, or replacing
either directory makes the live mechanics evidence unreplayable. Six loose
historical paths are not a durable substitute.

## Exact attempt manifest

The reservation freezes the exact top-level attempt projection before fixture
construction:

- campaign ID, attempt ID, and profile ID;
- source commit SHA, source-archive root, and archive size; the TAR global PAX
  header must be exactly `comment=<source-commit-sha>`;
- dependency-lock SHA and the nested qualification source/config/lock/test/doc
  SHA map;
- runtime-manifest root and size, with null runtime-image digest;
- RFC-8785 receipt root, SHA, and size;
- mode order and the 28,800-second hard total; and
- requested cgroup memory and pids maxima.

The five qualification-file hashes are attempt inputs. They are not written
back into the profile, lock, or documentation, which avoids a self-hash cycle.
The filesystem attempt directory is a short content-addressed name derived
from the manifest root; display campaign/attempt IDs remain inside the
reservation and terminal.

## Qualified CLI construction sequence

All paths below are absolute. Long options are unabbreviated, occur exactly
once, and use a separate value token; every entry rejects every unrelated
option. Successful builder stdout is exactly one canonical RFC-8785 artifact
with one LF and no diagnostic prefix. A direct source invocation is:

```text
<python> -I -S -B <qualification.py> --entry <entry> ...
```

First snapshot the cooperative host fingerprint:

```text
<python> -I -S -B <qualification.py> \
  --entry build-runtime-environment-manifest \
  --project-root <project-root> > <runtime-manifest.json>
```

Build the RFC prerequisite into a fresh directory; the attempt builder accepts
only `<rfc-bundle>/receipt.json` and replays that receipt's entire closed raw
bundle against the current source tree before emitting an attempt:

```text
<python> -I -S -B <qualification.py> \
  --entry build-rfc8785-prerequisite \
  --project-root <project-root> --output-root <fresh-rfc-bundle>
```

Create the source TAR with `git archive --format=tar <source-commit> --
<exact-23-pinned-paths>`. Git's global PAX `comment` must equal the supplied
40-hex commit. The builder and outer extraction verify that metadata, every
pinned archive file, all five current qualification-file hashes, the runtime
manifest, RFC receipt, and raw bundle before output or service launch:

```text
<python> -I -S -B <qualification.py> \
  --entry build-development-attempt-manifest \
  --project-root <project-root> \
  --campaign-id <campaign> --attempt-id <attempt> \
  --source-commit-sha <40-hex> --source-archive <source.tar> \
  --runtime-environment-manifest <runtime-manifest.json> \
  --rfc8785-prerequisite-receipt <rfc-bundle>/receipt.json \
  --cgroup-memory-max-bytes <finite-bytes> --cgroup-pids-max <finite-count> \
  > <attempt.json>
```

The PAX comment is an exact archive-to-asserted-commit binding; it is not a
signature or an independent proof that a cooperative producer used a
particular Git object. A later authority must authenticate the source archive.
The pending development outer entry then consumes the checked-in profile, the
new attempt, that same source TAR/runtime manifest/RFC bundle, and a fresh
output root.

## RFC-8785 prerequisite

The prerequisite builder and replay verifier bind canonical receipt bytes to:

- the source-only 18/18 KAT set;
- open-once sealed test, vector, and Node-oracle bytes;
- one Node vector check;
- exactly 50 deterministic `--number-bits` batches of 2,000 lowercase
  16-hex-bit patterns each (100,000 total); and
- exact fresh-replay stdout, argv, ordering, dependencies, and raw bundle.

Fresh replay relocates only the closed source-root positions to the retained
extracted archive. It does not reopen the builder's original worktree.

## External live supervisor and threat boundary

The child runs as a fresh user-systemd transient service, not a scope. The
launcher shape is exactly:

```text
systemd-run --user --pipe --wait --quiet --unit=<fresh-unit> \
  --property=Description=<attempt-bound-marker> \
  --property=Type=exec --property=RemainAfterExit=yes \
  --property=MemoryMax=<finite-bytes> \
  --property=TasksMax=<finite-count> \
  --property=RuntimeMaxSec=28800s \
  --property=KillMode=control-group \
  --property=KillSignal=SIGTERM --property=SendSIGKILL=yes \
  --property=FinalKillSignal=SIGKILL --property=TimeoutStopSec=30s \
  --working-directory=<retained-source-tree> \
  --setenv=LANG=C.UTF-8 --setenv=LC_ALL=C.UTF-8 -- \
  <direct-python> -I -S -B -c <open-once-bootstrap> ...
```

There is no `--collect`. `RemainAfterExit=yes` keeps the exited unit loaded so
the outer process can snapshot exact terminal properties. It then issues
`systemctl stop`, verifies the retained InvocationID/marker around control
actions, reaps the launcher, and proves exact `not-found` in bounded probes.
Failure cleanup is TERM, bounded grace, optional KILL, stop, reap, and bounded
absence. If cleanup cannot be proved, the only terminal is the explicit
`ATOMIC_FAILURE_CLEANUP_UNPROVEN_EXTERNAL_INTERVENTION_REQUIRED` negative
state; it is never ordinary closed success.

The current `/failure/4` cleanup command is exactly `systemctl --user kill
--kill-who=all --signal=SIGTERM|SIGKILL <unit>`, which is the systemd 249 CLI
shape. The retained `/failure/3` schema is replay-only and keeps its historical
command preimage; `/3` and `/4` field sets and command grammars are disjoint.
The producer never emits a new `/failure/3` terminal.

Before any failure-side TERM, KILL, or stop command, the outer process
atomically writes and fsyncs the stdout and stderr bytes already consumed from
the `systemd-run --pipe` relay. Cleanup then drains from the current pipe
offset into the same bounded bytearrays, so the final retained bytes equal the
prefix plus each tail byte exactly once. The two prefix files, two final raw
files, and one nonempty framed stdout/stderr bundle in the attempt-private CAS
are hash/size/path bound by `/failure/4` and replayed from retained dirfds.
Success still requires a nonempty canonical stdout artifact and exactly empty
stderr.

Every `/failure/4` also carries an exact failure-CAS snapshot. It distinguishes
an absent CAS directory from a present empty directory, lists every regular
self-addressed object in lexicographic digest order, rehashes each object from
an `O_NOFOLLOW` retained descriptor, and rejects any extra or missing object.
Only temporary names bound by `managed_atomic_residues` are excluded from that
object list. Partial failure-side CAS objects are explicitly unattributed
(`partial_cas_roles_attributed=false`); this exact set is not a claim that all
objects were produced by a particular causal step or have successful semantic
roles. An additional correctly self-addressed object is accepted only when it
is included in the terminal's exact CAS snapshot, and it remains unattributed.
When pipe-capture persistence is
verified, the framed pipe-bundle CAS row is separately cross-linked to its
specific capture role.

`capture_complete=true` means both pipe EOFs were observed, no tail remains,
and the launcher was waited/reaped under a verified cleanup projection. An
unproven cleanup may retain the actual prefix and any later bytes already
observed, but `/failure/4` has no positive launcher-reap fields with which a
pure replay could prove completeness. The producer therefore conservatively
uses `capture_complete=false` even if EOF/reap preceded a later ownership or
final-absence failure, and it does not claim a complete traceback. If prefix or
final capture reaches a stream cap, retained bytes stop exactly at that cap,
`capture_truncated_at_cap=true`, and the capture remains incomplete. If prefix
or final capture persistence itself is uncertain, cleanup still runs first and
the terminal uses the closed
`ATOMIC_FAILURE_PIPE_CAPTURE_PERSISTENCE_UNVERIFIED` negative disposition,
binding every retained partial raw artifact and any managed atomic residue.
Cleanup uncertainty has precedence: if cleanup and capture persistence are
both unproven, the terminal uses
`ATOMIC_FAILURE_CLEANUP_UNPROVEN_EXTERNAL_INTERVENTION_REQUIRED` while still
binding the persistence-unverified partial-artifact projection.
The terminal records `primary_failure_stage` and the primary exception type
separately from `cleanup_failure_exception_type`; cleanup failure cannot
overwrite an `INNER_EXECUTION` primary failure.

The systemctl transcript prevents accidental name collision and records
cooperative ownership. It is not an adversarial same-UID integrity sandbox.
The actual qualification prerequisite is a fresh dedicated user and
user-service-manager session with no concurrent same-UID unit mutation. The
local process-group cleanup additionally assumes the pinned cooperative
direct child creates no orphan descendants; it does not claim an adversarial
PID/PGID ownership primitive.

The end-to-end watchdog is 28,800 seconds. Normal work stops with a 3,000-second
reserve: 2,400 seconds are reserved for two bounded combined retained-manifest
scans,
metadata, fsync, verification, and terminal publication; 600 seconds are
reserved for cleanup control. No success may extend the deadline. Command
capture and process reaping use subordinate deadlines inside that bound. The
two-mode Auditor phase budget is exactly
`2 * (600 partition + 7,200 invariance + 3,600 statistics) = 22,800` seconds.
The 25,800-second normal-work window therefore retains exactly 3,000 seconds
for QV2 fixture/preparation/headroom, and the remaining 3,000 seconds are the
2,400-second terminal-publication plus 600-second cleanup-control reserve:
`22,800 + 3,000 + 3,000 = 28,800`. This deadline erratum changes neither the
pending calibration state nor the Task-4 projection-only/R03-R05-false boundary.
The
producer derives its failure tree, exact CAS snapshot, and managed-residue
binding in one fd-relative traversal; the prospective public verifier performs
one independent traversal across the trust boundary. Each traversal hashes a
regular inode at most once, including hard-linked target/temp aliases. No
producer digest cache is trusted by the verifier.

## Retained transaction, private CAS, and terminal

Reservation holds the output-root and attempt directory descriptors and uses
openat/no-follow operations throughout. Inputs, extracted source tree, raw RFC
bundle, evidence, and attempt-private CAS are retained under that inode.
The raw RFC destination directory is created, opened, and registered before
its first copied object is written, so pre-link and post-link uncertainty can
be reconciled from its retained parent descriptor without a directory-tree
search.
Archive inspection binds every regular file, explicit empty directory, and
implicit ancestor to the exact extracted full-tree manifest; path/type
conflicts, symlinks, devices, and extras are rejected.

Every successful or closed-failure attempt has exactly one O_EXCL
`terminal.json`. Before success becomes visible, a prospective pure verifier
replays the retained directory, successful semantic CAS roles, archive/tree
relation, inner/outer semantics, mechanics, and exact tree. A `/failure/4`
instead replays its exact self-addressed, explicitly unattributed partial CAS
set plus any separately cross-linked failure role. The commit is followed only
by bounded same-descriptor readback/fsync and parent-entry inode identity
checks.

If a nonterminal atomic link commits but cleanup/readback becomes uncertain,
the outer transaction reconciles the retained target and managed temporary
only through the required finite retained-parent descriptor registry; the
reconcile step performs no fallback tree traversal or content rehash.
Unremovable owned residue is not ignored: the negative terminal's closed
`managed_atomic_residues` rows bind its exact path, state, inode/link relation,
intended target hash, and observed size. In `evidence/cas`, the target name is
the exact lowercase intended SHA-256. A `STAGED_TEMP_ONLY` file may contain only
a partial prefix, so its actual retained-byte hash is bound by the tree row and
need not equal that intended target hash; a committed target/temp hard-link
must still hash exactly to it. Unknown temporaries or
same-byte/different-inode splices are rejected. Terminal-link uncertainty
itself forbids a second terminal.

Content deduplication applies to unique objects in the attempt-private CAS;
multiple semantic roles may reference the same digest. SHA-256 is a content
binding, not a signature or authority grant.

## Task-4 projection boundary and Step 5

Every inner, mechanics, and terminal success projection states exactly:

- the child projection was validated by pinned cooperative code;
- the pure verifier does not recompute all statistics;
- full results and worker artifacts are not retained;
- the artifact is not promotable to R04/R05;
- the host manifest is only a cooperative fingerprint;
- same-UID user-manager mutation is an outer TCB assumption;
- a fresh dedicated user/service-manager session is required;
- signature or CAS cannot promote the existing artifact; and
- Step 5 must rerun and retain full results, worker artifacts, the full runtime
  image, and the import/native closure.

Accordingly, Step 5 may consume Task-4 mechanics as diagnostic input, but it
must perform a new authoritative run. It may not merely sign or bless this
artifact.

## Rich-32 launch gate

Before any 1,024-row calibration, two separate rich-32 invariance-core tests
must run in the focused test process:

- `KNOWN_NULL`; and
- `KNOWN_BINDING_LEAK`.

Each must prove one baseline plus 80 schedules, exactly `32 x 80` transformed
consumer decodes, exact lightweight-versus-trusted projection equivalence for
every transformed envelope, and wall time no greater than 225 seconds. If
either mode exceeds 225 seconds, full 1,024 execution is prohibited. The gate
does not run 10,000 label permutations and never establishes R03, R04, or R05.
