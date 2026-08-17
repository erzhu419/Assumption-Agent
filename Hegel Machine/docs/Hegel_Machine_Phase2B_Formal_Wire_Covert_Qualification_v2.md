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

The resource plan has a closed key set partitioned into frozen static fields
and recorded dynamic fields. Retained comparison exact-compares only the
frozen fixture-dependent fields, but requires all dynamic fields, nested run
measurements, exact types, admissibility, and caps. Environmental RSS, CPU
capacity, worker count, wall time, runtime seed, p-values, and worker hashes
are recorded rather than mistaken for cross-revision fixture identity.

Both auditor receipts are projected, validated, and released in their issuing
process. Repeated candidate success does not consume the bounded auditor
receipt registry. No receipt DTO or live full-result object crosses the inner
process boundary.

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

The systemctl transcript prevents accidental name collision and records
cooperative ownership. It is not an adversarial same-UID integrity sandbox.
The actual qualification prerequisite is a fresh dedicated user and
user-service-manager session with no concurrent same-UID unit mutation. The
local process-group cleanup additionally assumes the pinned cooperative
direct child creates no orphan descendants; it does not claim an adversarial
PID/PGID ownership primitive.

The end-to-end watchdog is 28,800 seconds. Normal work stops with a 3,000-second
reserve: 2,400 seconds are reserved for two bounded retained-tree scans,
metadata, fsync, verification, and terminal publication; 600 seconds are
reserved for cleanup control. No success may extend the deadline. Command
capture and process reaping use subordinate deadlines inside that bound.

## Retained transaction, private CAS, and terminal

Reservation holds the output-root and attempt directory descriptors and uses
openat/no-follow operations throughout. Inputs, extracted source tree, raw RFC
bundle, evidence, and attempt-private CAS are retained under that inode.
Archive inspection binds every regular file, explicit empty directory, and
implicit ancestor to the exact extracted full-tree manifest; path/type
conflicts, symlinks, devices, and extras are rejected.

Every successful or closed-failure attempt has exactly one O_EXCL
`terminal.json`. A prospective pure verifier replays the retained directory,
CAS roles, archive/tree relation, inner/outer semantics, mechanics, and exact
tree before success becomes visible. The commit is followed only by bounded
same-descriptor readback/fsync and parent-entry inode identity checks.

If a nonterminal atomic link commits but cleanup/readback becomes uncertain,
the outer transaction reconciles the retained target and managed temporary.
Unremovable owned residue is not ignored: the negative terminal's closed
`managed_atomic_residues` rows bind its exact path, state, inode/link relation,
hash, and size. Unknown temporaries or same-byte/different-inode splices are
rejected. Terminal-link uncertainty itself forbids a second terminal.

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
