# Hegel Machine Q0.5b Shadow Retry Campaign v2r1 (record schema `/2`)

Status: `V2R1_FINAL_BOUND_AWAITING_EXACT_OWNER_AUTHORIZATION_NOT_EXECUTED`

This document accompanies
`config/phase3_q05b_shadow_retry_campaign_v2.json`.  The unchanged filename and
all structural schema tags ending in `/2` describe the retained record format;
the distinct contract ID, hash domains, roots, and trace IDs identify the new
v2r1 campaign instance.  The controller source, tests, document, exact V2R1
hash-domain registry, roots, and trace IDs are now frozen for binding.  The 13
binding summary values remain null only until the main agent derives and fills
them in the next freeze step, so this intermediate byte image is not yet a
loadable contract, execution capsule, authorization record, preflight receipt,
or execution result.  The frozen target supervisor remains exact
Commit A `7bc2044212e278443b1ca703115c7cb17591ac14`; its tree, parent,
commit-object bytes, seven required source hashes, static-policy root, target
contract digest, and target source-binding digest are independently replayed
by the controller and are not revised by v2r1.  No preflight, authorization,
or campaign execution has occurred.

The prior v2 capsule
`sha256:238b79a5ddf5adb379df9ac1b0ae7e762f7616f1cfbf19b93474315a9197ab62`
and authorization
`sha256:0f59cc4aa2f605bdd1388baf769961ccfebf2123c6d6c9ff467f7f61add29cea`
belong only to the completed prior-v2 terminal/completion chain.  That chain is
historical, terminal, non-resumable evidence.  Neither digest, its immutable
ledger, nor any prior-v2 root is v2r1 authority or an input that v2r1 may open,
access, reuse, copy, adopt, or resume.

No source-bound contract or self-consistent capsule is execution authority.
The read-only preflight may derive one exact capsule and authorization
template, but no clone, Cargo copy, authorization root, ledger root, artifact,
evidence directory, trajectory, Docker process, or supervisor invocation may
be created until the project owner authorizes that exact capsule digest.

## 1. Purpose and claim boundary

The campaign is a bounded response to one narrowly classified infrastructure
failure: an otherwise clean Q0.5b shadow invocation that fails only because a
held-actor resource-sampling gap exceeds the frozen 250 ms maximum.  It offers
three fixed, strictly sequential, one-shot **slots**, not an unbounded retry
loop.  A slot can enable its successor only after every clause of the exact gap
predicate has been durably proven.  Every other failure is terminal.

Even a successful slot establishes only one nonadopting shadow observation of
the Q0.5b mechanics.  It does not establish efficacy, causality,
generalization, C1/C2, Meta adoption, promotion, current status, Hegel Q1, an
`ACTIVE` transition, production eligibility, or any Operations Research
claim.

The hard campaign limits are:

- exactly three registered slots: `SLOT_01`, `SLOT_02`, `SLOT_03`;
- at most three supervisor invocations across the entire campaign;
- at most one invocation per slot;
- no reuse of a consumed slot;
- no parallel preparation or overlapping execution;
- no dynamically selected path and no `SLOT_04`;
- no generic retry after timeout, crash, signal, cleanup error, malformed
  output, artifact ambiguity, trajectory ambiguity, or binding ambiguity.

## 2. Binding and future authorization

Commit A is frozen as one exact 40-hex commit and tree.  The final contract
binds the raw bytes of the v2 isolation config, admission core,
artifact core, trajectory core, dual test file, trajectory test file, and
supervisor tool.  It also binds the config static-policy root and the target
source-binding digest.  The controller recomputes every value from the local
Git object store without network access.  Before capsule derivation it also
runs a bounded, strict, full `git fsck` rooted at the exact target commit with
reflogs and dangling diagnostics excluded and requires empty stdout/stderr;
therefore deterministic missing or corrupt local-object closure fails before
owner authorization rather than consuming a campaign slot.

The new v2r1 controller commit implements and tests its campaign core and tool
at the unchanged schema-v2 paths:

- `src/hegel_machine/phase3_q05b_shadow_retry_campaign_v2.py`;
- `tests/test_phase3_q05b_shadow_retry_campaign_v2_core.py`;
- `tools/phase3_q05b_shadow_retry_campaign_v2.py`;
- `tests/test_phase3_q05b_shadow_retry_campaign_v2_tool.py`;
- `docs/Hegel_Machine_Q05b_Shadow_Retry_Campaign_v2.md`.

Their raw/static hashes and campaign source-binding digest are frozen in the
final contract.  The campaign additionally binds the raw hash of the v1 safety
kernel at
`src/hegel_machine/phase3_q05b_shadow_nonadoption_trial_v1.py`.  Final binding
also closes the document hash, exact predicate digests, ledger grammar, and
contract digest without changing the slot registry or authority boundary.

The config deliberately contains neither its own raw hash nor a source-binding
literal embedded in executable code.  It directly binds the live tool's full
raw SHA-256.  The acyclic freeze order is: finalize all campaign source bytes;
write their raw hashes and static-surface hash into the config; compute the
campaign source-binding digest; then compute the full contract digest.  The
later capsule independently binds the final config raw hash, contract digest,
campaign source-binding digest, and live tool raw hash.  No normalized or
masked executable region participates in this chain.

Capsule, owner-confirmation, and authorization digests are deliberately not
fields of the contract: they can exist only after capsule derivation.  The
authorization template binds them externally, avoiding a contract/capsule
self-cycle.

All digest domains are stored as exact ASCII strings without an embedded NUL.
The implementation appends exactly one `0x00` byte before hashing the compact
canonical JSON preimage without its transport LF.  The contract freezes
sixteen named domains: contract, campaign source binding, capsule,
authorization, Cargo manifest, ledger record, trajectory binding, global
trajectory-event payloads, trajectory suffix order, other-nonfailure order,
failure rows, cleanup rows, the partition certificate, slot outcome, campaign
terminal, and campaign completion.

Only a later exact project-owner confirmation may authorize execution.  That
confirmation must bind the final contract, target commit and source binding,
campaign source binding, capsule, both external roots, all three slot path
records, maximum invocation count 3, the exact retry and success predicates,
the terminal-no-resume rule, and the forbidden-scope digest.  The owner record
is a local same-UID direction record, not a digital signature or external
identity attestation.

The fixed future roots are:

| Purpose | Absolute path |
|---|---|
| Owner authorization | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r1-authorization` |
| Durable campaign ledger | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r1-ledger` |

The controller worktree `/tmp/hegel-q05b-shadow-retry-v2r1-worktree` and all
seventeen prior-v2 authorization, ledger, clone, Cargo-cache, artifact,
evidence, and trace roots are frozen protected roots.  Prior-v2 root open,
access, and capsule/authorization/ledger reuse are all explicitly false.

The ledger and every slot freshness target must be absent at each preflight.
Authorization-root absence is intentionally not a preflight freshness claim:
it is enforced atomically and exclusively when the owner record is created;
the record is then expected to be present for an authorized run.  Future
directories must be ordinary, owner-controlled mode-`0700` directories;
immutable records must be ordinary, single-link, mode-`0444` files published
without replacement and fsynced with their held parent directory.

## 3. Fixed isolated slots

Every slot has its own clone, copied Cargo cache, external artifact location,
evidence root, trajectory root, stdout, and stderr.  None may be selected by a
caller or reused by another slot.

| Slot | Dedicated clone root | Cargo cache root | External artifact | Evidence root | Trace root |
|---|---|---|---|---|---|
| 01 | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r1-slot-01-clone` | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r1-slot-01-cargo-cache` | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r1-slot-01-artifact/phase3_q05b_dual_qualification_v2.json` | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r1-slot-01-evidence` | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r1-slot-01-trace` |
| 02 | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r1-slot-02-clone` | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r1-slot-02-cargo-cache` | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r1-slot-02-artifact/phase3_q05b_dual_qualification_v2.json` | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r1-slot-02-evidence` | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r1-slot-02-trace` |
| 03 | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r1-slot-03-clone` | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r1-slot-03-cargo-cache` | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r1-slot-03-artifact/phase3_q05b_dual_qualification_v2.json` | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r1-slot-03-evidence` | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r1-slot-03-trace` |

The stdout and stderr files are fixed beneath each corresponding evidence
root as `supervisor.stdout` and `supervisor.stderr`.  The external trajectory
binding is `trajectory-binding.json`, and the slot result is
`slot-result.json`, in that same evidence root.  Trace IDs are exactly
`q05b-shadow-retry-campaign-v2r1-slot-01`, `-02`, and `-03`.

All external artifact, evidence, trace, stdout, stderr, and cache paths are
outside Git roots.  Each dedicated clone is itself a new isolated Git root
outside every pre-existing Git or protected root.  The frozen supervisor
requires its immediate artifact at
`<clone>/Hegel Machine/artifacts/phase3_q05b_dual_qualification_v2.json`;
therefore this temporary supervisor path is separately named in the contract.
On exact success, those bytes must first pass public artifact replay and then
be copied without replacement to the slot's external artifact path.  The two
raw SHA-256 values must be equal.  On a retry-eligible gap failure, both paths
must be proven absent.

The three external artifact parent roots are separately frozen freshness
targets: `...slot-01-artifact`, `...slot-02-artifact`, and
`...slot-03-artifact`.  Each entire parent path must be absent before its slot
becomes eligible.  The same continuously lock-holding process may create the
eligible parent as mode `0700`; any pre-existing parent, file, alias, or
uncertain creation is terminal and cannot be reused.

The capsule binds each shared state parent's stable `dev`, `ino`, mount ID,
mode, and UID.  It deliberately excludes the parent directory's link count,
because authorized creation of sibling authorization, ledger, and earlier-slot
directories changes that count.  Every observed parent replay still records
the full identity including link count, and any present freshness target also
binds its own link count.  Parent inode or mount substitution remains fatal.

Before any slot is prepared, and again at every pre-spawn and post-return
boundary, the runner applies bidirectional lexical ancestry checks to all
fixed paths, component-no-follow traversal to campaign-opened paths, and
device/inode/mount-ID identity checks to existing campaign-opened roots.  It
rejects symlinks, special files, multiple-link files, shared observed inodes,
and observed aliases in that scope.  The controller does not intentionally
open protected historical, Meta, or production roots, so it does not claim to prove
hardlink or bind-mount exclusion against those unopened roots.  Its operational
threat boundary is an owner-controlled UID and mount namespace with no
concurrent same-UID or mount-namespace adversary.  A later slot's paths must
remain absent until its predecessor has durably reached exact retry
eligibility.

## 4. Strict execution sequence

The prior authorized v2 campaign terminalized before its first supervisor
invocation while verifying the `SLOT_01` clone.  Its `git clone --local`
copied the source repository's commit-graph, while later ref deletion and
pruning correctly removed two source-only stash/index commits.  The copied
graph still named those removed OIDs, so graph-enabled
`git fsck --unreachable` reported missing commits even though the target
Commit-A closure itself was complete.  Disabling commit-graph made that clone
fsck pass, identifying the copied stale graph—not the target closure—as the
failure source.

Each v2r1 slot instead uses a file-protocol `git clone --no-local` of the same
bound Commit-A object closure and a private, content/mode-manifest-equal copy
of the allowed Cargo cache.  Clone preparation removes every inherited source
ref, remote, reflog, source-only branch, and stash ref, then prunes.  It creates
exactly one permanent internal ref,
`refs/hegel/q05b-shadow-retry-v2r1-target`, bound to Commit A, and writes a new
commit-graph from commits reachable through only that ref.  The ref is not
temporary: its exact name and object ID are replayed before and after the
supervisor, eliminating a create/write/delete lifecycle window.  Preparation
and every later clone replay require `git commit-graph verify`, an empty-output
graph-enabled `git fsck --full --strict --unreachable --no-reflogs`, and the
same empty-output strict fsck with `core.commitGraph=false`.  Thus graph
metadata and the underlying object closure must independently agree.

Cargo
source entries may be owned only by root or the effective UID and may not be
group/world writable; every private-copy entry must be owned by the effective
UID.  Ownership is recorded in both evidence manifests but intentionally
excluded from their content-equality digest because the root-owned source is
copied into an owner-private tree.  No network is permitted during clone
preparation, cache preparation, or supervisor execution.  The supervisor is
invoked through `/usr/bin/python3.10 -I -S -B
-X utf8`, with only the fixed slot paths and the bound Commit-A value:

```text
tools/phase3_q05b_dual_qualification_v2.py
  --run
  --project-root <slot clone>/Hegel Machine
  --source-commit <bound Commit-A>
  --artifact <frozen in-clone supervisor artifact>
  --cargo-cache-source <slot-private cache>
  --trajectory-root <slot trace root>
  --trajectory-trace-id <slot fixed trace id>
```

The fixed local runtime inputs are `/usr/bin/git`, `/usr/bin/docker`, Docker
host `unix:///var/run/docker.sock`, clone source repository
`/tmp/hegel-q05b-shadow-retry-v2r1-worktree`,
project subpath `Hegel Machine`, and read-only Cargo copy source
`/home/erzhu419/.local/state/hegel-machine/rust-cargo-cache`.  The final runner
recognizes that the clone source is a linked worktree whose common Git store is
beneath the original protected repository.  It opens that store read-only and
binds the absolute worktree-Git/common/object paths plus their
device/inode/mount/mode/UID identities, repository format `0`, SHA-1 object
format, and SHA-256 of Git's NUL-delimited local-config listing into the
capsule.  Alternates, HTTP alternates, shallow/graft metadata, partial-clone or
promisor config, `.promisor` pack metadata, every local `include.*` or
`includeIf.*` directive, and every local `fsck.*` override are rejected before
authorization.  This prevents an external include, skip list, or severity
override from weakening the strict integrity gate.  This is the sole
explicitly permitted read-only original-repo
input; it does not authorize original worktree mutation.

The final runner
must replace the complete environment rather than inherit ambient credentials,
proxy configuration, agent sockets, tokens, or secrets.  The exact map freezes
file-only Git protocol, disabled prompts/hooks/fsmonitor/lazy fetch, null
global/system configuration, `/nonexistent` HOME/XDG/Docker config, fixed
locale/timezone/PATH, and a slot-substituted `core.worktree` value.

The supervisor monotonic timeout is exactly `7200` seconds.  Stdout and stderr
are independently bounded at `67,108,864` bytes.  Timeout cleanup uses a
`120`-second `SIGINT` grace followed by `SIGKILL` and a `30`-second reap
window.  Read-only Git commands are bounded at 120 seconds and 64 MiB per
stream; read-only Docker list/inspect commands are bounded at 60 seconds and
16 MiB per stream.  Truncation, timeout, incomplete reap, a surviving process
group, or wrapper-close uncertainty remains process/cleanup uncertainty.  A
registered Git command's ordinary nonzero status or required-empty-stream
violation is classified as a source/binding failure only when its bounded
receipt proves no timeout or overflow, direct-child reap, process-group
absence, and complete wrapper resource closure.  Spawn receipts, exception
receipts, close/cleanup faults, signals, and unregistered argv can never take
that reclassification path.  Every caught bounded-command failure
carries a structured receipt containing the fixed argv, label, phase,
and typed reason.  When a process result exists it also contains
return/timeout/overflow/reap/process-group facts, classification, and each
stream's byte count, raw SHA-256, first-1024-byte hex excerpt, and truncation
flag.  If setup, spawn, selector, read, or cleanup raises before a result, those
unavailable facts are explicitly null and any prior nested receipt is retained.
The receipt is authoritative when the authorized campaign includes it in an
immutable ledger terminal record and the chained completion record commits.
That completion verifier now accepts only the frozen receipt-schema registry:
bounded-command result and exception diagnostics, a spawn-registration receipt
only as a bounded-exception prior receipt, the pre-spawn Docker wrapper, an
explicit known-no-publication receipt, and the bounded omission marker.  It
checks exact fields, command label/argv, phase, terminal error linkage, stream
counts/digests, Docker rows and active-slot prefix, and a depth-bounded prior
chain.  Unknown schemas, extra fields, recursive receipts, and receipts whose
terminal/completion/ledger hashes were merely recomputed are non-authoritative.
An exception escaping the Docker inventory wrapper is first normalized to the
same exact-argv bounded-command exception diagnostic.  The pre-spawn Docker
wrapper may replace that nested diagnostic with the bounded omission marker
only when its canonical diagnostic bytes exceed the frozen 16-KiB decode bound;
the verifier rechecks that byte bound instead of accepting a small forged
omission.  Its terminal-detail cross-link uses the same frozen 4,096-character
error-detail truncation as terminal publication, while the receipt retains the
bounded full Docker diagnostic.
A failure before ledger creation, including
read-only preflight or authorization setup, can only return the same structured
receipt on CLI stdout.

Docker replay binds label key `org.hegel.q05b.source_commit` to the final exact
Commit-A value, plus the reserved execution-namespace and supervisor-slot
keys.  Before spawn and after reap, read-only enumeration must find cardinality
zero for the target commit label.  The wrapper may never issue Docker remove,
kill, stop, or another mutating command; only the bound supervisor may clean
containers in its own nonce/commit namespace.

The slot's durable consumed record must exist before spawn.  The supervisor
runs in a new outer process group with stdin from `/dev/null` and distinct
bounded binary stdout/stderr pipes.  Exact bytes are persisted to the fixed
evidence paths.  A timeout sends `SIGINT` to the whole group, permits only the
bound supervisor to clean its nonce/commit-labeled containers, then sends
`SIGKILL` to the group if needed.  The direct child must be waited exactly once
and the original process group must be absent; wrapper-owned pipes, selector,
and descriptors must close, and the read-only labeled
Docker inventory must be empty.  This is not cgroup proof against an escaped
or reparented descendant outside the original process group.  It relies on the
frozen trusted supervisor and owner-controlled namespace boundary.  Timeout,
signal, `ECHILD`, observed survivor, or cleanup uncertainty is terminal and
can never enable the next slot.

Before consumption and again immediately before spawn, Linux
`/proc/self/task` must prove that the controller has exactly one thread, and
`SIGCHLD` must be `SIG_DFL`; ignored or foreign child reaping is rejected before
any consumed record.  After the callback, `SIGINT` is blocked across the final
thread gate and the `Popen` handle stores, then the parent's prior mask is
restored.  Default fatal signals retain the last-durable-prefix evidence
ceiling.

There is no Python `preexec_fn`.  Instead, `Popen` performs `setsid` and its
first exec into a fixed, source-bound isolated Python 3.10 launcher.  In that
fresh single-threaded interpreter the launcher clears the signal mask,
normalizes `SIGINT` and `SIGCHLD`, writes exactly its decimal PID plus one LF to
a private `O_CLOEXEC` registration pipe, closes that descriptor, and `execve`s
the actual supervisor.  The descriptor is never present in the supervisor.  A
failure of the actual exec after valid registration exits with fixed status
127.  The parent accepts exactly one complete row followed by EOF and requires
it to match the PID stored by `Popen` whenever that store completed.

This registration closes the catchable Python exception windows between the
kernel fork/exec return and `Popen.pid` storage, between PID storage and
`_child_created`, and between `Popen.__init__` return and the caller's next
store: a valid receipt supplies the exact process-group/direct-child identity
for `SIGKILL`, direct `waitpid`, and group-absence proof.  A missing, torn,
malformed, additional-byte, or mismatched receipt, or any cleanup double fault,
is `FAIL_CAMPAIGN_SPAWN_UNCERTAIN`.  That error forbids the terminal record,
completion record, and every later ledger or slot write; it never claims that
cleanup succeeded.  A valid consistent receipt plus proven reap and group
absence may instead close an ordinary caught spawn failure.  Uncatchable
`SIGKILL`, machine power loss, kernel failure, and hostile same-UID or namespace
interference remain outside this mechanism's evidence ceiling; only the last
durable prefix may then be claimed.

If `Popen` has already reaped an initial-launcher exec failure, cleanup never
signals or probes that numeric PID/PGID: it may already have been reused and
cannot prove anything about the original process group.  The missing receipt
therefore remains spawn uncertainty.  Only an unreaped identity bound by
`Popen` or by the valid launcher receipt may be killed and directly waited.

## 5. External trajectory binding

Retry or success admission for an invoked slot requires an on-disk v2
trajectory and its complete external binding.  In-memory writer state is not
evidence.  A controller wrapper failure after spawn can terminate before raw
stream publication, trajectory replay, or binding; in that case the durable
consumed ledger row and, when terminal publication succeeds, its bounded-process
receipt are the controller evidence, while any unbound trace prefix remains
non-admissible.  For an admissible slot, a safe, same-pass disk replay must bind:

- campaign/slot identity and ordinal;
- bound target commit, supervisor tool hash, and trajectory-core hash;
- trace-root path plus device, inode, mount ID, mode, UID, and link count;
- trace ID, producer `Q05B_SUPERVISOR_V2`, and `CLOCK_MONOTONIC`;
- verification status and terminal-manifest outcome;
- terminal-manifest raw hash and manifest digest;
- event count, first/final event digests, and terminal event type;
- complete-verifier result, verifier error, torn-tail fact, closed/open span
  facts, open-span contexts, terminal error code/detail, and the bounded
  retained failure and cleanup rows in sequence order;
- the first and final event digests, final durable minimal event row, canonical
  digest of every verified raw event payload, and, for a complete failure, a
  maximum-256-KiB post-origin partition certificate; on success, independently
  extracted ten stage roots, artifact hashes, and
  prepublication authority root;
- process return/timeout/overflow/reap/process-group facts, stdout/stderr raw
  byte counts and hashes, and both artifact-presence/digest facts;
- held-parent and held-file identity evidence for both the in-clone and
  external artifact paths;
- full read-only Docker inventory diagnostics, before/after source evidence,
  future-slot freshness evidence, and cleanup evidence, not merely their
  aggregate roots;
- deterministic bounded clone-closure differences, including ignored files,
  empty directories, and `.git` state, if the before/after closure differs;
- before/after source replay roots and the cleanup replay root.

These fields are published as the slot's external trajectory-binding record
and closed by a domain-separated binding digest.  Event and manifest files
must be ordinary, no-follow, single-link mode-`0444` files under an ordinary
mode-`0700` trace root.  An incomplete prefix, torn tail, extra namespace,
wrong mode/link/identity, replay disagreement, or uncertain binding
publication is terminal; it is never retry evidence.

The raw immutable trace is authoritative and remains fully reconstructable;
the ledger does not duplicate every event.  The bounded certificate partitions
the origin-inclusive suffix through the terminal event into: every explicit
`FAIL` row (at most five for retry), every nonfailure
`HELD_ACTOR_ABORT_CLEANUP` `BEGIN`/`OK` row (at most twelve for retry), and all
other nonfailure events as an exact count plus ordered rolling root.  It also
binds the suffix count/root, global payload digest, per-class counts/digests,
retained/truncated facts, `unpartitioned_event_count`, and its own digest.  The
partition arithmetic must close exactly and retry requires zero unpartitioned
events and no retained-row truncation.  Thus the binding remains below the
4-MiB ledger ceiling even when long sampling runs make a duplicated full event
projection exceed that ceiling.

Each ordered root starts at
`SHA256(domain || 0x00 || "EMPTY")`.  For each selected event in increasing
global sequence it replaces the root with
`SHA256(domain || 0x00 || "ROW" || 0x00 || prior_root_32 ||
uint64_be(sequence) || record_digest_32)`.  The class precedence is `FAIL`
first, then nonfailure cleanup `BEGIN`/`OK`, then every other nonfailure
`BEGIN`/`OK`; anything else increments the explicit unpartitioned count.

For a trajectory that the strict verifier reports complete, the durable
namespace is the complete trajectory log.  Every instrumented operation first
publishes a hash-chained durable `BEGIN`; a catchably completed or failed
operation then appends `OK` or `FAIL`, with stage, substep, actor, detail, typed
error, and sample timing fields.  Producer-free replay therefore recovers both
the causal path and the reason for every catchably closed failure.  A durable
logger/publication failure, hard power loss, or uncatchable `SIGKILL` may leave
only the last durable prefix and its open-span contexts; the verifier reports
that incompleteness and never invents a closure or reason beyond the final
fsynced event.

## 6. Exact retry predicate

`SLOT_02` or `SLOT_03` may be opened only when **all** of the following are
true for the immediately preceding slot, in one coherent evidence replay:

1. The supervisor was spawned exactly once, returned normally, and was
   `waitpid`-reaped; its exit code is exactly `1`, with no signal, timeout, or
   wrapper exception.
2. Stdout is exactly one canonical ASCII/JSON single-LF v2 Q0.5b error object;
   stderr is empty.  The wire status is exactly
   `HELD_ACTOR_GAP_EXCEEDED`, `artifact_written` is false, all Q0.5b and Q1
   counts/masks are zero, receipts are null, Q1 and Q2 are `NOT_RUN`, and M3
   roots are null.
3. The disk trajectory safely replays as exactly `COMPLETE_FAIL`.
4. Exactly one failure row is the metric-complete origin.  Its operation is
   `LIVE_RESOURCE_SAMPLE` or `HELD_FINAL_RESOURCE_SAMPLE`; its error code is
   `HELD_ACTOR_GAP_EXCEEDED`; actor, role, sample ordinal, previous finish,
   scheduled start, actual start, finish, collection duration, scheduler
   lateness, target wait, gap, maximum, exceeded flag, and error detail are all
   non-null and arithmetically exact.
5. At that origin, target wait is exactly `50,000,000 ns`, maximum gap is
   exactly `250,000,000 ns`, observed gap is strictly greater than the maximum,
   and `gap_exceeded` is true.
6. Every other FAIL row is an exact registered causal propagation of that
   single origin, carries the same `HELD_ACTOR_GAP_EXCEEDED` code, and carries
   null sample metrics.  For `HELD_FINAL_RESOURCE_SAMPLE`, every layer and the
   stdout/terminal wire retain the raw origin detail.  A normal continuous
   `LIVE_RESOURCE_SAMPLE` path is origin -> sampler -> actor-close -> Stage 5
   (or trusted-host Stage 6) -> run.  If one endpoint sampler exits while
   Stage 4 is still coordinating the two endpoints, the separately registered
   early path is origin -> sampler -> Stage 4 -> run.  In either LIVE path the
   origin and sampler retain the raw detail, while every later layer and the
   stdout/terminal wire carry exactly one `held actor sampler failed: ` prefix.
   The bounded certificate's explicit `FAIL` projection must equal these
   failure rows exactly, and any additional `FAIL` therefore rejects retry.
   Every event from the origin through the terminal event must fall into the
   certificate's `FAIL`, cleanup-nonfailure, or other-nonfailure partition;
   counts must sum to the exact suffix count and
   `unpartitioned_event_count` must be zero.  Nonfailure `BEGIN`/`OK` events,
   including truthful Stage-4 completion, Stage-5 entry, peer sampling, and
   in-flight drain closures, are bound by their count and ordered root but are
   not an independent failure and do not disqualify an otherwise exact retry.
   Their full detail remains available in the source-bound, payload-digest-bound
   immutable raw trace.

   Every post-origin `HELD_ACTOR_ABORT_CLEANUP` nonfailure row is retained
   explicitly.  A registered cleanup span has the frozen lifecycle
   context, `ENTER` begin detail, exact zero-error OK detail, and a globally
   increasing canonical padded counter suffix.  Its matching `OK` need not be
   adjacent: other nonfailure events and registered FAIL propagation may
   interleave before it closes.  The same actor may have at most two cleanup
   pairs, covering local Stage-4/Stage-5 cleanup followed by the outer
   `run_actual_v1` sweep, and the origin actor must have at least one successful
   pair.  A malformed cleanup, a third cleanup for one actor, a missing cleanup
   for the origin actor, retained-row truncation, a nonzero unpartitioned count,
   a partition count/root/digest mismatch, a
   missing/collapsed/double-prefixed tier, an unregistered Stage-4 path, an
   independent failure, or a second origin rejects retry.

   These certificate and ordered-root checks belong to the same source-bound
   trajectory replay that constructs the external binding.  The pure campaign
   core and ledger-only verifier validate the retained rows, certificate
   arithmetic and digests, and binding/observation cross-links, but do not
   independently recreate the suffix, other-nonfailure, or global-payload roots
   from raw event files.  An independent audit of those roots must replay the
   authoritative immutable raw trace with the bound trajectory verifier.
7. Both the in-clone supervisor artifact and external artifact are absent,
   with anchored parent identities.
8. The direct child is reaped, the original process group is absent,
   wrapper-owned pipes/selector/descriptors are closed, temporary/staging
   publications are absent, the supervisor reports cleanup complete, and the
   nonce/commit-labeled Docker inventory is empty.  No cgroup or escaped
   descendant claim is made beyond the stated threat boundary.
9. Clone HEAD/tree/parent/clean state, every target raw hash, config policy,
   trajectory-core binding, Cargo source/copy manifests, and source bindings
   replay byte-exactly before and after.  The fixed credential-free
   environment, file-only Git policy, and bound Docker network-none policy
   replay exactly; this is not kernel packet-level network attestation.
10. The external trajectory binding, current slot outcome, and next-slot
    eligibility records are durably published, mutually consistent, and have
    no fsync, identity, or visibility uncertainty.
11. Under the owner-controlled namespace boundary, the controller did not
    intentionally open or mutate historical attempt-2/attempt-3, Meta,
    production, Q1, ACTIVE, adoption, promotion, or current state.

This is a conjunction, not a score.  Missing, false, null, contradictory, or
unverifiable evidence for any clause is terminal.  An error message that merely
contains the word “gap”, an exit code without the trajectory, or a
self-consistent trajectory without cleanup and binding certainty cannot
authorize another slot.

The only retry transitions are:

```text
SLOT_01 exact clean gap -> SLOT_02 intent
SLOT_02 exact clean gap -> SLOT_03 intent
SLOT_03 exact clean gap -> CAMPAIGN_RETRY_EXHAUSTED_TERMINAL_NO_SLOT_04
```

## 7. Exact success

A slot succeeds only if all success clauses close together:

- normal return, exit `0`, exact canonical single-LF success object on stdout,
  empty stderr, direct-child reap, and absence of the original process group;
- one ordinary single-link in-clone artifact at the frozen path;
- public artifact replay status exactly
  `Q05B_QUALIFICATION_20_OF_20_Q1_NOT_RUN`, with qualification count `20`,
  mask `1048575`, Q1 `NOT_RUN`, Q1 count/mask `0`, and eight null Q1 roots;
- exact equality between stdout and artifact roots, counts, masks, path, and
  raw digest;
- disk trajectory `COMPLETE_OK`, terminal event `OK`, no failure rows, and no
  open spans;
- durable external trajectory binding;
- byte-identical no-replace publication at the fixed external artifact path;
- the evidenced cleanup boundary above, zero matching Docker containers, and
  exact before/after source, config, Cargo, and target binding replay;
- a durably staged terminal success ledger record followed immediately by the
  verified, chained `campaign.completion.json` authoritative final commit,
  with no publication uncertainty and no intentional historical attempt-2/attempt-3, Meta,
  production, Q1/ACTIVE, adoption, promotion, or current access or mutation
  beyond the explicitly permitted bound clone and Cargo inputs.

Success immediately terminates the campaign.  Remaining slots stay absent and
cannot be invoked.  The retained artifact and evidence remain shadow-only and
cannot be copied, moved, linked, adopted, promoted, made current, or consumed
by production beyond the single fixed external artifact publication already
bound by this contract.

## 8. Durable ledger and terminal behavior

The future runner must hold one exclusive ledger lock and directory descriptor
continuously from first intent through the verified completion record.  The same
process alone may cross from a clean gap result into the next slot.  Every
record binds the contract, target binding, campaign source binding, capsule,
owner authorization, prior record digest, slot identity, and external
trajectory binding where applicable.

The permitted chain is:

```text
ABSENT_NOT_AUTHORIZED
  -> AUTHORIZED_CAMPAIGN_INTENT_STAGED
  -> CAMPAIGN_AUTHORIZATION_CONSUMED
  -> SLOT_01_INTENT_STAGED
  -> SLOT_01_CONSUMED_OUTCOME_UNKNOWN
  -> success terminal | failure terminal | exact-gap eligible
       exact-gap eligible -> SLOT_02_INTENT_STAGED
       -> SLOT_02_CONSUMED_OUTCOME_UNKNOWN
       -> success terminal | failure terminal | exact-gap eligible
            exact-gap eligible -> SLOT_03_INTENT_STAGED
            -> SLOT_03_CONSUMED_OUTCOME_UNKNOWN
            -> success terminal | failure terminal | retry-exhausted terminal
  -> CAMPAIGN_TERMINAL_STAGED
  -> CAMPAIGN_COMPLETION_COMMITTED
```

`campaign.terminal.json` is a staged outcome, not an authoritative final result
by itself.  Only an immediately chained, digest-valid
`campaign.completion.json` is the final commit; the completion body binds the
terminal-body and terminal-ledger digests, exact preceding filenames/count,
capsule, authorization, invocation count, and last-slot outcome.  The
completion ledger record and CLI result are fully built before its final
`O_CREAT`.  Every prepared ledger record is frozen as one canonical byte string;
the exact same bytes are structurally checked, published, reopened, and replayed.
Terminal progress is derived from those frozen published records rather than
mutable caller counters.  The producer-free ledger verifier rejects missing,
reordered, corrupt, terminal-only, and rehashed-but-semantically-impossible
chains.  It validates the exact intent and authorization bodies, slot-record
mapping and digests, trajectory-binding/outcome cross-links, pure-core decision,
next-slot eligibility, and terminal progress/status/error semantics for all ten
legal caught-error prefixes and every complete normal branch.  It does not, by
itself, recreate trajectory partition roots from raw event files.

Any crash after intent, any consumed-but-unknown state, and any uncertainty in
intent, consumption, outcome, trajectory-binding, external-artifact, failure,
terminal-record staging, or completion-record publication is terminal with
**no resume**.  Catchable
errors are assigned the most specific configured terminal disposition and, if
available, preserve their structured diagnostic receipt in
`error_receipt_or_null`.  If terminalization itself then fails, a bounded
composite receipt preserves both the original error snapshot and the
terminalization error without recursive growth.  Raw stdout/stderr files may
remain as a truthful partial prefix when failure occurs before their binding,
but that prefix grants no resume, retry, second write, or later supervisor call.
Before an immutable publication call, only an explicit helper receipt proving
`KNOWN_NO_PUBLICATION` may remain a typed pre-publication failure.  Any other
catchable fault across the `O_CREAT`, helper-return, or caller-store boundary is
publication uncertainty.  After creation may have occurred, any write, chmod,
file-fsync, directory-fsync, replay, stat, or immutable-record descriptor-close
uncertainty before verified replay is likewise publication uncertainty.  Every
such diagnostic binds a validated publication label and basename,
distinguishing stdout, stderr, evidence, authorization, artifact, ledger,
terminal, and completion publication.  No second terminal write is attempted
after ledger or publication uncertainty.  A later process may inspect the last
durable ledger prefix but cannot continue it.  Failure to publish a failure or
terminal/completion record does not restore authority and does not permit
another write or supervisor call.

Once the completion helper has fsynced, reopened, byte-replayed,
identity-checked, and root-replayed the exact final record, its protected commit
callback installs the authoritative prebuilt result before ordinary live-chain
bookkeeping.  If a catchable asynchronous fault lands at that callback's final
store boundary or at a callee-return/caller-store boundary, the controller may
recover success only by producer-free offline replay of the exact disk chain and
matching its completion, terminal, and ledger digests to the prebuilt result.
The cached or disk-recovered result is returned without a second terminal or
completion write.  Later ledger unlock and directory-descriptor close errors
are outside final-record validity, although every close is still attempted.
This rule terminates the publication regress; it does not exempt any immutable-
record close that precedes verified completion.  Unknown is never retry-
eligible.

## 9. Explicitly forbidden scope

This contract forbids network access; Meta and production access; Hegel Q1;
`ACTIVE`; adoption; promotion; current-pointer writes; generic retry;
`SLOT_04`; and all access to or mutation of historical attempt-2 or attempt-3
authorization, ledgers, worktrees, artifacts, or receipts.  It also forbids
copying, moving, linking, or exporting any campaign artifact beyond the one
fixed external per-slot artifact path.

The original repository, historical attempt roots, shared Cargo cache, Meta
repositories, and production roots are protected.  The implementation uses
the lexical, component-no-follow, and opened-root identity checks stated in
Section 3; it does not claim protected-root hardlink or bind-mount exclusion.
Protected Meta/production/historical paths are negative registry entries, not
campaign inputs, and the owner-controlled-namespace threat boundary is part of
the contract rather than an unstated assumption.

## 10. Present facts

At the current v2r1 frozen-source state, before the 13 binding summaries are
filled or any read-only preflight:

- target Commit A remains finalized and independently replayable; the v2r1
  controller source, tests, document, exact hash domains, roots, and trace IDs
  are frozen; the source hashes, static-surface hash, predicate digests,
  ledger-registry digest, forbidden-scope digest, source-binding digest, and
  contract digest comprise the 13 deliberately null values that the main agent
  will fill next and are not yet authority;
- the trajectory and campaign decision suites include real-control-flow gap
  propagation KATs, torn-prefix/open-span replay, sampler/process/deadline/join
  races, success and failure wire replay, and three-slot stop/retry/exhaustion
  tests; the campaign ledger KATs additionally cover all ten legal caught-error
  prefixes, real valid success, frozen prepared bytes, rehashed semantic
  forgeries, bounded composite receipts, and injected `O_CREAT`, helper-return,
  callback-store, and caller-return publication faults;
- no capsule has yet been owner-authorized, and source binding alone is not
  execution authority;
- no owner authorization record exists;
- no preflight has run;
- no external root or slot path has been created;
- supervisor invocation count is exactly zero;
- no Docker, Cargo, clone, artifact, evidence, trajectory, Q1, Meta,
  production, adoption, promotion, current, attempt-2, or attempt-3 action has
  occurred.

Therefore the frozen source surface carries status
`V2R1_FINAL_BOUND_AWAITING_EXACT_OWNER_AUTHORIZATION_NOT_EXECUTED`.  It cannot
pass contract loading or derive a capsule until the main agent independently
computes and fills all 13 binding summary values; no preflight, authorization,
or execution is represented by this status.
