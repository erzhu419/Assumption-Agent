# Hegel Machine Phase-3A Q0.5b Attempt-2 Meta-V11 Bridge v1

Status: `IMPLEMENTED_NOT_EXECUTED`

Canonical contract digest:
`sha256:e36ecb243a7f36d24c5a29f4fd6beb913c195848350f845ce615d8b0304535ed`.

This additive slice turns the owner's exact contextual confirmation into a
digest-bound, destination-scoped admission boundary for at most one Hegel
Q0.5b actual attempt-2.  It does not itself constitute execution evidence, a
cryptographic signature, Meta-wide adoption, promotion, a current-pointer
write, Q1 execution, an ACTIVE transition, or authority for attempt 3.

The bridge does not fabricate or synthesize a Meta report.  A real V11
post-transition qualification report is a required thirtieth upstream object.
Missing, empty, symlinked, non-file, null, or non-object report input fails
first with `MISSING_REAL_V11_REPORT`, exit `2`, empty stdout, a canonical error
on stderr, no bridge report, no ledger effect, no artifact, and no import or
call of the old actual tool.

## 1. Exact authorization

The bound owner authorization has schema
`hegel-phase3a-q05b-attempt2-owner-authorization/1`, role `PROJECT_OWNER`,
attempt ordinal `2`, and maximum actual invocations `1`.  It binds the exact
antecedent:

> 下一步若要进入 Hegel Q0.5b actual，需要你另行明确授权 attempt-2，并先补 digest-bound external authority/adoption bridge

and the exact confirmation `确认授权` with semantics
`EXACT_CONFIRMATION_OF_SELECTED_ANTECEDENT`.  Its domain-separated digest is
`sha256:9623deef9716248e19ec2eb38a4be6adb8bab2d0332666add76ac916d268e722`.

The ordered authorized actions are:

1. `CONSTRUCT_DIGEST_BOUND_META_V11_TO_HEGEL_EXTERNAL_AUTHORITY_ADOPTION_BRIDGE`;
2. `EXECUTE_HEGEL_Q05B_ACTUAL_ATTEMPT_2_ONLY_AFTER_BRIDGE_AND_ALL_EXISTING_RUNTIME_GATES`.

The ordered forbidden actions are `ATTEMPT_3_OR_RETRY`, `NETWORK_ACCESS`,
`Q1_EXECUTION`, `ACTIVE_TRANSITION`, `UNBOUND_META_ADOPTION`, and
`CURRENT_POINTER_WRITE_OUTSIDE_EXACT_BRIDGE_RECEIPT`.  The confirmation is
neither execution evidence nor an external signature.

## 2. Twelve admission gates

The wrapper admits the single call only after all twelve gates below pass in
the same process:

1. the exact owner-authorization body and domain digest;
2. one real, strict-object V11 report at the required absolute ordinary file;
3. 32 absolute, normalized, ordinary, non-symlink JSON files with distinct
   paths and inodes (all 496 unordered alias pairs are rejected);
4. 29 independent CLI digest anchors byte-equal to the bridge input registry;
5. all 30 V11 objects replayed through the public V11 verifier with all 11
   cumulative artifact maps;
6. the V11 disposition is exactly
   `QUALIFIED_FRESH_POST_TRANSITION_EVALUATOR_EPOCH`, its verified status is
   exact, and its four adoption/promotion/current fields remain withheld;
7. the Meta repository is clean at commit
   `fc9e781ce29ca0fef0f37ed97d07d5037adfdfff`, tree
   `932ea7acbea5e8fbf36ddbd11c648611e1fcdae0`, with all five frozen raw hashes;
8. attempt-1 ordinal, consumed failure, Stage-7 boundary, absent artifact,
   failure-manifest digest, errata raw hash/blob, and repair ancestry all match;
9. the requested Hegel source is one clean full-40 commit, the direct child of
   baseline `65dfc5f19450cee9510c63a85de32279160f291f`, with exactly the additive
   five files and no other diff;
10. every exact-five raw hash and every unchanged old Hegel path/raw hash
    matches the declared live source binding;
11. fixed project/artifact/Cargo/ledger paths, artifact absence, ledger mode,
    ledger state, and no-replace output aliases all pass; and
12. source-only import regimes prove the Meta verifier, the unchanged old
    Hegel actual, and artifact replay use exact-root `.py` sources rather than
    ignored bytecode or ambient import finders.

These gates prepare or consume a destination-scoped attempt input.  They do
not establish the 20 Q0.5b qualification predicates.

## 3. Public replay and input surface

The pure builder
`build_phase3_q05b_attempt2_meta_v11_bridge_v1` takes exactly 32 positional
objects: the V11 verifier's 30 objects followed by bridge input and bridge
contract.  Its verifier takes those 32 plus the bridge report and independently
binds the expected bridge-report digest and all input artifacts.

The first 30 CLI object flags preserve the V11 order from
`--competition-input` through
`--post-transition-qualification-report`.  They are followed by required
`--bridge-input`, default frozen `--bridge-contract`, required
`--meta-project-root`, `--hegel-project-root`, `--source-commit`, `--artifact`,
`--cargo-cache-source`, and `--ledger-root`, plus the exact 29
`--expected-...-digest` anchors.  There is no caller-selected output path.

The 11 cumulative V11 artifact maps have cardinalities
`2/4/7/9/12/15/18/21/24/26/29`.  The bridge report separately binds all 32 raw
input rows `{path,bytes,sha256}`.  Observed evidence values are never copied
into artifact metadata.

The five nonqualified V11 dispositions all produce
`META_V11_REPORT_NOT_QUALIFIED_BRIDGE_REFUSED`.  Their external-authority
decision, destination adoption receipt, attempt admission, and durable ledger
plan are null.  Only the exact qualified route produces
`META_V11_QUALIFIED_BRIDGE_ADMITTED_ATTEMPT2_PENDING_CONSUMPTION`.

The public report, consumed-record, success-link, and failure-link builders
and their standalone validators are untrusted structural constructors.  They
are not filesystem, ledger, execution, or adoption authority.  A report alone
omits the bridge input and 29 independent digest anchors; self-rehashing a
nested field cannot prove the 33-object chain.  Only the runner's in-process
full replay, live Git/raw checks, source-only checks, fixed-path checks, and
durable ledger transition can authorize the one old entrypoint call.

## 4. Source-only execution boundary

`-B` and `PYTHONDONTWRITEBYTECODE` prevent new cache writes but do not prevent
loading existing ignored `.pyc` files.  This slice therefore uses stronger
source-only regimes.

For Meta replay, preloaded `performance` modules are forbidden.  The exact
Meta root is temporarily first on `sys.path`; the bytecode cache prefix points
under unavailable `/dev/null`; `sys.meta_path` is restricted to
`BuiltinImporter`, `FrozenImporter`, and `PathFinder`; path hooks use
`FileFinder` with `SourceFileLoader` and the standard extension loader but no
`SourcelessFileLoader`; and the importer cache is fresh.  Every loaded project
module must map by module name to its exact ordinary `.py` source.  All new
`performance` modules and all interpreter import state are removed/restored
after replay.

The same exclusions remain active for the entire unchanged
`run_actual_v1(Path(project_root), source_commit, Path(artifact),
Path(cargo_cache))` call.  Before consumption, the wrapper permits only the
new bridge package/module to be preloaded, verifies the old tool's raw hash,
loads that exact source without invoking its CLI or `main`, verifies callable
identity, and records exact module-name-to-source paths.  Ambient custom
meta-path finders, path hooks, cached finders, in-tree bytecode, path traversal,
and arbitrary module/path pairings cannot satisfy the receipt.  The old
entrypoint is called exactly once only after consumption is durable.

Each terminal link binds three distinct source receipts.  The Meta live
receipt records its exact repo/commit/tree/clean identity, V11 contract digest,
and full `{path,device,inode,nlink,mode,size,sha256}` evidence for all five Meta
files.  The Hegel live receipt records HEAD/tree/clean, baseline/direct-child
and commit-count facts, attempt-1/repair ancestry, exact-five diff closure, and
the five live stat/hash rows.  The separate actual source-only receipt binds
the unchanged actual-tool hash, all six unchanged source/config/errata rows,
and every loaded exact-root `.py` origin.  Structural validators close the
receipt schemas and frozen semantics; only the runner's live observations make
their dynamic stat and Git fields execution authority.

`--preflight` never loads the old actual source, calls the old entrypoint,
starts Docker or Cargo, writes the ledger, or writes the artifact.

## 5. Durable exactly-once ledger

The only ledger root is
`/home/erzhu419/.local/state/hegel-machine/q05b-attempt2-ledger-v1`, mode
`0700`.  Its only permitted mode-`0444` ordinary single-link records are:

- `attempt-2.bridge-report.json`;
- `attempt-2.consumed.json`;
- `attempt-2.execution-link.json`;
- `attempt-2.failure-link.json`.

Every publication uses a held no-follow directory descriptor with
`O_CREAT|O_EXCL`, complete write, `fchmod(0444)`, file `fsync`, then directory
`fsync`.  Existing records are never replaced.

If the ledger directory is initially absent, the wrapper anchors its parent
with `O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC`, performs `mkdirat(0700)`, opens and
identity-checks the child, then fsyncs the child and parent before publishing
any record.  All ledger descriptors are close-on-exec and the ledger lock is
closed before the unchanged actual can start Docker or Cargo.  Read-only Git
checks use `GIT_OPTIONAL_LOCKS=0`, disabled credential prompts plus
system/global config, `GIT_NO_REPLACE_OBJECTS=1`, and no network-initiated
protocol.  Every call overrides repository-local `core.fsmonitor=false` and
`core.untrackedCache=false`; diff calls additionally use `--no-ext-diff` and
`--no-textconv`, so repository-local hooks or diff drivers cannot become an
execution path.

The state machine is:

```text
OPEN_ALL_FOUR_RECORDS_ABSENT
  -> BRIDGE_STAGED_NOT_CONSUMED
  -> CONSUMED_OUTCOME_UNKNOWN
  -> exactly one of CONSUMED_QUALIFIED_LINKED / CONSUMED_FAILED_CLOSED
```

A bridge-only staged file is resumable and has not spent authorization.  It
must byte-equal the current exact report.  Durable creation of
`attempt-2.consumed.json` is the sole spend.  From that point every restart is
terminal `ATTEMPT2_AUTHORIZATION_ALREADY_CONSUMED`; it never imports Meta or
the old actual, never retries, and never resumes an unknown outcome.  A crash
may leave `CONSUMED_OUTCOME_UNKNOWN`, which is deliberately terminal.  In a
concurrent race, runners may observe the same exact bridge stage, but the one
`O_EXCL` consumed winner alone may call actual.

Route precedence is strict: missing real V11 report first; then a valid
existing consumed ledger, before replacement-input replay; then remaining
input/source gates.  Invalid ledger combinations or noncanonical/tampered
records fail closed rather than becoming retry authority.

## 6. Outcomes and claim boundary

On old-actual failure, the authorization remains consumed and one canonical
`ATTEMPT2_FAILED_CLOSED` link is durably published.  A later invocation cannot
restore it.  On return, the fixed artifact is read without following symlinks,
strictly replayed through the unchanged public artifact decoder, and compared
to every run-result root, count, mask, path, hash, admission boundary, and final
delivery identity before `ATTEMPT2_QUALIFIED_LINKED` is published.

Failure linkage records `actual_returned=false` when the old call raised and
`actual_returned=true` when it returned before strict artifact or linkage
validation failed; neither post-consumption route permits fallback or retry.

A successful link still states only Q0.5b `20/20` and Q1
`NOT_RUN / 0 gates / mask 0 / eight null formal output roots`.  The bridge
never grants Meta-global adoption, Meta promotion, a Meta current pointer,
Hegel Q1, ACTIVE transition, network access, attempt 3, retry, or any
Operations Research claim.  It changes no prior file and is isolated from the
KG-op Operations Research paper/result line.

In particular: no Meta-global adoption, no Meta promotion, and no Meta current
pointer are produced by this destination-scoped bridge.

## 7. Operator modes

`--preflight` emits exactly one canonical bridge report on stdout and performs
zero writes.  Qualified and refused reports both exit `0`; refusal remains
strictly non-admitting.

`--run` is meaningful only for an admitted bridge.  Success emits the durable
execution link and exits `0`.  A consumed actual failure emits the durable
failure link and exits `1`.  Missing/invalid/refused/already-consumed inputs
emit canonical stderr, empty stdout, and exit `2`.

No actual attempt, Docker invocation, Cargo invocation, artifact publication,
or ledger consumption is evidence merely because this implementation and its
monkeypatched non-actual tests exist.
