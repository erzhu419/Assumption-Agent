# Hegel Machine Q0.5b Shadow Non-adoption Trial v1

Status: `SOURCE_BOUND / READ_ONLY_PREFLIGHT_VERIFIED / AUTHORIZED_CAPSULE_CONSUMED / TERMINAL_FAILED_CLOSED / NO_RETRY`.

Canonical contract digest: `sha256:c5691cd995939d31c1274620349294587c3a562598bba363b947d29cbdf29754`.

The exact pre-authorization read-only preflight completed on 2026-08-27 with capsule digest `sha256:40be382371150ac8847d99accf70ed96a8ee3a6ab738af14c534707cb56c0cd4` and preflight digest `sha256:dd8ab4650973beb990145d7d345f2189d9bd8be037e01bd196e6fec5b6afd79a`. It reported `execution_authorized=false`, no owner-authorization record, zero supervisor invocations, and absent clone, Cargo target, and ledger roots. The after-state replay confirmed those roots and the attempt-2 ledger/artifact remained absent and the attempt-2 worktree remained clean at the frozen commit/tree. The later authorized run independently rederived the same capsule before consuming it.

## Authorized one-shot terminal outcome

The owner subsequently authorized that exact capsule under authorization digest `sha256:efed7dc05d75caf07e785ae0a7ea6706572c44a4a5f93841857e736225935041`. The wrapper durably published intent `sha256:5e6910481ecc1633edd9c943265c3ac1bc5b9ff34743050a0c58e68f5eba4ff5` and consumption `sha256:de46180c5faf66ca0122fe5f342dae184629844c946220fad3114d2f98164dc6` before invoking the frozen supervisor exactly once. No second invocation is authorized or permitted.

The sole invocation returned exit code 1 and produced terminal failure receipt `sha256:90eb8984978a013e42bef582f41a272f551ae21a014f7f80aa49336131cb9eea`. Its canonical supervisor stdout SHA-256, `271bede54a995782de11f402b0090aaf42a8bc0171090b69ed77f1bf528322ad`, exactly binds the error wire with status `FAIL_Q05B_DUAL_POLICY`, detail `held actor sampler failed: Q05BDualSupervisorError:FAIL_Q05B_DUAL_POLICY: held actor sampling gap exceeded 250ms`, and `artifact_written=false`. The frozen sampler waits nominally 200 ms between collections but rejects a next-sample start more than 250 ms after the previous collection finishes, leaving at most 50 ms of scheduling slack. The terminal wire does not carry the actor id, and the supervisor removes its temporary evidence during failure cleanup, so the available durable evidence cannot distinguish whether the Python or Rust endpoint sampler crossed that bound.

Read-only Docker event history confirms that the isolated Rust test and release-build containers both exited 0, then both endpoint containers started and ran for about 378 seconds. The Python endpoint later exited 0 and was destroyed; the Rust endpoint received the failure-cleanup SIGTERM and was destroyed. Trusted-host replay never started, the qualification count and mask remained zero, and no receipt or Q0.5b artifact exists. The wrapper verified complete survivor absence. The failure record states `attempt2_state_touched=false`, `meta_or_production_state_touched=false`, and `adoption_or_current_pointer_written=false`; the attempt-2 ledger/artifact remained absent and its worktree remained clean. This is a mechanics-policy timing failure, not a Q0.5b qualification observation and not evidence for or against v3 efficacy. The consumed authorization is terminal with `no_retry=true`.

This release froze and implemented the contract, pure validation core, I/O tool, tests, and documentation for the now-consumed shadow-only Q0.5b mechanics trial. Before the later exact authorization, construction and mock/static verification created no trial clone, authorization record, ledger, artifact, supervisor process, Docker/Cargo invocation, Meta access, production access, adoption decision, current-pointer write, Q1 run, or ACTIVE transition. That construction-only direction was not itself execution authority.

Construction occurred on the separate local branch `codex/hegel-q05b-shadow-nonadoption-v1` in `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-nonadoption-dev-worktree`. The attempt-2 worktree remains clean and detached from these edits at `f2b0a1fd…`; neither construction nor the completed direct-clone shadow attempt advanced its branch, ledger, or artifact.

The wrapper has no supported shebang or import-and-call execution path. Its only preflight entry is the exact absolute argv below; the corresponding `--run` argv differs only in the final mode. That argv was forbidden until exact capsule authorization, was invoked once after authorization, and is now permanently forbidden for this consumed trial:

```text
/usr/bin/python3.10 -I -S -B -X utf8 /home/erzhu419/.local/state/hegel-machine/q05b-shadow-nonadoption-dev-worktree/Hegel Machine/tools/phase3_q05b_shadow_nonadoption_trial_v1.py --preflight
```

`main`, the public preflight function, and the public run function all reject any different interpreter, tool path, isolated/no-site/no-bytecode/UTF-8 flags, system `sys.path`, optimization or extra `-X` state. The exact `/usr/lib/python3.10` standard-library search roots are an explicit host trust base; their individual module bytes are not claimed as source-bound by this contract.

## Independent source and execution surface

The authorized preparation step made one fresh local `git clone --shared` from `/home/erzhu419/mine_code/Asumption Agent`, without fetch, pull, credentials, or network, and detached it at exact commit `f2b0a1fd37a2f488cea64fc2c8d1dd877639bf9f`, tree `12d354842f0359aa5b4ecbeebd83cbe123567788`. Its exact parent `65dfc5f19450cee9510c63a85de32279160f291f` is recorded only as the source parent/baseline; it is not the trial HEAD.

The dedicated clone root is `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-nonadoption-trial-v1-shared-clone`. The sole trial directly executed only that clone's unchanged `Hegel Machine/tools/phase3_q05b_dual_qualification_v1.py --run`, whose frozen raw SHA-256 is `af406bba2e4d11f8f59c1d7f91d3f84c887d1e90d10cc1749baf2a8610a7c5d2`. Its fixed artifact path is `Hegel Machine/artifacts/phase3_q05b_dual_qualification_v1.json`; no artifact was published there.

The checkout contains the attempt-2 additive files because those bytes are part of `f2b0a1fd…`. They are inert source-tree bytes for this trial. The trial may not import or call the attempt-2 bridge core or runner, validate or stage a Meta-V11 bridge, construct an attempt-2 adoption receipt, create an `attempt-2.*` record, or borrow authorization digest `sha256:9623deef9716248e19ec2eb38a4be6adb8bab2d0332666add76ac916d268e722`.

Every wrapper Git command disables hooks, fsmonitor, lazy fetch, replacement objects, global/system configuration, prompts, SSH, and non-file protocols. Each mutating Git subprocess runs under fixed umask `0022` with mandatory restoration, so the exact post-clone filesystem modes do not depend on the caller's ambient umask. Source shallow state, source alternates/http-alternates/grafts/replace refs, and the analogous forbidden clone metadata are rejected. After clone creation, the no-follow `.git` directory identity and exact shared alternates bytes are frozen and replayed around both later mutators; `.git/info/attributes` is forbidden. Checkout and origin removal use explicit absolute `--git-dir` and `--work-tree` rather than repository auto-discovery. The child supervisor additionally receives an exact `core.worktree` override to the isolated clone. Clone replay verifies the exact top-level and `.git` paths, detached HEAD, no tags or remotes, tracked cleanliness, and all other files—including ignored files—so the post-run exception is exactly the single clone-local artifact rather than an invisible `artifacts/*.json` ignore match.

## Protected state and independent roots

The current attempt-2 worktree, ledger, artifact, and Cargo cache are protected and must not be opened for mutation or used as trial outputs. Its read-only replay first verifies the exact discovered top-level and linked-worktree Git admin directory, then pins those absolute `--work-tree` and `--git-dir` paths for HEAD, tree, status, and final identity replay:

- `/home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree`;
- `/home/erzhu419/.local/state/hegel-machine/q05b-attempt2-ledger-v1`;
- `/home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine/artifacts/phase3_q05b_dual_qualification_v1.json`;
- `/home/erzhu419/.local/state/hegel-machine/rust-cargo-cache`.

Meta and all three production roots are also forbidden. The implementation rejects normalized lexical overlap, symlink components, existing same-inode/bind aliases, and either-direction ancestor overlap for trial, attempt-2, source, and Cargo paths using no-follow held-FD device, inode, and mount identities. Meta and production roots receive lexical/ancestry comparison only and are never opened. These boundaries are replayed at preflight, before ledger creation, immediately before authorization consumption, at spawn preparation, and after return. A concurrent change makes the shadow outcome non-claimable but never grants cleanup or retry authority.

The new roots are pairwise distinct and fixed:

- authorization: `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-nonadoption-trial-v1-authorization`;
- ledger: `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-nonadoption-trial-v1-ledger`;
- clone: `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-nonadoption-trial-v1-shared-clone`;
- Cargo cache: `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-nonadoption-trial-v1-cargo-cache`.

Preflight may read the protected legacy Cargo cache only to derive an exact regular-file tree manifest; it may not mutate it or invoke Cargo. Regular-file mode, link count, extended-attribute absence, bytes, size, and digest come from one no-follow held FD rather than mixed pathname snapshots. After intent and before consumption, an authorized run copies that manifest into the initially absent private trial cache using the same held-FD replay and exclusive writes, fsyncs the new tree, and proves source-before/source-after/target manifest equality. The direct supervisor receives only the isolated copy. Symlinks, hardlinks, special files, extended attributes, source drift, a pre-existing target, or any manifest mismatch are terminal before consumption.

## Capsule and owner confirmation

Source implementation, final source binding, and read-only capsule derivation must precede any execution request. Preflight binds the exact expected detached clone identity and the read-only source Cargo manifest without creating either target or invoking Docker. Only after the owner confirms that exact capsule digest may `--run` stage intent, materialize the clone and isolated Cargo copy, verify them against the capsule, publish consumption, and invoke the supervisor. The capsule binds the final source-binding digest; clone commit, tree, parent, detached state, no tags/remotes, cleanliness, alternates and required raw bytes; exact argv and Python/Git/Docker executable identities; environment; isolated Cargo cache; inputs, metrics, and seeds; resource limits; monotonic timeout and kill/reap/survivor rules; fixed artifact and ledger paths; the protected-path registry; and the claim ceiling.

Immediately before the sole authorization spend, the wrapper performs one unified replay of the live contract/core/tool/tests binding, frozen Git objects, attempt-2 protected state, all three executable identities, clone identity and raw inputs, source/target Cargo manifests, artifact absence, exact ledger bytes, and read-only Docker label absence. Spawn preparation repeats path, executable, clone, and Cargo checks. Identity replay, argv/environment materialization, umask setup, and signal-mask setup remain in `SPAWN_PREPARE`; an internal boundary callback changes the count to one only as the next operation enters `Popen`, so a pre-boundary failure remains count zero while a `Popen` ambiguity is conservatively count one.

The capsule is not persisted as a mutable input file. Read-only preflight deterministically rederives it and emits its canonical mapping; the authorized run independently rederived the same mapping and accepted only the fixed authorization record that names its digest. A self-hash-valid contract or capsule is not authority. Execution was forbidden until the project owner explicitly confirmed the exact final capsule digest together with all of these constraints: exactly one supervisor invocation; shadow-only and non-adopting; no attempt 2, attempt 3, retry, production, Meta bridge, adoption, promotion, current pointer, Hegel Q1, ACTIVE transition, or network. That exact confirmation was provided and consumed; the terminal failure now forbids any retry or second invocation.

## Exactly-once ledger

The independent ledger has this finite state machine:

```text
OPEN_ALL_FOUR_RECORDS_ABSENT
  -> INTENT_STAGED_UNCONSUMED
       -> INTENT_STAGED_PREPARATION_FAILED_CLOSED_UNCONSUMED
       or
       -> CONSUMED_OUTCOME_UNKNOWN
          -> exactly one of:
               CONSUMED_SHADOW_COMPLETED_NONADOPTING
               CONSUMED_SHADOW_FAILED_CLOSED
```

The records are `shadow-trial.intent.json`, `shadow-trial.consumed.json`, `shadow-trial.success.json`, and `shadow-trial.failure.json`. Every record is bounded canonical single-LF JSON in an ordinary single-link file, mode `0444`, published through a held no-follow ledger directory with `O_CREAT|O_EXCL`, complete write, `fchmod`, file `fsync`, and directory `fsync`. The held parent and ledger FDs are continuously tied back to the fixed pathname before and after every publication and again before success returns; rename-and-replace makes publication uncertain and terminal. Before and after either outcome publication the wrapper replays the exact intent/consumed bytes and enforces success XOR failure. An entry that observes any pre-existing ledger root rejects terminally before clone/Cargo preflight, with nullable wire facts rather than pretending it is a fresh pre-intent rejection. Intent-, consumption-, or success-publication uncertainty is transport-only and never causes a second ledger write.

Durable publication of `shadow-trial.consumed.json` immediately after the unified replay and before the direct supervisor call is the sole authorization spend. Read-only preflight and rejected pre-intent validation may be repeated only while the ledger root remains absent and are not trial invocations. Once intent is durably staged, any preparation failure is an unconsumed but terminal failure; once consumption is durable—or its publication outcome is uncertain—every crash, timeout, `BaseException`, Popen ambiguity, launch failure, invalid/missing/drifting artifact, cleanup uncertainty, survivor uncertainty, protected-state replay failure, or outcome-link publication failure is terminal and cannot authorize retry, resume, fallback, or a second supervisor call. Cleanup addresses the process group even when its leader has already exited; normal return also requires group absence, otherwise bounded SIGINT/SIGKILL and direct-child reap run and success is forbidden. The wrapper only inventories Docker and never removes or mutates a container.

## Failure states

The contract freezes ten exact terminal failure states: ledger-root/lock uncertainty; intent-publication uncertainty; unconsumed preparation failure; consumption-publication uncertainty; consumed pre-launch failure; supervisor spawn/timeout/cleanup uncertainty; nonzero return; missing/invalid artifact; failed post-return isolation/non-adoption replay; and uncertain success-outcome publication. The pure core freezes the canonical digest of the entire failure policy, so a self-rehashed contract cannot widen allowed facts. Each receipt carries typed nullable and relationally checked facts for consumption, supervisor return, artifact presence, cleanup, and survivor absence. Here `*_state_touched=false` means not mutated, consumed, or used as authority; attempt-2 read-only replay is explicitly allowed, while Meta/production paths are not opened. Uncertainty cannot be normalized to false, and `no_retry=true` is unconditional.

## Claim ceiling

An exact old-supervisor success (`Q05B_QUALIFICATION_20_OF_20_Q1_NOT_RUN`, 20 predicates, mask `1048575`) may be wrapped only as `SHADOW_Q05B_MECHANICS_QUALIFICATION_OBSERVED_NONADOPTING`. It is a capsule-bound observation that the frozen Q0.5b mechanics completed. It is not evidence of v3 efficacy, causal improvement, generalization, superiority over v1 Treatment or v2 GSCL, C1, C2, adoption eligibility, Meta adoption/promotion/current status, Hegel Q1, or production readiness. The clone-local artifact may not be copied, linked, ingested, promoted, or used to update any current pointer.

## Source-binding transition

Finalization replaced the construction placeholder with the source-binding status `SOURCE_CORE_TOOL_TESTS_PRESENT_AND_BOUND_EXECUTION_STILL_REQUIRES_OWNER_AUTHORIZATION`, binding the core raw bytes, focused-test raw bytes, a strict AST surface, and normalized tool bytes whose only masked region is one exact three-string data assignment. The canonical contract self-hash and the tool's contract/raw/normalized literal independently replay. That status records the construction-time authority boundary; the owner later supplied the separate exact authorization, and the resulting one-shot trial is now consumed and terminally failed closed.
