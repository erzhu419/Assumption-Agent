# Hegel Machine Phase 3A — Q1 target-blind start successor v6

## Purpose and authority

This protocol replaces, but never resumes, the terminal Q1-start-v5 campaign. One exact v6 owner authorization covers one logical campaign: it binds the pure, unconsumed successor-prestart candidate bundle and authorizes one fresh `Python -> Rust -> trusted host replay` actor sequence. The candidate bundle has no independent execution, adoption, or publication authority.

The authorized scope is exactly `TARGET_BLIND_Q1_START_V6_ONCE_WITH_BOUND_SUCCESSOR_PRESTART_BUNDLE`. Each actor may receive at most one durable intent and one Popen. Completed actors are never measured again.

## Bound terminal lineage

The capsule binds the authoritative Q1 prestart state `NOT_RUN`, 14/20, mask `16383`, with eight null output slots, and the immutable v2, v3, v4, and v5 terminal lineages. In particular, the v5 lineage is bound exactly as follows:

- capsule digest `sha256:d259d4d0c3c919ad69a6c3101e06f9b51402d44214df88bb582ed68c79b9c303`, raw capsule SHA-256 `sha256:5cb19e394721aed57ba2e2d8c09383fb1fba012d3339fb30943a95eee622c1bb`;
- authorization digest `sha256:915f07f1a4cf5f1a677ef9b70288431725cfe69d88d96053bf87ef7297aa65c0`, raw authorization SHA-256 `sha256:a9ae7abf988f8ab246f57d07acc3310234adce90fc7dbde31e4d5fd51d9b78ce`;
- consumption record digest `sha256:44a5983c19bfe7d6afe297fbad30f2816fc7729afa4973487972c21df18efc86`, raw consumption SHA-256 `sha256:92eb989b9db4d9128d05819bad8653da21757bee7aad1de60dc586d6e449f019`;
- terminal record digest `sha256:4ec9f4c84526b16dee764dceecc7ea23e34962097590dec60b9145f5621de3a4`, raw terminal SHA-256 `sha256:99a97c5927f0010dee466b6b22d93144effe205eb788ba8846ac914759da9506`;
- supervisor outcome raw SHA-256 `sha256:79748ea0411b8c5a9aa0073590df189ef7109181dfba98b1ca9872d3073329f5`;
- status `Q1_START_V5_OUTCOME_UNKNOWN_TERMINAL_NO_RESUME`, failure `Q1_START_POST_INTENT_EXCEPTION`, and exact detail `PermissionError: [Errno 1] Operation not permitted: '/home/erzhu419/.local/state/hegel-machine/q1-start-v5-run/python-output/signature-02/stream-04/chunk-000000.framed'`;
- actor starts `python=1`, `rust=0`, `host=0`, completed prefix `[]`, and the Python action conservatively counted once.

V5 is terminal no-resume. None of its output, scratch, build, candidate-bundle, run, or ledger work product may enter v6. The new authorization must independently permit the fresh v6 sequence.

## Ownership correction and startup probe

The reached v5 failure was outside the scientific method: its endpoint container wrote bind-mounted output as UID/GID `65534:65534`, while the host freezer ran as UID/GID `1000:1000`; the freezer therefore could not apply its required read-only transition to the output tree. V6 binds both container actors to exact UID/GID `1000:1000`, requires the controller's current effective UID/GID to be exactly `1000:1000`, emits run-spec `container_user` as `1000:1000`, and requires output freezing to occur under that same identity.

Before authorization consumption and before any actor intent, the startup smoke may perform exactly one ownership/freezability probe using a pinned local actor image with network disabled. The probe may create and freeze only its transient probe tree. Its normal path performs exactly one Docker `run`; if that bounded command is interrupted or its result shape is unavailable, the same smoke performs one `rm -f` cleanup attempt followed by one exact-name `docker ps -a` absence check. The failure receipt records all three Popen counts, whether Docker was invoked, and whether both the container and transient root are proven absent. The controller does not impose an earlier outer timeout that could kill the smoke before this cleanup receipt is produced. The probe and its failure cleanup are neither a campaign execution nor an actor measurement. Probe failure produces exact same-capsule pre-consumption PENDING, consumes no authorization, and creates no campaign or actor intent; the cooperating operator must resolve and prove any reported cleanup or absence failure before re-entry.

## Fresh local execution boundary

V6 uses independent paths:

- run root `/home/erzhu419/.local/state/hegel-machine/q1-start-v6-run`;
- ledger root `/home/erzhu419/.local/state/hegel-machine/q1-start-v6-ledger`;
- campaign lock `/home/erzhu419/.local/state/hegel-machine/q1-start-v6.campaign.lock`;
- Rust build `/home/erzhu419/.local/state/hegel-machine/q1-start-v6-build/hegel-q1-formal-archive-materializer`;
- successor bundle `/home/erzhu419/.local/state/hegel-machine/q1-prestart-successor-v6-bundle/candidate.json`;
- final artifact `/home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine/artifacts/phase3_q1_start_v6.cbor`.

The controller source closure retains the complete historical v5 config, documentation, successor candidate, campaign core, supervisor, and controller, and additionally binds their v6 successors. It also binds the v2/v3/v4 terminal lineage validators, the existing v1 formal-actor modules, pinned local container images, Rust sources and lockfile, and the read-only seccomp profile. The endpoint scratch envelope remains exactly `134217728` bytes. Target truth, split, and role inputs are absent. Network access is prohibited.

## PENDING and terminal rules

Before consumption, each controller entry may invoke exactly one bound independent `/usr/bin/python3.10 -I -S -B tools/phase3_q1_start_supervisor_v6.py --startup-smoke --run-spec ...` Popen. This smoke, including its single ownership/freezability probe, is neither a campaign-supervisor invocation nor an actor invocation. Its strict receipt must pass before consumption. A smoke, probe, local-path, dependency, or capacity failure produces exact same-capsule PENDING: it consumes no authorization, starts no campaign, and creates no actor intent.

After consumption, PENDING is allowed only at a clean completed prefix `[]`, `[python]`, or `[python, rust]`, before the next actor intent. Reopening replays all bindings and never repeats a completed actor. Once host intent exists, no PENDING route remains. Any post-intent unknown outcome, unverifiable actor artifact, unproved cleanup or absence, inconsistent ledger state, or other non-replayable condition is terminal no-resume.

Partial finalization may only complete a missing outcome or terminal record from exact durable evidence. It cannot call the supervisor or any actor again.

## Success and claim boundary

Success alone publishes Q1 as `COMPLETE`, 20/20, mask `1048575`, with eight non-null output roots. Q2 remains `NOT_RUN`; M3 roots, role evaluation, and outside certificates remain null or false. This campaign is target-blind, local-only, non-adopting, and non-promoting. It does not update a current pointer, enter `ACTIVE` or production, or make an external scientific claim.
