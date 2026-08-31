# Hegel Machine Phase 3A — Q1 target-blind start successor v7

The active outer instance is `v7r1`. It supersedes the unconsumed v7
authorization after startup smoke found that the supervisor compared the four
quantitative trusted-host limits against the complete seven-field bound
resource envelope. The repaired comparison includes the three already-bound
provenance fields. No v7 actor or campaign execution occurred, and no v7 run,
ledger, bundle, build, output, or scratch work product is reused.

## Purpose and authority

V7 is a distinct outer campaign. It does not resume Q1-start-v6 and cannot reuse any v2–v6 output, scratch, run, ledger, capsule, candidate-bundle, or build work product. One future exact v7 owner authorization may cover one logical target-blind campaign with the strict actor order `Python -> Rust -> trusted host replay`; each fresh v7 actor may receive at most one durable intent and one Popen.

The authorized scope is `TARGET_BLIND_Q1_START_V7_ONCE_WITH_BOUND_SUCCESSOR_PRESTART_BUNDLE`. The pure successor candidate has no execution, adoption, promotion, or publication authority.

## Failed v6 terminal lineage

The v7 contract binds the complete v6 terminal boundary:

- capsule digest `sha256:3c1eb4a851de182d63f47dae7ee7118e4eda362508af73273fc27d1a3e618dd8`, raw `sha256:af3c4d9c884a2784926271c18818685c7dbfaf19237a745f2d15158512dfc35f`;
- authorization digest `sha256:81136d206ce0abb58d5082e3594a6c74b979b72d0397c3f6ef68ed8578f12748`, raw `sha256:d5d72e5001501c94ad986fa25d028e7f4c91a85bc847efaf3f6972fb42df49ea`;
- consumption record digest `sha256:62f2eced52929e9afb6a96753ac1629534b79bc8df9b5477c45ad2ef9b72ad2c`, raw `sha256:9935bdaaa057d809f2eb8a534cd9cc1b086c1c16db89d2fc1cb810db741efed4`;
- supervisor terminal-outcome raw `sha256:5bd6c0cc0c88fa276ad5e3d56c6438fd54cc3eb94f3a2504046cf0b68030c50d` and controller outcome raw `sha256:024f1ed1124a7613ba2a474ca52026b5fb74a32b5c898accc2faaa0d7cc9dc25`;
- terminal record digest `sha256:b7feb61fec6a2049ede6903d3f43d47894b70345717a1653e557afea00f1cc0d`, raw `sha256:49d1b2e6e555173727ce3509e669137c9c6d1f5c66a4b3e0d3d7b4ea7c1ece10`;
- status `Q1_START_V6_INCONCLUSIVE_TERMINAL_NO_RESUME`, failure `Q1_START_ACTOR_NONZERO`, detail `host:137`;
- actor starts `python=1`, `rust=1`, `host=1`, completed prefix `[python,rust]`;
- Q1 remains `NOT_RUN`, 14/20, mask `16383`, with eight null output slots.

All three v6 actions conservatively count. V6 is terminal no-resume, and its work products are not reusable by v7.

## OOM evidence and formal-host memory ceiling

The external read-only Docker event receipt `/home/erzhu419/.local/state/hegel-machine/q1-start-v6-host-oom-docker-events.jsonl`, raw `sha256:ccd24646539216743d7fd7ff1e6e1a12a2684e303f684f68ae5342b59d22a52d`, records the exact v6 host-container lifecycle `create, attach, start, oom, die, destroy`, with exit code 137. It explains the known v6 host failure; it does not authorize resumption.

V7 binds `host_memory_bytes=2147483648`. This is a conservative formal-host ceiling, not a measured peak. The older capacity diagnostic did not measure the formal trusted-host replay under this campaign shape, so no diagnostic host resource peak may be reused or reported as a formal-host peak. The endpoint scratch observations remain the previously bound v6 capacity evidence and continue to derive the exact `134217728`-byte endpoint scratch guard.

## Fresh paths and closure

V7 uses fresh roots:

- run `/home/erzhu419/.local/state/hegel-machine/q1-start-v7r1-run`;
- ledger `/home/erzhu419/.local/state/hegel-machine/q1-start-v7r1-ledger`;
- lock `/home/erzhu419/.local/state/hegel-machine/q1-start-v7r1.campaign.lock`;
- Rust binary `/home/erzhu419/.local/state/hegel-machine/q1-start-v7r1-build/hegel-q1-formal-archive-materializer`;
- successor bundle `/home/erzhu419/.local/state/hegel-machine/q1-prestart-successor-v7r1-bundle/candidate.json`;
- artifact `/home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine/artifacts/phase3_q1_start_v7r1.cbor`.

The controller source closure retains the necessary v2–v6 lineage replay sources and binds the v7 config, documentation, candidate, campaign core, controller, supervisor, and their v7 tests. Target truth, split, and role inputs remain absent. Network access is prohibited.

## PENDING, terminal, and success boundaries

Before consumption, the bound independent startup smoke may run once per controller entry. A pre-consumption failure produces same-capsule PENDING without consuming authorization, starting the campaign, or invoking an actor. After consumption, PENDING is allowed only at clean prefixes `[]`, `[python]`, or `[python,rust]`; completed actors cannot run again. Host intent leaves no PENDING route. Any post-intent unknown or unverifiable state is terminal no-resume.

Only success may publish Q1 as `COMPLETE`, 20/20, mask `1048575`, with eight non-null output roots. Q2 remains `NOT_RUN`; M3 roots, role evaluation, and outside certificates remain null or false. V7 is local-only, target-blind, non-adopting, non-promoting, and cannot update a current pointer or enter `ACTIVE` or production.
