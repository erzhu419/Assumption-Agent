# Hegel Machine Q0.5b Shadow Retry Campaign v2r19 (record schema `/2`)

Status: `V2R19_FINAL_BOUND_AWAITING_EXACT_OWNER_AUTHORIZATION_NOT_EXECUTED`

This document accompanies
`config/phase3_q05b_shadow_retry_campaign_v2.json`.  The unchanged filename and
all structural schema tags ending in `/2` describe the retained record format;
the distinct contract ID, hash domains, roots, and trace IDs identify the new
v2r19 campaign instance under contract ID
`hegel-phase3a-q05b-shadow-retry-campaign-v2r19`.  The controller source,
tests, document, exact V2R19 hash-domain registry, roots, and trace IDs are
the source candidate that the final loadable contract must freeze.  Neither the
not-yet-final-bound
contract nor its binding summaries are an execution capsule, authorization
record, preflight receipt, or execution result.  The revised target supervisor
is Target C, occupying the retained campaign-contract `target_commit_a_binding`
role at exact commit
`303c0fcd45bb46c00b79bb2112bd1d777364e5bb`, tree
`d8a17e0ca7166eab5db46a87150c37b514f79648`, parent
`1e609a3123b5b80d5dada42f18758d3598467d5c`, and raw commit-object SHA-256
`d81a67bc0d54e88269fb1c2a6ac24306742d81b89b80b1715d5ac3053b13365a`.
Its thirteen required source hashes, static-policy root
`67b0cb763fa2c81b256c1831fd5d75f4cc70ab315c90a1cbf1ac2111d29cba96`,
command/mount/resource-policy root
`932064f93df61882cd0d35ebbb25a17818146b54c48fadf96967eaed49498a0f`,
trajectory-core raw SHA-256
`634826c13ee51c72c6e87c7b40f657ee315c3a91446ee12f2d807cdccc3030ed`,
target contract digest
`sha256:159b4e80b60d59005bd96b2b38136cb6504f0700319be0cd99886c3d487569f6`,
and target source-binding digest
`sha256:97963373458ffb333e959cfff6624d322c6da3f63cda66c0e444b7cb436d6713`
are independently replayed by the controller.
The read-only preflight accepts the controller source only when its clean live
`HEAD` is a non-merge commit with Commit A as its sole parent and its raw
NUL-delimited diff contains exactly the six campaign paths frozen by the
contract.  Thus the future frozen Commit B is the sole one-commit controller
layer admitted to capsule derivation; a clean later descendant, merge, path
omission, reordering, rename, or extra path is rejected.
No v2r19 preflight, final-commit live gauntlet, capsule derivation,
authorization, or campaign execution has occurred at this source-implementation
checkpoint.

The prior v2 capsule
`sha256:238b79a5ddf5adb379df9ac1b0ae7e762f7616f1cfbf19b93474315a9197ab62`
and authorization
`sha256:0f59cc4aa2f605bdd1388baf769961ccfebf2123c6d6c9ff467f7f61add29cea`
belong only to the completed prior-v2 terminal/completion chain.  That chain is
historical, terminal, non-resumable evidence.  Neither digest, its immutable
ledger, nor any prior-v2 root is v2r19 authority or an input that the v2r19
preflight, record-authorization, or run entry point may open, access, reuse,
copy, adopt, or resume.

The completed v2r1 capsule
`sha256:62d6d7a56107439261ce66788e5d803f7d1370733673416076131e6d20a16e61`
and authorization
`sha256:2a410cff6c59895f941075af5cc6035f61dea4ba4734d82eb0719c5b23a1bb90`
are also historical, terminal, and non-resumable.  V2r1 ran exactly two
supervisor slots.  Slot 1 exposed one exact Python sampling-gap origin and
correctly opened Slot 2.  Slot 2 durably recorded two independent exact gap
origins, Python and Rust, with seven total failure events.  Its five-row
projection was explicitly marked truncated, so the single-origin v2r1
predicate failed closed and did not open Slot 3.  The authoritative terminal
ledger record is
`sha256:b910df6dfea1e0bd8fd4bd11d8aa20d682f43898b495f8c643b668b970f8108b`
and the completion ledger record is
`sha256:b9e63585d7a6eb48d6195fa188ce1d40f2c79a169848f58cd10b82328be4702b`.
v2r19 protects the v2r1 worktree, authorization, ledger, and all fifteen slot
roots as negative historical registry entries.  The v2r19 preflight,
record-authorization, and run entry points must not open or reuse them.

The completed v2r2 campaign is likewise historical, terminal, non-resumable
evidence.  Its frozen controller commit is
`33ff8ba17addf914f06eb440b8b7a40e93654816`, its tree is
`bafb4291cdc003e514e6ae36800588f3e37242e7`, and its campaign source-binding
digest is
`sha256:f1bf112b268c5fdf1a104e9e66f92e21bc8187347657e0556ea47c3bd8815079`.
It retained target Commit A `7bc2044212e278443b1ca703115c7cb17591ac14`.
Its controller contract digest is
`sha256:02e297682d99e2efa7818b2b58562dd6b6dc3ef468e20b980240e2ab1e19c871`,
capsule digest is
`sha256:325fc2d916e021d811b5ca0bce2bba4a2f89282e5529d9471670a4c8cd34648d`,
owner-confirmation digest is
`sha256:a1e8b8d688d239aec60b30094e34a470bb889290d7101a8683a9a8fd951c0f52`,
and authorization digest is
`sha256:079730f604279e5da1e2aa81c22ae4a00806f5981659622fdec3da2efde836c9`.
It invoked all three strictly ordered one-shot slots.  Each slot durably
recorded a complete 42-event Python/Rust dual-origin exact-gap trace with all
seven failure rows and all eight cleanup rows retained, zero unpartitioned
post-origin events, no torn tail, and no open span.  Slot 3 therefore ended as
`CAMPAIGN_RETRY_EXHAUSTED_TERMINAL_NO_SLOT_04`.  The authoritative terminal
ledger record is
`sha256:bb312d9774a57723a4b59e3a31a4adfd0ad6abef6c3b12cd3bc852e9b6280f78`
with terminal-body digest
`sha256:61d2b63a73e781351a65d49c638498ddc3a66fe8972eaa1ed214b73f0e33edde`;
the authoritative completion ledger record is
`sha256:1d3c632731315906eea3ef104d4f73628ab8aeeb62af693055d69be02da03d09`.
Its completion-body digest is
`sha256:55b8b25cc990b56bbda03930d90bda14e9329546d136812ca8714a8b2d0cef85`.
The complete trajectories identified synchronous shared trajectory
publication, including its global writer lock and per-event fsync path, as
part of the observed sampler scheduling delay.  That evidence motivates the
v2r3 sampler-publication separation and workload-bound scheduling admission;
it grants no authority to reuse or resume v2r2.  V2r10 protects the v2r2
worktree, authorization, ledger, and all fifteen slot roots as negative
historical registry entries that its preflight, record-authorization, and run
entry points must not open.

The completed v2r3 campaign is a distinct historical terminal chain, not a
partially available v2r19 run.  Its controller Commit B is
`bcf22f321478ace834b2bc4c0953960812f74d45`, tree
`ef5f42336b5b6490396491062fe6b699c65b96d5`; its target Commit A is
`fdf8e22f6811fe3062b4409dad48cf9f044f2919`, tree
`7d3daf0088e24e223857f3f0daf789e2ce73c710`.  Its campaign source-binding,
contract, capsule, owner-confirmation, and authorization digests are,
respectively,
`sha256:2fe4c2b80e0218dd1aaefe99cf431701ef0350477f828254c1bb6f4a7cb81de9`,
`sha256:4be13ab3045cc67ff9087ab3e24c6e6327ec1a8efc0be99e4d2befb391282828`,
`sha256:150e92c65f870df8591655fba3d6798d60bfecf52693e9b30719385e93f0cc24`,
`sha256:43dc31c26c9853a5d8d507c64eb402d6e3b10d882e4df640fd75fbde16543ac7`,
and
`sha256:7a32ebc4fe0da30df71d74698733e7734080dc15138e3429af22bf54d5f25969`.
It invoked only `SLOT_01`.  Its `Q05B_SUPERVISOR_V3` trace
`q05b-shadow-retry-campaign-v2r3-slot-01` ended as an `INCOMPLETE_PREFIX`
after 5,410 durable events, with six open spans, no terminal manifest, and no
partition certificate.  Consequently, its authoritative decision was
`TERMINAL_NONRETRYABLE`, not retry eligibility.  The campaign terminal status
is `CAMPAIGN_TRAJECTORY_UNCERTAIN_TERMINAL_NO_RESUME`; its terminal-body and
completion-body digests are
`sha256:bd372c290bb4809479e3344da3a090a8f03a426bf04549d26174b41c72af089f`
and
`sha256:41280c9afdf42ab951617408b97f5526233b8eea94c63f09adc0fa6a8c229c96`,
and its terminal and completion ledger-record digests are
`sha256:6b4542757feefa7d26e4c537a66eae3f5486a0d1f3e30f74c4c340f0c833646f`
and
`sha256:b0adfb9cc65d5422918fb498f10fac534bb6c60ea71d6611c701553628984007`.
The eight-record ledger and completion record close that authority
permanently; `SLOT_02` and `SLOT_03` were never executed.  The absent
partition certificate and unknown unpartitioned-event count must not be
rewritten as zero or as complete-failure evidence.  V2r10 protects the v2r3
worktree, authorization root, ledger root, and all fifteen slot roots as
negative historical entries.  Its preflight, record-authorization, and run
entry points may not open, access, mutate, copy, resume, or reuse them, and may
not reuse any v2r3 capsule, owner confirmation, authorization, ledger prefix,
trace prefix, slot, or completion authority.

The completed v2r4 campaign is also historical, terminal, and non-resumable.
Its controller Commit B
`a2fac3ae10789eb2be34ad33f8ef9785d312eb53`, tree
`7d9e3c36ce10d00f2458c83d8bf16d60512f007e`, had exact parent and target
Commit A `abaedf57e9795515d9b7ca6b04f905d87f30d5e3`.  Its campaign
source-binding, contract, capsule, owner-confirmation, and authorization
digests were, respectively,
`sha256:76f03e0c55c54b95a9288d83d636f25bd04eb6a599bc6515e05231d92af7ce18`,
`sha256:09e2f40cceed32d38bd0e3d4a04f620593b76e8ff68f62e86ff35dd7ffa68cb8`,
`sha256:8646ad08037d681e8e619de66f5d23b15ebedccee0234108881488e758129a46`,
`sha256:cd083fef0b3f10636d972e4d24ffce2eb1152bf794bc40e5afdef5e474114742`,
and
`sha256:1bc5b4da4e67368a633ab4cb708099c0e8f09761e002fdb654397dd4e4d70e26`.
The authorization was consumed exactly once.  All three pre-consume
scheduling-admission windows failed; their timed durable counts were 570,
405, and 490 of 4,800, and all retained files were removed with absence
proved.  No slot was consumed, no supervisor was invoked, and no trajectory
or artifact was created.  The terminal status/error were
`CAMPAIGN_FAILED_CLOSED_TERMINAL` / `FAIL_CAMPAIGN_SCHEDULING_ADMISSION`;
terminal and completion ledger-record digests were
`sha256:b64f61ef3cf57419fa902dca6b58cf1bb49ad45b87137f8264099640305341bf`
and
`sha256:7d3a30a7fc8749e9a05a06f8c47ee01bcc06223925f20be50f49e3874ab8c74c`.

The v2r4 receipt exposed a gate-harness defect rather than a need to weaken
the 40-record/s requirement: the declared five-fsync path itself measured
42.81--44.35 records/s, but the retained-probe success `finally` performed an
undeclared sixth directory fsync after timing stopped.  That extra 5.58--5.91
ms per record reduced end-to-end worker throughput to 34.17--35.52 records/s
and filled the 64-record admission allowance.  V2r5 removes only that hidden
success-path sync, advances the campaign admission semantics to `/5`, and
retains the target five-fsync durability protocol and every frozen rate,
cardinality, queue, and gap threshold.  The v2r4 worktree and its seventeen
state roots are protected negative historical entries.  The v2r19 preflight,
record-authorization, and run entry points must not open them, and they cannot
be reused or resumed by v2r19.

The completed v2r5 campaign is an earlier executed campaign predecessor and
is terminal, non-resumable evidence only.  Its
frozen controller Commit B
was `8c1018518bb885ba4821ccaf10f2a3fb17f86a35`; its target was
`abaedf57e9795515d9b7ca6b04f905d87f30d5e3`.  Capsule
`sha256:ba1d195b17fea2cc461a50db50bc08fd45f63d2858ba608257a733ad3846d74d`
was authorized and consumed exactly once.  `SLOT_01` passed scheduling
admission, invoked the supervisor once, and closed a complete 6,338-event
trajectory with one exact Rust sampling-gap origin
(`290,330,097 ns > 250,000,000 ns`); its decision was
`RETRY_EXACT_GAP`, so and only so did `SLOT_02` become eligible.  `SLOT_02`
also passed scheduling admission and invoked the supervisor once.  It closed
a complete 13,602-event trajectory at Stage 8 with
`FAIL_Q05B_DUAL_ARTIFACT`: the isolated Python 3.10 process could not import
ambient top-level `tomli` (`No module named 'tomli'`).  That failure was
`TERMINAL_NONRETRYABLE`; `SLOT_03` was never invoked.  The campaign therefore
ended `CAMPAIGN_FAILED_CLOSED_TERMINAL` after two total supervisor
invocations.  Its exact twelve-record immutable ledger ends at completion
record digest
`sha256:a2ddd0e6493ed8023143052df672fa7425c39f2d5a4f0ba59df7d59bc9fd1f42`.
The v2r5 worktree, authorization, ledger, and fifteen slot roots are now
protected negative history.  The v2r19 preflight, record-authorization, and run
entry points must not open those roots, and none of the v2r5 capsule,
authorization, ledger prefix, paths, slots, or completion authority can be
reused by v2r19.

The v2r6 controller is a distinct failed historical source-bound version, not
a partially available v2r19 run.  Its exact controller commit was
`d0701ad308fca2d27818cbd6cab861dd47c8cdcd`, tree
`2bef012c37fc86c88e674697b2fb6e6a0fd4d97a`, with sole parent and unchanged
target Commit A `5d85bf858e80a22633f96de41259344c70e25519`.  Its campaign source-binding
and contract digests were
`sha256:883ae7ec454833bc3e6c1112422e88f0ccf428159309681b6779d14c00290ab8`
and
`sha256:47d7a676d836bc1c2c44ffb79f7b3de4c0ad658d95df3ca800ae750fff0be130`.
Exactly one CLI preflight returned exit code 1 and canonical
`CAMPAIGN_COMMAND_FAILED_CLOSED` error `FAIL_CAMPAIGN_SOURCE: target derived
binding differs`.  The 290-byte stdout had raw SHA-256
`8119c23e0df62403f01b3eddb5508664f0d49410e41f72b3196300c4c21fa1bb`;
stderr was empty.  The controller had derived the exact Target-A source
preimage but v2r6 incorrectly bound the domain digest computed with the
canonical transport LF retained,
`sha256:5dde4a55053e2e6ffc26f3c6b485bd1c92b001b68046431f878793d90e3c27c6`,
instead of the required no-final-LF digest
`sha256:8b24967b9fca37cb0f7f741b6ebc356dbeded597b567cca1ddd11c3ce61fe61a`.
The failure occurred before the Docker-label check, dependency child, capsule,
authorization lookup/publication, any state write or consumption, and any
supervisor spawn.  The 16 campaign freshness targets checked by the controller
were absent; independent before/after observation found all 17 v2r6 runtime
roots absent.  No capsule, owner confirmation, authorization, ledger record,
dependency-gate invocation, or supervisor invocation exists.  The clean v2r6
worktree and those 17 absent runtime roots are now 18 protected historical
roots.  The v2r19 preflight, record-authorization, and run entry points must not
open them, and their paths or failed binding cannot be reused by v2r19.

The completed v2r7 campaign is an earlier campaign predecessor and is
terminal, non-resumable evidence only.  Its frozen controller Commit B was
`18bbcc62b7ed74cc4c479494cda0a3323321f628`, tree
`dc01db925c911c29f8d64f6bdd82edb0c6863638`, with sole parent and target
Commit A `5d85bf858e80a22633f96de41259344c70e25519`, target tree
`aca3fc65be392c51c829544bfda8ae9149fc566b`.  Its campaign source-binding,
contract, capsule, owner-confirmation, dependency-preflight receipt, and
authorization digests were, respectively,
`sha256:78b2a44f0c511446d128a8d0bc2dbd8d7531afda081ac498a6a00cc7d02b7549`,
`sha256:7999c4a70812f45552c638ca01ad6b542858f5501bc42b717c71bbc47aebdbef`,
`sha256:568a76b249fc9f37c16e296e8b1e777bf80b83f7d7feccbdf1579837f7437b22`,
`sha256:8719cf0a11c2da930f1da7d345ba99b261304f89ae8b5aeba8deb488183d3e58`,
`sha256:d94e836471d6b4da87324197e9958a075e3ed79b74d0dbaf79d375402c34d221`,
and
`sha256:7f4b09c4e3f7f3f441b4e1572f3e08cf9bbe5451906adc069f8d0b66c4e44663`.
That authorization was consumed exactly once.  The authoritative five-record
ledger is, in order, `campaign.intent.json`,
`campaign.authorization-consumed.json`, `slot-01.intent.json`,
`campaign.terminal.json`, and `campaign.completion.json`, with record digests
`sha256:9cac7e1a6ec7e222d54c658282790e44f1fcc08b8e4f6be67463a3aee13b4f8e`,
`sha256:9f738ff92376f8fb71e4d66f80c31f0fc5f4e92d7a2db8047e5cffaac286204d`,
`sha256:ae2d395311e5f5a7f31d9c127985dfaff6eb62a23bed04eb8ee2e00fead2be8b`,
`sha256:27f492d3682edefd592688f9cf2b06f4d9d42cb3503ca364e215c7c80a05d2fa`,
and
`sha256:423a52720a358320db155d9496def9be3dc9a6008a7b1be63090956532bd190a`.
All three pre-consume scheduling-admission windows failed.  Their timed
durable-by-deadline counts were 3,010, 4,776, and 2,455 of 4,800; their late
durable counts were 0, 24, and 0; and their retained counts were 8,421,
10,211, and 7,866.  The middle window enqueued all 4,800 timed records with
zero drop, duplicate, or reorder and drained the final 24 records in
776,134,344 ns, but the frozen requirement is zero durability after the
deadline.  `SLOT_01` was staged but not consumed, no supervisor was invoked,
and no trajectory or artifact was created.  The terminal status/error were
`CAMPAIGN_FAILED_CLOSED_TERMINAL` /
`FAIL_CAMPAIGN_SCHEDULING_ADMISSION`; terminal-body and completion-body
digests were
`sha256:f5bd4b123af1cf3d4bb522cb5b644b3b2e81608d4fd53f5237b262dfc803fce4`
and
`sha256:df1fba1e98e5d2ddaf8fc247e5e1c584bfdddbf68a293c09d4ecff332e59ac6f`.
Its worktree, authorization, ledger, and fifteen slot roots are now 18
protected historical roots.  The v2r19 preflight, record-authorization, and run
entry points must not open them; none can be reused or resumed by v2r19.

The completed v2r8 campaign is an earlier campaign predecessor and is
terminal, non-resumable evidence only.  Its controller Commit B was
`b80473df5ae2515896d2e841b832636f09702f47`, tree
`3eab5e68b4800b5ea49c9c1edde4a0822e3ca724`; its target Commit A was
`8547fe0f084f3c914601b3f1a25f3957c53ab559`.  Its campaign tool and config
raw SHA-256 values were
`f756d70fadc6553e58fc8d80fc3c001b39de27ab09798a8db3a85966ed4a603a`
and
`d9a5f496c7b3dffb66b403ef3c88ae5cde4b5f83e12338200984d02c98de0fa6`.
Capsule
`sha256:9264b2cd48822bed11bd31650c491dc9fb06f9aa498dfbf75958e898824d839c`
and authorization
`sha256:99c1f29453b0c0cb9227c6536d60f9648170d6ff21dd907edcff1050fc9127fa`
were consumed exactly once.  The four-fsync scheduling admission succeeded in
its first and only window: all sixteen bursts passed, all 4,800 timed records
were durable by deadline, zero were late, dropped, duplicated, or reordered,
the queue high-water mark was 8, and all 10,211 retained records were removed
with absence proved.  `SLOT_01` then invoked the supervisor exactly once and
closed a complete 14,536-event `COMPLETE_FAIL` trajectory.  Its two failure
rows were `SUPERVISOR_STAGE` at
`STAGE_08_BUILD_AND_STRICTLY_DECODE_NINETEEN_ROW_CANDIDATE` and
`SUPERVISOR_RUN`, with exact detail `candidate evidence strict replay failed:
REJECT_Q05B_ARTIFACT_HOST: loaded module closure is incomplete`.  The artifact
was absent and that absence, cleanup, process-group absence, Docker absence,
and supervisor reap were all verified.  The decision was
`TERMINAL_NONRETRYABLE`; `SLOT_02` and `SLOT_03` were never invoked.

The authoritative v2r8 ledger has exactly eight records, in order:
`campaign.intent.json`, `campaign.authorization-consumed.json`,
`slot-01.intent.json`, `slot-01.consumed.json`,
`slot-01.trajectory-binding.json`, `slot-01.outcome.json`,
`campaign.terminal.json`, and `campaign.completion.json`.  Its terminal and
completion record digests are
`sha256:29a6b1670a41ca1902d0f8620381d98a27cee4ad438a7e43ec922bfdb157604c`
and
`sha256:95a6795f64b24412091cd2eb03f1da00546371161c2da4a680eafe8ae9171c6e`;
their body digests are
`sha256:c99cb483fc0f11eb5b066a190b95e073ed76bafbf68b26129bbd559489dd7b10`
and
`sha256:787c1742c70e7c304a597f2c6e7e325f7b3db35e58730eeb95fb1ee81b1eb172`.
The v2r8 worktree and its seventeen runtime roots are 18 protected historical
roots.  Six runtime roots are present and eleven are absent.  The v2r19
preflight, record-authorization, and run entry points must not open them; none
may be reused, resumed, or treated as v2r19 authority.

The unexecuted v2r9 controller is a retired source predecessor,
not a partially available v2r19 campaign.  Its Commit B is
`e9e038cf73f230ecef07a714c6a863be91189f7c`, tree
`d1d114407d14a04abc08db933473a00f122734e9`, with sole parent and Target A
`dd004ef75a67ccd14de8d6e6b6e4e0bd80c971b7`.  Its campaign source-binding
digest is
`sha256:b7a577919eaff4195b3bef97e3e55941eda30bb3cd31128a1018352057b810fd`
and its contract digest is
`sha256:0ec1075dd97d198d04ce5c49cb8d90c5eacb7de376f9753e3727c5c37a309daf`.
The derived v2r9 capsule
`sha256:83e16afcbd75d499b7f8649f822a8aacf9c4b6d97248d4d92d3e8b316d9c01e2`
is retired.  It was never owner-authorized: record-authorization, run, and
supervisor invocation counts are all zero.  No authorization or ledger digest
is asserted for v2r9, and neither the capsule nor any v2r9 authorization or
ledger namespace may be reused.  Its worktree and seventeen runtime roots are
18 protected historical entries; v2r19 preflight, record-authorization, and
run must not open the historical runtime roots.

The v2r9 source version fixed only the target host-closure expectation that
caused the Stage 8 failure: the pure target-blind
`hegel_machine.phase3_q05b_actual_admission_v2` module, which the supported
internal-host execution path loads before replay, is now one exact required
root.  The fresh-process known-answer test accepts the resulting exact
39-module closure and rejects both removal of that module and addition of any
unlisted module.  The historical v2r12 target retained that Target A unchanged.
V2r13 preserves this host-closure fix and the v2r8 four-fsync admission path,
rate, deadline, cardinality, queue, and gap requirements while its new Target A
repairs only the reachable held-control strict-replay contract: one file fsync
after `fchmod`, three directory fsyncs, and no success-cleanup fsync remain
unchanged.

The v2r10 controller is now a retired, non-resumable historical campaign
instance.  Its exact Commit B was
`4167f220a9a28a419150c9a656495e7173890f12`, tree
`c36dd1ba10292db40663a7552eccbbab62e76dfe`, with sole parent and unchanged
Target A `dd004ef75a67ccd14de8d6e6b6e4e0bd80c971b7`.  Its campaign
source-binding and contract digests were
`sha256:7021341bf68fca30ad5004351da920f65d3eec49c93924c09d7194b947f66931`
and
`sha256:0e0b759ed8292cdbbe4ebcf9b67d3e0f1ee32f9d59972c8170b63daef9eed9e4`.
Capsule
`sha256:4ba7e59fe951e3c5b7c5aee0d8b4a1c4d3f74abec6099b49ee2d903af0c362ab`,
owner confirmation
`sha256:280c6ad54cfff44c89ff6d7ca798158ba8890b8caeaa25b64b45b44a357342ba`,
and authorization
`sha256:fd9042ea3c72671962e80eff78264a9e8905144ef40cf5921fa5c7746c972da2`
belong only to v2r10 and are retired from v2r19 authority.

V2r10 record-authorization ran once and passed.  The resulting immutable
record was the regular mode-`0444`, single-link, 11,457-byte
`campaign.owner-authorization.json` in the v2r10 authorization root, with raw
SHA-256
`fee6dd6dd58fcfa1d67f104687e419a6439584ed1562357bdcec0575e8ccccb8`.
Its only run entry ran once, but the target full strict fsck reached the old
120-second generic Git bound before authorization lookup, ledger construction,
slot construction, scheduling admission, or supervisor launch.  The bounded
process returned `-9`, `timed_out=true`, empty stdout and stderr, and certain
cleanup.  The isolated dependency gate had not yet been invoked because the
v2r10 ordering placed fsck first.  Authorization was neither loaded nor
consumed; the ledger and all fifteen slot roots remained absent; supervisor
invocation count remained zero; and no process survived.  Its canonical CLI
stdout was 1,300 bytes with raw SHA-256
`b9409e278864053156d095c43ac4ab00992148b2d24f30ccd9422fcc463db777`.
An independent later read-only diagnostic gave the same fsck a 900-second outer
guard and observed exit 0, empty streams, and approximately 128 seconds of
elapsed time, establishing that the old budget—not the object closure—caused
the timeout.  That diagnostic was not a v2r10 run or authorization reuse.

The v2r10 worktree and its seventeen runtime roots are 18 protected historical
entries.  Only the authorization root exists; the ledger and fifteen slot
roots are absent.  V2r19 preflight, record-authorization, and run must not open,
reuse, resume, adopt, or mutate any of them, and neither the v2r10 capsule nor
authorization can authorize v2r19.

The completed v2r11 campaign is an earlier executed predecessor and is now
historical, terminal, and non-resumable.  Its exact controller Commit B was
`b448af3987a5cda9ad42172d60f22b5a70f90dd8`, tree
`57ae0c99998b80079748cd6ed35a6e6623dac7f1`, with sole parent and target
Commit A `dd004ef75a67ccd14de8d6e6b6e4e0bd80c971b7`.  Its campaign
source-binding and contract digests were
`sha256:51bec9e972309b86c56ee263f865795ceba0df657e91754d9b0fcfaa37e34956`
and
`sha256:77add7f3897f058417aa86aa0dc137db9bf3025a8c8b478087b3d25fe12aeda0`.
Capsule
`sha256:9b8075a79cc38dde0b24fc96868e915f52afdff03b365d698b76a88a2e9b008a`,
owner confirmation
`sha256:29644b5720d38055a94d635e0ef9e0e55abb4060efb26a89cae4209230deb58d`,
and authorization
`sha256:de4dc2992384d71244a32c98d42bb8ea3b853db7dd255297e159006273bc764b`
belong only to that completed chain.

V2r11 consumed its authorization exactly once, wrote the campaign intent and
`SLOT_01` intent, then exhausted all three scheduling-admission windows before
any slot consumption or supervisor launch.  Its exact timed records durable by
deadline were 645, 730, and 470 of 4,800; retained counts were 6,056, 6,141,
and 5,881.  The scheduling receipt used schema
`hegel-phase3a-q05b-shadow-retry-scheduling-admission-evidence/6`, bound
`SLOT_01`, selected no window, and recorded `passed=false`.  Under the frozen
v2r11 semantics this produced terminal status/error
`CAMPAIGN_FAILED_CLOSED_TERMINAL` /
`FAIL_CAMPAIGN_SCHEDULING_ADMISSION` with zero supervisor invocations.  The
terminal ledger/body digests are
`sha256:0845e1f697a02efd4eacc423d7df189d35d58ad5f8e6251e5228a376ae1e98b3`
and
`sha256:44ca26bc97eb5dd2c8ea9f6dda5d96c3af4074aebdd3ee9a07c1576e4ab9dce3`;
the completion ledger/body digests are
`sha256:46d25b7b97f6f846ea33dcad6ffb7aa70478248f6e48be1cc905f63871199cdf`
and
`sha256:396b6fba31059bc9fdc9eeac7e51e29172a875832809598bc09064bffbd82921`.
Producer-free offline replay returned
`AUTHORITATIVE_CAMPAIGN_COMPLETION_VERIFIED` for the exact five-record chain.
The run stdout was 81,643 bytes with raw SHA-256
`9f9764ac3f838956ce976e5e5c886bf65db36536bcc318c846cca096685857a0`;
stderr was empty.

The v2r11 worktree and all seventeen runtime roots are 18 protected historical
entries.  In registry order, the authorization and ledger roots, then the
`SLOT_01` clone, Cargo cache, and evidence roots are present; the `SLOT_01`
artifact and trace roots plus all ten `SLOT_02`/`SLOT_03` roots are absent,
for five present and twelve absent roots.  V2r19 never opens those roots and
may not reuse the v2r11 capsule, authorization, ledger, terminal, or completion
authority.  V2r13's rule that exact complete gate exhaustion is a
clean-checkpoint PENDING result is prospective only; it does not reopen or
reinterpret the completed v2r11 chain.

The completed v2r12 campaign is an executed predecessor of v2r19
and is now historical, terminal, and non-resumable.  Its controller Commit B
was `fb66aa373535c96afec1609c6755acf0fa69767f`, tree
`2e99f1cb312f2f57dd42d429b2f29179f30ba352`, with sole parent and target
Commit A `dd004ef75a67ccd14de8d6e6b6e4e0bd80c971b7`.  Its campaign
source-binding and contract digests were
`sha256:6404f881559b2d2467ab44be7f0ab846c2cd95ea51e6ea9e38f70701e08992cf`
and
`sha256:f883d0d42e27bed0aef59f5abfc382976b69087d283da13c225db3ea07c55e72`.
Capsule
`sha256:0ec5eb3fdc1b7bf044b3ebb5bf9debd1f3d0e0ebf4b4d75d4e60c0816ab55337`,
owner confirmation
`sha256:dd46459aae0801126682bf8257b655b7b1d66f1f7778888867f38f9c59411d9b`,
and authorization
`sha256:cc0d178b7643223a1d837a9a2128dbb982a847b0f89245a4bfe4b1073396bce2`
belong only to that completed chain.

V2r12 consumed its authorization once and invoked two strictly ordered
one-shot slots.  `SLOT_01` produced a complete 72-event, 36-span failure trace
with five retained failure rows, eight retained cleanup rows, no open span, and
no unpartitioned event.  It proved one exact `PYTHON_ENDPOINT` gap at role 1,
sample ordinal 2, sequence 27: previous finish `171265080722396`, scheduled
start `171265205722396`, actual start `171265348280502`, lateness
`142558106 ns`, and gap `267558106 ns` against a `250000000 ns` maximum.
Its decision was `RETRY_EXACT_GAP`, outcome digest was
`sha256:644c22785e18a1b8d9c82f78c06c05347bd6b7a46d85abc6a2a950daba314014`,
and partition-certificate digest was
`sha256:630e2b26a34ef3554d2d9149de02b67ac58a0effd43e285b051a7c76836f6e5a`.

`SLOT_02` reached Stage 8 after Stages 1 through 7 passed.  Its complete trace
contains 23,966 events, 11,983 closed spans, two retained failure rows, four
retained cleanup rows, no open span, and no unpartitioned event.  It then
failed non-retryably with `FAIL_Q05B_DUAL_ARTIFACT` because the producer's
held-control evidence contained `control_root_path` and `root_nlink` while the
strict decoder's exact field registry omitted them.  The exact detail was
`candidate evidence strict replay failed: REJECT_Q05B_ARTIFACT_SCHEMA: held
control identity field registry differs`.  Its decision was
`TERMINAL_NONRETRYABLE`, outcome digest was
`sha256:b207307055c11eef68c9b22212bf65b9e46cdc55bc2472c35403aef4697c53f5`,
canonical-event digest was
`sha256:0cd0987c9f4c18df78c2f21f95fe3e3bbf67feabc762dbe48a70b4aec0ed88c6`,
and partition-certificate digest was
`sha256:b1d8201f8d286811b49e364bca56518249b268f7494c0011aab33c968f8bfd4f`.
Cleanup, process, Docker, and artifact absence were certain for both slots;
`SLOT_03` was never invoked.

The exact twelve-record ledger closed with terminal ledger/body digests
`sha256:e9e1e31691fbde24dedd2eb1a17788ed1a2c816a1e54fde589029f2d8b73a058`
and
`sha256:1a72d7542b5495c2a08d891b8386a1eaea792b129f30d7e0c6ee5ba0998c6e71`,
then completion ledger/body digests
`sha256:a1dd9b0eaa1af0d60cc313dcf356e955d5d4905c5f420b066eec38c2f9f1e75a`
and
`sha256:1566ea6eb39cbef1d5d8a260308ee9542f0b9288f271f580c45c3acfa13b472f`.
Producer-free offline replay returned
`AUTHORITATIVE_CAMPAIGN_COMPLETION_VERIFIED`; the run stdout was 1,271 bytes
with raw SHA-256
`17ccffcd50525c4b0f351d2936d18c2161b925e0a1b0baad90135baa94b5294c`
and stderr was empty.  The v2r12 worktree and twenty runtime roots are 21
protected historical entries: ten runtime roots are present and ten absent.
V2r19 must not open, reuse, resume, adopt, or mutate any of them.

The completed v2r13 campaign is an executed predecessor of v2r19
and is historical, terminal, and non-resumable.  Its Controller B commit/tree
were `414048d12e2f920f30908ee9e3e554a015d97d59` and
`ac8a73745bc11afabaa63c2912bd0ffb132ceece`; its sole parent and Target A were
`0bde03ba2d00887e1590c12d235b96fe07d09b5c`.  Its contract, capsule, owner
confirmation, and authorization digests were respectively
`sha256:fa6e2cc3ca979c98e2d34e0ec2d65fbd01ea6031555e5aa3554e7d0ad319c9cf`,
`sha256:e9471b60c63741f48cba8f9c0b2b4f0fb5054e59ecd7baf0e3756e5d34de9ea0`,
`sha256:087ed59607fb67696cc304167aff3b2d9483a5a917e7f097fae16201ff2e174c`,
and
`sha256:8452455eaeebf1d1015e53dbb723736953b7623572dac681d34623260296d4ea`.

V2r13 consumed authorization once and invoked only `SLOT_01`.  The scheduling
gate passed window 1 with projected gap `127070404 ns`.  Stage 8 successfully
built and strictly decoded the nineteen-row candidate.  Stage 9 then failed
non-retryably with `FAIL_Q05B_DUAL_ARTIFACT`: the canonical artifact exceeded
the frozen 64 MiB maximum.  Its complete trace contains 14,512 events and
7,256 closed spans, no open span, two retained failure rows, four retained
cleanup rows, and no unpartitioned event.  Live Python, Rust, and trusted-host
resource samples numbered 1,575, 1,583, and 4,071; the 7,232 gap-checked rows
all remained within the gap limit.  Both supervisor and external artifacts
were absent, cleanup was certain, and `SLOT_02`/`SLOT_03` were never invoked.

The exact eight-record chain closed with terminal ledger/body digests
`sha256:ed385277b3e28d9989f83d20cd0da5af1da8e6f4b7ff7db5b03cd8fc2d0245dc`
and
`sha256:24a141bdbb6acea09bc9eb1d6e56a925664e7c199bd6d0cdac6ec4ae308de863`,
then completion ledger/body digests
`sha256:025a2bcd8c27b0deed9b186cf30070648ad592a0b2cc80ad677a99c7b9c33fb1`
and
`sha256:0ca34a0e5df75e9ae896225edde050cf9ef6c3bd1b250176cca76f1ef67c6e3e`.
Offline replay returned `AUTHORITATIVE_CAMPAIGN_COMPLETION_VERIFIED`.  The
v2r13 worktree and twenty runtime roots add 21 protected entries: six runtime
roots are present and fourteen absent.  V2r19 must not open, reuse, resume,
adopt, or mutate them.

The completed v2r14 campaign is an executed predecessor of v2r19
and is historical, terminal, and non-resumable.  Its Controller B commit/tree
were `cdd270a55dd80e32a352a1d1bbfb2da12a2fe195` and
`ddedf1afef1cce3dabd5081fa729fc264a7e24bd`; its sole parent and Target A were
`a6c3ac61f0e7d21d7d2dd64d14886a74f87dd97e`.  Its contract, capsule, owner
confirmation, and authorization digests were respectively
`sha256:b5c796d31372bbe297dc24ebb017b7a7cabad7768676b528580cd6993413f7d3`,
`sha256:1c12ecf0edf74c36fd4ade29ccdc8c276ef1774d65f30eb75139b0a13455c537`,
`sha256:3b1406d3a32c061859212243c765142b5627911a17c361a5df86ff0a572b4f00`,
and
`sha256:c947d37e06482c0fd47c48964c5a111d623f3051801348233e516abe1e8661e1`.

V2r14 consumed authorization once and invoked only `SLOT_01`.  Scheduling
admission passed window 2 with maximum projected gap `132131520 ns`.  The
target completed 1,757 Python samples and 1,757 Rust samples, then the
trusted-host held-final sampler failed in Stage 6 at sample ordinal 7,163:
the observed gap was `323536475 ns` against the frozen `250000000 ns`
maximum, with `198536475 ns` scheduler lateness.  The complete trace contains
21,384 events and 10,692 closed spans, no open span, four retained failure
rows, eight retained cleanup rows, and no unpartitioned event.  The failure
chain is exactly held-final origin -> actor close -> Stage 6 -> supervisor run.
The cleanup surface is two trusted-host pairs followed by one Python and one
Rust pair, all successful.  Both artifacts were absent and cleanup, process,
Docker, clone, and Cargo isolation were certain.

The frozen v2r14 classifier nevertheless returned `TERMINAL_NONRETRYABLE`:
its trusted-host single-origin branch allowed only trusted-host cleanup rows,
although the frozen target's outer failure sweep truthfully and idempotently
also cleans the completed Python and Rust endpoint actors.  This was a
controller/producer integration bug, not a rejection by the intended exact-gap
method.  V2r15 changed only that allowlist for a trusted-host Stage-6
single-origin failure: the trusted host remains the required cleanup actor,
while successful cleanup rows from all three registered actors are permitted.
Endpoint-origin Stage-4/Stage-5 rules remain endpoint-only.  Offline v2r19
reclassification of the immutable v2r14 observation is `RETRY_EXACT_GAP`; it
does not rewrite or reopen v2r14 history.

The exact eight-record v2r14 chain closed with terminal/completion body digests
`sha256:4663e1834e7283d3b673e57e2e176ce83ca805a3d8e0d4aab553c75892914cb1`
and
`sha256:838f42299d8febf53cad2b043bba4adde5153aed096e9859b707d85e49b86287`.
The v2r14 worktree and twenty runtime roots add 21 protected entries: six
runtime roots are present and fourteen absent.  V2r19 must not open, reuse,
resume, adopt, or mutate them.

The completed v2r15 campaign is an executed predecessor of v2r19
and is historical, terminal, and non-resumable.  Its Controller B commit/tree
were `f18a9fd415bd15ad5939b53abdc6659ae0d55583` and
`712515fb236d15381eccf14e43b52169eafcad87`; its sole parent and Target A were
`a6c3ac61f0e7d21d7d2dd64d14886a74f87dd97e`.  Its contract, capsule, owner
confirmation, authorization, and isolated Python 3.10 dependency-preflight
receipt digests were respectively
`sha256:c10945eadd2cc160f8351278945951ff3dc76755a276739273a71d13200f959b`,
`sha256:bf97840f2cd05acbee42a249e8c21049f518bfd9506b30fc2b44594a7fc4f5ad`,
`sha256:998bf75bbf372a25b54796d1ba7f75795052addc5b7bfe6dea9793a8490306a7`,
`sha256:cee1f621458900c653a31925712c97e5b44230df9a3ae85b7da08fbe64637737`,
and
`sha256:d7ee32ffbbab011f9156986938f644ea1cb9dd530cf1217ceb8e4ac84325e625`.

V2r15 consumed authorization exactly once.  During `SLOT_01` scheduling
admission, window 2 burst 15 had joined its admission workers, but the immediate
strict `/proc/self/task` replay still observed the controller task and one
known just-joined worker.  This happened before slot intent, consumption,
permanent-root creation, or supervisor `Popen`.  The frozen v2r15 cleanup-only
fallback mislabeled that observation as a terminal-drain failure and closed
with status/error `CAMPAIGN_FAILED_CLOSED_TERMINAL` /
`FAIL_CAMPAIGN_SCHEDULING_ADMISSION`.  Its exact terminal error detail was:

```text
burst 2:15 failed after zero-thread drain: RuntimeError:terminal drain policy was exceeded; cleanup=thread-proof:ShadowRetryCampaignErrorV2:FAIL_CAMPAIGN_PROCESS: spawn controller is not exactly single-threaded: task_ids=['3896108', '3903338']
```

A separate isolated diagnostic created and joined 20,000 threads and read
`/proc/self/task` immediately after each join.  It observed 58 bounded
post-join visibility hits; every task disappeared, and the maximum observed
visibility duration was `732430 ns`.  This identifies a real CPython/Linux
post-join task-visibility interval, not a 120-second terminal-drain overrun or
unknown live concurrency.  V2r16 therefore settles only already joined,
uniquely bound, `is_alive()==false` admission native task IDs for at most
`1,000,000,000 ns`, polling every `1,000,000 ns`, before repeating the original
strict single-thread controller proof.

The exact four-record v2r15 ledger contains `campaign.intent.json`,
`campaign.authorization-consumed.json`, `campaign.terminal.json`, and
`campaign.completion.json`, with record digests in that order:
`sha256:3175083e6ed2da1d2ef8b48146a09668085cab051c2f0d3142f07ce4ecf17e8e`,
`sha256:85d370a03745ff9bd1fe090c7f420f7edd5472f8315640091c7e33e67cb25d01`,
`sha256:714bec0ad2452b6ff509f5398682a44be998763f9f2161c5ef879a600dcd6a57`,
and
`sha256:0caae4a518865ea21f9de165a019022c2acec5de0074f52cdb9c2eb6ed3b12df`.
The terminal/completion body digests are
`sha256:b5aa67da443a99a1fb87c7c283054912bacbcc535c3d32471b3c33b37aae01f2`
and
`sha256:b49790f92103ea58a9196144a2dd87ed143961195df288b7e6a07c67c382db4d`.
There was no terminal error receipt, last consumed slot, slot outcome, or
supervisor invocation.  Of the twenty v2r15 runtime roots, only authorization
and ledger are present; the remaining eighteen, including every slot root, are
absent, and the transient scheduling-probe root is proved absent.  The v2r15
worktree plus those twenty roots add 21 protected historical entries.  V2r19
must not open, reuse, resume, adopt, or mutate them.

The completed v2r16 campaign is the immediate executed predecessor of v2r19
and is historical, terminal, and non-resumable.  Its Controller B commit/tree
were `332cb78da7461956e9142a3c46deeac5c5c9f947` and
`bdd0eb426bd7f980271e18858a163d48d3d3b308`; its sole parent and target commit
were `a6c3ac61f0e7d21d7d2dd64d14886a74f87dd97e`, with target tree
`67356f0f544fec59714fd046ac3a13f0e9ce06c9`.  Its campaign source-binding,
contract, capsule, owner-confirmation, isolated Python 3.10 dependency-preflight
receipt, and authorization digests were, respectively,
`sha256:ba88c0585144e9b32f8c3a1f22c9ecd4bd77c2971ae254e7383a036a8aa7eb61`,
`sha256:9de40655da439a1427230cd03f143ff16ac167f1bdcf313f50c9b39e4a4c3819`,
`sha256:14d19175c4e6b475a9f34e023b84a88708ad12bd12cc62deda7c3fe15d31df94`,
`sha256:50b3b690c16080811bcf1264cc4246a9eafb9474e57c67fdb3be43fb6579bebb`,
`sha256:457d2054b045781ae456a30e0d53fad1346751060b2596577675d1c16f32e3eb`,
and
`sha256:7ef011d49c9e479015903ed93fc46e0cce43cddcc2f8631a68704e1ef6a6633f`.

V2r16 consumed that authorization exactly once and invoked all three strictly
ordered one-shot slots.  Every slot used the historical
`Q05B_SUPERVISOR_V4` producer and `125,000,000 ns` target wait, closed and
replayed a `COMPLETE_FAIL` trajectory with no open span, torn tail, or
unpartitioned post-origin event, and proved cleanup and artifact absence.
`SLOT_01` retained 9,298 events, 4,649 closed spans, five failure rows, and six
cleanup rows.  Its exact Rust-endpoint origin was role 2, sample ordinal 2,317,
with gap `467,770,340 ns` against the `250,000,000 ns` maximum; its decision was
`RETRY_EXACT_GAP`, trajectory-binding digest was
`sha256:a0e7cc96768f77a4c043d6b188eba4b7de8e8db3a54383553f9ed62a37d30981`,
and outcome digest was
`sha256:cb67110f72b5cd791ac03a57fbf5307a91d9c9ba920aeb8b7c2224fa25a74209`.
`SLOT_02` retained 3,642 events, 1,821 closed spans, five failure rows, and six
cleanup rows.  Its exact Rust-endpoint origin was role 2, sample ordinal 2,
with gap `284,214,627 ns`; its decision was `RETRY_EXACT_GAP`, trajectory-binding
digest was
`sha256:c1cdf26db6a99defc3625b9f1aef6aaa13cce7529979a7c61349af40a71cedde`,
and outcome digest was
`sha256:c1e2392551b9ca0daf3ce043e0b1d666d059834cd6dcf831d538d3787ebd3615`.
`SLOT_03` retained 72 events, 36 closed spans, five failure rows, and eight
cleanup rows.  Its exact Python-endpoint origin was role 1, sample ordinal 2,
with gap `265,054,717 ns`; its decision was `EXHAUSTED_RETRYABLE_GAP`,
trajectory-binding digest was
`sha256:e2d6f3a86a1e306d622fedf1b05da440a57af863224bde31981c9dc61c21dfb9`,
and outcome digest was
`sha256:b412c52f7c88122efe4c5e59080c4c5e1886c84c230032fb0636d7fdad5398d4`.

The exact sixteen-record ledger therefore closed as
`CAMPAIGN_RETRY_EXHAUSTED_TERMINAL_NO_SLOT_04`, with three supervisor
invocations, no artifact publication, and Q1 and Q2 both `NOT_RUN`.  Its
terminal body/ledger-record digests were
`sha256:db5aba97b1cbfb68b02a240f489af007e795727b375a58f6419fc7bbb53a474c`
and
`sha256:f93a1fe9e1848ff451ce6e95082dd3f26c26bcdde9739780de4353c21024270e`;
its verified completion body/ledger-record digests were
`sha256:47111123200f6f60e58bb572bf5a3f949cb7a0e67ab437ea69e575b6cb6bc2a5`
and
`sha256:15f4b17d3eb5cbd6ff4d8803613554f66c617b777ee56859aee3a1c8d9a31dce`.
Of its twenty runtime roots, fourteen are present and six are absent.  The
v2r16 worktree plus those twenty roots add 21 protected historical entries;
v2r19 must not open, access, reuse, resume, adopt, or mutate them or reuse any
v2r16 capsule, authorization, ledger prefix, slot, trajectory, or completion
authority.

The v2r17 campaign is the immediate campaign predecessor of v2r19, but it is
not a terminal or completed chain.  Its Controller B commit/tree were
`b972b6b1e49749e370a0db13c8130e132a547423` and
`b01c6f61fe777e711c95ab3f808798446ad338f9`; its sole parent and Target-B
commit were `1e609a3123b5b80d5dada42f18758d3598467d5c`.  Its campaign
source-binding and contract digests were
`sha256:607b677f68a69f0d9fefe8f8b899b35029f22d62aa2480704fc67394d374a64d`
and
`sha256:ff086494a42e1a1d95be1b0b159a2aab2dfdb52a66653ce7a4f0d25fb58e8943`.
Capsule
`sha256:67402a6905e83445681ad8435ea7a827c6196aa2ccb2117e4abf387df4d07d6d`
was authorized exactly once under authorization digest
`sha256:b72a91158fbb001c486a9d44981829044f069f13132837e5cb58ab0c744a761a`.

The authoritative v2r17 ledger is exactly the two-record clean scheduling
checkpoint: `campaign.intent.json` has record digest
`sha256:8a8ae13645123381d92a1729be8631c8edd72b00509d3abc54cc9997233dc151`,
and `campaign.authorization-consumed.json` has record digest
`sha256:a52019715c8d563e8859a39eeb89b79d401ed6a87ca2019d1f1970cf1374bf74`
and names the intent digest as its predecessor.  Authorization was consumed
once, but no slot was consumed and no supervisor was invoked.  `SLOT_01`
exhausted all three complete scheduling windows and paused under the exact
`COMPLETE_GATE_EXHAUSTION` PENDING condition.  The maximum projected gaps were
50.856934, 50.795045, and 50.834151 ms; maximum critical-burst drains were
84.553151, 145.654742, and 147.592438 ms; the sustained phases durably retained
195, 255, and 2,145 timed records after the common 5,411-record prefill, with
queue high-water marks 63, 63, and 64.

The historical `/6` evidence called the remaining 11,805, 11,745, and 9,855
records `dropped`.  That field combined the first queue-capacity rejection with
the much larger set never offered after fail-fast; it is not evidence that each
of those records was individually offered and rejected.  V2r19 does not carry
that ambiguous projection forward: its `/7` evidence separately records the
typed terminal offer disposition, rejected records, records unoffered after
fail-fast, scheduled shortfall, full-duration observation, and the first exact
queue-capacity rejection.

The v2r17 checkpoint remains structurally resumable only under its own exact
same-capsule v2r17 authority; it is not v2r19 authority and v2r19 does not open
or resume it.  Only the v2r17 authorization and ledger roots are present; all
eighteen slot roots, including every permanent slot root and scheduling probe,
are absent.  There is no v2r17 terminal record, completion record, persistent
PENDING receipt, or corresponding terminal, completion, or PENDING-receipt
digest.  None is invented here.  The v2r17 worktree plus its twenty runtime
roots add 21 protected historical entries for v2r19.

The completed v2r18 campaign is the immediate executed predecessor of v2r19.
Its controller commit/tree were
`c1bebcf49734112bc22e5ca8562a50864b6e6531` and
`89e361187f7bf37277cb442251bc0a814d62cfcb`; its sole parent and Target C
were `303c0fcd45bb46c00b79bb2112bd1d777364e5bb`.  Contract
`sha256:fe3dbc1259466aa51373fe096c5c1465c08eb977c1ec969d62ef68394d4247d0`
produced capsule
`sha256:fe92bdfac4f010ae63d2123c21c5cf8cdb456a87fff34916b86f6ef0a6657e4f`,
which was authorized exactly once under authorization digest
`sha256:f75868a56979145fcbf3c48705fda50440efb589ce4d10b0bbb54cb06e16a2e2`.
Its authoritative ledger is exactly this eight-record chain:

| Sequence | Record | Ledger-record digest |
|---:|---|---|
| 1 | `campaign.intent.json` | `sha256:aefe4a2de66a7c67b6fd19142fe4210db7557c68a7bad07649d5b07644ed4ebf` |
| 2 | `campaign.authorization-consumed.json` | `sha256:a3f2539c30d1a64d4a432f93e227516cb537998dc860710f8f1932b9fb218aff` |
| 3 | `slot-01.intent.json` | `sha256:93e24ec4e1a24471ccf8afdea817afedc27312a4b6e1ba94827303eea67685b0` |
| 4 | `slot-01.consumed.json` | `sha256:e4346586d83058c418d05b00ea1638bc96989e7426893cb8a821c8c6b75adc17` |
| 5 | `slot-01.trajectory-binding.json` | `sha256:530009aa0356ac2c960a23e2c74e9bf6d4a310c98f15b53aa88536824dee9007` |
| 6 | `slot-01.outcome.json` | `sha256:10be84bfb7f01ca1fed5a07a7f563e899d92fc26b026214bf45ef436ba2b8689` |
| 7 | `campaign.terminal.json` | `sha256:85206cf092eef8723464010a2c32dee98d15f946c1705ba67f8aca843f18b6bb` |
| 8 | `campaign.completion.json` | `sha256:ffd3be4e3cbd04ff4de05198dad1897c8f5d636b3da717bfdc4190a4934cb7a9` |

Only `SLOT_01` was consumed and exactly one supervisor was invoked.  Producer
`Q05B_SUPERVISOR_V6` closed a valid `COMPLETE_FAIL` trajectory with 98 events,
49 closed spans, no open span or torn tail, five retained failure rows, ten
retained cleanup rows, 43 other nonfailure events, and zero unpartitioned
events.  The ten cleanup rows are exactly three successful Python cleanup
pairs and two successful Rust cleanup pairs.  Both supervisor and external
artifacts were absent.

The physical origin was Python role 1, sample ordinal 5.  Its previous finish,
scheduled start, native wake, and actual start were respectively
`10,717,689,231,348`, `10,717,789,231,348`, `10,718,019,131,989`, and
`10,718,019,225,400 ns`.  Therefore target wait was `100,000,000 ns`, gap was
`329,994,052 ns` against the `250,000,000 ns` maximum, native-wake lateness was
`229,900,641 ns`, trajectory-admission duration was `93,411 ns`, and total
scheduler lateness was `229,994,052 ns`.  These values close exactly:
scheduled start equals previous finish plus target wait; native-wake lateness
equals native wake minus scheduled start; trajectory-admission duration equals
actual start minus native wake; scheduler lateness equals actual start minus
scheduled start; and gap equals actual start minus previous finish.  The exact
origin detail was
`MAXIMUM_UNSAMPLED_GAP_EXCEEDED;native_wake_ns=10718019131989;native_wake_lateness_ns=229900641;trajectory_admission_duration_ns=93411`.

The v2r18 controller nevertheless persisted `TERMINAL_NONRETRYABLE` because
two controller assumptions had become stale relative to the reachable V6
producer: the live-origin grammar accepted only the bare
`MAXIMUM_UNSAMPLED_GAP_EXCEEDED` token, and the cleanup grammar allowed at most
two complete pairs per actor.  A read-only offline replay changed only those
two assumptions—accepting the exact enriched live-origin detail with strict
native-metric arithmetic and allowing three cleanup pairs per actor while
retaining the twelve-row ceiling—and then returned `RETRY_EXACT_GAP` with all
seven ordered decision-audit clauses passing or correctly not matching.  That
counterfactual replay wrote nothing, invoked no campaign or supervisor, and
cannot alter the already durable v2r18 outcome.

The v2r18 terminal/completion body digests are
`sha256:1e4e80e4d2e2642c20f577ef19e1a9da03ce00288bfc1ffd0760cd5898aee166`
and
`sha256:049a275562b35237045b24150c79405543be7a866721d9ed1bd3569d5289e012`.
The completion is the verified authoritative final commit, authorizes no later
campaign write, and permanently forbids resume, reuse, or opening `SLOT_02`.
Six of its twenty runtime roots are present and fourteen are absent.  The
v2r18 worktree plus those twenty roots add 21 protected historical entries for
v2r19.

Before the v2r19 contract was frozen, ten development review episodes performed
limited read-only historical review.  The disclosed scope is exactly:
selected v2 through v2r8 ledger/outcome/trajectory-binding JSON; v2r8 evidence
filenames only; the v2r8 trace tail of exactly nine events; the v2r12 audit;
the v2r13 authorization, ledger, slot outcome, complete trajectory, and
completion audit; the corresponding v2r14 authorization, ledger, slot outcome,
complete trajectory, and completion audit; and the v2r15 controller source,
config, authorization, ledger, scheduling terminal, and completion audit; and
the v2r16 authorization, ledger, three slot trajectories and outcomes,
terminal, and completion audit; and the v2r17 authorization, exact two-record
checkpoint, scheduling PENDING evidence, and old preauthorization-gauntlet
audit; and the v2r18 authorization, exact eight-record ledger, `SLOT_01`
trajectory and outcome, terminal/completion, and corrected-decision replay
audit.  The reviews wrote or mutated nothing and were neither execution
authority nor adoption or
reuse authority.  Accordingly, every `campaign_root_open_allowed=false` and
`campaign_root_access_allowed=false` fact is scoped to the v2r19
preauthorization gauntlet, preflight, record-authorization, and run; it does
not deny this disclosed earlier read-only review.

A separate preauthorization bound-source validation has so far confirmed only
that `/tmp/hegel-q05b-shadow-retry-v2r9-worktree` is clean at HEAD
`e9e038cf73f230ecef07a714c6a863be91189f7c` and tree
`d1d114407d14a04abc08db933473a00f122734e9`.  One full Campaign suite was also
run in that bound source worktree and passed.  In addition to exercising the
bound Campaign sources, v1 safety kernel, and Target-A paths, its test helper
directly read `rust/q1_archive_projection_oracle/Cargo.lock`; this statement
does not claim an exhaustive inventory of pytest or Python-import reads.  The
suite neither accessed nor wrote formal authorization, ledger, slot, evidence,
or runtime state.  It did use ephemeral pytest scratch and updated the
Git-ignored `.pytest_cache/v/cache/nodeids`; therefore the clean status porcelain
does not imply zero ignored-cache writes.

The one live strict-loader validation then passed.  It was limited to reading
the bound worktree Git HEAD/tree/clean metadata before the load and replaying
clean status after the strict loader; reading the exact six campaign
source-binding files; loading the bound tool without bytecode; and allowing its
strict loader to read the v1 safety kernel and the thirteen Target-A binding
paths.  The loader observed the exact bound v2r9 HEAD and tree above, equal raw
campaign source bytes, clean status before and after, and disabled bytecode
writes.  It made zero source-worktree or formal-state writes and opened no
authorization, ledger, slot, evidence, or runtime-state root.  This validation
is neither execution nor adoption/reuse authority and is distinct from the
ten-episode historical-review count above.

The old v2r17 one-shot development gauntlet at commit
`fb9219ac2d0853624cb8576fbd61ac3e75817b66`, tree
`4e6e387618c3055c766615da15aadc8c435bcba3`, completed `LIVE_ADMISSION` with
all cases passing.  Its 68,543-byte checkpoint is
`/tmp/hegel-q05b-preauth-gauntlet-fb9219ac.json`, raw SHA-256
`43be54a56e1621f5ba871dfa1b32a010b8e35c74e74cad2bc3bf043ec4b7545a`.
It created no authorization or ledger, wrote no formal root, contacted neither
Docker nor the network, invoked no supervisor, and removed its live-admission
scratch.  It is retained only as disclosed v2r17 development history.  It was
not run from the final v2r17 controller commit, is not v2r19 authority, and its
checkpoint and digest are not accepted as the v2r19 final-commit live-gauntlet
receipt.

Within that same scheduling-admission gate for `SLOT_01`, window 1 passed the
burst predicates (maximum drain 246,123,364 ns, projected gap 128,222,782 ns,
anchor 3,222,782 ns, queue high-water mark 7) but failed the sustained predicate:
5,411 prefill records plus 565 of 565 timed records were durable by the deadline,
none after it, while 4,235 further timed records were dropped as the queue reached
its high-water mark of 64.  The 5,976 durable records were retained and then
removed; terminal drain was zero, maximum sustained anchor was 12,876,911 ns
across 120 seconds, and cleanup absence was proved.  This exact capacity failure,
not a gap or anchor failure, caused the gate to move to window 2 under its bound
window policy.  Window 2 passed: maximum burst drain
214,795,028 ns, projected gap 127,169,763 ns, anchor 2,169,763 ns, burst queue
high-water mark 7; 5,411 prefill plus 4,800 of 4,800 timed records were durable
by the deadline, none after it and none dropped, for 10,211 retained and removed records, sustained
queue high-water mark 7, zero terminal drain, maximum sustained anchor
10,602,211 ns across 120 seconds, and proved cleanup absence.  This window-1 to
window-2 progression was one gate invocation, not a capsule retry, campaign run,
slot consumption, or second authorization.

V2r19 instead requires a new live gauntlet from the final clean controller
commit, after its unique parent has been proved equal to Target C and its raw
NUL-delimited diff has been proved equal to the exact six campaign paths.  It
uses the same real policy-`/8`, evidence-`/7` scheduling gate that the campaign
will use, not a copied surrogate.  Its stable external root is
`/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r19-preauthorization-live-gauntlet`,
and its immutable receipt basename is
`preauthorization-live-gauntlet.receipt.json`.  The gauntlet requires all
authorization, ledger, and formal slot roots absent, writes no formal root,
contacts neither Docker nor the network, invokes no supervisor, and removes
and parent-fsyncs its private scratch before publishing the complete receipt.
The complete dynamically generated receipt, its digest, raw SHA-256, byte
count, and external-root identity enter the capsule; the later owner
confirmation separately binds its receipt digest.  No receipt or receipt
digest exists at this source-implementation checkpoint, so none is written
into this document or contract as if already observed.

An earlier development live-checkpoint invocation failed in the dev import
path before the bound strict loader and before scratch creation.  It therefore
does not count as a strict-loader invocation and produced no live-admission
scratch state; its failure led to the bound-loader correction described above.

No source-bound contract or self-consistent capsule is execution authority.
Only after final binding may the read-only preflight derive one exact capsule
and authorization template, but no clone, Cargo copy, authorization root,
ledger root, artifact, evidence directory, trajectory, Docker process, or
supervisor invocation may be created until the project owner authorizes that
exact capsule digest.

## 1. Purpose and claim boundary

The campaign is a bounded response to one narrowly classified infrastructure
failure: an otherwise clean Q0.5b shadow invocation that fails only because a
held-actor resource-sampling gap exceeds the frozen 250 ms maximum.  It offers
three fixed, strictly sequential, one-shot **slots**, not an unbounded retry
loop.  A slot can enable its successor only after every clause of the exact gap
predicate has been durably proven.  Every other campaign failure is terminal;
the exact pre-consumption and scheduling PENDING conditions in Section 4 are
clean pauses rather than slot outcomes or successor-enabling failures.

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
binds thirteen raw paths in exact order: the v2 isolation config; the complete
six-file vendored Tomli closure under `src/hegel_machine/_vendor` (including
the `_vendor/__init__.py` namespace initializer outside the `tomli` directory,
plus the Tomli license); the admission, artifact, and trajectory cores; the
dual and trajectory tests; and the supervisor tool.  It also binds the config
static-policy root and the target source-binding digest.  The controller
recomputes every value from the local Git object store without network access.
After the isolated dependency gate it derives a provisional deterministic
capsule; record-authorization and run compare the supplied capsule digest at
that point.  It then runs a bounded, strict, full `git fsck` rooted at the exact
target commit with reflogs and dangling diagnostics excluded and requires empty
stdout/stderr.  After and only after a successful fsck, the entry performs one
independent host CPU topology/affinity admission.  That admission must pass
before authorization lookup or publication and before any ledger or slot-state
creation, write, or consumption and before any supervisor `Popen`.  Only both
admissions passing permit
construction of the authorization template or any authorization/state action.
Thus a wrong capsule digest is never PENDING, while deterministic missing or
corrupt local-object closure still fails before owner authorization rather than
consuming a campaign slot.

`Cargo.lock` is intentionally not a fourteenth target source-registry path.
Its provenance is separately bound as the exact Commit-A Git object
`Hegel Machine/rust/q1_archive_projection_oracle/Cargo.lock`, 6,493 bytes with
raw SHA-256
`c4b9d1eff522017c85b151696d082a0ff9f88bd2890c93d932f7467323a13b49`.
Before capsule construction, authorization lookup or publication, any ledger
or slot-state creation, write, or consumption, and any campaign-supervisor
spawn, a bounded child runs exactly
`/usr/bin/python3.10 -I -S -B -X utf8 -X
pycache_prefix=/nonexistent/hegel-q05b-v2r19-preflight-pyc -c PROGRAM` from
`/`.  It proves the pycache prefix absent before and after import, top-level
`tomli` unavailable and unloaded, bootstraps the frozen supervisor host
package, lazily imports the Stage-8 artifact module, proves
`artifact.tomllib.__name__ == 'hegel_machine._vendor.tomli'`, parses a smoke
TOML value and the real lockfile, and checks the 55-path host allowlist.  Its
canonical single-LF stdout and empty stderr form a deterministic receipt that
binds the six vendored files, Python flags and executable, loaded modules,
Cargo format 4, all 29 packages, the 21 registry-package projection, and full
bounded-process cleanup facts.  The v2r19 dependency program raw SHA-256 is
`a157917825ed1199c1dff4e5adc31ef85e6336b63c4830835e4d985c2d886d84`;
the vendored-closure digest is
`sha256:36f6aad68dfbc8ab661c73bc6972e0147e3aff74a1d220fb971367ae2289f4c2`;
and the registry-package-projection digest is
`sha256:ad4a01c6deeba9078364b5d1f2ad072e98e068ca0fba16ae0b0a28ea0f566963`.
These values are exact current v2r19 dependency-policy inputs, not final-binding
placeholders.
It also reads the target artifact module's
`ACTUAL_ARTIFACT_MAX_CANONICAL_BYTES` and target tool's
`ACTUAL_ARTIFACT_MAX_CANONICAL_BYTES_V2` and requires both exact integers to
equal the controller contract's `2,147,483,648`-byte artifact bound before any
authorization, ledger, slot state, or supervisor spawn.  The full receipt and
its replayed digest enter
the capsule; its producer-free validator reconstructs stdout from the retained
full child receipt and cross-binds the target and controller Git identities,
Python executable, runtime environment, frozen source hashes, and Commit-A
Cargo object against the surrounding capsule inputs.  The receipt digest also
enters owner confirmation.  Preflight, record-authorization, and run each
replay this gate freshly; it is never cached.

The v2r19 controller commit implements and tests its campaign core and tool
at the unchanged schema-v2 paths, and Commit B may differ from Commit A only
at these six paths:

- `config/phase3_q05b_shadow_retry_campaign_v2.json`;
- `docs/Hegel_Machine_Q05b_Shadow_Retry_Campaign_v2.md`;
- `src/hegel_machine/phase3_q05b_shadow_retry_campaign_v2.py`;
- `tests/test_phase3_q05b_shadow_retry_campaign_v2_core.py`;
- `tests/test_phase3_q05b_shadow_retry_campaign_v2_tool.py`;
- `tools/phase3_q05b_shadow_retry_campaign_v2.py`.

Their raw/static hashes and campaign source-binding digest are frozen in
the final contract.  The campaign additionally binds the raw hash of the v1 safety
kernel at
`src/hegel_machine/phase3_q05b_shadow_nonadoption_trial_v1.py`.  Final binding
also closes the document hash, exact predicate digests, decision-audit policy
digest, ledger grammar, and contract digest without changing the slot registry
or authority boundary.

The config deliberately contains neither its own raw hash nor a source-binding
literal embedded in executable code.  It directly binds the live tool's full
raw SHA-256.  The acyclic freeze order is: finalize all campaign source bytes;
write their raw hashes and static-surface hash into the config; compute the
campaign source-binding digest; then compute the full contract digest.  The
later capsule independently binds the final config raw hash, contract digest,
campaign source-binding digest, live tool raw hash, the full decision-audit
policy plus its digest and core schema, and the full exact prior-v2r18
historical-evidence projection.  No normalized or masked executable region
participates in this chain.

Capsule, owner-confirmation, and authorization digests are deliberately not
fields of the contract: they can exist only after capsule derivation.  The
authorization template binds them externally, avoiding a contract/capsule
self-cycle.

All digest domains are stored as exact ASCII strings without an embedded NUL.
The implementation appends exactly one `0x00` byte before hashing the compact
canonical JSON preimage without its transport LF.  The contract freezes
seventeen named domains: contract, campaign source binding, capsule,
authorization, Cargo manifest, ledger record, trajectory binding, global
trajectory-event payloads, trajectory suffix order, other-nonfailure order,
failure rows, cleanup rows, the partition certificate, decision-audit policy,
slot outcome, campaign terminal, and campaign completion.

Only a later exact project-owner confirmation may authorize execution.  That
confirmation must bind the final contract, target commit and source binding,
campaign source binding, capsule, Python 3.10 dependency-preflight receipt
digest, final-clean-controller preauthorization live-gauntlet receipt digest,
the decision-audit policy digest and core schema, the full exact prior-v2r18
historical-evidence projection, both external roots, all three slot path records, maximum invocation
count 3, the exact retry and success predicates, the terminal-no-resume rule,
the exact pre-consumption PENDING phase/condition pairs consisting of the target
fsck timeout plus the three host CPU topology/affinity conditions,
the `scheduling_pending_eligible_conditions_in_exact_order` value
`[COMPLETE_GATE_EXHAUSTION, KNOWN_DEAD_ADMISSION_TASK_QUIESCENCE_TIMEOUT]`,
and the forbidden-scope digest.  The owner record is a local same-UID direction
record, not a digital signature or external identity attestation.

The fixed future roots are:

| Purpose | Absolute path |
|---|---|
| Owner authorization | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r19-authorization` |
| Durable campaign ledger | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r19-ledger` |

The original repository, controller worktree
`/tmp/hegel-q05b-shadow-retry-v2r19-worktree`, trajectory-dev worktree, all
seventeen prior-v2 authorization/ledger/slot roots, the prior-v2r1 worktree
plus its seventeen authorization/ledger/slot roots, and the prior-v2r2
worktree plus its seventeen authorization/ledger/slot roots, and the prior-v2r3
worktree plus its seventeen authorization/ledger/slot roots, and the prior-v2r4
worktree plus its seventeen authorization/ledger/slot roots, and the prior-v2r5
worktree plus its seventeen authorization/ledger/slot roots, and the v2r6
worktree plus its seventeen absent runtime roots, and the v2r7 worktree plus
its seventeen authorization/ledger/slot roots, and the v2r8 worktree plus its
seventeen runtime roots, and the v2r9 worktree plus its seventeen runtime roots,
and the v2r10 worktree plus its seventeen runtime roots, and the v2r11
worktree plus its seventeen runtime roots, the v2r12 worktree plus its twenty
runtime roots, the v2r13 worktree plus its twenty runtime roots, and the v2r14
worktree plus its twenty runtime roots, and the v2r15 worktree plus its twenty
runtime roots, the v2r16 worktree plus its twenty runtime roots, the v2r17
worktree plus its twenty runtime roots, and the v2r18 worktree plus its twenty
runtime roots
are frozen protected roots.  Together with the fixed Cargo, Meta, production,
and attempt roots, the v2r19 registry contains exactly 373 protected roots.
Historical-root open/access by the v2r19 preauthorization gauntlet, preflight,
record-authorization, or run and capsule/authorization/ledger reuse are all
explicitly false.

The ledger and every slot freshness target must be absent at each preflight.
Authorization-root absence is intentionally not a preflight freshness claim:
it is enforced atomically and exclusively when the owner record is created;
the record is then expected to be present for an authorized run.  Future
directories must be ordinary, owner-controlled mode-`0700` directories;
immutable records must be ordinary, single-link, mode-`0444` files published
without replacement and fsynced with their held parent directory.

## 3. Fixed isolated slots

Every slot has its own clone, copied Cargo cache, external artifact location,
transient same-filesystem scheduling-probe root, evidence root, trajectory
root, stdout, and stderr.  The probe is a fixed path but is never permanent:
each gate invocation creates it, uses it for the admission workload, removes
it, fsyncs its held parent, and proves its absence before any current-slot
permanent root or intent is created.  No path may be caller-selected or reused
by another slot.

| Slot | Dedicated clone root | Cargo cache root | External artifact | Scheduling probe | Evidence root | Trace root |
|---|---|---|---|---|---|---|
| 01 | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r19-slot-01-clone` | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r19-slot-01-cargo-cache` | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r19-slot-01-artifact/phase3_q05b_dual_qualification_v2.json` | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r19-slot-01-scheduling-probe` | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r19-slot-01-evidence` | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r19-slot-01-trace` |
| 02 | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r19-slot-02-clone` | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r19-slot-02-cargo-cache` | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r19-slot-02-artifact/phase3_q05b_dual_qualification_v2.json` | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r19-slot-02-scheduling-probe` | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r19-slot-02-evidence` | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r19-slot-02-trace` |
| 03 | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r19-slot-03-clone` | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r19-slot-03-cargo-cache` | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r19-slot-03-artifact/phase3_q05b_dual_qualification_v2.json` | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r19-slot-03-scheduling-probe` | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r19-slot-03-evidence` | `/home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r19-slot-03-trace` |

The stdout and stderr files are fixed beneath each corresponding evidence
root as `supervisor.stdout` and `supervisor.stderr`.  The external trajectory
binding is `trajectory-binding.json`, and the slot result is
`slot-result.json`, in that same evidence root.  Trace IDs are exactly
`q05b-shadow-retry-campaign-v2r19-slot-01`, `-02`, and `-03`.

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
and observed aliases in that scope.  The v2r19 preflight,
record-authorization, and run entry points do not intentionally open protected
historical, Meta, or production roots, so the controller does not claim to
prove hardlink or bind-mount exclusion against those roots in those entry
points.  Its operational
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

Each v2r19 slot uses a file-protocol `git clone --no-local` of the bound
Commit-A object closure and a private, content/mode-manifest-equal copy
of the allowed Cargo cache.  Clone preparation removes every inherited source
ref, remote, reflog, source-only branch, and stash ref, then prunes.  It creates
exactly one permanent internal ref,
`refs/hegel/q05b-shadow-retry-v2r19-target`, bound to Commit A, and writes a new
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
`/tmp/hegel-q05b-shadow-retry-v2r19-worktree`,
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

Every preflight, record-authorization, and run entry performs the target
binding/object-byte and controller-cleanliness replay, then freshly passes the
isolated Python 3.10 vendored-Tomli/Cargo dependency gate.  Only after that
gate does it invoke the exact target closure command
`/usr/bin/git -C <source> fsck --full --strict --no-dangling --no-reflogs
<Commit-A>`.  That label-and-argv pair alone has a dedicated 300-second bound;
all other read-only Git commands retain the 120-second bound.  The target full
fsck is invoked exactly once per entry.  Its successful bounded receipt,
including elapsed time, is disclosed by preflight but elapsed time is not part
of capsule or runtime decision semantics.  An outer preauthorization review
uses 240 seconds as its headroom threshold and withholds an authorization
request when the observed elapsed time exceeds it.

Immediately after that fsck and before authorization lookup or publication and
before any ledger or slot-state creation, write, or consumption and before any
supervisor `Popen`, the same entry runs the independent host CPU
topology/affinity admission exactly once.  The
entry's effective affinity must contain every CPU from 0 through 31.  Against a
saved exact original affinity it then binds and reads back four masks in order:
CPU 24 for Python and the trusted-host sampler, CPU 25 for the Rust endpoint
sampler, CPU 26 for the trajectory publisher, and CPUs 27 through 31 for
supervisor control.  After the probe sequence, or immediately after the first
failed bind/readback, it restores and exactly reads back the original affinity
once.  A pass or PENDING result is impossible unless that
restore is exact; restore failure or uncertainty is terminal with no resume.
This is an entry-level Target-C host prerequisite.  It is independent of the
authorization-consumed 2/6/10 per-slot scheduling checkpoints and is not
repeated for every slot within one run.

Slot source replay rechecks the already bound target metadata, object bytes,
Cargo.lock, and clean controller state without repeating the original-repository
full fsck.  Clone preparation returns its already verified pre-spawn clone
receipt instead of immediately repeating it.  After supervisor return the
clone is verified once again.  Each meaningful clone boundary still performs
both graph-enabled and graph-disabled strict fsck, so commit-graph metadata and
the underlying object closure remain independently checked.

The tool has two exact PENDING classes, both returning exit 75.  First,
pre-consumption validation may pause for the exact target-fsck timeout or one of
three exact host CPU topology/affinity conditions.  The fsck condition requires
the 300-second bound to be reached with `timed_out=true`, no stream overflow,
zero stdout and stderr bytes, the direct child reaped, the original process
group absent, and all wrapper resources closed.  The CPU conditions are exactly
`SCHED_AFFINITY_API_UNAVAILABLE`,
`REQUIRED_EFFECTIVE_CPU_SET_UNAVAILABLE`, and
`EXACT_BIND_OR_REPLAY_FAILED_WITH_EXACT_RESTORE`; the third name records that
the requested probe failed but the saved original affinity was restored and
read back exactly.  Any restore failure or uncertainty is terminal and is not
PENDING-eligible.

For preflight and record-authorization, one of those exact four phase/condition
pairs produces schema
`hegel-phase3a-q05b-shadow-retry-preconsumption-pending/2`, status
`CAMPAIGN_PRECONSUMPTION_PENDING`, and truthful zero counts for all
authorization lookup/publication, ledger or slot-state creation/write/
consumption, Docker, and supervisor actions.  For `--run`, the same pre-lookup
condition instead produces schema
`hegel-phase3a-q05b-shadow-retry-run-validation-pending/2`, status
`CAMPAIGN_RUN_VALIDATION_PENDING`, and states that prior durable campaign state
is `UNINSPECTED_UNKNOWN`.  It claims only that this entry performed no
authorization lookup, ledger lookup/lock/write, slot mutation, Docker contact,
or `Popen`; it never falsely reports an existing clean-checkpoint campaign as
globally empty.  Reentry requires the same entry point and, when supplied, the
same capsule digest, followed by a fresh complete dependency, capsule, target
closure, and CPU-admission replay.  A wrong supplied capsule digest fails before
either validation condition can be classified as PENDING.

Second, scheduling admission has exactly two PENDING-eligible conditions:
`COMPLETE_GATE_EXHAUSTION` after exact complete three-window exhaustion, and
`KNOWN_DEAD_ADMISSION_TASK_QUIESCENCE_TIMEOUT` when only joined, uniquely
bound, not-alive admission task IDs remain visible for the full one-second
quiescence bound.  This exact order is the value bound by the future owner
confirmation field `scheduling_pending_eligible_conditions_in_exact_order`.
Both produce schema
`hegel-phase3a-q05b-shadow-retry-scheduling-admission-pending/2`, status
`CAMPAIGN_SCHEDULING_ADMISSION_PENDING`, and phase
`SLOT_SCHEDULING_ADMISSION`.  Either is available only after the transient
probe root has been removed, its parent fsynced, its absence proved, and all
current-slot permanent roots are proved absent.  `SLOT_01` binds the exact
two-record campaign-intent plus
authorization-consumed prefix; `SLOT_02` binds the exact six-record prefix with
the complete retry-eligible `SLOT_01` group; `SLOT_03` similarly binds the exact
ten-record prefix.  The receipt binds durable ledger,
authorization-consumption, and supervisor progress, and separately reports
this entry's retry-transition and Docker-inventory counts.  Its zero-side-
effect claim is only for the current slot: zero intent, clone, Cargo cache, artifact,
evidence, trace, consumption, Popen, terminal, or completion creation.
Neither eligible scheduling condition is an authoritative terminal receipt;
both write neither terminal nor completion.  Any alive joined task, unknown
task, missing controller task, `/proc` replay error, clock or sleep error,
true terminal-drain overrun, producer/publisher error, or cleanup error remains
terminal and cannot use this PENDING class.

No pending root, marker, pointer, or separate `--resume` mode exists.  The
same `--run` plus the same capsule may continue only from an exact clean
scheduling checkpoint: after fresh dependency, target-closure, and host CPU
topology/affinity replay it loads the authorization without consuming it again,
locks and strictly replays the exact 2/6/10 ledger prefix, and proves every
current/future permanent and probe root absent.  Reopening that checkpoint is
not another campaign
execution, authorization consumption, retry transition, scheduling gate, or
supervisor invocation; the continued slot's later actions are counted normally
if its freshly replayed gate passes.  This clean-boundary rule also covers a
process interruption that left exactly the same fully verified 2/6/10 prefix
and no residual current/future root.  It does not authorize any other crash
prefix or residual state.  Non-timeout fsck failure, nonempty stream, overflow,
probe-cleanup uncertainty, capsule mismatch, publication uncertainty, any
intent/consumed partial state, and network or unknown failure remain terminal
or fail-closed.  External Codex task orchestration may pause to repair other
path, file, format, or environment problems without changing campaign
authority.

The final runner
must replace the complete environment rather than inherit ambient credentials,
proxy configuration, agent sockets, tokens, or secrets.  The exact map freezes
file-only Git protocol, disabled prompts/hooks/fsmonitor/lazy fetch, null
global/system configuration, `/nonexistent` HOME/XDG/Docker config, fixed
locale/timezone/PATH, and a slot-substituted `core.worktree` value.

The target keeps the underlying `phase3_q05b_trajectory_v2.py` event and
terminal-manifest wire format at schema `/2`, but raises the supervisor's
`durable_trajectory_policy` to `/6`, retains the live container resource
transcript at schema `/3`, and freezes producer `Q05B_SUPERVISOR_V6`.  One
physical capacity-512 global FIFO facade is the
only route to the unchanged durable writer.  Ordinary live-sample `BEGIN`/`OK`
traffic is capped at 384 outstanding records, including the record in service,
leaving a physical 128-record reserve for recovery, failure, cleanup, control,
and terminal closure.  Normal admission polls at most every 10 ms and has an
absolute deadline at `previous_finish_ns + 250,000,000`; time spent waiting is
part of the actual unsampled gap.  If that deadline is reached before normal
admission, the same `BEGIN` is rolled back from normal accounting and submitted
once through the recovery reserve.  The sampling anchor is taken only after
that fallback enqueue, so the resulting real gap is strictly greater than the
hard maximum and propagates as exact `HELD_ACTOR_GAP_EXCEEDED`.  Equality at
exactly 250 ms remains accepted.

All sampler outer terminals, held-final sampling records, failure propagation,
cleanup, control, terminalization, and normal-admission fallback use the
recovery path.  At most seven recovery/control records may be outstanding:
two sampler fallback chains can each require recovery `BEGIN`, sample `FAIL`,
and outer `FAIL`, while the main controller can require one more synchronous
record.  A forged eighth outstanding record is an invariant violation.  The
publisher never permanently poisons later closure merely because normal
capacity was temporarily unavailable; a publisher fault still fails closed.
If the writer itself stops progressing, only the last durable prefix is
claimed.

The single publisher preserves total FIFO order.  A normal stop that arrives
before a sample `BEGIN` ends that sampler without creating an accepted span;
its outer `OK` then uses recovery capacity.  After quiescence, terminal-manifest
publication and close each require a drained writer and have no raw-writer
bypass.  Recovery-cap violation, publisher fault, barrier, or close failure is
fail-closed.  This leaves enough capacity to durably close a typed failure
trajectory whenever the writer continues to progress.

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
The canonical Q0.5b artifact has a separate `2,147,483,648`-byte bound.  The
outer controller applies that bound at exactly the in-clone replay,
export-source read, immutable external publication, anchored external reread,
and final external replay.  Generic
JSON, stdout, stderr, and command-capture limits remain 64 MiB.
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

At each exact pre-intent 2/6/10 ledger checkpoint, the controller first proves
the current slot's clone, Cargo cache, artifact parent, evidence, trace, and
fixed scheduling-probe paths absent.  It then creates only the transient probe
root as an owner-private mode-`0700` directory on the same filesystem as its
held parent and runs the exact scheduling-admission gate through that root
descriptor.  The gate therefore precedes the current slot intent, every
current-slot permanent root, slot consumption, and `Popen`.  Its policy schema
is `hegel-phase3a-q05b-shadow-retry-scheduling-admission-policy/8`; its
admission, sustained-publication, per-record, and chain-checkpoint evidence
schemas are `/7`.  The schema bump binds Target C's 100 ms sampler cadence,
the corresponding 50-record/s offered load, and truthful fail-fast capacity
accounting.  It admits at most three windows in strict order;
each window first observes at least 30 seconds of
`CLOCK_MONOTONIC` quiet time.  It runs sixteen independent
seven-record critical-enqueue probes: three controller records plus Python and
Rust `BEGIN`/`OK` pairs, with exact FIFO replay, high-water mark at most seven,
per-probe drain, and cleanup.  It does not treat that transient high-water mark
as sustained-rate evidence.
After those probes it exercises the same capacity-512 FIFO and immutable
four-fsync publication path under a separate continuous load in the same held
transient probe directory whose final files remain present throughout
measurement.
The retained probe's successful path contains exactly one file fsync and
three directory fsyncs.  Its measured duration ends only after immutable
readback, replay-descriptor close, and staging-absence proof.  Error cleanup
fsyncs the directory only after an actual staging/final namespace mutation;
there is no cleanup sync on the successful retained service path.  After the
gate returns, the controller removes the now-empty transient root, fsyncs its
parent, closes both descriptors, and proves the fixed pathname absent.  Only a
passed gate plus exact cleanup permits the current slot intent and permanent
state to be created.

The frozen runtime model has two sampler lanes, two trajectory records per
sample, and a `100,000,000 ns` sample period.  Its conservative arrival-rate
upper bound is therefore exactly 40 records per second.  Each admission window
first durably publishes and retains exactly 5,411 records, strictly exceeding
the old incomplete prefix before measurement begins, then runs a 120-second
timed epoch of 1,200 atomic five-record cycles at a 100 ms cadence without an
intermediate cleanup.  Each cycle contains Python `BEGIN`, Rust `BEGIN`, Python
`OK`, Rust `OK`, and one controller service-margin record.  The timed epoch
must durably retain exactly 6,000 records by its deadline, with no timed
durability after that deadline,
proving a service-rate floor of at least 50 records per second; equivalently it
must satisfy
`5 * arrival_rate_upper_bound <= 4 * durable_service_rate_floor`, reserving at
least 20 percent service headroom.  The final retained count is exactly 11,411,
and the whole timed interval therefore runs above the 5,410-event v2r3
incomplete-prefix cardinality rather than extrapolating from a transient
critical-burst high-water mark.

Every offered and durable record is canonical and bound by twelve exact
10-second epoch aggregates plus deterministic ordered chain digests.  Each
five-record timed cycle is offered atomically against the 64-record sustained
cap.  The first queue-capacity rejection fails the sustained phase fast and
records `timed_offer_terminal_disposition=QUEUE_CAPACITY_REJECTED`, the exact
`rejected_record_count`, `unoffered_after_failfast_record_count`,
`scheduled_shortfall_record_count`, `full_measurement_duration_observed=false`,
and one `first_queue_capacity_rejection_or_null` object binding its cycle,
due/attempt/lateness timestamps, queue depth, and five-record item count.  It
does not relabel records never offered after fail-fast as rejected or dropped.
A successful phase instead binds disposition `ENQUEUED`, all three counts zero,
the first-rejection object null, and full-duration observation true.

Replay must prove the same FIFO order, zero duplication or substitution, exact
typed offer/rejection/unoffered/shortfall arithmetic, exact epoch boundaries,
two enqueue anchors per enqueued timed cycle, and empty final backlog
after a terminal drain of at most `2,000,000,000 ns`.  That drain cannot count
toward the by-deadline 50-record/s floor, and any timed durability after the
deadline fails the window.  Cleanup removes all 11,411 final records and any
staging residue, fsyncs the directory, and proves absence.
The sustained atomic queue cap and maximum observed high-water mark are 64
records.  The critical-burst/runtime physical queue capacity remains 512, but
normal sampler traffic may occupy
at most 384 outstanding records, leaving 128 physically reserved while the
runtime recovery/control outstanding cap remains exactly seven.  Publisher
error is terminal, while a clean typed sustained-capacity rejection is a
truthful nonpassing window rather than an ambiguous aggregate.  Reservation
breach,
insufficient retained cardinality, arrival/service-rate inequality, reorder,
residue, evidence, clock, identity, drain, or close uncertainty fails closed.

For each sampler lane the critical metric is the interval from its scheduled
release to a `CLOCK_MONOTONIC` anchor taken immediately after enqueue returns.
It must be at most `25,000,000 ns`.  Adding the frozen `100,000,000 ns` runtime
sampling target yields a projected gap of at most `125,000,000 ns`, reserving
`125,000,000 ns` beneath the runtime hard maximum of `250,000,000 ns`.

A window passes only when all sustained-load, ordered-replay, service-margin,
enqueue-bound, reserve, and terminal-drain clauses pass.  The first passing
window is selected and no later window is opened.  A fully recorded nonpassing
window may open only the next admission window; uncertainty never may.  If all
three windows fail exactly and transient cleanup and absence are certain, the
controller emits the typed scheduling PENDING result described above.  It
leaves the exact 2/6/10 authoritative prefix unchanged, creates no current-slot
permanent root or intent, writes no terminal or completion, consumes no slot,
and performs no `Popen`.  That complete exhaustion receipt is never accepted
as terminal authority.  A nonexact gate failure or probe cleanup/absence
uncertainty remains terminal at the clean checkpoint and cannot be
reclassified as PENDING.

After every critical-burst or sustained-publication worker join, v2r19 binds
the workers' unique native task IDs and requires every corresponding Python
thread to report `is_alive()==false`.  It then polls `/proc/self/task` for up to
`1,000,000,000 ns` at `1,000,000 ns` intervals.  During this bounded settlement
the only permitted task set is the controller plus a subset of those known
joined IDs.  Once those IDs disappear, the original exact single-thread proof
runs unchanged and the gate continues transparently.  An unknown or alive task,
a missing controller task, or `/proc`, monotonic-clock, or sleep uncertainty is
terminal.  If only known joined IDs remain visible for the full bound and there
is no primary, producer, publisher, terminal-drain, or cleanup error, the gate
emits an intermediate exact joined-task-quiescence receipt.  Only an outer
runner holding an exact locked 2/6/10 ledger prefix may, after subsequent
empty-probe cleanup, parent fsync, pathname-absence proof, and
current-permanent-root absence proof, convert that receipt to
`KNOWN_DEAD_ADMISSION_TASK_QUIESCENCE_TIMEOUT` scheduling PENDING.  Thus the
v2r15 post-join visibility interval can pause at the same clean checkpoint,
while real concurrency or cleanup failure remains terminal.

The gate's exact claim ceiling is
`OBSERVED_PRECONSUME_BOUND_STAGE5_ENQUEUE_AND_120_SECOND_SAME_DIRECTORY_RETAINED_FILE_PUBLICATION_RATE_HEALTH_ONLY_NOT_A_RUNTIME_SCHEDULING_DURABILITY_OR_250MS_GAP_GUARANTEE`.
This is pre-consumption evidence for the exact sustained admission workload,
not a promise about later scheduler behavior, Docker execution, either runtime
sampler thread, filesystem durability timing, or a runtime gap.  Any exact
runtime gap is still classified solely from the durable trajectory grammar
below.

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
- trace ID, producer `Q05B_SUPERVISOR_V6`, and `CLOCK_MONOTONIC`;
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
`FAIL` row (at most seven for retry), every nonfailure
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
4. The complete failure projection contains either one metric-complete gap
   origin, preserving every v2r1 single-origin grammar, or exactly two
   independent metric-complete origins in one of only two frozen Stage-5
   combinations: Python `LIVE_RESOURCE_SAMPLE` plus Rust
   `LIVE_RESOURCE_SAMPLE`, or Python `HELD_FINAL_RESOURCE_SAMPLE` plus Rust
   `LIVE_RESOURCE_SAMPLE`.  No trusted-host origin, Rust held-final origin,
   duplicate actor, or third origin is admitted in a multi-origin projection.
   The first failure in global sequence remains the partition-certificate
   suffix origin and may be Rust, but Python is always the terminal cause in
   either accepted dual-origin combination because the frozen producer closes
   Python first and abort-cleans Rust after Python close propagation fails.
   Each origin independently binds its actor, role, operation, span, sample
   ordinal, prior finish, scheduled and actual start, finish, collection
   duration, scheduler lateness, target wait, gap, maximum, exceeded flag, and
   exact raw error detail; the two actors' ordinals and timings are not
   conflated.

   A live-resource origin additionally requires the exact ordered detail
   grammar
   `MAXIMUM_UNSAMPLED_GAP_EXCEEDED;native_wake_ns=<uint64>;native_wake_lateness_ns=<uint64>;trajectory_admission_duration_ns=<uint64>`.
   The three decimal values are parsed, not accepted as opaque text.  Native
   wake is no earlier than scheduled start and no later than actual start;
   trajectory-admission duration equals actual start minus native wake;
   native-wake lateness equals native wake minus scheduled start; and scheduler
   lateness equals actual start minus scheduled start.  Every value is a
   non-boolean unsigned 64-bit integer and must agree
   with the origin row's already exact scheduled/actual arithmetic.  A
   held-final origin retains the exact bare
   `MAXIMUM_UNSAMPLED_GAP_EXCEEDED` detail because its reachable producer path
   does not emit the live-origin native fields.
5. At every accepted origin, target wait is exactly `100,000,000 ns`, maximum
   gap is exactly `250,000,000 ns`, observed gap is strictly greater than the
   maximum, `gap_exceeded` is true, finish equals actual start, and collection
   duration is zero.
6. The single-origin propagation alternatives remain unchanged.  A held-final
   origin propagates through actor-close, endpoint Stage 5 or trusted-host
   Stage 6, and run with its raw detail.  A live origin normally propagates
   through sampler, actor-close, Stage 5 or Stage 6, and run; the separately
   frozen endpoint Stage-4 early path is origin -> sampler -> Stage 4 -> run.
   Its origin and sampler retain the raw detail, while later layers and the
   stdout/terminal wire carry exactly one `held actor sampler failed: ` prefix.

   The dual-origin grammar is a separate exact partial order, not a generic
   merge of single-origin failures.  Rust contributes exactly live origin ->
   sampler and no actor-close row.  Python contributes either live origin ->
   sampler -> actor-close or held-final origin -> actor-close.  Both actor-local
   chains must complete before the common Stage-5 failure, which must precede
   the terminal run failure.  Cross-actor interleaving is permitted only when
   those per-actor and common-tail orders remain true.  Stage 5, run, terminal
   manifest, and stdout all carry the Python terminal-cause detail: exactly one
   prefix for Python live, or the raw Python detail for Python held-final.  The
   live/live form has exactly seven FAIL rows; the Python-held-final/Rust-live
   form has exactly six.  The bounded projection capacity is seven, and retry
   requires every FAIL row to be retained, sequence ordered, digest bound, and
   untruncated.  Any additional or unmatched FAIL therefore rejects retry.

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
   interleave before it closes.  The same actor may have at most three cleanup
   pairs.  This admits the reachable V6 sequence in which Python closes three
   complete pairs while Rust closes two; the certificate still retains at most
   twelve cleanup rows in total.  A single origin requires at least one successful pair
   for that origin actor; either accepted dual-origin form requires at least one
   successful pair for both Python and Rust.  Endpoint Stage-4/Stage-5 paths may
   contain only endpoint cleanup actors.  A trusted-host Stage-6 single-origin
   path still requires at least one successful trusted-host pair, but may also
   contain successful cleanup pairs for Python and Rust because the frozen
   outer failure sweep idempotently cleans all three registered actors.  A
   malformed cleanup, an unregistered actor, a fourth cleanup for one actor,
   missing cleanup for any required origin actor, retained-row
   truncation, a nonzero unpartitioned count, a partition count/root/digest
   mismatch, a
   missing/collapsed/double-prefixed tier, an unregistered Stage-4 path, an
   independent failure, an unsupported second origin, or any third origin
   rejects retry.

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

The pure core returns the decision and its audit together through
`evaluate_slot_observation_with_audit_v2`; no caller reconstructs a reason
afterward.  The audit schema is
`hegel-phase3a-q05b-shadow-retry-campaign-decision-audit/2` with exactly
`schema_version`, `decision`, `matched_path_or_null`,
`first_reject_clause_or_null`, and `predicate_results`.  Predicate results are
limited to `PASS`, `NO_MATCH`, `ACCEPT`, `REJECT`, and `NOT_EVALUATED`, and the
clauses appear exactly as `OBSERVATION_SHAPE`, `SUCCESS`, `COMMON_EXACT`,
`GAP_GRAMMAR`, `WIRE_TERMINAL`, `PARTITION_CLEANUP`, and
`ARTIFACT_ABSENCE`.  The exact same core-returned audit is persisted in both
the external `slot-result.json` and the corresponding ledger slot outcome.
Producer-free replay recomputes it and requires an exact match; missing,
changed, reordered, or independently reconstructed audit evidence terminates
as `FAIL_CAMPAIGN_OBSERVATION_TERMINAL_NO_RESUME`.  The enclosing policy schema
is `hegel-phase3a-q05b-shadow-retry-slot-decision-audit/1`, and its final digest
is a capsule and owner-authorization binding input.

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

Each active runner process must hold one exclusive ledger lock and directory
descriptor continuously while it validates and advances the chain.  An exact
scheduling PENDING exit releases that process only after the clean 2/6/10
prefix, current/future-root absence, and probe cleanup have been verified; a
later same-capsule `--run` must reacquire the lock and replay that whole
checkpoint before advancing it.  Every record binds the contract, target
binding, campaign source binding, capsule, owner authorization, prior record
digest, slot identity, and external trajectory binding where applicable.

The permitted chain is:

```text
ABSENT_NOT_AUTHORIZED
  -> AUTHORIZED_CAMPAIGN_INTENT_STAGED
  -> CAMPAIGN_AUTHORIZATION_CONSUMED
  -> SLOT_01_CLEAN_SCHEDULING_CHECKPOINT
       -> exact gate exhaustion or clean known-dead-task timeout: PENDING, prefix unchanged
       -> gate pass: SLOT_01_INTENT_STAGED
  -> SLOT_01_CONSUMED_OUTCOME_UNKNOWN
  -> success terminal | failure terminal | exact-gap eligible
       exact-gap eligible -> SLOT_02_CLEAN_SCHEDULING_CHECKPOINT
       -> exact gate exhaustion or clean known-dead-task timeout: PENDING, prefix unchanged
       -> gate pass: SLOT_02_INTENT_STAGED
       -> SLOT_02_CONSUMED_OUTCOME_UNKNOWN
       -> success terminal | failure terminal | exact-gap eligible
            exact-gap eligible -> SLOT_03_CLEAN_SCHEDULING_CHECKPOINT
            -> exact gate exhaustion or clean known-dead-task timeout: PENDING, prefix unchanged
            -> gate pass: SLOT_03_INTENT_STAGED
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
the exact persisted/recomputed decision audit, next-slot eligibility, and
terminal progress/status/error semantics for all ten
legal caught-error prefixes and every complete normal branch.  It does not, by
itself, recreate trajectory partition roots from raw event files.

Only an exact locked 2/6/10 clean scheduling checkpoint with all current/future
permanent and probe roots absent may be continued by another same-capsule run;
no prior CLI receipt or durable pending marker is required.  Any crash after
intent, any consumed-but-unknown state, any other prefix, any residual root,
and any uncertainty in intent, consumption, outcome, trajectory-binding,
external-artifact, failure, terminal-record staging, or completion-record
publication is terminal with **no resume**.  Catchable
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
after ledger or publication uncertainty.  Apart from the exact clean
scheduling checkpoint above, a later process may inspect the last durable
ledger prefix but cannot continue it.  Failure to publish a failure or
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

At the current v2r19 source-implementation checkpoint:

- the completed v2, v2r1, v2r2, v2r3, v2r4, v2r5, v2r7, and v2r8 chains,
  plus the failed pre-capsule v2r6 source-binding preflight and the retired,
  unexecuted v2r9 source/capsule, and the retired v2r10 authorization plus
  pre-consumption fsck-timeout run, plus the completed five-record v2r11
  scheduling-failure chain, the completed twelve-record v2r12 two-slot chain,
  the completed eight-record v2r13 one-slot Stage-09 artifact-size terminal
  chain, and the completed eight-record v2r14 one-slot Stage-06 trusted-host
  gap chain, plus the completed four-record v2r15 pre-consumption scheduling
  terminal with zero supervisor invocations, and the completed sixteen-record
  v2r16 three-slot exact-gap retry-exhaustion chain, remain historical,
  non-resumable evidence; the v2r17 two-record `COMPLETE_GATE_EXHAUSTION`
  checkpoint remains current resumable authority only for v2r17 and is
  historical nonauthority for v2r19; the completed eight-record v2r18
  one-slot chain is authoritative terminal/completion history, and its
  corrected offline `RETRY_EXACT_GAP` audit is explanatory nonauthority only;
  v2r19 has distinct roots, trace IDs, clone ref, instance hash domains, and a
  373-entry protected historical registry; the trajectory wire is intentionally
  retained while the current producer remains V6;
- every v2r19 entry runs the isolated dependency gate before one dedicated
  300-second target full-fsck admission and then one independent host CPU
  topology/affinity admission before authorization lookup/publication or any
  ledger/slot-state creation, write, or consumption or supervisor `Popen`; the
  CPU admission requires
  effective CPUs 0 through 31, probes the exact 24 / 25 / 26 / 27--31 masks,
  and restores the original affinity exactly; the exact fsck timeout or one of
  the three CPU conditions returns the entrypoint-accurate `/2`
  preconsumption/run-validation PENDING result with exit 75, while restore
  failure or uncertainty is terminal and successful fsck elapsed time is
  disclosed only for the external 240-second authorization-headroom review;
  slot replay never repeats the target full fsck, and each prepared clone is
  strictly verified once before and once after its supervisor boundary;
- each per-slot scheduling gate runs in a transient same-filesystem probe root
  before current-slot intent or permanent state; exact three-window exhaustion
  or an exact one-second known-dead joined-task quiescence timeout cleans and
  proves that root absent, leaves the authoritative 2/6/10 prefix unchanged,
  writes no terminal/completion, and returns scheduling PENDING;
  this per-slot gate is separate from the entry-level CPU admission, and
  same-capsule continuation requires a fresh dependency/fsck/CPU replay, exact
  locked prefix verification, and current/future permanent/probe absence;
- the underlying trajectory event and terminal-manifest wire remains `/2`;
  the supervisor durability policy is `/6`, the live container resource
  transcript is `/3`, and producer `Q05B_SUPERVISOR_V6` separates
  sampling-sensitive enqueue anchors from its single FIFO publisher, reserves
  128 of 512 physical queue entries for recovery/failure/control/cleanup
  closure, and enforces a seven-record recovery/control outstanding cap;
- the campaign admission policy schema is `/8`, while its evidence and record
  schemas are `/7`; the policy retains the one-second/one-millisecond
  known-dead joined-task quiescence rule, and the gate replaces the v2r3
  transient five-record burst surrogate with a 120-second
  two-sampler workload, exact FIFO replay, 25 ms enqueue bound, 40 records/s
  actual sampler-production ceiling, 50 records/s offered load and
  durable-service floor,
  20 percent service margin, 125 ms projected-gap ceiling, 125 ms hard-gap
  reserve, 1,200 atomic five-record cycles, 6,000 timed records, and exactly
  11,411 retained records before consumption.  Its sustained atomic queue cap
  is 64 records, and a first capacity rejection truthfully separates rejected
  records, records unoffered after fail-fast, scheduled shortfall, and whether
  the full measurement duration was observed;
- live-resource exact-gap origins parse and arithmetically bind the V6
  `native_wake_ns`, `native_wake_lateness_ns`, and
  `trajectory_admission_duration_ns` detail fields; held-final origins retain
  the bare detail grammar; cleanup permits at most three complete pairs per
  actor under the unchanged twelve-row retention ceiling; and each slot
  persists the same seven-clause core decision audit in both the external slot
  result and ledger outcome for exact replay;
- the core's v2r19 failure-row, cleanup-row, and partition-certificate digest
  domains are instance-specific; the version-neutral slot-observation domain
  and retained `/2` record schemas are intentionally unchanged;
- final target commit: `303c0fcd45bb46c00b79bb2112bd1d777364e5bb`;
- final target tree: `d8a17e0ca7166eab5db46a87150c37b514f79648`;
- final target static-policy root:
  `67b0cb763fa2c81b256c1831fd5d75f4cc70ab315c90a1cbf1ac2111d29cba96`;
- final target contract digest:
  `sha256:159b4e80b60d59005bd96b2b38136cb6504f0700319be0cd99886c3d487569f6`;
- final target source-binding digest:
  `sha256:97963373458ffb333e959cfff6624d322c6da3f63cda66c0e444b7cb436d6713`;
- the campaign raw/static hashes, campaign source-binding digest, policy
  digests, and contract digest are currently
  explicit `null` final-binding placeholders and grant no execution authority.
  Their eventual final values are not guessed in this document because this
  document is itself one of the hashed campaign source inputs.  No v2r3,
  v2r4, v2r5, v2r6, v2r7, v2r8, or retired v2r9 binding
  value, no v2r10 binding/capsule/authorization value, no v2r11 capsule,
  authorization, ledger, terminal, or completion value, no v2r12 capsule,
  authorization, ledger, slot outcome, terminal, or completion value, and no
  v2r13 or v2r14 capsule, authorization, ledger, slot outcome, terminal, or
  completion value, and no v2r15 capsule, authorization, ledger, terminal, or
  completion value, and no v2r16 capsule, authorization, ledger, slot outcome,
  trajectory, terminal, or completion value, and no v2r17 capsule,
  authorization, ledger checkpoint, persistent PENDING receipt, or historical
  development-gauntlet value, and no v2r18 capsule, authorization, ledger,
  trajectory, outcome, terminal, completion, or offline corrected-decision
  value is a v2r19 current value;
- no v2r19 final-clean-commit live-gauntlet receipt, capsule, or Python
  dependency-preflight receipt has been derived or
  owner-authorized, and neither the implementation tree nor later source
  binding alone is execution authority;
- no v2r19 preflight, scheduling-admission window, supervisor invocation,
  campaign Docker/Cargo/clone/artifact/evidence/trajectory action, Meta/Q1/
  production action, adoption, promotion, or current-pointer write has run.

The exact offline campaign-tool result is recorded outside these self-bound
document bytes only after strict final-contract loading and digest replay.  A
single later read-only preflight may derive one exact capsule only from the
frozen v2r19 Commit B; this document does not itself represent preflight,
authorization, or execution.
