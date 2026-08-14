# Hegel Machine Phase-2B Actual-Unsealed 960 Readiness DAG Erratum v1

**Document ID:** `hegel-phase2b-actual-unsealed-960-readiness-dag-v1`

**Status:** `READINESS_ORDERING_ERRATUM_FROZEN_NOT_EXECUTED`

**Claim level:** `READINESS_ORDERING_ERRATUM_ONLY_NOT_GATE_EVIDENCE_NOT_EXECUTION`

## 1. Decision

The machine-readable companion is
`config/phase2b_actual_unsealed_960_readiness_dag_v1.json`. Its exact content
ID is:

```text
phase2b_actual_unsealed_960_readiness_dag_v1_775b86cb499db3a991686634335ec444f7e44c7caea92b1c2b948036b585ea84
```

It binds:

```text
owner authorization:
phase2b_c1_owner_actual_unsealed_960_authorization_v1_dd2cd6a98b429fe01f172daf16ea70404c91356e40e9e42daed4cb5416a5c50d

protocol:
phase2b_protocol_62ad411b5b5a0f912626c54e4bd822a8c585a8f612f5fce6040cce500a11756a

exact freeze:
phase2b_exact_freeze_ffa1fd4fed0b5c2c018803aa9f730b8c85c144efe7e4aa324256681d1c742cbe
```

This companion corrects one ordering defect in the owner authorization V1
prerequisite list. That list placed generator qualification, baseline
registration, and statistical validation before trusted-wire/covert-channel
qualification. The exact-freeze construction order instead places Priority 3
covert-channel wire work before Priority 5 Phase-2B statistical completion.

The stricter order in this companion must govern any future launch-readiness
evaluation. The historical owner authorization remains immutable. This
companion does not supersede it, change its scope, relax a condition, or create
evidence for any prerequisite.

The old list is therefore a **non-executable ordering conflict**, not an
alternative launch sequence. No consumer may silently treat the old list as
executable or claim that this companion silently supersedes the authorization.

## 2. Canonical readiness DAG

The 16 nodes are all initially unsatisfied:

| Node | Requirement | Dependencies |
|---|---|---|
| R01 | exact owner authorization active and not revoked | none |
| R02 | execution branch replays protocol, freeze, and authorization bindings | R01 |
| R03 | formal RFC 8785 trusted-wire implementation qualification | R02 |
| R04 | namespace-aware covert-auditor implementation qualification | R03 |
| R05 | integrated wire/covert qualification on synthetic/dev inputs only | R04 |
| R06 | independent generator implementation qualification without actual cohort generation | R05 |
| R07 | all three exact baseline revisions/artifacts registered before generation | R05 |
| R08 | a permitted statistical validation version passes | R06 and R07 |
| R09 | live six-role Docker qualification passes under the disclosed threat model | R08 |
| R10 | durable custody, ledger, CAS, denominator, and crash/retry behavior pass | R09 |
| R11 | execution-configuration freeze and pregeneration commitments are cross-bound | R10 |
| R12 | the independent launch overlay is authorized and verified without allocating IDs | R11 |
| R13 | the verified overlay allocates fresh campaign and attempt IDs | R12 |
| R14 | the actual-unsealed 960 cohort is generated and public-payload commitments are bound | R13 |
| R15 | the actual corpus passes namespace, metadata, capacity, and atomic covert audit | R14 |
| R16 | recognizer and three-baseline inference admission is unlocked | R15 |

The generator and baseline branches may be constructed independently after R05;
R08 joins them. No downstream node may be admitted from list position alone.
Every dependency needs separately verified evidence.

Authorization and verification of the launch overlay at R12 precedes fresh ID
allocation at R13. The old authorization list's opposite textual order is not
an authorization to allocate IDs before the overlay is admitted.

R05 is only implementation qualification on synthetic/development inputs. It
cannot substitute for R15. Actual-corpus covert audit occurs only after R14
generation and must pass before any recognizer or baseline inference is
admitted at R16.

## 3. R03-R05 qualification boundary

R03-R05 require an implementation and qualification unit that freezes and
tests, on synthetic/development data only:

1. RFC 8785 canonical public-case JSON;
2. a fixed 65,536-byte envelope and explicit capacity-boundary tests;
3. independent `K_shuffle`, `K_id`, and `K_padding` roles, separated from the
   latent-generator and answer seeds;
4. post-global-shuffle HMAC-SHA256 public IDs truncated to 128 bits with UUIDv4
   version and variant bits;
5. namespace-aware, globally consistent renaming over all ten frozen ID
   namespaces, with at least 32 renamings;
6. metadata-predictivity gates, including 10,000 label permutations,
   Holm-Bonferroni family-wise alpha 0.01, normalized mutual information no more
   than 0.02, and balanced-accuracy advantage no more than 0.05;
7. 32 global case-order permutations and 16 within-case observation-order
   permutations;
8. `standard_error` disabled by default; and
9. exact evidence bindings for source commit, source archive, dependency lock,
   runtime image, synthetic fixture root, and qualification receipt; and
10. atomic whole-qualification failure on schema/closure, envelope-capacity,
    renaming-invariance, covert-gate, source/answer binding, key-independence,
    RFC 8785 duplicate/Unicode/number vectors, unclassified/ambiguous namespace,
    or statistical-sufficiency failure.

`standard_error` has exact status `STANDARD_ERROR_UNSUPPORTED`; encountering it
aborts the whole batch with no partial output. Existing accepted-JCS-subset
mechanics are not a complete RFC 8785 qualification, and synthetic covert
statistics capped at 64 cases are not actual-960 capacity or corpus
qualification.

This contract states what later evidence must prove. This readiness artifact
does not implement the wire builder or auditor and does not assert that R03,
R04, or R05 has passed. A separate implementation qualification may later bind
its own source, test vectors, qualification receipt, and content IDs to these
requirements.

## 4. Non-execution boundary

The loader accepts only strict canonical ASCII JSON with no trailing newline,
duplicate keys, floats, non-finite numbers, unknown fields, alternate order, or
content-ID drift. A successful verification yields a privately issued,
deeply immutable result. It proves only that this exact ordering companion was
verified.

At publication, the exact state is:

```yaml
node_count: 16
satisfied_node_count: 0
all_nodes_satisfied: false
formal_trusted_wire_implementation_qualified: false
formal_covert_channel_audit_passed: false
synthetic_dev_qualification_executed: false
formal_corpus_audit_executed: false
actual_960_capacity_qualified: false
generator_exists: false
baseline_registrations_exist: false
prerequisites_verified: false
campaign_id: null
attempt_id: null
campaign_id_allocation_authorized: false
actual_campaign_eligible: false
execution_authorized: false
formal_holdout_generation_authorized: false
actual_holdout_generated: false
actual_corpus_audit_executed: false
inference_admission_unlocked: false
inference_performed: false
docker_invoked: false
verifier_filesystem_accessed: false
network_accessed: false
keys_generated: false
scoring_performed: false
actual_execution_claimed: false
shadow_only: true
```

The pure verifier has no filesystem, Docker, network, subprocess,
key-generation, ID-allocation, holdout-generation, covert-audit execution, or
scoring capability. Reading the config bytes is the caller's responsibility;
the verifier only consumes bytes already supplied to it.

## 5. Exact allowed statement

> The Phase-2B actual-unsealed-960 readiness order has been frozen as a strict
> additive companion to the immutable owner authorization V1. Formal trusted
> RFC 8785 wire and namespace-aware covert-auditor implementation qualification,
> using synthetic/development inputs only, must precede generator, baseline,
> and statistical validation work. No readiness node, implementation
> qualification, actual-corpus audit, campaign identity, execution, score, or
> Phase-2B/C1 result is established by this artifact.
