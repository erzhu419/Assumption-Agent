# Hegel Machine Phase-2B C1 Owner Authorization From Zero: Actual-Unsealed 960 v1

**Document ID:** `hegel-phase2b-c1-owner-authorization-from-zero-actual-unsealed-960-v1`
**Authority class:** `OWNER_ACCEPTED_CONTAINER_TECHNICAL_ACTORS_V1`
**Status:** `OWNER_AUTHORIZED_PROSPECTIVE_CONDITIONAL_CONSTRUCTION_AND_EXECUTION`
**Claim level:** `PROSPECTIVE_CONDITIONAL_AUTHORIZATION_NOT_IMPLEMENTATION_NOT_EXECUTION_NOT_EVIDENCE`

## 1. Owner decision

The owner prospectively authorizes construction from zero of:

1. an independent Phase-2B holdout generator;
2. registration campaigns for exactly the frozen embedding, semantic-only LLM,
   and flat typed baselines;
3. the readiness, custody, ledger, CAS, Docker qualification, chronology,
   scoring, and campaign machinery required to evaluate a future attempt; and
4. at most one fresh `actual-unsealed-960` campaign, but only after every
   frozen prerequisite has passed and a separate independent launch overlay has
   allocated and bound fresh campaign and attempt IDs.

The exact machine-readable policy is
`config/phase2b_c1_owner_actual_unsealed_960_authorization_v1.json`. Its body is
strict canonical ASCII JSON and has content ID:

```text
phase2b_c1_owner_actual_unsealed_960_authorization_v1_dd2cd6a98b429fe01f172daf16ea70404c91356e40e9e42daed4cb5416a5c50d
```

This is a prospective scope decision. It is not evidence that any authorized
component currently exists.

## 2. Non-self-executing preflight boundary

The authorization itself fixes the following exact state:

```yaml
construction: true
actual_unsealed_960_campaign_conditionally_authorized: true
authorization_is_self_executing: false
campaign_id: null
attempt_id: null
campaign_id_allocation_authorized: false
actual_campaign_eligible: false
execution_authorized: false
formal_holdout_generation_authorized: false
shadow_only: true
prerequisites_verified: false
independent_launch_overlay_verified: false
```

The policy therefore cannot be passed directly to a campaign runner. No field
in this artifact may be interpreted as a launch token, registry, Docker
receipt, custodian attestation, baseline registration, holdout commitment,
prediction archive, score, gate result, or C1 exit record.

## 3. Frozen identities and legacy preservation

The policy binds the existing Phase-2B protocol and exact freeze
`hegel-freeze-p2b-p3-v1.0.1`, the
owner-accepted same-admin actor class and profile V2, the logical-stage V1 wire
schema/manifest, and the V2 authorization-policy identity. It does not edit or supersede those
artifacts. In particular, the legacy logical-stage fields remain:

```yaml
legacy_registry_policy_authorization_literal: NOT_AUTHORIZED_BY_FROZEN_ACTOR_PROFILE_V2_OR_EXECUTION_REGISTRY_V1
legacy_actual_campaign_eligible: false
legacy_actual_execution_claimed: false
legacy_execution_authorized: false
legacy_logical_stage_fields_unchanged: true
```

Logical-stage V2 mechanics may verify a caller-supplied signed prefix, but this
new policy does not retrospectively promote any old token. A later launch
overlay must independently bind this authorization, fresh execution evidence,
and a fresh campaign/attempt identity.

## 4. Exact construction scope

### 4.1 Independent generator

The generator must be built independently under the `CUSTODIAN` data boundary.
It must use a fresh seed and must not reuse Phase-2A fixtures, historical
holdouts, synthetic answer keys, prediction-derived feedback, or scorer-derived
expectations. Generator source, seed, answers, and gate manifests must never be
mounted into the recognizer or baseline containers.

The intended future cohort remains 720 main cases plus the separate 240-case
semantic-conflict challenge, with the independently derived 496 legal and 76
invalid transformation pairs. Building code or fixtures does not generate this
cohort. Any cohort produced under this authorization is `actual-unsealed` and
cannot be relabelled as the formally sealed Phase-2B holdout.

### 4.2 Three baseline registration campaigns

Before generation, all three implementations must be independently frozen and
registered against their existing BaselineSpec content IDs:

| Baseline | BaselineSpec content ID | Revision policy |
|---|---|---|
| `embedding_nearest_prototype` | `phase2b_baseline_spec_d14fa83a6843b92d1a502bba97b982323d1de8bf0abe22763d8c1cb67c97cfa4` | exact 40-hex commit |
| `frozen_llm_semantic_only` | `phase2b_baseline_spec_c07005a2e32423f5e11a2160367b089d8f69208289239bfc18821fb351f13fa8` | exact 40-hex commit |
| `flat_learned_typed` | `phase2b_baseline_spec_0e7b098030bf7e5a36f2deb0befd40fbabc55ce538a78765be2aa33d6b7af6d2` | repository dependency lock |

Registration must bind implementation, source, configuration, image/SBOM when
applicable, artifact digest, and a receipt proving that registration preceded
generation. Registration is not a prediction archive and does not show that a
baseline ran on the future 960 inputs.

### 4.3 Readiness and campaign machinery

Construction may include pure contracts, validators, Dockerfiles, actor
workers, runbooks, negative vectors, durable-state code, and independent replay
tools. The six formal technical roles remain governance anchor, custodian,
store receipt, time attestor, recognizer, and scorer/evaluator. Baseline runners
must use a separately versioned execution profile rather than silently changing
the frozen six-role signature DTO.

The machine body freezes this exact threat-model declaration:

```yaml
same_admin_controller: true
owner_accepted_threat_model: true
organizational_independence: false
independent_human_actors: false
technical_role_independence_profile_required: true
technical_role_independence_live_verified: false
remote_attestation: false
hardware_key_nonexportability: false
malicious_admin_resistance: false
```

`technical_role_independence_profile_required=true` is a prospective profile
requirement. It is not live evidence; only a future independent qualification
may establish that evidence outside this authorization artifact.

## 5. Preconditions for the separate launch overlay

The independent launch overlay must fail closed unless it verifies, in order:

1. this exact content ID is active and not revoked;
2. all frozen protocol/profile/logical-stage bindings replay on the selected
   execution branch;
3. the independent generator is implemented and qualified;
4. all three baseline registrations were frozen before generation;
5. a permitted validation version passed;
6. the trusted wire builder and covert-channel audit passed;
7. live six-role Docker qualification passed under the disclosed same-admin
   threat model;
8. durable custody, CAS, ledger, attempt denominator, and crash/retry behavior
   are verified;
9. execution-freeze and data commitments are cross-bound; and
10. only then, the separate overlay allocates fresh campaign and attempt IDs and
    independently authorizes launch.

This file does not implement that overlay and cannot satisfy any prerequisite
by declaring it true.

## 6. Claim boundary

The loader/verifier may claim only that the exact prospective authorization
bytes and content ID were verified. It does not authenticate an owner
signature or observe owner identity. It has no filesystem, Docker, network,
key-generation, campaign allocation, scoring, or execution path.

The machine key `authorization_artifact_permanent_nonclaims` fixes all of the
following as claims that this authorization artifact itself can never make:

```text
implementation_exists
generator_exists
baseline_registrations_exist
owner_signature_authenticated
docker_invoked
filesystem_accessed
network_accessed
keys_generated
holdout_generated
holdout_formally_sealed
actual_execution_claimed
scoring_performed
formal_gate_evaluation_performed
effect_evidence
formal_phase2b_exit_claim
c1_exit_evidence
active_promotion_authorized
```

This namespace does not prohibit future external execution evidence from
establishing an applicable live fact. It prevents this static authorization
artifact from impersonating that evidence or flipping the fact itself.

The exact-freeze formal holdout NO-GO remains unchanged. An actual-unsealed
campaign, even if later executed successfully, is not a formal sealed holdout
and cannot by itself support formal Phase-2B exit, C1 effectiveness, or ACTIVE
promotion.

## 7. Versioning and revocation

This V1 artifact is immutable and has authorization epoch 1. It supersedes no
prior authorization. Any change to scope, actor topology, frozen identity,
baseline family, campaign class, or claim boundary requires a new major
authorization version and a new content ID; the historical file must remain.

Revocation requires a separate owner artifact. A future launch overlay must
check the latest revocation status before allocating IDs or admitting any
stage. Revocation is prospective by default and cannot manufacture, erase, or
upgrade execution evidence. Retroactive invalidation must explicitly identify
the affected campaign/attempt or logical sequence in the later revocation
artifact.

## 8. Exact allowed statement

> The owner has prospectively authorized construction of an independent
> generator, three frozen baseline registration campaigns, and supporting
> readiness/campaign machinery, and has conditionally authorized at most one
> future fresh actual-unsealed 960-case campaign after independently verified
> prerequisites and a separate launch overlay. No campaign or attempt ID is
> allocated here, and no implementation, formal seal, execution, score, gate,
> effect, or Phase-2B/C1 exit is claimed.
