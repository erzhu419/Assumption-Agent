from __future__ import annotations

import ast
from dataclasses import dataclass, fields, replace
from enum import Enum
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from hegel_machine import phase2b_c1_three_baseline_registration_v1 as mechanics


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "src/hegel_machine/phase2b_c1_three_baseline_registration_v1.py"
CONFIG_PATH = ROOT / "config/phase2b_c1_three_baseline_registration_profile_v1.json"
DOC_PATH = ROOT / "docs/Hegel_Machine_Phase2B_C1_Three_Baseline_Registration_Profile_v1.md"

PROFILE_DOMAIN = b"HEGEL/PHASE2B/C1/THREE_BASELINE/PROFILE/V1\x00"
PROFILE_PREFIX = "phase2b_c1_three_baseline_registration_profile_v1_"
FILE_MANIFEST_DOMAIN = b"HEGEL/PHASE2B/C1/THREE_BASELINE/ARTIFACT_FILE_MANIFEST/V1\x00"
R05_DOMAIN = b"HEGEL/PHASE2B/C1/THREE_BASELINE/R05_PREDECESSOR_CANDIDATE/V1\x00"
KEY_DOMAIN = b"HEGEL/PHASE2B/C1/THREE_BASELINE/KEY/V1\x00"
IMPLEMENTATION_DOMAIN = b"HEGEL/PHASE2B/C1/THREE_BASELINE/IMPLEMENTATION/V1\x00"
BUILDER_SUBJECT_DOMAIN = b"HEGEL/PHASE2B/C1/THREE_BASELINE/BUILDER_SUBJECT/V1\x00"
BUILDER_SIGNATURE_DOMAIN = b"HEGEL/PHASE2B/C1/THREE_BASELINE/BUILDER_SIGNATURE/V1\x00"
BUILDER_ATTESTATION_DOMAIN = b"HEGEL/PHASE2B/C1/THREE_BASELINE/BUILDER_ATTESTATION/V1\x00"
QUALIFIER_SIGNATURE_DOMAIN = b"HEGEL/PHASE2B/C1/THREE_BASELINE/QUALIFIER_SIGNATURE/V1\x00"
QUALIFIER_ATTESTATION_DOMAIN = b"HEGEL/PHASE2B/C1/THREE_BASELINE/QUALIFIER_ATTESTATION/V1\x00"
RECORD_DOMAIN = b"HEGEL/PHASE2B/C1/THREE_BASELINE/RICH_RECORD/V1\x00"
CUSTODIAN_SIGNATURE_DOMAIN = b"HEGEL/PHASE2B/C1/THREE_BASELINE/CUSTODIAN_SIGNATURE/V1\x00"
CUSTODIAN_ATTESTATION_DOMAIN = b"HEGEL/PHASE2B/C1/THREE_BASELINE/CUSTODIAN_ATTESTATION/V1\x00"
STORE_SIGNATURE_DOMAIN = b"HEGEL/PHASE2B/C1/THREE_BASELINE/STORE_SIGNATURE/V1\x00"
STORE_RECEIPT_DOMAIN = b"HEGEL/PHASE2B/C1/THREE_BASELINE/STORE_RECEIPT/V1\x00"
TIME_SIGNATURE_DOMAIN = b"HEGEL/PHASE2B/C1/THREE_BASELINE/TIME_SIGNATURE/V1\x00"
TIME_TOKEN_DOMAIN = b"HEGEL/PHASE2B/C1/THREE_BASELINE/TIME_TOKEN/V1\x00"
FINAL_DOMAIN = b"HEGEL/PHASE2B/C1/THREE_BASELINE/FINAL_REGISTRATION/V1\x00"
REGISTRY_ROOT_DOMAIN = b"HEGEL/PHASE2B/C1/THREE_BASELINE/REGISTRY_ROOT/V1\x00"
REGISTRY_ID_DOMAIN = b"HEGEL/PHASE2B/C1/THREE_BASELINE/REGISTRY_ID/V1\x00"
SNAPSHOT_DOMAIN = b"HEGEL/PHASE2B/C1/THREE_BASELINE/INPUT_SNAPSHOT/V1\x00"

FILE_MANIFEST_PREFIX = "phase2b_c1_baseline_artifact_file_manifest_v1_"
R05_PREFIX = "phase2b_c1_r05_predecessor_candidate_v1_"
KEY_PREFIX = "phase2b_c1_baseline_candidate_key_v1_"
IMPLEMENTATION_PREFIX = "phase2b_c1_baseline_implementation_v1_"
BUILDER_ATTESTATION_PREFIX = "phase2b_c1_baseline_builder_attestation_candidate_v1_"
QUALIFIER_ATTESTATION_PREFIX = "phase2b_c1_baseline_qualifier_attestation_candidate_v1_"
RECORD_PREFIX = "phase2b_c1_rich_baseline_registration_record_v1_"
CUSTODIAN_ATTESTATION_PREFIX = "phase2b_c1_baseline_custodian_attestation_candidate_v1_"
STORE_RECEIPT_PREFIX = "phase2b_c1_baseline_store_if_absent_receipt_candidate_v1_"
TIME_TOKEN_PREFIX = "phase2b_c1_baseline_declared_time_token_candidate_v1_"
FINAL_PREFIX = "phase2b_c1_final_baseline_registration_candidate_v1_"
REGISTRY_PREFIX = "phase2b_c1_three_baseline_registration_candidate_registry_v1_"

PROFILE_ID = (
    "phase2b_c1_three_baseline_registration_profile_v1_"
    "27cde92a67d2d92f83e52c9b60c84ace1138df7df134756b38a4cbfb98f2166d"
)

PUBLIC_KEYS_HEX = (
    "5b5d8b025ce4a8bd61767c6652985095148f98fa6029778754f8e20d38c91186",
    "cfb351f5c7b8b0240679d849a0e1f100106193430e2bf3b71f3a043e064341f1",
    "c9617fbe0b93bf1381911f73b186fc72aceb27a0098df0b662e2299b99bee42d",
    "f2a8239e57c751651b63ed0956bcb04bd019de39e510b928754a3d0c5eb83c56",
    "3f4a416adf0eec08059418f6d9979c37e47e4002e6bfb25122e0a41d8f073057",
)

# TEST_ONLY_NON_AUTHORITY literals were produced once, in memory, from five
# deterministic test-only seeds under explicit fixture-only authorization.
# No private bytes were persisted.  Runtime tests only parse and verify the raw
# public keys/signatures below; they never derive a private key or sign.
SIGNATURES_HEX = (
    "cb798af9cd8264f1a3b3d6fd5fc0aa06f26693998494797a153d0d156473c7500451b3548f5b7678286f2b726c8566081b537c74f956e5a1d5d5712e39573507",
    "92398da5430612fde8a86f5801988d83c6e08079029d9962c3902d59fc040b5fe06faa8d158a420b025311df17b6a3b9082e9d24a8dced53726389c1ca5a5c0e",
    "132b82e70bb3bffb4c0ac7e8a48eee670c647538971db26e965f707c1394302f0d42f4463da3504b53fb0b474574fbd1c3510bad836db905ac6f0f32e2aba801",
    "1880c94543273756da06a250b57c3d530f5231cdac926354da9de06f54bc01648be5bdfa9587ebe20c7aeff2773888e86e255f791aa5b2e4600b55dda5924b08",
    "d316a8c5460bfef7de6df219d47a8c695c1b16a43c69258b96c4d10f1f854a9d27ff695155f15b66c4745bf4e3937759f22727e6557739700682ad655dc1ca04",
    "7fd7f17acc4bcf22b261ebfb3b9e48a073413bcfc4992896bd0804f3bd593e36ed67a37200d4b9b2e96bdf61104304bbe6ff9dfcefd6cf972c0ecb6547096a09",
    "36bb6da915348db949d3483a4dabc0bc6f104c8d51a45060f5de087a4d979791e275057026b633ed7539ad80b3934b2c65cacac28d60b602a7385851170fe50c",
    "945d422ad6e01550c75d8a22cfde77b09da0c8009b266a2503e2c9605bc83980722fe96780b9aa6192f864e1a1b9d07b5d2aa201e34d249a36e4d1418fa5cb07",
    "69d72e9a82691ab2aed2c1db1b2afc87ad7750d0a65c3fc760e2c1fa4067935023e54cf098826dae6cef55586b4293bd9378feb2419e0582c56a8605e38aeb08",
    "72afc8766b987f0c5894a3509da160ec790b79411eba74596979559f00bb14ee847986d94a7ab898f038b2ce6c4500aa19f86b158c654bfabc33db2e8d5a0b09",
    "ee17ed7b1dbedecb0fd8934ce811e73ddb8395312cb14e389b580381797ecc762a69e7d5c11ef262770f735e3e96f761837919d187e663b4a10e3fdab1846e07",
    "acf521a1e724e98efeee3a5a00adfad52ee702f17ff8a1624138b2b515782fbd95086fb0a42022704c02465deee94440447d94b3cab11861bd2e93b2d5598501",
    "bbdf76955afb39221697917e0264d5bd437a64bd69e7e08c3f5ebf8da83095a371b490d65deca4ed82a1351560c6764b5a017ce788a7483a2bc5164a7df1c204",
    "bca4111ef4a39661cf971835660cef0f974c80eb90950d618b4d479b90095a4b00d9c632b07c6006fd986ebb573cb628ce4578fa4c8410935edeb069836afd05",
    "c271f4aec249aaedacd667865970886c7355ba606d73d07eaed6f36e19b0ec7988ffd88524543ffa4dbf1c91dfcdaca6cb090f2a24634bbd3dde2506ba017b01",
)

PREIMAGE_SHA256 = (
    "dd6c48eae3a74bc4a65d2171389a1af932427a709e0bd2321c579fdd3b33f3a8",
    "65938ea25c7bd594aa061e904c575ca9cde1b6b99111d3eff53c83c53ebc8115",
    "1d0b99230f92ddacdbddaf7c0a5f1ae55ea023b89d0165310b635d728a83708e",
    "ace0ffcaef93674891d9e6e778c08cbc00c12213ba2a91b32e911e22a10131fd",
    "65edd6ba236f0061a1f412a84dc226bc132c00122ac21859b5b4e35cd983dc8f",
    "9c9071b24e3f9cbd64d7a6961f5fc1ee6e58d5363d36851632e97ab75e28ef89",
    "7f1fe3ea0032ce575f2f87fea7579f12a4cc828b5b3cf1b4efec9e17ca87e79f",
    "ee6955407706d088120b6a701d8a07fef677b0bad2f2fd2867e696b3f043ddc2",
    "11554a07b6950d449837b32ae68199c062c9ff1ec4366531dbce2c1d005659a2",
    "69dc5ed893813c8fc6265b1e916d1616e518e77370e15af165e776b961e075ce",
    "0817b56dd339c9aaf6519ecaad56c3765d643fb33bd302f963b5b2119461ff95",
    "19dc00593f6863d7090e78ca4cb7cd4c2f368a63e6dd1fe531efd41648a93b56",
    "17f9a140269585f88386a06e3f00385c313b7bfe4b827a1a5cb1850d757afa87",
    "4cc15a349e0560f117e0e7a8129f250b79c31f6f1391070d760c335772c09d93",
    "992e28051c63ad79693cebcf9cfb0e06cc7350a6fda1126c245ef95dae9dbd30",
)

BASELINE_ROWS = (
    (
        mechanics.BaselineRegistrationKindV1.EMBEDDING_NEAREST_PROTOTYPE,
        "phase2b_baseline_spec_d14fa83a6843b92d1a502bba97b982323d1de8bf0abe22763d8c1cb67c97cfa4",
        "sentence-transformers/all-mpnet-base-v2",
        mechanics.BaselineRevisionPolicyV1.EXACT_40_HEX_COMMIT_REQUIRED,
        "a" * 40,
        None,
        None,
    ),
    (
        mechanics.BaselineRegistrationKindV1.FROZEN_LLM_SEMANTIC_ONLY,
        "phase2b_baseline_spec_c07005a2e32423f5e11a2160367b089d8f69208289239bfc18821fb351f13fa8",
        "Qwen/Qwen2.5-7B-Instruct",
        mechanics.BaselineRevisionPolicyV1.EXACT_40_HEX_COMMIT_REQUIRED,
        "b" * 40,
        None,
        None,
    ),
    (
        mechanics.BaselineRegistrationKindV1.FLAT_LEARNED_TYPED,
        "phase2b_baseline_spec_0e7b098030bf7e5a36f2deb0befd40fbabc55ce538a78765be2aa33d6b7af6d2",
        "sklearn.ensemble.HistGradientBoostingClassifier",
        mechanics.BaselineRevisionPolicyV1.REPOSITORY_DEPENDENCY_LOCK_REQUIRED,
        None,
        "c" * 40,
        hashlib.sha256(b"TEST HGB dependency lock").hexdigest(),
    ),
)

REQUIRED_ROLES = (
    (
        mechanics.BaselineArtifactRoleV1.SOURCE_ARCHIVE,
        mechanics.BaselineArtifactRoleV1.MODEL_WEIGHTS,
        mechanics.BaselineArtifactRoleV1.MODEL_CONFIGURATION,
        mechanics.BaselineArtifactRoleV1.TOKENIZER,
        mechanics.BaselineArtifactRoleV1.POOLING_CONFIGURATION,
        mechanics.BaselineArtifactRoleV1.DECODER_CONFIGURATION,
        mechanics.BaselineArtifactRoleV1.SBOM,
        mechanics.BaselineArtifactRoleV1.DEVELOPMENT_STATE,
        mechanics.BaselineArtifactRoleV1.OFFLINE_QUALIFICATION_REPORT,
        mechanics.BaselineArtifactRoleV1.LICENSE,
    ),
    (
        mechanics.BaselineArtifactRoleV1.SOURCE_ARCHIVE,
        mechanics.BaselineArtifactRoleV1.MODEL_WEIGHTS,
        mechanics.BaselineArtifactRoleV1.MODEL_CONFIGURATION,
        mechanics.BaselineArtifactRoleV1.TOKENIZER,
        mechanics.BaselineArtifactRoleV1.GENERATION_CONFIGURATION,
        mechanics.BaselineArtifactRoleV1.DECODER_CONFIGURATION,
        mechanics.BaselineArtifactRoleV1.SBOM,
        mechanics.BaselineArtifactRoleV1.DEVELOPMENT_STATE,
        mechanics.BaselineArtifactRoleV1.OFFLINE_QUALIFICATION_REPORT,
        mechanics.BaselineArtifactRoleV1.LICENSE,
    ),
    (
        mechanics.BaselineArtifactRoleV1.SOURCE_ARCHIVE,
        mechanics.BaselineArtifactRoleV1.DEPENDENCY_LOCK,
        mechanics.BaselineArtifactRoleV1.FEATURE_SCHEMA,
        mechanics.BaselineArtifactRoleV1.SERIALIZED_ESTIMATOR,
        mechanics.BaselineArtifactRoleV1.DECODER_CONFIGURATION,
        mechanics.BaselineArtifactRoleV1.SBOM,
        mechanics.BaselineArtifactRoleV1.DEVELOPMENT_STATE,
        mechanics.BaselineArtifactRoleV1.OFFLINE_QUALIFICATION_REPORT,
        mechanics.BaselineArtifactRoleV1.LICENSE,
    ),
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _cid(prefix: str, domain: bytes, body: object) -> str:
    return prefix + hashlib.sha256(domain + _canonical(body)).hexdigest()


def _root(domain: bytes, body: object) -> str:
    return hashlib.sha256(domain + _canonical(body)).hexdigest()


def _signed_id(prefix: str, domain: bytes, unsigned: dict[str, object], signature: bytes) -> str:
    return _cid(prefix, domain, {**unsigned, "signature_hex": signature.hex()})


def _preimage(domain: bytes, unsigned: dict[str, object]) -> bytes:
    return domain + _canonical(unsigned)


def _label_sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _snapshot_json_oracle(value: object) -> object:
    if value is None:
        return {"type": "none"}
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "int", "value": value}
    if type(value) is str:
        return {"type": "str", "value": value}
    if type(value) is bytes:
        return {"type": "bytes", "value_hex": value.hex()}
    if isinstance(value, Enum):
        return {
            "type": "enum",
            "class": type(value).__name__,
            "value": value.value,
        }
    if type(value) is tuple:
        return {
            "type": "tuple",
            "items": [_snapshot_json_oracle(item) for item in value],
        }
    if hasattr(type(value), "__dataclass_fields__"):
        return {
            "type": "dataclass",
            "class": type(value).__name__,
            "fields": [
                {
                    "name": item.name,
                    "value": _snapshot_json_oracle(getattr(value, item.name)),
                }
                for item in fields(type(value))
            ],
        }
    raise TypeError(type(value))


def _snapshot_commitment_oracle(graph: "KatGraph") -> str:
    body = _snapshot_json_oracle((graph.r05, graph.keys, graph.finals))
    return hashlib.sha256(SNAPSHOT_DOMAIN + _canonical(body)).hexdigest()


def _candidate_bindings_projection(
    value: mechanics.BaselineCallerSuppliedCandidateBindingsV1,
) -> dict[str, object]:
    return {
        "source_root_candidate_sha256": value.source_root_candidate_sha256,
        "source_archive_root_candidate_sha256": (
            value.source_archive_root_candidate_sha256
        ),
        "configuration_root_candidate_sha256": (
            value.configuration_root_candidate_sha256
        ),
        "decoder_root_candidate_sha256": value.decoder_root_candidate_sha256,
        "sbom_root_candidate_sha256": value.sbom_root_candidate_sha256,
        "runtime_image_candidate_digest": value.runtime_image_candidate_digest,
        "development_state_root_candidate_sha256": (
            value.development_state_root_candidate_sha256
        ),
        "offline_qualification_root_candidate_sha256": (
            value.offline_qualification_root_candidate_sha256
        ),
    }


def _revision_projection(upstream, repository_commit, dependency_lock) -> dict[str, object]:
    return {
        "upstream_revision_40hex": upstream,
        "repository_source_commit_40hex": repository_commit,
        "repository_dependency_lock_sha256": dependency_lock,
    }


@dataclass(frozen=True)
class KatGraph:
    r05: mechanics.R05QualificationPredecessorCandidateV1
    keys: tuple[mechanics.BaselineCandidatePublicKeyV1, ...]
    finals: tuple[mechanics.FinalBaselineRegistrationCandidateV1, ...]
    preimages: tuple[bytes, ...]


def _literal_provider(index: int, preimage: bytes) -> bytes:
    assert hashlib.sha256(preimage).hexdigest() == PREIMAGE_SHA256[index]
    return bytes.fromhex(SIGNATURES_HEX[index])


def _build_graph(signature_provider=_literal_provider) -> KatGraph:
    r05_body = {
        "readiness_node_id": "R05_SYNTHETIC_DEV_WIRE_COVERT_QUALIFICATION_VERIFIED",
        "qualification_receipt_id": (
            "phase2b_formal_wire_covert_qualification_receipt_v2_" + "2" * 64
        ),
        "source_commit_sha": "1" * 40,
        "source_archive_root_sha256": _label_sha("R05 source archive"),
        "dependency_lock_sha256": _label_sha("R05 dependency lock"),
        "runtime_image_digest": "sha256:" + _label_sha("R05 runtime image"),
        "synthetic_fixture_root_sha256": _label_sha("R05 synthetic fixture"),
        "candidate_scope": "SUPPLIED_R05_PREDECESSOR_CANDIDATE_NOT_AUTHORITY",
        "actual_corpus_input_used": False,
    }
    r05 = mechanics.R05QualificationPredecessorCandidateV1(
        **r05_body,
        candidate_id=_cid(R05_PREFIX, R05_DOMAIN, r05_body),
    )

    keys = []
    for purpose, public_hex in zip(
        mechanics.BaselineSignaturePurposeV1,
        PUBLIC_KEYS_HEX,
        strict=True,
    ):
        public = bytes.fromhex(public_hex)
        body = {
            "purpose": purpose.value,
            "key_epoch": 1,
            "algorithm": "Ed25519",
            "public_key_hex": public_hex,
        }
        keys.append(
            mechanics.BaselineCandidatePublicKeyV1(
                purpose=purpose,
                key_epoch=1,
                algorithm="Ed25519",
                public_key=public,
                key_id=_cid(KEY_PREFIX, KEY_DOMAIN, body),
            )
        )
    keys_tuple = tuple(keys)

    finals = []
    preimages = []
    previous_final_id = None
    previous_time_token_id = None
    for baseline_index, row in enumerate(BASELINE_ROWS):
        (
            kind,
            baseline_spec_id,
            implementation,
            revision_policy,
            upstream_revision,
            repository_commit,
            dependency_lock,
        ) = row
        roles = sorted(REQUIRED_ROLES[baseline_index], key=lambda item: item.value)
        file_rows = []
        for role_index, role in enumerate(roles):
            path = f"artifact/{role_index:02d}-{role.value.lower()}.bin"
            artifact_sha256 = _label_sha(f"{kind.value}:{path}")
            if role is mechanics.BaselineArtifactRoleV1.DEPENDENCY_LOCK:
                artifact_sha256 = dependency_lock
            file_rows.append(
                mechanics.BaselineArtifactFileV1(
                    role=role,
                    relative_path=path,
                    size_bytes=role_index + 1,
                    sha256=artifact_sha256,
                )
            )
        files = tuple(sorted(file_rows, key=lambda item: item.relative_path))
        file_projection = [
            {
                "role": item.role.value,
                "relative_path": item.relative_path,
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
            }
            for item in files
        ]
        manifest_root = _root(FILE_MANIFEST_DOMAIN, file_projection)
        manifest = mechanics.BaselineArtifactFileManifestV1(
            files=files,
            file_count=len(files),
            total_bytes=sum(item.size_bytes for item in files),
            complete=True,
            manifest_root_sha256=manifest_root,
            manifest_id=FILE_MANIFEST_PREFIX + manifest_root,
        )
        bindings = mechanics.BaselineCallerSuppliedCandidateBindingsV1(
            source_root_candidate_sha256=_label_sha(f"{kind.value}:source"),
            source_archive_root_candidate_sha256=_label_sha(
                f"{kind.value}:archive"
            ),
            configuration_root_candidate_sha256=_label_sha(
                f"{kind.value}:config"
            ),
            decoder_root_candidate_sha256=_label_sha(f"{kind.value}:decoder"),
            sbom_root_candidate_sha256=_label_sha(f"{kind.value}:sbom"),
            runtime_image_candidate_digest=(
                "sha256:" + _label_sha(f"{kind.value}:image")
            ),
            development_state_root_candidate_sha256=_label_sha(
                f"{kind.value}:dev-state"
            ),
            offline_qualification_root_candidate_sha256=_label_sha(
                f"{kind.value}:qualification"
            ),
        )
        implementation_body = {
            "kind": kind.value,
            "baseline_spec_id": baseline_spec_id,
            "implementation": implementation,
            "revision_policy": revision_policy.value,
            "revision_binding": _revision_projection(
                upstream_revision, repository_commit, dependency_lock
            ),
            "artifact_manifest_id": manifest.manifest_id,
            "caller_supplied_candidate_bindings": (
                _candidate_bindings_projection(bindings)
            ),
        }
        implementation_id = _cid(
            IMPLEMENTATION_PREFIX, IMPLEMENTATION_DOMAIN, implementation_body
        )

        builder_subject = {
            "profile_id": PROFILE_ID,
            "kind": kind.value,
            "baseline_spec_id": baseline_spec_id,
            "implementation_id": implementation_id,
            "artifact_manifest_id": manifest.manifest_id,
            "caller_supplied_candidate_bindings": (
                _candidate_bindings_projection(bindings)
            ),
        }
        builder_unsigned = {
            "purpose": "BASELINE_RUNTIME_BUILDER_ATTESTATION",
            "key_epoch": 1,
            "signer_key_id": keys_tuple[0].key_id,
            **builder_subject,
            "builder_subject_root_sha256": _root(
                BUILDER_SUBJECT_DOMAIN, builder_subject
            ),
        }
        builder_preimage = _preimage(BUILDER_SIGNATURE_DOMAIN, builder_unsigned)
        builder_signature = signature_provider(len(preimages), builder_preimage)
        preimages.append(builder_preimage)
        builder = mechanics.BaselineRuntimeBuilderAttestationCandidateV1(
            purpose=mechanics.BaselineSignaturePurposeV1.BASELINE_RUNTIME_BUILDER_ATTESTATION,
            key_epoch=1,
            signer_key_id=keys_tuple[0].key_id,
            profile_id=PROFILE_ID,
            kind=kind,
            baseline_spec_id=baseline_spec_id,
            implementation_id=implementation_id,
            artifact_manifest_id=manifest.manifest_id,
            candidate_bindings=bindings,
            builder_subject_root_sha256=builder_unsigned[
                "builder_subject_root_sha256"
            ],
            signature=builder_signature,
            attestation_id=_signed_id(
                BUILDER_ATTESTATION_PREFIX,
                BUILDER_ATTESTATION_DOMAIN,
                builder_unsigned,
                builder_signature,
            ),
        )

        qualifier_unsigned = {
            "purpose": "BASELINE_OFFLINE_QUALIFIER_ATTESTATION",
            "key_epoch": 1,
            "signer_key_id": keys_tuple[1].key_id,
            "profile_id": PROFILE_ID,
            "kind": kind.value,
            "baseline_spec_id": baseline_spec_id,
            "implementation_id": implementation_id,
            "builder_attestation_id": builder.attestation_id,
            "runtime_image_candidate_digest": (
                bindings.runtime_image_candidate_digest
            ),
            "offline_qualification_root_candidate_sha256": (
                bindings.offline_qualification_root_candidate_sha256
            ),
        }
        qualifier_preimage = _preimage(
            QUALIFIER_SIGNATURE_DOMAIN, qualifier_unsigned
        )
        qualifier_signature = signature_provider(len(preimages), qualifier_preimage)
        preimages.append(qualifier_preimage)
        qualifier = mechanics.BaselineOfflineQualifierAttestationCandidateV1(
            purpose=mechanics.BaselineSignaturePurposeV1.BASELINE_OFFLINE_QUALIFIER_ATTESTATION,
            key_epoch=1,
            signer_key_id=keys_tuple[1].key_id,
            profile_id=PROFILE_ID,
            kind=kind,
            baseline_spec_id=baseline_spec_id,
            implementation_id=implementation_id,
            builder_attestation_id=builder.attestation_id,
            runtime_image_candidate_digest=(
                bindings.runtime_image_candidate_digest
            ),
            offline_qualification_root_candidate_sha256=(
                bindings.offline_qualification_root_candidate_sha256
            ),
            signature=qualifier_signature,
            attestation_id=_signed_id(
                QUALIFIER_ATTESTATION_PREFIX,
                QUALIFIER_ATTESTATION_DOMAIN,
                qualifier_unsigned,
                qualifier_signature,
            ),
        )

        record_body = {
            "version": "hegel-machine-phase2b-c1-three-baseline-registration-mechanics/1",
            "claim_level": "MECHANICS_ONLY_NOT_REGISTRATIONS",
            "profile_id": PROFILE_ID,
            "protocol_id": mechanics.PROTOCOL_ID_V1,
            "exact_freeze_id": mechanics.EXACT_FREEZE_ID_V1,
            "owner_authorization_id": mechanics.OWNER_AUTHORIZATION_ID_V1,
            "readiness_dag_id": mechanics.READINESS_DAG_ID_V1,
            "r05_predecessor_candidate_id": r05.candidate_id,
            "kind": kind.value,
            "baseline_spec_id": baseline_spec_id,
            "implementation": implementation,
            "revision_policy": revision_policy.value,
            "revision_binding": _revision_projection(
                upstream_revision, repository_commit, dependency_lock
            ),
            "artifact_manifest_id": manifest.manifest_id,
            "artifact_manifest_root_sha256": manifest.manifest_root_sha256,
            "caller_supplied_candidate_bindings": (
                _candidate_bindings_projection(bindings)
            ),
            "implementation_id": implementation_id,
            "builder_attestation_id": builder.attestation_id,
            "qualifier_attestation_id": qualifier.attestation_id,
            "previous_final_registration_id": previous_final_id,
            "holdout_generation_count": 0,
            "actual_case_count": 0,
            "campaign_id": None,
            "attempt_id": None,
        }
        record = mechanics.RichBaselineRegistrationRecordV1(
            version=record_body["version"],
            claim_level=record_body["claim_level"],
            profile_id=PROFILE_ID,
            protocol_id=mechanics.PROTOCOL_ID_V1,
            exact_freeze_id=mechanics.EXACT_FREEZE_ID_V1,
            owner_authorization_id=mechanics.OWNER_AUTHORIZATION_ID_V1,
            readiness_dag_id=mechanics.READINESS_DAG_ID_V1,
            r05_predecessor_candidate_id=r05.candidate_id,
            kind=kind,
            baseline_spec_id=baseline_spec_id,
            implementation=implementation,
            revision_policy=revision_policy,
            upstream_revision_40hex=upstream_revision,
            repository_source_commit_40hex=repository_commit,
            repository_dependency_lock_sha256=dependency_lock,
            artifact_manifest=manifest,
            candidate_bindings=bindings,
            implementation_id=implementation_id,
            builder_attestation_id=builder.attestation_id,
            qualifier_attestation_id=qualifier.attestation_id,
            previous_final_registration_id=previous_final_id,
            holdout_generation_count=0,
            actual_case_count=0,
            campaign_id=None,
            attempt_id=None,
            record_id=_cid(RECORD_PREFIX, RECORD_DOMAIN, record_body),
        )

        custodian_unsigned = {
            "purpose": "BASELINE_REGISTRATION_CUSTODIAN_ATTESTATION",
            "key_epoch": 1,
            "signer_key_id": keys_tuple[2].key_id,
            "profile_id": PROFILE_ID,
            "kind": kind.value,
            "record_id": record.record_id,
            "r05_predecessor_candidate_id": r05.candidate_id,
            "previous_final_registration_id": previous_final_id,
        }
        custodian_preimage = _preimage(
            CUSTODIAN_SIGNATURE_DOMAIN, custodian_unsigned
        )
        custodian_signature = signature_provider(len(preimages), custodian_preimage)
        preimages.append(custodian_preimage)
        custodian = mechanics.BaselineRegistrationCustodianAttestationCandidateV1(
            purpose=mechanics.BaselineSignaturePurposeV1.BASELINE_REGISTRATION_CUSTODIAN_ATTESTATION,
            key_epoch=1,
            signer_key_id=keys_tuple[2].key_id,
            profile_id=PROFILE_ID,
            kind=kind,
            record_id=record.record_id,
            r05_predecessor_candidate_id=r05.candidate_id,
            previous_final_registration_id=previous_final_id,
            signature=custodian_signature,
            attestation_id=_signed_id(
                CUSTODIAN_ATTESTATION_PREFIX,
                CUSTODIAN_ATTESTATION_DOMAIN,
                custodian_unsigned,
                custodian_signature,
            ),
        )

        object_key = (
            f"phase2b/c1/baseline-registration/v1/{kind.value}/{record.record_id}.json"
        )
        store_unsigned = {
            "purpose": "STORE_IF_ABSENT_RECEIPT",
            "key_epoch": 1,
            "signer_key_id": keys_tuple[3].key_id,
            "profile_id": PROFILE_ID,
            "kind": kind.value,
            "record_id": record.record_id,
            "custodian_attestation_id": custodian.attestation_id,
            "object_key": object_key,
            "previous_generation": 0,
            "new_generation": 1,
            "condition": "IF_ABSENT",
            "stored_content_sha256": hashlib.sha256(_canonical(record_body)).hexdigest(),
        }
        store_preimage = _preimage(STORE_SIGNATURE_DOMAIN, store_unsigned)
        store_signature = signature_provider(len(preimages), store_preimage)
        preimages.append(store_preimage)
        store = mechanics.BaselineStoreIfAbsentReceiptCandidateV1(
            purpose=mechanics.BaselineSignaturePurposeV1.STORE_IF_ABSENT_RECEIPT,
            key_epoch=1,
            signer_key_id=keys_tuple[3].key_id,
            profile_id=PROFILE_ID,
            kind=kind,
            record_id=record.record_id,
            custodian_attestation_id=custodian.attestation_id,
            object_key=object_key,
            previous_generation=0,
            new_generation=1,
            condition="IF_ABSENT",
            stored_content_sha256=store_unsigned["stored_content_sha256"],
            signature=store_signature,
            receipt_id=_signed_id(
                STORE_RECEIPT_PREFIX,
                STORE_RECEIPT_DOMAIN,
                store_unsigned,
                store_signature,
            ),
        )

        time_unsigned = {
            "purpose": "DECLARED_TIME_TOKEN",
            "key_epoch": 1,
            "signer_key_id": keys_tuple[4].key_id,
            "profile_id": PROFILE_ID,
            "kind": kind.value,
            "record_id": record.record_id,
            "custodian_attestation_id": custodian.attestation_id,
            "store_receipt_id": store.receipt_id,
            "logical_sequence": baseline_index,
            "previous_time_token_id": previous_time_token_id,
            "declared_unix_seconds": 1_700_000_000 + baseline_index,
            "declared_nanosecond": baseline_index,
        }
        time_preimage = _preimage(TIME_SIGNATURE_DOMAIN, time_unsigned)
        time_signature = signature_provider(len(preimages), time_preimage)
        preimages.append(time_preimage)
        time_token = mechanics.BaselineDeclaredTimeTokenCandidateV1(
            purpose=mechanics.BaselineSignaturePurposeV1.DECLARED_TIME_TOKEN,
            key_epoch=1,
            signer_key_id=keys_tuple[4].key_id,
            profile_id=PROFILE_ID,
            kind=kind,
            record_id=record.record_id,
            custodian_attestation_id=custodian.attestation_id,
            store_receipt_id=store.receipt_id,
            logical_sequence=baseline_index,
            previous_time_token_id=previous_time_token_id,
            declared_unix_seconds=time_unsigned["declared_unix_seconds"],
            declared_nanosecond=time_unsigned["declared_nanosecond"],
            signature=time_signature,
            token_id=_signed_id(
                TIME_TOKEN_PREFIX,
                TIME_TOKEN_DOMAIN,
                time_unsigned,
                time_signature,
            ),
        )
        final_body = {
            "profile_id": PROFILE_ID,
            "kind": kind.value,
            "record_id": record.record_id,
            "builder_attestation_id": builder.attestation_id,
            "qualifier_attestation_id": qualifier.attestation_id,
            "custodian_attestation_id": custodian.attestation_id,
            "store_receipt_id": store.receipt_id,
            "time_token_id": time_token.token_id,
        }
        final = mechanics.FinalBaselineRegistrationCandidateV1(
            record=record,
            builder_attestation=builder,
            qualifier_attestation=qualifier,
            custodian_attestation=custodian,
            store_receipt=store,
            time_token=time_token,
            final_registration_id=_cid(FINAL_PREFIX, FINAL_DOMAIN, final_body),
        )
        finals.append(final)
        previous_final_id = final.final_registration_id
        previous_time_token_id = time_token.token_id
    return KatGraph(r05, keys_tuple, tuple(finals), tuple(preimages))


def _verify(graph: KatGraph):
    return mechanics.verify_three_baseline_registration_mechanics_v1(
        r05_predecessor=graph.r05,
        candidate_keys=graph.keys,
        final_registrations=graph.finals,
    )


def _rebuild_final_with_declared_time(
    final: mechanics.FinalBaselineRegistrationCandidateV1,
    *,
    seconds: int,
    nanosecond: int,
) -> mechanics.FinalBaselineRegistrationCandidateV1:
    old = final.time_token
    unsigned = {
        "purpose": old.purpose.value,
        "key_epoch": old.key_epoch,
        "signer_key_id": old.signer_key_id,
        "profile_id": old.profile_id,
        "kind": old.kind.value,
        "record_id": old.record_id,
        "custodian_attestation_id": old.custodian_attestation_id,
        "store_receipt_id": old.store_receipt_id,
        "logical_sequence": old.logical_sequence,
        "previous_time_token_id": old.previous_time_token_id,
        "declared_unix_seconds": seconds,
        "declared_nanosecond": nanosecond,
    }
    token = mechanics.BaselineDeclaredTimeTokenCandidateV1(
        purpose=old.purpose,
        key_epoch=old.key_epoch,
        signer_key_id=old.signer_key_id,
        profile_id=old.profile_id,
        kind=old.kind,
        record_id=old.record_id,
        custodian_attestation_id=old.custodian_attestation_id,
        store_receipt_id=old.store_receipt_id,
        logical_sequence=old.logical_sequence,
        previous_time_token_id=old.previous_time_token_id,
        declared_unix_seconds=seconds,
        declared_nanosecond=nanosecond,
        signature=old.signature,
        token_id=_signed_id(
            TIME_TOKEN_PREFIX,
            TIME_TOKEN_DOMAIN,
            unsigned,
            old.signature,
        ),
    )
    final_body = {
        "profile_id": PROFILE_ID,
        "kind": final.record.kind.value,
        "record_id": final.record.record_id,
        "builder_attestation_id": final.builder_attestation.attestation_id,
        "qualifier_attestation_id": final.qualifier_attestation.attestation_id,
        "custodian_attestation_id": final.custodian_attestation.attestation_id,
        "store_receipt_id": final.store_receipt.receipt_id,
        "time_token_id": token.token_id,
    }
    return replace(
        final,
        time_token=token,
        final_registration_id=_cid(FINAL_PREFIX, FINAL_DOMAIN, final_body),
    )


def _standalone_time_candidate(
    *,
    seconds: int,
    nanosecond: int,
) -> mechanics.BaselineDeclaredTimeTokenCandidateV1:
    graph = _build_graph()
    old = graph.finals[0].time_token
    unsigned = {
        "purpose": old.purpose.value,
        "key_epoch": old.key_epoch,
        "signer_key_id": old.signer_key_id,
        "profile_id": old.profile_id,
        "kind": old.kind.value,
        "record_id": old.record_id,
        "custodian_attestation_id": old.custodian_attestation_id,
        "store_receipt_id": old.store_receipt_id,
        "logical_sequence": old.logical_sequence,
        "previous_time_token_id": old.previous_time_token_id,
        "declared_unix_seconds": seconds,
        "declared_nanosecond": nanosecond,
    }
    signature = bytes(64)
    return mechanics.BaselineDeclaredTimeTokenCandidateV1(
        purpose=old.purpose,
        key_epoch=old.key_epoch,
        signer_key_id=old.signer_key_id,
        profile_id=old.profile_id,
        kind=old.kind,
        record_id=old.record_id,
        custodian_attestation_id=old.custodian_attestation_id,
        store_receipt_id=old.store_receipt_id,
        logical_sequence=old.logical_sequence,
        previous_time_token_id=old.previous_time_token_id,
        declared_unix_seconds=seconds,
        declared_nanosecond=nanosecond,
        signature=signature,
        token_id=_signed_id(
            TIME_TOKEN_PREFIX,
            TIME_TOKEN_DOMAIN,
            unsigned,
            signature,
        ),
    )


def test_independent_profile_canonical_hash_and_checked_in_bytes() -> None:
    wrapper = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    body = wrapper["profile_body"]
    assert wrapper["profile_id"] == _cid(PROFILE_PREFIX, PROFILE_DOMAIN, body)
    assert wrapper["profile_id"] == PROFILE_ID
    assert mechanics.phase2b_c1_three_baseline_registration_profile_v1() == wrapper
    assert CONFIG_PATH.read_bytes() == _canonical(wrapper) + b"\n"
    assert body["canonicalization"] == "RFC8785_CANONICAL_BODY_PLUS_ONE_LF"
    assert body["safety_boundary"]["artifact_roots_replayed"] is False
    assert body["baselines_in_order"][0]["baseline_spec_id"].endswith(
        "d14fa83a6843b92d1a502bba97b982323d1de8bf0abe22763d8c1cb67c97cfa4"
    )
    assert body["baselines_in_order"][1]["baseline_spec_id"].endswith(
        "c07005a2e32423f5e11a2160367b089d8f69208289239bfc18821fb351f13fa8"
    )
    assert body["baselines_in_order"][2]["baseline_spec_id"].endswith(
        "0e7b098030bf7e5a36f2deb0befd40fbabc55ce538a78765be2aa33d6b7af6d2"
    )


def test_literal_fifteen_signature_kat_graph_reaches_real_ed25519_success() -> None:
    graph = _build_graph()
    assert len(graph.preimages) == len(SIGNATURES_HEX) == 15
    assert tuple(
        item.record.artifact_manifest.manifest_root_sha256
        for item in graph.finals
    ) == (
        "0731ed146b39f35be836a0d7f28ffc70cbe0d557ddd190ccb0d7b5bdef23c63b",
        "3cfbec12bec5985f48b40d36fe4b48be71c2d4cfa7a6ebcc6ec3c50b4d240049",
        "0a573131bda9269b3fc3aad64a147de956c18982e7df8b3afc0ea9dc59c1901d",
    )
    result = _verify(graph)
    assert type(result) is mechanics.ThreeBaselineRegistrationMechanicsV1
    assert result.disposition is (
        mechanics.ThreeBaselineRegistrationDispositionV1.MECHANICS_ONLY_NOT_REGISTRATIONS
    )
    assert result.verified_signature_count == 15
    assert result.fifteen_caller_supplied_ed25519_signatures_verified is True
    assert result.r05_authority_verified is False
    assert result.r07_baseline_registrations_frozen is False
    assert result.r08_statistical_validation_unlocked is False
    assert result.aggregate_snapshot_persisted is False
    assert result.baseline_registrations_exist is False
    assert result.artifact_roots_replayed is False
    assert result.result_object_unforgeable is False
    assert result.cooperative_computation_only is True
    assert result.downstream_fresh_reverification_required is True
    assert result.durable_cas_verified is False
    assert result.authoritative_time_verified is False
    assert result.actual_execution_claimed is False
    assert result.input_snapshot_commitment_sha256 == _snapshot_commitment_oracle(
        graph
    )
    assert result.input_snapshot_commitment_sha256 == (
        "48b8987df884dfa5b09ec6d1ff6428fe68d282f9aaa781b7aa548c47eed81304"
    )
    assert result.result_id == (
        "phase2b_c1_three_baseline_registration_mechanics_result_v1_"
        "e91d92fb8a273392bc2b3f093dcdd2efd13fcd944168926d311b0af0ac95954d"
    )
    assert tuple(item.kind.value for item in result.legacy_projection_candidates) == tuple(
        item.value for item in mechanics.BaselineRegistrationKindV1
    )
    assert all(
        item.eligible_for_execution_freeze is False
        and item.r07_registration_verified is False
        for item in result.legacy_projection_candidates
    )
    assert tuple(
        item.projection_id for item in result.legacy_projection_candidates
    ) == (
        "phase2b_c1_legacy_baseline_projection_candidate_v1_63e8a8b5bf8e9f17d49653081d99ad086d03e00980daa3c852dfbe23d5eaf62c",
        "phase2b_c1_legacy_baseline_projection_candidate_v1_9293a674a9c203ced9e3212b03dae6c202982b64fe72161d0982a3e19df68536",
        "phase2b_c1_legacy_baseline_projection_candidate_v1_8020a290814d588281743fd7574e38e572b201f3d6b6fcce139b68c3e6ae6013",
    )
    final_ids = tuple(item.final_registration_id for item in graph.finals)
    expected_root = _root(REGISTRY_ROOT_DOMAIN, list(final_ids))
    expected_id = _cid(
        REGISTRY_PREFIX,
        REGISTRY_ID_DOMAIN,
        {
            "profile_id": PROFILE_ID,
            "r05_predecessor_candidate_id": graph.r05.candidate_id,
            "final_registration_candidate_ids": list(final_ids),
            "candidate_registry_root_sha256": expected_root,
        },
    )
    assert result.candidate_registry_root_sha256 == expected_root
    assert result.candidate_registry_id == expected_id


def test_fresh_real_verify_order_is_three_baselines_by_five_purposes(monkeypatch) -> None:
    graph = _build_graph()
    calls = []
    real_verify = mechanics._verify_ed25519

    def recording_verify(public_key: bytes, signature: bytes, message: bytes) -> None:
        calls.append((public_key, signature, message))
        real_verify(public_key, signature, message)

    monkeypatch.setattr(mechanics, "_verify_ed25519", recording_verify)
    result = _verify(graph)
    assert type(result) is mechanics.ThreeBaselineRegistrationMechanicsV1
    assert len(calls) == 15
    expected_public = tuple(
        bytes.fromhex(PUBLIC_KEYS_HEX[purpose_index])
        for _baseline_index in range(3)
        for purpose_index in range(5)
    )
    assert tuple(item[0] for item in calls) == expected_public
    assert tuple(item[2] for item in calls) == graph.preimages


@pytest.mark.parametrize("verification_boundary", range(15))
def test_mutation_at_every_signature_boundary_rejects_atomically(
    monkeypatch,
    verification_boundary: int,
) -> None:
    graph = _build_graph()
    real_verify = mechanics._verify_ed25519
    call_count = 0

    def verify_then_mutate(
        public_key: bytes,
        signature: bytes,
        message: bytes,
    ) -> None:
        nonlocal call_count
        real_verify(public_key, signature, message)
        if call_count == verification_boundary:
            object.__setattr__(
                graph.r05,
                "candidate_scope",
                "MUTATED_DURING_FRESH_VERIFICATION",
            )
        call_count += 1

    monkeypatch.setattr(mechanics, "_verify_ed25519", verify_then_mutate)
    result = _verify(graph)
    assert type(result) is mechanics.ThreeBaselineRegistrationRejectionV1
    assert result.reason is (
        mechanics.ThreeBaselineRegistrationReasonV1.INPUT_MUTATED_DURING_VERIFICATION
    )
    assert result.verified_signature_count == 0
    assert result.final_registration_candidate_ids == ()
    assert result.legacy_projection_candidates == ()
    assert result.partial_output_published is False


def test_cooperative_result_requires_full_fresh_reverification(monkeypatch) -> None:
    graph = _build_graph()
    result = _verify(graph)
    assert type(result) is mechanics.ThreeBaselineRegistrationMechanicsV1

    real_verify = mechanics._verify_ed25519
    calls = 0

    def counting_real_verify(
        public_key: bytes,
        signature: bytes,
        message: bytes,
    ) -> None:
        nonlocal calls
        real_verify(public_key, signature, message)
        calls += 1

    monkeypatch.setattr(mechanics, "_verify_ed25519", counting_real_verify)
    assert mechanics.verify_three_baseline_registration_mechanics_result_v1(
        r05_predecessor=graph.r05,
        candidate_keys=graph.keys,
        final_registrations=graph.finals,
        result=result,
    )
    assert calls == 15

    forged_id = replace(result, result_id=result.result_id[:-1] + "0")
    assert not mechanics.verify_three_baseline_registration_mechanics_result_v1(
        r05_predecessor=graph.r05,
        candidate_keys=graph.keys,
        final_registrations=graph.finals,
        result=forged_id,
    )
    forged_claim = replace(result, artifact_roots_replayed=True)
    assert not mechanics.verify_three_baseline_registration_mechanics_result_v1(
        r05_predecessor=graph.r05,
        candidate_keys=graph.keys,
        final_registrations=graph.finals,
        result=forged_claim,
    )

    object.__setattr__(graph.r05, "actual_corpus_input_used", True)
    assert not mechanics.verify_three_baseline_registration_mechanics_result_v1(
        r05_predecessor=graph.r05,
        candidate_keys=graph.keys,
        final_registrations=graph.finals,
        result=result,
    )


def test_result_and_private_helper_are_explicitly_cooperative_not_evidence() -> None:
    graph = _build_graph()
    result = _verify(graph)
    assert type(result) is mechanics.ThreeBaselineRegistrationMechanicsV1
    assert result.result_object_unforgeable is False
    assert result.downstream_fresh_reverification_required is True
    # Ordinary dataclass construction remains possible; it is intentionally
    # non-evidence until verify_*_result reruns the complete graph.
    fabricated = replace(result, verified_signature_count=0)
    assert type(fabricated) is mechanics.ThreeBaselineRegistrationMechanicsV1
    assert not mechanics.verify_three_baseline_registration_mechanics_result_v1(
        r05_predecessor=graph.r05,
        candidate_keys=graph.keys,
        final_registrations=graph.finals,
        result=fabricated,
    )
    assert tuple(inspect.signature(mechanics._success).parameters) == ("snapshot",)
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "_ISSUANCE_TOKEN" not in source
    assert "privately issued" not in source


@pytest.mark.parametrize("signature_index", range(15))
def test_every_signature_tamper_rejects_atomically(signature_index: int) -> None:
    def tampered_provider(index: int, _preimage: bytes) -> bytes:
        signature = bytes.fromhex(SIGNATURES_HEX[index])
        if index == signature_index:
            return bytes([signature[0] ^ 1]) + signature[1:]
        return signature

    # The independent oracle rebuilds every affected content ID and downstream
    # link.  Rejection therefore reaches real Ed25519 verification instead of
    # being explained by a stale candidate ID.
    graph = _build_graph(tampered_provider)
    result = _verify(graph)
    assert type(result) is mechanics.ThreeBaselineRegistrationRejectionV1
    assert result.reason is mechanics.ThreeBaselineRegistrationReasonV1.SIGNATURE_INVALID
    assert result.final_registration_candidate_ids == ()
    assert result.legacy_projection_candidates == ()
    assert result.verified_signature_count == 0
    assert result.partial_output_published is False


@pytest.mark.parametrize(
    ("baseline_index", "component", "attribute", "prefix"),
    [
        (baseline_index, component, attribute, prefix)
        for baseline_index in range(3)
        for component, attribute, prefix in (
            ("record.artifact_manifest", "manifest_id", FILE_MANIFEST_PREFIX),
            ("record", "implementation_id", IMPLEMENTATION_PREFIX),
            ("builder_attestation", "attestation_id", BUILDER_ATTESTATION_PREFIX),
            ("qualifier_attestation", "attestation_id", QUALIFIER_ATTESTATION_PREFIX),
            ("record", "record_id", RECORD_PREFIX),
            ("custodian_attestation", "attestation_id", CUSTODIAN_ATTESTATION_PREFIX),
            ("store_receipt", "receipt_id", STORE_RECEIPT_PREFIX),
            ("time_token", "token_id", TIME_TOKEN_PREFIX),
            ("self", "final_registration_id", FINAL_PREFIX),
        )
    ],
)
def test_every_per_registration_content_id_tamper_rejects_atomically(
    baseline_index: int,
    component: str,
    attribute: str,
    prefix: str,
) -> None:
    graph = _build_graph()
    target = graph.finals[baseline_index]
    if component == "record.artifact_manifest":
        target = target.record.artifact_manifest
    elif component == "self":
        pass
    else:
        target = getattr(target, component)
    object.__setattr__(target, attribute, prefix + "f" * 64)
    result = _verify(graph)
    assert type(result) is mechanics.ThreeBaselineRegistrationRejectionV1
    assert result.reason is (
        mechanics.ThreeBaselineRegistrationReasonV1.STRUCTURAL_CLOSURE_INVALID
    )
    assert result.final_registration_candidate_ids == ()
    assert result.partial_output_published is False


@pytest.mark.parametrize("key_index", range(5))
def test_every_key_id_tamper_rejects_atomically(key_index: int) -> None:
    graph = _build_graph()
    object.__setattr__(graph.keys[key_index], "key_id", KEY_PREFIX + "f" * 64)
    result = _verify(graph)
    assert type(result) is mechanics.ThreeBaselineRegistrationRejectionV1
    assert result.final_registration_candidate_ids == ()
    assert result.partial_output_published is False


def test_r05_candidate_id_tamper_rejects_atomically() -> None:
    graph = _build_graph()
    object.__setattr__(graph.r05, "candidate_id", R05_PREFIX + "f" * 64)
    result = _verify(graph)
    assert type(result) is mechanics.ThreeBaselineRegistrationRejectionV1
    assert result.final_registration_candidate_ids == ()
    assert result.partial_output_published is False


def test_sequence_reorder_and_cross_baseline_splice_reject_atomically() -> None:
    graph = _build_graph()
    reordered = KatGraph(graph.r05, graph.keys, tuple(reversed(graph.finals)), graph.preimages)
    assert type(_verify(reordered)) is mechanics.ThreeBaselineRegistrationRejectionV1

    graph = _build_graph()
    object.__setattr__(
        graph.finals[1].record,
        "previous_final_registration_id",
        graph.finals[2].final_registration_id,
    )
    result = _verify(graph)
    assert type(result) is mechanics.ThreeBaselineRegistrationRejectionV1
    assert result.partial_output_published is False

    graph = _build_graph()
    object.__setattr__(
        graph.finals[2].time_token,
        "previous_time_token_id",
        graph.finals[0].time_token.token_id,
    )
    assert type(_verify(graph)) is mechanics.ThreeBaselineRegistrationRejectionV1


@pytest.mark.parametrize(
    "component",
    (
        "builder_attestation",
        "qualifier_attestation",
        "custodian_attestation",
        "store_receipt",
        "time_token",
    ),
)
def test_every_signed_component_cross_baseline_splice_rejects(
    component: str,
) -> None:
    graph = _build_graph()
    object.__setattr__(
        graph.finals[1],
        component,
        getattr(graph.finals[0], component),
    )
    result = _verify(graph)
    assert type(result) is mechanics.ThreeBaselineRegistrationRejectionV1
    assert result.final_registration_candidate_ids == ()
    assert result.partial_output_published is False


@pytest.mark.parametrize("baseline_index", range(3))
def test_every_declared_time_logical_sequence_tamper_rejects(
    baseline_index: int,
) -> None:
    graph = _build_graph()
    object.__setattr__(
        graph.finals[baseline_index].time_token,
        "logical_sequence",
        (baseline_index + 1) % 3,
    )
    result = _verify(graph)
    assert type(result) is mechanics.ThreeBaselineRegistrationRejectionV1
    assert result.final_registration_candidate_ids == ()


def test_revision_manifest_store_and_r05_tamper_reject() -> None:
    mutations = (
        lambda graph: object.__setattr__(
            graph.finals[0].record, "upstream_revision_40hex", "f" * 39
        ),
        lambda graph: object.__setattr__(
            graph.finals[2].record, "repository_dependency_lock_sha256", None
        ),
        lambda graph: object.__setattr__(
            graph.finals[0].record.artifact_manifest, "complete", False
        ),
        lambda graph: object.__setattr__(
            graph.finals[1].store_receipt, "new_generation", 2
        ),
        lambda graph: object.__setattr__(
            graph.finals[2].record,
            "r05_predecessor_candidate_id",
            R05_PREFIX + "f" * 64,
        ),
    )
    for mutation in mutations:
        graph = _build_graph()
        mutation(graph)
        result = _verify(graph)
        assert type(result) is mechanics.ThreeBaselineRegistrationRejectionV1
        assert result.final_registration_candidate_ids == ()
        assert result.partial_output_published is False


def test_purpose_key_duplication_and_wrong_key_binding_reject() -> None:
    graph = _build_graph()
    object.__setattr__(graph.keys[1], "public_key", graph.keys[0].public_key)
    assert type(_verify(graph)) is mechanics.ThreeBaselineRegistrationRejectionV1

    graph = _build_graph()
    object.__setattr__(
        graph.finals[0].builder_attestation,
        "signer_key_id",
        graph.keys[1].key_id,
    )
    assert type(_verify(graph)) is mechanics.ThreeBaselineRegistrationRejectionV1


def test_wrong_input_and_backend_unavailable_fail_closed(monkeypatch) -> None:
    graph = _build_graph()
    wrong = mechanics.verify_three_baseline_registration_mechanics_v1(
        r05_predecessor=object(),
        candidate_keys=graph.keys,
        final_registrations=graph.finals,
    )
    assert wrong.reason is mechanics.ThreeBaselineRegistrationReasonV1.WRONG_INPUT_TYPE
    monkeypatch.setattr(mechanics, "_Ed25519PublicKey", None)
    unavailable = _verify(graph)
    assert unavailable.reason is (
        mechanics.ThreeBaselineRegistrationReasonV1.SIGNATURE_BACKEND_UNAVAILABLE
    )


def test_ineligible_projection_cannot_enter_old_execution_protocol() -> None:
    from hegel_machine.phase2b_freeze_v1 import frozen_phase2b_exact_freeze
    from hegel_machine.phase2b_protocol import (
        BaselineRegistration,
        ExecutionFreezeManifest,
        frozen_phase2b_protocol,
    )

    graph = _build_graph()
    result = _verify(graph)
    assert type(result) is mechanics.ThreeBaselineRegistrationMechanicsV1
    assert "BaselineRegistration" not in mechanics.__all__
    assert all(
        type(item) is mechanics.LegacyBaselineProjectionCandidateV1
        and not isinstance(item, BaselineRegistration)
        for item in result.legacy_projection_candidates
    )
    projected = mechanics.project_rich_candidate_to_ineligible_legacy_projection_v1(
        graph.finals[0].record
    )
    assert type(projected) is mechanics.LegacyBaselineProjectionCandidateV1
    assert projected.eligible_for_execution_freeze is False
    assert projected.r07_registration_verified is False

    protocol = frozen_phase2b_protocol()
    exact_freeze = frozen_phase2b_exact_freeze()
    with pytest.raises((AttributeError, KeyError, TypeError, ValueError)):
        ExecutionFreezeManifest(
            protocol_id=protocol.protocol_id,
            exact_freeze_id=exact_freeze.freeze_id,
            git_commit="a" * 40,
            recognizer_image_digest="sha256:" + "b" * 64,
            configuration_sha256="c" * 64,
            theory_version_id="test-only-non-authority",
            adapter_implementation_sha256="d" * 64,
            selector_implementation_sha256="e" * 64,
            verifier_registry_sha256="f" * 64,
            baseline_registrations=result.legacy_projection_candidates,
            isolation_profile_id=protocol.isolation_profile.profile_id,
        )


@pytest.mark.parametrize(
    "bad_path",
    ("", "/absolute.bin", "../escape.bin", "a/../escape.bin", "a//b", "a\\b"),
)
def test_artifact_path_traversal_and_nonportable_forms_reject(bad_path: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        mechanics.BaselineArtifactFileV1(
            role=mechanics.BaselineArtifactRoleV1.SOURCE_ARCHIVE,
            relative_path=bad_path,
            size_bytes=0,
            sha256="0" * 64,
        )


def test_duplicate_artifact_path_and_role_reject_before_manifest_acceptance() -> None:
    class ArtifactSubclass(mechanics.BaselineArtifactFileV1):
        pass

    subclass_item = ArtifactSubclass(
        mechanics.BaselineArtifactRoleV1.SOURCE_ARCHIVE,
        "subclass.bin",
        1,
        "0" * 64,
    )
    with pytest.raises(TypeError, match="exact type"):
        mechanics.artifact_file_manifest_root_v1((subclass_item,))

    by_path = (
        mechanics.BaselineArtifactFileV1(
            mechanics.BaselineArtifactRoleV1.SOURCE_ARCHIVE,
            "same.bin",
            1,
            "1" * 64,
        ),
        mechanics.BaselineArtifactFileV1(
            mechanics.BaselineArtifactRoleV1.LICENSE,
            "same.bin",
            1,
            "2" * 64,
        ),
    )
    root = mechanics.artifact_file_manifest_root_v1(by_path)
    with pytest.raises(ValueError, match="paths"):
        mechanics.BaselineArtifactFileManifestV1(
            by_path,
            2,
            2,
            True,
            root,
            FILE_MANIFEST_PREFIX + root,
        )

    by_role = (
        mechanics.BaselineArtifactFileV1(
            mechanics.BaselineArtifactRoleV1.LICENSE,
            "a.bin",
            1,
            "3" * 64,
        ),
        mechanics.BaselineArtifactFileV1(
            mechanics.BaselineArtifactRoleV1.LICENSE,
            "b.bin",
            1,
            "4" * 64,
        ),
    )
    root = mechanics.artifact_file_manifest_root_v1(by_role)
    with pytest.raises(ValueError, match="roles"):
        mechanics.BaselineArtifactFileManifestV1(
            by_role,
            2,
            2,
            True,
            root,
            FILE_MANIFEST_PREFIX + root,
        )


@pytest.mark.parametrize(
    "bad_size",
    (True, -1, 2**53, 1.5, float("inf"), float("nan")),
)
def test_artifact_size_exact_safe_integer_domain_rejects(bad_size: object) -> None:
    graph = _build_graph()
    object.__setattr__(
        graph.finals[0].record.artifact_manifest.files[0],
        "size_bytes",
        bad_size,
    )
    result = _verify(graph)
    assert type(result) is mechanics.ThreeBaselineRegistrationRejectionV1
    assert result.reason is (
        mechanics.ThreeBaselineRegistrationReasonV1.STRUCTURAL_CLOSURE_INVALID
    )
    assert result.partial_output_published is False


def test_safe_integer_boundaries_and_manifest_aggregate_overflow() -> None:
    zero = mechanics.BaselineArtifactFileV1(
        mechanics.BaselineArtifactRoleV1.SOURCE_ARCHIVE,
        "zero.bin",
        0,
        "5" * 64,
    )
    maximum = mechanics.BaselineArtifactFileV1(
        mechanics.BaselineArtifactRoleV1.LICENSE,
        "maximum.bin",
        2**53 - 1,
        "6" * 64,
    )
    assert zero.size_bytes == 0
    assert maximum.size_bytes == 2**53 - 1
    one = mechanics.BaselineArtifactFileV1(
        mechanics.BaselineArtifactRoleV1.SBOM,
        "one.bin",
        1,
        "7" * 64,
    )
    overflow_files = tuple(sorted((one, maximum), key=lambda item: item.relative_path))
    root = mechanics.artifact_file_manifest_root_v1(overflow_files)
    with pytest.raises(ValueError, match="total bytes"):
        mechanics.BaselineArtifactFileManifestV1(
            overflow_files,
            2,
            2**53 - 1,
            True,
            root,
            FILE_MANIFEST_PREFIX + root,
        )


def test_declared_time_safe_boundaries_and_current_epoch_are_reachable() -> None:
    for seconds, nanosecond in (
        (0, 0),
        (1_700_000_000, 123_456_789),
        (2**53 - 1, 999_999_999),
    ):
        token = _standalone_time_candidate(
            seconds=seconds,
            nanosecond=nanosecond,
        )
        assert (token.declared_unix_seconds, token.declared_nanosecond) == (
            seconds,
            nanosecond,
        )

    for seconds, nanosecond in (
        (-1, 0),
        (2**53, 0),
        (True, 0),
        (0, -1),
        (0, 1_000_000_000),
        (0, True),
    ):
        with pytest.raises((TypeError, ValueError)):
            _standalone_time_candidate(
                seconds=seconds,
                nanosecond=nanosecond,
            )


@pytest.mark.parametrize("relation", ("equal", "decrease"))
def test_declared_time_equality_and_decrease_reject_before_crypto(
    monkeypatch,
    relation: str,
) -> None:
    graph = _build_graph()
    previous = graph.finals[1].time_token
    seconds = previous.declared_unix_seconds
    nanosecond = previous.declared_nanosecond
    if relation == "decrease":
        seconds -= 1
    replacement = _rebuild_final_with_declared_time(
        graph.finals[2],
        seconds=seconds,
        nanosecond=nanosecond,
    )
    graph = KatGraph(
        graph.r05,
        graph.keys,
        (*graph.finals[:2], replacement),
        graph.preimages,
    )

    def crypto_must_not_run(*_args) -> None:
        pytest.fail("aggregate declared-time closure must reject before crypto")

    monkeypatch.setattr(mechanics, "_verify_ed25519", crypto_must_not_run)
    result = _verify(graph)
    assert type(result) is mechanics.ThreeBaselineRegistrationRejectionV1
    assert result.reason is (
        mechanics.ThreeBaselineRegistrationReasonV1.STRUCTURAL_CLOSURE_INVALID
    )


def test_manifest_root_store_object_content_splice_and_replay_reject() -> None:
    mutations = (
        lambda graph: object.__setattr__(
            graph.finals[0].record.artifact_manifest,
            "manifest_root_sha256",
            "f" * 64,
        ),
        lambda graph: object.__setattr__(
            graph.finals[1].store_receipt,
            "object_key",
            graph.finals[0].store_receipt.object_key,
        ),
        lambda graph: object.__setattr__(
            graph.finals[1].store_receipt,
            "stored_content_sha256",
            graph.finals[0].store_receipt.stored_content_sha256,
        ),
        lambda graph: object.__setattr__(
            graph.finals[2],
            "store_receipt",
            graph.finals[1].store_receipt,
        ),
    )
    for mutation in mutations:
        graph = _build_graph()
        mutation(graph)
        result = _verify(graph)
        assert type(result) is mechanics.ThreeBaselineRegistrationRejectionV1
        assert result.final_registration_candidate_ids == ()
        assert result.partial_output_published is False


def test_snapshot_walker_exact_types_bounds_cycles_and_true_bytes_copy(monkeypatch) -> None:
    original = bytes(bytearray(range(32)))
    frozen = mechanics._freeze_snapshot_node(
        original,
        depth=0,
        active_ids=set(),
        budget=mechanics._SnapshotBudgetV1(),
    )
    assert frozen[1] == original
    assert frozen[1] is not original

    graph = _build_graph()
    object.__setattr__(graph.keys[0], "public_key", bytes(range(32)) * 129)
    assert type(_verify(graph)) is mechanics.ThreeBaselineRegistrationRejectionV1

    graph = _build_graph()
    object.__setattr__(graph.r05, "candidate_scope", "x" * 4097)
    assert type(_verify(graph)) is mechanics.ThreeBaselineRegistrationRejectionV1

    graph = _build_graph()
    deeply_nested: object = b"leaf"
    for _ in range(40):
        deeply_nested = (deeply_nested,)
    object.__setattr__(graph.finals[0].builder_attestation, "signature", deeply_nested)
    assert type(_verify(graph)) is mechanics.ThreeBaselineRegistrationRejectionV1

    graph = _build_graph()
    object.__setattr__(
        graph.finals[0].record.artifact_manifest,
        "files",
        (graph.finals[0],),
    )
    assert type(_verify(graph)) is mechanics.ThreeBaselineRegistrationRejectionV1

    graph = _build_graph()
    oversized_tuple = tuple(graph.keys[0] for _ in range(8193))
    oversized = KatGraph(graph.r05, oversized_tuple, graph.finals, graph.preimages)
    assert type(_verify(oversized)) is mechanics.ThreeBaselineRegistrationRejectionV1

    graph = _build_graph()
    monkeypatch.setattr(mechanics, "_MAX_SNAPSHOT_NODES", 8)
    result = _verify(graph)
    assert type(result) is mechanics.ThreeBaselineRegistrationRejectionV1
    assert result.partial_output_published is False


def test_real_kat_snapshot_budget_measurements_have_fixed_headroom() -> None:
    graph = _build_graph()
    snapshot = mechanics._freeze_graph_inputs(graph.r05, graph.keys, graph.finals)
    assert snapshot.node_count == 543
    assert snapshot.utf8_bytes == 24_784
    assert snapshot.raw_bytes == 1_120
    assert snapshot.tuple_entry_count == 40
    assert snapshot.max_depth == 7
    assert len(snapshot.canonical_bytes) == 64_177
    assert snapshot.canonical_upper_bound == 294_688
    assert snapshot.node_count < mechanics._MAX_SNAPSHOT_NODES
    assert snapshot.utf8_bytes < mechanics._MAX_SNAPSHOT_TOTAL_UTF8_BYTES
    assert snapshot.raw_bytes < mechanics._MAX_SNAPSHOT_TOTAL_RAW_BYTES
    assert snapshot.tuple_entry_count < mechanics._MAX_SNAPSHOT_TOTAL_TUPLE_ENTRIES
    assert len(snapshot.canonical_bytes) < mechanics._MAX_SNAPSHOT_CANONICAL_BYTES


def test_cumulative_raw_and_tuple_budgets_count_repeated_references() -> None:
    raw_leaf = bytes(4_096)
    with pytest.raises(mechanics._SnapshotBudgetExceeded):
        mechanics._freeze_bounded_value((raw_leaf,) * 300)

    tuple_leaf = (None,) * 3_000
    with pytest.raises(mechanics._SnapshotBudgetExceeded):
        mechanics._freeze_bounded_value((tuple_leaf, tuple_leaf, tuple_leaf))


def test_repeated_reference_four_kib_leaf_amplification_rejects_before_crypto(
    monkeypatch,
) -> None:
    graph = _build_graph()
    leaf = graph.finals[0].record.artifact_manifest.files[0]
    object.__setattr__(leaf, "relative_path", "x" * 4_096)
    object.__setattr__(
        graph.finals[0].record.artifact_manifest,
        "files",
        (leaf,) * 300,
    )
    crypto_calls = 0

    def crypto_must_not_run(*_args) -> None:
        nonlocal crypto_calls
        crypto_calls += 1

    monkeypatch.setattr(mechanics, "_verify_ed25519", crypto_must_not_run)
    result = _verify(graph)
    assert crypto_calls == 0
    assert type(result) is mechanics.ThreeBaselineRegistrationRejectionV1
    assert result.reason is (
        mechanics.ThreeBaselineRegistrationReasonV1.RESOURCE_BUDGET_EXCEEDED
    )
    assert result.final_registration_candidate_ids == ()
    assert result.legacy_projection_candidates == ()
    assert result.verified_signature_count == 0
    assert result.partial_output_published is False


def test_nonrecursive_graph_shell_guard_rejects_before_snapshot(monkeypatch) -> None:
    graph = _build_graph()
    freeze_calls = 0

    def snapshot_must_not_run(*_args) -> None:
        nonlocal freeze_calls
        freeze_calls += 1
        pytest.fail("fixed graph shell must reject before snapshot traversal")

    monkeypatch.setattr(mechanics, "_freeze_graph_inputs", snapshot_must_not_run)
    bad_inputs = (
        (graph.keys + (graph.keys[0],), graph.finals),
        ((graph.keys[1], graph.keys[0], *graph.keys[2:]), graph.finals),
        (graph.keys, graph.finals + (graph.finals[0],)),
        (graph.keys, tuple(reversed(graph.finals))),
    )
    for keys, finals in bad_inputs:
        result = mechanics.verify_three_baseline_registration_mechanics_v1(
            r05_predecessor=graph.r05,
            candidate_keys=keys,
            final_registrations=finals,
        )
        assert type(result) is mechanics.ThreeBaselineRegistrationRejectionV1
        assert result.partial_output_published is False
    assert freeze_calls == 0


def test_bad_result_fixed_fields_reject_before_result_snapshot(monkeypatch) -> None:
    graph = _build_graph()
    result = _verify(graph)
    assert type(result) is mechanics.ThreeBaselineRegistrationMechanicsV1
    oversized = replace(
        result,
        final_registration_candidate_ids=(result.final_registration_candidate_ids[0],)
        * 10_000,
    )
    freeze_calls = 0

    def result_snapshot_must_not_run(*_args) -> None:
        nonlocal freeze_calls
        freeze_calls += 1
        pytest.fail("fixed result fields must reject before result traversal")

    monkeypatch.setattr(
        mechanics,
        "_freeze_result_for_comparison",
        result_snapshot_must_not_run,
    )
    assert not mechanics.verify_three_baseline_registration_mechanics_result_v1(
        r05_predecessor=graph.r05,
        candidate_keys=graph.keys,
        final_registrations=graph.finals,
        result=oversized,
    )
    assert freeze_calls == 0


@pytest.mark.parametrize("target", ("graph", "result"))
def test_canonical_upper_bound_rejects_before_projection_materialization(
    monkeypatch,
    target: str,
) -> None:
    graph = _build_graph()
    result = _verify(graph)
    projection_calls = 0

    def projection_must_not_run(*_args) -> None:
        nonlocal projection_calls
        projection_calls += 1
        pytest.fail("canonical upper bound must precede tagged projection")

    monkeypatch.setattr(mechanics, "_MAX_SNAPSHOT_CANONICAL_BYTES", 1_000)
    monkeypatch.setattr(mechanics, "_snapshot_json_value", projection_must_not_run)
    with pytest.raises(mechanics._SnapshotBudgetExceeded):
        if target == "graph":
            mechanics._freeze_graph_inputs(graph.r05, graph.keys, graph.finals)
        else:
            mechanics._freeze_result_for_comparison(result)
    assert projection_calls == 0


def test_memory_error_becomes_tiny_atomic_resource_rejection(monkeypatch) -> None:
    graph = _build_graph()

    def raise_memory_error(*_args) -> None:
        raise MemoryError

    monkeypatch.setattr(mechanics, "_freeze_graph_inputs", raise_memory_error)
    result = _verify(graph)
    assert type(result) is mechanics.ThreeBaselineRegistrationRejectionV1
    assert result.reason is (
        mechanics.ThreeBaselineRegistrationReasonV1.RESOURCE_BUDGET_EXCEEDED
    )
    assert result.final_registration_candidate_ids == ()
    assert result.legacy_projection_candidates == ()
    assert result.verified_signature_count == 0
    assert result.partial_output_published is False


def test_production_ast_has_no_side_effect_or_private_signing_surface() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_import_roots = {
        "asyncio",
        "boto3",
        "ftplib",
        "multiprocessing",
        "os",
        "paramiko",
        "pathlib",
        "pickle",
        "shutil",
        "socket",
        "sqlite3",
        "ssl",
        "subprocess",
        "tempfile",
        "threading",
        "requests",
        "urllib",
        "httpx",
        "docker",
        "secrets",
        "time",
    }
    imported_roots = set()
    forbidden_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {
                "open",
                "exec",
                "eval",
                "compile",
            }:
                forbidden_calls.append(node.func.id)
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "check_call",
                "check_output",
                "connect",
                "create_connection",
                "generate",
                "from_private_bytes",
                "open",
                "popen",
                "private_bytes",
                "read_bytes",
                "read_text",
                "request",
                "rmtree",
                "run",
                "sign",
                "system",
                "write_text",
                "write_bytes",
                "mkdir",
                "unlink",
            }:
                forbidden_calls.append(node.func.attr)
    assert imported_roots.isdisjoint(forbidden_import_roots)
    assert forbidden_calls == []
    assert "Ed25519PrivateKey" not in source
    assert "from .phase2b_protocol import" not in source
    assert "deepcopy" not in inspect.getsource(mechanics._freeze_snapshot_node)
    assert "sorted(" not in inspect.getsource(mechanics._freeze_snapshot_node)
    assert "campaign_id" in source and "attempt_id" in source
    assert DOC_PATH.is_file()
