"""Pure Q1 prestart-successor bundle derivation.

This module carries the already-authoritative Gates 1--10 forward, binds the
terminal boundary of the failed Q1-start-v3 campaign, and derives fresh
Gates 11--14 for the corrected resource envelope and actor implementation.
It has no filesystem, process, network, authorization, publication, ledger,
or Q1 execution facility.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import PurePosixPath
import re
from typing import Final, Mapping, NoReturn

from .strict_cbor_v1 import (
    canonical_cbor_decode,
    canonical_cbor_encode,
    content_hash,
)


SCHEMA_VERSION: Final = "hegel-phase3a-q1-prestart-successor-candidate/4"
ACTION_ID: Final = "phase3-q1-prestart-successor-admission-v4"
STATUS: Final = "Q1_PRESTART_SUCCESSOR_CANDIDATE_NOT_AUTHORIZED_NOT_CONSUMED"

BUNDLE_ROOT_DOMAIN: Final = "HEGEL/Q1/PRESTART_SUCCESSOR/BUNDLE/V4"
RECEIPT_ROOT_DOMAIN: Final = "HEGEL/Q1/PRESTART_SUCCESSOR/RECEIPT/V4"
RUN_GENESIS_ROOT_DOMAIN: Final = "HEGEL/Q1/PRESTART_SUCCESSOR/RUN_GENESIS/V4"
GATE_EVIDENCE_ROOT_DOMAIN: Final = "HEGEL/Q1/PRESTART_SUCCESSOR/GATE_EVIDENCE/V4"
RESOURCE_ENVELOPE_ROOT_DOMAIN: Final = (
    "HEGEL/Q1/PRESTART_SUCCESSOR/RESOURCE_ENVELOPE/V4"
)
ACTOR_SOURCE_MANIFEST_ROOT_DOMAIN: Final = (
    "HEGEL/Q1/PRESTART_SUCCESSOR/ACTOR_SOURCE_MANIFEST/V4"
)
ACTOR_SOURCE_CLOSURE_ROOT_DOMAIN: Final = (
    "HEGEL/Q1/PRESTART_SUCCESSOR/ACTOR_SOURCE_CLOSURE/V4"
)
ACTOR_IMPLEMENTATION_ROOT_DOMAIN: Final = (
    "HEGEL/Q1/PRESTART_SUCCESSOR/ACTOR_IMPLEMENTATION/V4"
)
MOUNT_MANIFEST_ROOT_DOMAIN: Final = (
    "HEGEL/Q1/PRESTART_SUCCESSOR/MOUNT_MANIFEST/V4"
)
INPUT_ABSENCE_ROOT_DOMAIN: Final = (
    "HEGEL/Q1/PRESTART_SUCCESSOR/INPUT_ABSENCE/V4"
)

RUN_GENESIS_SCHEMA_ID: Final = b"hegel-q1-prestart-successor-run-genesis/4"
RECEIPT_SCHEMA_ID: Final = b"hegel-q1-prestart-successor-receipt/4"
RESOURCE_ENVELOPE_SCHEMA_VERSION: Final = "hegel-q1-formal-resource-envelope/4"
V6_OBSERVATIONS_SCHEMA_VERSION: Final = (
    "hegel-q1-v6-scratch-observations-bound-for-successor/4"
)
MOUNT_MANIFEST_SCHEMA_VERSION: Final = "hegel-q1-successor-mount-manifest/4"
INPUT_ABSENCE_SCHEMA_VERSION: Final = "hegel-q1-successor-input-absence/4"

Q1_GATE_COUNT: Final = 14
Q1_GATE_TOTAL: Final = 20
Q1_GATE_MASK: Final = 0x003FFF
NULL_OUTPUT_SLOTS: Final = [None] * 8
ACTORS: Final = ("python", "rust", "host")
ACTOR_IMPLEMENTATION_REGISTRY: Final = {
    "python": (
        "hegel-python-q1-formal-archive-materializer-v1",
        "materialize-full-node6-formal-archive-v1",
    ),
    "rust": (
        "hegel-rust-q1-formal-archive-materializer-v1",
        "materialize-full-node6-formal-archive-v1",
    ),
    "host": (
        "hegel-host-q1-formal-archive-replay-v1",
        "replay-full-node6-formal-archive-v1",
    ),
}
ACTOR_ENTRYPOINT_RELATIVE_PATHS: Final = {
    "python": "tools/phase3_q1_full_node6_formal_materializer_entrypoint_v1.py",
    "rust": "rust/q1_full_capacity_preflight_engine/src/formal_materializer_main.rs",
    "host": "tools/phase3_q1_formal_host_replay_v1.py",
}
ACTOR_BOUND_ENVIRONMENT_ROOTS: Final = {
    "python": {
        "dependency_closure_root": "sha256:e11ed2fcb6cec54356798a26dffc66639d7350d8d41bbeb28547b8ef4d038bf8",
        "runtime_identity_root": "sha256:0e21719d45c91c48e3963bc216e4ba6ee2542c259f5d9f2dc0f8d10a100b379d",
        "offline_isolation_root": "sha256:af18eea093f5f40e8b5de10b341bb5607be17bd53e9b2ab8aad90ad8c23546b9",
    },
    "rust": {
        "dependency_closure_root": "sha256:67d0cb7284c2903c0d7f3546e736cc046242ab081da1538dfc05cae266d1db32",
        "runtime_identity_root": "sha256:f0c32a0b43d4bce269232fc41dc332e471d9bda8b276be88a16c717445f91597",
        "offline_isolation_root": "sha256:7bfa1658612508d71c22ffd2d7c83cf133ae6e466f66acd296d14382fbd85f09",
    },
    "host": {
        "dependency_closure_root": "sha256:b6fb63473be69f82941058b1b9903f47bd0c89cf75d23f0dc6cfc97d1e822598",
        "runtime_identity_root": "sha256:57a682f2907caddc415cb4f48b41ca3ebfc13b1c74a50462833cace978cf40ab",
        "offline_isolation_root": "sha256:750b5f86e2c29174df0795d094e6f02c3baf965561c20b7882c86037bf9e11aa",
    },
}
MAX_JSON_BYTES: Final = 16 * 1024 * 1024

BASE_ARTIFACT_RAW_SHA256: Final = (
    "sha256:23642c242a8e247a75fddaf2c94e0934958b1bbc26e51914aa2411734c4d7d50"
)
BASE_CAPSULE_DIGEST: Final = (
    "sha256:dc921cb1e3309d089fda588a3339cbec11d3e4e109ad4f688b6d260180736e04"
)
BASE_AUTHORIZATION_DIGEST: Final = (
    "sha256:4d9eba529e915d741c64f95fdcafdc0444db1f1840bc49dc27f81af33a595b5b"
)
BASE_RECEIPT_ROOT: Final = (
    "sha256:abe1704077b0af63d33c21d16dc926e10959771f38f4e4800e162a18cfb3c08a"
)
BASE_RUN_GENESIS_ROOT: Final = (
    "sha256:bb05d8ef1a5ccfe4ef8923c226a6ba7c14d62291a7e41e53ceb6d2a6ebeb825d"
)
BASE_SOURCE_COMMIT: Final = "a42fb84aeadc251cc595cae62bfc008a4d993d81"
BASE_SOURCE_TREE: Final = "f3a66048d5680de16f3bcb0258171d30afccea9f"
BASE_SOURCE_CLOSURE_ROOT: Final = (
    "sha256:21f571e9989fb4d9e33815adf21775cc02f22168625d33a0a6d9bbe5cb1f2a1a"
)
BASE_ACTOR_SOURCE_CLOSURE_ROOT: Final = (
    "sha256:e1307f496b37e392e452d6a1d9263859a72eee84e65131ffd663296298313cc0"
)
BASE_ACTOR_IMPLEMENTATION_ROOTS: Final = {
    "python": "sha256:ba64efbc30663e29f3e04413059cf13128287aba2e9dac4c3fb1d9ae37c10744",
    "rust": "sha256:e339f14550f06465c7daf8a515741099e56d6c4c410127984842b62067d10ed0",
    "host": "sha256:6de3d5e9ac7dfb86566614d0a1907517c8cfd00a7f28e83c3f7ab4ccbd7f2dd5",
}

Q1_START_V2_TERMINAL_RAW_SHA256: Final = (
    "sha256:05dd15cad8a6e17410fdb070728598053378ee8a59188a5655b951c0ee839560"
)
Q1_START_V2_TERMINAL_RECORD_DIGEST: Final = (
    "sha256:95aefaf81249deecccf70be47bfb2981d74f1d1bf43c6e092b583c3e02e63946"
)
Q1_START_V2_CAPSULE_DIGEST: Final = (
    "sha256:b32d15bfea02fc4838bfe05bb79a3fc93411e094b8517b3b557de9e76e23697b"
)
Q1_START_V2_CAPSULE_RAW_SHA256: Final = (
    "sha256:8a2580045ed57932bbcb7956302826a0dd3471884051bf67636d4709f65bab74"
)
Q1_START_V2_AUTHORIZATION_DIGEST: Final = (
    "sha256:7a7b82c52b939e5dec812c85687497bb1b9760832501e645c8ad62e61d87e375"
)
Q1_START_V2_AUTHORIZATION_RAW_SHA256: Final = (
    "sha256:86f4a1cef3381f4057d53c0de7aec999837485fef9c2f985f580b7784d91fc9d"
)
Q1_START_V2_CONSUMPTION_RECORD_DIGEST: Final = (
    "sha256:6ba931716028396aeca66a68d75effe2a08c91feede44a5afeeb241a9758b75e"
)
Q1_START_V2_CONSUMPTION_RAW_SHA256: Final = (
    "sha256:f732dde6d968aa3928391845defa0e451f29db52ea2836c5e619990ba3851f93"
)
Q1_START_V2_OUTCOME_RAW_SHA256: Final = (
    "sha256:a7a3e8957f8355d985df16bb31738bfa98abbed99fbee5eacbd699d8df07902b"
)

Q1_START_V3_TERMINAL_RAW_SHA256: Final = (
    "sha256:8785b57108e653b1e53444c9a1cb23facd95c6ecaf5b2447e93f2077758fd96e"
)
Q1_START_V3_TERMINAL_RECORD_DIGEST: Final = (
    "sha256:d48ff0bdcb296a28f7075f1bb09c846c5c6a5aa2153e5bf9da68bef1d9f5f7f7"
)
Q1_START_V3_CAPSULE_DIGEST: Final = (
    "sha256:3a2cf664970071c0529a200ef7e0b59a520dea8d3be0fe999c6208d836825dd9"
)
Q1_START_V3_CAPSULE_RAW_SHA256: Final = (
    "sha256:894920ee7252235655d72283fead637868b25a002ef513c5882b9025757d5c7e"
)
Q1_START_V3_AUTHORIZATION_DIGEST: Final = (
    "sha256:88867c73d6bad295a16648ea3cf6f34d0c1d42db11eed53ab62669d394f08765"
)
Q1_START_V3_AUTHORIZATION_RAW_SHA256: Final = (
    "sha256:e2523560b3321cd1f7d6d15c3de72a9ade503723bbacbfc7a560ba0b65ad2750"
)
Q1_START_V3_CONSUMPTION_RECORD_DIGEST: Final = (
    "sha256:f1dc4c6bfdd38447923ae4c84c9b3063441bc41682606b293e0232f3f2c23622"
)
Q1_START_V3_CONSUMPTION_RAW_SHA256: Final = (
    "sha256:e15d7acecdc0441e991088b5ac07ad42de8852b6d323a90d848f0a69c08ab1c5"
)
Q1_START_V3_OUTCOME_RAW_SHA256: Final = (
    "sha256:d53c41877d920944f58f5c57ea9c45d6529b4d56d99ee7e68fe5b508b8c642ad"
)

FORMAL_OUTPUT_LAYOUT_ROOT: Final = (
    "sha256:f481fbd24eabbf2669ff653972c7c811b451e5874a31d2445b4528bd1c19cd82"
)

GATE_NAMES: Final = (
    "POST_SHRINK6_NORMATIVE_AND_Q1_FREEZE_BYTES_BOUND",
    "Q0_14_OF_14_RECEIPT_AND_SOURCE_REPLAYED",
    "V16_FULL_DSL_STRICT_ROOT_FAMILY_REQUALIFIED",
    "PRODUCTION_UNIVERSE_ROWS_AND_ADAPTERS_DUAL_QUALIFIED",
    "Q1_BEHAVIOR_BOTTOM_AND_COLLISION_CODEC_QUALIFIED",
    "Q1_CONSTRUCTION_SIGNATURE_AND_MULTIPLICITY_POLICY_QUALIFIED",
    "Q1_PARETO_DOMINANCE_AND_CLASS_MDL_QUALIFIED",
    "FULL_OPERATOR_CONGRUENCE_AND_REWRITE_PROFILES_QUALIFIED",
    "STRUCTURAL_INDUCTION_AND_FIXED_POINT_CONTRACT_QUALIFIED",
    "Q1_ARCHIVE_WIRE_ROOT_DAG_RESOURCE_PROJECTION_AND_GOLDENS_PASS",
    "TARGET_BLIND_CAPACITY_PREFLIGHT_AND_CORRECTED_RESOURCE_ENVELOPE_BOUND",
    "SUCCESSOR_ACTOR_IMPLEMENTATIONS_AND_SOURCE_MANIFESTS_BOUND",
    "SUCCESSOR_TARGET_TRUTH_SPLIT_ROLE_INPUTS_ABSENT",
    "SUCCESSOR_RUN_GENESIS_NON_NULL_AND_EIGHT_OUTPUT_SLOTS_NULL",
)

# These ten compact rows are the exact authoritative projection carried from
# the immutable 14/20 artifact.  Their subject bytes and evidence roots are
# already fixed by BASE_ARTIFACT_RAW_SHA256; retaining them avoids pretending
# that the successor re-ran Gates 1--10.
LEGACY_GATE_ROWS_1_TO_10: Final = (
    (1, GATE_NAMES[0], "sha256:8e49d42b8b1b0ad340000016d15816aba1998c124ff082fe0294dd3a0fe928fd", "8358201df8d3ff3ede2cbead98e7901a3e82b91c460ad1d5eb0d1af78938e7b2d23b955820e106aa9ccab264aae73df8c3fc69de957bff9ec9bb630155cf765e348c84a4f35820d8a8b019e8e1af84f80be86e492d68bf72f138f0cb9e0e686c9c0cf90987e459"),
    (2, GATE_NAMES[1], "sha256:112d92900aaed8d1ada2d29c8466b09febc13ded1cc39cdefa3963886b0b747a", "8358201d8d530cbccf133af97da900164b89c9385db21d873bfff039ee7ce07c45f28d5820ee198614e94cf425202f9c667836fc6ad61fda02c9439a689eb90012c5798ad254ae201905526747980ff8e46e70f9ec96de93d3eb"),
    (3, GATE_NAMES[2], "sha256:ff88e392353696fbfb7e48746b2f09646957e974428e18f9d71a47232a3f5577", "835820ee198614e94cf425202f9c667836fc6ad61fda02c9439a689eb90012c5798ad258203fefacd3db59294f2b6d44a5d0b813e73af3ec84742a24ab846bbdacae6c1f1b5820e3b3df3e81b7632c7c713ef5ec84913f990ad8232a25b851f20c46ac7416bfcb"),
    (4, GATE_NAMES[3], "sha256:51d82174f07ee39bac3395d8913f398179c3ad0ad9502b75c41e66510c165e47", "845820ee198614e94cf425202f9c667836fc6ad61fda02c9439a689eb90012c5798ad25820b7e6eed1174ceee1944bd540cbb0ace4c5c7fc0dffea79dd956a51f7a2410a0558201a46f9967ad48df6ec1d9be609de701413ce86171fb0f92495a982bef0f40ff558201a1bb3ac14311ee84b073521b9b96cd4d4ed404de08ccc5ee1174e494de671ec"),
    (5, GATE_NAMES[4], "sha256:bff93fc0f95ce655ff1fa264953745054b748eabb857c679048cfe1da93d32aa", "845820ee198614e94cf425202f9c667836fc6ad61fda02c9439a689eb90012c5798ad2582006d219b403c0302862badc020edc6409f44f78fe26e8d30a23d178dae4410c5258200afef42d4544fddbe07d0b8ac5c3b4f86bfcff0e7d51131b6acda12a8fda18ea5820ce8f77867675413d58b5e446f87a16c661e2cfcf27c27aef545e23ae63352671"),
    (6, GATE_NAMES[5], "sha256:4d8ca6d124e737f157f72157e33ae2a968aa715582aacd4aeea9ba52296f19c5", "835820ee198614e94cf425202f9c667836fc6ad61fda02c9439a689eb90012c5798ad2582006d219b403c0302862badc020edc6409f44f78fe26e8d30a23d178dae4410c525820dd7e13b445371e53e61fcd5233fc771079546eaaec4a9f98ea44f6569ccb4250"),
    (7, GATE_NAMES[6], "sha256:b9d7fa58f6d2acfebdc03faad76d88c8eaee0227b9b4f2324cd859ca649e50f5", "835820ee198614e94cf425202f9c667836fc6ad61fda02c9439a689eb90012c5798ad2582006d219b403c0302862badc020edc6409f44f78fe26e8d30a23d178dae4410c525820231d13d18b0b7cf0553ab9a7a42c71de2af480dab1c2aef27eb39246464a64f4"),
    (8, GATE_NAMES[7], "sha256:5d2ce1e822279c2bd17e1f5a4771fd5f8e4b100147e2073c06b17965c80c9bce", "835820ee198614e94cf425202f9c667836fc6ad61fda02c9439a689eb90012c5798ad2582006d219b403c0302862badc020edc6409f44f78fe26e8d30a23d178dae4410c5258209fd7e4b5126a749baf111f14ed8adb97b8b6c2323ec6d00367428ac0e53c3bc0"),
    (9, GATE_NAMES[8], "sha256:1487d1001af21ca6dc76f4afa3da1cc84bfe49d5e153248113d35c493fcb7b00", "835820ee198614e94cf425202f9c667836fc6ad61fda02c9439a689eb90012c5798ad2582073c319b54c53af75c3dbe01d64cd74aeb903c7d2edf1491cea7a8b7eb38194f458203808c771a65a8dad53fa711e7605d63824b8b1dfc119d90eba42ae6786bf5118"),
    (10, GATE_NAMES[9], "sha256:3b0090098a0f25c76668d242fa61619394c56ed0ec4dc6424f595a8a0c76a059", "8358207bbbe13efbc43a128c7310fa3a3a32c0545fe0d55acf03d6212742ebdef3c700582006d219b403c0302862badc020edc6409f44f78fe26e8d30a23d178dae4410c525820e1307f496b37e392e452d6a1d9263859a72eee84e65131ffd663296298313cc0"),
)

EXPECTED_V6_SCRATCH_OBSERVATIONS: Final = {
    "schema_version": V6_OBSERVATIONS_SCHEMA_VERSION,
    "v6_capsule_digest": "sha256:a22cc6660e01c56b5670cca770d29d06c40bd06851d18868cbff07b61e918a29",
    "v6_artifact_raw_sha256": "sha256:73c319b54c53af75c3dbe01d64cd74aeb903c7d2edf1491cea7a8b7eb38194f4",
    "v6_python_stdout_raw_sha256": "sha256:9c71a2ba7e2bc41e5cc0390c8f53eab0dbb000ea916ccb5be4a75cc56bf85442",
    "v6_rust_stdout_raw_sha256": "sha256:c5e641f18ab42982e92035b89131b81c0c61d73dcd139fad015bc75ef21cb779",
    "projected_peak_formal_scratch_bytes": 1_646_592,
    "python_actual_preflight_peak_scratch_bytes": 40_452_166,
    "rust_actual_preflight_peak_scratch_bytes": 3_194_598,
    "selected_worst_peak_scratch_bytes": 40_452_166,
}

EXPECTED_CORRECTED_RESOURCE_ENVELOPE: Final = {
    "schema_version": RESOURCE_ENVELOPE_SCHEMA_VERSION,
    "scratch_guard_formula": (
        "pow2ceil(max(1,2*max(projected_peak_formal_scratch_bytes,"
        "python_actual_preflight_peak_scratch_bytes,"
        "rust_actual_preflight_peak_scratch_bytes)))"
    ),
    "per_signature_formal_count_guards": [
        {
            "input_signature_id": 1,
            "maximum_raw_operator_applications": 32_768,
            "maximum_behavior_classes": 1_024,
            "maximum_visible_frontier_points": 4_096,
            "maximum_visible_frontier_points_per_class": 32,
            "maximum_continuation_bank_points": 8_192,
            "maximum_continuation_bank_points_per_class": 4_096,
            "maximum_work_queue_points": 16_384,
            "maximum_saturation_rounds": 7,
        },
        {
            "input_signature_id": 2,
            "maximum_raw_operator_applications": 32_768,
            "maximum_behavior_classes": 256,
            "maximum_visible_frontier_points": 1_024,
            "maximum_visible_frontier_points_per_class": 256,
            "maximum_continuation_bank_points": 8_192,
            "maximum_continuation_bank_points_per_class": 4_096,
            "maximum_work_queue_points": 16_384,
            "maximum_saturation_rounds": 7,
        },
    ],
    "endpoint": {
        "output_bytes": 16_777_216,
        "scratch_bytes": 134_217_728,
        "memory_bytes": 2_147_483_648,
        "wall_time_seconds": 3_600,
    },
    "trusted_host_replay": {
        "output_bytes": 2_097_152,
        "scratch_bytes": 1,
        "memory_bytes": 67_108_864,
        "wall_time_seconds": 3_600,
    },
}

_TAGGED_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")


class Q1PrestartSuccessorCandidateError(RuntimeError):
    """Stable rejection from the pure successor-candidate derivation."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Q1PrestartSuccessorCandidateError(code, detail)


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail("REJECT_Q1_SUCCESSOR_JSON", f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    _fail("REJECT_Q1_SUCCESSOR_JSON", f"non-finite JSON token {value!r}")


def _finite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _fail("REJECT_Q1_SUCCESSOR_JSON", "non-finite JSON number")
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            _fail("REJECT_Q1_SUCCESSOR_JSON", "JSON keys must be strings")
        for child in value.values():
            _finite(child)
    elif type(value) in (list, tuple):
        for child in value:
            _finite(child)


def canonical_json_bytes_v4(value: object) -> bytes:
    _finite(value)
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        _fail("REJECT_Q1_SUCCESSOR_JSON", str(error))


def _decode_canonical_json(payload: bytes, label: str) -> dict[str, object]:
    if type(payload) is not bytes or not payload or len(payload) > MAX_JSON_BYTES:
        _fail("REJECT_Q1_SUCCESSOR_INPUT", f"{label} byte bound differs")
    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except Q1PrestartSuccessorCandidateError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail("REJECT_Q1_SUCCESSOR_JSON", f"{label}: {error}")
    if type(value) is not dict or canonical_json_bytes_v4(value) != payload:
        _fail("REJECT_Q1_SUCCESSOR_JSON", f"{label} is not canonical JSON")
    return value


def _json_wire(value: object) -> object:
    """Convert canonical JSON values into the map-free strict-CBOR profile."""

    if value is None or value is False or value is True or type(value) is int:
        return value
    if type(value) is str:
        try:
            return value.encode("ascii")
        except UnicodeEncodeError:
            _fail("REJECT_Q1_SUCCESSOR_JSON", "wire text must be ASCII")
    if type(value) in (list, tuple):
        return tuple(_json_wire(child) for child in value)
    if type(value) is dict:
        return tuple(
            (key.encode("ascii"), _json_wire(value[key])) for key in sorted(value)
        )
    _fail("REJECT_Q1_SUCCESSOR_JSON", f"unsupported wire type {type(value).__name__}")


def _tagged(value: object, label: str) -> str:
    if type(value) is not str or _TAGGED_SHA256.fullmatch(value) is None:
        _fail("REJECT_Q1_SUCCESSOR_DIGEST", f"{label} must be tagged SHA-256")
    return value


def _hex40(value: object, label: str) -> str:
    if type(value) is not str or _HEX40.fullmatch(value) is None:
        _fail("REJECT_Q1_SUCCESSOR_SOURCE", f"{label} must be 40 lowercase hex")
    return value


def _root_bytes(value: object, label: str) -> bytes:
    return bytes.fromhex(_tagged(value, label)[7:])


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        _fail("REJECT_Q1_SUCCESSOR_SHAPE", f"{label} must be a mapping")
    result = deepcopy(dict(value))
    if any(type(key) is not str for key in result):
        _fail("REJECT_Q1_SUCCESSOR_SHAPE", f"{label} keys must be strings")
    return result


def _exact_keys(value: dict[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        _fail("REJECT_Q1_SUCCESSOR_SHAPE", f"{label} key closure differs")


def _reject_predecessor_runtime_paths(value: object) -> None:
    if type(value) is str:
        normalized = value.replace("\\", "/")
        for generation in ("v2", "v3"):
            if f"q1-start-{generation}-run/" in normalized and any(
                marker in normalized
                for marker in (
                    "python-output",
                    "python-scratch",
                    "rust-output",
                    "rust-scratch",
                    "host-output",
                    "host-scratch",
                )
            ):
                _fail(
                    f"REJECT_Q1_SUCCESSOR_{generation.upper()}_WORK_PRODUCT",
                    f"Q1-start-{generation} output/scratch paths cannot enter the successor",
                )
    elif type(value) is dict:
        for child in value.values():
            _reject_predecessor_runtime_paths(child)
    elif type(value) in (list, tuple):
        for child in value:
            _reject_predecessor_runtime_paths(child)


def _raw_sha(payload: bytes) -> str:
    return "sha256:" + sha256(payload).hexdigest()


def _legacy_gate_dict(row: tuple[int, str, str, str]) -> dict[str, object]:
    return {
        "gate_id": row[0],
        "gate_name": row[1],
        "evidence_root": row[2],
        "subjects_cbor_hex": row[3],
    }


def _validate_base_artifact(payload: bytes) -> tuple[list[dict[str, object]], dict[str, object]]:
    if _raw_sha(payload) != BASE_ARTIFACT_RAW_SHA256:
        _fail("REJECT_Q1_SUCCESSOR_BASE", "authoritative prestart artifact differs")
    value = _decode_canonical_json(payload, "authoritative prestart artifact")
    if (
        value.get("schema_version")
        != "hegel-phase3a-q1-prestart-authoritative-amendment-artifact/1"
        or value.get("status") != "Q1_PRESTART_14_OF_20_APPLIED_Q1_NOT_RUN"
        or value.get("capsule_digest") != BASE_CAPSULE_DIGEST
        or value.get("authorization_digest") != BASE_AUTHORIZATION_DIGEST
        or value.get("authorization_consumption_count") != 1
        or type(value.get("authorization_consumption_count")) is not int
        or value.get("campaign_execution_count") != 1
        or type(value.get("campaign_execution_count")) is not int
        or value.get("prestart_admission_receipt_root") != BASE_RECEIPT_ROOT
        or value.get("run_genesis_root") != BASE_RUN_GENESIS_ROOT
        or value.get("source_commit") != BASE_SOURCE_COMMIT
        or value.get("source_tree") != BASE_SOURCE_TREE
        or value.get("q1_state") != "NOT_RUN"
        or value.get("q1_gate_count") != Q1_GATE_COUNT
        or type(value.get("q1_gate_count")) is not int
        or value.get("q1_gate_mask") != Q1_GATE_MASK
        or type(value.get("q1_gate_mask")) is not int
        or value.get("q1_output_slots") != NULL_OUTPUT_SLOTS
        or value.get("q1_execution_started") is not False
        or value.get("popen_count") != 0
        or type(value.get("popen_count")) is not int
    ):
        _fail("REJECT_Q1_SUCCESSOR_BASE", "authoritative prestart boundary differs")
    rows = value.get("gate_evidence_rows")
    expected = [_legacy_gate_dict(row) for row in LEGACY_GATE_ROWS_1_TO_10]
    if type(rows) is not list or rows[:10] != expected:
        _fail("REJECT_Q1_SUCCESSOR_BASE", "authoritative Gates 1 through 10 differ")
    return deepcopy(expected), value


def _validate_q1_start_v3_terminal(payload: bytes) -> dict[str, object]:
    if _raw_sha(payload) != Q1_START_V3_TERMINAL_RAW_SHA256:
        _fail("REJECT_Q1_SUCCESSOR_V3_TERMINAL", "Q1-start-v3 terminal bytes differ")
    value = _decode_canonical_json(payload, "Q1-start-v3 terminal")
    outcome = value.get("outcome")
    if (
        type(outcome) is not dict
        or value.get("schema_version") != "hegel-phase3a-q1-start-terminal/3"
        or value.get("status") != "Q1_START_V3_INCONCLUSIVE_TERMINAL_NO_RESUME"
        or value.get("record_digest") != Q1_START_V3_TERMINAL_RECORD_DIGEST
        or value.get("capsule_digest") != Q1_START_V3_CAPSULE_DIGEST
        or value.get("authorization_digest") != Q1_START_V3_AUTHORIZATION_DIGEST
        or value.get("consumption_record_digest")
        != Q1_START_V3_CONSUMPTION_RECORD_DIGEST
        or value.get("authorization_consumption_count") != 1
        or value.get("logical_campaign_count") != 1
        or outcome.get("status") != "Q1_START_V3_INCONCLUSIVE_TERMINAL_NO_RESUME"
        or outcome.get("failure_code") != "REJECT_Q1_START_SUCCESSOR_BUNDLE"
        or outcome.get("failure_detail")
        != "validator import failed: AttributeError: module 'importlib' has no attribute 'util'"
        or outcome.get("actor_start_count_by_actor")
        != {"host": 0, "python": 0, "rust": 0}
        or outcome.get("q1_state") != "NOT_RUN"
        or outcome.get("q1_gate_count") != Q1_GATE_COUNT
        or outcome.get("q1_gate_mask") != Q1_GATE_MASK
        or outcome.get("q1_output_slots") != NULL_OUTPUT_SLOTS
        or outcome.get("network_accessed") is not False
        or outcome.get("target_truth_accessed") is not False
        or outcome.get("split_accessed") is not False
        or outcome.get("role_evaluation_performed") is not False
    ):
        _fail("REJECT_Q1_SUCCESSOR_V3_TERMINAL", "Q1-start-v3 terminal boundary differs")
    return value


def _base_chain() -> dict[str, object]:
    return {
        "base_prestart_artifact_raw_sha256": BASE_ARTIFACT_RAW_SHA256,
        "base_prestart_capsule_digest": BASE_CAPSULE_DIGEST,
        "base_prestart_authorization_digest": BASE_AUTHORIZATION_DIGEST,
        "base_prestart_authorization_consumption_count": 1,
        "base_prestart_campaign_execution_count": 1,
        "base_prestart_receipt_root": BASE_RECEIPT_ROOT,
        "base_run_genesis_root": BASE_RUN_GENESIS_ROOT,
        "base_source_commit": BASE_SOURCE_COMMIT,
        "base_source_tree": BASE_SOURCE_TREE,
        "base_source_closure_root": BASE_SOURCE_CLOSURE_ROOT,
        "base_actor_source_closure_root": BASE_ACTOR_SOURCE_CLOSURE_ROOT,
        "base_actor_implementation_roots": deepcopy(BASE_ACTOR_IMPLEMENTATION_ROOTS),
        "q1_start_v2_terminal_raw_sha256": Q1_START_V2_TERMINAL_RAW_SHA256,
        "q1_start_v2_terminal_record_digest": Q1_START_V2_TERMINAL_RECORD_DIGEST,
        "q1_start_v2_capsule_raw_sha256": Q1_START_V2_CAPSULE_RAW_SHA256,
        "q1_start_v2_capsule_digest": Q1_START_V2_CAPSULE_DIGEST,
        "q1_start_v2_authorization_raw_sha256": Q1_START_V2_AUTHORIZATION_RAW_SHA256,
        "q1_start_v2_authorization_digest": Q1_START_V2_AUTHORIZATION_DIGEST,
        "q1_start_v2_consumption_raw_sha256": Q1_START_V2_CONSUMPTION_RAW_SHA256,
        "q1_start_v2_consumption_record_digest": Q1_START_V2_CONSUMPTION_RECORD_DIGEST,
        "q1_start_v2_outcome_raw_sha256": Q1_START_V2_OUTCOME_RAW_SHA256,
        "q1_start_v2_actor_start_count_by_actor": {"host": 0, "python": 0, "rust": 0},
        "q1_start_v2_completed_prefix": [],
        "q1_start_v2_actor_actions_conservatively_counted": False,
        "q1_start_v2_resume_allowed": False,
        "q1_start_v3_terminal_raw_sha256": Q1_START_V3_TERMINAL_RAW_SHA256,
        "q1_start_v3_terminal_record_digest": Q1_START_V3_TERMINAL_RECORD_DIGEST,
        "q1_start_v3_capsule_raw_sha256": Q1_START_V3_CAPSULE_RAW_SHA256,
        "q1_start_v3_capsule_digest": Q1_START_V3_CAPSULE_DIGEST,
        "q1_start_v3_authorization_raw_sha256": Q1_START_V3_AUTHORIZATION_RAW_SHA256,
        "q1_start_v3_authorization_digest": Q1_START_V3_AUTHORIZATION_DIGEST,
        "q1_start_v3_consumption_raw_sha256": Q1_START_V3_CONSUMPTION_RAW_SHA256,
        "q1_start_v3_consumption_record_digest": Q1_START_V3_CONSUMPTION_RECORD_DIGEST,
        "q1_start_v3_outcome_raw_sha256": Q1_START_V3_OUTCOME_RAW_SHA256,
        "q1_start_v3_actor_start_count_by_actor": {"host": 0, "python": 0, "rust": 0},
        "q1_start_v3_completed_prefix": [],
        "q1_start_v3_actor_actions_conservatively_counted": False,
        "q1_start_v3_resume_allowed": False,
    }


def _validate_source_identity(value: object) -> dict[str, object]:
    result = _mapping(value, "source identity")
    _exact_keys(
        result,
        {"source_commit", "source_tree", "source_closure_root"},
        "source identity",
    )
    _hex40(result["source_commit"], "source commit")
    _hex40(result["source_tree"], "source tree")
    _tagged(result["source_closure_root"], "source closure root")
    if (
        result["source_commit"] == BASE_SOURCE_COMMIT
        or result["source_tree"] == BASE_SOURCE_TREE
        or result["source_closure_root"] == BASE_SOURCE_CLOSURE_ROOT
    ):
        _fail("REJECT_Q1_SUCCESSOR_SOURCE", "successor source identity is not fresh")
    return result


def actor_source_manifest_root_v4(actor_id: str, rows: object) -> str:
    if actor_id not in ACTORS or type(rows) is not list or not rows:
        _fail("REJECT_Q1_SUCCESSOR_ACTOR", "actor manifest registry differs")
    checked: list[tuple[bytes, int, bytes]] = []
    paths: list[str] = []
    for raw in rows:
        row = _mapping(raw, f"{actor_id} source row")
        _exact_keys(
            row,
            {"relative_path", "byte_count", "raw_sha256"},
            f"{actor_id} source row",
        )
        path = row["relative_path"]
        if (
            type(path) is not str
            or not path
            or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
        ):
            _fail("REJECT_Q1_SUCCESSOR_ACTOR", "actor source path differs")
        count = row["byte_count"]
        if type(count) is not int or count < 0:
            _fail("REJECT_Q1_SUCCESSOR_ACTOR", "actor source byte count differs")
        paths.append(path)
        checked.append(
            (path.encode("ascii"), count, _root_bytes(row["raw_sha256"], "source row"))
        )
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        _fail("REJECT_Q1_SUCCESSOR_ACTOR", "actor source rows must be unique and sorted")
    return "sha256:" + content_hash(
        ACTOR_SOURCE_MANIFEST_ROOT_DOMAIN,
        (2, actor_id.encode("ascii"), tuple(checked)),
    ).hex()


def _validate_actor_source_manifests(
    value: object,
) -> tuple[dict[str, object], dict[str, object]]:
    manifests = _mapping(value, "actor source manifests")
    _exact_keys(manifests, set(ACTORS), "actor source manifests")
    roots: dict[str, object] = {}
    checked: dict[str, object] = {}
    for actor in ACTORS:
        manifest = _mapping(manifests[actor], f"{actor} source manifest")
        _exact_keys(
            manifest,
            {"source_manifest_root", "rows"},
            f"{actor} source manifest",
        )
        computed = actor_source_manifest_root_v4(actor, manifest["rows"])
        if manifest["source_manifest_root"] != computed:
            _fail("REJECT_Q1_SUCCESSOR_ACTOR", f"{actor} source manifest root differs")
        roots[actor] = computed
        checked[actor] = manifest
    return checked, roots


def actor_implementation_root_v4(
    actor_id: str,
    descriptor: object,
    source_manifest_root: object,
) -> str:
    """Mechanically bind one actor implementation to its verified manifest."""

    if actor_id not in ACTORS:
        _fail("REJECT_Q1_SUCCESSOR_ACTOR", "actor implementation registry differs")
    value = _mapping(descriptor, f"{actor_id} implementation descriptor")
    _exact_keys(
        value,
        {
            "implementation_id",
            "action_id",
            "entrypoint_relative_path",
            "entrypoint_raw_sha256",
            "dependency_closure_root",
            "runtime_identity_root",
            "offline_isolation_root",
        },
        f"{actor_id} implementation descriptor",
    )
    expected_implementation, expected_action = ACTOR_IMPLEMENTATION_REGISTRY[actor_id]
    if (
        value["implementation_id"] != expected_implementation
        or value["action_id"] != expected_action
        or value["entrypoint_relative_path"]
        != ACTOR_ENTRYPOINT_RELATIVE_PATHS[actor_id]
    ):
        _fail("REJECT_Q1_SUCCESSOR_ACTOR", f"{actor_id} implementation identity differs")
    for field in (
        "entrypoint_raw_sha256",
        "dependency_closure_root",
        "runtime_identity_root",
        "offline_isolation_root",
    ):
        _tagged(value[field], f"{actor_id} {field}")
    if any(
        value[field] != expected
        for field, expected in ACTOR_BOUND_ENVIRONMENT_ROOTS[actor_id].items()
    ):
        _fail(
            "REJECT_Q1_SUCCESSOR_ACTOR",
            f"{actor_id} bound dependency/runtime/isolation roots differ",
        )
    return "sha256:" + content_hash(
        ACTOR_IMPLEMENTATION_ROOT_DOMAIN,
        (
            2,
            actor_id.encode("ascii"),
            expected_implementation.encode("ascii"),
            expected_action.encode("ascii"),
            _root_bytes(source_manifest_root, f"{actor_id} source manifest"),
            ACTOR_ENTRYPOINT_RELATIVE_PATHS[actor_id].encode("ascii"),
            _root_bytes(value["entrypoint_raw_sha256"], f"{actor_id} entrypoint"),
            _root_bytes(value["dependency_closure_root"], f"{actor_id} dependency"),
            _root_bytes(value["runtime_identity_root"], f"{actor_id} runtime"),
            _root_bytes(value["offline_isolation_root"], f"{actor_id} isolation"),
            _root_bytes(FORMAL_OUTPUT_LAYOUT_ROOT, "formal output layout"),
            (),
            False,
            False,
            False,
        ),
    ).hex()


def _validate_actor_implementation_descriptors(
    value: object,
    manifest_roots: dict[str, object],
    manifests: dict[str, object] | None,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    descriptors = _mapping(value, "actor implementation descriptors")
    _exact_keys(descriptors, set(ACTORS), "actor implementation descriptors")
    implementation_roots: dict[str, object] = {}
    offline_roots: dict[str, object] = {}
    checked: dict[str, object] = {}
    for actor in ACTORS:
        descriptor = _mapping(
            descriptors[actor], f"{actor} implementation descriptor"
        )
        root = actor_implementation_root_v4(
            actor, descriptor, manifest_roots[actor]
        )
        if root == BASE_ACTOR_IMPLEMENTATION_ROOTS[actor]:
            _fail(
                "REJECT_Q1_SUCCESSOR_ACTOR",
                f"{actor} derived implementation root is not fresh",
            )
        if manifests is not None:
            matching_rows = [
                row
                for row in manifests[actor]["rows"]  # type: ignore[index]
                if row["relative_path"] == ACTOR_ENTRYPOINT_RELATIVE_PATHS[actor]
            ]
            if (
                len(matching_rows) != 1
                or matching_rows[0]["raw_sha256"]
                != descriptor["entrypoint_raw_sha256"]
            ):
                _fail(
                    "REJECT_Q1_SUCCESSOR_ACTOR",
                    f"{actor} fixed entrypoint row/hash differs from its source manifest",
                )
        implementation_roots[actor] = root
        offline_roots[actor] = descriptor["offline_isolation_root"]
        checked[actor] = descriptor
    return checked, implementation_roots, offline_roots


def _validate_actor_roots(
    value: object,
    label: str,
    *,
    reject_base_implementation: bool,
) -> dict[str, object]:
    roots = _mapping(value, label)
    _exact_keys(roots, set(ACTORS), label)
    for actor in ACTORS:
        _tagged(roots[actor], f"{label} {actor}")
        if reject_base_implementation and roots[actor] == BASE_ACTOR_IMPLEMENTATION_ROOTS[actor]:
            _fail("REJECT_Q1_SUCCESSOR_ACTOR", f"{actor} implementation root is not fresh")
    return roots


def _actor_source_closure_root(manifest_roots: dict[str, object]) -> str:
    root = "sha256:" + content_hash(
        ACTOR_SOURCE_CLOSURE_ROOT_DOMAIN,
        tuple((actor.encode("ascii"), _root_bytes(manifest_roots[actor], actor)) for actor in ACTORS),
    ).hex()
    if root == BASE_ACTOR_SOURCE_CLOSURE_ROOT:
        _fail("REJECT_Q1_SUCCESSOR_ACTOR", "actor source closure is not fresh")
    return root


def _validate_mount_manifest(
    value: object,
    manifests: dict[str, object],
) -> tuple[dict[str, object], str]:
    mount = _mapping(value, "mount manifest")
    _exact_keys(mount, {"schema_version", "rows"}, "mount manifest")
    rows = mount["rows"]
    if mount["schema_version"] != MOUNT_MANIFEST_SCHEMA_VERSION or type(rows) is not list or len(rows) != 3:
        _fail("REJECT_Q1_SUCCESSOR_MOUNT", "mount manifest shape differs")
    for actor, raw in zip(ACTORS, rows, strict=True):
        row = _mapping(raw, f"{actor} mount row")
        _exact_keys(
            row,
            {
                "actor_id",
                "read_only_source_relative_paths",
                "target_truth_mounted",
                "split_mounted",
                "role_mounted",
            },
            f"{actor} mount row",
        )
        source_rows = manifests[actor]["rows"]  # type: ignore[index]
        expected_paths = [source["relative_path"] for source in source_rows]  # type: ignore[index]
        if (
            row["actor_id"] != actor
            or row["read_only_source_relative_paths"] != expected_paths
            or row["target_truth_mounted"] is not False
            or row["split_mounted"] is not False
            or row["role_mounted"] is not False
        ):
            _fail("REJECT_Q1_SUCCESSOR_MOUNT", f"{actor} mount row differs")
    return mount, "sha256:" + content_hash(
        MOUNT_MANIFEST_ROOT_DOMAIN, _json_wire(mount)
    ).hex()


def _validate_input_absence(value: object) -> tuple[dict[str, object], str]:
    absence = _mapping(value, "input absence manifest")
    _exact_keys(
        absence,
        {
            "schema_version",
            "formal_input_keys",
            "target_truth_present",
            "split_present",
            "role_present",
        },
        "input absence manifest",
    )
    if (
        absence["schema_version"] != INPUT_ABSENCE_SCHEMA_VERSION
        or absence["formal_input_keys"] != []
        or absence["target_truth_present"] is not False
        or absence["split_present"] is not False
        or absence["role_present"] is not False
    ):
        _fail("REJECT_Q1_SUCCESSOR_ABSENCE", "input absence manifest differs")
    return absence, "sha256:" + content_hash(
        INPUT_ABSENCE_ROOT_DOMAIN, _json_wire(absence)
    ).hex()


def _validate_v6_observations(value: object) -> dict[str, object]:
    observations = _mapping(value, "v6 scratch observations")
    scratch_fields = (
        "projected_peak_formal_scratch_bytes",
        "python_actual_preflight_peak_scratch_bytes",
        "rust_actual_preflight_peak_scratch_bytes",
    )
    scratch_values = tuple(observations.get(field) for field in scratch_fields)
    if any(type(item) is not int or item < 0 for item in scratch_values):
        _fail("REJECT_Q1_SUCCESSOR_RESOURCE", "scratch observations must be uints")
    if observations.get("selected_worst_peak_scratch_bytes") != max(scratch_values):
        _fail(
            "REJECT_Q1_SUCCESSOR_RESOURCE",
            "selected scratch observation is not the mechanical maximum",
        )
    if observations != EXPECTED_V6_SCRATCH_OBSERVATIONS:
        _fail("REJECT_Q1_SUCCESSOR_RESOURCE", "bound v6 scratch observations differ")
    return observations


def _validate_resource_envelope(
    value: object,
    observations: dict[str, object],
) -> tuple[dict[str, object], str]:
    envelope = _mapping(value, "corrected resource envelope")
    endpoint = envelope.get("endpoint")
    selected = observations["selected_worst_peak_scratch_bytes"]
    mechanically_derived = 1 << (max(1, 2 * selected) - 1).bit_length()  # type: ignore[operator]
    if type(endpoint) is not dict or endpoint.get("scratch_bytes") != mechanically_derived:
        _fail(
            "REJECT_Q1_SUCCESSOR_RESOURCE",
            "endpoint scratch guard is not pow2ceil twice the selected maximum",
        )
    if mechanically_derived != 134_217_728:
        _fail(
            "REJECT_Q1_SUCCESSOR_RESOURCE",
            "bound v6 observations do not derive the 128 MiB scratch guard",
        )
    if envelope != EXPECTED_CORRECTED_RESOURCE_ENVELOPE:
        _fail("REJECT_Q1_SUCCESSOR_RESOURCE", "corrected 128 MiB resource envelope differs")
    return envelope, "sha256:" + content_hash(
        RESOURCE_ENVELOPE_ROOT_DOMAIN, _json_wire(envelope)
    ).hex()


def _authority_object() -> tuple[object, ...]:
    return (
        b"NOT_RUN",
        Q1_GATE_COUNT,
        Q1_GATE_MASK,
        Q1_GATE_TOTAL,
        tuple(NULL_OUTPUT_SLOTS),
        False,
        False,
        False,
        False,
    )


def _source_object(source: dict[str, object]) -> tuple[object, ...]:
    return (
        2,
        b"hegel-q1-prestart-successor-source-identity/4",
        bytes.fromhex(source["source_commit"]),  # type: ignore[arg-type]
        bytes.fromhex(source["source_tree"]),  # type: ignore[arg-type]
        _root_bytes(source["source_closure_root"], "source closure root"),
    )


def _root_rows(roots: dict[str, object]) -> tuple[tuple[bytes, bytes], ...]:
    return tuple(
        (actor.encode("ascii"), _root_bytes(roots[actor], actor)) for actor in ACTORS
    )


def _genesis_object(
    *,
    run_id: bytes,
    created_at_unix_seconds: int,
    source_identity: dict[str, object],
    base_chain: dict[str, object],
    v6_scratch_observations: dict[str, object],
    corrected_resource_envelope_root: str,
    actor_source_manifest_roots: dict[str, object],
    actor_source_closure_root: str,
    actor_implementation_roots: dict[str, object],
    actor_offline_isolation_roots: dict[str, object],
    formal_output_layout_root: str,
    mount_manifest_root: str,
    input_absence_root: str,
) -> tuple[object, ...]:
    return (
        2,
        RUN_GENESIS_SCHEMA_ID,
        run_id,
        _source_object(source_identity),
        _json_wire(base_chain),
        _json_wire(v6_scratch_observations),
        _root_bytes(corrected_resource_envelope_root, "resource envelope root"),
        _root_rows(actor_source_manifest_roots),
        _root_bytes(actor_source_closure_root, "actor source closure root"),
        _root_rows(actor_implementation_roots),
        _root_rows(actor_offline_isolation_roots),
        _root_bytes(formal_output_layout_root, "formal output layout root"),
        _root_bytes(mount_manifest_root, "mount manifest root"),
        _root_bytes(input_absence_root, "input absence root"),
        _authority_object(),
        created_at_unix_seconds,
    )


def _new_gate_subjects(
    *,
    genesis: tuple[object, ...],
    genesis_root: str,
    base_chain: dict[str, object],
    v6_observations: dict[str, object],
    resource_envelope: dict[str, object],
    resource_root: str,
    manifest_roots: dict[str, object],
    actor_source_closure_root: str,
    implementation_roots: dict[str, object],
    offline_roots: dict[str, object],
    formal_output_layout_root: str,
    mount_manifest_root: str,
    input_absence_root: str,
) -> tuple[tuple[object, ...], ...]:
    return (
        (
            _json_wire(v6_observations),
            _json_wire(resource_envelope),
            _root_bytes(resource_root, "resource root"),
            _root_bytes(base_chain["q1_start_v3_terminal_record_digest"], "terminal"),
            _root_bytes(base_chain["q1_start_v3_capsule_digest"], "capsule"),
        ),
        (
            _root_rows(manifest_roots),
            _root_bytes(actor_source_closure_root, "actor source closure"),
            _root_rows(implementation_roots),
            _root_rows(offline_roots),
            _root_bytes(formal_output_layout_root, "output layout"),
        ),
        (
            _root_bytes(actor_source_closure_root, "actor source closure"),
            _root_bytes(mount_manifest_root, "mount manifest"),
            _root_bytes(input_absence_root, "input absence"),
            False,
            False,
            False,
        ),
        (
            _root_bytes(genesis_root, "genesis root"),
            genesis,
            tuple(NULL_OUTPUT_SLOTS),
            False,
            _root_bytes(base_chain["base_prestart_receipt_root"], "base receipt"),
            _root_bytes(base_chain["q1_start_v3_terminal_record_digest"], "terminal"),
        ),
    )


def _gate_evidence_root(
    source_identity: dict[str, object],
    gate_id: int,
    subjects: tuple[object, ...],
) -> str:
    return "sha256:" + content_hash(
        GATE_EVIDENCE_ROOT_DOMAIN,
        (
            _source_object(source_identity),
            gate_id,
            GATE_NAMES[gate_id - 1].encode("ascii"),
            subjects,
        ),
    ).hex()


def _receipt_object(
    *,
    source_identity: dict[str, object],
    base_chain: dict[str, object],
    genesis: tuple[object, ...],
    genesis_root: str,
    gate_rows: list[dict[str, object]],
) -> tuple[object, ...]:
    rows = tuple(
        (
            row["gate_id"],
            row["gate_name"].encode("ascii"),  # type: ignore[union-attr]
            True,
            _root_bytes(row["evidence_root"], "gate evidence root"),
        )
        for row in gate_rows
    )
    return (
        2,
        RECEIPT_SCHEMA_ID,
        _json_wire(base_chain),
        _source_object(source_identity),
        _root_bytes(genesis_root, "genesis root"),
        genesis,
        Q1_GATE_COUNT,
        rows,
        Q1_GATE_MASK,
        _authority_object(),
    )


TOP_LEVEL_KEYS: Final = {
    "action_id",
    "actor_implementation_descriptors",
    "actor_implementation_roots",
    "actor_offline_isolation_roots",
    "actor_source_closure_root",
    "actor_source_manifests",
    "actor_source_manifest_roots",
    "authorization_consumption_count",
    "authorized",
    "base_chain",
    "bundle_root",
    "campaign_execution_count",
    "candidate_only",
    "corrected_resource_envelope",
    "corrected_resource_envelope_root",
    "formal_output_layout_root",
    "gate_evidence_rows",
    "input_absence_manifest",
    "input_absence_root",
    "intended_q1_gate_count",
    "intended_q1_gate_mask",
    "intended_q1_state",
    "mount_manifest",
    "mount_manifest_root",
    "network_accessed",
    "popen_count",
    "published",
    "q1_execution_started",
    "q1_output_slots",
    "receipt_cbor_hex",
    "receipt_root",
    "role_evaluation_performed",
    "run_genesis_cbor_hex",
    "run_genesis_root",
    "schema_version",
    "source_identity",
    "split_accessed",
    "status",
    "target_truth_accessed",
    "v6_scratch_observations",
}


def successor_bundle_root_v4(value: object) -> str:
    bundle = _mapping(value, "successor bundle")
    bundle.pop("bundle_root", None)
    return "sha256:" + content_hash(BUNDLE_ROOT_DOMAIN, _json_wire(bundle)).hex()


def validate_q1_prestart_successor_candidate_v4(
    value: object,
) -> dict[str, object]:
    bundle = _mapping(value, "successor bundle")
    _exact_keys(bundle, TOP_LEVEL_KEYS, "successor bundle")
    _reject_predecessor_runtime_paths(bundle)
    if (
        bundle["schema_version"] != SCHEMA_VERSION
        or bundle["action_id"] != ACTION_ID
        or bundle["status"] != STATUS
        or bundle["candidate_only"] is not True
        or bundle["authorized"] is not False
        or bundle["authorization_consumption_count"] != 0
        or type(bundle["authorization_consumption_count"]) is not int
        or bundle["campaign_execution_count"] != 0
        or type(bundle["campaign_execution_count"]) is not int
        or bundle["popen_count"] != 0
        or type(bundle["popen_count"]) is not int
        or bundle["published"] is not False
        or bundle["q1_execution_started"] is not False
        or bundle["intended_q1_state"] != "NOT_RUN"
        or bundle["intended_q1_gate_count"] != Q1_GATE_COUNT
        or bundle["intended_q1_gate_mask"] != Q1_GATE_MASK
        or bundle["q1_output_slots"] != NULL_OUTPUT_SLOTS
        or bundle["network_accessed"] is not False
        or bundle["target_truth_accessed"] is not False
        or bundle["split_accessed"] is not False
        or bundle["role_evaluation_performed"] is not False
    ):
        _fail("REJECT_Q1_SUCCESSOR_AUTHORITY", "candidate authority boundary differs")
    base_chain = _mapping(bundle["base_chain"], "base chain")
    if base_chain != _base_chain():
        _fail("REJECT_Q1_SUCCESSOR_BASE", "base chain identity differs")
    source = _validate_source_identity(bundle["source_identity"])
    observations = _validate_v6_observations(bundle["v6_scratch_observations"])
    resource, resource_root = _validate_resource_envelope(
        bundle["corrected_resource_envelope"], observations
    )
    if bundle["corrected_resource_envelope_root"] != resource_root:
        _fail("REJECT_Q1_SUCCESSOR_RESOURCE", "resource envelope root differs")
    manifests, manifest_roots = _validate_actor_source_manifests(
        bundle["actor_source_manifests"]
    )
    if bundle["actor_source_manifest_roots"] != manifest_roots:
        _fail("REJECT_Q1_SUCCESSOR_ACTOR", "derived actor source manifest roots differ")
    descriptors, implementation_roots, offline_roots = (
        _validate_actor_implementation_descriptors(
            bundle["actor_implementation_descriptors"],
            manifest_roots,
            manifests,
        )
    )
    if bundle["actor_implementation_descriptors"] != descriptors:
        _fail("REJECT_Q1_SUCCESSOR_ACTOR", "actor implementation descriptors differ")
    if bundle["actor_implementation_roots"] != implementation_roots:
        _fail("REJECT_Q1_SUCCESSOR_ACTOR", "derived actor implementation roots differ")
    if bundle["actor_offline_isolation_roots"] != offline_roots:
        _fail("REJECT_Q1_SUCCESSOR_ACTOR", "derived offline isolation roots differ")
    actor_source_closure_root = _actor_source_closure_root(manifest_roots)
    if bundle["actor_source_closure_root"] != actor_source_closure_root:
        _fail("REJECT_Q1_SUCCESSOR_ACTOR", "actor source closure root differs")
    mount, mount_root = _validate_mount_manifest(
        bundle["mount_manifest"], manifests
    )
    if bundle["mount_manifest"] != mount or bundle["mount_manifest_root"] != mount_root:
        _fail("REJECT_Q1_SUCCESSOR_MOUNT", "derived mount manifest root differs")
    absence, absence_root = _validate_input_absence(
        bundle["input_absence_manifest"]
    )
    if (
        bundle["input_absence_manifest"] != absence
        or bundle["input_absence_root"] != absence_root
    ):
        _fail("REJECT_Q1_SUCCESSOR_ABSENCE", "derived input absence root differs")
    if bundle["formal_output_layout_root"] != FORMAL_OUTPUT_LAYOUT_ROOT:
        _fail("REJECT_Q1_SUCCESSOR_ACTOR", "formal output layout root differs")
    try:
        run_genesis_payload = bytes.fromhex(bundle["run_genesis_cbor_hex"])  # type: ignore[arg-type]
        genesis = canonical_cbor_decode(run_genesis_payload)
    except (TypeError, ValueError) as error:
        _fail("REJECT_Q1_SUCCESSOR_GENESIS", str(error))
    if type(genesis) is not tuple or len(genesis) != 16 or genesis[:2] != (
        2,
        RUN_GENESIS_SCHEMA_ID,
    ):
        _fail("REJECT_Q1_SUCCESSOR_GENESIS", "run genesis wire differs")
    if (
        genesis[3] != _source_object(source)
        or genesis[4] != _json_wire(base_chain)
        or genesis[5] != _json_wire(observations)
        or genesis[6] != _root_bytes(resource_root, "resource root")
        or genesis[7] != _root_rows(manifest_roots)
        or genesis[8] != _root_bytes(actor_source_closure_root, "actor closure")
        or genesis[9] != _root_rows(implementation_roots)
        or genesis[10] != _root_rows(offline_roots)
        or genesis[11] != _root_bytes(FORMAL_OUTPUT_LAYOUT_ROOT, "output layout")
        or genesis[12] != _root_bytes(mount_root, "mount root")
        or genesis[13] != _root_bytes(absence_root, "absence root")
        or genesis[14] != _authority_object()
        or type(genesis[15]) is not int
        or genesis[15] < 0
        or type(genesis[2]) is not bytes
        or len(genesis[2]) != 16
    ):
        _fail("REJECT_Q1_SUCCESSOR_GENESIS", "run genesis binding differs")
    genesis_root = "sha256:" + content_hash(RUN_GENESIS_ROOT_DOMAIN, genesis).hex()
    if bundle["run_genesis_root"] != genesis_root:
        _fail("REJECT_Q1_SUCCESSOR_GENESIS", "run genesis root differs")
    rows = bundle["gate_evidence_rows"]
    if type(rows) is not list or len(rows) != Q1_GATE_COUNT:
        _fail("REJECT_Q1_SUCCESSOR_GATE", "fourteen gate rows are required")
    if rows[:10] != [_legacy_gate_dict(row) for row in LEGACY_GATE_ROWS_1_TO_10]:
        _fail("REJECT_Q1_SUCCESSOR_GATE", "legacy gate projection differs")
    expected_subjects = _new_gate_subjects(
        genesis=genesis,
        genesis_root=genesis_root,
        base_chain=base_chain,
        v6_observations=observations,
        resource_envelope=resource,
        resource_root=resource_root,
        manifest_roots=manifest_roots,
        actor_source_closure_root=actor_source_closure_root,
        implementation_roots=implementation_roots,
        offline_roots=offline_roots,
        formal_output_layout_root=FORMAL_OUTPUT_LAYOUT_ROOT,
        mount_manifest_root=mount_root,
        input_absence_root=absence_root,
    )
    for gate_id, (row, subjects) in enumerate(zip(rows[10:], expected_subjects, strict=True), 11):
        expected = {
            "gate_id": gate_id,
            "gate_name": GATE_NAMES[gate_id - 1],
            "evidence_root": _gate_evidence_root(source, gate_id, subjects),
            "subjects_cbor_hex": canonical_cbor_encode(subjects).hex(),
        }
        if row != expected:
            _fail("REJECT_Q1_SUCCESSOR_GATE", f"Gate {gate_id} evidence differs")
    receipt = _receipt_object(
        source_identity=source,
        base_chain=base_chain,
        genesis=genesis,
        genesis_root=genesis_root,
        gate_rows=rows,  # type: ignore[arg-type]
    )
    receipt_payload = canonical_cbor_encode(receipt)
    receipt_root = "sha256:" + content_hash(RECEIPT_ROOT_DOMAIN, receipt).hex()
    if (
        bundle["receipt_cbor_hex"] != receipt_payload.hex()
        or bundle["receipt_root"] != receipt_root
    ):
        _fail("REJECT_Q1_SUCCESSOR_RECEIPT", "successor receipt replay differs")
    if bundle["bundle_root"] != successor_bundle_root_v4(bundle):
        _fail("REJECT_Q1_SUCCESSOR_BUNDLE", "successor bundle root differs")
    return deepcopy(bundle)


def replay_q1_prestart_successor_candidate_v4(
    payload: bytes,
) -> dict[str, object]:
    return validate_q1_prestart_successor_candidate_v4(
        _decode_canonical_json(payload, "successor bundle")
    )


def derive_q1_prestart_successor_candidate_v4(
    base_authoritative_artifact_payload: bytes,
    q1_start_v3_terminal_payload: bytes,
    *,
    v6_scratch_observations: Mapping[str, object],
    corrected_resource_envelope: Mapping[str, object],
    source_identity: Mapping[str, object],
    actor_source_manifests: Mapping[str, object],
    actor_implementation_descriptors: Mapping[str, object],
    mount_manifest: Mapping[str, object],
    input_absence_manifest: Mapping[str, object],
    formal_output_layout_root: str,
    run_id: bytes,
    created_at_unix_seconds: int,
) -> dict[str, object]:
    """Derive one deterministic, unconsumed successor-prestart bundle."""

    legacy_rows, _base = _validate_base_artifact(base_authoritative_artifact_payload)
    _validate_q1_start_v3_terminal(q1_start_v3_terminal_payload)
    all_inputs = {
        "v6_scratch_observations": dict(v6_scratch_observations),
        "corrected_resource_envelope": dict(corrected_resource_envelope),
        "source_identity": dict(source_identity),
        "actor_source_manifests": dict(actor_source_manifests),
        "actor_implementation_descriptors": dict(actor_implementation_descriptors),
        "mount_manifest": dict(mount_manifest),
        "input_absence_manifest": dict(input_absence_manifest),
    }
    _reject_predecessor_runtime_paths(all_inputs)
    observations = _validate_v6_observations(v6_scratch_observations)
    resource, resource_root = _validate_resource_envelope(
        corrected_resource_envelope, observations
    )
    source = _validate_source_identity(source_identity)
    manifests, manifest_roots = _validate_actor_source_manifests(actor_source_manifests)
    descriptors, implementation_roots, offline_roots = (
        _validate_actor_implementation_descriptors(
            actor_implementation_descriptors,
            manifest_roots,
            manifests,
        )
    )
    actor_source_closure_root = _actor_source_closure_root(manifest_roots)
    mount, mount_root = _validate_mount_manifest(mount_manifest, manifests)
    absence, absence_root = _validate_input_absence(input_absence_manifest)
    if formal_output_layout_root != FORMAL_OUTPUT_LAYOUT_ROOT:
        _fail("REJECT_Q1_SUCCESSOR_ACTOR", "formal output layout root differs")
    if type(run_id) is not bytes or len(run_id) != 16:
        _fail("REJECT_Q1_SUCCESSOR_GENESIS", "run ID must be exactly 16 bytes")
    if type(created_at_unix_seconds) is not int or created_at_unix_seconds < 0:
        _fail("REJECT_Q1_SUCCESSOR_GENESIS", "creation time must be uint64")
    base_chain = _base_chain()
    genesis = _genesis_object(
        run_id=run_id,
        created_at_unix_seconds=created_at_unix_seconds,
        source_identity=source,
        base_chain=base_chain,
        v6_scratch_observations=observations,
        corrected_resource_envelope_root=resource_root,
        actor_source_manifest_roots=manifest_roots,
        actor_source_closure_root=actor_source_closure_root,
        actor_implementation_roots=implementation_roots,
        actor_offline_isolation_roots=offline_roots,
        formal_output_layout_root=formal_output_layout_root,
        mount_manifest_root=mount_root,
        input_absence_root=absence_root,
    )
    genesis_payload = canonical_cbor_encode(genesis)
    genesis_root = "sha256:" + content_hash(RUN_GENESIS_ROOT_DOMAIN, genesis).hex()
    subjects = _new_gate_subjects(
        genesis=genesis,
        genesis_root=genesis_root,
        base_chain=base_chain,
        v6_observations=observations,
        resource_envelope=resource,
        resource_root=resource_root,
        manifest_roots=manifest_roots,
        actor_source_closure_root=actor_source_closure_root,
        implementation_roots=implementation_roots,
        offline_roots=offline_roots,
        formal_output_layout_root=formal_output_layout_root,
        mount_manifest_root=mount_root,
        input_absence_root=absence_root,
    )
    new_rows = [
        {
            "gate_id": gate_id,
            "gate_name": GATE_NAMES[gate_id - 1],
            "evidence_root": _gate_evidence_root(source, gate_id, row),
            "subjects_cbor_hex": canonical_cbor_encode(row).hex(),
        }
        for gate_id, row in enumerate(subjects, 11)
    ]
    gate_rows = legacy_rows + new_rows
    receipt = _receipt_object(
        source_identity=source,
        base_chain=base_chain,
        genesis=genesis,
        genesis_root=genesis_root,
        gate_rows=gate_rows,
    )
    receipt_payload = canonical_cbor_encode(receipt)
    bundle: dict[str, object] = {
        "action_id": ACTION_ID,
        "actor_implementation_descriptors": descriptors,
        "actor_implementation_roots": implementation_roots,
        "actor_offline_isolation_roots": offline_roots,
        "actor_source_closure_root": actor_source_closure_root,
        "actor_source_manifests": manifests,
        "actor_source_manifest_roots": manifest_roots,
        "authorization_consumption_count": 0,
        "authorized": False,
        "base_chain": base_chain,
        "campaign_execution_count": 0,
        "candidate_only": True,
        "corrected_resource_envelope": resource,
        "corrected_resource_envelope_root": resource_root,
        "formal_output_layout_root": formal_output_layout_root,
        "gate_evidence_rows": gate_rows,
        "input_absence_manifest": absence,
        "input_absence_root": absence_root,
        "intended_q1_gate_count": Q1_GATE_COUNT,
        "intended_q1_gate_mask": Q1_GATE_MASK,
        "intended_q1_state": "NOT_RUN",
        "mount_manifest": mount,
        "mount_manifest_root": mount_root,
        "network_accessed": False,
        "popen_count": 0,
        "published": False,
        "q1_execution_started": False,
        "q1_output_slots": deepcopy(NULL_OUTPUT_SLOTS),
        "receipt_cbor_hex": receipt_payload.hex(),
        "receipt_root": "sha256:" + content_hash(RECEIPT_ROOT_DOMAIN, receipt).hex(),
        "role_evaluation_performed": False,
        "run_genesis_cbor_hex": genesis_payload.hex(),
        "run_genesis_root": genesis_root,
        "schema_version": SCHEMA_VERSION,
        "source_identity": source,
        "split_accessed": False,
        "status": STATUS,
        "target_truth_accessed": False,
        "v6_scratch_observations": observations,
    }
    bundle["bundle_root"] = successor_bundle_root_v4(bundle)
    return validate_q1_prestart_successor_candidate_v4(bundle)


__all__ = [
    "ACTION_ID",
    "ACTOR_BOUND_ENVIRONMENT_ROOTS",
    "ACTOR_ENTRYPOINT_RELATIVE_PATHS",
    "ACTOR_IMPLEMENTATION_REGISTRY",
    "ACTOR_IMPLEMENTATION_ROOT_DOMAIN",
    "ACTOR_SOURCE_CLOSURE_ROOT_DOMAIN",
    "ACTOR_SOURCE_MANIFEST_ROOT_DOMAIN",
    "BASE_ARTIFACT_RAW_SHA256",
    "BUNDLE_ROOT_DOMAIN",
    "EXPECTED_CORRECTED_RESOURCE_ENVELOPE",
    "EXPECTED_V6_SCRATCH_OBSERVATIONS",
    "FORMAL_OUTPUT_LAYOUT_ROOT",
    "INPUT_ABSENCE_SCHEMA_VERSION",
    "MOUNT_MANIFEST_SCHEMA_VERSION",
    "Q1_START_V3_AUTHORIZATION_DIGEST",
    "Q1_START_V3_AUTHORIZATION_RAW_SHA256",
    "Q1_START_V3_CAPSULE_DIGEST",
    "Q1_START_V3_CAPSULE_RAW_SHA256",
    "Q1_START_V3_CONSUMPTION_RAW_SHA256",
    "Q1_START_V3_CONSUMPTION_RECORD_DIGEST",
    "Q1_START_V3_OUTCOME_RAW_SHA256",
    "Q1_START_V3_TERMINAL_RAW_SHA256",
    "Q1_START_V3_TERMINAL_RECORD_DIGEST",
    "Q1_START_V2_AUTHORIZATION_DIGEST",
    "Q1_START_V2_AUTHORIZATION_RAW_SHA256",
    "Q1_START_V2_CAPSULE_DIGEST",
    "Q1_START_V2_CAPSULE_RAW_SHA256",
    "Q1_START_V2_CONSUMPTION_RAW_SHA256",
    "Q1_START_V2_CONSUMPTION_RECORD_DIGEST",
    "Q1_START_V2_OUTCOME_RAW_SHA256",
    "Q1_START_V2_TERMINAL_RAW_SHA256",
    "Q1_START_V2_TERMINAL_RECORD_DIGEST",
    "Q1PrestartSuccessorCandidateError",
    "SCHEMA_VERSION",
    "STATUS",
    "TOP_LEVEL_KEYS",
    "actor_implementation_root_v4",
    "actor_source_manifest_root_v4",
    "canonical_json_bytes_v4",
    "derive_q1_prestart_successor_candidate_v4",
    "replay_q1_prestart_successor_candidate_v4",
    "successor_bundle_root_v4",
    "validate_q1_prestart_successor_candidate_v4",
]
