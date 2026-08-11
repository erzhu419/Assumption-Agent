"""Strict qualification-only Q0.5b actual-artifact assembler and replayer.

This module is pure: it starts no process, opens no file, publishes nothing,
and cannot advance formal Q1.  It accepts complete evidence preimages, replays
their production wire contracts, derives every receipt/evidence root, and
returns one canonical-JSON value suitable for a later atomic publisher.
"""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha1, sha256
import ast
import io
import json
import re
import tarfile
from typing import Final, Mapping, NoReturn

try:  # Python 3.11 stdlib; the project test runtime also supports 3.10+tomli.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - interpreter dependent
    import tomli as tomllib

from . import phase3_q05b_host_replay_v1 as _host
from . import phase3_q05b_negative_vectors_v1 as _negative
from . import phase3_q05b_actual_admission_v1 as _admission
from . import phase3_q1_qualification_wire_v1 as _wire
from .strict_cbor_v1 import canonical_cbor_decode, canonical_cbor_encode, content_hash


ARTIFACT_SCHEMA_VERSION: Final = "hegel-phase3a-q05b-actual-artifact/1"
ACTUAL_ARTIFACT_MAX_CANONICAL_BYTES: Final = 64 * 1024 * 1024
HOST_EXECUTION_BINDING_ROOT_DOMAIN: Final = "HEGEL/Q05B/ACTUAL/HOST_EXECUTION_BINDING/V1"
RESOURCE_EVIDENCE_ROOT_DOMAIN: Final = "HEGEL/Q05B/ACTUAL/RESOURCE/V1"
ISOLATION_EVIDENCE_ROOT_DOMAIN: Final = "HEGEL/Q05B/ACTUAL/ISOLATION/V1"
BUNDLE_EVIDENCE_ROOT_DOMAIN: Final = "HEGEL/Q05B/ACTUAL/BUNDLE/V1"
PREDICATE11_EVIDENCE_ROOT_DOMAIN: Final = "HEGEL/Q05B/ACTUAL/PREDICATE11/V1"
PREDICATE19_EVIDENCE_ROOT_DOMAIN: Final = "HEGEL/Q05B/ACTUAL/PREDICATE19/V1"
PREDICATE_EVIDENCE_ROOT_DOMAIN: Final = "HEGEL/Q05B/ACTUAL/PREDICATE/V1"
ARTIFACT_SET_ROOT_DOMAIN: Final = "HEGEL/Q05B/QUALIFICATION/ARTIFACT_SET/V1"
ACTUAL_ADMISSION_ARTIFACT_EVIDENCE_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-actual-admission-artifact-evidence/1"
)
ACTUAL_ADMISSION_ARTIFACT_EVIDENCE_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/ACTUAL/ADMISSION_ARTIFACT_EVIDENCE/V1\x00"
)
DOCKER_OWNERSHIP_LABEL_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/DOCKER/OWNERSHIP_LABELS/V1\x00"
)
DOCKER_OWNED_INSPECT_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/DOCKER/OWNED_INSPECT/V1\x00"
)

SECTION_NAMES: Final = (
    "source_wire_profile",
    "five_sidecars",
    "endpoint_stdout_set",
    "host_stage",
    "actor_rows",
    "cargo_build_binary",
    "final_resource_rows",
    "negative_corpus",
    "scratch_rows",
    "actual_admission",
    "semantic_execution",
)

CLOSED_Q1_AUTHORITY: Final = {
    "active_transition_allowed": False,
    "certificate_active": False,
    "formal_output_roots": [None] * 8,
    "gate_count": 0,
    "gate_mask": 0,
    "gate_total": 20,
    "formal_fixed_point_claimed": False,
    "m3_formal_roots": None,
    "outside_certificate_issued": False,
    "q1_receipt": None,
    "q2_state": "NOT_RUN",
    "state": "NOT_RUN",
}

STAGE8_CANDIDATE_REGISTRY_KEYS: Final = frozenset({
    "actual_admission_evidence_root",
    "bundle_evidence_root",
    "candidate_receipt_cbor_hex",
    "candidate_receipt_root",
    "closed_q1_authority",
    "host_execution_binding_root",
    "isolation_evidence_root",
    "ordered_predicate_rows",
    "qualification_count",
    "qualification_mask",
    "resource_evidence_root",
})
STAGE9_DERIVED_REGISTRY_KEYS: Final = frozenset({
    "actual_admission_evidence_root",
    "artifact_set_root",
    "bundle_evidence_root",
    "candidate_receipt_cbor_hex",
    "candidate_receipt_root",
    "closed_q1_authority",
    "final_receipt_cbor_hex",
    "final_receipt_root",
    "host_execution_binding_root",
    "isolation_evidence_root",
    "ordered_predicate_rows",
    "qualification_count",
    "qualification_mask",
    "resource_evidence_root",
})


class Q05BActualArtifactError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Q05BActualArtifactError(code, detail)


def _canonical_json(value: object) -> bytes:
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
        _fail("REJECT_Q05B_ARTIFACT_JSON", str(error))


def _json_root(domain: str, value: object) -> bytes:
    return sha256(domain.encode("ascii") + b"\x00" + _canonical_json(value)).digest()


def _object(value: object, keys: set[str], name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        _fail("REJECT_Q05B_ARTIFACT_SCHEMA", f"{name} field registry differs")
    return value


def _require_type_exact_v1(value: object, expected: object, name: str) -> None:
    if type(value) is not type(expected):
        _fail("REJECT_Q05B_ARTIFACT_SCHEMA", f"{name} type differs")
    if type(expected) is dict:
        assert type(value) is dict
        if set(value) != set(expected):
            _fail("REJECT_Q05B_ARTIFACT_SCHEMA", f"{name} fields differ")
        for key in expected:
            _require_type_exact_v1(value[key], expected[key], f"{name}.{key}")
    elif type(expected) is list:
        assert type(value) is list
        if len(value) != len(expected):
            _fail("REJECT_Q05B_ARTIFACT_SCHEMA", f"{name} length differs")
        for index, (item, expected_item) in enumerate(
            zip(value, expected, strict=True)
        ):
            _require_type_exact_v1(item, expected_item, f"{name}[{index}]")
    elif value != expected:
        _fail("REJECT_Q05B_ARTIFACT_SCHEMA", f"{name} value differs")


def _hex(value: object, length: int, name: str) -> bytes:
    if (
        type(value) is not str
        or type(length) is not int
        or length < 0
        or len(value) != 2 * length
        or re.fullmatch(r"[0-9a-f]+", value) is None
    ):
        _fail("REJECT_Q05B_ARTIFACT_HEX", f"{name} is not exact lowercase hex")
    return bytes.fromhex(value)


def _hex_any(value: object, name: str) -> bytes:
    if type(value) is not str or len(value) % 2:
        _fail("REJECT_Q05B_ARTIFACT_HEX", f"{name} is not even-length text hex")
    if value == "":
        return b""
    return _hex(value, len(value) // 2, name)


def _strict_json(payload: bytes) -> dict[str, object]:
    if type(payload) is not bytes or not payload.endswith(b"\n"):
        _fail("REJECT_Q05B_ARTIFACT_JSON", "artifact must end in one LF")

    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if type(key) is not str or key in result:
                _fail("REJECT_Q05B_ARTIFACT_JSON", "duplicate/non-string JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: _fail(
                "REJECT_Q05B_ARTIFACT_JSON", f"non-finite token {token}"
            ),
            parse_float=lambda token: _fail(
                "REJECT_Q05B_ARTIFACT_JSON", f"float token {token}"
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        _fail("REJECT_Q05B_ARTIFACT_JSON", str(error))
    if type(value) is not dict or _canonical_json(value) != payload:
        _fail("REJECT_Q05B_ARTIFACT_JSON", "artifact JSON is not canonical")
    return value


def _sidecars_v1(value: object):
    if type(value) is not list or len(value) != 5:
        _fail("REJECT_Q05B_ARTIFACT_SIDECAR", "five sidecar rows required")
    payloads: list[bytes] = []
    expected_paths = [path.decode("ascii") for path in _wire.ORDERED_OUTPUT_RELATIVE_PATHS]
    for index, row_value in enumerate(value):
        row = _object(
            row_value,
            {"cbor_hex", "content_root", "length", "mode", "path", "raw_sha256"},
            f"sidecar[{index}]",
        )
        payload = _hex_any(row["cbor_hex"], "sidecar CBOR")
        if (
            row["path"] != expected_paths[index]
            or type(row["mode"]) is not int
            or row["mode"] != 0o444
            or type(row["length"]) is not int
            or row["length"] != len(payload)
            or row["raw_sha256"] != sha256(payload).hexdigest()
        ):
            _fail("REJECT_Q05B_ARTIFACT_SIDECAR", f"sidecar row {index} differs")
        payloads.append(payload)
    leaf = _wire.decode_full_v16_leaf_manifest_v1(payloads[0])
    odd = _wire.decode_node3_partition_evidence_v1(payloads[1])
    sink = _wire.decode_node3_partition_evidence_v1(payloads[2])
    sidecar = _wire.replay_sidecar_manifest_v1(payloads[3], tuple(payloads[:3]))
    golden = _wire.decode_node3_golden_manifest_v1(payloads[4])
    roots = (
        leaf.manifest_root,
        odd.evidence_root,
        sink.evidence_root,
        sidecar.manifest_root,
        golden.manifest_root,
    )
    for index, (row, root) in enumerate(zip(value, roots, strict=True)):
        if row["content_root"] != root.hex():
            _fail("REJECT_Q05B_ARTIFACT_SIDECAR", f"sidecar root {index} differs")
    if (
        golden.full_leaf_manifest_root != leaf.manifest_root
        or golden.sidecar_manifest_root != sidecar.manifest_root
    ):
        _fail("REJECT_Q05B_ARTIFACT_SIDECAR", "golden bundle binding differs")
    partition_replays = (
        _host.strict_replay_partition_streams_v1(odd),
        _host.strict_replay_partition_streams_v1(sink),
    )
    shadow = _host._shadow_assembler_v1(  # noqa: SLF001
        leaf, (odd, sink), golden, partition_replays
    )
    return tuple(payloads), leaf, (odd, sink), sidecar, golden, partition_replays, shadow


def _actor_replay_v1(actor_id: str, stdout: bytes, replayed):
    payloads, leaf, partitions, sidecar, golden, partition_replays, shadow = replayed
    envelope = _wire.validate_actor_stdout_envelope_v1(stdout)
    implementation = dict(_wire.ACTOR_IMPLEMENTATION_ID_REGISTRY)[actor_id]
    if (
        envelope["actor_id"] != actor_id
        or envelope["implementation_id"] != implementation
        or envelope["neutral_manifest_length"] != len(payloads[4])
        or envelope["neutral_manifest_raw_sha256"] != sha256(payloads[4]).hexdigest()
        or envelope["neutral_manifest_root"] != golden.manifest_root.hex()
        or envelope["sidecar_manifest_length"] != len(payloads[3])
        or envelope["sidecar_manifest_raw_sha256"] != sha256(payloads[3]).hexdigest()
        or envelope["sidecar_manifest_root"] != sidecar.manifest_root.hex()
    ):
        _fail("REJECT_Q05B_ARTIFACT_STDOUT", f"{actor_id} envelope differs")
    replay_root = content_hash(
        _host.HOST_REPLAY_ROOT_DOMAIN,
        (
            actor_id.encode("ascii"),
            tuple(sha256(payload).digest() for payload in payloads),
            tuple(item.canonical_object() for item in partition_replays),
            shadow.root,
            golden.canonical_object()[-1],
        ),
    )
    return _host.ActorSidecarReplayV1(
        actor_id,
        implementation,
        stdout,
        payloads,
        leaf,
        partitions,
        sidecar,
        golden,
        partition_replays,
        shadow,
        replay_root,
    )


def _dual_replay_v1(endpoint: dict[str, object], replayed, host_source: bytes, host_runtime: bytes):
    python_stdout = _hex_any(endpoint["python_stdout_hex"], "Python stdout")
    rust_stdout = _hex_any(endpoint["rust_stdout_hex"], "Rust stdout")
    manifest = _hex_any(endpoint["manifest_hex"], "stdout manifest")
    _host.validate_sealed_actor_stdout_set_v1(python_stdout, rust_stdout, manifest)
    python = _actor_replay_v1("PYTHON_ENDPOINT", python_stdout, replayed)
    rust = _actor_replay_v1("RUST_ENDPOINT", rust_stdout, replayed)
    payloads, _leaf, _partitions, sidecar, golden, partition_replays, shadow = replayed
    file_rows = tuple(
        (index, _wire.ORDERED_OUTPUT_RELATIVE_PATHS[index], len(payload), sha256(payload).digest())
        for index, payload in enumerate(payloads)
    )
    materialized = tuple(item.materialized_replay_roots for item in partition_replays)
    counting = tuple(item.counting_replay_roots for item in partition_replays)
    traces = tuple(item.trace_replay_roots for item in partition_replays)
    preimages = {
        6: (b"NEUTRAL_GOLDEN_MANIFEST_PYTHON_RUST_HOST_BYTE_EQUAL",) + (sha256(payloads[4]).digest(),) * 3,
        7: (b"SIDECAR_MANIFEST_PYTHON_RUST_HOST_BYTE_EQUAL", sha256(payloads[3]).digest(), sha256(payloads[3]).digest(), sidecar.manifest_root, sidecar.manifest_root),
        8: (b"SIDECAR_RAW_SHA_LENGTH_CONTENT_ROOT_REPLAY", file_rows, sidecar.canonical_object(), sha256(python_stdout).digest(), sha256(rust_stdout).digest(), sha256(manifest).digest()),
        11: (b"TRUSTED_HOST_READ_ONLY_REPLAY_QUALIFIED", python.host_replay_root, rust.host_replay_root, sha256(payloads[4]).digest(), sha256(manifest).digest(), host_source, host_runtime),
        12: (b"STRICT_PARTITION_MANIFEST_BUNDLE_ASSEMBLER_REPLAY", shadow.canonical_object, shadow.root, tuple(item.record_set_replay_root for item in partition_replays), tuple(item.coverage_replay_root for item in partition_replays)),
        14: (b"COUNTING_DISCARD_AND_MATERIALIZED_ENCODER_EQUAL", materialized, counting),
        15: (b"EXTERNAL_SORT_RUN_AND_MERGE_REPLAY_PASS", traces, tuple(item.scratch_ledger_roots for item in partition_replays)),
        17: (b"OUTPUT_AND_METADATA_FORMULA_REPLAY_PASS", file_rows, len(payloads[4]), len(payloads[3]), len(payloads[4]), len(payloads[3]), golden.canonical_object()[-1]),
    }
    host_rows = tuple(
        (predicate_id, content_hash(_host.HOST_PREDICATE_EVIDENCE_ROOT_DOMAIN, (predicate_id, preimages[predicate_id])))
        for predicate_id in (6, 7, 8, 12, 14, 15, 17)
    )
    pred11_component = content_hash(_host.HOST_PREDICATE_EVIDENCE_ROOT_DOMAIN, (11, preimages[11]))
    dual_root = content_hash(
        _host.HOST_REPLAY_ROOT_DOMAIN,
        (python.host_replay_root, rust.host_replay_root, sha256(payloads[4]).digest(), sha256(manifest).digest(), host_source, host_runtime, shadow.root, host_rows, pred11_component, (11, 13, 16, 18, 19), golden.canonical_object()[-1]),
    )
    return _host.DualHostReplayV1(python, rust, payloads[4], sha256(payloads[4]).digest(), sha256(manifest).digest(), host_source, host_runtime, shadow.root, host_rows, pred11_component, (11, 13, 16, 18, 19), dual_root)


@lru_cache(maxsize=1)
def _production_negative_corpus_v1() -> _negative.Q05BNegativeVectorCorpusV1:
    return _negative.run_q05b_negative_vector_corpus_v1()


def _negative_v1(value: object) -> _negative.Q05BNegativeVectorCorpusV1:
    section = _object(value, {"canonical_cbor_hex", "category13_root", "category18_root", "corpus_root"}, "negative_corpus")
    if type(section["canonical_cbor_hex"]) is not str:
        _fail("REJECT_Q05B_ARTIFACT_NEGATIVE", "negative corpus hex type differs")
    payload = _hex_any(section["canonical_cbor_hex"], "negative corpus")
    decoded = canonical_cbor_decode(payload)
    if type(decoded) is not tuple or len(decoded) != 5 or decoded[:2] != (1, _negative.NEGATIVE_VECTOR_CORPUS_SCHEMA_ID):
        _fail("REJECT_Q05B_ARTIFACT_NEGATIVE", "negative corpus wire differs")
    raw_rows = decoded[2]
    if type(raw_rows) is not tuple:
        _fail("REJECT_Q05B_ARTIFACT_NEGATIVE", "negative rows differ")
    rows = tuple(_negative.Q05BNegativeVectorRowV1(*row) for row in raw_rows)
    corpus = _negative.Q05BNegativeVectorCorpusV1(rows)
    if corpus.canonical_object() != decoded or payload != canonical_cbor_encode(decoded):
        _fail("REJECT_Q05B_ARTIFACT_NEGATIVE", "negative corpus replay differs")
    production = _production_negative_corpus_v1()
    if (
        corpus.canonical_object() != production.canonical_object()
        or payload != canonical_cbor_encode(production.canonical_object())
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_NEGATIVE",
            "negative production-validator replay differs",
        )
    roots = dict(corpus.category_roots)
    if section["corpus_root"] != corpus.corpus_root.hex() or section["category13_root"] != roots[13].hex() or section["category18_root"] != roots[18].hex():
        _fail("REJECT_Q05B_ARTIFACT_NEGATIVE", "negative roots differ")
    return corpus


def _validate_sealed_tree_identity_v1(
    value: object,
    payloads: Mapping[str, bytes],
    expected_directories: tuple[str, ...],
    name: str,
    expected_modes: Mapping[str, int] | None = None,
) -> dict[str, object]:
    identity = _object(
        value,
        {
            "directory_rows", "file_rows", "manifest_sha256", "root_device",
            "root_inode", "root_mode", "root_nlink", "root_path",
            "schema_version",
        },
        name,
    )
    if (
        identity["schema_version"]
        != "hegel-phase3a-q05b-sealed-tree-identity/1"
        or type(identity["root_path"]) is not str
        or not identity["root_path"].startswith("/")
        or ".." in identity["root_path"].split("/")
        or any(type(identity[field]) is not int or identity[field] < 1 for field in ("root_device", "root_inode", "root_nlink"))
        or identity["root_mode"] != 0o555
    ):
        _fail("REJECT_Q05B_ARTIFACT_ISOLATION", f"{name} root differs")
    directory_rows = identity["directory_rows"]
    if type(directory_rows) is not list or [row[0] if type(row) is list and row else None for row in directory_rows] != list(expected_directories):
        _fail("REJECT_Q05B_ARTIFACT_ISOLATION", f"{name} directory registry differs")
    for row in directory_rows:
        if type(row) is not list or len(row) != 9 or type(row[0]) is not str or any(type(row[index]) is not int for index in range(1, 9)) or row[3] < 1 or row[6] != 0o555:
            _fail("REJECT_Q05B_ARTIFACT_ISOLATION", f"{name} directory row differs")
    file_rows = identity["file_rows"]
    if type(file_rows) is not list or [row[0] if type(row) is list and row else None for row in file_rows] != list(payloads):
        _fail("REJECT_Q05B_ARTIFACT_ISOLATION", f"{name} file registry differs")
    for row, (path, payload) in zip(file_rows, payloads.items(), strict=True):
        mode = 0o444 if expected_modes is None else expected_modes[path]
        if type(row) is not list or len(row) != 11 or row[0] != path or any(type(row[index]) is not int for index in range(1, 10)) or row[3] != 1 or row[6] != mode or row[7] != len(payload) or row[10] != sha256(payload).hexdigest():
            _fail("REJECT_Q05B_ARTIFACT_ISOLATION", f"{name} file row differs")
    body = dict(identity); observed = body.pop("manifest_sha256")
    if observed != sha256(_canonical_json(body)).hexdigest():
        _fail("REJECT_Q05B_ARTIFACT_ISOLATION", f"{name} manifest differs")
    return identity


def _validate_actor_rows(
    value: object,
    commit: str,
    payload_table: dict[str, tuple[int, str, bytes]],
    actor_path_rows: object,
    config: dict[str, object],
) -> tuple[dict[str, object], ...]:
    if type(value) is not list or len(value) != 3:
        _fail("REJECT_Q05B_ARTIFACT_ACTOR", "three actor rows required")
    expected_ids = ("PYTHON_ENDPOINT", "RUST_ENDPOINT", "TRUSTED_HOST_REPLAY")
    if (
        type(actor_path_rows) is not list
        or len(actor_path_rows) != 3
        or any(type(row) is not list or len(row) != 2 for row in actor_path_rows)
        or [row[0] for row in actor_path_rows] != list(expected_ids)
    ):
        _fail("REJECT_Q05B_ARTIFACT_SOURCE", "actor source path registry differs")
    policy = config.get("source_allowlist_policy")
    if type(policy) is not dict or type(policy.get("actor_rows")) is not list:
        _fail("REJECT_Q05B_ARTIFACT_SOURCE", "source allowlist policy differs")
    policy_rows = policy["actor_rows"]
    if len(policy_rows) != 3:
        _fail("REJECT_Q05B_ARTIFACT_SOURCE", "source allowlist policy rows differ")
    union_paths: set[str] = set()
    for index, (path_row, policy_row) in enumerate(
        zip(actor_path_rows, policy_rows, strict=True), start=1
    ):
        paths = path_row[1]
        if (
            type(policy_row) is not list
            or len(policy_row) != 4
            or policy_row[:2] != [index, path_row[0]]
            or policy_row[2] != len(paths)
            or type(policy_row[2]) is not int
            or policy_row[3] != sha256(_canonical_json(paths)).hexdigest()
        ):
            _fail("REJECT_Q05B_ARTIFACT_SOURCE", "actor frozen path registry differs")
        union_paths.update(paths)
    if union_paths != set(payload_table):
        _fail("REJECT_Q05B_ARTIFACT_SOURCE", "Git blob table is not exact allowlist union")
    result = []
    for expected, raw, path_row in zip(expected_ids, value, actor_path_rows, strict=True):
        row = _object(raw, {"actor_id", "command", "control_evidence", "runtime_identity_sha256", "snapshot_identity", "source_evidence"}, expected)
        source = _object(row["source_evidence"], {"actor_id", "allowlist_count", "blob_rows", "commit", "git_blob_manifest_sha256", "path_registry_sha256", "source_identity_sha256"}, "source_evidence")
        paths = path_row[1]
        if type(paths) is not list or paths != sorted(paths) or len(set(paths)) != len(paths) or any(type(path) is not str or path not in payload_table for path in paths):
            _fail("REJECT_Q05B_ARTIFACT_SOURCE", f"{expected} source paths differ")
        blob_rows = [
            [path, payload_table[path][0], payload_table[path][1], len(payload_table[path][2]), sha256(payload_table[path][2]).hexdigest()]
            for path in paths
        ]
        snapshot_payloads = {path: payload_table[path][2] for path in paths}
        snapshot_directories = sorted({"/".join(path.split("/")[:depth]) for path in paths for depth in range(1, len(path.split("/")))})
        _validate_sealed_tree_identity_v1(
            row["snapshot_identity"], snapshot_payloads,
            tuple(snapshot_directories), f"{expected} snapshot",
            {path: 0o555 if payload_table[path][0] == 0o100755 else 0o444 for path in paths},
        )
        raw_digest = sha256()
        for path in paths:
            payload = payload_table[path][2]; path_bytes = path.encode("utf-8")
            raw_digest.update(len(path_bytes).to_bytes(4, "big")); raw_digest.update(path_bytes)
            raw_digest.update(len(payload).to_bytes(8, "big")); raw_digest.update(payload)
        if (
            row["actor_id"] != expected
            or source["actor_id"] != expected
            or source["commit"] != commit
            or type(source["allowlist_count"]) is not int
            or source["allowlist_count"] != len(paths)
            or source["blob_rows"] != blob_rows
            or source["source_identity_sha256"] != raw_digest.hexdigest()
            or source["path_registry_sha256"] != sha256(_canonical_json(paths)).hexdigest()
            or source["git_blob_manifest_sha256"] != sha256(_canonical_json(blob_rows)).hexdigest()
            or type(row["command"]) is not list
            or not row["command"]
            or any(type(item) is not str or not item for item in row["command"])
        ):
            _fail("REJECT_Q05B_ARTIFACT_ACTOR", f"{expected} evidence differs")
        _hex(source["source_identity_sha256"], 32, "source identity")
        _hex(row["runtime_identity_sha256"], 32, "runtime identity")
        result.append(row)
    return tuple(result)


def _validate_actor_control_v1(
    value: object, actor_id: str, command: object
) -> dict[str, object]:
    control = _object(
        value,
        {
            "actor_id", "command_sha256", "completion_evidence",
            "container_id", "explicit_remove_exit_code",
            "final_resource_transcript", "held_final_resource",
            "mount_registry_sha256", "post_exit_inspect_sha256",
            "release_evidence", "schema_version", "stdout_hex",
            "stdout_sha256",
        },
        "actor control evidence",
    )
    stdout = _hex_any(control["stdout_hex"], "actor control stdout")
    if (
        control["schema_version"]
        != "hegel-phase3a-q05b-held-actor-complete-evidence/1"
        or control["actor_id"] != actor_id
        or type(control["container_id"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", control["container_id"]) is None
        or control["command_sha256"] != sha256(_canonical_json(command)).hexdigest()
        or type(control["mount_registry_sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", control["mount_registry_sha256"]) is None
        or type(control["post_exit_inspect_sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", control["post_exit_inspect_sha256"]) is None
        or control["stdout_sha256"] != sha256(stdout).hexdigest()
        or type(control["explicit_remove_exit_code"]) is not int
        or control["explicit_remove_exit_code"] != 0
    ):
        _fail("REJECT_Q05B_ARTIFACT_ACTOR", f"{actor_id} control binding differs")
    for name, expected_files in (
        ("completion_evidence", ("actor.stdout", "done", "exit-code")),
        ("release_evidence", ("actor.stdout", "done", "exit-code", "release")),
    ):
        item = _object(
            control[name],
            {
                "actor_id", "actor_stdout_hex", "file_rows", "manifest_sha256",
                "root_device", "root_inode", "root_mode", "schema_version",
            },
            name,
        )
        body = dict(item); observed = body.pop("manifest_sha256")
        rows = item["file_rows"]
        if (
            item["schema_version"] != "hegel-phase3a-q05b-held-control-evidence/1"
            or item["actor_id"] != actor_id
            or item["actor_stdout_hex"] != control["stdout_hex"]
            or type(item["root_device"]) is not int
            or type(item["root_inode"]) is not int
            or item["root_mode"] not in (0o700, 0o555)
            or type(rows) is not list
            or [row[0] if type(row) is list and row else None for row in rows]
            != list(expected_files)
            or any(
                type(row) is not list
                or len(row) != 11
                or any(type(row[index]) is not int for index in range(1, 10))
                or row[3] != 1
                or row[6] != 0o444
                or type(row[10]) is not str
                or re.fullmatch(r"[0-9a-f]{64}", row[10]) is None
                for row in rows
            )
            or observed != sha256(_canonical_json(body)).hexdigest()
        ):
            _fail("REJECT_Q05B_ARTIFACT_ACTOR", f"{actor_id} {name} differs")
    held = _object(
        control["held_final_resource"],
        {
            "actor_child_complete_held", "anchored_collection",
            "completion_manifest_sha256", "fresh_after_done_collection",
            "sample_preimage", "sample_root",
        },
        "held final resource",
    )
    if (
        held["actor_child_complete_held"] is not True
        or held["anchored_collection"] is not True
        or held["fresh_after_done_collection"] is not True
        or held["completion_manifest_sha256"]
        != control["completion_evidence"]["manifest_sha256"]
        or type(held["sample_preimage"]) is not dict
        or held["sample_root"]
        != _json_root(
            "HEGEL/Q05B/ACTUAL/HELD_RESOURCE_SAMPLE/V1",
            held["sample_preimage"],
        ).hex()
    ):
        _fail("REJECT_Q05B_ARTIFACT_ACTOR", f"{actor_id} held sample differs")
    return control


def _decoded_json_value(payload: bytes, name: str) -> object:
    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if type(key) is not str or key in result:
                _fail("REJECT_Q05B_ARTIFACT_JSON", f"{name} duplicate key")
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8", "strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: _fail(
                "REJECT_Q05B_ARTIFACT_JSON", f"{name} non-finite {token}"
            ),
            parse_float=lambda token: _fail(
                "REJECT_Q05B_ARTIFACT_JSON", f"{name} float {token}"
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        _fail("REJECT_Q05B_ARTIFACT_JSON", f"{name}: {error}")


def _strict_seccomp_json_object_v1(
    payload: bytes,
    name: str,
) -> dict[str, object]:
    """Decode one seccomp profile without JSON alias or extension semantics."""

    value = _decoded_json_value(payload, name)
    if type(value) is not dict:
        _fail(
            "REJECT_Q05B_ARTIFACT_ISOLATION",
            f"{name} must be one JSON object",
        )
    return value


def _json_value_type_exact_v1(observed: object, expected: object) -> bool:
    """Compare JSON trees without Python's bool/int equality alias."""

    if type(observed) is not type(expected):
        return False
    if type(expected) is dict:
        assert type(observed) is dict
        return set(observed) == set(expected) and all(
            _json_value_type_exact_v1(observed[key], expected[key])
            for key in expected
        )
    if type(expected) is list:
        assert type(observed) is list
        return len(observed) == len(expected) and all(
            _json_value_type_exact_v1(item, expected_item)
            for item, expected_item in zip(observed, expected, strict=True)
        )
    return observed == expected


def _validate_inspect_security_options_v1(
    observed: object,
    command_security_options: list[str],
    sealed_policy_payload: bytes,
    sealed_policy_evidence: Mapping[str, object],
    name: str,
) -> None:
    """Bind Docker 29's inline seccomp inspect value to sealed source bytes.

    Docker consumes ``seccomp=/sealed/path`` but reports the loaded profile as
    ``seccomp={...}``.  The inspect suffix is evidence text, never a path to
    open.  Dict key order and whitespace are normalized; list order and every
    JSON value type remain exact.
    """

    if (
        type(command_security_options) is not list
        or len(command_security_options) != 2
        or any(type(item) is not str for item in command_security_options)
        or command_security_options[0] != "no-new-privileges"
        or type(sealed_policy_evidence) is not dict
        or type(sealed_policy_evidence.get("absolute_path")) is not str
        or command_security_options[1]
        != f"seccomp={sealed_policy_evidence['absolute_path']}"
        or type(sealed_policy_payload) is not bytes
        or sealed_policy_evidence.get("payload_sha256")
        != sha256(sealed_policy_payload).hexdigest()
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_ISOLATION",
            f"{name} sealed seccomp command binding differs",
        )
    if (
        type(observed) is not list
        or len(observed) != 2
        or any(type(item) is not str for item in observed)
        or observed[0] != "no-new-privileges"
        or not observed[1].startswith("seccomp=")
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_ISOLATION",
            f"{name} security option registry differs",
        )
    inline_text = observed[1].removeprefix("seccomp=")
    try:
        inline_payload = inline_text.encode("utf-8", "strict")
    except UnicodeError as error:
        _fail("REJECT_Q05B_ARTIFACT_JSON", f"{name} inline seccomp: {error}")
    sealed_value = _strict_seccomp_json_object_v1(
        sealed_policy_payload,
        f"{name} sealed seccomp",
    )
    inline_value = _strict_seccomp_json_object_v1(
        inline_payload,
        f"{name} inline seccomp",
    )
    sealed_canonical = _canonical_json(sealed_value)
    inline_canonical = _canonical_json(inline_value)
    if (
        not _json_value_type_exact_v1(inline_value, sealed_value)
        or inline_canonical != sealed_canonical
        or sha256(inline_canonical).digest() != sha256(sealed_canonical).digest()
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_ISOLATION",
            f"{name} inline seccomp differs from sealed policy",
        )


_LIVE_SAMPLE_BASE_KEYS: Final = {
    "anchored_collection", "captured_while_running", "cgroup_directory_device",
    "cgroup_directory_inode", "cgroup_path", "cgroup_payload_rows",
    "cgroup_sha256", "container_id", "cpuset_cpus", "inspect_after_payload_hex",
    "inspect_after_sha256", "inspect_payload_hex", "inspect_sha256",
    "memory_current_bytes", "memory_events", "memory_limit_bytes",
    "memory_peak_bytes", "memory_swap_limit_bytes", "mount_command_sha256",
    "mount_registry_sha256", "nofile_hard", "nofile_soft", "oom_killed",
    "pids_current", "pids_limit", "pids_peak", "proc_cgroup_payload_hex",
    "proc_cgroup_sha256", "proc_limits_payload_hex", "proc_limits_sha256",
    "proc_pid_directory_device", "proc_pid_directory_inode", "role_id",
    "sample_duration_ns", "sample_monotonic_ns", "sample_ordinal",
    "schema_version",
}


def _decimal_line(payload: bytes, name: str) -> int:
    if re.fullmatch(rb"0\n|[1-9][0-9]*\n", payload) is None:
        _fail("REJECT_Q05B_ARTIFACT_RESOURCE", f"{name} differs")
    return int(payload)


def _replay_live_sample_v1(
    value: object,
    role_row: list[object],
    ordinal: int,
    held: bool,
) -> dict[str, object]:
    expected_keys = set(_LIVE_SAMPLE_BASE_KEYS)
    if held:
        expected_keys |= {
            "actor_child_complete_held", "completion_manifest_sha256",
            "fresh_after_done_collection",
        }
    sample = _object(value, expected_keys, f"live sample {ordinal}")
    container_id = sample["container_id"]
    if (
        type(container_id) is not str
        or re.fullmatch(r"[0-9a-f]{64}", container_id) is None
        or sample["schema_version"]
        != "hegel-phase3a-q05b-live-container-resource-transcript/1"
        or sample["role_id"] != role_row[0]
        or type(sample["role_id"]) is not int
        or sample["captured_while_running"] is not True
        or sample["anchored_collection"] is not True
        or sample["sample_ordinal"] != ordinal
        or type(sample["sample_ordinal"]) is not int
        or type(sample["sample_monotonic_ns"]) is not int
        or sample["sample_monotonic_ns"] < 0
        or type(sample["sample_duration_ns"]) is not int
        or sample["sample_duration_ns"] < 0
        or type(sample["proc_pid_directory_device"]) is not int
        or sample["proc_pid_directory_device"] < 1
        or type(sample["proc_pid_directory_inode"]) is not int
        or sample["proc_pid_directory_inode"] < 1
    ):
        _fail("REJECT_Q05B_ARTIFACT_RESOURCE", "live sample header differs")
    raw: dict[str, bytes] = {}
    for payload_field, digest_field in (
        ("inspect_payload_hex", "inspect_sha256"),
        ("inspect_after_payload_hex", "inspect_after_sha256"),
        ("proc_cgroup_payload_hex", "proc_cgroup_sha256"),
        ("proc_limits_payload_hex", "proc_limits_sha256"),
    ):
        raw[payload_field] = _hex_any(sample[payload_field], payload_field)
        if sample[digest_field] != sha256(raw[payload_field]).hexdigest():
            _fail("REJECT_Q05B_ARTIFACT_RESOURCE", f"{digest_field} differs")
    cgroup_rows = sample["cgroup_payload_rows"]
    expected_names = (
        "memory.current", "memory.events", "memory.peak", "pids.current",
        "pids.peak",
    )
    if type(cgroup_rows) is not list or [row[0] if type(row) is list and row else None for row in cgroup_rows] != list(expected_names):
        _fail("REJECT_Q05B_ARTIFACT_RESOURCE", "cgroup payload registry differs")
    cgroup_payloads: dict[str, bytes] = {}
    digest = sha256()
    if type(sample["cgroup_path"]) is not str or container_id not in sample["cgroup_path"] or ".." in sample["cgroup_path"].split("/"):
        _fail("REJECT_Q05B_ARTIFACT_RESOURCE", "cgroup path differs")
    path_bytes = sample["cgroup_path"].encode("ascii", "strict")
    digest.update(len(path_bytes).to_bytes(4, "big")); digest.update(path_bytes)
    for field in ("cgroup_directory_device", "cgroup_directory_inode"):
        if type(sample[field]) is not int or sample[field] < 1:
            _fail("REJECT_Q05B_ARTIFACT_RESOURCE", "cgroup identity differs")
        digest.update(sample[field].to_bytes(8, "big"))
    for row, name in zip(cgroup_rows, expected_names, strict=True):
        if type(row) is not list or len(row) != 2:
            _fail("REJECT_Q05B_ARTIFACT_RESOURCE", "cgroup row differs")
        payload = _hex_any(row[1], f"cgroup {name}")
        cgroup_payloads[name] = payload
        name_bytes = name.encode("ascii")
        digest.update(len(name_bytes).to_bytes(4, "big")); digest.update(name_bytes)
        digest.update(len(payload).to_bytes(8, "big")); digest.update(payload)
    if sample["cgroup_sha256"] != digest.hexdigest():
        _fail("REJECT_Q05B_ARTIFACT_RESOURCE", "cgroup transcript digest differs")
    proc_match = re.fullmatch(rb"0::(/[^\r\n]*)\n", raw["proc_cgroup_payload_hex"])
    if proc_match is None or proc_match.group(1).decode("ascii") != sample["cgroup_path"]:
        _fail("REJECT_Q05B_ARTIFACT_RESOURCE", "proc cgroup binding differs")
    limits = raw["proc_limits_payload_hex"].decode("ascii", "strict")
    nofile = [line for line in limits.splitlines() if line.startswith("Max open files")]
    if (
        len(nofile) != 1
        or re.fullmatch(r"Max open files +256 +256 +files *", nofile[0])
        is None
    ):
        _fail("REJECT_Q05B_ARTIFACT_RESOURCE", "proc nofile limit differs")
    events: dict[str, int] = {}
    for line in cgroup_payloads["memory.events"].splitlines():
        match = re.fullmatch(rb"([a-z_]+) (0|[1-9][0-9]*)", line)
        if match is None or match.group(1).decode() in events:
            _fail("REJECT_Q05B_ARTIFACT_RESOURCE", "memory events differ")
        events[match.group(1).decode()] = int(match.group(2))
    required_events = {"low", "high", "max", "oom", "oom_kill", "oom_group_kill"}
    memory_current = _decimal_line(cgroup_payloads["memory.current"], "memory.current")
    memory_peak = _decimal_line(cgroup_payloads["memory.peak"], "memory.peak")
    pids_current = _decimal_line(cgroup_payloads["pids.current"], "pids.current")
    pids_peak = _decimal_line(cgroup_payloads["pids.peak"], "pids.peak")
    if not required_events <= set(events) or any(events[name] for name in ("oom", "oom_kill", "oom_group_kill")):
        _fail("REJECT_Q05B_ARTIFACT_RESOURCE", "OOM event differs")
    expected_values = {
        "cpuset_cpus": role_row[2], "memory_limit_bytes": 14 * 1024**3,
        "memory_swap_limit_bytes": 14 * 1024**3, "pids_limit": 128,
        "nofile_soft": 256, "nofile_hard": 256, "oom_killed": False,
        "memory_current_bytes": memory_current, "memory_peak_bytes": memory_peak,
        "pids_current": pids_current, "pids_peak": pids_peak,
        "memory_events": [[name, events[name]] for name in sorted(events)],
    }
    if any(sample[field] != expected for field, expected in expected_values.items()) or memory_current > memory_peak or memory_peak > 14 * 1024**3 or pids_current > pids_peak or pids_peak > 128:
        _fail("REJECT_Q05B_ARTIFACT_RESOURCE", "resource values differ")
    before = _decoded_json_value(raw["inspect_payload_hex"], "live inspect before")
    after = _decoded_json_value(raw["inspect_after_payload_hex"], "live inspect after")
    for document in (before, after):
        if type(document) is not list or len(document) != 1 or type(document[0]) is not dict or document[0].get("Id") != container_id or type(document[0].get("State")) is not dict or document[0]["State"].get("Running") is not True:
            _fail("REJECT_Q05B_ARTIFACT_RESOURCE", "live Docker inspect differs")
    if before[0]["State"].get("Pid") != after[0]["State"].get("Pid"):
        _fail("REJECT_Q05B_ARTIFACT_RESOURCE", "live Docker PID differs")
    if held and (
        sample["actor_child_complete_held"] is not True
        or sample["fresh_after_done_collection"] is not True
        or type(sample["completion_manifest_sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", sample["completion_manifest_sha256"]) is None
    ):
        _fail("REJECT_Q05B_ARTIFACT_RESOURCE", "held final sample differs")
    return sample


def _validate_resources_v1(value: object, config: dict[str, object]) -> list[dict[str, object]]:
    keys = {
        "actor_exit_code", "container_id", "continuous_sampling_through_child_completion",
        "explicit_remove_admitted_after_this_transcript", "final_memory_peak_bytes",
        "final_pids_peak", "fresh_held_final_before_release", "live_sample_objects",
        "maximum_inter_sample_gap_ns", "oom_killed", "peak_scope",
        "post_exit_inspect_hex", "post_exit_inspect_sha256",
        "post_exit_zero_and_no_oom", "post_release_wrapper_only_exits",
        "role_id", "sample_count", "sample_rows", "sampling_interval_milliseconds",
        "schema_version", "transcript_sha256",
    }
    roles = config.get("resource_roles")
    if type(value) is not list or len(value) != 3 or type(roles) is not list or len(roles) != 3:
        _fail("REJECT_Q05B_ARTIFACT_RESOURCE", "three resource transcripts required")
    result = []
    for expected_role, raw, role_row in zip(range(1, 4), value, roles, strict=True):
        if type(role_row) is not list or len(role_row) != 8 or role_row[0] != expected_role:
            _fail("REJECT_Q05B_ARTIFACT_RESOURCE", "resource role policy differs")
        row = _object(raw, keys, f"resource[{expected_role}]")
        samples = row["live_sample_objects"]
        if type(samples) is not list or not samples:
            _fail("REJECT_Q05B_ARTIFACT_RESOURCE", "live samples differ")
        replayed = [
            _replay_live_sample_v1(item, role_row, index, index == len(samples))
            for index, item in enumerate(samples, start=1)
        ]
        maximum_gap = 0
        for prior, current in zip(replayed, replayed[1:]):
            gap = current["sample_monotonic_ns"] - (prior["sample_monotonic_ns"] + prior["sample_duration_ns"])
            if gap < 0 or gap > 250_000_000:
                _fail("REJECT_Q05B_ARTIFACT_RESOURCE", "sampling gap differs")
            maximum_gap = max(maximum_gap, gap)
        sample_rows = [[index, sample["inspect_sha256"], sample["proc_cgroup_sha256"], sample["cgroup_sha256"], sample["proc_limits_sha256"], sample["memory_peak_bytes"], sample["pids_peak"], sample["memory_events"], sample.get("actor_child_complete_held", False), sample.get("completion_manifest_sha256"), sample["sample_monotonic_ns"], sample["sample_duration_ns"]] for index, sample in enumerate(replayed, start=1)]
        post_payload = _hex_any(row["post_exit_inspect_hex"], "post-exit inspect")
        post = _decoded_json_value(post_payload, "post-exit inspect")
        if type(post) is not list or len(post) != 1 or type(post[0]) is not dict or post[0].get("Id") != row["container_id"] or type(post[0].get("State")) is not dict or type(post[0].get("HostConfig")) is not dict or post[0]["State"].get("Running") is not False or post[0]["State"].get("OOMKilled") is not False or post[0]["State"].get("ExitCode") != 0 or post[0]["HostConfig"].get("AutoRemove") is not False:
            _fail("REJECT_Q05B_ARTIFACT_RESOURCE", "post-exit Docker state differs")
        body = dict(row); observed_root = body.pop("transcript_sha256")
        if (
            row["schema_version"] != "hegel-phase3a-q05b-final-container-resource-transcript/1"
            or row["container_id"] != replayed[0]["container_id"]
            or row["role_id"] != expected_role or type(row["role_id"]) is not int
            or row["sampling_interval_milliseconds"] != 250
            or row["continuous_sampling_through_child_completion"] is not True
            or row["fresh_held_final_before_release"] is not True
            or row["post_release_wrapper_only_exits"] is not True
            or row["post_exit_zero_and_no_oom"] is not True
            or row["peak_scope"] != "CHILD_PLUS_WRAPPER_THROUGH_HELD_FINAL_SAMPLE"
            or row["actor_exit_code"] != 0 or type(row["actor_exit_code"]) is not int
            or row["oom_killed"] is not False
            or row["sample_count"] != len(replayed) or type(row["sample_count"]) is not int
            or row["sample_rows"] != sample_rows
            or row["maximum_inter_sample_gap_ns"] != maximum_gap
            or type(row["maximum_inter_sample_gap_ns"]) is not int
            or row["final_memory_peak_bytes"] != replayed[-1]["memory_peak_bytes"]
            or row["final_pids_peak"] != replayed[-1]["pids_peak"]
            or row["post_exit_inspect_sha256"] != sha256(post_payload).hexdigest()
            or row["explicit_remove_admitted_after_this_transcript"] is not True
            or observed_root != sha256(_canonical_json(body)).hexdigest()
        ):
            _fail("REJECT_Q05B_ARTIFACT_RESOURCE", f"resource {expected_role} differs")
        result.append(row)
    return result


def _docker_run_principal_tokens_v1(
    command: object,
    name: str,
) -> tuple[str, list[list[str]], list[str]]:
    """Replay the label-bearing Docker-run principal without admission state.

    Causal namespace/name/value validation is deliberately deferred to the
    post-admission authority join.  This helper freezes only the production
    argv encoding: one ``--name`` followed immediately by the three ordered
    reserved ``--label=key=value`` tokens.
    """

    if (
        type(command) is not list
        or not command
        or any(type(item) is not str or not item for item in command)
    ):
        _fail("REJECT_Q05B_ARTIFACT_ISOLATION", f"{name} command differs")
    name_indexes = [index for index, item in enumerate(command) if item == "--name"]
    label_indexes = [
        index
        for index, item in enumerate(command)
        if item == "--label" or item.startswith("--label=")
    ]
    if (
        len(name_indexes) != 1
        or name_indexes[0] + 4 >= len(command)
        or label_indexes != list(range(name_indexes[0] + 2, name_indexes[0] + 5))
        or any(command[index] == "--label" for index in label_indexes)
    ):
        _fail("REJECT_Q05B_ARTIFACT_ISOLATION", f"{name} Docker principal differs")
    container_name = command[name_indexes[0] + 1]
    if (
        re.fullmatch(
            r"hegel-q05b-[0-9a-f]{64}-(?:rust-test|rust-release|python|rust|host)",
            container_name,
        )
        is None
    ):
        _fail("REJECT_Q05B_ARTIFACT_ISOLATION", f"{name} Docker name differs")
    labels: list[list[str]] = []
    for index, expected_key in zip(
        label_indexes, _admission.DOCKER_RESERVED_LABEL_KEYS, strict=True
    ):
        payload = command[index].removeprefix("--label=")
        key, separator, value = payload.partition("=")
        if separator != "=" or key != expected_key or not value:
            _fail(
                "REJECT_Q05B_ARTIFACT_ISOLATION",
                f"{name} Docker label differs",
            )
        labels.append([key, value])
    normalized = command[: name_indexes[0] + 2] + command[name_indexes[0] + 5 :]
    return container_name, labels, normalized


def _command_mount_registry_v1(
    command: object, role_id: int, actor_id: str, config: dict[str, object]
) -> tuple[str, str]:
    if type(command) is not list or not command or any(type(item) is not str or not item for item in command):
        _fail("REJECT_Q05B_ARTIFACT_ACTOR", f"{actor_id} command differs")
    docker = config.get("docker")
    images = config.get("images")
    configured = config.get("actor_commands")
    if type(docker) is not dict or type(images) is not dict or type(configured) is not dict:
        _fail("REJECT_Q05B_ARTIFACT_ACTOR", "Docker command policy differs")
    actor_key = {1: "python", 2: "rust", 3: "trusted_host"}[role_id]
    image_key = {1: "python_endpoint", 2: "rust_runtime", 3: "trusted_host"}[role_id]
    cpuset = {1: "0-11", 2: "12-23", 3: "0-11"}[role_id]
    name, _labels, normalized = _docker_run_principal_tokens_v1(
        command, actor_id
    )
    if "--mount" not in normalized:
        _fail("REJECT_Q05B_ARTIFACT_ACTOR", "Docker mount registry is absent")
    first_mount = normalized.index("--mount")
    prefix = normalized[:first_mount]
    if len(prefix) != 29 or re.fullmatch(r"--cidfile=/[^,]+\.cid", prefix[5]) is None or re.fullmatch(r"--security-opt=seccomp=/[^,]+", prefix[11]) is None or re.fullmatch(r"--user=[1-9][0-9]*:[1-9][0-9]*", prefix[20]) is None:
        _fail("REJECT_Q05B_ARTIFACT_ACTOR", f"{actor_id} Docker prefix differs")
    expected_prefix = [docker["executable"], f"--host={docker['host']}", "run", "--name", name, prefix[5], f"--pull={docker['pull_policy']}", f"--network={docker['network']}", "--read-only", f"--cap-drop={docker['cap_drop']}", "--security-opt=no-new-privileges", prefix[11], f"--ipc={docker['ipc']}", "--cgroupns=private", f"--pids-limit={docker['pids_limit']}", f"--ulimit=nofile={docker['nofile_ulimit']}", f"--memory={docker['memory']}", f"--memory-swap={docker['memory_swap']}", f"--cpuset-cpus={cpuset}", f"--tmpfs={docker['runtime_tmpfs']}", prefix[20], "-e", "HOME=/tmp", "-e", "LANG=C.UTF-8", "-e", "LC_ALL=C.UTF-8", "-e", "TZ=UTC"]
    if prefix != expected_prefix:
        _fail("REJECT_Q05B_ARTIFACT_ACTOR", f"{actor_id} Docker prefix differs")
    marker = "hegel-q05b-held-actor"
    if marker not in command or command[command.index(marker) - 3:command.index(marker)] != ["/bin/sh", "-ceu", command[command.index(marker) - 1]]:
        _fail("REJECT_Q05B_ARTIFACT_ACTOR", f"{actor_id} held wrapper differs")
    wrapper = command[command.index(marker) - 1]
    held_policy = config.get("held_actor_protocol")
    if type(held_policy) is not dict or sha256(wrapper.encode()).hexdigest() != held_policy.get("wrapper_script_sha256"):
        _fail("REJECT_Q05B_ARTIFACT_ACTOR", "held wrapper identity differs")
    payload = command[command.index(marker) + 1:]
    expected_payload = list(configured[actor_key])
    if role_id == 3:
        expected_payload[-3] = command[command.index(marker) + 1:][-3]
        expected_payload[-1] = command[command.index(marker) + 1:][-1]
        for value in (expected_payload[-3], expected_payload[-1]):
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                _fail("REJECT_Q05B_ARTIFACT_ACTOR", "host command identity differs")
    if payload != expected_payload:
        _fail("REJECT_Q05B_ARTIFACT_ACTOR", f"{actor_id} payload command differs")
    mounts: dict[str, tuple[str, bool]] = {}
    for index, item in enumerate(command):
        if item != "--mount":
            continue
        if index + 1 >= len(command):
            _fail("REJECT_Q05B_ARTIFACT_ACTOR", "Docker mount is truncated")
        match = re.fullmatch(r"type=bind,src=([^,]+),dst=([^,]+)(,readonly)?", command[index + 1])
        if match is None or match.group(2) in mounts:
            _fail("REJECT_Q05B_ARTIFACT_ACTOR", "Docker mount differs")
        source, destination, readonly = match.groups()
        if not source.startswith("/") or ".." in source.split("/") or "docker.sock" in source:
            _fail("REJECT_Q05B_ARTIFACT_ACTOR", "Docker mount source differs")
        mounts[destination] = (source, readonly is None)
    expected_destinations = {
        1: ("/control", "/output", "/snapshot"),
        2: ("/control", "/output", "/runtime/hegel-q1-archive-projection-oracle"),
        3: ("/control", "/inputs/python", "/inputs/rust", "/inputs/stdout/manifest.json", "/inputs/stdout/python.stdout", "/inputs/stdout/rust.stdout", "/snapshot", "/staging"),
    }[role_id]
    rows = [[destination, *mounts[destination]] for destination in sorted(mounts)] if set(mounts) == set(expected_destinations) else []
    if tuple(sorted(mounts)) != expected_destinations:
        _fail("REJECT_Q05B_ARTIFACT_ACTOR", "Docker mount destination registry differs")
    mount_order = {
        1: ("/snapshot", "/output", "/control"),
        2: ("/runtime/hegel-q1-archive-projection-oracle", "/output", "/control"),
        3: ("/snapshot", "/inputs/python", "/inputs/rust", "/inputs/stdout/python.stdout", "/inputs/stdout/rust.stdout", "/inputs/stdout/manifest.json", "/control", "/staging"),
    }[role_id]
    observed_order = tuple(
        re.fullmatch(r"type=bind,src=([^,]+),dst=([^,]+)(,readonly)?", command[index + 1]).group(2)
        for index, item in enumerate(command)
        if item == "--mount"
    )
    if observed_order != mount_order:
        _fail("REJECT_Q05B_ARTIFACT_ACTOR", "Docker mount order differs")
    image_index = command.index(images[image_key]) if command.count(images[image_key]) == 1 else -1
    if image_index < 0 or command[image_index + 1:command.index(marker) - 3] != []:
        _fail("REJECT_Q05B_ARTIFACT_ACTOR", "Docker image/payload boundary differs")
    container_argv = command[image_index + 1:]
    security_options = [prefix[10].removeprefix("--security-opt="), prefix[11].removeprefix("--security-opt=")]
    runtime_policy = config.get("runtime_command_inspect_policy")
    if type(runtime_policy) is not dict or type(runtime_policy.get("environment_rows")) is not list or type(runtime_policy.get("working_directory_rows")) is not list:
        _fail("REJECT_Q05B_ARTIFACT_ACTOR", "runtime command policy differs")
    environment_policy_row = runtime_policy["environment_rows"][role_id - 1]
    working_policy_row = runtime_policy["working_directory_rows"][role_id - 1]
    if environment_policy_row[:2] != [role_id, actor_id] or working_policy_row[:2] != [role_id, actor_id] or type(environment_policy_row[2]) is not list or type(working_policy_row[2]) is not str:
        _fail("REJECT_Q05B_ARTIFACT_ACTOR", "runtime role policy differs")
    environments = environment_policy_row[2]
    working_directory = working_policy_row[2]
    mount_tail: list[str] = []
    for destination in mount_order:
        source, writable = mounts[destination]
        mount_tail.extend(["--mount", f"type=bind,src={source},dst={destination}" + ("" if writable else ",readonly")])
    if working_directory:
        mount_tail.extend(["-w", working_directory])
    mount_tail.extend([images[image_key], *container_argv])
    if normalized[first_mount:] != mount_tail:
        _fail("REJECT_Q05B_ARTIFACT_ACTOR", "Docker argv has extra/reordered fields")
    command_sha = sha256(_canonical_json(command)).hexdigest()
    body = {"command_sha256": command_sha, "container_argv": container_argv, "environment_rows": environments, "mount_rows": rows, "role_id": role_id, "schema_version": "hegel-phase3a-q05b-sealed-command-mount-registry/1", "security_options": security_options, "working_directory": working_directory}
    return command_sha, sha256(_canonical_json(body)).hexdigest()


def _replay_control_identity_v1(
    value: object, actor_id: str, stdout: bytes, released: bool
) -> dict[str, object]:
    item = _object(value, {"actor_id", "actor_stdout_hex", "file_rows", "manifest_sha256", "root_device", "root_inode", "root_mode", "schema_version"}, "held control identity")
    names = ("actor.stdout", "done", "exit-code") + (("release",) if released else ())
    payloads = {
        "actor.stdout": stdout,
        "done": b"ACTOR_COMPLETE_HELD\n",
        "exit-code": b"0\n",
        "release": b"HOST_FINAL_SAMPLE_SEALED\n",
    }
    rows = item["file_rows"]
    body = dict(item); observed = body.pop("manifest_sha256")
    if (
        item["schema_version"] != "hegel-phase3a-q05b-held-control-evidence/1"
        or item["actor_id"] != actor_id
        or item["actor_stdout_hex"] != stdout.hex()
        or type(item["root_device"]) is not int or type(item["root_inode"]) is not int
        or item["root_mode"] != (0o555 if released else 0o700)
        or type(rows) is not list or [row[0] if type(row) is list and row else None for row in rows] != list(names)
        or observed != sha256(_canonical_json(body)).hexdigest()
    ):
        _fail("REJECT_Q05B_ARTIFACT_ACTOR", f"{actor_id} control identity differs")
    for row, name in zip(rows, names, strict=True):
        payload = payloads[name]
        if type(row) is not list or len(row) != 11 or any(type(row[index]) is not int for index in range(1, 10)) or row[3] != 1 or row[6] != 0o444 or row[7] != len(payload) or row[10] != sha256(payload).hexdigest():
            _fail("REJECT_Q05B_ARTIFACT_ACTOR", f"{actor_id} control file differs")
    return item


def _validate_seccomp_evidence_v1(
    value: object, relative_path: str, payload: bytes
) -> dict[str, object]:
    evidence = _object(value, {"absolute_path", "file_ctime_ns", "file_device", "file_gid", "file_inode", "file_mode", "file_mtime_ns", "file_nlink", "file_size", "file_uid", "manifest_sha256", "payload_sha256", "schema_version", "snapshot_relative_path"}, "seccomp evidence")
    body = dict(evidence); observed = body.pop("manifest_sha256")
    if evidence["schema_version"] != "hegel-phase3a-q05b-sealed-policy-file/1" or evidence["snapshot_relative_path"] != relative_path or type(evidence["absolute_path"]) is not str or not evidence["absolute_path"].startswith("/") or ".." in evidence["absolute_path"].split("/") or any(type(evidence[field]) is not int for field in ("file_ctime_ns", "file_device", "file_gid", "file_inode", "file_mode", "file_mtime_ns", "file_nlink", "file_size", "file_uid")) or evidence["file_nlink"] != 1 or evidence["file_mode"] != 0o444 or evidence["file_size"] != len(payload) or evidence["payload_sha256"] != sha256(payload).hexdigest() or observed != sha256(_canonical_json(body)).hexdigest():
        _fail("REJECT_Q05B_ARTIFACT_ISOLATION", "seccomp evidence differs")
    return evidence


def _validate_live_inspect_policy_v1(
    sample: dict[str, object], command: list[str], role_id: int,
    config: dict[str, object], pinned_image_id: str,
    runtime_seccomp_payload: bytes,
    runtime_seccomp_evidence: Mapping[str, object],
) -> None:
    expected_name, command_labels, _normalized = _docker_run_principal_tokens_v1(
        command, f"role {role_id}"
    )
    command_label_map = dict(command_labels)
    docker = config["docker"]
    images = config["images"]
    image = images[{1: "python_endpoint", 2: "rust_runtime", 3: "trusted_host"}[role_id]]
    image_index = command.index(image)
    container_argv = command[image_index + 1:]
    workdir = "/snapshot" if role_id in (1, 3) else ""
    user = next(item.removeprefix("--user=") for item in command if item.startswith("--user="))
    security = [item.removeprefix("--security-opt=") for item in command if item.startswith("--security-opt=")]
    runtime_policy = config["runtime_command_inspect_policy"]
    environment_rows = [tuple(row) for row in runtime_policy["environment_rows"][role_id - 1][2]]
    expected_mounts: dict[str, tuple[str, bool]] = {}
    for index, item in enumerate(command):
        if item == "--mount":
            match = re.fullmatch(r"type=bind,src=([^,]+),dst=([^,]+)(,readonly)?", command[index + 1])
            expected_mounts[match.group(2)] = (match.group(1), match.group(3) is None)
    for field in ("inspect_payload_hex", "inspect_after_payload_hex"):
        decoded = _decoded_json_value(_hex_any(sample[field], field), field)
        if type(decoded) is not list or len(decoded) != 1 or type(decoded[0]) is not dict:
            _fail("REJECT_Q05B_ARTIFACT_RESOURCE", "Docker inspect shape differs")
        document = decoded[0]
        state = document.get("State"); host = document.get("HostConfig"); container = document.get("Config"); mounts = document.get("Mounts")
        if type(state) is not dict or type(host) is not dict or type(container) is not dict or type(mounts) is not list:
            _fail("REJECT_Q05B_ARTIFACT_RESOURCE", "Docker inspect sections differ")
        observed_env: list[tuple[str, str]] = []
        if type(container.get("Env")) is list:
            for item in container["Env"]:
                if type(item) is not str or "=" not in item:
                    _fail("REJECT_Q05B_ARTIFACT_RESOURCE", "Docker environment row differs")
                observed_env.append(tuple(item.split("=", 1)))
        observed_mounts: dict[str, tuple[object, object]] = {}
        for mount in mounts:
            if type(mount) is not dict or mount.get("Type") != "bind" or type(mount.get("Destination")) is not str or mount["Destination"] in observed_mounts:
                _fail("REJECT_Q05B_ARTIFACT_RESOURCE", "Docker inspect mount row differs")
            observed_mounts[mount["Destination"]] = (mount.get("Source"), mount.get("RW"))
        observed_labels = container.get("Labels")
        _validate_inspect_security_options_v1(
            host.get("SecurityOpt"),
            security,
            runtime_seccomp_payload,
            runtime_seccomp_evidence,
            f"role {role_id} {field}",
        )
        if (
            document.get("Id") != sample["container_id"]
            or document.get("Name") != f"/{expected_name}"
            or document.get("Image") != pinned_image_id
            or state.get("Running") is not True or state.get("OOMKilled") is not False
            or type(state.get("Pid")) is not int or state["Pid"] < 1
            or container.get("Image") != image or container.get("User") != user
            or container.get("Entrypoint") is not None or container.get("Cmd") != container_argv
            or container.get("WorkingDir") != workdir or tuple(sorted(observed_env)) != tuple(environment_rows)
            or type(observed_labels) is not dict
            or any(
                type(key) is not str or type(value) is not str
                for key, value in observed_labels.items()
            )
            or any(observed_labels.get(key) != value for key, value in command_label_map.items())
            or any(
                key in _admission.DOCKER_RESERVED_LABEL_KEYS
                and key not in command_label_map
                for key in observed_labels
            )
            or host.get("AutoRemove") is not False or host.get("NetworkMode") != docker["network"]
            or host.get("ReadonlyRootfs") is not True or host.get("CapDrop") != [docker["cap_drop"]]
            or host.get("IpcMode") != docker["ipc"]
            or host.get("PidMode") != "" or host.get("CgroupnsMode") != "private" or host.get("UsernsMode") != ""
            or host.get("Privileged") is not False or host.get("Devices") != [] or host.get("DeviceRequests") is not None
            or host.get("CpusetCpus") != sample["cpuset_cpus"]
            or host.get("Memory") != 14 * 1024**3 or type(host.get("Memory")) is not int
            or host.get("MemorySwap") != 14 * 1024**3 or type(host.get("MemorySwap")) is not int
            or host.get("PidsLimit") != 128 or type(host.get("PidsLimit")) is not int
            or host.get("Tmpfs") != {"/tmp": docker["runtime_tmpfs"].removeprefix("/tmp:")}
            or observed_mounts != expected_mounts
            or host.get("Ulimits") != [{"Name": "nofile", "Hard": 256, "Soft": 256}]
        ):
            _fail("REJECT_Q05B_ARTIFACT_RESOURCE", "Docker inspect policy differs")


def _validate_docker_absence_v1(
    value: object, target: str, name: str
) -> dict[str, object]:
    evidence = _object(
        value,
        {
            "container_identity", "inspect_exit_code", "inspect_stderr_hex",
            "inspect_stderr_sha256", "inspect_stdout_hex",
            "inspect_stdout_sha256", "schema_version",
        },
        name,
    )
    stdout = _hex_any(evidence["inspect_stdout_hex"], f"{name} stdout")
    stderr = _hex_any(evidence["inspect_stderr_hex"], f"{name} stderr")
    authoritative = {
        (b"", f"Error: No such object: {target}\n".encode("ascii")),
        (b"", f"Error: No such container: {target}\n".encode("ascii")),
        (
            b"",
            f"Error response from daemon: No such container: {target}\n".encode(
                "ascii"
            ),
        ),
        (b"[]\n", f"error: no such object: {target}\n".encode("ascii")),
    }
    if (
        evidence["schema_version"]
        != _admission.DOCKER_AUTHORITATIVE_ABSENCE_SCHEMA_VERSION
        or evidence["container_identity"] != target
        or type(evidence["inspect_exit_code"]) is not int
        or evidence["inspect_exit_code"] != 1
        or evidence["inspect_stdout_sha256"] != sha256(stdout).hexdigest()
        or evidence["inspect_stderr_sha256"] != sha256(stderr).hexdigest()
        or (stdout, stderr) not in authoritative
    ):
        _fail("REJECT_Q05B_ARTIFACT_ISOLATION", f"{name} differs")
    return evidence


def _validate_docker_cid_payload_v1(
    cid: dict[str, object],
    expected_container_id: object,
    name: str,
) -> bytes:
    """Require the pinned Docker 29.1.3 exact 64-byte cidfile payload."""

    if (
        type(expected_container_id) is not str
        or re.fullmatch(r"[0-9a-f]{64}", expected_container_id) is None
    ):
        _fail("REJECT_Q05B_ARTIFACT_ISOLATION", f"{name} container ID differs")
    payload = _hex_any(cid.get("payload_hex"), name)
    expected_payload = expected_container_id.encode("ascii")
    if (
        cid.get("container_id") != expected_container_id
        or type(cid.get("file_size")) is not int
        or cid["file_size"] != 64
        or len(payload) != 64
        or payload != expected_payload
        or re.fullmatch(rb"[0-9a-f]{64}", payload) is None
        or cid.get("payload_sha256") != sha256(payload).hexdigest()
    ):
        _fail("REJECT_Q05B_ARTIFACT_ISOLATION", f"{name} payload differs")
    return payload


def _validate_actor_control_exact_v1(
    value: object,
    actor: dict[str, object],
    role_id: int,
    stdout: bytes,
    resource: dict[str, object],
    config: dict[str, object],
    runtime_seccomp_payload: bytes,
    pinned_image_id: str,
) -> dict[str, object]:
    control = _object(value, {"actor_id", "cidfile_evidence", "cleanup_target_kind", "command_sha256", "completion_evidence", "container_id", "container_name_was_never_a_destructive_target", "continuous_sample_count", "control_root_nlink", "control_root_path", "docker_absence_evidence", "docker_execution_authority_manifest_sha256", "docker_execution_slot_row", "explicit_remove_command", "explicit_remove_exit_code", "final_resource_transcript", "held_final_resource", "live_ownership_inspect_evidence", "mount_registry_sha256", "ownership_label_root", "post_exit_inspect_hex", "post_exit_inspect_sha256", "post_ownership_inspect_evidence", "precreate_absence_evidence", "release_evidence", "schema_version", "seccomp_evidence", "stderr_length", "stderr_sha256", "stdout_hex", "stdout_length", "stdout_sha256"}, "actor control evidence")
    actor_id = actor["actor_id"]
    command_sha, mount_sha = _command_mount_registry_v1(actor["command"], role_id, actor_id, config)
    completion = _replay_control_identity_v1(control["completion_evidence"], actor_id, stdout, False)
    release = _replay_control_identity_v1(control["release_evidence"], actor_id, stdout, True)
    cid = _object(control["cidfile_evidence"], {"cidfile_path", "container_id", "file_device", "file_gid", "file_inode", "file_mode", "file_nlink", "file_size", "file_uid", "manifest_sha256", "parent_device", "parent_inode", "parent_mode", "parent_nlink", "payload_hex", "payload_sha256", "relative_name", "schema_version"}, "cidfile evidence")
    cid_payload = _validate_docker_cid_payload_v1(
        cid, control["container_id"], "actor cidfile"
    )
    cid_body = dict(cid); cid_root = cid_body.pop("manifest_sha256")
    post = _hex_any(control["post_exit_inspect_hex"], "actor post-exit inspect")
    absence = _validate_docker_absence_v1(
        control["docker_absence_evidence"], control["container_id"],
        "actor Docker absence",
    )
    seccomp = _validate_seccomp_evidence_v1(control["seccomp_evidence"], config["seccomp"]["runtime_profile"], runtime_seccomp_payload)
    cid_tokens = [item.removeprefix("--cidfile=") for item in actor["command"] if item.startswith("--cidfile=")]
    seccomp_tokens = [item.removeprefix("--security-opt=seccomp=") for item in actor["command"] if item.startswith("--security-opt=seccomp=")]
    command_security = [
        item.removeprefix("--security-opt=")
        for item in actor["command"]
        if item.startswith("--security-opt=")
    ]
    post_value = _decoded_json_value(post, "actor post-exit inspect")
    post_host = (
        post_value[0].get("HostConfig")
        if type(post_value) is list
        and len(post_value) == 1
        and type(post_value[0]) is dict
        else None
    )
    if type(post_host) is not dict:
        _fail(
            "REJECT_Q05B_ARTIFACT_RESOURCE",
            "actor post-exit inspect HostConfig differs",
        )
    _validate_inspect_security_options_v1(
        post_host.get("SecurityOpt"),
        command_security,
        runtime_seccomp_payload,
        seccomp,
        "actor post-exit inspect",
    )
    if (
        cid["schema_version"] != "hegel-phase3a-q05b-sealed-docker-cidfile/1"
        or cid_tokens != [cid["cidfile_path"]]
        or seccomp_tokens != [seccomp["absolute_path"]]
        or cid["container_id"] != control["container_id"]
        or cid["file_size"] != len(cid_payload) or cid["file_mode"] != 0o444 or cid["file_nlink"] != 1 or cid["parent_mode"] != 0o700
        or cid_root != sha256(_canonical_json(cid_body)).hexdigest()
        or control["schema_version"] != "hegel-phase3a-q05b-held-actor-complete-evidence/1"
        or type(control["control_root_path"]) is not str
        or not control["control_root_path"].startswith("/")
        or ".." in control["control_root_path"].split("/")
        or type(control["control_root_nlink"]) is not int
        or control["control_root_nlink"] < 1
        or control["actor_id"] != actor_id or control["container_id"] != resource["container_id"]
        or control["command_sha256"] != command_sha or control["mount_registry_sha256"] != mount_sha
        or control["continuous_sample_count"] != len(resource["live_sample_objects"]) - 1
        or type(control["continuous_sample_count"]) is not int or control["continuous_sample_count"] < 1
        or control["held_final_resource"] != resource["live_sample_objects"][-1]
        or control["held_final_resource"]["completion_manifest_sha256"] != completion["manifest_sha256"]
        or control["final_resource_transcript"] != resource
        or control["post_exit_inspect_hex"] != resource["post_exit_inspect_hex"]
        or control["post_exit_inspect_sha256"] != sha256(post).hexdigest()
        or control["stdout_hex"] != stdout.hex() or control["stdout_length"] != len(stdout)
        or control["stdout_sha256"] != sha256(stdout).hexdigest()
        or control["stderr_length"] != 0 or control["stderr_sha256"] != sha256(b"").hexdigest()
        or control["explicit_remove_exit_code"] != 0 or type(control["explicit_remove_exit_code"]) is not int
        or absence["schema_version"]
        != _admission.DOCKER_AUTHORITATIVE_ABSENCE_SCHEMA_VERSION
        or absence["container_identity"] != control["container_id"]
    ):
        _fail("REJECT_Q05B_ARTIFACT_ACTOR", f"{actor_id} complete control differs")
    for sample in resource["live_sample_objects"]:
        _validate_live_inspect_policy_v1(
            sample,
            actor["command"],
            role_id,
            config,
            pinned_image_id,
            runtime_seccomp_payload,
            seccomp,
        )
    return control


def _mount_sources_v1(command: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for index, item in enumerate(command):
        if item == "--mount":
            match = re.fullmatch(
                r"type=bind,src=([^,]+),dst=([^,]+)(,readonly)?",
                command[index + 1],
            )
            if match is None or match.group(2) in result:
                _fail("REJECT_Q05B_ARTIFACT_ISOLATION", "mount source registry differs")
            result[match.group(2)] = match.group(1)
    return result


def _validate_cargo_execution_v1(
    transcript: dict[str, object], suffix: list[object], config: dict[str, object],
    rust_source_identity: str, name: str, build_seccomp_payload: bytes,
    pinned_image_id: str,
) -> None:
    command = transcript["command"]
    docker = config["docker"]; image = config["images"]["rust_build"]
    if type(command) is not list or any(type(item) is not str or not item for item in command) or "--mount" not in command:
        _fail("REJECT_Q05B_ARTIFACT_CARGO", f"{name} command differs")
    container_name, command_labels, normalized = _docker_run_principal_tokens_v1(
        command, name
    )
    first_mount = normalized.index("--mount")
    prefix = normalized[:first_mount]
    if len(prefix) != 29 or re.fullmatch(r"--cidfile=/[^,]+\.cid", prefix[5]) is None or re.fullmatch(r"--security-opt=seccomp=/[^,]+", prefix[11]) is None or re.fullmatch(r"--user=[1-9][0-9]*:[1-9][0-9]*", prefix[20]) is None:
        _fail("REJECT_Q05B_ARTIFACT_CARGO", f"{name} Docker prefix differs")
    expected_prefix = [docker["executable"], f"--host={docker['host']}", "run", "--name", container_name, prefix[5], f"--pull={docker['pull_policy']}", f"--network={docker['network']}", "--read-only", f"--cap-drop={docker['cap_drop']}", "--security-opt=no-new-privileges", prefix[11], f"--ipc={docker['ipc']}", "--cgroupns=private", f"--pids-limit={docker['pids_limit']}", f"--ulimit=nofile={docker['nofile_ulimit']}", f"--memory={docker['memory']}", f"--memory-swap={docker['memory_swap']}", "--cpuset-cpus=12-23", f"--tmpfs={docker['build_tmpfs']}", prefix[20], "-e", "HOME=/tmp", "-e", "LANG=C.UTF-8", "-e", "LC_ALL=C.UTF-8", "-e", "TZ=UTC"]
    if prefix != expected_prefix:
        _fail("REJECT_Q05B_ARTIFACT_CARGO", f"{name} Docker prefix differs")
    seccomp = _validate_seccomp_evidence_v1(transcript["seccomp_evidence"], config["seccomp"]["build_profile"], build_seccomp_payload)
    if prefix[5] != f"--cidfile={transcript['cidfile_evidence']['cidfile_path']}" or prefix[11] != f"--security-opt=seccomp={seccomp['absolute_path']}":
        _fail("REJECT_Q05B_ARTIFACT_CARGO", f"{name} control path binding differs")
    mounts: list[tuple[str, str, bool, str]] = []
    index = first_mount
    while index < len(normalized) and normalized[index] == "--mount":
        if index + 1 >= len(normalized):
            _fail("REJECT_Q05B_ARTIFACT_CARGO", "Cargo mount is truncated")
        match = re.fullmatch(r"type=bind,src=([^,]+),dst=([^,]+)(,readonly)?", normalized[index + 1])
        if match is None or not match.group(1).startswith("/") or ".." in match.group(1).split("/"):
            _fail("REJECT_Q05B_ARTIFACT_CARGO", "Cargo mount differs")
        mounts.append((match.group(2), match.group(1), match.group(3) is None, normalized[index + 1]))
        index += 2
    if [(row[0], row[2]) for row in mounts] != [("/snapshot", False), ("/cargo-home", False), ("/target-output", True)]:
        _fail("REJECT_Q05B_ARTIFACT_CARGO", "Cargo mount registry differs")
    expected_tail = []
    for row in mounts: expected_tail.extend(["--mount", row[3]])
    expected_tail.extend(["-e", "CARGO_HOME=/cargo-home", "-e", "CARGO_NET_OFFLINE=true", "-e", "CARGO_TARGET_DIR=/target-output", "-e", f"HEGEL_Q05B_RUST_SOURCE_IDENTITY_SHA256={rust_source_identity}", "-w", "/snapshot/rust/q1_archive_projection_oracle", image, *suffix])
    if normalized[first_mount:] != expected_tail:
        _fail("REJECT_Q05B_ARTIFACT_CARGO", f"{name} Docker argv differs")
    inspect = _decoded_json_value(_hex_any(transcript["live_inspect_hex"], f"{name} live inspect"), f"{name} live inspect")
    if type(inspect) is not list or len(inspect) != 1 or type(inspect[0]) is not dict:
        _fail("REJECT_Q05B_ARTIFACT_CARGO", f"{name} inspect shape differs")
    document = inspect[0]; state = document.get("State"); host = document.get("HostConfig"); container = document.get("Config"); observed_mounts = document.get("Mounts")
    if type(state) is not dict or type(host) is not dict or type(container) is not dict or type(observed_mounts) is not list:
        _fail("REJECT_Q05B_ARTIFACT_CARGO", f"{name} inspect sections differ")
    mount_map: dict[str, tuple[object, object]] = {}
    for mount in observed_mounts:
        if type(mount) is not dict or mount.get("Type") != "bind" or mount.get("Destination") in mount_map:
            _fail("REJECT_Q05B_ARTIFACT_CARGO", "Cargo inspect mount differs")
        mount_map[mount.get("Destination")] = (mount.get("Source"), mount.get("RW"))
    observed_env: list[tuple[str, str]] = []
    if type(container.get("Env")) is not list:
        _fail("REJECT_Q05B_ARTIFACT_CARGO", "Cargo inspect environment differs")
    for item in container["Env"]:
        if type(item) is not str or "=" not in item:
            _fail("REJECT_Q05B_ARTIFACT_CARGO", "Cargo inspect environment row differs")
        observed_env.append(tuple(item.split("=", 1)))
    runtime_env = config["runtime_command_inspect_policy"]["environment_rows"][1][2]
    expected_env = dict(runtime_env)
    expected_env.update({"CARGO_HOME": "/cargo-home", "CARGO_NET_OFFLINE": "true", "CARGO_TARGET_DIR": "/target-output", "HEGEL_Q05B_RUST_SOURCE_IDENTITY_SHA256": rust_source_identity})
    observed_labels = container.get("Labels")
    command_security = [
        item.removeprefix("--security-opt=")
        for item in normalized
        if item.startswith("--security-opt=")
    ]
    _validate_inspect_security_options_v1(
        host.get("SecurityOpt"),
        command_security,
        build_seccomp_payload,
        seccomp,
        f"{name} live inspect",
    )
    if (
        document.get("Id") != transcript["cidfile_evidence"]["container_id"] or document.get("Name") != f"/{container_name}" or document.get("Image") != pinned_image_id or state.get("Running") is not True or state.get("OOMKilled") is not False or type(state.get("Pid")) is not int or state["Pid"] < 1
        or container.get("Image") != image or container.get("User") != prefix[20].removeprefix("--user=") or container.get("Entrypoint") is not None or container.get("Cmd") != suffix or container.get("WorkingDir") != "/snapshot/rust/q1_archive_projection_oracle" or tuple(sorted(observed_env)) != tuple(sorted(expected_env.items()))
        or type(observed_labels) is not dict
        or any(type(key) is not str or type(value) is not str for key, value in observed_labels.items())
        or any(observed_labels.get(key) != value for key, value in command_labels)
        or host.get("AutoRemove") is not False or host.get("NetworkMode") != docker["network"] or host.get("ReadonlyRootfs") is not True or host.get("CapDrop") != [docker["cap_drop"]] or host.get("IpcMode") != docker["ipc"] or host.get("Privileged") is not False or host.get("Devices") != [] or host.get("DeviceRequests") is not None or host.get("CpusetCpus") != "12-23" or host.get("Memory") != 14 * 1024**3 or host.get("MemorySwap") != 14 * 1024**3 or host.get("PidsLimit") != 128 or host.get("Tmpfs") != {"/tmp": docker["build_tmpfs"].removeprefix("/tmp:")} or host.get("Ulimits") != [{"Name": "nofile", "Hard": 256, "Soft": 256}]
        or mount_map != {row[0]: (row[1], row[2]) for row in mounts}
    ):
        _fail("REJECT_Q05B_ARTIFACT_CARGO", f"{name} inspect policy differs")
    post = _decoded_json_value(_hex_any(transcript["post_inspect_hex"], f"{name} post inspect"), f"{name} post inspect")
    post_config = post[0].get("Config") if type(post) is list and len(post) == 1 and type(post[0]) is dict else None
    post_labels = post_config.get("Labels") if type(post_config) is dict else None
    post_host = post[0].get("HostConfig") if type(post) is list and len(post) == 1 and type(post[0]) is dict else None
    if type(post_host) is dict:
        _validate_inspect_security_options_v1(
            post_host.get("SecurityOpt"),
            command_security,
            build_seccomp_payload,
            seccomp,
            f"{name} post inspect",
        )
    if type(post) is not list or len(post) != 1 or type(post[0]) is not dict or post[0].get("Id") != transcript["cidfile_evidence"]["container_id"] or post[0].get("Name") != f"/{container_name}" or type(post_config) is not dict or type(post_labels) is not dict or any(post_labels.get(key) != value for key, value in command_labels) or type(post[0].get("State")) is not dict or type(post_host) is not dict or post[0]["State"].get("Running") is not False or post[0]["State"].get("OOMKilled") is not False or post[0]["State"].get("ExitCode") != 0 or post_host.get("AutoRemove") is not False:
        _fail("REJECT_Q05B_ARTIFACT_CARGO", f"{name} post-exit state differs")


def _validate_binary_detach_evidence_v1(
    value: object,
    target_output_root_path: str,
    binary_path: str,
    binary_sha256: str,
    binary_size: int,
    sealed_identity: dict[str, object],
) -> dict[str, object]:
    keys = {
        "schema_version",
        "source_path",
        "detached_path",
        "source_parent_before",
        "source_parent_after",
        "source_fd_before",
        "source_fd_after",
        "source_path_before",
        "source_path_after",
        "source_sha256_before",
        "source_sha256_after",
        "detached_parent_before",
        "detached_parent_after",
        "detached_fd",
        "detached_path_identity",
        "detached_sha256",
        "source_and_detached_bytes_equal",
        "manifest_sha256",
    }
    detach = _object(value, keys, "binary detach evidence")
    body = dict(detach)
    manifest = body.pop("manifest_sha256")
    parent_keys = {"device", "inode", "nlink", "uid", "gid", "mode"}
    file_keys = parent_keys | {"size", "mtime_ns", "ctime_ns"}
    parent_rows = [
        _object(detach[name], parent_keys, name)
        for name in (
            "source_parent_before",
            "source_parent_after",
            "detached_parent_before",
            "detached_parent_after",
        )
    ]
    file_rows = [
        _object(detach[name], file_keys, name)
        for name in (
            "source_fd_before",
            "source_fd_after",
            "source_path_before",
            "source_path_after",
            "detached_fd",
            "detached_path_identity",
        )
    ]
    source_parent = parent_rows[0]
    detached_parent = parent_rows[2]
    source_identity = file_rows[0]
    detached_identity = file_rows[4]
    expected_source_path = (
        target_output_root_path.rstrip("/")
        + "/release/hegel-q1-archive-projection-oracle"
    )
    expected_detached_path = (
        target_output_root_path.rstrip("/")
        + "/runtime-binary/hegel-q1-archive-projection-oracle"
    )
    hashes = (
        detach["source_sha256_before"],
        detach["source_sha256_after"],
        detach["detached_sha256"],
    )
    if (
        detach["schema_version"]
        != "hegel-phase3a-q05b-detached-cargo-release-binary/1"
        or type(manifest) is not str
        or re.fullmatch(r"[0-9a-f]{64}", manifest) is None
        or manifest != sha256(_canonical_json(body)).hexdigest()
        or detach["source_path"] != expected_source_path
        or detach["detached_path"] != expected_detached_path
        or binary_path != expected_detached_path
        or any(
            any(type(item) is not int or item < 0 for item in row.values())
            for row in parent_rows + file_rows
        )
        or parent_rows[0] != parent_rows[1]
        or parent_rows[2] != parent_rows[3]
        or file_rows[0] != file_rows[1]
        or file_rows[1] != file_rows[2]
        or file_rows[2] != file_rows[3]
        or file_rows[4] != file_rows[5]
        or any(
            type(value) is not str
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in hashes
        )
        or hashes != (binary_sha256, binary_sha256, binary_sha256)
        or detach["source_and_detached_bytes_equal"] is not True
        or source_parent["inode"] <= 0
        or source_parent["nlink"] < 2
        or detached_parent["inode"] <= 0
        or detached_parent["nlink"] < 2
        or detached_parent["mode"] != 0o700
        or source_identity["inode"] <= 0
        or source_identity["nlink"] < 1
        or source_identity["mode"] != 0o755
        or source_identity["size"] != binary_size
        or detached_identity["inode"] <= 0
        or detached_identity["nlink"] != 1
        or detached_identity["mode"] != 0o755
        or detached_identity["size"] != binary_size
        or (source_identity["device"], source_identity["inode"])
        == (detached_identity["device"], detached_identity["inode"])
        or detached_parent["uid"] != detached_identity["uid"]
        or detached_parent["gid"] != detached_identity["gid"]
        or any(
            sealed_identity[field] != detached_identity[field]
            for field in (
                "device", "inode", "nlink", "uid", "gid", "size",
                "mtime_ns",
            )
        )
        or sealed_identity["mode"] != 0o555
        or sealed_identity["ctime_ns"] < detached_identity["ctime_ns"]
        or sealed_identity["sha256"] != binary_sha256
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_CARGO",
            "detached Cargo binary evidence differs",
        )
    return detach


def _validate_cargo_v1(
    value: object, config: dict[str, object], rust_source_identity: str,
    rust_snapshot_path: str, build_seccomp_payload: bytes,
    pinned_image_id: str,
) -> dict[str, object]:
    cargo = _object(value, {"binary_detach_evidence", "binary_file_identity", "binary_hex", "binary_path", "binary_runtime_identity_sha256", "binary_sha256", "cargo_snapshot_post_build", "lock_hex", "locked_packages", "rust_image_inspect_hex", "rust_image_inspect_sha256", "rust_release_build", "rust_snapshot_post_build", "rust_test", "schema_version", "sealed_cargo_files", "sealed_cargo_manifest_sha256", "sealed_cargo_tree", "target_output_root_path"}, "cargo_build_binary")
    if cargo["schema_version"] != "hegel-phase3a-q05b-cargo-build-binary-evidence/1":
        _fail("REJECT_Q05B_ARTIFACT_CARGO", "Cargo evidence schema differs")
    lock = _hex_any(cargo["lock_hex"], "Cargo.lock")
    binary = _hex_any(cargo["binary_hex"], "Rust binary")
    image_inspect = _hex_any(cargo["rust_image_inspect_hex"], "Rust image inspect")
    binary_identity = _object(cargo["binary_file_identity"], {"ctime_ns", "device", "gid", "inode", "mode", "mtime_ns", "nlink", "path", "sha256", "size", "uid"}, "Rust binary identity")
    if type(cargo["binary_path"]) is not str or not cargo["binary_path"].startswith("/") or ".." in cargo["binary_path"].split("/") or binary_identity["path"] != cargo["binary_path"] or any(type(binary_identity[field]) is not int for field in ("ctime_ns", "device", "gid", "inode", "mode", "mtime_ns", "nlink", "size", "uid")) or binary_identity["nlink"] != 1 or binary_identity["mode"] != 0o555 or binary_identity["size"] != len(binary) or binary_identity["sha256"] != sha256(binary).hexdigest():
        _fail("REJECT_Q05B_ARTIFACT_CARGO", "Rust binary file identity differs")
    if cargo["binary_sha256"] != sha256(binary).hexdigest() or cargo["rust_image_inspect_sha256"] != sha256(image_inspect).hexdigest():
        _fail("REJECT_Q05B_ARTIFACT_CARGO", "binary/image identity differs")
    runtime = sha256(b"HEGEL/Q05B/RUST_RUNTIME_IDENTITY/V1\x00" + len(binary).to_bytes(8, "big") + binary).hexdigest()
    if cargo["binary_runtime_identity_sha256"] != runtime:
        _fail("REJECT_Q05B_ARTIFACT_CARGO", "Rust runtime identity differs")
    try:
        lock_value = tomllib.loads(lock.decode("utf-8", "strict"))
    except (UnicodeError, tomllib.TOMLDecodeError) as error:
        _fail("REJECT_Q05B_ARTIFACT_CARGO", f"Cargo.lock differs: {error}")
    raw_lock_packages = lock_value.get("package") if type(lock_value) is dict else None
    if type(raw_lock_packages) is not list:
        _fail("REJECT_Q05B_ARTIFACT_CARGO", "Cargo.lock package registry differs")
    parsed_packages = []
    for item in raw_lock_packages:
        if type(item) is not dict:
            _fail("REJECT_Q05B_ARTIFACT_CARGO", "Cargo.lock package row differs")
        source = item.get("source")
        if source is None:
            continue
        if (
            source != "registry+https://github.com/rust-lang/crates.io-index"
            or type(item.get("name")) is not str
            or type(item.get("version")) is not str
            or type(item.get("checksum")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", item["checksum"]) is None
        ):
            _fail("REJECT_Q05B_ARTIFACT_CARGO", "Cargo.lock registry row differs")
        parsed_packages.append([item["name"], item["version"], item["checksum"]])
    parsed_packages.sort()
    packages = cargo["locked_packages"]
    files = cargo["sealed_cargo_files"]
    if type(packages) is not list or not packages or type(files) is not list or not files:
        _fail("REJECT_Q05B_ARTIFACT_CARGO", "Cargo package/file registry differs")
    file_rows = []
    payload_by_path = {}
    for raw in files:
        if type(raw) is not list or len(raw) != 3:
            _fail("REJECT_Q05B_ARTIFACT_CARGO", "sealed Cargo row differs")
        path, mode, payload_hex = raw; payload = _hex_any(payload_hex, "sealed Cargo file")
        if type(path) is not str or path.startswith("/") or ".." in path.split("/") or type(mode) is not int or mode not in (0o100644, 0o100755) or path in payload_by_path:
            _fail("REJECT_Q05B_ARTIFACT_CARGO", "sealed Cargo path differs")
        payload_by_path[path] = payload
        file_rows.append([path, mode, len(payload), sha256(payload).hexdigest()])
    if list(payload_by_path) != sorted(payload_by_path) or cargo["sealed_cargo_manifest_sha256"] != sha256(_canonical_json(file_rows)).hexdigest():
        _fail("REJECT_Q05B_ARTIFACT_CARGO", "sealed Cargo manifest differs")
    cargo_directories = sorted({"/".join(path.split("/")[:depth]) for path in payload_by_path for depth in range(1, len(path.split("/")))})
    cargo_tree = _validate_sealed_tree_identity_v1(
        cargo["sealed_cargo_tree"], payload_by_path, tuple(cargo_directories),
        "sealed Cargo tree",
        {row[0]: 0o555 if row[1] == 0o100755 else 0o444 for row in files},
    )
    if cargo["cargo_snapshot_post_build"] != cargo["sealed_cargo_tree"] or type(cargo["rust_snapshot_post_build"]) is not dict or cargo["rust_snapshot_post_build"].get("root_path") != rust_snapshot_path:
        _fail("REJECT_Q05B_ARTIFACT_CARGO", "post-build sealed snapshots differ")
    if type(cargo["target_output_root_path"]) is not str or not cargo["target_output_root_path"].startswith("/") or ".." in cargo["target_output_root_path"].split("/") or not cargo["binary_path"].startswith(cargo["target_output_root_path"].rstrip("/") + "/"):
        _fail("REJECT_Q05B_ARTIFACT_CARGO", "Cargo target path differs")
    _validate_binary_detach_evidence_v1(
        cargo["binary_detach_evidence"],
        cargo["target_output_root_path"],
        cargo["binary_path"],
        cargo["binary_sha256"],
        len(binary),
        binary_identity,
    )
    if packages != parsed_packages or not packages:
        _fail("REJECT_Q05B_ARTIFACT_CARGO", "Cargo.lock parsed package set differs")
    expected_material: dict[str, tuple[int, bytes]] = {}
    registry_ids: set[str] = set()

    def admit(path: str, mode: int, payload: bytes) -> None:
        prior = expected_material.get(path)
        if prior is not None and prior != (mode, payload):
            _fail("REJECT_Q05B_ARTIFACT_CARGO", "Cargo material collision")
        expected_material[path] = (mode, payload)

    for row in packages:
        if type(row) is not list or len(row) != 3 or any(type(item) is not str or not item for item in row) or re.fullmatch(r"[0-9a-f]{64}", row[2]) is None:
            _fail("REJECT_Q05B_ARTIFACT_CARGO", "locked package row differs")
        stem = f"{row[0]}-{row[1]}"
        matches = [(path, payload) for path, payload in payload_by_path.items() if re.fullmatch(rf"registry/cache/[A-Za-z0-9._-]+/{re.escape(stem)}\.crate", path)]
        if len(matches) != 1 or sha256(matches[0][1]).hexdigest() != row[2]:
            _fail("REJECT_Q05B_ARTIFACT_CARGO", "locked crate checksum differs")
        archive_path, archive = matches[0]
        registry_id = archive_path.split("/")[2]
        registry_ids.add(registry_id)
        admit(archive_path, 0o100644, archive)
        seen: set[str] = set()
        try:
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
                for member in sorted(bundle.getmembers(), key=lambda item: item.name):
                    parts = member.name.split("/")
                    if member.isdir():
                        continue
                    if not member.isfile() or len(parts) < 2 or parts[0] != stem or any(part in ("", ".", "..") for part in parts):
                        _fail("REJECT_Q05B_ARTIFACT_CARGO", "crate member differs")
                    relative = "/".join(parts[1:])
                    stream = bundle.extractfile(member)
                    payload = stream.read() if stream is not None else None
                    if relative in seen or payload is None or len(payload) != member.size or member.size > 64 * 1024 * 1024:
                        _fail("REJECT_Q05B_ARTIFACT_CARGO", "crate member payload differs")
                    seen.add(relative)
                    admit(f"registry/src/{registry_id}/{stem}/{relative}", 0o100755 if member.mode & 0o100 else 0o100644, payload)
        except (OSError, tarfile.TarError) as error:
            _fail("REJECT_Q05B_ARTIFACT_CARGO", f"crate archive replay differs: {error}")
        if not seen:
            _fail("REJECT_Q05B_ARTIFACT_CARGO", "crate archive is empty")
        admit(f"registry/src/{registry_id}/{stem}/.cargo-ok", 0o100644, b'{"v":1}')
        index_matches = [(path, payload) for path, payload in payload_by_path.items() if path.startswith(f"registry/index/{registry_id}/.cache/") and path.endswith(f"/{row[0]}")]
        if len(index_matches) != 1:
            _fail("REJECT_Q05B_ARTIFACT_CARGO", "registry index entry differs")
        index_path, index_payload = index_matches[0]
        fields = index_payload[5:].split(b"\x00") if len(index_payload) >= 8 and index_payload[0] == 3 else []
        if fields and fields[-1] == b"":
            fields.pop()
        if len(fields) < 3 or (len(fields) - 1) % 2:
            _fail("REJECT_Q05B_ARTIFACT_CARGO", "registry index framing differs")
        locked_matches = 0
        for offset in range(1, len(fields), 2):
            version = fields[offset].decode("ascii", "strict")
            document = _decoded_json_value(fields[offset + 1], "registry index")
            if type(document) is not dict or document.get("name") != row[0] or document.get("vers") != version:
                _fail("REJECT_Q05B_ARTIFACT_CARGO", "registry index identity differs")
            if version == row[1]:
                locked_matches += 1
                if document.get("cksum") != row[2]:
                    _fail("REJECT_Q05B_ARTIFACT_CARGO", "registry index checksum differs")
        if locked_matches != 1:
            _fail("REJECT_Q05B_ARTIFACT_CARGO", "locked registry version differs")
        admit(index_path, 0o100644, index_payload)
    if len(registry_ids) != 1:
        _fail("REJECT_Q05B_ARTIFACT_CARGO", "Cargo registry identity differs")
    registry_id = next(iter(registry_ids))
    for path in (f"registry/index/{registry_id}/config.json", ".package-cache", ".package-cache-mutate"):
        if path not in payload_by_path:
            _fail("REJECT_Q05B_ARTIFACT_CARGO", "Cargo fixed material is absent")
        admit(path, 0o100644, payload_by_path[path])
    if payload_by_path[".package-cache"] != b"" or payload_by_path[".package-cache-mutate"] != b"":
        _fail("REJECT_Q05B_ARTIFACT_CARGO", "Cargo package cache sentinel differs")
    observed_material = {
        row[0]: (row[1], payload_by_path[row[0]]) for row in files
    }
    if observed_material != expected_material:
        _fail("REJECT_Q05B_ARTIFACT_CARGO", "sealed Cargo closure differs")
    docker = config.get("docker")
    images = config.get("images")
    rust_policy = config.get("rust_build_policy")
    if type(docker) is not dict or type(images) is not dict or type(rust_policy) is not dict or type(rust_policy.get("commands")) is not list:
        _fail("REJECT_Q05B_ARTIFACT_CARGO", "Rust build policy differs")
    for name in ("rust_test", "rust_release_build"):
        transcript = _object(cargo[name], {"cidfile_evidence", "cleanup_target_kind", "command", "command_sha256", "container_name_was_never_a_destructive_target", "docker_absence_evidence", "docker_execution_authority_manifest_sha256", "docker_execution_slot_row", "evidence_sha256", "exit_code", "explicit_remove_command", "live_inspect_hex", "live_inspect_sha256", "live_ownership_inspect_evidence", "ownership_label_root", "post_inspect_hex", "post_inspect_sha256", "post_ownership_inspect_evidence", "precreate_absence_evidence", "schema_version", "seccomp_evidence", "stderr_hex", "stderr_length", "stderr_sha256", "stdout_hex", "stdout_length", "stdout_sha256"}, name)
        live_inspect = _hex_any(transcript["live_inspect_hex"], f"{name} live inspect")
        post_inspect = _hex_any(transcript["post_inspect_hex"], f"{name} post inspect")
        stderr = _hex_any(transcript["stderr_hex"], f"{name} stderr")
        stdout = _hex_any(transcript["stdout_hex"], f"{name} stdout")
        command = transcript["command"]
        suffix = rust_policy["commands"][0 if name == "rust_test" else 1]
        cid = _object(transcript["cidfile_evidence"], {"cidfile_path", "container_id", "file_device", "file_gid", "file_inode", "file_mode", "file_nlink", "file_size", "file_uid", "manifest_sha256", "parent_device", "parent_inode", "parent_mode", "parent_nlink", "payload_hex", "payload_sha256", "relative_name", "schema_version"}, f"{name} cidfile")
        cid_payload = _validate_docker_cid_payload_v1(
            cid, cid["container_id"], f"{name} cidfile"
        )
        cid_body = dict(cid); cid_root = cid_body.pop("manifest_sha256")
        absence = _validate_docker_absence_v1(
            transcript["docker_absence_evidence"], cid["container_id"],
            f"{name} Docker absence",
        )
        evidence_body = dict(transcript); evidence_root = evidence_body.pop("evidence_sha256")
        if transcript["schema_version"] != "hegel-phase3a-q05b-offline-rust-container-run/1" or transcript["command_sha256"] != sha256(_canonical_json(command)).hexdigest() or transcript["exit_code"] != 0 or type(transcript["exit_code"]) is not int or transcript["live_inspect_sha256"] != sha256(live_inspect).hexdigest() or transcript["post_inspect_sha256"] != sha256(post_inspect).hexdigest() or transcript["stderr_sha256"] != sha256(stderr).hexdigest() or transcript["stderr_length"] != len(stderr) or type(transcript["stderr_length"]) is not int or transcript["stdout_sha256"] != sha256(stdout).hexdigest() or transcript["stdout_length"] != len(stdout) or type(transcript["stdout_length"]) is not int or cid["schema_version"] != "hegel-phase3a-q05b-sealed-docker-cidfile/1" or cid["file_size"] != len(cid_payload) or cid["file_mode"] != 0o444 or cid["file_nlink"] != 1 or cid_root != sha256(_canonical_json(cid_body)).hexdigest() or absence["schema_version"] != _admission.DOCKER_AUTHORITATIVE_ABSENCE_SCHEMA_VERSION or absence["container_identity"] != cid["container_id"] or evidence_root != sha256(_canonical_json(evidence_body)).hexdigest():
            _fail("REJECT_Q05B_ARTIFACT_CARGO", f"{name} transcript differs")
        _validate_cargo_execution_v1(
            transcript, suffix, config, rust_source_identity, name,
            build_seccomp_payload, pinned_image_id,
        )
        command = transcript["command"]
        mount_sources = {}
        for index, item in enumerate(command):
            if item == "--mount":
                match = re.fullmatch(r"type=bind,src=([^,]+),dst=([^,]+)(,readonly)?", command[index + 1])
                mount_sources[match.group(2)] = match.group(1)
        if mount_sources != {"/snapshot": rust_snapshot_path, "/cargo-home": cargo_tree["root_path"], "/target-output": cargo["target_output_root_path"]}:
            _fail("REJECT_Q05B_ARTIFACT_CARGO", "Cargo mount/source binding differs")
    image_value = _decoded_json_value(image_inspect, "Rust image inspect")
    if type(image_value) is not list or len(image_value) != 1 or type(image_value[0]) is not dict or type(image_value[0].get("RepoDigests")) is not list or images["rust_build"] not in image_value[0]["RepoDigests"]:
        _fail("REJECT_Q05B_ARTIFACT_CARGO", "Rust image inspect identity differs")
    return cargo


def _validate_scratch_v1(
    value: object, producer_roots: tuple[bytes, bytes, bytes]
) -> list[dict[str, object]]:
    expected_ids = ("PYTHON_ENDPOINT", "RUST_ENDPOINT", "TRUSTED_HOST_REPLAY")
    if type(value) is not list or len(value) != 3:
        _fail("REJECT_Q05B_ARTIFACT_SCRATCH", "three scratch rows required")
    result = []
    for expected, producer_root, raw in zip(expected_ids, producer_roots, value, strict=True):
        row = _object(raw, {"actor_id", "partition_scratch_ledger_roots", "producer_replay_root", "scratch_root"}, "scratch row")
        roots = row["partition_scratch_ledger_roots"]
        if type(roots) is not list or len(roots) != 2 or any(type(partition) is not list or len(partition) != 4 for partition in roots):
            _fail("REJECT_Q05B_ARTIFACT_SCRATCH", "scratch ledger shape differs")
        for partition in roots:
            for root in partition:
                _hex(root, 32, "scratch ledger root")
        preimage = {"actor_id": expected, "partition_scratch_ledger_roots": roots, "producer_replay_root": producer_root.hex()}
        expected_root = _json_root("HEGEL/Q05B/ACTUAL/SCRATCH_ACTOR/V1", preimage).hex()
        if row["actor_id"] != expected or row["producer_replay_root"] != producer_root.hex() or row["scratch_root"] != expected_root:
            _fail("REJECT_Q05B_ARTIFACT_SCRATCH", "scratch root differs")
        result.append(row)
    return result


def _validate_host_control_v1(
    payload: bytes,
    loaded_rows: object,
    loaded_root: str,
    witness: bytes,
    witness_root: str,
    dual_root: str,
    host_actor: dict[str, object],
) -> dict[str, object]:
    value = _strict_json(payload)
    _object(
        value,
        {
            "action_id", "actor_id", "file_count", "final_isolation_root",
            "implementation_id", "loaded_module_root", "loaded_module_rows",
            "q1_formal_roots", "q1_gate_count", "q1_gate_mask",
            "q1_output_slots", "q1_state", "qualification_receipt",
            "runtime_identity_sha256", "schema_version",
            "semantic_replay_root", "source_identity_sha256", "status",
            "witness_length", "witness_relative_path", "witness_root",
            "witness_sha256",
        },
        "host control",
    )
    if (
        value["action_id"] != "trusted-host-semantic-replay-v1"
        or value["actor_id"] != "TRUSTED_HOST_REPLAY"
        or value["file_count"] != 6
        or type(value["file_count"]) is not int
        or value["final_isolation_root"] is not None
        or value["implementation_id"]
        != "HEGEL_Q1_BOUNDED_NODE3_PROJECTION_HOST_REPLAY_V1"
        or value["loaded_module_rows"] != loaded_rows
        or value["loaded_module_root"] != loaded_root
        or value["q1_formal_roots"] is not None
        or value["q1_gate_count"] != 0
        or type(value["q1_gate_count"]) is not int
        or value["q1_gate_mask"] != 0
        or type(value["q1_gate_mask"]) is not int
        or value["q1_output_slots"] != [None] * 8
        or value["q1_state"] != "NOT_RUN"
        or value["qualification_receipt"] is not None
        or value["runtime_identity_sha256"] != host_actor["runtime_identity_sha256"]
        or value["schema_version"]
        != "hegel-phase3a-q05b-host-semantic-control-envelope/1"
        or value["semantic_replay_root"] != dual_root
        or value["source_identity_sha256"]
        != host_actor["source_evidence"]["source_identity_sha256"]
        or value["status"] != "HOST_SEMANTIC_WITNESS_EMITTED_NOT_RECEIPT"
        or value["witness_length"] != len(witness)
        or type(value["witness_length"]) is not int
        or value["witness_relative_path"]
        != "host-semantic-witness.json"
        or value["witness_root"] != witness_root
        or value["witness_sha256"] != sha256(witness).hexdigest()
    ):
        _fail("REJECT_Q05B_ARTIFACT_HOST", "host control binding differs")
    return value


def _validate_loaded_modules_v1(
    value: object, payload_table: dict[str, tuple[int, str, bytes]]
) -> tuple[list[list[object]], str]:
    if type(value) is not list or not value:
        _fail("REJECT_Q05B_ARTIFACT_HOST", "loaded module rows differ")
    forbidden = {
        "hegel_machine.__init__",
        "hegel_machine.phase3_dsl_v1",
        "hegel_machine.phase3_m25_rows_v1",
        "hegel_machine.phase3_m25_split_v1",
        "hegel_machine.phase3_m25_formal_static_basis_v1",
    }
    expected_modules = {
        "phase3_q05b_host_replay_v1", "phase3_q05b_negative_vectors_v1"
    }
    pending = list(expected_modules)
    while pending:
        module = pending.pop()
        path = f"src/hegel_machine/{module}.py"
        if path not in payload_table:
            _fail("REJECT_Q05B_ARTIFACT_HOST", f"loaded source absent: {path}")
        try:
            tree = ast.parse(payload_table[path][2], filename=path)
        except (SyntaxError, UnicodeError) as error:
            _fail("REJECT_Q05B_ARTIFACT_HOST", f"loaded source parse differs: {error}")
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                if node.module:
                    imports.add(node.module.split(".", 1)[0])
                else:
                    imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        for dependency in imports - expected_modules:
            dependency_path = f"src/hegel_machine/{dependency}.py"
            if dependency_path in payload_table:
                expected_modules.add(dependency)
                pending.append(dependency)
    expected_names = ["hegel_machine"] + sorted(
        f"hegel_machine.{module}" for module in expected_modules
    )
    names: list[str] = []
    for index, raw in enumerate(value):
        if (
            type(raw) is not list
            or len(raw) != 3
            or type(raw[0]) is not str
            or raw[0] in forbidden
            or raw[0] in names
        ):
            _fail("REJECT_Q05B_ARTIFACT_HOST", "loaded module row differs")
        name, path, digest = raw
        if index == 0:
            if [name, path, digest] != ["hegel_machine", None, None]:
                _fail("REJECT_Q05B_ARTIFACT_HOST", "empty package row differs")
        elif (
            type(path) is not str
            or path not in payload_table
            or path.endswith("/__init__.py")
            or digest != sha256(payload_table[path][2]).hexdigest()
        ):
            _fail("REJECT_Q05B_ARTIFACT_HOST", "loaded module source differs")
        names.append(name)
    if names != sorted(names):
        _fail("REJECT_Q05B_ARTIFACT_HOST", "loaded module rows are not ordered")
    if names != expected_names:
        _fail("REJECT_Q05B_ARTIFACT_HOST", "loaded module closure is incomplete")
    root = sha256(
        b"HEGEL/Q05B/HOST/LOADED_MODULE_CLOSURE/V1\x00" + _canonical_json(value)
    ).hexdigest()
    return value, root


def _git_object_oid(kind: bytes, payload: bytes) -> str:
    return sha1(kind + b" " + str(len(payload)).encode("ascii") + b"\x00" + payload).hexdigest()


def _parse_git_tree_v1(payload: bytes) -> dict[str, tuple[int, str]]:
    rows: dict[str, tuple[int, str]] = {}
    offset = 0
    while offset < len(payload):
        space = payload.find(b" ", offset)
        nul = payload.find(b"\x00", space + 1)
        if space < 0 or nul < 0 or nul + 21 > len(payload):
            _fail("REJECT_Q05B_ARTIFACT_SOURCE", "Git tree framing differs")
        mode_raw = payload[offset:space]
        name_raw = payload[space + 1:nul]
        try:
            mode = int(mode_raw, 8)
            name = name_raw.decode("utf-8", "strict")
        except (ValueError, UnicodeError):
            _fail("REJECT_Q05B_ARTIFACT_SOURCE", "Git tree entry differs")
        if (
            mode not in (0o40000, 0o100644, 0o100755)
            or not name
            or "/" in name
            or name in (".", "..")
            or name in rows
        ):
            _fail("REJECT_Q05B_ARTIFACT_SOURCE", "Git tree entry policy differs")
        rows[name] = (mode, payload[nul + 1:nul + 21].hex())
        offset = nul + 21
    return rows


def _replay_git_object_closure_v1(
    source: dict[str, object],
    payload_table: dict[str, tuple[int, str, bytes]],
) -> None:
    commit_payload = _hex_any(source["git_commit_object_hex"], "Git commit object")
    if _git_object_oid(b"commit", commit_payload) != source["source_commit"]:
        _fail("REJECT_Q05B_ARTIFACT_SOURCE", "Git commit object identity differs")
    match = re.match(rb"tree ([0-9a-f]{40})\n", commit_payload)
    if match is None:
        _fail("REJECT_Q05B_ARTIFACT_SOURCE", "Git commit tree header differs")
    root_oid = match.group(1).decode("ascii")
    tree_rows = source["git_tree_object_rows"]
    if type(tree_rows) is not list or not tree_rows:
        _fail("REJECT_Q05B_ARTIFACT_SOURCE", "Git tree object registry differs")
    tree_payloads: dict[str, bytes] = {}
    for raw in tree_rows:
        if type(raw) is not list or len(raw) != 2:
            _fail("REJECT_Q05B_ARTIFACT_SOURCE", "Git tree object row differs")
        oid, payload_hex = raw
        payload = _hex_any(payload_hex, "Git tree object")
        if type(oid) is not str or oid != _git_object_oid(b"tree", payload) or oid in tree_payloads:
            _fail("REJECT_Q05B_ARTIFACT_SOURCE", "Git tree object identity differs")
        tree_payloads[oid] = payload
    if list(tree_payloads) != sorted(tree_payloads):
        _fail("REJECT_Q05B_ARTIFACT_SOURCE", "Git tree object rows are not ordered")
    prefix = source["project_tree_prefix"]
    if type(prefix) is not str or not prefix or prefix.startswith("/") or any(part in ("", ".", "..") for part in prefix.split("/")):
        _fail("REJECT_Q05B_ARTIFACT_SOURCE", "project tree prefix differs")
    used: set[str] = set()

    def descend(tree_oid: str, parts: list[str]) -> tuple[int, str]:
        current = tree_oid
        for index, part in enumerate(parts):
            if current not in tree_payloads:
                _fail("REJECT_Q05B_ARTIFACT_SOURCE", "Git tree object is absent")
            used.add(current)
            entries = _parse_git_tree_v1(tree_payloads[current])
            if part not in entries:
                _fail("REJECT_Q05B_ARTIFACT_SOURCE", "Git path inclusion is absent")
            mode, oid = entries[part]
            if index + 1 == len(parts):
                return mode, oid
            if mode != 0o40000:
                _fail("REJECT_Q05B_ARTIFACT_SOURCE", "Git path traverses a non-tree")
            current = oid
        _fail("REJECT_Q05B_ARTIFACT_SOURCE", "empty Git inclusion path")

    for path, (mode, oid, _payload) in payload_table.items():
        observed_mode, observed_oid = descend(root_oid, prefix.split("/") + path.split("/"))
        if observed_mode != mode or observed_oid != oid:
            _fail("REJECT_Q05B_ARTIFACT_SOURCE", f"Git blob inclusion differs: {path}")
    if used != set(tree_payloads):
        _fail("REJECT_Q05B_ARTIFACT_SOURCE", "Git tree closure has unused objects")
    external = source["external_commit_replay"]
    if external["tree_oid"] != root_oid:
        _fail("REJECT_Q05B_ARTIFACT_SOURCE", "external/root tree identity differs")


def _isolation_config_v1(
    payload_table: dict[str, tuple[int, str, bytes]]
) -> dict[str, object]:
    path = "config/phase3_q05b_dual_isolation_v1.json"
    if path not in payload_table:
        _fail("REJECT_Q05B_ARTIFACT_SOURCE", "frozen isolation config blob is absent")
    value = _decoded_json_value(payload_table[path][2], "isolation config")
    expected_top = {
        "actor_commands", "actual_preconditions", "artifact_layout", "claim_scope",
        "docker", "dry_run_authority", "engineering_status", "execution_protocol",
        "held_actor_protocol", "images", "live_resource_evidence_policy",
        "mount_policy", "profile_id", "qualification_receipt_protocol",
        "resource_roles", "runtime_command_inspect_policy", "rust_build_policy",
        "schema_version", "seccomp", "source_allowlist_policy",
        "source_snapshot_policy", "stdout_capture_policy",
    }
    if type(value) is not dict or set(value) != expected_top or value.get("schema_version") != "hegel-phase3a-q05b-dual-isolation/1":
        _fail("REJECT_Q05B_ARTIFACT_SOURCE", "isolation config differs")
    try:
        admitted_config = _admission.validate_commit_a_actual_config_bytes_v1(
            payload_table[path][2]
        )
    except _admission.Q05BActualAdmissionError as error:
        _fail("REJECT_Q05B_ARTIFACT_SOURCE", error.detail)
    if admitted_config != value:
        _fail("REJECT_Q05B_ARTIFACT_SOURCE", "isolation config replay differs")
    actual = value.get("actual_preconditions")
    authority = value.get("dry_run_authority")
    receipt = value.get("qualification_receipt_protocol")
    artifact = value.get("artifact_layout")
    if (
        type(actual) is not dict
        or type(authority) is not dict
        or authority.get("active_transition_allowed") is not False
        or authority.get("artifact_written") is not False
        or authority.get("formal_fixed_point_claimed") is not False
        or authority.get("m3_formal_roots") is not None
        or authority.get("outside_certificate_issued") is not False
        or authority.get("q1_formal_output_roots") != [None] * 8
        or authority.get("q1_gate_count") != 0
        or type(authority.get("q1_gate_count")) is not int
        or authority.get("q1_gate_mask") != 0
        or type(authority.get("q1_gate_mask")) is not int
        or authority.get("q1_gate_total") != 20
        or type(authority.get("q1_gate_total")) is not int
        or authority.get("q1_receipt") is not None
        or authority.get("q1_state") != "NOT_RUN"
        or authority.get("q2_state") != "NOT_RUN"
        or type(receipt) is not dict
        or receipt.get("candidate_count") != 19
        or type(receipt.get("candidate_count")) is not int
        or receipt.get("candidate_mask") != 0x7FFFF
        or type(receipt.get("candidate_mask")) is not int
        or receipt.get("final_count") != 20
        or type(receipt.get("final_count")) is not int
        or receipt.get("final_mask") != 0xFFFFF
        or type(receipt.get("final_mask")) is not int
        or value.get("claim_scope") != "Q05B_TARGET_BLIND_QUALIFICATION_ONLY"
        or value.get("profile_id") != "hegel-phase3a-q05b-three-actor-offline-qualification-v1"
        or type(artifact) is not dict
        or artifact.get("artifact_set_root_domain") != ARTIFACT_SET_ROOT_DOMAIN
        or artifact.get("format") != "ONE_CANONICAL_JSON_OBJECT_PLUS_LF"
        or artifact.get("mode") != "0444"
        or artifact.get("atomic_publication") != "DIRFD_NOFOLLOW_FSYNC_LINK_NOREPLACE_UNLINK_FSYNC"
    ):
        _fail("REJECT_Q05B_ARTIFACT_SOURCE", "actual isolation authority differs")
    expected_docker = {
        "attempt_nonce_length_bytes": 32,
        "auto_remove": False,
        "build_tmpfs": "/tmp:rw,exec,nosuid,nodev,size=8g,mode=1777",
        "cap_drop": "ALL",
        "cgroup_namespace": "private",
        "command_label_policy": "EXACTLY_THREE_RESERVED_LABELS_IN_FROZEN_ORDER",
        "config_label_policy": "PINNED_IMAGE_BASE_LABELS_UNION_EXACT_THREE_RESERVED_LABELS",
        "container_name_usage": "READ_ONLY_DISCOVERY_ONLY",
        "destructive_target": "OWNERSHIP_VALIDATED_64_LOWERHEX_CONTAINER_ID_ONLY",
        "docker_inventory_baseline_scope": "RUN_AUDIT_ONLY_NOT_ADMISSION_OR_CLAIM_EVIDENCE",
        "docker_socket_mounted_into_actor": False,
        "executable": "/usr/bin/docker",
        "execution_slot_rows": [
            [1, "RUST_TEST", "rust-test"],
            [2, "RUST_RELEASE", "rust-release"],
            [3, "PYTHON_ENDPOINT", "python"],
            [4, "RUST_ENDPOINT", "rust"],
            [5, "TRUSTED_HOST_REPLAY", "host"],
        ],
        "explicit_remove_after_post_exit_inspect": True,
        "foreign_name_collision_policy": "ZERO_MUTATION_FAIL_CLOSED",
        "host": "unix:///var/run/docker.sock",
        "initial_name_absence_sample_count": 2,
        "ipc": "none",
        "memory": "14g",
        "memory_swap": "14g",
        "network": "none",
        "no_new_privileges": True,
        "nofile_ulimit": "256:256",
        "ownership_namespace_derivation": "SHA256(DOMAIN_NUL_ATTEMPT_NONCE_32_SOURCE_COMMIT_ASCII_40)",
        "ownership_namespace_domain_ascii": "HEGEL/Q05B/DOCKER/OWNERSHIP_NAMESPACE/V1",
        "pids_limit": 128,
        "precreate_name_absence_sample_count": 2,
        "pull_policy": "never",
        "python_pinned_image_base_label_rows": [],
        "remove_by_container_name_forbidden": True,
        "reserved_label_keys": list(_admission.DOCKER_RESERVED_LABEL_KEYS),
        "root_filesystem_read_only": True,
        "run_as_caller_uid_gid": True,
        "runtime_tmpfs": "/tmp:rw,noexec,nosuid,nodev,size=2g,mode=1777",
        "rust_pinned_image_base_label_rows": [
            list(row) for row in _admission.DOCKER_RUST_BASE_LABEL_ROWS
        ],
        "unique_container_name_template": "hegel-q05b-{FULL64_EXECUTION_NAMESPACE}-{SLOT_SUFFIX}",
        "unknown_daemon_or_ownership_state_policy": "ZERO_MUTATION_FAIL_CLOSED",
    }
    expected_images = {"python_endpoint": "python@sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3", "rust_build": "rust@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89", "rust_runtime": "rust@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89", "trusted_host": "python@sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3"}
    expected_roles = [[1, "PYTHON_ENDPOINT", "0-11", "ENDPOINTS_PARALLEL", "14g", "14g", 128, "256:256"], [2, "RUST_ENDPOINT", "12-23", "ENDPOINTS_PARALLEL", "14g", "14g", 128, "256:256"], [3, "TRUSTED_HOST_REPLAY", "0-11", "AFTER_BOTH_ENDPOINTS_EXIT", "14g", "14g", 128, "256:256"]]
    try:
        _require_type_exact_v1(value.get("docker"), expected_docker, "docker")
        _require_type_exact_v1(value.get("images"), expected_images, "images")
        _require_type_exact_v1(value.get("resource_roles"), expected_roles, "resource_roles")
    except Q05BActualArtifactError:
        _fail("REJECT_Q05B_ARTIFACT_SOURCE", "claim-critical Docker profile differs")
    return value


def _validate_pinned_image_rows_v1(
    value: object, config: dict[str, object]
) -> dict[str, dict[str, object]]:
    if type(value) is not list or len(value) != 2:
        _fail("REJECT_Q05B_ARTIFACT_SOURCE", "pinned image row registry differs")
    expected = (("python", config["images"]["python_endpoint"]), ("rust", config["images"]["rust_build"]))
    result: dict[str, dict[str, object]] = {}
    keys = {"architecture", "evidence_sha256", "image_id", "os", "raw_inspect_hex", "raw_inspect_sha256", "repo_digests", "requested_reference", "schema_version"}
    for raw, (label, reference) in zip(value, expected, strict=True):
        if type(raw) is not list or len(raw) != 2 or raw[0] != label:
            _fail("REJECT_Q05B_ARTIFACT_SOURCE", "pinned image row order differs")
        evidence = _object(raw[1], keys, f"{label} pinned image")
        payload = _hex_any(evidence["raw_inspect_hex"], f"{label} image inspect")
        decoded = _decoded_json_value(payload, f"{label} image inspect")
        if type(decoded) is not list or len(decoded) != 1 or type(decoded[0]) is not dict:
            _fail("REJECT_Q05B_ARTIFACT_SOURCE", "pinned image inspect shape differs")
        document = decoded[0]; image_config = document.get("Config")
        body = dict(evidence); observed_root = body.pop("evidence_sha256")
        if (
            evidence["schema_version"] != "hegel-phase3a-q05b-pinned-local-image-evidence/1"
            or evidence["requested_reference"] != reference
            or type(evidence["image_id"]) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}", evidence["image_id"]) is None
            or evidence["image_id"] != document.get("Id")
            or evidence["repo_digests"] != document.get("RepoDigests")
            or type(evidence["repo_digests"]) is not list
            or reference not in evidence["repo_digests"]
            or any(type(item) is not str for item in evidence["repo_digests"])
            or evidence["os"] != "linux" or document.get("Os") != "linux"
            or type(evidence["architecture"]) is not str or not evidence["architecture"]
            or evidence["architecture"] != document.get("Architecture")
            or type(image_config) is not dict or type(image_config.get("Env")) is not list
            or any(type(item) is not str for item in image_config["Env"])
            or evidence["raw_inspect_sha256"] != sha256(payload).hexdigest()
            or observed_root != sha256(_canonical_json(body)).hexdigest()
        ):
            _fail("REJECT_Q05B_ARTIFACT_SOURCE", f"pinned {label} image differs")
        result[label] = evidence
    if config["images"]["python_endpoint"] != config["images"]["trusted_host"] or config["images"]["rust_build"] != config["images"]["rust_runtime"]:
        _fail("REJECT_Q05B_ARTIFACT_SOURCE", "actor/build image registry differs")
    return result


_ACTUAL_STAGE_1_TO_3_NAMES: Final = (
    "FREEZE_SOURCE_IMAGES_AND_COMMAND_REGISTRY",
    "MATERIALIZE_AND_SEAL_THREE_SOURCE_SNAPSHOTS_AND_CARGO",
    "OFFLINE_RUST_TEST_AND_RELEASE_BUILD_AND_SEAL_BINARY",
)

_ACTUAL_STAGE_1_EVIDENCE_KEYS: Final = frozenset({
    "config_hex",
    "config_sha256",
    "fixed_artifact_path",
    "layout",
    "cargo_cache_source",
    "cargo_cache_root_identity",
    "source_evidence",
    "source_object_closure",
    "image_evidence",
    "planned_commands",
    "docker_execution_authority",
    "q1_authority",
})
_ACTUAL_STAGE_2_EVIDENCE_KEYS: Final = frozenset({
    "snapshot_evidence",
    "cargo_lock_hex",
    "cargo_lock_sha256",
    "cargo_evidence",
    "seccomp_evidence",
})


def _actual_prior_stage_rows_v1(
    value: object,
    source_commit: str,
) -> list[dict[str, object]]:
    if type(value) is not list or len(value) != 3:
        _fail(
            "REJECT_Q05B_ARTIFACT_ADMISSION",
            "actual admission prior-stage registry differs",
        )
    keys = {
        "candidate_receipt_hex", "evidence", "final_receipt_hex",
        "q1_authority", "qualification_count", "qualification_mask",
        "schema_version", "source_commit", "stage_evidence_root",
        "stage_id", "stage_name", "status",
    }
    expected_q1 = {
        "certificate_active": False,
        "formal_output_roots": [None] * 8,
        "gate_count": 0,
        "gate_mask": 0,
        "state": "NOT_RUN",
    }
    result: list[dict[str, object]] = []
    for expected_id, (raw, expected_name) in enumerate(
        zip(value, _ACTUAL_STAGE_1_TO_3_NAMES, strict=True), start=1
    ):
        row = _object(raw, keys, f"actual admission stage {expected_id}")
        if (
            row["schema_version"] != _admission.ACTUAL_STAGE_SCHEMA_VERSION
            or row["source_commit"] != source_commit
            or type(row["stage_id"]) is not int
            or row["stage_id"] != expected_id
            or row["stage_name"] != expected_name
            or row["status"] != "STAGE_COMPLETE_IN_MEMORY_NOT_PUBLISHED"
            or type(row["qualification_count"]) is not int
            or row["qualification_count"] != 0
            or type(row["qualification_mask"]) is not int
            or row["qualification_mask"] != 0
            or row["candidate_receipt_hex"] is not None
            or row["final_receipt_hex"] is not None
            or type(row["evidence"]) is not dict
            or (
                expected_id == 1
                and set(row["evidence"]) != _ACTUAL_STAGE_1_EVIDENCE_KEYS
            )
            or (
                expected_id == 2
                and set(row["evidence"]) != _ACTUAL_STAGE_2_EVIDENCE_KEYS
            )
        ):
            _fail(
                "REJECT_Q05B_ARTIFACT_ADMISSION",
                f"actual admission stage {expected_id} identity differs",
            )
        _require_type_exact_v1(
            row["q1_authority"], expected_q1,
            f"actual admission stage {expected_id} Q1 authority",
        )
        body = dict(row)
        observed_root = body.pop("stage_evidence_root")
        expected_root = sha256(
            _admission.ACTUAL_STAGE_EVIDENCE_ROOT_DOMAIN
            + expected_id.to_bytes(2, "big")
            + _admission.canonical_json_bytes_v1(body)
        ).hexdigest()
        if observed_root != expected_root:
            _fail(
                "REJECT_Q05B_ARTIFACT_ADMISSION",
                f"actual admission stage {expected_id} root differs",
            )
        result.append(row)
    return result


def _consume_work_root_replay_v1(
    value: object,
    work_root_identity: Mapping[str, object],
) -> dict[str, object]:
    replay = _object(
        value,
        {
            "schema_version", "absolute_path", "device", "inode", "nlink",
            "mode", "path_matches_anchored_descriptor",
        },
        "actual admission consume work-root replay",
    )
    try:
        work = _admission.validate_work_root_identity_v1(work_root_identity)
    except _admission.Q05BActualAdmissionError as error:
        _fail("REJECT_Q05B_ARTIFACT_ADMISSION", error.detail)
    if (
        replay["schema_version"]
        != "hegel-phase3a-q05b-admission-work-root-replay/1"
        or replay["absolute_path"] != work["absolute_path"]
        or type(replay["device"]) is not int
        or replay["device"] != work["device"]
        or type(replay["inode"]) is not int
        or replay["inode"] != work["inode"]
        or type(replay["nlink"]) is not int
        or replay["nlink"] != work["nlink"]
        or type(replay["mode"]) is not int
        or replay["mode"] != 0o700
        or replay["path_matches_anchored_descriptor"] is not True
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_ADMISSION",
            "actual admission consume work-root replay differs",
        )
    return replay


def _checkpoint_dynamic_authority_v1(
    value: object,
    source_commit: str,
    stage_5_evidence: Mapping[str, object],
    issue_record: Mapping[str, object],
    consumed_marker: Mapping[str, object],
    checkpoint_1: Mapping[str, object],
    stage_5_mount_launch_replay_rows: object,
    five_sidecars: Mapping[str, object],
    endpoint_stdout_set: Mapping[str, object],
) -> dict[str, object]:
    try:
        dynamic = _admission.validate_dynamic_mount_authority_set_v1(
            value,
            source_commit,
            stage_5_evidence,
            issue_record=issue_record,
            consumed_marker_evidence=consumed_marker,
            checkpoint_1=checkpoint_1,
            mount_launch_replay_rows=stage_5_mount_launch_replay_rows,
        )
    except _admission.Q05BActualAdmissionError as error:
        _fail("REJECT_Q05B_ARTIFACT_ADMISSION", error.detail)
    if any(
        _admission.canonical_json_bytes_v1(observed)
        != _admission.canonical_json_bytes_v1(expected)
        for observed, expected in (
            (dynamic["python_output_tree"], five_sidecars["python_output_tree"]),
            (dynamic["rust_output_tree"], five_sidecars["rust_output_tree"]),
            (dynamic["stdout_tree"], endpoint_stdout_set["sealed_stdout_tree"]),
        )
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_ADMISSION",
            "fresh checkpoint Stage-5 external authority differs",
        )
    return dynamic


def _fresh_runtime_checkpoint_rows_v1(
    value: object,
    *,
    source_commit: str,
    artifact_path: str,
    issue_record: Mapping[str, object],
    consumed_marker: Mapping[str, object],
    issue_fresh_runtime: Mapping[str, object],
    consume_absence: Mapping[str, object],
    stage_5_evidence: Mapping[str, object],
    stage_5_mount_launch_replay_rows: object,
    five_sidecars: Mapping[str, object],
    endpoint_stdout_set: Mapping[str, object],
) -> list[dict[str, object]]:
    if type(value) is not list or len(value) != 3:
        _fail(
            "REJECT_Q05B_ARTIFACT_ADMISSION",
            "fresh checkpoint registry differs",
        )
    keys = {
        "schema_version", "source_commit", "artifact_path", "checkpoint_id",
        "checkpoint_name", "attempt_id", "boundary_root",
        "issue_record_root", "consumed_marker_root",
        "issue_fresh_runtime_evidence_root",
        "issue_fresh_runtime_evidence_sha256",
        "observed_fresh_runtime_evidence",
        "observed_fresh_runtime_evidence_root",
        "observed_fresh_runtime_evidence_sha256", "canonical_sets_byte_equal",
        "artifact_absence_evidence", "mount_binding_rows",
        "mount_registry_root", "dynamic_authority_set",
        "dynamic_authority_root", "checkpoint_root",
    }
    issue_bytes = _admission.canonical_json_bytes_v1(issue_fresh_runtime)
    result: list[dict[str, object]] = []
    prior_dynamic: dict[str, object] | None = None
    for raw, (checkpoint_id, checkpoint_name) in zip(
        value, _admission.ACTUAL_FRESH_RUNTIME_CHECKPOINT_REGISTRY, strict=True
    ):
        checkpoint = _object(
            raw, keys, f"fresh runtime checkpoint {checkpoint_id}"
        )
        try:
            observed = _admission.validate_fresh_runtime_evidence_set_v1(
                checkpoint["observed_fresh_runtime_evidence"], source_commit
            )
            bindings = [
                _admission.validate_actor_mount_binding_v1(row)
                for row in checkpoint["mount_binding_rows"]
            ] if type(checkpoint["mount_binding_rows"]) is list else []
            mount_root = _admission.checkpoint_mount_registry_root_v1(
                checkpoint_id, bindings
            )
            absence = _admission.validate_artifact_absence_evidence_v1(
                checkpoint["artifact_absence_evidence"], artifact_path
            )
        except _admission.Q05BActualAdmissionError as error:
            _fail("REJECT_Q05B_ARTIFACT_ADMISSION", error.detail)
        expected_roles = {
            1: (1, 2),
            2: (3,),
            3: (1, 2, 3),
        }[checkpoint_id]
        if (
            checkpoint["schema_version"]
            != _admission.ACTUAL_FRESH_RUNTIME_CHECKPOINT_SCHEMA_VERSION
            or checkpoint["source_commit"] != source_commit
            or checkpoint["artifact_path"] != artifact_path
            or type(checkpoint["checkpoint_id"]) is not int
            or checkpoint["checkpoint_id"] != checkpoint_id
            or checkpoint["checkpoint_name"] != checkpoint_name
            or checkpoint["attempt_id"] != issue_record["attempt_id"]
            or checkpoint["boundary_root"] != issue_record["boundary_root"]
            or checkpoint["issue_record_root"]
            != issue_record["issue_record_root"]
            or checkpoint["consumed_marker_root"]
            != consumed_marker["consumed_marker_root"]
            or checkpoint["issue_fresh_runtime_evidence_root"]
            != issue_fresh_runtime["fresh_runtime_evidence_root"]
            or checkpoint["issue_fresh_runtime_evidence_sha256"]
            != sha256(issue_bytes).hexdigest()
            or _admission.canonical_json_bytes_v1(observed) != issue_bytes
            or checkpoint["observed_fresh_runtime_evidence_root"]
            != issue_fresh_runtime["fresh_runtime_evidence_root"]
            or checkpoint["observed_fresh_runtime_evidence_sha256"]
            != sha256(issue_bytes).hexdigest()
            or checkpoint["canonical_sets_byte_equal"] is not True
            or absence != consume_absence
            or tuple(row["role_id"] for row in bindings) != expected_roles
            or checkpoint["mount_registry_root"] != mount_root
        ):
            _fail(
                "REJECT_Q05B_ARTIFACT_ADMISSION",
                f"fresh runtime checkpoint {checkpoint_id} differs",
            )
        if checkpoint_id == 1:
            if (
                checkpoint["dynamic_authority_set"] is not None
                or checkpoint["dynamic_authority_root"] is not None
            ):
                _fail(
                    "REJECT_Q05B_ARTIFACT_ADMISSION",
                    "endpoint checkpoint carried Stage-5 authority",
                )
        else:
            dynamic = _checkpoint_dynamic_authority_v1(
                checkpoint["dynamic_authority_set"],
                source_commit,
                stage_5_evidence,
                issue_record,
                consumed_marker,
                result[0],
                stage_5_mount_launch_replay_rows,
                five_sidecars,
                endpoint_stdout_set,
            )
            if checkpoint["dynamic_authority_root"] != dynamic["dynamic_authority_root"]:
                _fail(
                    "REJECT_Q05B_ARTIFACT_ADMISSION",
                    "fresh checkpoint dynamic authority root differs",
                )
            if prior_dynamic is not None and dynamic != prior_dynamic:
                _fail(
                    "REJECT_Q05B_ARTIFACT_ADMISSION",
                    "fresh checkpoint dynamic authority bytes differ",
                )
            prior_dynamic = dynamic
            role3 = bindings[-1]
            sources = {row["destination"]: row for row in role3["source_rows"]}
            if (
                sources.get("/inputs/python", {}).get("authority_evidence")
                != dynamic["python_output_tree"]
                or sources.get("/inputs/rust", {}).get("authority_evidence")
                != dynamic["rust_output_tree"]
            ):
                _fail(
                    "REJECT_Q05B_ARTIFACT_ADMISSION",
                    "host checkpoint external tree authority differs",
                )
        body = dict(checkpoint)
        checkpoint_root = body.pop("checkpoint_root")
        if checkpoint_root != sha256(
            _admission.ACTUAL_FRESH_RUNTIME_CHECKPOINT_ROOT_DOMAIN
            + checkpoint_id.to_bytes(1, "big")
            + _admission.canonical_json_bytes_v1(body)
        ).hexdigest():
            _fail(
                "REJECT_Q05B_ARTIFACT_ADMISSION",
                f"fresh runtime checkpoint {checkpoint_id} root differs",
            )
        result.append(checkpoint)
    return result


def _cross_stage_5_endpoint_completion_authority_v1(
    actor_completion_rows: object,
    bindings: list[dict[str, object]],
    launches: list[dict[str, object]],
) -> None:
    if (
        type(actor_completion_rows) is not list
        or len(actor_completion_rows) != 2
        or len(bindings) != 2
        or len(launches) != 2
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_ADMISSION",
            "Stage-5 endpoint completion authority registry differs",
        )
    for completion, binding, launch in zip(
        actor_completion_rows, bindings, launches, strict=True
    ):
        registry = binding["command_mount_registry"]
        if (
            type(completion) is not dict
            or completion.get("actor_id") != binding["actor_id"]
            or completion.get("actor_id") != launch["actor_id"]
            or completion.get("command_sha256") != registry["command_sha256"]
            or completion.get("command_sha256") != launch["command_sha256"]
            or completion.get("mount_registry_sha256")
            != registry["registry_sha256"]
            or completion.get("mount_registry_sha256")
            != launch["mount_registry_sha256"]
            or type(completion.get("seccomp_evidence")) is not dict
            or _admission.canonical_json_bytes_v1(
                completion["seccomp_evidence"]
            ) != _admission.canonical_json_bytes_v1(
                binding["seccomp_row"]["authority_evidence"]
            )
        ):
            _fail(
                "REJECT_Q05B_ARTIFACT_ADMISSION",
                "Stage-5 endpoint completion authority differs",
            )


def _reconstruct_actual_stage_5_evidence_v1(
    *,
    source_commit: str,
    actor_completion_rows: object,
    five_sidecars: Mapping[str, object],
    endpoint_stdout_set: Mapping[str, object],
    strict_endpoint_replay_roots: object,
    issue_record: Mapping[str, object],
    consumed_marker: Mapping[str, object],
    consume_work_root_replay: Mapping[str, object],
    consume_git_source_transcript: Mapping[str, object],
    consume_artifact_absence_evidence: Mapping[str, object],
    fresh_runtime_checkpoint_rows: object,
    stage_5_live_marker_replay: object,
    stage_5_mount_launch_replay_rows: object,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    """Reconstruct the full concrete `_stage(5)` row without sidecar copies."""

    if (
        type(actor_completion_rows) is not list
        or len(actor_completion_rows) != 2
        or any(type(row) is not dict for row in actor_completion_rows)
        or type(strict_endpoint_replay_roots) is not list
        or len(strict_endpoint_replay_roots) != 2
        or type(fresh_runtime_checkpoint_rows) is not list
        or len(fresh_runtime_checkpoint_rows) != 3
        or type(fresh_runtime_checkpoint_rows[0]) is not dict
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_ADMISSION",
            "Stage-5 external reconstruction registry differs",
        )
    checkpoint_1 = fresh_runtime_checkpoint_rows[0]
    mount_rows = checkpoint_1.get("mount_binding_rows")
    if type(mount_rows) is not list or len(mount_rows) != 2:
        _fail(
            "REJECT_Q05B_ARTIFACT_ADMISSION",
            "Stage-5 checkpoint-1 mount registry differs",
        )
    try:
        bindings = [
            _admission.validate_actor_mount_binding_v1(row) for row in mount_rows
        ]
        if [row["role_id"] for row in bindings] != [1, 2]:
            raise _admission.Q05BActualAdmissionError(
                "Stage-5 checkpoint-1 mount role order differs"
            )
        if (
            type(stage_5_mount_launch_replay_rows) is not list
            or len(stage_5_mount_launch_replay_rows) != 2
        ):
            raise _admission.Q05BActualAdmissionError(
                "Stage-5 launch replay registry differs"
            )
        launches = [
            _admission.validate_actor_mount_launch_replay_v1(launch, binding)
            for launch, binding in zip(
                stage_5_mount_launch_replay_rows, bindings, strict=True
            )
        ]
        live = _admission.validate_actual_admission_live_marker_replay_surface_v1(
            stage_5_live_marker_replay,
            "STAGE_05_BEFORE_EVIDENCE",
            issue_record,
            consumed_marker,
        )
    except _admission.Q05BActualAdmissionError as error:
        _fail("REJECT_Q05B_ARTIFACT_ADMISSION", error.detail)
    checkpoint_root = checkpoint_1.get("checkpoint_root")
    _hex(checkpoint_root, 32, "Stage-5 checkpoint-1 root")
    injected = {
        "actual_admission_attempt_id": issue_record["attempt_id"],
        "actual_admission_boundary_root": issue_record["boundary_root"],
        "actual_admission_issue_record_root": issue_record["issue_record_root"],
        "actual_admission_consumed_marker_evidence": dict(consumed_marker),
        "actual_admission_work_root_replay": dict(consume_work_root_replay),
        "actual_admission_consume_git_source_transcript": dict(
            consume_git_source_transcript
        ),
        "actual_admission_consume_artifact_absence": dict(
            consume_artifact_absence_evidence
        ),
        "actual_admission_fresh_checkpoint_root_rows": [
            [
                1,
                _admission.ACTUAL_FRESH_RUNTIME_CHECKPOINT_REGISTRY[0][1],
                checkpoint_root,
            ]
        ],
        "actual_actor_mount_binding_root_rows": [
            [row["role_id"], row["actor_id"], row["mount_binding_root"]]
            for row in bindings
        ],
        "actual_actor_mount_launch_root_rows": [
            [row["role_id"], row["actor_id"], row["launch_replay_root"]]
            for row in launches
        ],
        "actual_admission_live_marker_replay": live,
    }
    try:
        stage_5 = _admission.build_actual_stage_5_evidence_v1(
            source_commit,
            actor_completion_rows,
            five_sidecars,
            endpoint_stdout_set,
            strict_endpoint_replay_roots,
            injected,
        )
        stage_5 = _admission.validate_actual_stage_5_evidence_v1(
            stage_5,
            source_commit,
            issue_record=issue_record,
            consumed_marker_evidence=consumed_marker,
            checkpoint_1=checkpoint_1,
            mount_launch_replay_rows=launches,
        )
    except _admission.Q05BActualAdmissionError as error:
        _fail("REJECT_Q05B_ARTIFACT_ADMISSION", error.detail)
    _cross_stage_5_endpoint_completion_authority_v1(
        actor_completion_rows, bindings, launches
    )
    return stage_5, live, launches


def build_actual_admission_artifact_evidence_v1(
    *,
    source_commit: str,
    artifact_path: str,
    commit_a_config_bytes: bytes,
    commit_a_config_git_blob_oid: str,
    prior_stage_evidence_rows: object,
    issue_record: object,
    consumed_marker_evidence: object,
    consume_work_root_replay: object,
    consume_git_source_transcript: object,
    consume_artifact_absence_evidence: object,
    fresh_runtime_checkpoint_rows: object,
    pre_artifact_live_marker_replay: object,
    anti_replay_scope: object,
    stage_5_evidence: Mapping[str, object] | None,
    stage_5_actor_completion_rows: object,
    stage_5_strict_endpoint_replay_roots: object,
    stage_5_live_marker_replay: object,
    stage_5_mount_launch_replay_rows: object,
    five_sidecars: Mapping[str, object],
    endpoint_stdout_set: Mapping[str, object],
) -> dict[str, object]:
    """Build/replay the causal admission section ending before Stage 8."""

    if (
        type(source_commit) is not str
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        or type(artifact_path) is not str
        or not artifact_path.startswith("/")
        or ".." in artifact_path.split("/")
        or type(commit_a_config_git_blob_oid) is not str
        or re.fullmatch(r"[0-9a-f]{40}", commit_a_config_git_blob_oid) is None
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_ADMISSION",
            "actual admission source/config identity differs",
        )
    try:
        _admission.validate_commit_a_actual_config_bytes_v1(
            commit_a_config_bytes
        )
        stages = _actual_prior_stage_rows_v1(
            prior_stage_evidence_rows, source_commit
        )
        record, boundary_surface = (
            _admission.validate_actual_admission_issue_record_v1(issue_record)
        )
        consumed = (
            _admission.validate_actual_admission_consumed_marker_evidence_v1(
                consumed_marker_evidence, record
            )
        )
        _require_type_exact_v1(
            anti_replay_scope,
            _admission.ACTUAL_ADMISSION_RUN_LOCAL_ANTI_REPLAY_SCOPE,
            "actual admission anti-replay scope",
        )
        bundle_payload = bytes.fromhex(boundary_surface["precondition_bundle_hex"])
        decision_payload = bytes.fromhex(boundary_surface["decision_hex"])
        bundle_value = _decoded_json_value(
            bundle_payload, "actual admission precondition bundle"
        )
        if type(bundle_value) is not dict:
            raise _admission.Q05BActualAdmissionError(
                "actual admission bundle is not one object"
            )
        bundle = _admission.validate_actual_precondition_bundle_object_v1(
            bundle_value,
            source_commit,
            commit_a_config_bytes,
            artifact_path,
        )
        decision = _admission.decode_actual_admission_decision_v1(
            decision_payload,
            commit_a_config_bytes,
            source_commit,
            artifact_path,
            bundle,
        )
        boundary = _admission.decode_stage3_to4_admission_boundary_v1(
            _admission.canonical_json_bytes_v1(boundary_surface),
            source_commit,
            commit_a_config_bytes,
            artifact_path,
            bundle,
            decision,
        )
        consume_git = _admission.validate_git_source_transcript_v1(
            consume_git_source_transcript, source_commit
        )
        consume_absence = _admission.validate_artifact_absence_evidence_v1(
            consume_artifact_absence_evidence, artifact_path
        )
        live = (
            _admission.validate_actual_admission_live_marker_replay_surface_v1(
                pre_artifact_live_marker_replay,
                "PRE_ARTIFACT_ASSEMBLY",
                record,
                consumed,
            )
        )
    except (KeyError, TypeError, ValueError, _admission.Q05BActualAdmissionError) as error:
        _fail(
            "REJECT_Q05B_ARTIFACT_ADMISSION",
            getattr(error, "detail", "actual admission replay differs"),
        )
    stage_roots = [[row["stage_id"], row["stage_evidence_root"]] for row in stages]
    if bundle["prior_stage_root_rows"] != stage_roots:
        _fail(
            "REJECT_Q05B_ARTIFACT_ADMISSION",
            "actual admission bundle/stage roots differ",
        )
    stage1_config_hex = stages[0]["evidence"].get("config_hex")
    if (
        type(stage1_config_hex) is not str
        or stage1_config_hex != commit_a_config_bytes.hex()
        or boundary["source_commit"] != source_commit
        or boundary["artifact_path"] != artifact_path
        or record["anti_replay_scope"] != anti_replay_scope
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_ADMISSION",
            "actual admission config/boundary cross differs",
        )
    issue_git = bundle["ordered_precondition_rows"][0]["preimage"][
        "git_source_transcript"
    ]
    issue_absence = bundle["ordered_precondition_rows"][3]["preimage"][
        "artifact_absence_evidence"
    ]
    if consume_git != issue_git or consume_absence != issue_absence:
        _fail(
            "REJECT_Q05B_ARTIFACT_ADMISSION",
            "consume Git/artifact absence differs from issue bytes",
        )
    work_replay = _consume_work_root_replay_v1(
        consume_work_root_replay, bundle["work_root_identity"]
    )
    if (
        work_replay["device"] != consumed["work_root_device"]
        or work_replay["inode"] != consumed["work_root_inode"]
        or live["work_root_device"] != work_replay["device"]
        or live["work_root_inode"] != work_replay["inode"]
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_ADMISSION",
            "consume/live work-root authority differs",
        )
    row_preimages = [
        row["preimage"] for row in bundle["ordered_precondition_rows"]
    ]
    try:
        issue_fresh = _admission.build_fresh_runtime_evidence_set_v1(
            source_commit,
            row_preimages[4]["image_rows"],
            row_preimages[5]["actor_rows"],
            row_preimages[6]["cargo_material_identity"],
            row_preimages[6]["cargo_snapshot_evidence"],
            row_preimages[6]["cargo_tree_evidence"],
            row_preimages[7]["seccomp_rows"],
            row_preimages[7]["binary_identity"],
        )
    except _admission.Q05BActualAdmissionError as error:
        _fail("REJECT_Q05B_ARTIFACT_ADMISSION", error.detail)
    reconstructed_stage_5, stage_5_live, stage_5_launches = (
        _reconstruct_actual_stage_5_evidence_v1(
            source_commit=source_commit,
            actor_completion_rows=stage_5_actor_completion_rows,
            five_sidecars=five_sidecars,
            endpoint_stdout_set=endpoint_stdout_set,
            strict_endpoint_replay_roots=stage_5_strict_endpoint_replay_roots,
            issue_record=record,
            consumed_marker=consumed,
            consume_work_root_replay=work_replay,
            consume_git_source_transcript=consume_git,
            consume_artifact_absence_evidence=consume_absence,
            fresh_runtime_checkpoint_rows=fresh_runtime_checkpoint_rows,
            stage_5_live_marker_replay=stage_5_live_marker_replay,
            stage_5_mount_launch_replay_rows=stage_5_mount_launch_replay_rows,
        )
    )
    if stage_5_evidence is not None and (
        type(stage_5_evidence) is not dict
        or _admission.canonical_json_bytes_v1(stage_5_evidence)
        != _admission.canonical_json_bytes_v1(reconstructed_stage_5)
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_ADMISSION",
            "concrete/reconstructed full Stage-5 bytes differ",
        )
    checkpoints = _fresh_runtime_checkpoint_rows_v1(
        fresh_runtime_checkpoint_rows,
        source_commit=source_commit,
        artifact_path=artifact_path,
        issue_record=record,
        consumed_marker=consumed,
        issue_fresh_runtime=issue_fresh,
        consume_absence=consume_absence,
        stage_5_evidence=reconstructed_stage_5,
        stage_5_mount_launch_replay_rows=stage_5_launches,
        five_sidecars=five_sidecars,
        endpoint_stdout_set=endpoint_stdout_set,
    )
    config_binding = {
        "relative_path": "config/phase3_q05b_dual_isolation_v1.json",
        "git_blob_oid": commit_a_config_git_blob_oid,
        "raw_length": len(commit_a_config_bytes),
        "raw_sha256": sha256(commit_a_config_bytes).hexdigest(),
    }
    marker = record["issued_marker_evidence"]
    spending = _decoded_json_value(
        bytes.fromhex(consumed["spending_intent_hex"]),
        "actual admission spending intent",
    )
    root_registry = {
        "commit_a_config_sha256": config_binding["raw_sha256"],
        "stage_1_root": stages[0]["stage_evidence_root"],
        "stage_2_root": stages[1]["stage_evidence_root"],
        "stage_3_root": stages[2]["stage_evidence_root"],
        "precondition_registry_root": bundle["precondition_registry_root"],
        "issue_git_source_transcript_root": issue_git["transcript_root"],
        "issue_fresh_runtime_evidence_root": issue_fresh[
            "fresh_runtime_evidence_root"
        ],
        "precondition_bundle_root": bundle["bundle_root"],
        "decision_root": decision["decision_root"],
        "boundary_root": boundary["boundary_root"],
        "issued_marker_root": marker["issued_marker_root"],
        "issue_record_root": record["issue_record_root"],
        "spending_intent_root": spending["spending_intent_root"],
        "consumed_marker_root": consumed["consumed_marker_root"],
        "consume_git_source_transcript_root": consume_git["transcript_root"],
        "consume_after_spend_before_endpoints_checkpoint_root": checkpoints[0][
            "checkpoint_root"
        ],
        "stage6_before_host_launch_checkpoint_root": checkpoints[1][
            "checkpoint_root"
        ],
        "stage7_before_predicate19_checkpoint_root": checkpoints[2][
            "checkpoint_root"
        ],
        "stage_5_evidence_root": reconstructed_stage_5[
            "stage_evidence_root"
        ],
        "stage5_mount_binding_root_rows": [
            [row["role_id"], row["actor_id"], row["mount_binding_root"]]
            for row in checkpoints[0]["mount_binding_rows"]
        ],
        "stage5_live_marker_replay_root": stage_5_live[
            "live_marker_replay_root"
        ],
        "stage5_mount_launch_replay_root_rows": [
            [row["role_id"], row["actor_id"], row["launch_replay_root"]]
            for row in stage_5_launches
        ],
        "dynamic_authority_root": checkpoints[1]["dynamic_authority_root"],
        "pre_artifact_live_marker_replay_root": live[
            "live_marker_replay_root"
        ],
    }
    body: dict[str, object] = {
        "schema_version": ACTUAL_ADMISSION_ARTIFACT_EVIDENCE_SCHEMA_VERSION,
        "source_commit": source_commit,
        "artifact_path": artifact_path,
        "commit_a_config_binding": config_binding,
        "prior_stage_evidence_rows": stages,
        "issue_record": record,
        "consumed_marker_evidence": consumed,
        "consume_work_root_replay": work_replay,
        "consume_git_source_transcript": consume_git,
        "consume_artifact_absence_evidence": consume_absence,
        "fresh_runtime_checkpoint_rows": checkpoints,
        "stage_5_live_marker_replay": stage_5_live,
        "stage_5_mount_launch_replay_rows": stage_5_launches,
        "pre_artifact_live_marker_replay": live,
        "anti_replay_scope": dict(_admission.ACTUAL_ADMISSION_RUN_LOCAL_ANTI_REPLAY_SCOPE),
        "root_registry": root_registry,
    }
    result = dict(body)
    result["actual_admission_evidence_root"] = sha256(
        ACTUAL_ADMISSION_ARTIFACT_EVIDENCE_ROOT_DOMAIN
        + _admission.canonical_json_bytes_v1(body)
    ).hexdigest()
    return result


def _replay_actual_admission_artifact_evidence_v1(
    value: object,
    *,
    source_commit: str,
    commit_a_config_bytes: bytes,
    commit_a_config_git_blob_oid: str,
    stage_5_actor_completion_rows: object,
    stage_5_strict_endpoint_replay_roots: object,
    five_sidecars: Mapping[str, object],
    endpoint_stdout_set: Mapping[str, object],
) -> dict[str, object]:
    section = _object(
        value,
        {
            "schema_version", "source_commit", "artifact_path",
            "commit_a_config_binding", "prior_stage_evidence_rows",
            "issue_record", "consumed_marker_evidence",
            "consume_work_root_replay", "consume_git_source_transcript",
            "consume_artifact_absence_evidence",
            "fresh_runtime_checkpoint_rows",
            "stage_5_live_marker_replay",
            "stage_5_mount_launch_replay_rows",
            "pre_artifact_live_marker_replay", "anti_replay_scope",
            "root_registry", "actual_admission_evidence_root",
        },
        "actual_admission",
    )
    expected = build_actual_admission_artifact_evidence_v1(
        source_commit=source_commit,
        artifact_path=section["artifact_path"],
        commit_a_config_bytes=commit_a_config_bytes,
        commit_a_config_git_blob_oid=commit_a_config_git_blob_oid,
        prior_stage_evidence_rows=section["prior_stage_evidence_rows"],
        issue_record=section["issue_record"],
        consumed_marker_evidence=section["consumed_marker_evidence"],
        consume_work_root_replay=section["consume_work_root_replay"],
        consume_git_source_transcript=section["consume_git_source_transcript"],
        consume_artifact_absence_evidence=section[
            "consume_artifact_absence_evidence"
        ],
        fresh_runtime_checkpoint_rows=section["fresh_runtime_checkpoint_rows"],
        pre_artifact_live_marker_replay=section[
            "pre_artifact_live_marker_replay"
        ],
        anti_replay_scope=section["anti_replay_scope"],
        stage_5_evidence=None,
        stage_5_actor_completion_rows=stage_5_actor_completion_rows,
        stage_5_strict_endpoint_replay_roots=(
            stage_5_strict_endpoint_replay_roots
        ),
        stage_5_live_marker_replay=section["stage_5_live_marker_replay"],
        stage_5_mount_launch_replay_rows=section[
            "stage_5_mount_launch_replay_rows"
        ],
        five_sidecars=five_sidecars,
        endpoint_stdout_set=endpoint_stdout_set,
    )
    if section != expected:
        _fail(
            "REJECT_Q05B_ARTIFACT_ADMISSION",
            "actual admission section replay differs",
        )
    return section


def _actual_admission_decision_for_docker_v1(
    actual_admission: Mapping[str, object],
) -> dict[str, object]:
    """Read the already-replayed decision bytes for the post-replay join."""

    try:
        _record, boundary = _admission.validate_actual_admission_issue_record_v1(
            actual_admission["issue_record"]
        )
        decision = _decoded_json_value(
            _hex_any(boundary["decision_hex"], "Docker admission decision"),
            "Docker admission decision",
        )
        if type(decision) is not dict:
            raise TypeError("decision")
        return decision
    except (
        KeyError,
        TypeError,
        ValueError,
        _admission.Q05BActualAdmissionError,
    ) as error:
        _fail(
            "REJECT_Q05B_ARTIFACT_ADMISSION",
            getattr(error, "detail", "Docker admission decision differs"),
        )


def _pinned_image_base_labels_v1(
    evidence: Mapping[str, object],
    name: str,
) -> dict[str, str]:
    try:
        decoded = _decoded_json_value(
            _hex_any(evidence["raw_inspect_hex"], f"{name} image inspect"),
            f"{name} image inspect",
        )
        if (
            type(decoded) is not list
            or len(decoded) != 1
            or type(decoded[0]) is not dict
            or type(decoded[0].get("Config")) is not dict
        ):
            raise TypeError("image inspect")
        raw_labels = decoded[0]["Config"].get("Labels")
        if raw_labels is None:
            labels: dict[str, str] = {}
        elif type(raw_labels) is dict and all(
            type(key) is str and type(value) is str
            for key, value in raw_labels.items()
        ):
            labels = dict(raw_labels)
        else:
            raise TypeError("image labels")
    except (KeyError, TypeError, ValueError):
        _fail(
            "REJECT_Q05B_ARTIFACT_ISOLATION",
            f"{name} pinned image labels differ",
        )
    if set(labels) & set(_admission.DOCKER_RESERVED_LABEL_KEYS):
        _fail(
            "REJECT_Q05B_ARTIFACT_ISOLATION",
            f"{name} pinned image owns a reserved Docker label",
        )
    return labels


def _docker_command_matches_slot_v1(
    command: object,
    slot: Mapping[str, object],
    name: str,
) -> list[str]:
    container_name, labels, normalized = _docker_run_principal_tokens_v1(
        command, name
    )
    if (
        type(slot) is not dict
        or container_name != slot.get("container_name")
        or labels != slot.get("labels")
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_ISOLATION",
            f"{name} Docker authority binding differs",
        )
    return normalized


def _docker_inspect_principal_v1(
    payload: bytes,
    *,
    command: list[str],
    slot: Mapping[str, object],
    container_id: str,
    image: str,
    name: str,
) -> dict[str, object]:
    decoded = _decoded_json_value(payload, name)
    if type(decoded) is not list or len(decoded) != 1 or type(decoded[0]) is not dict:
        _fail("REJECT_Q05B_ARTIFACT_ISOLATION", f"{name} shape differs")
    document = decoded[0]
    container = document.get("Config")
    expected_rows = slot.get("expected_container_labels")
    if (
        type(container_id) is not str
        or re.fullmatch(r"[0-9a-f]{64}", container_id) is None
        or type(container) is not dict
        or type(expected_rows) is not list
        or document.get("Id") != container_id
        or document.get("Name") != f"/{slot.get('container_name')}"
        or container.get("Image") != image
        or command.count(image) != 1
        or container.get("Cmd") != command[command.index(image) + 1 :]
        or container.get("Labels") != dict(expected_rows)
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_ISOLATION",
            f"{name} ownership principal differs",
        )
    return document


def _docker_owned_inspect_evidence_v1(
    value: object,
    payload: bytes,
    *,
    authority: Mapping[str, object],
    slot: Mapping[str, object],
    command: list[str],
    container_id: str,
    image: str,
    ownership_label_root: str,
    name: str,
) -> dict[str, object]:
    evidence = _object(
        value,
        {
            "schema_version",
            "docker_execution_authority_manifest_sha256",
            "slot_id",
            "slot",
            "container_id",
            "container_name",
            "ownership_label_root",
            "image",
            "command_sha256",
            "inspect_hex",
            "inspect_sha256",
            "ownership_inspect_root",
        },
        name,
    )
    _docker_inspect_principal_v1(
        payload,
        command=command,
        slot=slot,
        container_id=container_id,
        image=image,
        name=name,
    )
    body = {
        "schema_version": "hegel-phase3a-q05b-docker-owned-inspect/1",
        "docker_execution_authority_manifest_sha256": authority["manifest_sha256"],
        "slot_id": slot["slot_id"],
        "slot": slot["slot"],
        "container_id": container_id,
        "container_name": slot["container_name"],
        "ownership_label_root": ownership_label_root,
        "image": image,
        "command_sha256": sha256(_canonical_json(command)).hexdigest(),
        "inspect_hex": payload.hex(),
        "inspect_sha256": sha256(payload).hexdigest(),
    }
    expected = {
        **body,
        "ownership_inspect_root": sha256(
            DOCKER_OWNED_INSPECT_ROOT_DOMAIN + _canonical_json(body)
        ).hexdigest(),
    }
    if evidence != expected:
        _fail(
            "REJECT_Q05B_ARTIFACT_ISOLATION",
            f"{name} evidence differs",
        )
    return evidence


def _docker_success_ownership_v1(
    success: Mapping[str, object],
    *,
    authority: Mapping[str, object],
    slot: Mapping[str, object],
    command: list[str],
    container_id: object,
    cidfile: object,
    live_payload: bytes,
    post_payload: bytes,
    image: str,
    config: Mapping[str, object],
    name: str,
) -> None:
    _docker_command_matches_slot_v1(command, slot, name)
    if (
        type(container_id) is not str
        or re.fullmatch(r"[0-9a-f]{64}", container_id) is None
        or type(cidfile) is not dict
        or cidfile.get("container_id") != container_id
    ):
        _fail("REJECT_Q05B_ARTIFACT_ISOLATION", f"{name} owned CID differs")
    labels = slot["labels"]
    expected_label_root = sha256(
        DOCKER_OWNERSHIP_LABEL_ROOT_DOMAIN + _canonical_json(labels)
    ).hexdigest()
    try:
        precreate = _admission.validate_docker_precreate_absence_v1(
            success["precreate_absence_evidence"], authority
        )
    except (KeyError, TypeError, _admission.Q05BActualAdmissionError) as error:
        _fail(
            "REJECT_Q05B_ARTIFACT_ISOLATION",
            getattr(error, "detail", f"{name} precreate absence differs"),
        )
    if (
        success.get("docker_execution_authority_manifest_sha256")
        != authority["manifest_sha256"]
        or success.get("docker_execution_slot_row") != slot
        or success.get("ownership_label_root") != expected_label_root
        or precreate.get("slot_id") != slot["slot_id"]
        or precreate.get("container_name") != slot["container_name"]
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_ISOLATION",
            f"{name} Docker ownership evidence differs",
        )
    _docker_owned_inspect_evidence_v1(
        success.get("live_ownership_inspect_evidence"),
        live_payload,
        authority=authority,
        slot=slot,
        command=command,
        container_id=container_id,
        image=image,
        ownership_label_root=expected_label_root,
        name=f"{name} live owned inspect",
    )
    _docker_owned_inspect_evidence_v1(
        success.get("post_ownership_inspect_evidence"),
        post_payload,
        authority=authority,
        slot=slot,
        command=command,
        container_id=container_id,
        image=image,
        ownership_label_root=expected_label_root,
        name=f"{name} post owned inspect",
    )
    docker = config.get("docker")
    expected_remove = [
        docker.get("executable") if type(docker) is dict else None,
        f"--host={docker.get('host')}" if type(docker) is dict else None,
        "rm",
        container_id,
    ]
    if (
        success.get("explicit_remove_command") != expected_remove
        or success.get("cleanup_target_kind")
        != "OWNERSHIP_VALIDATED_CONTAINER_ID"
        or success.get("container_name_was_never_a_destructive_target") is not True
        or slot["container_name"] in expected_remove
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_ISOLATION",
            f"{name} Docker cleanup authority differs",
        )
    absence = _validate_docker_absence_v1(
        success.get("docker_absence_evidence"),
        container_id,
        f"{name} Docker removal absence",
    )
    if absence["container_identity"] != container_id:
        _fail(
            "REJECT_Q05B_ARTIFACT_ISOLATION",
            f"{name} Docker cleanup target differs",
        )


def _cross_docker_execution_ownership_surfaces_v1(
    config: Mapping[str, object],
    pinned_images: Mapping[str, Mapping[str, object]],
    actor_rows: object,
    cargo: Mapping[str, object],
    authority_value: object,
    decision: object,
    planned: object,
) -> None:
    """Pure core for the post-replay five-slot Docker ownership join."""

    try:
        authority = (
            _admission.cross_docker_execution_authority_to_admission_decision_v1(
                authority_value, decision
            )
        )
        slots = {
            row["slot"]: row for row in authority["ordered_slot_rows"]
        }
        if (
            type(actor_rows) not in (list, tuple)
            or len(actor_rows) != 3
            or type(planned) is not dict
            or set(planned)
            != {"rust_test", "rust_release", "python", "rust", "host_template"}
            or set(slots)
            != {
                "RUST_TEST",
                "RUST_RELEASE",
                "PYTHON_ENDPOINT",
                "RUST_ENDPOINT",
                "TRUSTED_HOST_REPLAY",
            }
        ):
            raise TypeError("Docker ownership registry")
    except (
        KeyError,
        TypeError,
        ValueError,
        _admission.Q05BActualAdmissionError,
    ) as error:
        _fail(
            "REJECT_Q05B_ARTIFACT_ISOLATION",
            getattr(error, "detail", "Docker execution authority differs"),
        )

    python_base = _pinned_image_base_labels_v1(
        pinned_images["python"], "Python"
    )
    rust_base = _pinned_image_base_labels_v1(pinned_images["rust"], "Rust")
    for slot_name, slot in slots.items():
        base = rust_base if slot_name in {
            "RUST_TEST", "RUST_RELEASE", "RUST_ENDPOINT"
        } else python_base
        expected = dict(base)
        for key, value in slot["labels"]:
            if key in expected:
                _fail(
                    "REJECT_Q05B_ARTIFACT_ISOLATION",
                    "pinned image reserved Docker label collision",
                )
            expected[key] = value
        if slot["expected_container_labels"] != [
            [key, expected[key]] for key in sorted(expected)
        ]:
            _fail(
                "REJECT_Q05B_ARTIFACT_ISOLATION",
                f"{slot_name} pinned image label union differs",
            )

    planned_slots = (
        ("rust_test", "RUST_TEST"),
        ("rust_release", "RUST_RELEASE"),
        ("python", "PYTHON_ENDPOINT"),
        ("rust", "RUST_ENDPOINT"),
        ("host_template", "TRUSTED_HOST_REPLAY"),
    )
    for command_key, slot_name in planned_slots:
        _docker_command_matches_slot_v1(
            planned[command_key], slots[slot_name], f"planned {command_key}"
        )
    names = [slots[name]["container_name"] for _, name in planned_slots]
    if len(set(names)) != 5:
        _fail(
            "REJECT_Q05B_ARTIFACT_ISOLATION",
            "Docker execution names are not unique by slot",
        )

    actor_specs = (
        (actor_rows[0], slots["PYTHON_ENDPOINT"], config["images"]["python_endpoint"]),
        (actor_rows[1], slots["RUST_ENDPOINT"], config["images"]["rust_runtime"]),
        (actor_rows[2], slots["TRUSTED_HOST_REPLAY"], config["images"]["trusted_host"]),
    )
    for actor, slot, image in actor_specs:
        command = actor["command"]
        control = actor["control_evidence"]
        resource = control["final_resource_transcript"]
        live_samples = resource["live_sample_objects"]
        for sample_index, sample in enumerate(live_samples, 1):
            for field in ("inspect_payload_hex", "inspect_after_payload_hex"):
                _docker_inspect_principal_v1(
                    _hex_any(sample[field], f"actor sample {field}"),
                    command=command,
                    slot=slot,
                    container_id=control["container_id"],
                    image=image,
                    name=f"{slot['slot']} sample {sample_index} {field}",
                )
        held_live = _hex_any(
            control["held_final_resource"]["inspect_payload_hex"],
            f"{slot['slot']} held live inspect",
        )
        post = _hex_any(
            control["post_exit_inspect_hex"], f"{slot['slot']} post inspect"
        )
        _docker_success_ownership_v1(
            control,
            authority=authority,
            slot=slot,
            command=command,
            container_id=control["container_id"],
            cidfile=control["cidfile_evidence"],
            live_payload=held_live,
            post_payload=post,
            image=image,
            config=config,
            name=slot["slot"],
        )

    for transcript_key, slot_name in (
        ("rust_test", "RUST_TEST"),
        ("rust_release_build", "RUST_RELEASE"),
    ):
        transcript = cargo[transcript_key]
        slot = slots[slot_name]
        _docker_success_ownership_v1(
            transcript,
            authority=authority,
            slot=slot,
            command=transcript["command"],
            container_id=transcript["cidfile_evidence"]["container_id"],
            cidfile=transcript["cidfile_evidence"],
            live_payload=_hex_any(
                transcript["live_inspect_hex"], f"{slot_name} live inspect"
            ),
            post_payload=_hex_any(
                transcript["post_inspect_hex"], f"{slot_name} post inspect"
            ),
            image=config["images"]["rust_build"],
            config=config,
            name=slot_name,
        )


def _cross_docker_execution_ownership_v1(
    config: Mapping[str, object],
    pinned_images: Mapping[str, Mapping[str, object]],
    actor_rows: object,
    cargo: Mapping[str, object],
    actual_admission: Mapping[str, object],
) -> None:
    """Extract already-replayed surfaces, then perform the causal join."""

    try:
        stage_1 = actual_admission["prior_stage_evidence_rows"][0]["evidence"]
        authority = stage_1["docker_execution_authority"]
        planned = stage_1["planned_commands"]
    except (KeyError, IndexError, TypeError):
        _fail(
            "REJECT_Q05B_ARTIFACT_ISOLATION",
            "Docker Stage-1 ownership surface differs",
        )
    _cross_docker_execution_ownership_surfaces_v1(
        config,
        pinned_images,
        actor_rows,
        cargo,
        authority,
        _actual_admission_decision_for_docker_v1(actual_admission),
        planned,
    )


def _cross_prior_stage12_top_v1(
    source: Mapping[str, object],
    payload_table: Mapping[str, tuple[int, str, bytes]],
    config: Mapping[str, object],
    pinned_images: Mapping[str, Mapping[str, object]],
    actor_rows: object,
    cargo: Mapping[str, object],
    actual_admission: Mapping[str, object],
) -> None:
    """Join production Stage-1/2 bytes to independently replayed top sections.

    This is deliberately a post-replay equality join.  It adds no serialized
    field and contributes to no evidence root, so neither side can acquire its
    expected bytes from the other while its own root is being validated.
    """

    actor_ids = (
        "PYTHON_ENDPOINT",
        "RUST_ENDPOINT",
        "TRUSTED_HOST_REPLAY",
    )
    try:
        stages = actual_admission["prior_stage_evidence_rows"]
        issue_record = actual_admission["issue_record"]
        if (
            type(stages) is not list
            or len(stages) != 3
            or type(actor_rows) not in (list, tuple)
            or len(actor_rows) != 3
            or type(issue_record) is not dict
        ):
            raise KeyError("Stage-1/2 join registry")
        stage_1 = stages[0]["evidence"]
        stage_2 = stages[1]["evidence"]
        boundary = _decoded_json_value(
            _hex_any(issue_record["pure_boundary_hex"], "issued boundary"),
            "issued boundary",
        )
        bundle = _decoded_json_value(
            _hex_any(
                boundary["precondition_bundle_hex"],
                "admission precondition bundle",
            ),
            "admission precondition bundle",
        )
        ordered = bundle["ordered_precondition_rows"]
        work_identity = bundle["work_root_identity"]
        if (
            type(stage_1) is not dict
            or set(stage_1) != _ACTUAL_STAGE_1_EVIDENCE_KEYS
            or type(stage_2) is not dict
            or set(stage_2) != _ACTUAL_STAGE_2_EVIDENCE_KEYS
            or type(ordered) is not list
            or len(ordered) != 12
            or type(work_identity) is not dict
        ):
            raise KeyError("Stage-1/2 evidence registry")
        preimages = [row["preimage"] for row in ordered]
    except (KeyError, TypeError, ValueError):
        _fail(
            "REJECT_Q05B_ARTIFACT_ADMISSION",
            "Stage-1/2 top join registry differs",
        )

    config_path = "config/phase3_q05b_dual_isolation_v1.json"
    config_payload = payload_table[config_path][2]
    source_path_rows = source["actor_source_path_rows"]
    if (
        type(source_path_rows) is not list
        or [row[0] for row in source_path_rows] != list(actor_ids)
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_SOURCE",
            "Stage-1 actor source registry differs",
        )

    project_prefix = source["project_tree_prefix"]
    project_git_prefix = f"{project_prefix}/" if project_prefix else ""
    expected_source_evidence: dict[str, object] = {}
    for actor_id, actor, path_row in zip(
        actor_ids, actor_rows, source_path_rows, strict=True
    ):
        paths = path_row[1]
        condensed = actor["source_evidence"]
        blob_rows = [
            [
                path,
                payload_table[path][0],
                payload_table[path][1],
                len(payload_table[path][2]),
                sha256(payload_table[path][2]).hexdigest(),
            ]
            for path in paths
        ]
        blob_preimages = [
            [*row, payload_table[row[0]][2].hex()] for row in blob_rows
        ]
        expected_source_evidence[actor_id] = {
            "schema_version": "hegel-phase3a-q05b-actor-source-evidence/1",
            "actor_id": actor_id,
            "commit": source["source_commit"],
            "project_git_prefix": project_git_prefix,
            "path_registry_sha256": condensed["path_registry_sha256"],
            "source_identity_sha256": condensed["source_identity_sha256"],
            "rows": blob_rows,
            "blob_preimage_rows": blob_preimages,
        }

    commit_payload = _hex_any(
        source["git_commit_object_hex"], "Stage-1 Git commit object"
    )
    tree_payloads = {
        row[0]: _hex_any(row[1], "Stage-1 Git tree object")
        for row in source["git_tree_object_rows"]
    }
    project_tree_object_id = source["external_commit_replay"]["tree_oid"]
    for component in project_prefix.split("/") if project_prefix else ():
        try:
            mode, project_tree_object_id = _parse_git_tree_v1(
                tree_payloads[project_tree_object_id]
            )[component]
        except (KeyError, TypeError, ValueError):
            _fail(
                "REJECT_Q05B_ARTIFACT_SOURCE",
                "Stage-1 project tree binding differs",
            )
        if mode != 0o40000:
            _fail(
                "REJECT_Q05B_ARTIFACT_SOURCE",
                "Stage-1 project tree binding differs",
            )
    closure_body: dict[str, object] = {
        "schema_version": "hegel-phase3a-q05b-git-source-object-closure/1",
        "commit": source["source_commit"],
        "commit_payload_hex": commit_payload.hex(),
        "commit_payload_sha256": sha256(commit_payload).hexdigest(),
        "root_tree_object_id": source["external_commit_replay"]["tree_oid"],
        "project_tree_prefix": project_prefix,
        "project_tree_object_id": project_tree_object_id,
        "allowlist_union": list(payload_table),
        "tree_object_rows": source["git_tree_object_rows"],
    }
    expected_closure = dict(closure_body)
    expected_closure["closure_sha256"] = sha256(
        _canonical_json(closure_body)
    ).hexdigest()

    cache_path = stage_1["cargo_cache_source"]
    cache_identity = stage_1["cargo_cache_root_identity"]
    if (
        type(cache_path) is not str
        or not cache_path.startswith("/")
        or cache_path == "/"
        or cache_path.endswith("/")
        or any(part in ("", ".", "..") for part in cache_path.split("/")[1:])
        or type(cache_identity) is not list
        or len(cache_identity) != 4
        or any(type(item) is not int for item in cache_identity)
        or cache_identity[0] < 0
        or cache_identity[1] <= 0
        or cache_identity[2] < 1
        or not 0 <= cache_identity[3] <= 0o7777
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_SOURCE",
            "Stage-1 external Cargo cache preimage differs",
        )

    work_root = work_identity["absolute_path"]
    work_prefix = work_root.rstrip("/")

    def work_path(relative: str) -> str:
        return f"{work_prefix}/{relative}"

    expected_layout = {
        "python_snapshot": work_path("snapshots/python"),
        "rust_snapshot": work_path("snapshots/rust"),
        "host_snapshot": work_path("snapshots/host"),
        "cargo_home": work_path("cargo-home"),
        "target_output": work_path("target-output"),
        "cargo_release_binary": work_path(
            "target-output/release/hegel-q1-archive-projection-oracle"
        ),
        "runtime_binary_parent": work_path("target-output/runtime-binary"),
        "python_output": work_path("python-output"),
        "python_control": work_path("python-control"),
        "python_cid_parent": work_path("python-cid"),
        "python_cidfile": work_path("python-cid/python.cid"),
        "rust_output": work_path("rust-output"),
        "rust_control": work_path("rust-control"),
        "rust_cid_parent": work_path("rust-cid"),
        "rust_cidfile": work_path("rust-cid/rust.cid"),
        "host_output": work_path("host-output-unused"),
        "host_control": work_path("host-control"),
        "host_cid_parent": work_path("host-cid"),
        "host_cidfile": work_path("host-cid/host.cid"),
        "host_staging": work_path("host-staging"),
        "build_cid_parent": work_path("build-cid"),
        "build_test_cidfile": work_path("build-cid/test.cid"),
        "build_release_cidfile": work_path("build-cid/release.cid"),
        "stdout_root": work_path("stdout"),
        "binary": work_path(
            "target-output/runtime-binary/hegel-q1-archive-projection-oracle"
        ),
    }
    if (
        stage_1["layout"] != expected_layout
        or work_identity["layout_sha256"]
        != sha256(_canonical_json(expected_layout)).hexdigest()
        or cache_path == work_root
        or cache_path.startswith(work_prefix + "/")
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_ISOLATION",
            "Stage-1 work layout differs",
        )

    host_template = list(actor_rows[2]["command"])
    for flag in (
        "--host-source-identity-root-hex",
        "--host-runtime-identity-root-hex",
    ):
        positions = [index for index, item in enumerate(host_template) if item == flag]
        if len(positions) != 1 or positions[0] + 1 >= len(host_template):
            _fail(
                "REJECT_Q05B_ARTIFACT_ISOLATION",
                "Stage-1 host command template differs",
            )
        host_template[positions[0] + 1] = "0" * 64
    expected_planned_commands = {
        "python": actor_rows[0]["command"],
        "rust": actor_rows[1]["command"],
        "host_template": host_template,
        "rust_test": cargo["rust_test"]["command"],
        "rust_release": cargo["rust_release_build"]["command"],
    }

    runtime_seccomp_path = work_path(
        f"snapshots/host/{config['seccomp']['runtime_profile']}"
    )
    build_seccomp_path = work_path(
        f"snapshots/host/{config['seccomp']['build_profile']}"
    )

    def one_option(command: object, prefix: str, name: str) -> str:
        if type(command) is not list:
            _fail(
                "REJECT_Q05B_ARTIFACT_ISOLATION", f"{name} command differs"
            )
        values = [item.removeprefix(prefix) for item in command if item.startswith(prefix)]
        if len(values) != 1:
            _fail(
                "REJECT_Q05B_ARTIFACT_ISOLATION", f"{name} option differs"
            )
        return values[0]

    python_mounts = _mount_sources_v1(expected_planned_commands["python"])
    rust_mounts = _mount_sources_v1(expected_planned_commands["rust"])
    host_mounts = _mount_sources_v1(expected_planned_commands["host_template"])
    expected_runtime_mounts = (
        {
            "/snapshot": expected_layout["python_snapshot"],
            "/output": expected_layout["python_output"],
            "/control": expected_layout["python_control"],
        },
        {
            "/runtime/hegel-q1-archive-projection-oracle": expected_layout[
                "binary"
            ],
            "/output": expected_layout["rust_output"],
            "/control": expected_layout["rust_control"],
        },
        {
            "/snapshot": expected_layout["host_snapshot"],
            "/inputs/python": expected_layout["python_output"],
            "/inputs/rust": expected_layout["rust_output"],
            "/inputs/stdout/python.stdout": work_path("stdout/python.stdout"),
            "/inputs/stdout/rust.stdout": work_path("stdout/rust.stdout"),
            "/inputs/stdout/manifest.json": work_path("stdout/manifest.json"),
            "/control": expected_layout["host_control"],
            "/staging": expected_layout["host_staging"],
        },
    )
    build_mounts = [
        _mount_sources_v1(expected_planned_commands[name])
        for name in ("rust_test", "rust_release")
    ]
    expected_build_mounts = {
        "/snapshot": expected_layout["rust_snapshot"],
        "/cargo-home": expected_layout["cargo_home"],
        "/target-output": expected_layout["target_output"],
    }
    if (
        (python_mounts, rust_mounts, host_mounts) != expected_runtime_mounts
        or any(mounts != expected_build_mounts for mounts in build_mounts)
        or one_option(
            expected_planned_commands["python"], "--cidfile=", "Python"
        )
        != expected_layout["python_cidfile"]
        or one_option(expected_planned_commands["rust"], "--cidfile=", "Rust")
        != expected_layout["rust_cidfile"]
        or one_option(
            expected_planned_commands["host_template"], "--cidfile=", "host"
        )
        != expected_layout["host_cidfile"]
        or one_option(
            expected_planned_commands["rust_test"], "--cidfile=", "Rust test"
        )
        != expected_layout["build_test_cidfile"]
        or one_option(
            expected_planned_commands["rust_release"],
            "--cidfile=",
            "Rust release",
        )
        != expected_layout["build_release_cidfile"]
        or any(
            one_option(command, "--security-opt=seccomp=", "runtime")
            != runtime_seccomp_path
            for command in expected_planned_commands.values()
            if command
            in (
                expected_planned_commands["python"],
                expected_planned_commands["rust"],
                expected_planned_commands["host_template"],
            )
        )
        or any(
            one_option(command, "--security-opt=seccomp=", "build")
            != build_seccomp_path
            for command in (
                expected_planned_commands["rust_test"],
                expected_planned_commands["rust_release"],
            )
        )
        or cache_path
        in {
            source_path
            for command in expected_planned_commands.values()
            for source_path in _mount_sources_v1(command).values()
        }
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_ISOLATION",
            "Stage-1 planned command/layout binding differs",
        )

    project_root = preimages[0]["git_source_transcript"]["project_root"]
    expected_artifact_path = (
        project_root.rstrip("/")
        + "/"
        + config["artifact_layout"]["relative_path"]
    )
    expected_stage_1 = {
        "config_hex": config_payload.hex(),
        "config_sha256": sha256(config_payload).hexdigest(),
        "fixed_artifact_path": expected_artifact_path,
        "layout": expected_layout,
        "cargo_cache_source": cache_path,
        "cargo_cache_root_identity": cache_identity,
        "source_evidence": expected_source_evidence,
        "source_object_closure": expected_closure,
        "image_evidence": {
            "python": pinned_images["python"],
            "rust": pinned_images["rust"],
        },
        "planned_commands": expected_planned_commands,
        "docker_execution_authority": stage_1["docker_execution_authority"],
        "q1_authority": config["dry_run_authority"],
    }
    if (
        stage_1 != expected_stage_1
        or actual_admission["artifact_path"] != expected_artifact_path
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_SOURCE",
            "Stage-1 top evidence differs",
        )

    expected_image_rows = [
        {
            "label": label,
            "reference": pinned_images[label]["requested_reference"],
            "evidence": pinned_images[label],
            "evidence_root": _admission.fresh_runtime_evidence_object_root_v1(
                "PINNED_IMAGE", label, pinned_images[label]
            ),
        }
        for label in ("python", "rust")
    ]
    if preimages[4]["image_rows"] != expected_image_rows:
        _fail(
            "REJECT_Q05B_ARTIFACT_SOURCE",
            "Stage-1 admission image rows differ",
        )

    expected_snapshots = {
        actor_id: actor["snapshot_identity"]
        for actor_id, actor in zip(actor_ids, actor_rows, strict=True)
    }
    lock_payload = _hex_any(cargo["lock_hex"], "Cargo.lock")
    cargo_lock_path = "rust/q1_archive_projection_oracle/Cargo.lock"
    if (
        cargo_lock_path not in payload_table
        or lock_payload != payload_table[cargo_lock_path][2]
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_CARGO",
            "Stage-2 Cargo.lock/source binding differs",
        )

    cargo_file_rows: list[list[object]] = []
    for path, mode, payload_hex in cargo["sealed_cargo_files"]:
        payload = _hex_any(payload_hex, "Stage-2 sealed Cargo file")
        cargo_file_rows.append(
            [path, mode, len(payload), sha256(payload).hexdigest()]
        )
    cargo_tree = cargo["sealed_cargo_tree"]
    cargo_snapshot_body: dict[str, object] = {
        "schema_version": "hegel-phase3a-q05b-sealed-snapshot-identity/1",
        "root_device": cargo_tree["root_device"],
        "root_inode": cargo_tree["root_inode"],
        "root_mode": cargo_tree["root_mode"],
        "file_rows": cargo_tree["file_rows"],
    }
    cargo_snapshot = dict(cargo_snapshot_body)
    cargo_snapshot["manifest_sha256"] = sha256(
        _canonical_json(cargo_snapshot_body)
    ).hexdigest()
    expected_cargo_evidence = {
        "schema_version": "hegel-phase3a-q05b-sealed-cargo-home/1",
        "locked_registry_package_count": len(cargo["locked_packages"]),
        "locked_packages": cargo["locked_packages"],
        "file_count": len(cargo_file_rows),
        "file_rows": cargo_file_rows,
        "file_preimage_rows": cargo["sealed_cargo_files"],
        "manifest_sha256": cargo["sealed_cargo_manifest_sha256"],
        "sealed_snapshot_identity": cargo_snapshot,
        "root_mode": "0555",
        "file_modes": "0444_OR_0555",
        "cargo_home_mount": "READ_ONLY_PREUNPACKED",
        "root_path": cargo_tree["root_path"],
        "root_nlink": cargo_tree["root_nlink"],
        "sealed_tree_identity": cargo_tree,
    }

    host_snapshot = expected_snapshots["TRUSTED_HOST_REPLAY"]
    host_file_rows = {row[0]: row for row in host_snapshot["file_rows"]}

    def expected_seccomp(relative_path: str) -> dict[str, object]:
        try:
            row = host_file_rows[relative_path]
            payload = payload_table[relative_path][2]
        except (KeyError, TypeError):
            _fail(
                "REJECT_Q05B_ARTIFACT_ISOLATION",
                "Stage-2 seccomp snapshot registry differs",
            )
        body: dict[str, object] = {
            "schema_version": "hegel-phase3a-q05b-sealed-policy-file/1",
            "absolute_path": (
                host_snapshot["root_path"].rstrip("/") + "/" + relative_path
            ),
            "snapshot_relative_path": relative_path,
            "file_device": row[1],
            "file_inode": row[2],
            "file_nlink": row[3],
            "file_uid": row[4],
            "file_gid": row[5],
            "file_mode": row[6],
            "file_size": row[7],
            "file_mtime_ns": row[8],
            "file_ctime_ns": row[9],
            "payload_sha256": sha256(payload).hexdigest(),
        }
        result = dict(body)
        result["manifest_sha256"] = sha256(_canonical_json(body)).hexdigest()
        return result

    runtime_relative = config["seccomp"]["runtime_profile"]
    build_relative = config["seccomp"]["build_profile"]
    expected_seccomp_evidence = {
        "runtime": expected_seccomp(runtime_relative),
        "build": expected_seccomp(build_relative),
    }
    expected_stage_2 = {
        "snapshot_evidence": expected_snapshots,
        "cargo_lock_hex": lock_payload.hex(),
        "cargo_lock_sha256": sha256(lock_payload).hexdigest(),
        "cargo_evidence": expected_cargo_evidence,
        "seccomp_evidence": expected_seccomp_evidence,
    }
    if stage_2 != expected_stage_2:
        _fail(
            "REJECT_Q05B_ARTIFACT_CARGO",
            "Stage-2 top evidence differs",
        )

    if (
        any(
            actor["control_evidence"]["seccomp_evidence"]
            != expected_seccomp_evidence["runtime"]
            for actor in actor_rows
        )
        or cargo["rust_test"]["seccomp_evidence"]
        != expected_seccomp_evidence["build"]
        or cargo["rust_release_build"]["seccomp_evidence"]
        != expected_seccomp_evidence["build"]
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_ISOLATION",
            "Stage-2 seccomp consumer binding differs",
        )

    expected_actor_rows: list[dict[str, object]] = []
    for actor_id in actor_ids:
        source_evidence = expected_source_evidence[actor_id]
        snapshot = expected_snapshots[actor_id]
        snapshot_registry = [
            [row[0], row[6], row[7], row[10]]
            for row in snapshot["file_rows"]
        ]
        source_identity = {
            "schema_version": (
                "hegel-phase3a-q05b-fresh-actor-source-identity/1"
            ),
            "actor_id": actor_id,
            "source_commit": source["source_commit"],
            "project_git_prefix": project_git_prefix,
            "path_registry_sha256": source_evidence[
                "path_registry_sha256"
            ],
            "source_identity_sha256": source_evidence[
                "source_identity_sha256"
            ],
            "blob_count": len(source_evidence["rows"]),
            "snapshot_file_registry_sha256": sha256(
                _canonical_json(snapshot_registry)
            ).hexdigest(),
            "stage_1_source_evidence_sha256": sha256(
                _canonical_json(source_evidence)
            ).hexdigest(),
        }
        expected_actor_rows.append(
            {
                "actor_id": actor_id,
                "source_identity": source_identity,
                "source_identity_root": (
                    _admission.fresh_runtime_evidence_object_root_v1(
                        "ACTOR_SOURCE", actor_id, source_identity
                    )
                ),
                "snapshot_evidence": snapshot,
                "snapshot_evidence_root": (
                    _admission.fresh_runtime_evidence_object_root_v1(
                        "ACTOR_SNAPSHOT", actor_id, snapshot
                    )
                ),
            }
        )
    if preimages[5]["actor_rows"] != expected_actor_rows:
        _fail(
            "REJECT_Q05B_ARTIFACT_ACTOR",
            "Stage-1/2 admission actor rows differ",
        )

    cargo_material_identity = {
        "schema_version": (
            "hegel-phase3a-q05b-fresh-cargo-material-identity/1"
        ),
        "root_path": expected_cargo_evidence["root_path"],
        "root_nlink": expected_cargo_evidence["root_nlink"],
        "file_count": expected_cargo_evidence["file_count"],
        "locked_registry_package_count": expected_cargo_evidence[
            "locked_registry_package_count"
        ],
        "locked_packages_sha256": sha256(
            _canonical_json(expected_cargo_evidence["locked_packages"])
        ).hexdigest(),
        "file_registry_sha256": sha256(
            _canonical_json(expected_cargo_evidence["file_rows"])
        ).hexdigest(),
        "material_manifest_sha256": expected_cargo_evidence[
            "manifest_sha256"
        ],
        "sealed_snapshot_manifest_sha256": cargo_snapshot[
            "manifest_sha256"
        ],
        "sealed_tree_manifest_sha256": cargo_tree["manifest_sha256"],
        "stage_2_cargo_evidence_sha256": sha256(
            _canonical_json(expected_cargo_evidence)
        ).hexdigest(),
    }
    if (
        preimages[6]["cargo_lock_sha256"] != sha256(lock_payload).hexdigest()
        or preimages[6]["cargo_material_identity"]
        != cargo_material_identity
        or preimages[6]["cargo_snapshot_evidence"] != cargo_snapshot
        or preimages[6]["cargo_tree_evidence"] != cargo_tree
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_CARGO",
            "Stage-2 admission Cargo material differs",
        )

    expected_seccomp_rows = [
        {
            "label": label,
            "relative_path": relative,
            "evidence": expected_seccomp_evidence[label],
            "evidence_root": _admission.fresh_runtime_evidence_object_root_v1(
                "SECCOMP_POLICY", label, expected_seccomp_evidence[label]
            ),
        }
        for label, relative in (
            ("runtime", runtime_relative),
            ("build", build_relative),
        )
    ]
    if preimages[7]["seccomp_rows"] != expected_seccomp_rows:
        _fail(
            "REJECT_Q05B_ARTIFACT_ISOLATION",
            "Stage-2 admission seccomp rows differ",
        )
    if preimages[8]["planned_command_registry_sha256"] != sha256(
        _canonical_json(expected_planned_commands)
    ).hexdigest():
        _fail(
            "REJECT_Q05B_ARTIFACT_ISOLATION",
            "Stage-1 admission planned command registry differs",
        )


def _cross_cargo_actual_admission_v1(
    cargo: Mapping[str, object],
    actual_admission: Mapping[str, object],
) -> None:
    """Join independently replayed Cargo and admission evidence.

    This adds no serialized field or root.  It only rejects when the top-level
    Cargo preimage and the already-replayed admission history describe
    different Stage-3 build/detach/binary authorities.
    """

    try:
        stages = actual_admission["prior_stage_evidence_rows"]
        issue_record = actual_admission["issue_record"]
        checkpoints = actual_admission["fresh_runtime_checkpoint_rows"]
        if (
            type(stages) is not list
            or len(stages) != 3
            or type(stages[2]) is not dict
            or type(issue_record) is not dict
            or type(checkpoints) is not list
            or len(checkpoints) != 3
        ):
            raise KeyError("join registry")
        stage_3_row = stages[2]
        stage_3 = stage_3_row["evidence"]
        boundary = _decoded_json_value(
            _hex_any(issue_record["pure_boundary_hex"], "issued boundary"),
            "issued boundary",
        )
        bundle = _decoded_json_value(
            _hex_any(
                boundary["precondition_bundle_hex"],
                "admission precondition bundle",
            ),
            "admission precondition bundle",
        )
        ordered = bundle["ordered_precondition_rows"]
        if type(ordered) is not list or len(ordered) != 12:
            raise KeyError("precondition registry")
        row_7 = ordered[6]["preimage"]
        row_8 = ordered[7]["preimage"]
    except (KeyError, TypeError, ValueError):
        _fail(
            "REJECT_Q05B_ARTIFACT_CARGO",
            "Cargo/admission join registry differs",
        )

    stage_3_keys = {
        "rust_test",
        "rust_release_build",
        "binary_detach",
        "binary",
        "rust_snapshot_post_build",
        "cargo_snapshot_post_build",
        "cargo_tree_post_build",
    }
    if type(stage_3) is not dict or set(stage_3) != stage_3_keys:
        _fail(
            "REJECT_Q05B_ARTIFACT_CARGO",
            "admission Stage-3 evidence fields differ",
        )
    binary_identity = cargo["binary_file_identity"]
    if type(binary_identity) is not dict:
        _fail(
            "REJECT_Q05B_ARTIFACT_CARGO",
            "Cargo/admission binary identity differs",
        )
    sealed_binary_body: dict[str, object] = {
        "schema_version": "hegel-phase3a-q05b-sealed-prebuilt-rust-binary/1",
        "binary_path": binary_identity["path"],
        "device": binary_identity["device"],
        "inode": binary_identity["inode"],
        "nlink": binary_identity["nlink"],
        "uid": binary_identity["uid"],
        "gid": binary_identity["gid"],
        "mode": binary_identity["mode"],
        "size": binary_identity["size"],
        "mtime_ns": binary_identity["mtime_ns"],
        "ctime_ns": binary_identity["ctime_ns"],
        "sha256": binary_identity["sha256"],
        "payload_hex": cargo["binary_hex"],
    }
    sealed_binary = dict(sealed_binary_body)
    sealed_binary["manifest_sha256"] = sha256(
        _canonical_json(sealed_binary_body)
    ).hexdigest()
    if (
        stage_3["rust_test"] != cargo["rust_test"]
        or stage_3["rust_release_build"] != cargo["rust_release_build"]
        or stage_3["binary_detach"] != cargo["binary_detach_evidence"]
        or stage_3["binary"] != sealed_binary
        or stage_3["rust_snapshot_post_build"]
        != cargo["rust_snapshot_post_build"]
        or stage_3["cargo_tree_post_build"]
        != cargo["cargo_snapshot_post_build"]
        or stage_3["cargo_tree_post_build"] != cargo["sealed_cargo_tree"]
        or row_7["cargo_snapshot_evidence"]
        != stage_3["cargo_snapshot_post_build"]
        or row_7["cargo_tree_evidence"]
        != stage_3["cargo_tree_post_build"]
        or row_7["cargo_lock_sha256"]
        != sha256(_hex_any(cargo["lock_hex"], "Cargo.lock")).hexdigest()
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_CARGO",
            "top Cargo and admission Stage-3 evidence differ",
        )

    expected_offline = {
        "schema_version": "hegel-phase3a-q05b-fresh-offline-build-identity/1",
        "stage_3_root": stage_3_row["stage_evidence_root"],
        "rust_test_transcript_sha256": sha256(
            _admission.canonical_json_bytes_v1(stage_3["rust_test"])
        ).hexdigest(),
        "rust_release_build_transcript_sha256": sha256(
            _admission.canonical_json_bytes_v1(
                stage_3["rust_release_build"]
            )
        ).hexdigest(),
        "rust_snapshot_manifest_sha256": stage_3[
            "rust_snapshot_post_build"
        ]["manifest_sha256"],
        "cargo_snapshot_manifest_sha256": stage_3[
            "cargo_snapshot_post_build"
        ]["manifest_sha256"],
        "cargo_tree_manifest_sha256": stage_3[
            "cargo_tree_post_build"
        ]["manifest_sha256"],
        "binary_manifest_sha256": sealed_binary["manifest_sha256"],
        "stage_3_evidence_sha256": sha256(
            _admission.canonical_json_bytes_v1(stage_3)
        ).hexdigest(),
    }
    if row_7["offline_build_identity"] != expected_offline:
        _fail(
            "REJECT_Q05B_ARTIFACT_CARGO",
            "admission offline identity differs from Stage-3 bytes",
        )

    expected_fresh_binary = {
        "schema_version": (
            "hegel-phase3a-q05b-fresh-prebuilt-rust-binary-identity/1"
        ),
        "binary_path": sealed_binary["binary_path"],
        "device": sealed_binary["device"],
        "inode": sealed_binary["inode"],
        "nlink": sealed_binary["nlink"],
        "uid": sealed_binary["uid"],
        "gid": sealed_binary["gid"],
        "mode": sealed_binary["mode"],
        "size": sealed_binary["size"],
        "mtime_ns": sealed_binary["mtime_ns"],
        "ctime_ns": sealed_binary["ctime_ns"],
        "sha256": sealed_binary["sha256"],
        "sealed_binary_manifest_sha256": sealed_binary[
            "manifest_sha256"
        ],
        "stage_3_binary_evidence_sha256": sha256(
            _admission.canonical_json_bytes_v1(sealed_binary)
        ).hexdigest(),
    }
    if row_8["binary_identity"] != expected_fresh_binary:
        _fail(
            "REJECT_Q05B_ARTIFACT_CARGO",
            "admission fresh binary differs from Stage-3 bytes",
        )
    for checkpoint in checkpoints:
        try:
            observed_binary = checkpoint[
                "observed_fresh_runtime_evidence"
            ]["binary"]["identity"]
        except (KeyError, TypeError):
            _fail(
                "REJECT_Q05B_ARTIFACT_CARGO",
                "checkpoint binary join registry differs",
            )
        if observed_binary != expected_fresh_binary:
            _fail(
                "REJECT_Q05B_ARTIFACT_CARGO",
                "checkpoint binary differs from top Cargo bytes",
            )


def _replay_actual_evidence_v1(
    evidence: Mapping[str, object], *, candidate_only: bool
) -> dict[str, object]:
    if type(evidence) is not dict or set(evidence) != set(SECTION_NAMES):
        _fail("REJECT_Q05B_ARTIFACT_SCHEMA", "evidence section registry differs")
    source = _object(evidence["source_wire_profile"], {"actor_source_path_rows", "external_commit_replay", "full_leaf_manifest_root", "git_blob_payload_table", "git_commit_object_hex", "git_tree_object_rows", "pinned_image_rows", "project_tree_prefix", "q0_receipt_root", "q1_projection_profile_root", "q1_semantic_binding_root", "qualification_predicate_registry_root", "qualification_tag_registry_root", "qualification_wire_profile_root", "source_commit", "source_commit_raw20_hex"}, "source_wire_profile")
    commit = source["source_commit"]
    if type(commit) is not str or re.fullmatch(r"[0-9a-f]{40}", commit) is None or source["source_commit_raw20_hex"] != commit:
        _fail("REJECT_Q05B_ARTIFACT_SOURCE", "source commit wire differs")
    source_raw = _hex(commit, 20, "source commit")
    table_value = source["git_blob_payload_table"]
    if type(table_value) is not list or not table_value or any(type(row) is not list or len(row) != 4 for row in table_value):
        _fail("REJECT_Q05B_ARTIFACT_SOURCE", "Git blob payload table differs")
    payload_table: dict[str, tuple[int, str, bytes]] = {}
    for row in table_value:
        path, mode, oid, payload_hex = row
        payload = _hex_any(payload_hex, "Git blob payload")
        expected_oid = sha1(b"blob " + str(len(payload)).encode("ascii") + b"\x00" + payload).hexdigest()
        if type(path) is not str or not path or path.startswith("/") or ".." in path.split("/") or type(mode) is not int or mode not in (0o100644, 0o100755) or oid != expected_oid or path in payload_table:
            _fail("REJECT_Q05B_ARTIFACT_SOURCE", "Git blob payload row differs")
        payload_table[path] = (mode, oid, payload)
    if list(payload_table) != sorted(payload_table):
        _fail("REJECT_Q05B_ARTIFACT_SOURCE", "Git blob payload table is not ordered")
    external = _object(source["external_commit_replay"], {"commit", "head_clean_before", "head_clean_after", "tree_oid"}, "external_commit_replay")
    _require_type_exact_v1(
        external,
        {"commit": commit, "head_clean_before": True, "head_clean_after": True, "tree_oid": external["tree_oid"]},
        "external_commit_replay",
    )
    if type(external["tree_oid"]) is not str or re.fullmatch(r"[0-9a-f]{40}", external["tree_oid"]) is None:
        _fail("REJECT_Q05B_ARTIFACT_SOURCE", "external Commit-A replay differs")
    _replay_git_object_closure_v1(source, payload_table)
    config = _isolation_config_v1(payload_table)
    pinned_images = _validate_pinned_image_rows_v1(source["pinned_image_rows"], config)
    seccomp_policy = config["seccomp"]
    if type(seccomp_policy) is not dict or set(seccomp_policy) != {"build_profile", "build_profile_sha256", "default_seccomp_forbidden", "runtime_profile", "runtime_profile_sha256"} or seccomp_policy["default_seccomp_forbidden"] is not True:
        _fail("REJECT_Q05B_ARTIFACT_SOURCE", "seccomp policy differs")
    for profile_field, digest_field in (("runtime_profile", "runtime_profile_sha256"), ("build_profile", "build_profile_sha256")):
        path = seccomp_policy[profile_field]
        if type(path) is not str or path not in payload_table or seccomp_policy[digest_field] != sha256(payload_table[path][2]).hexdigest():
            _fail("REJECT_Q05B_ARTIFACT_SOURCE", "seccomp Commit-A binding differs")
        _strict_seccomp_json_object_v1(
            payload_table[path][2],
            f"Commit-A {profile_field}",
        )
    negative = _negative_v1(evidence["negative_corpus"])
    negative_cbor = _hex_any(
        evidence["negative_corpus"]["canonical_cbor_hex"], "negative corpus"
    )
    sidecar_section = _object(
        evidence["five_sidecars"],
        {"canonical_rows", "python_output_tree", "rust_output_tree"},
        "five_sidecars",
    )
    sidecar_rows = sidecar_section["canonical_rows"]
    replayed = _sidecars_v1(sidecar_rows)
    payloads, leaf, _partitions, _sidecar, golden, partition_replays, shadow = replayed
    sidecar_payload_map = dict(sorted({
        row["path"]: payload for row, payload in zip(sidecar_rows, payloads, strict=True)
    }.items()))
    python_output_tree = _validate_sealed_tree_identity_v1(
        sidecar_section["python_output_tree"], sidecar_payload_map,
        ("neutral", "preimages"), "Python output tree",
    )
    rust_output_tree = _validate_sealed_tree_identity_v1(
        sidecar_section["rust_output_tree"], sidecar_payload_map,
        ("neutral", "preimages"), "Rust output tree",
    )
    if python_output_tree["root_path"] == rust_output_tree["root_path"]:
        _fail("REJECT_Q05B_ARTIFACT_ISOLATION", "endpoint output roots are not distinct")
    expected_source = {
        "full_leaf_manifest_root": leaf.manifest_root.hex(),
        "q0_receipt_root": _wire.Q0_SATURATION_RECEIPT_ROOT_FROM_Q1_PREREGISTRATION.hex(),
        "q1_projection_profile_root": golden.q1_projection_profile_root.hex(),
        "q1_semantic_binding_root": golden.q1_semantic_binding_root.hex(),
        "qualification_predicate_registry_root": _wire.QUALIFICATION_PREDICATE_REGISTRY_ROOT.hex(),
        "qualification_tag_registry_root": _wire.QUALIFICATION_TAG_REGISTRY_ROOT.hex(),
        "qualification_wire_profile_root": _wire.qualification_wire_profile_root_v1().hex(),
        "source_commit": commit,
        "source_commit_raw20_hex": commit,
        "git_blob_payload_table": source["git_blob_payload_table"],
        "actor_source_path_rows": source["actor_source_path_rows"],
        "external_commit_replay": source["external_commit_replay"],
        "git_commit_object_hex": source["git_commit_object_hex"],
        "git_tree_object_rows": source["git_tree_object_rows"],
        "pinned_image_rows": source["pinned_image_rows"],
        "project_tree_prefix": source["project_tree_prefix"],
    }
    if source != expected_source:
        _fail("REJECT_Q05B_ARTIFACT_SOURCE", "source/wire profile replay differs")
    endpoint = _object(evidence["endpoint_stdout_set"], {"manifest_hex", "python_stdout_hex", "rust_stdout_hex", "sealed_stdout_tree"}, "endpoint_stdout_set")
    config_path = "config/phase3_q05b_dual_isolation_v1.json"
    actor_rows = _validate_actor_rows(
        evidence["actor_rows"], commit, payload_table,
        source["actor_source_path_rows"], config,
    )
    actor_map = {row["actor_id"]: row for row in actor_rows}
    dual = _dual_replay_v1(endpoint, replayed, _hex(actor_map["TRUSTED_HOST_REPLAY"]["source_evidence"]["source_identity_sha256"], 32, "host source"), _hex(actor_map["TRUSTED_HOST_REPLAY"]["runtime_identity_sha256"], 32, "host runtime"))
    for actor_id, actor_replay, endpoint_key in (
        ("PYTHON_ENDPOINT", dual.python, "python_stdout_hex"),
        ("RUST_ENDPOINT", dual.rust, "rust_stdout_hex"),
    ):
        envelope = _wire.validate_actor_stdout_envelope_v1(actor_replay.stdout_payload)
        actor = actor_map[actor_id]
        if (
            envelope["source_identity_sha256"]
            != actor["source_evidence"]["source_identity_sha256"]
            or envelope["runtime_identity_sha256"]
            != actor["runtime_identity_sha256"]
            or actor["control_evidence"]["stdout_hex"] != endpoint[endpoint_key]
        ):
            _fail("REJECT_Q05B_ARTIFACT_ACTOR", f"{actor_id} envelope binding differs")
    cargo = _validate_cargo_v1(
        evidence["cargo_build_binary"], config,
        actor_map["RUST_ENDPOINT"]["source_evidence"]["source_identity_sha256"],
        actor_map["RUST_ENDPOINT"]["snapshot_identity"]["root_path"],
        payload_table[config["seccomp"]["build_profile"]][2],
        pinned_images["rust"]["image_id"],
    )
    if cargo["rust_image_inspect_hex"] != pinned_images["rust"]["raw_inspect_hex"] or cargo["rust_image_inspect_sha256"] != pinned_images["rust"]["raw_inspect_sha256"]:
        _fail("REJECT_Q05B_ARTIFACT_CARGO", "Rust pinned image preimage differs")
    if (
        cargo["binary_runtime_identity_sha256"]
        != actor_map["RUST_ENDPOINT"]["runtime_identity_sha256"]
        or cargo["rust_snapshot_post_build"]
        != actor_map["RUST_ENDPOINT"]["snapshot_identity"]
    ):
        _fail("REJECT_Q05B_ARTIFACT_CARGO", "Rust actor/binary identity differs")
    stdout_tree = _validate_sealed_tree_identity_v1(
        endpoint["sealed_stdout_tree"],
        {"manifest.json": _hex_any(endpoint["manifest_hex"], "stdout manifest"), "python.stdout": _hex_any(endpoint["python_stdout_hex"], "Python stdout"), "rust.stdout": _hex_any(endpoint["rust_stdout_hex"], "Rust stdout")},
        (), "sealed stdout tree",
    )
    host_stage = _object(evidence["host_stage"], {"host_control_stdout_hex", "host_execution_binding_preimage", "loaded_module_root", "loaded_module_rows", "staged_sidecar_rows", "staging_tree", "witness_hex", "witness_root"}, "host_stage")
    expected_staged_rows = [
        [row["path"], len(payload), sha256(payload).hexdigest(), 0o444]
        for row, payload in zip(sidecar_rows, payloads, strict=True)
    ]
    if host_stage["staged_sidecar_rows"] != expected_staged_rows:
        _fail("REJECT_Q05B_ARTIFACT_HOST", "host staged sidecars differ")
    witness = _hex_any(host_stage["witness_hex"], "host witness")
    decoded_witness = _host.decode_host_semantic_witness_v1(
        witness, dual, negative_cbor, negative.corpus_root,
        negative.category_roots,
    )
    if host_stage["witness_root"] != decoded_witness["witness_root"]:
        _fail("REJECT_Q05B_ARTIFACT_HOST", "host witness root differs")
    staging_tree = _validate_sealed_tree_identity_v1(
        host_stage["staging_tree"],
        dict(sorted({**{f"sidecars/{path}": payload for path, payload in sidecar_payload_map.items()}, "host-semantic-witness.json": witness}.items())),
        ("sidecars", "sidecars/neutral", "sidecars/preimages"), "host staging tree",
    )
    host_control = _hex_any(host_stage["host_control_stdout_hex"], "host control")
    loaded_rows, loaded_root = _validate_loaded_modules_v1(
        host_stage["loaded_module_rows"], payload_table
    )
    if host_stage["loaded_module_root"] != loaded_root:
        _fail("REJECT_Q05B_ARTIFACT_HOST", "loaded module root differs")
    host_control_value = _validate_host_control_v1(
        host_control,
        loaded_rows,
        loaded_root,
        witness,
        decoded_witness["witness_root"],
        dual.dual_replay_root.hex(),
        actor_map["TRUSTED_HOST_REPLAY"],
    )
    if actor_map["TRUSTED_HOST_REPLAY"]["control_evidence"]["stdout_hex"] != host_stage["host_control_stdout_hex"]:
        _fail("REJECT_Q05B_ARTIFACT_HOST", "host held stdout differs")
    resources = _validate_resources_v1(evidence["final_resource_rows"], config)
    actor_stdouts = (
        _hex_any(endpoint["python_stdout_hex"], "Python actor stdout"),
        _hex_any(endpoint["rust_stdout_hex"], "Rust actor stdout"),
        host_control,
    )
    for index, (actor, stdout, resource) in enumerate(
        zip(actor_rows, actor_stdouts, resources, strict=True), start=1
    ):
        control = _validate_actor_control_exact_v1(
            actor["control_evidence"], actor, index, stdout, resource, config,
            payload_table[config["seccomp"]["runtime_profile"]][2],
            pinned_images["rust" if index == 2 else "python"]["image_id"],
        )
        for sample in resource["live_sample_objects"]:
            if (
                sample["mount_command_sha256"] != control["command_sha256"]
                or sample["mount_registry_sha256"] != control["mount_registry_sha256"]
            ):
                _fail("REJECT_Q05B_ARTIFACT_RESOURCE", "sample mount binding differs")
    actual_admission = _replay_actual_admission_artifact_evidence_v1(
        evidence["actual_admission"],
        source_commit=commit,
        commit_a_config_bytes=payload_table[config_path][2],
        commit_a_config_git_blob_oid=payload_table[config_path][1],
        stage_5_actor_completion_rows=[
            actor_rows[0]["control_evidence"],
            actor_rows[1]["control_evidence"],
        ],
        stage_5_strict_endpoint_replay_roots=[
            dual.python.host_replay_root.hex(),
            dual.rust.host_replay_root.hex(),
        ],
        five_sidecars=sidecar_section,
        endpoint_stdout_set=endpoint,
    )
    _cross_prior_stage12_top_v1(
        source,
        payload_table,
        config,
        pinned_images,
        actor_rows,
        cargo,
        actual_admission,
    )
    _cross_cargo_actual_admission_v1(cargo, actual_admission)
    _cross_docker_execution_ownership_v1(
        config,
        pinned_images,
        actor_rows,
        cargo,
        actual_admission,
    )
    python_mounts = _mount_sources_v1(actor_rows[0]["command"])
    rust_mounts = _mount_sources_v1(actor_rows[1]["command"])
    host_mounts = _mount_sources_v1(actor_rows[2]["command"])
    expected_python_mounts = {
        "/snapshot": actor_rows[0]["snapshot_identity"]["root_path"],
        "/output": python_output_tree["root_path"],
        "/control": actor_rows[0]["control_evidence"]["control_root_path"],
    }
    expected_rust_mounts = {
        "/runtime/hegel-q1-archive-projection-oracle": cargo["binary_path"],
        "/output": rust_output_tree["root_path"],
        "/control": actor_rows[1]["control_evidence"]["control_root_path"],
    }
    stdout_root = stdout_tree["root_path"].rstrip("/")
    expected_host_mounts = {
        "/snapshot": actor_rows[2]["snapshot_identity"]["root_path"],
        "/inputs/python": python_output_tree["root_path"],
        "/inputs/rust": rust_output_tree["root_path"],
        "/inputs/stdout/python.stdout": stdout_root + "/python.stdout",
        "/inputs/stdout/rust.stdout": stdout_root + "/rust.stdout",
        "/inputs/stdout/manifest.json": stdout_root + "/manifest.json",
        "/control": actor_rows[2]["control_evidence"]["control_root_path"],
        "/staging": staging_tree["root_path"],
    }
    if python_mounts != expected_python_mounts or rust_mounts != expected_rust_mounts or host_mounts != expected_host_mounts:
        _fail("REJECT_Q05B_ARTIFACT_ISOLATION", "actor mount/preimage binding differs")
    host_payload = actor_rows[2]["command"][actor_rows[2]["command"].index("hegel-q05b-held-actor") + 1:]
    if host_payload[-3] != actor_rows[2]["source_evidence"]["source_identity_sha256"] or host_payload[-1] != actor_rows[2]["runtime_identity_sha256"]:
        _fail("REJECT_Q05B_ARTIFACT_ISOLATION", "host command identity binding differs")
    scratch = _validate_scratch_v1(
        evidence["scratch_rows"],
        (
            dual.python.host_replay_root,
            dual.rust.host_replay_root,
            bytes.fromhex(decoded_witness["host_scratch_evidence_root"]),
        ),
    )
    expected_scratch_registries = (
        [
            [root.hex() for root in replay.scratch_ledger_roots]
            for replay in dual.python.partition_replays
        ],
        [
            [root.hex() for root in replay.scratch_ledger_roots]
            for replay in dual.rust.partition_replays
        ],
        decoded_witness["host_scratch_partition_roots"],
    )
    if any(
        row["partition_scratch_ledger_roots"] != expected
        for row, expected in zip(scratch, expected_scratch_registries, strict=True)
    ):
        _fail("REJECT_Q05B_ARTIFACT_SCRATCH", "scratch ledger replay differs")
    resource_preimage = {"final_resource_rows": resources}
    resource_root = _json_root(RESOURCE_EVIDENCE_ROOT_DOMAIN, resource_preimage)
    expected_host_binding = {"host_actor_row": actor_map["TRUSTED_HOST_REPLAY"], "host_control_sha256": sha256(host_control).hexdigest(), "host_final_resource": resources[2], "loaded_module_root": loaded_root, "semantic_replay_root": dual.dual_replay_root.hex(), "witness_root": decoded_witness["witness_root"]}
    if host_stage["host_execution_binding_preimage"] != expected_host_binding:
        _fail("REJECT_Q05B_ARTIFACT_HOST", "host execution binding preimage differs")
    host_binding_root = _json_root(HOST_EXECUTION_BINDING_ROOT_DOMAIN, expected_host_binding)
    isolation_preimage = {
        "actual_admission": evidence["actual_admission"],
        "actor_rows": evidence["actor_rows"],
        "cargo_build_binary": cargo,
        "endpoint_stdout_set": endpoint,
        "final_resource_rows": resources,
        "five_sidecars": evidence["five_sidecars"],
        "host_stage": evidence["host_stage"],
        "negative_corpus": evidence["negative_corpus"],
        "scratch_rows": scratch,
        "source_wire_profile": evidence["source_wire_profile"],
    }
    isolation_root = _json_root(ISOLATION_EVIDENCE_ROOT_DOMAIN, isolation_preimage)
    bundle_preimage = {"actual_admission_evidence_root": actual_admission["actual_admission_evidence_root"], "five_sidecars": evidence["five_sidecars"], "host_witness_root": decoded_witness["witness_root"], "scratch_rows": scratch, "semantic_replay_root": dual.dual_replay_root.hex()}
    bundle_root = _json_root(BUNDLE_EVIDENCE_ROOT_DOMAIN, bundle_preimage)
    semantic = _object(evidence["semantic_execution"], {"bundle_preimage", "host_execution_binding_preimage", "isolation_preimage", "resource_preimage", "semantic_component_root"}, "semantic_execution")
    if semantic != {"bundle_preimage": bundle_preimage, "host_execution_binding_preimage": expected_host_binding, "isolation_preimage": isolation_preimage, "resource_preimage": resource_preimage, "semantic_component_root": dual.predicate11_semantic_component_root.hex()}:
        _fail("REJECT_Q05B_ARTIFACT_SEMANTIC", "semantic execution preimage differs")
    negative_roots = dict(negative.category_roots)
    host_predicates = dict(dual.predicate_evidence_rows)
    implementation_roots = tuple(_json_root(ISOLATION_EVIDENCE_ROOT_DOMAIN, {"actor": row}) for row in actor_rows)
    if len(set(implementation_roots)) != 3:
        _fail("REJECT_Q05B_ARTIFACT_ACTOR", "implementation roots are not distinct")
    preimages = {
        1: source,
        2: [[row["path"], row["raw_sha256"]] for row in sidecar_rows],
        3: [leaf.manifest_root.hex(), len(leaf.rows)],
        4: [golden.q1_semantic_binding_root.hex(), golden.q1_projection_profile_root.hex(), source["q0_receipt_root"]],
        5: CLOSED_Q1_AUTHORITY,
        6: [dual.dual_replay_root.hex(), host_predicates[6].hex()],
        7: host_predicates[7].hex(),
        8: host_predicates[8].hex(),
        9: actor_rows[0],
        10: actor_rows[1],
        12: [host_predicates[12].hex(), shadow.root.hex()],
        13: {
            "category13_root": negative_roots[13].hex(),
            "rust_offline_test": cargo["rust_test"],
        },
        14: host_predicates[14].hex(),
        15: host_predicates[15].hex(),
        16: scratch,
        17: host_predicates[17].hex(),
        18: negative_roots[18].hex(),
    }
    predicate_rows = []
    for predicate_id, name in _wire.QUALIFICATION_PREDICATE_REGISTRY[:19]:
        if predicate_id == 11:
            root = _json_root(PREDICATE11_EVIDENCE_ROOT_DOMAIN, {"host_execution_binding_root": host_binding_root.hex(), "semantic_component_root": dual.predicate11_semantic_component_root.hex()})
        elif predicate_id == 19:
            root = _json_root(PREDICATE19_EVIDENCE_ROOT_DOMAIN, {"isolation_evidence_root": isolation_root.hex(), "passed": True})
        else:
            root = _json_root(PREDICATE_EVIDENCE_ROOT_DOMAIN, {"predicate_id": predicate_id, "predicate_name": name.hex(), "preimage": preimages[predicate_id]})
        predicate_rows.append((predicate_id, name, True, root))
    predicate_rows_t = tuple(predicate_rows)
    candidate = _wire.Q05BQualificationCandidateReceiptV1(source_raw, golden.q1_semantic_binding_root, golden.q1_projection_profile_root, _wire.Q0_SATURATION_RECEIPT_ROOT_FROM_Q1_PREREGISTRATION, leaf.manifest_root, implementation_roots, (golden.manifest_root,) * 3, tuple(row[2] for row in golden.bounded_state_rows), bundle_root, isolation_root, resource_root, _wire.pre_receipt_evidence_root_v1(source_raw, predicate_rows_t), predicate_rows_t)
    decoded_candidate = _wire.decode_qualification_candidate_receipt_v1(candidate.canonical_bytes)
    candidate_result = {
        "actual_admission_evidence_root": actual_admission[
            "actual_admission_evidence_root"
        ],
        "bundle_evidence_root": bundle_root.hex(),
        "candidate_receipt_cbor_hex": decoded_candidate.canonical_bytes.hex(),
        "candidate_receipt_root": decoded_candidate.receipt_root.hex(),
        "closed_q1_authority": CLOSED_Q1_AUTHORITY,
        "host_execution_binding_root": host_binding_root.hex(),
        "isolation_evidence_root": isolation_root.hex(),
        "ordered_predicate_rows": [
            [row[0], row[1].decode("ascii"), row[2], row[3].hex()]
            for row in predicate_rows_t
        ],
        "qualification_count": 19,
        "qualification_mask": 0x7FFFF,
        "resource_evidence_root": resource_root.hex(),
    }
    candidate_result = validate_stage8_candidate_registry_v1(
        candidate_result,
        actual_admission["actual_admission_evidence_root"],
    )
    if candidate_only:
        return candidate_result
    final = _wire.Q05BQualificationReceiptV1(decoded_candidate)
    decoded_final = _wire.decode_qualification_receipt_v1(final.canonical_bytes)
    if decoded_final.candidate_receipt.canonical_bytes != candidate.canonical_bytes:
        _fail("REJECT_Q05B_ARTIFACT_RECEIPT", "candidate/final mismatch")
    sections = {name: evidence[name] for name in SECTION_NAMES}
    derived = {"actual_admission_evidence_root": actual_admission["actual_admission_evidence_root"], "artifact_set_root": None, "bundle_evidence_root": bundle_root.hex(), "candidate_receipt_cbor_hex": candidate.canonical_bytes.hex(), "candidate_receipt_root": candidate.receipt_root.hex(), "closed_q1_authority": CLOSED_Q1_AUTHORITY, "final_receipt_cbor_hex": final.canonical_bytes.hex(), "final_receipt_root": final.receipt_root.hex(), "host_execution_binding_root": host_binding_root.hex(), "isolation_evidence_root": isolation_root.hex(), "ordered_predicate_rows": [[row[0], row[1].decode("ascii"), row[2], row[3].hex()] for row in predicate_rows_t], "qualification_count": 20, "qualification_mask": 0xFFFFF, "resource_evidence_root": resource_root.hex()}
    artifact = {"derived": derived, "schema_version": ARTIFACT_SCHEMA_VERSION, "sections": sections, "status": "Q05B_QUALIFICATION_20_OF_20_Q1_NOT_RUN"}
    body = json.loads(_canonical_json(artifact).decode("ascii")); body["derived"].pop("artifact_set_root")
    artifact["derived"]["artifact_set_root"] = _json_root(ARTIFACT_SET_ROOT_DOMAIN, body).hex()
    validate_stage9_derived_registry_v1(
        artifact["derived"],
        candidate_result,
        actual_admission["actual_admission_evidence_root"],
    )
    return artifact


def _strict_stage_receipt_hex_v1(value: object, name: str) -> bytes:
    payload = _hex_any(value, name)
    if not payload:
        _fail("REJECT_Q05B_ARTIFACT_RECEIPT", f"{name} is empty")
    return payload


def _validate_stage_predicate_rows_v1(
    value: object,
    decoded_candidate: object,
) -> list[object]:
    if type(value) is not list or len(value) != 19:
        _fail(
            "REJECT_Q05B_ARTIFACT_RECEIPT",
            "Stage-8 candidate requires exactly 19 predicate rows",
        )
    decoded_rows = decoded_candidate.predicate_rows_1_through_19
    for expected, row, decoded in zip(
        _wire.QUALIFICATION_PREDICATE_REGISTRY[:19],
        value,
        decoded_rows,
        strict=True,
    ):
        predicate_id, predicate_name = expected
        if (
            type(row) is not list
            or len(row) != 4
            or type(row[0]) is not int
            or row[0] != predicate_id
            or type(row[1]) is not str
            or row[1] != predicate_name.decode("ascii")
            or type(row[2]) is not bool
            or row[2] is not True
        ):
            _fail(
                "REJECT_Q05B_ARTIFACT_RECEIPT",
                f"Stage-8 predicate row {predicate_id} differs",
            )
        evidence_root = _hex(row[3], 32, f"predicate {predicate_id} root")
        if (
            type(decoded) is not tuple
            or len(decoded) != 4
            or type(decoded[0]) is not int
            or decoded[0] != predicate_id
            or decoded[1] != predicate_name
            or type(decoded[2]) is not bool
            or decoded[2] is not True
            or decoded[3] != evidence_root
        ):
            _fail(
                "REJECT_Q05B_ARTIFACT_RECEIPT",
                f"Stage-8 predicate row {predicate_id} does not match CBOR",
            )
    return value


def validate_stage8_candidate_registry_v1(
    value: object,
    actual_admission_evidence_root: str,
) -> dict[str, object]:
    """Strictly bind the exact 11-key Stage-8 adapter to candidate CBOR."""

    candidate = _object(
        value,
        set(STAGE8_CANDIDATE_REGISTRY_KEYS),
        "Stage-8 candidate registry",
    )
    admission_root = _hex(
        actual_admission_evidence_root,
        32,
        "expected actual admission evidence root",
    )
    if _hex(
        candidate["actual_admission_evidence_root"],
        32,
        "Stage-8 actual admission evidence root",
    ) != admission_root:
        _fail(
            "REJECT_Q05B_ARTIFACT_ADMISSION",
            "Stage-8 admission root differs from admitted section",
        )
    for name in (
        "bundle_evidence_root",
        "candidate_receipt_root",
        "host_execution_binding_root",
        "isolation_evidence_root",
        "resource_evidence_root",
    ):
        _hex(candidate[name], 32, f"Stage-8 {name}")
    if (
        type(candidate["qualification_count"]) is not int
        or candidate["qualification_count"] != 19
        or type(candidate["qualification_mask"]) is not int
        or candidate["qualification_mask"] != 0x7FFFF
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_RECEIPT",
            "Stage-8 qualification count/mask differs",
        )
    _require_type_exact_v1(
        candidate["closed_q1_authority"],
        CLOSED_Q1_AUTHORITY,
        "Stage-8 closed Q1 authority",
    )
    candidate_payload = _strict_stage_receipt_hex_v1(
        candidate["candidate_receipt_cbor_hex"],
        "Stage-8 candidate receipt CBOR",
    )
    try:
        decoded = _wire.decode_qualification_candidate_receipt_v1(
            candidate_payload
        )
    except Exception as error:
        _fail(
            "REJECT_Q05B_ARTIFACT_RECEIPT",
            f"Stage-8 candidate CBOR strict decode failed: {error}",
        )
    if (
        decoded.canonical_bytes != candidate_payload
        or decoded.receipt_root
        != _hex(
            candidate["candidate_receipt_root"],
            32,
            "Stage-8 candidate receipt root",
        )
        or decoded.bundle_evidence_root
        != _hex(candidate["bundle_evidence_root"], 32, "Stage-8 bundle root")
        or decoded.isolation_evidence_root
        != _hex(
            candidate["isolation_evidence_root"],
            32,
            "Stage-8 isolation root",
        )
        or decoded.resource_evidence_root
        != _hex(
            candidate["resource_evidence_root"],
            32,
            "Stage-8 resource root",
        )
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_RECEIPT",
            "Stage-8 candidate CBOR/root registry differs",
        )
    _validate_stage_predicate_rows_v1(
        candidate["ordered_predicate_rows"], decoded
    )
    return candidate


def validate_stage9_derived_registry_v1(
    value: object,
    stage8_candidate: Mapping[str, object],
    actual_admission_evidence_root: str,
) -> dict[str, object]:
    """Strictly bind the exact 14-key derived registry to Stage 8 and receipts."""

    candidate = validate_stage8_candidate_registry_v1(
        stage8_candidate, actual_admission_evidence_root
    )
    derived = _object(
        value,
        set(STAGE9_DERIVED_REGISTRY_KEYS),
        "Stage-9 derived registry",
    )
    for name in (
        "actual_admission_evidence_root",
        "artifact_set_root",
        "bundle_evidence_root",
        "candidate_receipt_root",
        "final_receipt_root",
        "host_execution_binding_root",
        "isolation_evidence_root",
        "resource_evidence_root",
    ):
        _hex(derived[name], 32, f"Stage-9 {name}")
    for name in (
        "actual_admission_evidence_root",
        "bundle_evidence_root",
        "candidate_receipt_cbor_hex",
        "candidate_receipt_root",
        "closed_q1_authority",
        "host_execution_binding_root",
        "isolation_evidence_root",
        "ordered_predicate_rows",
        "resource_evidence_root",
    ):
        try:
            _require_type_exact_v1(
                derived[name],
                candidate[name],
                f"Stage-9 candidate-era field {name}",
            )
        except Q05BActualArtifactError:
            _fail(
                "REJECT_Q05B_ARTIFACT_RECEIPT",
                f"Stage-9 candidate-era field {name} differs",
            )
    if (
        derived["actual_admission_evidence_root"]
        != actual_admission_evidence_root
        or type(derived["qualification_count"]) is not int
        or derived["qualification_count"] != 20
        or type(derived["qualification_mask"]) is not int
        or derived["qualification_mask"] != 0xFFFFF
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_RECEIPT",
            "Stage-9 qualification/admission transition differs",
        )
    _require_type_exact_v1(
        derived["closed_q1_authority"],
        CLOSED_Q1_AUTHORITY,
        "Stage-9 closed Q1 authority",
    )
    candidate_payload = _strict_stage_receipt_hex_v1(
        derived["candidate_receipt_cbor_hex"],
        "Stage-9 candidate receipt CBOR",
    )
    final_payload = _strict_stage_receipt_hex_v1(
        derived["final_receipt_cbor_hex"],
        "Stage-9 final receipt CBOR",
    )
    try:
        decoded_candidate = _wire.decode_qualification_candidate_receipt_v1(
            candidate_payload
        )
        decoded_final = _wire.decode_qualification_receipt_v1(final_payload)
    except Exception as error:
        _fail(
            "REJECT_Q05B_ARTIFACT_RECEIPT",
            f"Stage-9 receipt strict decode failed: {error}",
        )
    if (
        decoded_candidate.receipt_root
        != _hex(derived["candidate_receipt_root"], 32, "Stage-9 candidate root")
        or decoded_final.receipt_root
        != _hex(derived["final_receipt_root"], 32, "Stage-9 final root")
        or decoded_final.candidate_receipt.canonical_bytes != candidate_payload
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_RECEIPT",
            "Stage-9 candidate/final receipt binding differs",
        )
    return derived


def replay_actual_evidence_1_19_v1(
    evidence: Mapping[str, object],
) -> dict[str, object]:
    """Strictly replay all embedded preimages and emit only the 1--19 candidate."""

    return _replay_actual_evidence_v1(evidence, candidate_only=True)


def build_actual_artifact_v1(evidence: Mapping[str, object]) -> dict[str, object]:
    """Strictly replay evidence, then add predicate 20 and the final receipt."""

    if type(evidence) is not dict:
        _fail(
            "REJECT_Q05B_ARTIFACT_SCHEMA",
            "artifact evidence must be an exact object",
        )
    artifact = _replay_actual_evidence_v1(evidence, candidate_only=False)
    if len(_canonical_json(artifact)) > ACTUAL_ARTIFACT_MAX_CANONICAL_BYTES:
        _fail(
            "REJECT_Q05B_ARTIFACT_SIZE",
            "artifact canonical bytes exceed frozen maximum",
        )
    return artifact


def canonical_actual_artifact_bytes_v1(value: object) -> bytes:
    if type(value) is not dict:
        _fail("REJECT_Q05B_ARTIFACT_SCHEMA", "artifact value must be exact object")
    payload = _canonical_json(value)
    if len(payload) > ACTUAL_ARTIFACT_MAX_CANONICAL_BYTES:
        _fail(
            "REJECT_Q05B_ARTIFACT_SIZE",
            "artifact canonical bytes exceed frozen maximum",
        )
    replayed = decode_and_replay_actual_artifact_v1(payload)
    if replayed != value:
        _fail("REJECT_Q05B_ARTIFACT_REPLAY", "artifact value differs from replay")
    return payload


def decode_and_replay_actual_artifact_v1(payload: bytes) -> dict[str, object]:
    if (
        type(payload) is not bytes
        or len(payload) > ACTUAL_ARTIFACT_MAX_CANONICAL_BYTES
    ):
        _fail(
            "REJECT_Q05B_ARTIFACT_SIZE",
            "artifact payload must be exact bytes within frozen maximum",
        )
    value = _strict_json(payload)
    _object(value, {"derived", "schema_version", "sections", "status"}, "artifact")
    if value["schema_version"] != ARTIFACT_SCHEMA_VERSION or value["status"] != "Q05B_QUALIFICATION_20_OF_20_Q1_NOT_RUN":
        _fail("REJECT_Q05B_ARTIFACT_SCHEMA", "artifact header differs")
    sections = _object(value["sections"], set(SECTION_NAMES), "sections")
    rebuilt = build_actual_artifact_v1(sections)
    if rebuilt != value:
        _fail("REJECT_Q05B_ARTIFACT_REPLAY", "artifact does not equal strict replay")
    return rebuilt


def actual_artifact_summary_v1(value: object) -> dict[str, object]:
    replayed = decode_and_replay_actual_artifact_v1(_canonical_json(value))
    derived = replayed["derived"]
    payload = _canonical_json(replayed)
    return {"artifact_set_root": derived["artifact_set_root"], "canonical_sha256": sha256(payload).hexdigest(), "candidate_receipt_root": derived["candidate_receipt_root"], "final_receipt_root": derived["final_receipt_root"]}


__all__ = [
    "ACTUAL_ADMISSION_ARTIFACT_EVIDENCE_ROOT_DOMAIN",
    "ACTUAL_ADMISSION_ARTIFACT_EVIDENCE_SCHEMA_VERSION",
    "ACTUAL_ARTIFACT_MAX_CANONICAL_BYTES", "ARTIFACT_SCHEMA_VERSION",
    "ARTIFACT_SET_ROOT_DOMAIN", "BUNDLE_EVIDENCE_ROOT_DOMAIN",
    "CLOSED_Q1_AUTHORITY", "HOST_EXECUTION_BINDING_ROOT_DOMAIN", "ISOLATION_EVIDENCE_ROOT_DOMAIN",
    "PREDICATE11_EVIDENCE_ROOT_DOMAIN", "PREDICATE19_EVIDENCE_ROOT_DOMAIN", "Q05BActualArtifactError",
    "RESOURCE_EVIDENCE_ROOT_DOMAIN", "actual_artifact_summary_v1",
    "STAGE8_CANDIDATE_REGISTRY_KEYS", "STAGE9_DERIVED_REGISTRY_KEYS",
    "build_actual_admission_artifact_evidence_v1", "build_actual_artifact_v1",
    "canonical_actual_artifact_bytes_v1", "decode_and_replay_actual_artifact_v1",
    "replay_actual_evidence_1_19_v1", "validate_stage8_candidate_registry_v1",
    "validate_stage9_derived_registry_v1",
]
