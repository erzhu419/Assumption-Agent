from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import fcntl
import functools
import hashlib
import http.client
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..models import stable_hash
from .semantic_assignment_operator_v1 import (
    ALL_DESTINATIONS,
    OfflineMiniLMEncoder,
    PUBLIC_DEFAULT_DESTINATION,
    _atomic_write_json,
    _canonical_json_bytes,
    _payload_hash,
    _sha256_bytes,
    _sha256_file,
    build_semantic_assignment_plan,
    load_operator_asset,
)


OA_FEASIBILITY_VERSION = "semantic_assignment_public_oa_feasibility_v1"
OA_PACK_VERSION = "semantic_assignment_public_oa_pack_v1"
OA_REPORT_VERSION = "semantic_assignment_public_oa_report_v1"
OA_DECISION_LOCK_VERSION = "semantic_assignment_public_oa_decision_lock_v1"
OPENALEX_API = "https://api.openalex.org/works"
EXPECTED_STRATA = (
    "LLM",
    "trapped_ion_and_qc",
    "black_hole",
    "DNA",
    "music_history",
    "unrelated_public_default",
)

MAXIMUM_JSON_BYTES = 32 * 1024 * 1024
MAXIMUM_PDF_BYTES = 64 * 1024 * 1024
MAXIMUM_EXTRACTED_CHARACTERS = 4096
PDF_PAGES = 2
PDFTOTEXT_PATH = Path("/usr/bin/pdftotext")
NETWORK_ISOLATION_POLICY = "linux_netns_loopback_only_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OPENALEX_ID = re.compile(r"(?:https?://openalex\.org/)?(W[0-9]+)\Z", re.I)
_ARXIV_ID = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/)?([0-9]{4}\.[0-9]{4,5}|[a-z-]+/[0-9]{7})(?:v[0-9]+)?(?:\.pdf)?",
    re.I,
)


class OaFeasibilityError(RuntimeError):
    pass


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise OaFeasibilityError(f"{label} is not a sha256 digest")
    return value


def _read_json(path: str | Path, *, maximum: int = MAXIMUM_JSON_BYTES) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if resolved.stat().st_size > maximum:
        raise OaFeasibilityError("JSON input exceeds its byte bound")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OaFeasibilityError("JSON input is unreadable") from error
    if not isinstance(payload, dict):
        raise OaFeasibilityError("JSON input must be an object")
    return payload


def _verify_self_hash(
    payload: Mapping[str, Any], *, hash_field: str, label: str
) -> str:
    declared = _require_sha256(payload.get(hash_field), f"{label} hash")
    body = dict(payload)
    del body[hash_field]
    if stable_hash(body) != declared:
        raise OaFeasibilityError(f"{label} hash mismatch")
    return declared


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode(
        "utf-8"
    ) + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise FileExistsError("exclusive OA artifact already exists") from error


def _safe_public_https_url(value: str) -> str:
    if not isinstance(value, str) or any(
        ord(character) <= 32 or character.isspace() for character in value
    ):
        raise OaFeasibilityError("OA PDF URL contains unsafe characters")
    try:
        parsed = urllib.parse.urlparse(value)
        host = (parsed.hostname or "").casefold()
        port = parsed.port or 443
    except (TypeError, ValueError) as error:
        raise OaFeasibilityError("OA PDF URL is malformed") from error
    if (
        parsed.scheme.casefold() != "https"
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or host == "localhost"
        or host.endswith(".localhost")
        or host == "arxiv.org"
        or host.endswith(".arxiv.org")
    ):
        raise OaFeasibilityError("OA PDF URL is not safe public HTTPS")
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, ValueError) as error:
        raise OaFeasibilityError("OA PDF host cannot be resolved") from error
    ips = {row[4][0].split("%", 1)[0] for row in addresses}
    try:
        globally_routable = bool(ips) and all(
            ipaddress.ip_address(address).is_global for address in ips
        )
    except ValueError as error:
        raise OaFeasibilityError("OA PDF host address is malformed") from error
    if not globally_routable:
        raise OaFeasibilityError("OA PDF host is not globally routable")
    return urllib.parse.urlunparse(parsed)


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> urllib.request.Request | None:
        safe_url = _safe_public_https_url(new_url)
        return super().redirect_request(
            request, file_pointer, code, message, headers, safe_url
        )


def _with_acquisition_guard(function: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
        output_root = kwargs.get("output_root")
        preregistration_path = kwargs.get("preregistration_path")
        if output_root is None or preregistration_path is None:
            raise TypeError(
                "output_root and preregistration_path must be passed by keyword"
            )
        preregistration = load_preregistration(preregistration_path)
        repository_root = (
            Path(preregistration_path).expanduser().resolve(strict=True).parents[1]
        )
        expected_root = (
            repository_root / preregistration["formal_paths"]["pack_root"]
        ).resolve()
        root = Path(output_root).expanduser().resolve()
        if root != expected_root:
            raise OaFeasibilityError("OA pack output path is not preregistered")
        root.mkdir(parents=True, exist_ok=True)
        handle = (root / ".acquisition.guard").open("a+", encoding="utf-8")
        try:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise FileExistsError("OA acquisition is already running") from error
            return function(*args, **kwargs)
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    return wrapped


def load_preregistration(path: str | Path) -> dict[str, Any]:
    resolved_path = Path(path).expanduser().resolve(strict=True)
    preregistration = _read_json(resolved_path)
    if preregistration.get("preregistration_version") != OA_FEASIBILITY_VERSION:
        raise OaFeasibilityError("OA preregistration version mismatch")
    _verify_self_hash(
        preregistration, hash_field="manifest_hash", label="OA preregistration"
    )
    if preregistration.get("decision_budget") != 1:
        raise OaFeasibilityError("OA decision budget must be one")
    acquisition = preregistration.get("acquisition")
    if not isinstance(acquisition, dict):
        raise OaFeasibilityError("OA acquisition policy is missing")
    strata = acquisition.get("strata")
    if not isinstance(strata, list) or tuple(
        str(row.get("stratum")) for row in strata if isinstance(row, dict)
    ) != EXPECTED_STRATA:
        raise OaFeasibilityError("OA stratum order mismatch")
    if acquisition.get("records_per_stratum") != 10:
        raise OaFeasibilityError("OA stratum size mismatch")
    if acquisition.get("total_records") != 60:
        raise OaFeasibilityError("OA total pack size mismatch")
    evaluation = preregistration.get("evaluation", {})
    if (
        evaluation.get("required_correct") != 60
        or evaluation.get("required_evidence_valid") != 60
        or evaluation.get("network_isolation") != NETWORK_ISOLATION_POLICY
    ):
        raise OaFeasibilityError("OA exact decision threshold mismatch")
    bindings = preregistration.get("implementation_bindings")
    if not isinstance(bindings, dict):
        raise OaFeasibilityError("OA implementation binding is missing")
    files = bindings.get("files")
    if not isinstance(files, list) or stable_hash(files) != bindings.get(
        "file_set_hash"
    ):
        raise OaFeasibilityError("OA implementation file set is malformed")
    repository_root = resolved_path.parents[1]
    for row in files:
        if not isinstance(row, dict) or set(row) != {
            "relative_path",
            "file_sha256",
        }:
            raise OaFeasibilityError("OA implementation file row is malformed")
        relative = Path(str(row["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise OaFeasibilityError("OA implementation path is unsafe")
        implementation = (repository_root / relative).resolve(strict=True)
        if repository_root not in implementation.parents:
            raise OaFeasibilityError("OA implementation escaped repository")
        if _sha256_file(implementation) != row["file_sha256"]:
            raise OaFeasibilityError("OA implementation binding drifted")
    return preregistration


def verify_extraction_runtime(
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    declared = preregistration.get("extraction_runtime")
    if not isinstance(declared, dict):
        raise OaFeasibilityError("OA extraction runtime binding is missing")
    expected_path = Path(str(declared.get("pdftotext_path") or ""))
    if expected_path != PDFTOTEXT_PATH or not PDFTOTEXT_PATH.is_file():
        raise OaFeasibilityError("OA pdftotext path drifted")
    if _sha256_file(PDFTOTEXT_PATH) != declared.get("pdftotext_sha256"):
        raise OaFeasibilityError("OA pdftotext binary drifted")
    try:
        completed = subprocess.run(
            [str(PDFTOTEXT_PATH), "-v"],
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise OaFeasibilityError("OA pdftotext version probe failed") from error
    version_line = (completed.stderr or completed.stdout).splitlines()
    if completed.returncode != 0 or not version_line or (
        version_line[0].strip() != declared.get("pdftotext_version_line")
    ):
        raise OaFeasibilityError("OA pdftotext version drifted")
    if (
        declared.get("pdf_pages") != PDF_PAGES
        or declared.get("maximum_extracted_characters")
        != MAXIMUM_EXTRACTED_CHARACTERS
    ):
        raise OaFeasibilityError("OA extraction policy drifted")
    return dict(declared)


def _canonical_openalex_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    match = _OPENALEX_ID.fullmatch(value.strip())
    return match.group(1).upper() if match else None


def _canonical_doi(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = urllib.parse.unquote(value.strip()).casefold()
    normalized = re.sub(
        r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", normalized
    ).strip()
    return normalized if normalized.startswith("10.") and "/" in normalized else None


def _abstract_text(work: Mapping[str, Any]) -> str:
    inverted = work.get("abstract_inverted_index")
    if not isinstance(inverted, dict):
        return ""
    positions: list[tuple[int, str]] = []
    for token, indexes in inverted.items():
        if not isinstance(token, str) or not isinstance(indexes, list):
            continue
        for index in indexes:
            if isinstance(index, int) and not isinstance(index, bool) and index >= 0:
                positions.append((index, token))
    positions.sort()
    return " ".join(token for _, token in positions)


def _work_text(work: Mapping[str, Any]) -> str:
    title = work.get("title") or work.get("display_name") or ""
    return " ".join(f"{title} {_abstract_text(work)}".split())


def _topic_text(work: Mapping[str, Any]) -> str:
    names: list[str] = []
    primary = work.get("primary_topic")
    if isinstance(primary, dict) and isinstance(primary.get("display_name"), str):
        names.append(primary["display_name"])
    topics = work.get("topics")
    if isinstance(topics, list):
        names.extend(
            row["display_name"]
            for row in topics
            if isinstance(row, dict) and isinstance(row.get("display_name"), str)
        )
    concepts = work.get("concepts")
    if isinstance(concepts, list):
        names.extend(
            row["display_name"]
            for row in concepts
            if isinstance(row, dict) and isinstance(row.get("display_name"), str)
        )
    return " ".join(names)


def _arxiv_ids(work: Mapping[str, Any]) -> set[str]:
    values: list[str] = []
    ids = work.get("ids")
    if isinstance(ids, dict):
        values.extend(str(value) for value in ids.values())
    for key in ("locations",):
        rows = work.get(key)
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    values.extend(
                        str(row.get(field) or "")
                        for field in ("landing_page_url", "pdf_url")
                    )
    found: set[str] = set()
    for value in values:
        for match in _ARXIV_ID.finditer(value):
            found.add(match.group(1).casefold())
    return found


def _matches_regex_list(text: str, values: object, *, require_all: bool) -> bool:
    if values in (None, []):
        return True
    if not isinstance(values, list) or any(not isinstance(row, str) for row in values):
        raise OaFeasibilityError("metadata regex policy is malformed")
    matches = [re.search(pattern, text, flags=re.I) is not None for pattern in values]
    return all(matches) if require_all else any(matches)


def metadata_candidate(
    work: Mapping[str, Any],
    *,
    stratum: Mapping[str, Any],
    known_train_arxiv_ids: set[str],
) -> dict[str, Any] | None:
    work_id = _canonical_openalex_id(work.get("id"))
    if work_id is None or work.get("type") != "article" or work.get("is_retracted"):
        return None
    date = work.get("publication_date")
    if not isinstance(date, str) or not (
        stratum["from_publication_date"] <= date <= stratum["to_publication_date"]
    ):
        return None
    location = work.get("best_oa_location")
    if not isinstance(location, dict):
        return None
    pdf_url = location.get("pdf_url")
    license_value = location.get("license")
    if (
        location.get("is_oa") is not True
        or not isinstance(pdf_url, str)
        or not pdf_url.startswith("https://")
        or not isinstance(license_value, str)
        or not license_value.strip()
    ):
        return None
    host = (urllib.parse.urlparse(pdf_url).hostname or "").casefold()
    if host == "arxiv.org" or host.endswith(".arxiv.org"):
        return None
    if _arxiv_ids(work).intersection(known_train_arxiv_ids):
        return None
    text = _work_text(work)
    topics = _topic_text(work)
    if not text or not topics:
        return None
    if not _matches_regex_list(
        text, stratum.get("required_text_all_regex"), require_all=True
    ):
        return None
    if not _matches_regex_list(
        text, stratum.get("required_text_any_regex"), require_all=False
    ):
        return None
    if not _matches_regex_list(
        topics, stratum.get("required_topic_any_regex"), require_all=False
    ):
        return None
    excluded_text = stratum.get("excluded_text_any_regex")
    if excluded_text and _matches_regex_list(text, excluded_text, require_all=False):
        return None
    doi = _canonical_doi(work.get("doi"))
    return {
        "openalex_id": work_id,
        "doi": doi,
        "pdf_url": pdf_url,
        "license": license_value,
        "expected_destination": stratum["expected_destination"],
        "stratum": stratum["stratum"],
    }


def rank_metadata_candidates(
    works: Iterable[Mapping[str, Any]],
    *,
    stratum: Mapping[str, Any],
    seed: str,
    known_train_arxiv_ids: set[str],
) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for work in works:
        candidate = metadata_candidate(
            work,
            stratum=stratum,
            known_train_arxiv_ids=known_train_arxiv_ids,
        )
        if candidate is None:
            continue
        candidate["sampling_hash"] = _sha256_bytes(
            f"{seed}||{candidate['openalex_id']}".encode("utf-8")
        )
        candidates.setdefault(candidate["openalex_id"], candidate)
    ranked = sorted(
        candidates.values(),
        key=lambda row: (row["sampling_hash"], row["openalex_id"]),
    )
    for rank, row in enumerate(ranked):
        row["sampling_rank"] = rank
    return ranked


def _http_json(url: str, *, timeout: float, user_agent: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(MAXIMUM_JSON_BYTES + 1)
    except (OSError, urllib.error.URLError) as error:
        raise OaFeasibilityError("OpenAlex metadata transport failed") from error
    if len(raw) > MAXIMUM_JSON_BYTES:
        raise OaFeasibilityError("OpenAlex response exceeds bound")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OaFeasibilityError("OpenAlex response is invalid") from error
    if not isinstance(value, dict):
        raise OaFeasibilityError("OpenAlex response is malformed")
    return value


def fetch_stratum_metadata(
    stratum: Mapping[str, Any],
    *,
    acquisition: Mapping[str, Any],
) -> list[dict[str, Any]]:
    per_page = int(acquisition["metadata_per_page"])
    maximum = int(acquisition["maximum_metadata_results_per_stratum"])
    cursor = "*"
    results: list[dict[str, Any]] = []
    while len(results) < maximum:
        query = {
            "search": stratum["search"],
            "filter": (
                f"from_publication_date:{stratum['from_publication_date']},"
                f"to_publication_date:{stratum['to_publication_date']},"
                "is_oa:true,has_fulltext:true,type:article"
            ),
            "per-page": min(per_page, maximum - len(results)),
            "cursor": cursor,
        }
        url = OPENALEX_API + "?" + urllib.parse.urlencode(query)
        payload = _http_json(
            url,
            timeout=float(acquisition["metadata_timeout_seconds"]),
            user_agent=str(acquisition["user_agent"]),
        )
        page = payload.get("results")
        if not isinstance(page, list):
            raise OaFeasibilityError("OpenAlex results are malformed")
        results.extend(row for row in page if isinstance(row, dict))
        next_cursor = (payload.get("meta") or {}).get("next_cursor")
        if not page or not isinstance(next_cursor, str) or not next_cursor:
            break
        cursor = next_cursor
    return results[:maximum]


def _normalize_pdf_text(raw: str) -> str:
    cleaned = "".join(
        character if character in "\n\t" or ord(character) >= 32 else " "
        for character in raw
    )
    return re.sub(r"\s+", " ", cleaned).strip()[:MAXIMUM_EXTRACTED_CHARACTERS]


def _extract_pdf(path: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="semantic-oa-extract-") as folder:
        output = Path(folder) / "text.txt"
        try:
            completed = subprocess.run(
                [
                    str(PDFTOTEXT_PATH),
                    "-f",
                    "1",
                    "-l",
                    str(PDF_PAGES),
                    "-nopgbrk",
                    str(path),
                    str(output),
                ],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        if completed.returncode != 0 or not output.is_file():
            return ""
        return _normalize_pdf_text(
            output.read_text(encoding="utf-8", errors="replace")
        )


def _download_task_identity(candidate: Mapping[str, Any]) -> str:
    sampling_hash = _require_sha256(
        candidate.get("sampling_hash"), "OA candidate sampling hash"
    )
    return stable_hash(
        {
            "stratum": candidate.get("stratum"),
            "openalex_id": candidate.get("openalex_id"),
            "sampling_hash": sampling_hash,
        }
    )


def _download_one(
    candidate: Mapping[str, Any],
    *,
    staging_root: Path,
    timeout: float,
    user_agent: str,
) -> dict[str, Any]:
    keep_temporary = False
    try:
        safe_initial_url = _safe_public_https_url(str(candidate["pdf_url"]))
    except OaFeasibilityError:
        return {"valid": False, "error_type": "unsafe_source_url"}
    request = urllib.request.Request(
        safe_initial_url,
        headers={"User-Agent": user_agent, "Accept": "application/pdf"},
    )
    try:
        opener = urllib.request.build_opener(_SafeRedirectHandler())
        with opener.open(request, timeout=timeout) as response:
            final_url = _safe_public_https_url(str(response.geturl()))
            content_type = str(response.headers.get("Content-Type") or "").casefold()
            raw = response.read(MAXIMUM_PDF_BYTES + 1)
    except (
        OSError,
        ValueError,
        http.client.HTTPException,
        urllib.error.URLError,
        OaFeasibilityError,
    ):
        return {"valid": False, "error_type": "transport_unavailable"}
    if len(raw) > MAXIMUM_PDF_BYTES:
        return {"valid": False, "error_type": "pdf_too_large"}
    if not raw.startswith(b"%PDF-") or (
        content_type and "pdf" not in content_type and "octet-stream" not in content_type
    ):
        return {"valid": False, "error_type": "not_pdf"}
    digest = _sha256_bytes(raw)
    task_identity = _download_task_identity(candidate)
    temporary = staging_root / f"{digest}.{task_identity}.tmp.pdf"
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
        text = _extract_pdf(temporary)
        if not text:
            return {"valid": False, "error_type": "extraction_unavailable"}
        keep_temporary = True
        return {
            "valid": True,
            "candidate": dict(candidate),
            "temporary_path": str(temporary),
            "pdf_sha256": digest,
            "size_bytes": len(raw),
            "evidence_text_sha256": _sha256_bytes(text.encode("utf-8")),
            "final_url_hash": stable_hash({"source_url": final_url}),
        }
    except OSError:
        return {"valid": False, "error_type": "local_staging_failed"}
    finally:
        if not keep_temporary:
            try:
                temporary.unlink()
            except (FileNotFoundError, UnboundLocalError):
                pass


def _known_train_arxiv_ids(train_pack: Mapping[str, Any]) -> set[str]:
    records = train_pack.get("records")
    if not isinstance(records, list):
        raise OaFeasibilityError("TRAIN pack records are missing")
    values: set[str] = set()
    for row in records:
        if not isinstance(row, dict):
            continue
        filename = str(row.get("filename") or "")
        match = _ARXIV_ID.search(filename)
        if match:
            values.add(match.group(1).casefold())
    return values


@_with_acquisition_guard
def acquire_oa_pack(
    *,
    preregistration_path: str | Path,
    train_pack_path: str | Path,
    operator_asset_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    preregistration = load_preregistration(preregistration_path)
    verify_extraction_runtime(preregistration)
    train_pack = _read_json(train_pack_path)
    train_pack_hash = _verify_self_hash(
        train_pack, hash_field="manifest_hash", label="TRAIN pack"
    )
    operator_asset = load_operator_asset(operator_asset_path)
    freeze = preregistration["operator_freeze"]
    if (
        operator_asset["candidate_id"] != freeze["candidate_id"]
        or train_pack_hash != freeze["consumed_train_pack_manifest_hash"]
        or train_pack.get("records_hash")
        != freeze["consumed_train_records_hash"]
        or operator_asset.get("train_pack_manifest_hash") != train_pack_hash
        or operator_asset.get("train_records_hash") != train_pack.get(
            "records_hash"
        )
    ):
        raise OaFeasibilityError("frozen candidate identity mismatch")
    if operator_asset["manifest_hash"] != freeze["operator_asset_manifest_hash"]:
        raise OaFeasibilityError("frozen operator asset mismatch")
    prereg_root = Path(preregistration_path).expanduser().resolve(strict=True).parents[1]
    root = Path(output_root).expanduser().resolve()
    expected_root = (prereg_root / preregistration["formal_paths"]["pack_root"]).resolve()
    if root != expected_root:
        raise OaFeasibilityError("OA pack output path is not preregistered")
    lock_path = root / "pack.lock.json"
    if lock_path.exists():
        raise FileExistsError("OA pack is already locked")
    staging = root / ".staging"
    pdf_root = root / "pdfs"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    pdf_root.mkdir(parents=True, exist_ok=True)
    acquisition = preregistration["acquisition"]
    strata = acquisition["strata"]
    known_ids = _known_train_arxiv_ids(train_pack)
    forbidden_train_pdf_hashes = {
        _require_sha256(row.get("content_sha256"), "TRAIN PDF hash")
        for row in train_pack["records"]
        if isinstance(row, dict)
    }
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(acquisition["metadata_workers"])
    ) as executor:
        futures = {
            row["stratum"]: executor.submit(
                fetch_stratum_metadata, row, acquisition=acquisition
            )
            for row in strata
        }
        metadata_by_stratum = {
            key: future.result() for key, future in futures.items()
        }
    metadata_query_hashes = {
        row["stratum"]: stable_hash(
            {
                "api_base": OPENALEX_API,
                "search": row["search"],
                "from_publication_date": row["from_publication_date"],
                "to_publication_date": row["to_publication_date"],
                "metadata_per_page": acquisition["metadata_per_page"],
                "maximum_metadata_results": acquisition[
                    "maximum_metadata_results_per_stratum"
                ],
            }
        )
        for row in strata
    }
    metadata_result_set_hashes = {
        stratum: stable_hash(
            [
                {
                    "openalex_id_hash": stable_hash(
                        {"openalex_id": _canonical_openalex_id(work.get("id"))}
                    ),
                    "work_text_hash": stable_hash({"work_text": _work_text(work)}),
                    "topic_text_hash": stable_hash(
                        {"topic_text": _topic_text(work)}
                    ),
                    "best_pdf_url_hash": stable_hash(
                        {
                            "pdf_url": (
                                (work.get("best_oa_location") or {}).get(
                                    "pdf_url"
                                )
                                if isinstance(
                                    work.get("best_oa_location"), dict
                                )
                                else None
                            )
                        }
                    ),
                }
                for work in works
            ]
        )
        for stratum, works in metadata_by_stratum.items()
    }
    ranked_by_stratum = {
        row["stratum"]: rank_metadata_candidates(
            metadata_by_stratum[row["stratum"]],
            stratum=row,
            seed=str(acquisition["seed"]),
            known_train_arxiv_ids=known_ids,
        )[: int(acquisition["maximum_download_candidates_per_stratum"])]
        for row in strata
    }
    qualified_candidate_set_hashes = {
        stratum: stable_hash(
            [
                {
                    "openalex_id_hash": stable_hash(
                        {"openalex_id": row["openalex_id"]}
                    ),
                    "doi_hash": (
                        stable_hash({"doi": str(row["doi"]).casefold()})
                        if row.get("doi")
                        else None
                    ),
                    "pdf_url_hash": stable_hash(
                        {"pdf_url": row["pdf_url"]}
                    ),
                    "license_hash": stable_hash(
                        {"license": row["license"]}
                    ),
                    "sampling_hash": row["sampling_hash"],
                    "sampling_rank": row["sampling_rank"],
                }
                for row in ranked
            ]
        )
        for stratum, ranked in ranked_by_stratum.items()
    }
    tasks = [
        candidate
        for row in strata
        for candidate in ranked_by_stratum[row["stratum"]]
    ]
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(acquisition["download_workers"])
    ) as executor:
        futures = [
            executor.submit(
                _download_one,
                candidate,
                staging_root=staging,
                timeout=float(acquisition["download_timeout_seconds"]),
                user_agent=str(acquisition["user_agent"]),
            )
            for candidate in tasks
        ]
        downloaded = [future.result() for future in futures]
    attempt_ledger = [
        {
            "stratum": candidate["stratum"],
            "sampling_hash": candidate["sampling_hash"],
            "sampling_rank": candidate["sampling_rank"],
            "openalex_id_hash": stable_hash(
                {"openalex_id": candidate["openalex_id"]}
            ),
            "doi_hash": (
                stable_hash({"doi": _canonical_doi(candidate.get("doi"))})
                if _canonical_doi(candidate.get("doi"))
                else None
            ),
            "transport_valid": result.get("valid") is True,
            "error_type": result.get("error_type"),
            "pdf_sha256": result.get("pdf_sha256"),
            "final_url_hash": result.get("final_url_hash"),
        }
        for candidate, result in zip(tasks, downloaded)
    ]
    by_key = {
        (
            row.get("candidate", {}).get("stratum"),
            row.get("candidate", {}).get("openalex_id"),
        ): row
        for row in downloaded
        if row.get("valid") is True
    }
    selected: list[dict[str, Any]] = []
    seen_openalex: set[str] = set()
    seen_doi: set[str] = set()
    seen_pdf: set[str] = set()
    transport_failures: dict[str, int] = {}
    selection_ledger: list[dict[str, Any]] = []
    required = int(acquisition["records_per_stratum"])
    try:
        for stratum in strata:
            count = 0
            for candidate in ranked_by_stratum[stratum["stratum"]]:
                disposition = {
                    "stratum": candidate["stratum"],
                    "sampling_hash": candidate["sampling_hash"],
                    "sampling_rank": candidate["sampling_rank"],
                    "openalex_id_hash": stable_hash(
                        {"openalex_id": candidate["openalex_id"]}
                    ),
                    "doi_hash": (
                        stable_hash(
                            {"doi": _canonical_doi(candidate.get("doi"))}
                        )
                        if _canonical_doi(candidate.get("doi"))
                        else None
                    ),
                }
                if count >= required:
                    selection_ledger.append(
                        {**disposition, "disposition": "quota_after_10"}
                    )
                    continue
                result = by_key.get((stratum["stratum"], candidate["openalex_id"]))
                if result is None:
                    transport_failures[stratum["stratum"]] = (
                        transport_failures.get(stratum["stratum"], 0) + 1
                    )
                    selection_ledger.append(
                        {**disposition, "disposition": "transport_failure"}
                    )
                    continue
                if result["pdf_sha256"] in forbidden_train_pdf_hashes:
                    selection_ledger.append(
                        {**disposition, "disposition": "train_content_overlap"}
                    )
                    continue
                doi_key = _canonical_doi(candidate.get("doi")) or ""
                if candidate["openalex_id"] in seen_openalex:
                    selection_ledger.append(
                        {**disposition, "disposition": "dedupe_openalex"}
                    )
                    continue
                if doi_key and doi_key in seen_doi:
                    selection_ledger.append(
                        {**disposition, "disposition": "dedupe_doi"}
                    )
                    continue
                if result["pdf_sha256"] in seen_pdf:
                    selection_ledger.append(
                        {**disposition, "disposition": "dedupe_pdf"}
                    )
                    continue
                source = Path(result["temporary_path"])
                destination = pdf_root / f"{result['pdf_sha256']}.pdf"
                if destination.exists():
                    if _sha256_file(destination) != result["pdf_sha256"]:
                        raise OaFeasibilityError("existing OA PDF content drifted")
                    source.unlink(missing_ok=True)
                else:
                    os.replace(source, destination)
                seen_openalex.add(candidate["openalex_id"])
                if doi_key:
                    seen_doi.add(doi_key)
                seen_pdf.add(result["pdf_sha256"])
                selected.append(
                    {
                        "record_id": stable_hash(
                            {
                                "seed": acquisition["seed"],
                                "openalex_id": candidate["openalex_id"],
                            }
                        ),
                        "openalex_id_hash": stable_hash(
                            {"openalex_id": candidate["openalex_id"]}
                        ),
                        "doi_hash": (
                            stable_hash({"doi": doi_key}) if doi_key else None
                        ),
                        "license_hash": stable_hash(
                            {"license": candidate["license"]}
                        ),
                        "source_url_hash": result["final_url_hash"],
                        "stratum": candidate["stratum"],
                        "expected_destination": candidate[
                            "expected_destination"
                        ],
                        "sampling_rank": candidate["sampling_rank"],
                        "pdf_relative_path": f"pdfs/{result['pdf_sha256']}.pdf",
                        "pdf_sha256": result["pdf_sha256"],
                        "size_bytes": result["size_bytes"],
                        "evidence_text_sha256": result[
                            "evidence_text_sha256"
                        ],
                    }
                )
                selection_ledger.append(
                    {**disposition, "disposition": "selected"}
                )
                count += 1
            if count != required:
                raise OaFeasibilityError(
                    f"OA acquisition insufficient:{stratum['stratum']}"
                )
    finally:
        for row in downloaded:
            temporary = row.get("temporary_path")
            if temporary:
                Path(str(temporary)).unlink(missing_ok=True)
    selected.sort(key=lambda row: (EXPECTED_STRATA.index(row["stratum"]), row["sampling_rank"]))
    if len(selected) != len(EXPECTED_STRATA) * required:
        raise OaFeasibilityError("OA pack size mismatch")
    pack: dict[str, Any] = {
        "pack_version": OA_PACK_VERSION,
        "preregistration_manifest_hash": preregistration["manifest_hash"],
        "candidate_id": operator_asset["candidate_id"],
        "operator_asset_manifest_hash": operator_asset["manifest_hash"],
        "seed": acquisition["seed"],
        "record_count": len(selected),
        "records_per_stratum": required,
        "strata": list(EXPECTED_STRATA),
        "records": selected,
        "records_hash": stable_hash(selected),
        "selection_hash": stable_hash(selected),
        "pdf_content_set_hash": stable_hash(sorted(seen_pdf)),
        "metadata_query_hashes": metadata_query_hashes,
        "metadata_result_set_hashes": metadata_result_set_hashes,
        "qualified_candidate_set_hashes": qualified_candidate_set_hashes,
        "download_attempt_count": len(attempt_ledger),
        "download_attempt_ledger": attempt_ledger,
        "download_attempt_ledger_hash": stable_hash(attempt_ledger),
        "selection_ledger": selection_ledger,
        "selection_ledger_hash": stable_hash(selection_ledger),
        "selection_disposition_counts": dict(
            sorted(Counter(row["disposition"] for row in selection_ledger).items())
        ),
        "transport_failure_counts": dict(sorted(transport_failures.items())),
        "prediction_started": False,
        "semantic_outcome_used_for_selection": False,
        "operator_created_extracted_text_artifact": False,
        "raw_title_abstract_or_text_persisted": False,
        "acquisition_online_calls_only": True,
    }
    pack["pack_hash"] = stable_hash(pack)
    _write_json_exclusive(lock_path, pack)
    try:
        staging.rmdir()
    except OSError:
        pass
    return pack


def verify_locked_pack(
    pack: Mapping[str, Any],
    *,
    pack_root: str | Path,
    forbidden_train_pdf_hashes: set[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    if pack.get("pack_version") != OA_PACK_VERSION:
        raise OaFeasibilityError("OA pack version mismatch")
    _verify_self_hash(pack, hash_field="pack_hash", label="OA pack")
    if (
        pack.get("record_count") != 60
        or pack.get("records_per_stratum") != 10
        or tuple(pack.get("strata") or ()) != EXPECTED_STRATA
        or pack.get("prediction_started") is not False
        or pack.get("semantic_outcome_used_for_selection") is not False
    ):
        raise OaFeasibilityError("OA pack design boundary mismatch")
    records = pack.get("records")
    if not isinstance(records, list) or len(records) != pack.get("record_count"):
        raise OaFeasibilityError("OA pack records are malformed")
    if stable_hash(records) != pack.get("records_hash"):
        raise OaFeasibilityError("OA pack record hash mismatch")
    attempts = pack.get("download_attempt_ledger")
    selections = pack.get("selection_ledger")
    if (
        not isinstance(attempts, list)
        or len(attempts) != pack.get("download_attempt_count")
        or stable_hash(attempts) != pack.get("download_attempt_ledger_hash")
        or not isinstance(selections, list)
        or stable_hash(selections) != pack.get("selection_ledger_hash")
    ):
        raise OaFeasibilityError("OA acquisition ledger is malformed")
    dispositions = {
        "selected",
        "transport_failure",
        "train_content_overlap",
        "dedupe_openalex",
        "dedupe_doi",
        "dedupe_pdf",
        "quota_after_10",
    }
    if len(attempts) != len(selections):
        raise OaFeasibilityError("OA acquisition ledgers have unequal length")
    for attempt, selection in zip(attempts, selections):
        if not isinstance(attempt, dict) or not isinstance(selection, dict):
            raise OaFeasibilityError("OA acquisition ledger row is malformed")
        identity = (
            "stratum",
            "sampling_hash",
            "sampling_rank",
            "openalex_id_hash",
            "doi_hash",
        )
        if any(attempt.get(key) != selection.get(key) for key in identity) or (
            selection.get("disposition") not in dispositions
        ):
            raise OaFeasibilityError("OA acquisition ledger identity drifted")
        rank = attempt.get("sampling_rank")
        if (
            attempt.get("stratum") not in EXPECTED_STRATA
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 0
        ):
            raise OaFeasibilityError("OA acquisition ledger rank is malformed")
        _require_sha256(attempt.get("sampling_hash"), "OA sampling hash")
        _require_sha256(attempt.get("openalex_id_hash"), "OA ledger OpenAlex hash")
        doi_hash = attempt.get("doi_hash")
        if doi_hash is not None:
            _require_sha256(doi_hash, "OA ledger DOI hash")
        if attempt.get("transport_valid") is True:
            if attempt.get("error_type") is not None:
                raise OaFeasibilityError("valid OA transport has an error")
            _require_sha256(attempt.get("pdf_sha256"), "OA attempt PDF hash")
            _require_sha256(
                attempt.get("final_url_hash"), "OA attempt final URL hash"
            )
        elif attempt.get("transport_valid") is False:
            if (
                not isinstance(attempt.get("error_type"), str)
                or not attempt["error_type"]
                or attempt.get("pdf_sha256") is not None
                or attempt.get("final_url_hash") is not None
            ):
                raise OaFeasibilityError("failed OA transport row is malformed")
        else:
            raise OaFeasibilityError("OA transport validity is malformed")
    for stratum in EXPECTED_STRATA:
        ranks = [
            int(row["sampling_rank"])
            for row in selections
            if row.get("stratum") == stratum
        ]
        if ranks != list(range(len(ranks))) or len(ranks) > 40:
            raise OaFeasibilityError("OA candidate ranks are not canonical")
    disposition_counts = dict(
        sorted(Counter(row["disposition"] for row in selections).items())
    )
    if disposition_counts != pack.get("selection_disposition_counts"):
        raise OaFeasibilityError("OA selection disposition counts drifted")
    if pack.get("selection_hash") != pack.get("records_hash"):
        raise OaFeasibilityError("OA selection hash does not bind records")
    for field in (
        "metadata_query_hashes",
        "metadata_result_set_hashes",
        "qualified_candidate_set_hashes",
    ):
        values = pack.get(field)
        if (
            not isinstance(values, dict)
            or set(values) != set(EXPECTED_STRATA)
            or any(
                not isinstance(value, str) or not _SHA256.fullmatch(value)
                for value in values.values()
            )
        ):
            raise OaFeasibilityError("OA acquisition hash ledger is malformed")
    root = Path(pack_root).expanduser().resolve(strict=True)
    texts: list[str] = []
    normalized: list[dict[str, Any]] = []
    content_hashes: list[str] = []
    seen_record_ids: set[str] = set()
    seen_openalex_hashes: set[str] = set()
    seen_doi_hashes: set[str] = set()
    expected_by_stratum = {
        "LLM": "LLM",
        "trapped_ion_and_qc": "trapped_ion_and_qc",
        "black_hole": "black_hole",
        "DNA": "DNA",
        "music_history": "music_history",
        "unrelated_public_default": "music_history",
    }
    previous_order: tuple[int, int] | None = None
    for row in records:
        if not isinstance(row, dict):
            raise OaFeasibilityError("OA record is malformed")
        relative = Path(str(row.get("pdf_relative_path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise OaFeasibilityError("OA PDF path is unsafe")
        path = (root / relative).resolve(strict=True)
        if root not in path.parents:
            raise OaFeasibilityError("OA PDF escaped pack root")
        digest = _require_sha256(row.get("pdf_sha256"), "OA PDF hash")
        record_id = _require_sha256(row.get("record_id"), "OA record id")
        openalex_hash = _require_sha256(
            row.get("openalex_id_hash"), "OA OpenAlex id hash"
        )
        doi_hash = row.get("doi_hash")
        if doi_hash is not None:
            _require_sha256(doi_hash, "OA DOI hash")
        _require_sha256(row.get("license_hash"), "OA license hash")
        _require_sha256(row.get("source_url_hash"), "OA source URL hash")
        stratum = row.get("stratum")
        rank = row.get("sampling_rank")
        if (
            stratum not in expected_by_stratum
            or row.get("expected_destination") != expected_by_stratum[stratum]
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 0
        ):
            raise OaFeasibilityError("OA stratum commitment is invalid")
        order = (EXPECTED_STRATA.index(stratum), rank)
        if previous_order is not None and order <= previous_order:
            raise OaFeasibilityError("OA records are not canonically ordered")
        previous_order = order
        if (
            record_id in seen_record_ids
            or openalex_hash in seen_openalex_hashes
            or digest in content_hashes
            or (doi_hash is not None and doi_hash in seen_doi_hashes)
        ):
            raise OaFeasibilityError("OA record identity is duplicated")
        seen_record_ids.add(record_id)
        seen_openalex_hashes.add(openalex_hash)
        if doi_hash is not None:
            seen_doi_hashes.add(doi_hash)
        if path.stat().st_size != row.get("size_bytes") or _sha256_file(path) != digest:
            raise OaFeasibilityError("OA PDF content drifted")
        text = _extract_pdf(path)
        if not text or _sha256_bytes(text.encode("utf-8")) != row.get(
            "evidence_text_sha256"
        ):
            raise OaFeasibilityError("OA evidence extraction drifted")
        if row.get("expected_destination") not in ALL_DESTINATIONS:
            raise OaFeasibilityError("OA gold destination is invalid")
        normalized.append(dict(row))
        texts.append(text)
        content_hashes.append(digest)
    if stable_hash(sorted(content_hashes)) != pack.get("pdf_content_set_hash"):
        raise OaFeasibilityError("OA PDF content set drifted")
    if Counter(row["stratum"] for row in normalized) != Counter(
        {stratum: 10 for stratum in EXPECTED_STRATA}
    ):
        raise OaFeasibilityError("OA stratum counts drifted")
    if any(
        not isinstance(value, str) or not _SHA256.fullmatch(value)
        for value in forbidden_train_pdf_hashes
    ):
        raise OaFeasibilityError("TRAIN PDF exclusion set is malformed")
    records_by_pair = {
        (row["stratum"], row["sampling_rank"]): row for row in normalized
    }
    seen_openalex: set[str] = set()
    seen_doi: set[str] = set()
    seen_pdf: set[str] = set()
    selected_counts: Counter[str] = Counter()
    replayed_transport_failures: Counter[str] = Counter()
    replayed_selected_pairs: set[tuple[str, int]] = set()
    for attempt, selection in zip(attempts, selections):
        stratum = str(attempt["stratum"])
        pair = (stratum, int(attempt["sampling_rank"]))
        transport_valid = attempt["transport_valid"] is True
        pdf_sha256 = attempt.get("pdf_sha256")
        openalex_hash = str(attempt["openalex_id_hash"])
        doi_hash = attempt.get("doi_hash")
        if selected_counts[stratum] >= 10:
            expected_disposition = "quota_after_10"
        elif not transport_valid:
            expected_disposition = "transport_failure"
            replayed_transport_failures[stratum] += 1
        elif pdf_sha256 in forbidden_train_pdf_hashes:
            expected_disposition = "train_content_overlap"
        elif openalex_hash in seen_openalex:
            expected_disposition = "dedupe_openalex"
        elif doi_hash is not None and doi_hash in seen_doi:
            expected_disposition = "dedupe_doi"
        elif pdf_sha256 in seen_pdf:
            expected_disposition = "dedupe_pdf"
        else:
            expected_disposition = "selected"
        if selection["disposition"] != expected_disposition:
            raise OaFeasibilityError("OA selection ledger cannot be replayed")
        if expected_disposition != "selected":
            if pair in records_by_pair:
                raise OaFeasibilityError("unselected OA attempt has a record")
            continue
        record = records_by_pair.get(pair)
        if record is None or (
            record["openalex_id_hash"] != openalex_hash
            or record.get("doi_hash") != doi_hash
            or record["pdf_sha256"] != pdf_sha256
            or record["source_url_hash"] != attempt["final_url_hash"]
        ):
            raise OaFeasibilityError("selected OA attempt does not bind its record")
        replayed_selected_pairs.add(pair)
        selected_counts[stratum] += 1
        seen_openalex.add(openalex_hash)
        if doi_hash is not None:
            seen_doi.add(doi_hash)
        seen_pdf.add(str(pdf_sha256))
    if replayed_selected_pairs != set(records_by_pair) or len(
        replayed_selected_pairs
    ) != 60:
        raise OaFeasibilityError("OA selection ledger does not bind records")
    if dict(sorted(replayed_transport_failures.items())) != pack.get(
        "transport_failure_counts"
    ):
        raise OaFeasibilityError("OA transport failure counts drifted")
    return normalized, texts


def _network_isolation_probe() -> bool:
    try:
        rows = Path("/proc/net/dev").read_text(encoding="utf-8").splitlines()[2:]
        interfaces = {
            row.split(":", 1)[0].strip() for row in rows if ":" in row
        }
    except OSError:
        return False
    if interfaces != {"lo"}:
        return False
    probes = (("1.1.1.1", 443), ("8.8.8.8", 53))
    for host, port in probes:
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return False
        except OSError:
            continue
    return True


def _reserve_decision_lock(
    path: Path, *, preregistration_hash: str, pack_hash: str, candidate_id: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lock_version": OA_DECISION_LOCK_VERSION,
        "decision_ordinal": 1,
        "state": "reserved",
        "preregistration_manifest_hash": preregistration_hash,
        "pack_hash": pack_hash,
        "candidate_id": candidate_id,
    }
    payload["lock_hash"] = stable_hash(payload)
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise FileExistsError("OA decision budget is already consumed") from error


def _oa_evidence_payload(
    records: Sequence[Mapping[str, Any]], texts: Sequence[str]
) -> dict[str, Any]:
    contract_hash = stable_hash(
        {
            "policy": OA_FEASIBILITY_VERSION,
            "record_ids": [row["record_id"] for row in records],
        }
    )
    files: list[dict[str, Any]] = []
    for row, text in zip(records, texts):
        text_hash = _sha256_bytes(text.encode("utf-8"))
        file_id = stable_hash(
            {"contract_hash": contract_hash, "record_id": row["record_id"]}
        )
        evidence_id = stable_hash(
            {
                "contract_hash": contract_hash,
                "file_id": file_id,
                "text_sha256": text_hash,
            }
        )
        files.append(
            {
                "file_id": file_id,
                "filename": f"{row['record_id']}.pdf",
                "content_sha256": row["pdf_sha256"],
                "size_bytes": row["size_bytes"],
                "media_type": "pdf",
                "extraction_status": "ok",
                "evidence": [
                    {
                        "evidence_id": evidence_id,
                        "kind": "pdf_first_pages_text",
                        "text": text,
                        "text_sha256": text_hash,
                        "truncated": len(text) == MAXIMUM_EXTRACTED_CHARACTERS,
                    }
                ],
            }
        )
    body: dict[str, Any] = {
        "runtime_policy": "typed_assignment_prepare_plan_apply_reconcile_v3",
        "contract_hash": contract_hash,
        "destinations": list(ALL_DESTINATIONS),
        "public_default": PUBLIC_DEFAULT_DESTINATION,
        "extraction_policy": {
            "pdf_pages": PDF_PAGES,
            "maximum_characters": MAXIMUM_EXTRACTED_CHARACTERS,
        },
        "files": files,
    }
    body["evidence_set_hash"] = _payload_hash(body)
    return body


def evaluate_oa_pack(
    *,
    preregistration_path: str | Path,
    pack_root: str | Path,
    operator_asset_path: str | Path,
    runtime_asset_path: str | Path,
    snapshot_root: str | Path,
    report_path: str | Path,
    decision_lock_path: str | Path,
) -> dict[str, Any]:
    preregistration = load_preregistration(preregistration_path)
    verify_extraction_runtime(preregistration)
    prereg_root = Path(preregistration_path).expanduser().resolve(strict=True).parents[1]
    freeze = preregistration["operator_freeze"]
    train_pack_path = (
        prereg_root / str(freeze["consumed_train_pack_path"])
    ).resolve(strict=True)
    train_pack = _read_json(train_pack_path)
    train_pack_hash = _verify_self_hash(
        train_pack, hash_field="manifest_hash", label="TRAIN pack"
    )
    if (
        train_pack_hash != freeze["consumed_train_pack_manifest_hash"]
        or train_pack.get("records_hash") != freeze["consumed_train_records_hash"]
    ):
        raise OaFeasibilityError("OA TRAIN exclusion identity mismatch")
    forbidden_train_pdf_hashes = {
        _require_sha256(row.get("content_sha256"), "TRAIN PDF hash")
        for row in train_pack.get("records", [])
        if isinstance(row, dict)
    }
    if len(forbidden_train_pdf_hashes) != freeze["consumed_train_record_count"]:
        raise OaFeasibilityError("OA TRAIN exclusion set is incomplete")
    root = Path(pack_root).expanduser().resolve(strict=True)
    pack = _read_json(root / "pack.lock.json")
    _verify_self_hash(pack, hash_field="pack_hash", label="OA pack")
    asset = load_operator_asset(operator_asset_path)
    runtime_asset = _read_json(runtime_asset_path)
    runtime_asset_hash = _verify_self_hash(
        runtime_asset, hash_field="manifest_hash", label="runtime asset"
    )
    if (
        asset["candidate_id"] != freeze["candidate_id"]
        or asset["manifest_hash"] != freeze["operator_asset_manifest_hash"]
        or pack.get("candidate_id") != asset["candidate_id"]
        or pack.get("preregistration_manifest_hash")
        != preregistration["manifest_hash"]
        or pack.get("operator_asset_manifest_hash") != asset["manifest_hash"]
        or runtime_asset_hash != freeze["runtime_asset_manifest_hash"]
        or runtime_asset.get("runtime_required_file_set_hash")
        != freeze["runtime_required_file_set_hash"]
        or runtime_asset.get("snapshot_revision") != freeze["snapshot_revision"]
        or runtime_asset.get("weights_sha256") != freeze["weights_sha256"]
        or asset.get("runtime_asset_manifest_hash") != runtime_asset_hash
        or asset.get("runtime_required_file_set_hash")
        != runtime_asset.get("runtime_required_file_set_hash")
    ):
        raise OaFeasibilityError("OA evaluation candidate identity mismatch")
    report = Path(report_path).expanduser().resolve()
    lock = Path(decision_lock_path).expanduser().resolve()
    expected_paths = preregistration["formal_paths"]
    if root != (prereg_root / expected_paths["pack_root"]).resolve():
        raise OaFeasibilityError("OA pack input path is not preregistered")
    if report != (prereg_root / expected_paths["report"]).resolve() or lock != (
        prereg_root / expected_paths["decision_lock"]
    ).resolve():
        raise OaFeasibilityError("OA formal output path is not preregistered")
    if report.exists():
        raise FileExistsError("OA formal report already exists")
    if os.environ.get("SEMANTIC_ASSIGNMENT_NETWORK_ISOLATION") != (
        NETWORK_ISOLATION_POLICY
    ) or not _network_isolation_probe():
        raise OaFeasibilityError("OA evaluation network isolation is absent")
    _reserve_decision_lock(
        lock,
        preregistration_hash=preregistration["manifest_hash"],
        pack_hash=pack["pack_hash"],
        candidate_id=asset["candidate_id"],
    )
    records, texts = verify_locked_pack(
        pack,
        pack_root=root,
        forbidden_train_pdf_hashes=forbidden_train_pdf_hashes,
    )
    model = OfflineMiniLMEncoder(
        runtime_asset_path=runtime_asset_path,
        snapshot_root=snapshot_root,
    )
    evidence = _oa_evidence_payload(records, texts)
    plan, operator_receipt = build_semantic_assignment_plan(
        evidence_payload=evidence,
        operator_asset=asset,
        encoder=model,
        runtime_receipt=model.runtime_receipt,
    )
    predicted_by_file = {
        row["file_id"]: row["destination"] for row in plan["assignments"]
    }
    evidence_file_ids = [row["file_id"] for row in evidence["files"]]
    outcomes: list[dict[str, Any]] = []
    correct = 0
    by_stratum: dict[str, dict[str, int]] = {}
    for record, file_id in zip(records, evidence_file_ids):
        predicted = predicted_by_file[file_id]
        expected = record["expected_destination"]
        matched = predicted == expected
        correct += int(matched)
        aggregate = by_stratum.setdefault(record["stratum"], {"correct": 0, "total": 0})
        aggregate["correct"] += int(matched)
        aggregate["total"] += 1
        outcomes.append(
            {
                "record_id": record["record_id"],
                "stratum": record["stratum"],
                "expected_destination": expected,
                "predicted_destination": predicted,
                "correct": matched,
            }
        )
    required = int(preregistration["evaluation"]["required_correct"])
    passed = len(records) == required and len(texts) == required and correct == required
    result: dict[str, Any] = {
        "report_version": OA_REPORT_VERSION,
        "decision_ordinal": 1,
        "preregistration_manifest_hash": preregistration["manifest_hash"],
        "pack_hash": pack["pack_hash"],
        "candidate_id": asset["candidate_id"],
        "operator_asset_manifest_hash": asset["manifest_hash"],
        "record_count": len(records),
        "evidence_valid_count": len(texts),
        "correct_count": correct,
        "required_correct": required,
        "feasibility_passed": passed,
        "by_stratum": by_stratum,
        "outcomes": outcomes,
        "outcomes_hash": stable_hash(outcomes),
        "operator_receipt": operator_receipt,
        "network_namespace_policy": NETWORK_ISOLATION_POLICY,
        "network_probe_blocked": True,
        "offline_evaluation_only": True,
        "ruoli_calls": 0,
        "online_judge_calls": 0,
        "agent_calls": 0,
        "hipporag_calls": 0,
        "raw_benchmark_calls": 0,
        "valid_failure_retry_authorized": False,
        "same_pack_post_failure_tuning_authorized": False,
        "operator_created_extracted_text_artifact": False,
        "operator_logged_raw_text": False,
        "incumbent_authorized": False,
        "promotion_authorized": False,
        "sealed_split_accessed": False,
    }
    result["decision_hash"] = stable_hash(
        {
            "preregistration_manifest_hash": result[
                "preregistration_manifest_hash"
            ],
            "pack_hash": result["pack_hash"],
            "candidate_id": result["candidate_id"],
            "outcomes_hash": result["outcomes_hash"],
            "feasibility_passed": passed,
        }
    )
    result["report_hash"] = stable_hash(result)
    _atomic_write_json(report, result)
    completed_lock = {
        "lock_version": OA_DECISION_LOCK_VERSION,
        "decision_ordinal": 1,
        "state": "completed",
        "preregistration_manifest_hash": preregistration["manifest_hash"],
        "pack_hash": pack["pack_hash"],
        "candidate_id": asset["candidate_id"],
        "decision_hash": result["decision_hash"],
        "report_hash": result["report_hash"],
        "feasibility_passed": passed,
    }
    completed_lock["lock_hash"] = stable_hash(completed_lock)
    _atomic_write_json(lock, completed_lock)
    return result


def _command_acquire(args: argparse.Namespace) -> int:
    acquire_oa_pack(
        preregistration_path=args.preregistration,
        train_pack_path=args.train_pack,
        operator_asset_path=args.operator_asset,
        output_root=args.output_root,
    )
    return 0


def _command_evaluate(args: argparse.Namespace) -> int:
    evaluate_oa_pack(
        preregistration_path=args.preregistration,
        pack_root=args.pack_root,
        operator_asset_path=args.operator_asset,
        runtime_asset_path=args.runtime_asset,
        snapshot_root=args.snapshot_root,
        report_path=args.report,
        decision_lock_path=args.decision_lock,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    acquire = subparsers.add_parser("acquire")
    acquire.add_argument("--preregistration", required=True)
    acquire.add_argument("--train-pack", required=True)
    acquire.add_argument("--operator-asset", required=True)
    acquire.add_argument("--output-root", required=True)
    acquire.set_defaults(function=_command_acquire)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--preregistration", required=True)
    evaluate.add_argument("--pack-root", required=True)
    evaluate.add_argument("--operator-asset", required=True)
    evaluate.add_argument("--runtime-asset", required=True)
    evaluate.add_argument("--snapshot-root", required=True)
    evaluate.add_argument("--report", required=True)
    evaluate.add_argument("--decision-lock", required=True)
    evaluate.set_defaults(function=_command_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
