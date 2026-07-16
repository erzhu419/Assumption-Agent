from __future__ import annotations

import json
from pathlib import Path
import stat
import subprocess
import tempfile
from typing import Any, Iterator
import zipfile

import pytest

from assumption_agent.benchmarks import musique_evaluator_portfolio_acquisition_v1 as study
from assumption_agent.benchmarks.musique_official_core_comparison_v1 import (
    _selection_secret_commitment,
    generate_selection_secret,
)
from assumption_agent.models import stable_hash


@pytest.fixture
def private_tmp_path() -> Iterator[Path]:
    parent = Path(__file__).resolve().parents[1] / "artifacts"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="musique-portfolio-acquisition-", dir=parent) as root:
        yield Path(root)


def _row(index: int) -> dict[str, Any]:
    token = f"signal{index:04d}"
    return {
        "id": f"official-dev-{index:04d}",
        "question": f"Which signal belongs to record {index}?",
        "answer": token,
        "answer_aliases": [f"the {token}", f"signal {index:04d}"],
        "answerable": True,
        "paragraphs": [
            {
                "idx": 0,
                "title": "root",
                "paragraph_text": f"Root evidence for {index}.",
                "is_supporting": True,
            },
            {
                "idx": 1,
                "title": "leaf",
                "paragraph_text": f"Leaf evidence contains {token}.",
                "is_supporting": True,
            },
            *[
                {
                    "idx": position,
                    "title": f"distractor-{position}",
                    "paragraph_text": f"Distractor {position} for {index}.",
                    "is_supporting": False,
                }
                for position in range(2, 5)
            ],
        ],
    }


def _git(project: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(project), *arguments],
        check=True,
        capture_output=True,
    )


def _commit_all(project: Path, message: str) -> None:
    _git(project, "add", "-A")
    subprocess.run(
        [
            "git",
            "-C",
            str(project),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-q",
            "-m",
            message,
        ],
        check=True,
        capture_output=True,
    )


def _prepare(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Path]:
    project = root / "project"
    project.mkdir()
    _git(project, "init", "-q")
    (project / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
    (project / "impl.py").write_text("VERSION = 1\n", encoding="utf-8")
    artifacts = project / "artifacts"
    artifacts.mkdir()
    secret_path = artifacts / "selection.key"
    generate_selection_secret(project=project, output=secret_path)

    archive_path = artifacts / "musique.zip"
    source_text = "\n".join(json.dumps(_row(index)) for index in range(300)) + "\n"
    monkeypatch.setattr(
        study,
        "OFFICIAL_DEV_MEMBER_SHA256",
        study._sha256_bytes(source_text.encode("utf-8")),
    )
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            f"data/{study.OFFICIAL_DEV_MEMBER_BASENAME}",
            source_text,
        )

    manifests = project / "manifests"
    manifests.mkdir()
    design = {
        "schema": study.PORTFOLIO_DESIGN_SCHEMA,
        "status": "fixed_before_residual_selection",
        "raw_content_persisted": False,
        "cohort": {"start": 96, "stop": 264},
    }
    design["design_sha256"] = stable_hash(design)
    design_path = project / study.PORTFOLIO_DESIGN_RELATIVE
    design_path.write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    secret = secret_path.read_bytes()
    prior_implementation = {
        "files": [{"path": "impl.py", "sha256": study._sha256_file(project / "impl.py")}],
    }
    prior_implementation["set_sha256"] = stable_hash(prior_implementation["files"])
    monkeypatch.setattr(
        study.prior,
        "_implementation_binding",
        lambda _project: prior_implementation,
    )
    prior_prereg: dict[str, Any] = {
        "schema": study.prior.PREREGISTRATION_SCHEMA,
        "eligibility": {
            "normalizer": "musique_official_core_comparison_v1._normalize_source_row"
        },
        "implementation": prior_implementation,
        "selection": {
            "algorithm": "ascending_hmac_sha256_private_secret_and_official_item_id_v1",
            "selected_count": 96,
            "selection_secret_commitment_sha256": _selection_secret_commitment(secret),
        },
        "source": {
            "repository": "https://github.com/StonyBrookNLP/musique.git",
            "license": {"spdx": "CC-BY-4.0"},
            "official_archive_sha256": study._sha256_file(archive_path),
            "source_split": "official_dev",
        },
    }
    prior_prereg["preregistration_sha256"] = stable_hash(prior_prereg)
    prior_prereg_path = project / study.PRIOR_PREREGISTRATION_RELATIVE
    prior_prereg_path.write_text(
        json.dumps(prior_prereg, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    prior_receipt: dict[str, Any] = {
        "schema": study.prior.ACQUISITION_SCHEMA,
        "counts": {"selected_rows": 96, "eligible_rows": 300},
        "source": {
            "archive_sha256": study._sha256_file(archive_path),
            "source_split": "official_dev",
            "official_dev_member_sha256": study._sha256_bytes(
                source_text.encode("utf-8")
            ),
        },
        "commitments": {
            "selection_secret_commitment_sha256": _selection_secret_commitment(
                secret
            )
        },
        "ordering": {
            "preregistration_sha256": prior_prereg["preregistration_sha256"]
        },
    }
    prior_receipt["acquisition_sha256"] = stable_hash(prior_receipt)
    prior_path = project / study.PRIOR_ACQUISITION_RELATIVE
    prior_path.write_text(
        json.dumps(prior_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _commit_all(project, "initial")

    monkeypatch.setattr(study, "IMPLEMENTATION_RELATIVE_FILES", ("impl.py",))
    monkeypatch.setattr(study, "PRIOR_LINEAGE_RELATIVE_FILES", ())
    monkeypatch.setattr(study, "EXPECTED_ELIGIBLE_ROWS", 300)
    monkeypatch.setattr(
        study, "OFFICIAL_ARCHIVE_SHA256", study._sha256_file(archive_path)
    )
    monkeypatch.setattr(
        study, "PRIOR_ACQUISITION_FILE_SHA256", study._sha256_file(prior_path)
    )
    monkeypatch.setattr(
        study,
        "PRIOR_ACQUISITION_SHA256",
        prior_receipt["acquisition_sha256"],
    )
    monkeypatch.setattr(
        study,
        "PRIOR_PREREGISTRATION_FILE_SHA256",
        study._sha256_file(prior_prereg_path),
    )
    monkeypatch.setattr(
        study,
        "PRIOR_PREREGISTRATION_SHA256",
        prior_prereg["preregistration_sha256"],
    )
    monkeypatch.setattr(
        study,
        "official_source_receipt",
        lambda _path: {
            "repository": "https://github.com/StonyBrookNLP/musique.git",
            "commit": study.OFFICIAL_SOURCE_COMMIT,
            "license": {"spdx": "CC-BY-4.0"},
        },
    )
    official_repository = project / "official"
    official_repository.mkdir()
    return {
        "project": project,
        "artifacts": artifacts,
        "secret": secret_path,
        "archive": archive_path,
        "prior_prereg": prior_prereg_path,
        "prior": prior_path,
        "official": official_repository,
    }


def _preregister(paths: dict[str, Path]) -> Path:
    payload = study.build_preregistration(
        project=paths["project"],
        official_repository=paths["official"],
        selection_secret_path=paths["secret"],
        prior_preregistration_path=paths["prior_prereg"],
        prior_acquisition_receipt_path=paths["prior"],
    )
    output = paths["project"] / study.PREREGISTRATION_RELATIVE
    study._write_json_exclusive(
        output, payload, hash_field="preregistration_sha256", mode=0o644
    )
    _commit_all(paths["project"], "preregister")
    return output


def _acquire_and_commit(
    paths: dict[str, Path], prereg: Path
) -> tuple[Path, Path, Path, dict[str, Any]]:
    private_root = paths["artifacts"] / "portfolio-pack"
    locator = paths["artifacts"] / "portfolio-custody" / "locator.json"
    public_receipt = paths["project"] / study.ACQUISITION_RELATIVE
    receipt = study.acquire_private_blocks(
        project=paths["project"],
        preregistration_path=prereg,
        official_repository=paths["official"],
        selection_secret_path=paths["secret"],
        prior_preregistration_path=paths["prior_prereg"],
        prior_acquisition_receipt_path=paths["prior"],
        source_archive_path=paths["archive"],
        private_root=private_root,
        private_locator_path=locator,
        public_receipt_path=public_receipt,
    )
    study._write_json_exclusive(
        public_receipt, receipt, hash_field="acquisition_sha256", mode=0o644
    )
    _commit_all(paths["project"], "acquisition")
    return public_receipt, private_root, locator, receipt


def test_preregistration_is_zero_row_and_binds_exact_continuation(
    private_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prepare(private_tmp_path, monkeypatch)
    monkeypatch.setattr(
        study,
        "_iter_source_rows",
        lambda _raw: (_ for _ in ()).throw(AssertionError("source rows opened")),
    )
    monkeypatch.setattr(
        study.zipfile,
        "ZipFile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("source archive opened")
        ),
    )
    payload = study.build_preregistration(
        project=paths["project"],
        official_repository=paths["official"],
        selection_secret_path=paths["secret"],
        prior_preregistration_path=paths["prior_prereg"],
        prior_acquisition_receipt_path=paths["prior"],
    )
    assert payload["selection"]["previous_rank_window_stop_exclusive"] == 96
    assert payload["selection"]["rank_window_start_inclusive"] == 96
    assert payload["selection"]["rank_window_stop_exclusive"] == 264
    assert payload["selection"]["selected_count"] == 168
    assert payload["safety"]["dataset_rows_read"] == 0
    assert payload["prior_acquisition_binding"]["private_block_files_opened"] == 0


def test_acquisition_forms_exact_disjoint_continuation_and_strict_loaders(
    private_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prepare(private_tmp_path, monkeypatch)
    prereg = _preregister(paths)
    private_root = paths["artifacts"] / "portfolio-pack"
    locator = paths["artifacts"] / "portfolio-custody" / "locator.json"
    public_receipt = paths["project"] / "manifests" / "portfolio-acquisition.json"
    receipt = study.acquire_private_blocks(
        project=paths["project"],
        preregistration_path=prereg,
        official_repository=paths["official"],
        selection_secret_path=paths["secret"],
        prior_preregistration_path=paths["prior_prereg"],
        prior_acquisition_receipt_path=paths["prior"],
        source_archive_path=paths["archive"],
        private_root=private_root,
        private_locator_path=locator,
        public_receipt_path=public_receipt,
    )
    study._write_json_exclusive(
        public_receipt, receipt, hash_field="acquisition_sha256", mode=0o644
    )
    loaded_receipt, commitments = study.load_acquisition_binding(public_receipt)
    assert loaded_receipt["counts"]["selected_rows"] == 168
    assert loaded_receipt["counts"]["selected_previous_rank_window_overlap"] == 0
    assert [row.block for row in commitments] == list(study.BLOCK_ORDER)
    assert [row.count for row in commitments] == [24, 24, 24, 24, 48, 24]

    selected_ids: set[str] = set()
    for commitment in commitments:
        rows = study.load_private_block(
            private_root / f"{commitment.block}.jsonl",
            commitment=commitment,
            expected_block=commitment.block,
        )
        selected_ids.update(row["item_id"] for row in rows)
    assert len(selected_ids) == 168
    secret = paths["secret"].read_bytes()
    ordered_ids = sorted(
        (f"official-dev-{index:04d}" for index in range(300)),
        key=lambda item_id: (study._selection_key(item_id, secret), item_id),
    )
    assert selected_ids == set(ordered_ids[96:264])
    assert not selected_ids.intersection(ordered_ids[:96])
    serialized = json.dumps(loaded_receipt, sort_keys=True)
    assert "official-dev-" not in serialized
    assert "Which signal" not in serialized
    assert str(private_root) not in serialized


def test_preflight_failure_does_not_consume_or_open_source(
    private_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prepare(private_tmp_path, monkeypatch)
    prereg = _preregister(paths)
    bad_parent = paths["artifacts"] / "not-a-directory"
    bad_parent.write_text("blocked\n", encoding="utf-8")
    monkeypatch.setattr(
        study.zipfile,
        "ZipFile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("source archive opened before marker")
        ),
    )
    with pytest.raises((FileExistsError, NotADirectoryError)):
        study.acquire_private_blocks(
            project=paths["project"],
            preregistration_path=prereg,
            official_repository=paths["official"],
            selection_secret_path=paths["secret"],
            prior_preregistration_path=paths["prior_prereg"],
            prior_acquisition_receipt_path=paths["prior"],
            source_archive_path=paths["archive"],
            private_root=paths["artifacts"] / "pack",
            private_locator_path=bad_parent / "locator.json",
            public_receipt_path=paths["project"] / "manifests" / "acq.json",
        )
    assert not (paths["project"] / study.CONSUMPTION_RELATIVE).exists()


def test_consumption_marker_forbids_replay(
    private_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prepare(private_tmp_path, monkeypatch)
    prereg = _preregister(paths)
    arguments = {
        "project": paths["project"],
        "preregistration_path": prereg,
        "official_repository": paths["official"],
        "selection_secret_path": paths["secret"],
        "prior_preregistration_path": paths["prior_prereg"],
        "prior_acquisition_receipt_path": paths["prior"],
        "source_archive_path": paths["archive"],
        "private_root": paths["artifacts"] / "pack",
        "private_locator_path": paths["artifacts"] / "locator" / "locator.json",
        "public_receipt_path": paths["project"] / "manifests" / "acq.json",
    }
    study.acquire_private_blocks(**arguments)
    marker = paths["project"] / study.CONSUMPTION_RELATIVE
    assert marker.is_file()
    with pytest.raises(FileExistsError, match="already consumed"):
        study.acquire_private_blocks(**arguments)


def test_canonical_prior_receipt_rejects_alternate_copy(
    private_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prepare(private_tmp_path, monkeypatch)
    alternate = paths["project"] / "manifests" / "alternate.json"
    alternate.write_bytes(paths["prior"].read_bytes())
    _commit_all(paths["project"], "alternate")
    with pytest.raises(
        study.MuSiQuePortfolioAcquisitionError, match="fixed canonical path"
    ):
        study.build_preregistration(
            project=paths["project"],
            official_repository=paths["official"],
            selection_secret_path=paths["secret"],
            prior_preregistration_path=paths["prior_prereg"],
            prior_acquisition_receipt_path=alternate,
        )


def test_canonical_prior_preregistration_and_dependency_drift_are_rejected(
    private_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prepare(private_tmp_path, monkeypatch)
    alternate = paths["project"] / "manifests" / "alternate-prereg.json"
    alternate.write_bytes(paths["prior_prereg"].read_bytes())
    _commit_all(paths["project"], "alternate prior prereg")
    with pytest.raises(
        study.MuSiQuePortfolioAcquisitionError, match="fixed canonical path"
    ):
        study.build_preregistration(
            project=paths["project"],
            official_repository=paths["official"],
            selection_secret_path=paths["secret"],
            prior_preregistration_path=alternate,
            prior_acquisition_receipt_path=paths["prior"],
        )

    drifted = {"files": [], "set_sha256": stable_hash([])}
    monkeypatch.setattr(study.prior, "_implementation_binding", lambda _root: drifted)
    with pytest.raises(
        study.MuSiQuePortfolioAcquisitionError, match="dependency closure drifted"
    ):
        study.build_preregistration(
            project=paths["project"],
            official_repository=paths["official"],
            selection_secret_path=paths["secret"],
            prior_preregistration_path=paths["prior_prereg"],
            prior_acquisition_receipt_path=paths["prior"],
        )


def test_public_private_output_overlap_fails_before_marker_or_pack_creation(
    private_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prepare(private_tmp_path, monkeypatch)
    prereg = _preregister(paths)
    private_root = paths["artifacts"] / "overlap-pack"
    with pytest.raises(
        study.MuSiQuePortfolioAcquisitionError, match="must be disjoint"
    ):
        study.acquire_private_blocks(
            project=paths["project"],
            preregistration_path=prereg,
            official_repository=paths["official"],
            selection_secret_path=paths["secret"],
            prior_preregistration_path=paths["prior_prereg"],
            prior_acquisition_receipt_path=paths["prior"],
            source_archive_path=paths["archive"],
            private_root=private_root,
            private_locator_path=paths["artifacts"] / "locator" / "locator.json",
            public_receipt_path=private_root / "public.json",
        )
    assert not private_root.exists()
    assert not (paths["project"] / study.CONSUMPTION_RELATIVE).exists()


def test_private_block_tamper_is_rejected(
    private_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prepare(private_tmp_path, monkeypatch)
    prereg = _preregister(paths)
    private_root = paths["artifacts"] / "pack"
    public_receipt = paths["project"] / "manifests" / "acq.json"
    receipt = study.acquire_private_blocks(
        project=paths["project"],
        preregistration_path=prereg,
        official_repository=paths["official"],
        selection_secret_path=paths["secret"],
        prior_preregistration_path=paths["prior_prereg"],
        prior_acquisition_receipt_path=paths["prior"],
        source_archive_path=paths["archive"],
        private_root=private_root,
        private_locator_path=paths["artifacts"] / "locator" / "locator.json",
        public_receipt_path=public_receipt,
    )
    study._write_json_exclusive(
        public_receipt, receipt, hash_field="acquisition_sha256", mode=0o644
    )
    _loaded, commitments = study.load_acquisition_binding(public_receipt)
    target = private_root / "A_form_0.jsonl"
    target.write_bytes(target.read_bytes() + b"{}\n")
    with pytest.raises(study.MuSiQuePortfolioAcquisitionError, match="file hash"):
        study.load_private_block(target, commitment=commitments[0])


def test_live_loader_accepts_only_complete_canonical_committed_chain(
    private_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prepare(private_tmp_path, monkeypatch)
    prereg = _preregister(paths)
    receipt_path, _pack, _locator, _receipt = _acquire_and_commit(paths, prereg)
    loaded, blocks = study.load_acquisition_binding_live(
        project=paths["project"],
        path=receipt_path,
        selection_secret_path=paths["secret"],
    )
    assert loaded["counts"]["selected_rows"] == 168
    assert sum(row.count for row in blocks) == 168

    alternate = paths["project"] / "manifests" / "alternate-acquisition.json"
    alternate.write_bytes(receipt_path.read_bytes())
    _commit_all(paths["project"], "alternate acquisition")
    with pytest.raises(
        study.MuSiQuePortfolioAcquisitionError, match="fixed canonical path"
    ):
        study.load_acquisition_binding_live(
            project=paths["project"],
            path=alternate,
            selection_secret_path=paths["secret"],
        )


@pytest.mark.parametrize("tamper", ["receipt", "preregistration", "marker", "secret"])
def test_live_loader_rejects_rehashed_chain_tampering(
    private_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    paths = _prepare(private_tmp_path, monkeypatch)
    prereg = _preregister(paths)
    receipt_path, _pack, _locator, _receipt = _acquire_and_commit(paths, prereg)
    if tamper == "receipt":
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        payload["preregistration_sha256"] = "f" * 64
        payload["acquisition_sha256"] = stable_hash(
            {key: value for key, value in payload.items() if key != "acquisition_sha256"}
        )
        receipt_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _commit_all(paths["project"], "forged receipt")
    elif tamper == "preregistration":
        payload = json.loads(prereg.read_text(encoding="utf-8"))
        payload["claim_boundary"]["performance_claim_before_measurement"] = True
        payload["preregistration_sha256"] = stable_hash(
            {
                key: value
                for key, value in payload.items()
                if key != "preregistration_sha256"
            }
        )
        prereg.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _commit_all(paths["project"], "forged preregistration")
    elif tamper == "marker":
        marker_path = paths["project"] / study.CONSUMPTION_RELATIVE
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
        payload["source_member_sha256"] = "a" * 64
        payload["consumption_sha256"] = stable_hash(
            {key: value for key, value in payload.items() if key != "consumption_sha256"}
        )
        marker_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        paths["secret"].write_bytes(b"x" * 32)
        paths["secret"].chmod(0o600)

    with pytest.raises(study.MuSiQuePortfolioAcquisitionError):
        study.load_acquisition_binding_live(
            project=paths["project"],
            path=receipt_path,
            selection_secret_path=paths["secret"],
        )


def test_plain_loader_rejects_rehashed_extra_nested_key_and_public_content(
    private_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prepare(private_tmp_path, monkeypatch)
    prereg = _preregister(paths)
    receipt_path, _pack, _locator, _receipt = _acquire_and_commit(paths, prereg)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["source"]["question"] = "private question must never be public"
    payload["acquisition_sha256"] = stable_hash(
        {key: value for key, value in payload.items() if key != "acquisition_sha256"}
    )
    receipt_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(study.MuSiQuePortfolioAcquisitionError):
        study.load_acquisition_binding(receipt_path)


def test_source_hash_is_first_read_only_after_durable_marker(
    private_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prepare(private_tmp_path, monkeypatch)
    prereg = _preregister(paths)
    original = study._sha256_file
    observed = {"source_hash_calls": 0}

    def guarded(path: Path) -> str:
        if Path(path).resolve() == paths["archive"].resolve():
            observed["source_hash_calls"] += 1
            assert (paths["project"] / study.CONSUMPTION_RELATIVE).is_file()
        return original(path)

    monkeypatch.setattr(study, "_sha256_file", guarded)
    study.acquire_private_blocks(
        project=paths["project"],
        preregistration_path=prereg,
        official_repository=paths["official"],
        selection_secret_path=paths["secret"],
        prior_preregistration_path=paths["prior_prereg"],
        prior_acquisition_receipt_path=paths["prior"],
        source_archive_path=paths["archive"],
        private_root=paths["artifacts"] / "pack",
        private_locator_path=paths["artifacts"] / "locator" / "locator.json",
        public_receipt_path=paths["project"] / "manifests" / "source-order.json",
    )
    assert observed["source_hash_calls"] == 1


@pytest.mark.parametrize("fault", ["hardlink", "chmod", "directory_fsync", "space"])
def test_persistence_faults_leave_no_pack_or_marker(
    private_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    pack = private_tmp_path / "private" / "deep" / "pack"
    locator = private_tmp_path / "locator" / "deep" / "locator.json"
    marker = private_tmp_path / "marker" / "deep" / "consumed.json"
    public = private_tmp_path / "public" / "deep" / "receipt.json"
    if fault == "hardlink":
        monkeypatch.setattr(
            study.os,
            "link",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no hardlink")),
        )
    elif fault == "chmod":
        monkeypatch.setattr(
            study.os,
            "chmod",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no chmod")),
        )
    elif fault == "directory_fsync":
        monkeypatch.setattr(
            study,
            "_fsync_directory",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no fsync")),
        )
    else:
        usage = type("Usage", (), {"total": 1, "used": 1, "free": 0})()
        monkeypatch.setattr(study.shutil, "disk_usage", lambda _path: usage)
    with pytest.raises(OSError if fault != "space" else study.MuSiQuePortfolioAcquisitionError):
        study._preflight_persistence(
            pack_root=pack,
            locator=locator,
            consumption_path=marker,
            public_receipt_path=public,
        )
    assert not pack.exists()
    assert not marker.exists()


def test_preflight_creates_nested_parents_uses_formal_writer_and_checks_all_outputs(
    private_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pack = private_tmp_path / "private" / "nested" / "pack"
    locator = private_tmp_path / "locator" / "nested" / "locator.json"
    marker = private_tmp_path / "marker" / "nested" / "consumed.json"
    public = private_tmp_path / "public" / "nested" / "receipt.json"
    original_write = study._atomic_write_exclusive
    write_calls: list[Path] = []
    space_calls: list[Path] = []

    def observed_write(path: Path, raw: bytes, *, mode: int) -> None:
        write_calls.append(path.parent)
        original_write(path, raw, mode=mode)

    real_usage = study.shutil.disk_usage

    def observed_usage(path: Path):
        space_calls.append(Path(path))
        return real_usage(path)

    monkeypatch.setattr(study, "_atomic_write_exclusive", observed_write)
    monkeypatch.setattr(study.shutil, "disk_usage", observed_usage)
    study._preflight_persistence(
        pack_root=pack,
        locator=locator,
        consumption_path=marker,
        public_receipt_path=public,
    )
    assert pack.is_dir()
    assert stat.S_IMODE(pack.stat().st_mode) == 0o700
    assert set(space_calls) == {pack, locator.parent, marker.parent, public.parent}
    assert {pack, locator.parent, marker.parent, public.parent}.issubset(
        set(write_calls)
    )
    pack.rmdir()


def test_cli_has_no_prior_private_pack_surface(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        study.main(["acquire", "--help"])
    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    assert "--prior-private" not in help_text
    assert "--prior-preregistration" in help_text
