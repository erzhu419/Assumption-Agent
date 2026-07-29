from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Iterator

import pytest

from replication_runtime.wikisql_uao_runtime_qualification import (
    alias_runtime,
)


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    """Use a native Linux filesystem because the contract requires mode 0700."""

    with tempfile.TemporaryDirectory(
        prefix="wikisql-uao-alias-", dir="/tmp"
    ) as value:
        yield Path(value)


def _identity(root: Path) -> dict[str, object]:
    rows = []
    for path in sorted(
        root.rglob("*"),
        key=lambda value: value.relative_to(root).as_posix(),
    ):
        metadata = path.lstat()
        relative = path.relative_to(root).as_posix()
        if stat.S_ISDIR(metadata.st_mode):
            rows.append({"kind": "directory", "path": relative})
        elif stat.S_ISREG(metadata.st_mode):
            raw = path.read_bytes()
            rows.append(
                {
                    "kind": "file",
                    "path": relative,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "size": len(raw),
                }
            )
        else:
            raise AssertionError("test identity accepts direct trees only")
    raw = json.dumps(
        rows,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return {
        "entry_count": len(rows),
        "tree_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    work = tmp_path / "work"
    llm = tmp_path / "verified-smollm"
    embedding = tmp_path / "verified-minilm"
    for root in (work, llm, embedding):
        root.mkdir(mode=0o700)
    (llm / "weights.bin").write_bytes(b"fixed llm")
    (embedding / "weights.bin").write_bytes(b"fixed embedding")
    return work, llm, embedding


def test_bind_short_aliases_returns_serializable_exact_receipt(
    tmp_path: Path,
) -> None:
    work, llm, embedding = _roots(tmp_path)
    llm_before = _identity(llm)
    embedding_before = _identity(embedding)

    receipt = alias_runtime.bind_and_verify_short_model_aliases(
        writable_root=work,
        llm_model_root=llm,
        embedding_model_root=embedding,
        identity_fn=_identity,
    )

    json.dumps(receipt, allow_nan=False, sort_keys=True)
    alias_root = work / alias_runtime.ALIAS_DIRECTORY
    assert stat.S_IMODE(alias_root.stat().st_mode) == 0o700
    assert alias_root.resolve(strict=True) == alias_root
    for alias, target in (
        (alias_runtime.LLM_ALIAS, llm),
        (alias_runtime.EMBEDDING_ALIAS, embedding),
    ):
        link = alias_root / alias
        assert link.is_symlink()
        assert os.readlink(link) == str(target)
        assert link.resolve(strict=True) == target
        assert os.path.samefile(link, target)
        assert receipt["aliases"][alias]["samefile"] is True
    assert receipt["derived_hipporag_component"] == (
        "Transformers_smollm2_Transformers_minilm"
    )
    assert receipt["derived_hipporag_component_utf8_bytes"] == 40
    assert receipt["filesystem_name_max_bytes"] >= 40
    assert receipt["model_content_changed"] is False
    assert _identity(llm) == llm_before
    assert _identity(embedding) == embedding_before
    body = {
        key: value
        for key, value in receipt.items()
        if key != "self_sha256"
    }
    assert receipt["self_sha256"] == hashlib.sha256(
        json.dumps(
            body,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


@pytest.mark.parametrize("case", ("relative", "missing", "symlink"))
def test_bind_short_aliases_rejects_non_direct_model_roots(
    tmp_path: Path,
    case: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work, llm, embedding = _roots(tmp_path)
    if case == "relative":
        monkeypatch.chdir(tmp_path)
        llm = Path("verified-smollm")
    elif case == "missing":
        llm = tmp_path / "missing"
    else:
        linked = tmp_path / "linked-smollm"
        linked.symlink_to(llm, target_is_directory=True)
        llm = linked

    with pytest.raises(alias_runtime.WikiSQLUAOAliasRuntimeError):
        alias_runtime.bind_and_verify_short_model_aliases(
            writable_root=work,
            llm_model_root=llm,
            embedding_model_root=embedding,
            identity_fn=_identity,
        )


def test_bind_short_aliases_requires_fresh_alias_root(
    tmp_path: Path,
) -> None:
    work, llm, embedding = _roots(tmp_path)
    (work / alias_runtime.ALIAS_DIRECTORY).mkdir(mode=0o700)
    with pytest.raises(
        alias_runtime.WikiSQLUAOAliasRuntimeError,
        match="not fresh",
    ):
        alias_runtime.bind_and_verify_short_model_aliases(
            writable_root=work,
            llm_model_root=llm,
            embedding_model_root=embedding,
            identity_fn=_identity,
        )


def test_bind_short_aliases_fails_when_name_max_is_below_40(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work, llm, embedding = _roots(tmp_path)
    monkeypatch.setattr(alias_runtime.os, "pathconf", lambda *_: 39)
    with pytest.raises(
        alias_runtime.WikiSQLUAOAliasRuntimeError,
        match="exceeds NAME_MAX",
    ):
        alias_runtime.bind_and_verify_short_model_aliases(
            writable_root=work,
            llm_model_root=llm,
            embedding_model_root=embedding,
            identity_fn=_identity,
        )


def test_bind_short_aliases_accepts_name_max_exactly_40(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work, llm, embedding = _roots(tmp_path)
    monkeypatch.setattr(alias_runtime.os, "pathconf", lambda *_: 40)
    receipt = alias_runtime.bind_and_verify_short_model_aliases(
        writable_root=work,
        llm_model_root=llm,
        embedding_model_root=embedding,
        identity_fn=_identity,
    )
    assert receipt["filesystem_name_max_bytes"] == 40


def test_bind_short_aliases_rejects_tree_identity_drift(
    tmp_path: Path,
) -> None:
    work, llm, embedding = _roots(tmp_path)
    calls: dict[Path, int] = {}

    def drifting(root: Path) -> dict[str, object]:
        calls[root] = calls.get(root, 0) + 1
        return {"generation": calls[root], **_identity(root)}

    with pytest.raises(
        alias_runtime.WikiSQLUAOAliasRuntimeError,
        match="identity changed",
    ):
        alias_runtime.bind_and_verify_short_model_aliases(
            writable_root=work,
            llm_model_root=llm,
            embedding_model_root=embedding,
            identity_fn=drifting,
        )


def test_bind_short_aliases_rejects_samefile_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work, llm, embedding = _roots(tmp_path)
    real_samefile = alias_runtime.os.path.samefile

    def mismatched(left: object, right: object) -> bool:
        if Path(left).is_symlink():
            return False
        return real_samefile(left, right)

    monkeypatch.setattr(alias_runtime.os.path, "samefile", mismatched)
    with pytest.raises(
        alias_runtime.WikiSQLUAOAliasRuntimeError,
        match="binding drifted",
    ):
        alias_runtime.bind_and_verify_short_model_aliases(
            writable_root=work,
            llm_model_root=llm,
            embedding_model_root=embedding,
            identity_fn=_identity,
        )
