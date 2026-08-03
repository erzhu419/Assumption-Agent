#!/usr/bin/env python3
"""CLI for prepare, finalize-index and verify-commit publication audits."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import stat
import sys


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from hegel_machine.phase3_m25_commit_b_publication_audit_v1 import (  # noqa: E402
    AUDIT_RECEIPT_REPOSITORY_PATH,
    CommitBPublicationAuditError,
    EXTERNAL_STATUS_REPOSITORY_PATH,
    build_external_status_from_worktree_v1,
    canonical_json_v1,
    finalize_staged_commit_b_publication_v1,
    run_commit_b_publication_actor_audit_v1,
    verify_commit_b_publication_commit_v1,
)


def _repository(path: str) -> Path:
    requested = Path(os.path.abspath(path))
    try:
        metadata = requested.lstat()
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise CommitBPublicationAuditError(
            "FAIL_COMMIT_B_AUDIT_PATH_POLICY", f"repository is unavailable: {exc}"
        ) from exc
    if (
        requested != resolved
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise CommitBPublicationAuditError(
            "FAIL_COMMIT_B_AUDIT_PATH_POLICY",
            "repository must be a real absolute directory without aliasing",
        )
    return requested


def _open_parent_without_symlinks(anchor: Path, parts: tuple[str, ...]) -> int:
    if not parts or any(part in {"", ".", ".."} or "/" in part for part in parts):
        raise CommitBPublicationAuditError(
            "FAIL_COMMIT_B_AUDIT_PATH_POLICY", "output path is not canonical"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        current = os.open(anchor, flags)
    except OSError as exc:
        raise CommitBPublicationAuditError(
            "FAIL_COMMIT_B_AUDIT_PATH_POLICY", f"output anchor is unavailable: {exc}"
        ) from exc
    try:
        for component in parts[:-1]:
            following = os.open(component, flags, dir_fd=current)
            os.close(current)
            current = following
        return current
    except BaseException:
        os.close(current)
        raise


def _write_exclusive_anchored(
    anchor: Path, parts: tuple[str, ...], payload: bytes, *, mode: int
) -> None:
    parent = _open_parent_without_symlinks(anchor, parts)
    basename = parts[-1]
    write_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor: int | None = None
    created_identity: tuple[int, int] | None = None

    def cleanup_created() -> None:
        if created_identity is None:
            return
        try:
            observed = os.stat(basename, dir_fd=parent, follow_symlinks=False)
            if (observed.st_dev, observed.st_ino) == created_identity:
                os.unlink(basename, dir_fd=parent)
                os.fsync(parent)
        except OSError:
            pass

    try:
        descriptor = os.open(basename, write_flags, mode, dir_fd=parent)
        opened = os.fstat(descriptor)
        created_identity = (opened.st_dev, opened.st_ino)
        os.fchmod(descriptor, mode)
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("short publication-audit write")
            offset += written
        os.fsync(descriptor)
        created = os.fstat(descriptor)
        if not stat.S_ISREG(created.st_mode) or stat.S_IMODE(created.st_mode) != mode:
            raise OSError("publication-audit output mode/type differs")
        os.fsync(parent)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        cleanup_created()
        os.close(parent)
        raise
    else:
        os.close(descriptor)

    try:
        replay_parent = _open_parent_without_symlinks(anchor, parts)
    except BaseException:
        cleanup_created()
        os.close(parent)
        raise
    try:
        try:
            original_parent = os.fstat(parent)
            reopened_parent = os.fstat(replay_parent)
            if (original_parent.st_dev, original_parent.st_ino) != (
                reopened_parent.st_dev,
                reopened_parent.st_ino,
            ):
                raise OSError("publication-audit output parent identity changed")
            replay = os.open(
                basename,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                dir_fd=replay_parent,
            )
            try:
                reopened = os.fstat(replay)
                if (
                    reopened.st_dev,
                    reopened.st_ino,
                    reopened.st_size,
                    stat.S_IMODE(reopened.st_mode),
                ) != (created.st_dev, created.st_ino, len(payload), mode):
                    raise OSError("publication-audit output identity changed after fsync")
            finally:
                os.close(replay)
        except BaseException:
            cleanup_created()
            raise
    finally:
        os.close(replay_parent)
        os.close(parent)


def _write_repository_exclusive(
    repository: Path, repository_path: str, payload: bytes, *, mode: int
) -> None:
    _write_exclusive_anchored(
        repository, tuple(repository_path.split("/")), payload, mode=mode
    )


def _write_external_exclusive(path_text: str, payload: bytes, *, mode: int) -> None:
    if (
        not os.path.isabs(path_text)
        or os.path.normpath(path_text) != path_text
        or os.path.abspath(path_text) != path_text
    ):
        raise CommitBPublicationAuditError(
            "FAIL_COMMIT_B_AUDIT_PATH_POLICY",
            "repo-external output must be a canonical absolute path",
        )
    path = Path(path_text)
    _write_exclusive_anchored(Path(path.anchor), tuple(path.parts[1:]), payload, mode=mode)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=str(PROJECT.parent))
    sub = parser.add_subparsers(dest="command", required=True)
    render = sub.add_parser("render-status")
    render.add_argument("--basis-commit", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--basis-commit", required=True)
    prepare.add_argument("--output")
    finalize = sub.add_parser("finalize-index")
    finalize.add_argument("--basis-commit", required=True)
    finalize.add_argument("--output", help="optional repo-external final receipt")
    verify = sub.add_parser("verify-commit")
    verify.add_argument("--basis-commit", required=True)
    verify.add_argument("--publication-commit", required=True)
    verify.add_argument("--finalize-receipt", required=True, type=Path)
    verify.add_argument("--output", help="optional repo-external verification receipt")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = _repository(args.repository)
    try:
        if args.command == "render-status":
            payload = build_external_status_from_worktree_v1(
                repository=repository, basis_commit=args.basis_commit
            )
            _write_repository_exclusive(
                repository, EXTERNAL_STATUS_REPOSITORY_PATH, payload, mode=0o644
            )
            sys.stdout.buffer.write(
                canonical_json_v1(
                    {
                        "status": "PASS_EXTERNAL_STATUS_WRITTEN_NOT_STAGED",
                        "basis_commit_sha1": args.basis_commit,
                        "output_repository_path": EXTERNAL_STATUS_REPOSITORY_PATH,
                        "status_sha256": hashlib.sha256(payload).hexdigest(),
                        "index_authority_deferred_to_prepare": True,
                        "formal_gate_delta": 0,
                        "m3_start_or_state_transition": False,
                    }
                )
            )
            return 0
        if args.command == "prepare":
            result = run_commit_b_publication_actor_audit_v1(
                repository=repository, basis_commit=args.basis_commit
            )
            expected = repository / AUDIT_RECEIPT_REPOSITORY_PATH
            if args.output is not None and args.output != expected.as_posix():
                raise CommitBPublicationAuditError(
                    "FAIL_COMMIT_B_AUDIT_PATH_POLICY",
                    "prepare receipt must use the unique frozen in-repository path",
                )
            _write_repository_exclusive(
                repository,
                AUDIT_RECEIPT_REPOSITORY_PATH,
                result.canonical_receipt_bytes,
                mode=0o644,
            )
            sys.stdout.buffer.write(
                canonical_json_v1(
                    {
                        "status": "PASS_PREPARE_RECEIPT_WRITTEN_NOT_STAGED",
                        "basis_commit_sha1": args.basis_commit,
                        "output_repository_path": AUDIT_RECEIPT_REPOSITORY_PATH,
                        "receipt_sha256": result.receipt["receipt_sha256"],
                        "formal_host_replay": dict(result.host_formal_replay),
                        "formal_gate_delta": 0,
                        "m3_start_or_state_transition": False,
                    }
                )
            )
            return 0
        if args.command == "finalize-index":
            result = finalize_staged_commit_b_publication_v1(
                repository=repository, basis_commit=args.basis_commit
            )
        else:
            result = verify_commit_b_publication_commit_v1(
                repository=repository,
                basis_commit=args.basis_commit,
                publication_commit=args.publication_commit,
                finalize_receipt_path=args.finalize_receipt,
            )
        payload = canonical_json_v1(result)
        if args.output:
            output = Path(args.output)
            if not output.is_absolute():
                raise CommitBPublicationAuditError(
                    "FAIL_COMMIT_B_AUDIT_PATH_POLICY",
                    "final/verification receipt output must be an absolute path",
                )
            try:
                output.relative_to(repository)
            except ValueError:
                pass
            else:
                raise CommitBPublicationAuditError(
                    "FAIL_COMMIT_B_AUDIT_PATH_POLICY",
                    "final/verification receipt output must remain outside the repository",
                )
            _write_external_exclusive(args.output, payload, mode=0o600)
        sys.stdout.buffer.write(payload)
        return 0
    except (CommitBPublicationAuditError, OSError) as exc:
        code = getattr(exc, "code", "FAIL_COMMIT_B_PUBLICATION_AUDIT_CLI")
        sys.stderr.write(str(code) + "\n")
        return 70


if __name__ == "__main__":
    raise SystemExit(main())
