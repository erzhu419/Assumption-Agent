"""Build the closed-choice Qwen asset manifest without loading model weights.

This qualification-only command hashes the exact model tree, loads only the
local config/tokenizer, and records the live torch/transformers/CUDA runtime.
The subsequent actual qualification is therefore the unique weight load and
model execution.  No dataset, story, scorer, logits, or label is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import secrets
import stat
import sys
from typing import Callable, Sequence

from . import closed_choice_qwen_runtime as qwen_closed
from . import closed_choice_worker as closed
from . import contract
from . import worker


VERSION = "gscl_narrative_closed_choice_manifest_builder_v1"
ATTENTION_IMPLEMENTATION = "sdpa"


def _publish_validated_once(
    path: Path,
    raw: bytes,
    *,
    validate_pending: Callable[[Path], None],
) -> str:
    """Validate a same-directory pending inode, then publish without replace."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    if (
        not absolute.parent.exists()
        or absolute.parent.is_symlink()
        or not stat.S_ISDIR(absolute.parent.stat().st_mode)
        or absolute.exists()
        or absolute.is_symlink()
        or not callable(validate_pending)
    ):
        raise closed.ClosedChoiceError(
            "closed_choice_manifest_output_invalid"
        )
    pending: Path | None = None
    descriptor: int | None = None
    for _ in range(32):
        candidate = absolute.with_name(
            f".{absolute.name}.pending-{secrets.token_hex(16)}"
        )
        try:
            descriptor = os.open(
                candidate,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_CLOEXEC
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except FileExistsError:
            continue
        pending = candidate
        break
    if pending is None or descriptor is None:
        raise closed.ClosedChoiceError(
            "closed_choice_manifest_pending_unavailable"
        )
    expected = (hashlib.sha256(raw).hexdigest(), len(raw))
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(raw)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise closed.ClosedChoiceError(
                    "closed_choice_manifest_write_failed"
                )
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None

        validate_pending(pending)
        read_descriptor = os.open(
            pending,
            os.O_RDONLY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            observed = worker._stable_file_hash_from_fd(
                read_descriptor,
                maximum=worker.MAXIMUM_MODEL_MANIFEST_BYTES,
            )
        finally:
            os.close(read_descriptor)
        if observed != expected:
            raise closed.ClosedChoiceError(
                "closed_choice_manifest_pending_changed"
            )
        pending_metadata = pending.stat(follow_symlinks=False)
        try:
            os.link(
                pending,
                absolute,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise closed.ClosedChoiceError(
                "closed_choice_manifest_output_invalid"
            ) from exc
        try:
            directory_descriptor = os.open(
                absolute.parent,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except Exception:
            try:
                final_metadata = absolute.stat(
                    follow_symlinks=False
                )
                if (
                    final_metadata.st_dev == pending_metadata.st_dev
                    and final_metadata.st_ino
                    == pending_metadata.st_ino
                ):
                    absolute.unlink()
            except FileNotFoundError:
                pass
            raise
        return expected[0]
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            pending.unlink()
        except FileNotFoundError:
            pass


def _context_limit(config: object, tokenizer: object) -> int:
    values = [
        value
        for value in (
            getattr(config, "max_position_embeddings", None),
            getattr(tokenizer, "model_max_length", None),
        )
        if isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value < 10**8
    ]
    if not values:
        raise closed.ClosedChoiceError(
            "closed_choice_context_limit_unavailable"
        )
    return min(values)


def build_manifest_without_weight_load(
    *, model_root: Path, output_path: Path
) -> bytes:
    """Create one exact manifest using config/tokenizer/runtime only."""

    model_root = Path(os.path.abspath(os.fspath(model_root)))
    output_path = Path(os.path.abspath(os.fspath(output_path)))
    try:
        output_path.relative_to(model_root)
    except ValueError:
        pass
    else:
        raise closed.ClosedChoiceError(
            "closed_choice_manifest_inside_model_tree"
        )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import torch
        import transformers
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:
        raise closed.ClosedChoiceError(
            "closed_choice_manifest_runtime_unavailable"
        ) from exc
    if not torch.cuda.is_available():
        raise closed.ClosedChoiceError(
            "closed_choice_manifest_cuda_unavailable"
        )
    torch_origin = getattr(torch, "__file__", None)
    transformers_origin = getattr(
        transformers, "__file__", None
    )
    if (
        not isinstance(torch_origin, str)
        or not torch_origin
        or not isinstance(transformers_origin, str)
        or not transformers_origin
    ):
        raise closed.ClosedChoiceError(
            "closed_choice_manifest_module_origin_unavailable"
        )
    try:
        config = AutoConfig.from_pretrained(
            model_root,
            local_files_only=True,
            trust_remote_code=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_root,
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise closed.ClosedChoiceError(
            "closed_choice_manifest_config_tokenizer_failed"
        ) from exc
    projection = qwen_closed._config_projection(config)
    architectures = projection["architectures"]
    if (
        projection["values"].get("model_type") != "qwen2"
        or architectures != ["Qwen2ForCausalLM"]
    ):
        raise closed.ClosedChoiceError(
            "closed_choice_manifest_architecture_invalid"
        )
    critical = {
        key: getattr(config, key, None)
        for key in worker.QWEN_ARCHITECTURE
    }
    if critical != worker.QWEN_ARCHITECTURE:
        raise closed.ClosedChoiceError(
            "closed_choice_manifest_critical_config_invalid"
        )
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise closed.ClosedChoiceError(
            "closed_choice_manifest_chat_template_unavailable"
        )
    declarations = {
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "chat_template_sha256": hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest(),
        "context_limit": _context_limit(config, tokenizer),
        "critical_config": critical,
        "loaded_config_sha256": contract.semantic_sha256(
            projection
        ),
        "model_class": "Qwen2ForCausalLM",
        "special_token_ids": {
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "tokenizer_class": tokenizer.__class__.__name__,
    }
    cudnn_version = torch.backends.cudnn.version()
    if not isinstance(cudnn_version, int):
        raise closed.ClosedChoiceError(
            "closed_choice_manifest_cudnn_version_unavailable"
        )
    capability = torch.cuda.get_device_capability(0)
    runtime_requirements = {
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": cudnn_version,
        "gpu_compute_capability": [
            int(capability[0]),
            int(capability[1]),
        ],
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "python_executable_sha256": (
            worker._hash_runtime_executable()
        ),
        "python_implementation": (
            platform.python_implementation()
        ),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_distribution_sha256": (
            worker._distribution_closure_sha256(
                "torch",
                required_module_origins=(Path(torch_origin),),
            )
        ),
        "transformers_version": str(
            transformers.__version__
        ),
        "transformers_distribution_sha256": (
            worker._distribution_closure_sha256(
                "transformers",
                required_module_origins=(
                    Path(transformers_origin),
                ),
            )
        ),
    }
    raw = worker.build_model_asset_manifest_qualification_only(
        model_root=model_root,
        declarations=declarations,
        runtime_requirements=runtime_requirements,
    )
    expected_sha256 = hashlib.sha256(raw).hexdigest()

    def validate_pending(pending: Path) -> None:
        # Decode and rescan through the exact loader before the final pathname
        # exists.  The subsequent hard-link publication preserves this exact
        # validated inode and cannot overwrite an existing result.
        verified = worker.load_model_asset_manifest(
            manifest_path=pending,
            model_root=model_root,
        )
        if verified.manifest_file_sha256 != expected_sha256:
            raise closed.ClosedChoiceError(
                "closed_choice_manifest_postwrite_binding_invalid"
            )

    _publish_validated_once(
        output_path,
        raw,
        validate_pending=validate_pending,
    )
    return raw


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    raw = build_manifest_without_weight_load(
        model_root=arguments.model,
        output_path=arguments.output,
    )
    print(
        json.dumps(
            {
                "model_weight_load_count": 0,
                "official_source_access_count": 0,
                "output_file_sha256": hashlib.sha256(raw).hexdigest(),
                "status": "PASS_CLOSED_CHOICE_MANIFEST_BUILT_SOURCE_FREE",
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        closed.ClosedChoiceError,
        contract.NarrativeExtractorRuntimeError,
    ) as exc:
        issue_id = getattr(exc, "issue_id", "runtime_error")
        print(
            f"{VERSION} failed closed: {issue_id}",
            file=sys.stderr,
        )
        raise SystemExit(2) from None


__all__ = [
    "ATTENTION_IMPLEMENTATION",
    "VERSION",
    "build_manifest_without_weight_load",
]
