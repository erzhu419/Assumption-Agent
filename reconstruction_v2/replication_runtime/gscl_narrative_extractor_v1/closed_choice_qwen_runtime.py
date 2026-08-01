"""Exact local-Qwen backend for program-owned closed-choice extraction.

The runtime performs batched teacher-forced forward passes and conditional
log-likelihood ranking.  It never calls a generation API.  Formal execution
accepts only the exact runtime type constructed from the verified model asset
manifest and never accepts caller-provided scorers, logits, records, parsers,
or receipts.
"""

from __future__ import annotations

import hashlib
from importlib import import_module
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
from types import MappingProxyType
from typing import Mapping

from . import closed_choice_worker as closed
from . import contract
from . import worker


VERSION = "gscl_narrative_closed_choice_qwen_runtime_v1"
RUNTIME_RECEIPT_SCHEMA = f"{VERSION}.runtime_receipt.v1"
DOUBLE_RUN_RECEIPT_SCHEMA = f"{VERSION}.double_run_receipt.v1"
DEVICE = "cuda:0"
TORCH_SEED = worker.TORCH_SEED
FORMAL_SCORING_BATCH_SIZE = 4
CRITICAL_DEPENDENCY_DISTRIBUTIONS = (
    ("huggingface_hub", "huggingface-hub"),
    ("numpy", "numpy"),
    ("safetensors", "safetensors"),
    ("tokenizers", "tokenizers"),
)

_CONSTRUCTING_MARKER = object()
_VERIFIED_RUNTIME_MARKER = object()
_BACKEND_MARKER = object()


def _config_projection(config: object) -> dict[str, object]:
    """Stable architecture projection shared by builder and loaded runtime."""

    architectures = getattr(config, "architectures", None)
    if (
        type(architectures) is not list
        or not architectures
        or any(not isinstance(value, str) for value in architectures)
    ):
        raise closed.ClosedChoiceError(
            "closed_choice_model_architecture_unavailable"
        )
    keys = (
        "bos_token_id",
        "eos_token_id",
        "hidden_size",
        "intermediate_size",
        "max_position_embeddings",
        "model_type",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_theta",
        "sliding_window",
        "tie_word_embeddings",
        "use_sliding_window",
        "vocab_size",
    )
    return {
        "architectures": list(architectures),
        "config_class": config.__class__.__name__,
        "values": {
            key: getattr(config, key, None) for key in keys
        },
    }


def _source_closure_sha256() -> str:
    rows: list[dict[str, str]] = []
    for module in (contract, closed, worker):
        path = Path(module.__file__)
        descriptor = os.open(
            path,
            os.O_RDONLY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            digest, _ = worker._stable_file_hash_from_fd(
                descriptor, maximum=4 * 1024 * 1024
            )
        finally:
            os.close(descriptor)
        rows.append(
            {
                "module": module.__name__,
                "source_sha256": digest,
            }
        )
    own_path = Path(__file__)
    descriptor = os.open(
        own_path,
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        own_digest, _ = worker._stable_file_hash_from_fd(
            descriptor, maximum=4 * 1024 * 1024
        )
    finally:
        os.close(descriptor)
    rows.append(
        {"module": __name__, "source_sha256": own_digest}
    )
    return contract.semantic_sha256(rows)


def _critical_dependency_closure() -> dict[str, object]:
    """Bind the imported non-Torch dependencies that affect model execution."""

    rows: list[dict[str, str]] = []
    for module_name, distribution_name in (
        CRITICAL_DEPENDENCY_DISTRIBUTIONS
    ):
        try:
            module = import_module(module_name)
            version = importlib_metadata.version(distribution_name)
        except (
            ImportError,
            importlib_metadata.PackageNotFoundError,
        ) as exc:
            raise closed.ClosedChoiceError(
                "closed_choice_critical_dependency_unavailable"
            ) from exc
        origin = getattr(module, "__file__", None)
        if (
            not isinstance(origin, str)
            or not origin
            or not isinstance(version, str)
            or not version
        ):
            raise closed.ClosedChoiceError(
                "closed_choice_critical_dependency_origin_unavailable"
            )
        try:
            distribution_sha256 = (
                worker._distribution_closure_sha256(
                    distribution_name,
                    required_module_origins=(Path(origin),),
                )
            )
        except contract.NarrativeExtractorRuntimeError:
            raise
        except Exception as exc:
            raise closed.ClosedChoiceError(
                "closed_choice_critical_dependency_hash_failed"
            ) from exc
        rows.append(
            {
                "distribution": distribution_name,
                "distribution_sha256": distribution_sha256,
                "module": module_name,
                "version": version,
            }
        )
    body: dict[str, object] = {
        "dependencies": rows,
        "schema": (
            "gscl_narrative_closed_choice_critical_dependencies_v1"
        ),
    }
    return {
        **body,
        "self_sha256": contract.semantic_sha256(body),
    }


def _decode_receipt(raw: bytes) -> Mapping[str, object]:
    value = json.loads(raw.decode("ascii"))
    if type(value) is not dict:
        raise closed.ClosedChoiceError(
            "closed_choice_runtime_receipt_invalid"
        )
    return MappingProxyType(value)


class _ExactTeacherForcedBackend:
    """Ephemeral adapter carrying a private per-selection nonce."""

    __slots__ = ("_marker", "_nonce", "_owner")

    def __init__(
        self,
        owner: "LocalQwenClosedChoiceRuntime",
        nonce: object,
        marker: object,
    ) -> None:
        if (
            marker is not _BACKEND_MARKER
            or type(owner) is not LocalQwenClosedChoiceRuntime
        ):
            raise closed.ClosedChoiceError(
                "closed_choice_backend_authority_invalid"
            )
        self._marker = marker
        self._nonce = nonce
        self._owner = owner

    @property
    def runtime_commitment(self) -> str:
        self._require_active()
        return self._owner._teacher_forced_backend_commitment

    def _require_active(self) -> None:
        if (
            type(self) is not _ExactTeacherForcedBackend
            or self._marker is not _BACKEND_MARKER
            or type(self._owner) is not LocalQwenClosedChoiceRuntime
            or self._owner._active_nonce is not self._nonce
        ):
            raise closed.ClosedChoiceError(
                "closed_choice_backend_authority_invalid"
            )

    def score_batch(
        self, pairs: tuple[closed.PromptAnswer, ...]
    ) -> tuple[closed.TeacherForcedScore, ...]:
        self._require_active()
        return LocalQwenClosedChoiceRuntime._score_batch_bound(
            self._owner,
            pairs,
            nonce=self._nonce,
        )

    def count_program_owned_completion_tokens(
        self, completion: str
    ) -> int:
        self._require_active()
        return (
            LocalQwenClosedChoiceRuntime._count_completion_tokens_bound(
                self._owner,
                completion,
                nonce=self._nonce,
            )
        )


class LocalQwenClosedChoiceRuntime:
    """Manifest-bound fp16 CUDA runtime with no free-generation surface."""

    __slots__ = (
        "_active_nonce",
        "_double_run_receipt_bytes",
        "_double_run_receipt_sha256",
        "_manifest",
        "_marker",
        "_model",
        "_model_runtime_closure_sha256",
        "_parameter_binding",
        "_runtime_receipt_bytes",
        "_runtime_receipt_sha256",
        "_selection_receipt_commitments",
        "_teacher_forced_backend_commitment",
        "_tokenizer",
        "_torch",
        "_transformers",
        "execution_closure",
    )

    def __init__(
        self,
        *,
        model_root: Path,
        manifest: worker.ModelAssetManifest,
    ) -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if (
            type(manifest) is not worker.ModelAssetManifest
            or manifest._marker is not worker._VERIFIED_MANIFEST_MARKER
        ):
            raise closed.ClosedChoiceError(
                "closed_choice_model_manifest_not_verified"
            )
        try:
            import torch
            import transformers
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise closed.ClosedChoiceError(
                "closed_choice_local_model_runtime_unavailable"
            ) from exc
        if not torch.cuda.is_available():
            raise closed.ClosedChoiceError(
                "closed_choice_cuda_runtime_unavailable"
            )
        torch.manual_seed(TORCH_SEED)
        torch.cuda.manual_seed_all(TORCH_SEED)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_root,
                local_files_only=True,
                trust_remote_code=False,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_root,
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=torch.float16,
                use_safetensors=True,
                attn_implementation=manifest.declarations[
                    "attention_implementation"
                ],
            ).to(DEVICE)
        except Exception as exc:
            raise closed.ClosedChoiceError(
                "closed_choice_local_model_load_failed"
            ) from exc
        model.eval()

        self._torch = torch
        self._transformers = transformers
        self._tokenizer = tokenizer
        self._model = model
        self._manifest = manifest
        self._active_nonce = None
        self._selection_receipt_commitments: list[str] = []
        self._marker = _CONSTRUCTING_MARKER
        self._parameter_binding = self._capture_parameter_binding()

        runtime_receipt = self._runtime_receipt_payload()
        if (
            runtime_receipt["environment"]
            != dict(manifest.runtime_requirements)
            or self._loaded_declarations()
            != dict(manifest.declarations)
        ):
            raise closed.ClosedChoiceError(
                "closed_choice_runtime_environment_drifted"
            )
        self._runtime_receipt_bytes = contract.canonical_json_bytes(
            runtime_receipt
        )
        self._runtime_receipt_sha256 = hashlib.sha256(
            self._runtime_receipt_bytes
        ).hexdigest()
        self._teacher_forced_backend_commitment = (
            contract.semantic_sha256(
                {
                    "model_asset_manifest_sha256": (
                        manifest.manifest_file_sha256
                    ),
                    "prompt_closure_sha256": (
                        closed.PROMPT_CLOSURE_SHA256
                    ),
                    "runtime_receipt_sha256": (
                        self._runtime_receipt_sha256
                    ),
                    "scoring_policy_sha256": contract.semantic_sha256(
                        dict(closed.SCORING_POLICY)
                    ),
                }
            )
        )

        first = self._select_story(
            worker.DETERMINISM_CANARY_STORY,
            record=False,
        )
        second = self._select_story(
            worker.DETERMINISM_CANARY_STORY,
            record=False,
        )
        if first != second:
            raise closed.ClosedChoiceError(
                "closed_choice_target_double_run_not_exact"
            )
        double_body: dict[str, object] = {
            "canonical_completion_commitment": hashlib.sha256(
                first.canonical_completion.encode("utf-8")
            ).hexdigest(),
            "free_form_generation_count": 0,
            "model_asset_manifest_sha256": (
                manifest.manifest_file_sha256
            ),
            "prompt_closure_sha256": (
                closed.PROMPT_CLOSURE_SHA256
            ),
            "repeat_count": 2,
            "repeat_exact": True,
            "runtime_receipt_sha256": (
                self._runtime_receipt_sha256
            ),
            "schema": DOUBLE_RUN_RECEIPT_SCHEMA,
            "selection_receipt_commitment": hashlib.sha256(
                first.receipt_bytes
            ).hexdigest(),
            "story_commitment": hashlib.sha256(
                worker.DETERMINISM_CANARY_STORY.encode("utf-8")
            ).hexdigest(),
            "wire_completion_token_count": (
                first.wire_completion_token_count
            ),
        }
        double_receipt = {
            **double_body,
            "self_sha256": contract.semantic_sha256(double_body),
        }
        self._double_run_receipt_bytes = (
            contract.canonical_json_bytes(double_receipt)
        )
        self._double_run_receipt_sha256 = hashlib.sha256(
            self._double_run_receipt_bytes
        ).hexdigest()
        self._model_runtime_closure_sha256 = (
            contract.semantic_sha256(
                {
                    "double_run_receipt_sha256": (
                        self._double_run_receipt_sha256
                    ),
                    "model_asset_manifest_sha256": (
                        manifest.manifest_file_sha256
                    ),
                    "runtime_receipt_sha256": (
                        self._runtime_receipt_sha256
                    ),
                    "teacher_forced_backend_commitment": (
                        self._teacher_forced_backend_commitment
                    ),
                }
            )
        )
        self.execution_closure = contract.ExecutionClosure(
            prompt_sha256=closed.PROMPT_CLOSURE_SHA256,
            parser_closure_sha256=worker._parser_closure_sha256(),
            model_asset_manifest_sha256=(
                manifest.manifest_file_sha256
            ),
            model_runtime_closure_sha256=(
                self._model_runtime_closure_sha256
            ),
            target_double_run_receipt_sha256=(
                self._double_run_receipt_sha256
            ),
        )
        self._marker = _VERIFIED_RUNTIME_MARKER
        self._validate_formal_binding()

    @property
    def runtime_receipt(self) -> Mapping[str, object]:
        return _decode_receipt(self._runtime_receipt_bytes)

    @property
    def target_double_run_receipt(self) -> Mapping[str, object]:
        return _decode_receipt(self._double_run_receipt_bytes)

    @property
    def selection_receipt_commitments(self) -> tuple[str, ...]:
        self._validate_formal_binding()
        return tuple(self._selection_receipt_commitments)

    def _capture_parameter_binding(
        self,
    ) -> tuple[tuple[object, ...], ...]:
        return tuple(
            (
                name,
                int(parameter.data_ptr()),
                int(parameter._version),
                tuple(int(value) for value in parameter.shape),
                str(parameter.dtype),
                str(parameter.device),
            )
            for name, parameter in self._model.named_parameters()
        )

    def _context_limit(self) -> int:
        values = [
            value
            for value in (
                getattr(
                    self._model.config,
                    "max_position_embeddings",
                    None,
                ),
                getattr(
                    self._tokenizer, "model_max_length", None
                ),
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

    def _runtime_environment(self) -> dict[str, object]:
        torch = self._torch
        torch_origin = getattr(torch, "__file__", None)
        transformers_origin = getattr(
            self._transformers, "__file__", None
        )
        if (
            not isinstance(torch_origin, str)
            or not torch_origin
            or not isinstance(transformers_origin, str)
            or not transformers_origin
        ):
            raise closed.ClosedChoiceError(
                "closed_choice_runtime_module_origin_unavailable"
            )
        cudnn_version = torch.backends.cudnn.version()
        if not isinstance(cudnn_version, int):
            raise closed.ClosedChoiceError(
                "closed_choice_cudnn_version_unavailable"
            )
        capability = torch.cuda.get_device_capability(0)
        return {
            "attention_implementation": str(
                getattr(
                    self._model.config,
                    "_attn_implementation",
                    None,
                )
            ),
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
                    required_module_origins=(
                        Path(torch_origin),
                    ),
                )
            ),
            "transformers_version": str(
                self._transformers.__version__
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

    def _lightweight_runtime_environment(
        self,
    ) -> dict[str, object]:
        """Recheck live state without re-hashing multi-GB distributions."""

        torch = self._torch
        cudnn_version = torch.backends.cudnn.version()
        if not isinstance(cudnn_version, int):
            raise closed.ClosedChoiceError(
                "closed_choice_cudnn_version_unavailable"
            )
        capability = torch.cuda.get_device_capability(0)
        return {
            "attention_implementation": str(
                getattr(
                    self._model.config,
                    "_attn_implementation",
                    None,
                )
            ),
            "cuda_version": str(torch.version.cuda),
            "cudnn_version": cudnn_version,
            "gpu_compute_capability": [
                int(capability[0]),
                int(capability[1]),
            ],
            "gpu_name": str(torch.cuda.get_device_name(0)),
            "python_implementation": (
                platform.python_implementation()
            ),
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "transformers_version": str(
                self._transformers.__version__
            ),
        }

    def _loaded_declarations(self) -> dict[str, object]:
        chat_template = getattr(
            self._tokenizer, "chat_template", None
        )
        if not isinstance(chat_template, str) or not chat_template:
            raise closed.ClosedChoiceError(
                "closed_choice_chat_template_unavailable"
            )
        return {
            "attention_implementation": str(
                getattr(
                    self._model.config,
                    "_attn_implementation",
                    None,
                )
            ),
            "chat_template_sha256": hashlib.sha256(
                chat_template.encode("utf-8")
            ).hexdigest(),
            "context_limit": self._context_limit(),
            "critical_config": {
                key: getattr(self._model.config, key, None)
                for key in worker.QWEN_ARCHITECTURE
            },
            "loaded_config_sha256": contract.semantic_sha256(
                _config_projection(self._model.config)
            ),
            "model_class": self._model.__class__.__name__,
            "special_token_ids": {
                "bos_token_id": self._tokenizer.bos_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
                "pad_token_id": self._tokenizer.pad_token_id,
            },
            "tokenizer_class": (
                self._tokenizer.__class__.__name__
            ),
        }

    def _runtime_receipt_payload(self) -> dict[str, object]:
        return {
            "critical_dependency_closure": (
                _critical_dependency_closure()
            ),
            "deterministic_algorithms": True,
            "device": DEVICE,
            "dtype": "float16",
            "environment": self._runtime_environment(),
            "formal_scoring_batch_size": (
                FORMAL_SCORING_BATCH_SIZE
            ),
            "free_form_generation_count": 0,
            "local_files_only": True,
            "maximum_context_tokens": (
                closed.MAXIMUM_CONTEXT_TOKENS
            ),
            "runtime_code_closure_sha256": (
                _source_closure_sha256()
            ),
            "schema": RUNTIME_RECEIPT_SCHEMA,
            "score_operation": (
                "teacher_forced_forward_log_softmax"
            ),
            "seed": TORCH_SEED,
            "tf32": False,
            "trust_remote_code": False,
        }

    def _validate_formal_binding(
        self, *, constructing: bool = False
    ) -> None:
        expected_marker = (
            _CONSTRUCTING_MARKER
            if constructing
            else _VERIFIED_RUNTIME_MARKER
        )
        runtime_receipt = _decode_receipt(
            self._runtime_receipt_bytes
        )
        recorded_environment = runtime_receipt.get("environment")
        if not isinstance(recorded_environment, Mapping):
            raise closed.ClosedChoiceError(
                "closed_choice_formal_runtime_binding_drifted"
            )
        lightweight_environment = (
            self._lightweight_runtime_environment()
        )
        if (
            type(self) is not LocalQwenClosedChoiceRuntime
            or self._marker is not expected_marker
            or type(self._manifest) is not worker.ModelAssetManifest
            or self._manifest._marker
            is not worker._VERIFIED_MANIFEST_MARKER
            or self._active_nonce is not None
            or self._capture_parameter_binding()
            != self._parameter_binding
            or any(
                parameter.device.type != "cuda"
                or (
                    parameter.is_floating_point()
                    and parameter.dtype != self._torch.float16
                )
                for parameter in self._model.parameters()
            )
            or self._torch.backends.cuda.matmul.allow_tf32
            or self._torch.backends.cudnn.allow_tf32
            or not self._torch.are_deterministic_algorithms_enabled()
            or self._loaded_declarations()
            != dict(self._manifest.declarations)
            or any(
                recorded_environment.get(key) != value
                for key, value in lightweight_environment.items()
            )
            or runtime_receipt.get(
                "runtime_code_closure_sha256"
            )
            != _source_closure_sha256()
            or runtime_receipt.get("free_form_generation_count")
            != 0
            or runtime_receipt.get("score_operation")
            != "teacher_forced_forward_log_softmax"
            or hashlib.sha256(
                self._runtime_receipt_bytes
            ).hexdigest()
            != self._runtime_receipt_sha256
        ):
            raise closed.ClosedChoiceError(
                "closed_choice_formal_runtime_binding_drifted"
            )
        if constructing:
            return
        expected_model_runtime = contract.semantic_sha256(
            {
                "double_run_receipt_sha256": (
                    self._double_run_receipt_sha256
                ),
                "model_asset_manifest_sha256": (
                    self._manifest.manifest_file_sha256
                ),
                "runtime_receipt_sha256": (
                    self._runtime_receipt_sha256
                ),
                "teacher_forced_backend_commitment": (
                    self._teacher_forced_backend_commitment
                ),
            }
        )
        if (
            hashlib.sha256(
                self._double_run_receipt_bytes
            ).hexdigest()
            != self._double_run_receipt_sha256
            or expected_model_runtime
            != self._model_runtime_closure_sha256
            or self.execution_closure.prompt_sha256
            != closed.PROMPT_CLOSURE_SHA256
            or self.execution_closure.parser_closure_sha256
            != worker._parser_closure_sha256()
            or (
                self.execution_closure
                .model_asset_manifest_sha256
            )
            != self._manifest.manifest_file_sha256
            or (
                self.execution_closure
                .model_runtime_closure_sha256
            )
            != expected_model_runtime
            or (
                self.execution_closure
                .target_double_run_receipt_sha256
            )
            != self._double_run_receipt_sha256
        ):
            raise closed.ClosedChoiceError(
                "closed_choice_formal_runtime_binding_drifted"
            )

    def _token_ids(self, text: str) -> list[int]:
        try:
            encoded = self._tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                truncation=False,
            )
            values = encoded["input_ids"]
        except Exception as exc:
            raise closed.ClosedChoiceError(
                "closed_choice_tokenizer_failed"
            ) from exc
        if (
            type(values) is not list
            or not values
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in values
            )
        ):
            raise closed.ClosedChoiceError(
                "closed_choice_tokenizer_output_invalid"
            )
        return values

    def _prompt_answer_token_ids(
        self, pair: closed.PromptAnswer
    ) -> tuple[list[int], list[int], list[int]]:
        """Prove that the separately tokenized boundary is text-exact."""

        if type(pair) is not closed.PromptAnswer:
            raise closed.ClosedChoiceError(
                "closed_choice_prompt_answer_invalid"
            )
        prompt = self._token_ids(pair.prompt)
        answer = self._token_ids(pair.answer)
        sequence = prompt + answer
        combined = self._token_ids(pair.prompt + pair.answer)
        if sequence != combined:
            raise closed.ClosedChoiceError(
                "closed_choice_token_boundary_invalid"
            )
        return prompt, answer, sequence

    def _score_batch_bound(
        self,
        pairs: tuple[closed.PromptAnswer, ...],
        *,
        nonce: object,
    ) -> tuple[closed.TeacherForcedScore, ...]:
        if (
            self._active_nonce is not nonce
            or type(pairs) is not tuple
            or not 1 <= len(pairs) <= FORMAL_SCORING_BATCH_SIZE
            or any(type(pair) is not closed.PromptAnswer for pair in pairs)
        ):
            raise closed.ClosedChoiceError(
                "closed_choice_score_authority_invalid"
            )
        prompt_ids: list[list[int]] = []
        answer_ids: list[list[int]] = []
        combined: list[list[int]] = []
        for pair in pairs:
            prompt, answer, sequence = (
                self._prompt_answer_token_ids(pair)
            )
            if (
                len(sequence) > closed.MAXIMUM_CONTEXT_TOKENS
                or len(sequence) > self._context_limit()
            ):
                raise closed.ClosedChoiceAbstention(
                    "closed_choice_context_too_long",
                    pre_model=False,
                )
            prompt_ids.append(prompt)
            answer_ids.append(answer)
            combined.append(sequence)

        pad_token = self._tokenizer.pad_token_id
        if pad_token is None:
            pad_token = self._tokenizer.eos_token_id
        if (
            isinstance(pad_token, bool)
            or not isinstance(pad_token, int)
            or pad_token < 0
        ):
            raise closed.ClosedChoiceError(
                "closed_choice_pad_token_invalid"
            )
        maximum = max(len(row) for row in combined)
        input_rows = [
            row + [pad_token] * (maximum - len(row))
            for row in combined
        ]
        mask_rows = [
            [1] * len(row) + [0] * (maximum - len(row))
            for row in combined
        ]
        torch = self._torch
        input_tensor = torch.tensor(
            input_rows, dtype=torch.long, device=DEVICE
        )
        mask_tensor = torch.tensor(
            mask_rows, dtype=torch.long, device=DEVICE
        )
        torch.manual_seed(TORCH_SEED)
        torch.cuda.manual_seed_all(TORCH_SEED)
        try:
            with torch.inference_mode():
                output = self._model(
                    input_ids=input_tensor,
                    attention_mask=mask_tensor,
                    use_cache=False,
                    return_dict=True,
                )
                logits = output.logits
                scores: list[closed.TeacherForcedScore] = []
                for row_index, answer in enumerate(answer_ids):
                    prompt_count = len(prompt_ids[row_index])
                    positions = torch.arange(
                        prompt_count - 1,
                        prompt_count + len(answer) - 1,
                        device=DEVICE,
                    )
                    selected_logits = logits[
                        row_index, positions, :
                    ].float()
                    targets = torch.tensor(
                        answer, dtype=torch.long, device=DEVICE
                    )
                    target_logits = selected_logits.gather(
                        1, targets.unsqueeze(1)
                    ).squeeze(1)
                    log_probabilities = target_logits - torch.logsumexp(
                        selected_logits, dim=1
                    )
                    total = float(
                        log_probabilities.sum().cpu().item()
                    )
                    quantized = round(
                        total * closed.LOGPROB_QUANTIZATION_SCALE
                    )
                    scores.append(
                        closed.TeacherForcedScore(
                            total_logprob_microunits=quantized,
                            answer_token_count=len(answer),
                            context_and_answer_token_count=len(
                                combined[row_index]
                            ),
                        )
                    )
        except closed.ClosedChoiceError:
            raise
        except Exception as exc:
            raise closed.ClosedChoiceError(
                "closed_choice_model_forward_failed"
            ) from exc
        return tuple(scores)

    def _count_completion_tokens_bound(
        self, completion: str, *, nonce: object
    ) -> int:
        if self._active_nonce is not nonce:
            raise closed.ClosedChoiceError(
                "closed_choice_tokenizer_authority_invalid"
            )
        return len(self._token_ids(completion))

    def _select_story(
        self, story_text: str, *, record: bool
    ) -> closed.ClosedChoiceDecision:
        if self._active_nonce is not None:
            raise closed.ClosedChoiceError(
                "closed_choice_runtime_reentrant"
            )
        self._validate_formal_binding(
            constructing=self._marker is _CONSTRUCTING_MARKER
        )
        nonce = object()
        self._active_nonce = nonce
        backend = _ExactTeacherForcedBackend(
            self, nonce, _BACKEND_MARKER
        )
        try:
            engine = closed._ClosedChoiceEngine(
                closed._ENGINE_MARKER
            )
            decision = engine.select(
                story_text,
                backend=backend,
                narrative_parser=worker._independent_parser,
                scoring_batch_size=FORMAL_SCORING_BATCH_SIZE,
            )
        finally:
            self._active_nonce = None
            # Revalidate even when tokenization, scoring, or parsing raises.
            # A failed item cannot bypass the post-call custody check.
            self._validate_formal_binding(
                constructing=(
                    self._marker is _CONSTRUCTING_MARKER
                )
            )
        if record:
            self._selection_receipt_commitments.append(
                hashlib.sha256(decision.receipt_bytes).hexdigest()
            )
        return decision


def process_formal_pack(
    pack: contract.StoryOnlyInputPack,
    *,
    runtime: LocalQwenClosedChoiceRuntime,
) -> list[dict[str, object]]:
    """Exact formal ABI: admitted pack plus exact bound runtime only."""

    trusted = contract.require_formal_story_only_pack(pack)
    if type(runtime) is not LocalQwenClosedChoiceRuntime:
        raise closed.ClosedChoiceError(
            "closed_choice_formal_runtime_not_verified"
        )
    runtime._validate_formal_binding()
    results: list[dict[str, object]] = []
    for request, story_commitment in zip(
        trusted.requests, trusted.story_commitments
    ):
        try:
            decision = (
                LocalQwenClosedChoiceRuntime._select_story(
                    runtime,
                    request.story_text,
                    record=True,
                )
            )
        except closed.ClosedChoiceAbstention as exc:
            results.append(
                contract.invalid_result(
                    ordinal=request.ordinal,
                    story_commitment=story_commitment,
                    error_code=(
                        "SPAN_CATALOG_UNAVAILABLE"
                        if exc.pre_model
                        else "INPUT_TOO_LONG"
                    ),
                )
            )
            continue
        except (
            closed.ClosedChoiceError,
            contract.NarrativeExtractorRuntimeError,
        ):
            results.append(
                contract.invalid_result(
                    ordinal=request.ordinal,
                    story_commitment=story_commitment,
                    error_code="MODEL_RUNTIME_ERROR",
                )
            )
            continue
        results.append(
            contract.valid_result(
                ordinal=request.ordinal,
                story_commitment=story_commitment,
                completion=decision.canonical_completion,
                completion_token_count=(
                    decision.wire_completion_token_count
                ),
                wire_completion_sha256=hashlib.sha256(
                    decision.completion.encode("utf-8")
                ).hexdigest(),
            )
        )
    runtime._validate_formal_binding()
    return results


__all__ = [
    "DOUBLE_RUN_RECEIPT_SCHEMA",
    "FORMAL_SCORING_BATCH_SIZE",
    "LocalQwenClosedChoiceRuntime",
    "RUNTIME_RECEIPT_SCHEMA",
    "VERSION",
    "process_formal_pack",
]
