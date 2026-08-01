"""Memory-bounded teacher-forced Qwen conditional likelihood.

The frozen v1 implementation asks the causal-LM head for logits at every
sequence position.  For Qwen's large vocabulary, the resulting
``[batch, full_sequence, vocabulary]`` tensor is the dominant transient
allocation even though only answer-prediction positions are scored.

This independent v2 component preserves the v1 scoring definition:

* prompt and answer are tokenized separately and their boundary is proved
  exact;
* every answer token is scored under teacher forcing;
* fp16 logits are converted to fp32 before ``target - logsumexp``;
* the sum is quantized with Python's round-to-even at the v1 scale.

The primary strategy passes the union of required sequence indices through
Qwen/Transformers' tensor ``logits_to_keep`` API and rejects any output whose
shape is not exactly ``[batch, requested_positions, vocabulary]``.  When that
API is absent, the frozen fallback calls the Qwen base decoder with a KV cache
and applies ``lm_head`` only to one final hidden position per step.  It groups
finite candidates by prompt and answer length, so the fallback remains
batched and never materializes prompt-length vocabulary logits.

The public fake builder and arbitrary-pair method are explicitly
qualification-only.  The exact runtime is loaded from a verified v1 asset
manifest, owns its model, exposes no arbitrary-pair scoring method, and can
only run the program-owned closed-choice engine through ``select_story``.
Python markers prevent accidental API substitution; the containing fixed,
hashed process remains the actual adversarial trust boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from replication_runtime.gscl_narrative_extractor_v1 import (
    closed_choice_worker as closed,
)
from replication_runtime.gscl_narrative_extractor_v1 import contract
from replication_runtime.gscl_narrative_extractor_v1 import worker

from . import closed_choice as hierarchical
from .contract import (
    ClosedChoiceV2Abstention,
    ClosedChoiceV2Error,
)


VERSION = "gscl_narrative_extractor_v2_memory_safe_qwen_v1"
FIXED_CANARY_SCHEMA = (
    "gscl_narrative_extractor_v2_fixed_teacher_forced_canary_v1"
)
DEVICE = "cuda:0"
DTYPE_NAME = "float16"
SCORING_BATCH_SIZE = 4
# Preserve v2's admitted answer-token domain.  Sparse positions are chunked
# below, rather than silently inheriting v1's obsolete 512-token ceiling.
MAXIMUM_ANSWER_TOKENS = hierarchical.MAXIMUM_CONTEXT_TOKENS
MAXIMUM_SPARSE_POSITIONS = 128
MAXIMUM_VOCABULARY_SIZE = 262_144
MAXIMUM_SPARSE_LOGIT_ELEMENTS = 64_000_000
MAXIMUM_FALLBACK_GROUPS = SCORING_BATCH_SIZE
TORCH_SEED = worker.TORCH_SEED

SPARSE_STRATEGY = "tensor_logits_to_keep"
FALLBACK_STRATEGY = "grouped_incremental_kv_cache"
SCORING_SEMANTICS = MappingProxyType(
    {
        "answer_boundary": (
            "tokenize(prompt)+tokenize(answer)==tokenize(prompt+answer)"
        ),
        "fallback": FALLBACK_STRATEGY,
        "free_form_generation_count": 0,
        "logit_accumulation_dtype": "float32",
        "logprob_quantization_scale": (
            closed.LOGPROB_QUANTIZATION_SCALE
        ),
        "maximum_answer_tokens": MAXIMUM_ANSWER_TOKENS,
        "maximum_batch_size": SCORING_BATCH_SIZE,
        "maximum_context_tokens": (
            hierarchical.MAXIMUM_CONTEXT_TOKENS
        ),
        "maximum_sparse_logit_elements": (
            MAXIMUM_SPARSE_LOGIT_ELEMENTS
        ),
        "maximum_sparse_positions_per_forward": (
            MAXIMUM_SPARSE_POSITIONS
        ),
        "maximum_vocabulary_size": MAXIMUM_VOCABULARY_SIZE,
        "primary": SPARSE_STRATEGY,
        "score_operation": (
            "teacher_forced_target_minus_logsumexp"
        ),
        "version": VERSION,
    }
)

_FIXED_SHORT_CANARY_PAIR = closed.PromptAnswer(
    candidate_key="fixed_public_canary.short.polarity",
    prompt="Fixed public canary:\n",
    answer="polarity=positive",
)
_FIXED_LONG_CANARY_PAIR = closed.PromptAnswer(
    candidate_key="fixed_public_canary.long.chunked",
    prompt=(
        "Treat this as a fixed public teacher-forced chunking canary.\n"
        "Candidate completion:\n"
    ),
    answer=" a" * 130,
)


def _fixed_pair_commitment(pair: closed.PromptAnswer) -> str:
    return contract.semantic_sha256(
        {
            "answer": pair.answer,
            "candidate_key": pair.candidate_key,
            "prompt": pair.prompt,
        }
    )


FIXED_SHORT_CANARY_PAIR_SHA256 = _fixed_pair_commitment(
    _FIXED_SHORT_CANARY_PAIR
)
FIXED_LONG_CANARY_PAIR_SHA256 = _fixed_pair_commitment(
    _FIXED_LONG_CANARY_PAIR
)


def _fixed_canary_closure_is_frozen() -> bool:
    return (
        type(_FIXED_SHORT_CANARY_PAIR) is closed.PromptAnswer
        and type(_FIXED_LONG_CANARY_PAIR) is closed.PromptAnswer
        and _fixed_pair_commitment(_FIXED_SHORT_CANARY_PAIR)
        == FIXED_SHORT_CANARY_PAIR_SHA256
        and _fixed_pair_commitment(_FIXED_LONG_CANARY_PAIR)
        == FIXED_LONG_CANARY_PAIR_SHA256
    )

_CONSTRUCTION_MARKER = object()
_QUALIFICATION_RUNTIME_MARKER = object()
_EXACT_RUNTIME_MARKER = object()
_BACKEND_MARKER = object()


class MemorySafeQwenError(closed.ClosedChoiceError):
    """A memory-safe scorer contract or custody violation."""


@dataclass(frozen=True, slots=True)
class _EncodedPair:
    prompt: tuple[int, ...]
    answer: tuple[int, ...]
    combined: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not self.prompt
            or not self.answer
            or self.combined != self.prompt + self.answer
        ):
            raise MemorySafeQwenError(
                "memory_safe_encoded_pair_invalid"
            )


def _explicit_tensor_logits_to_keep(
    model: object, *, exact: bool = False
) -> bool:
    """Return true only for an explicit tensor-capable forward parameter.

    Qualification fakes may leave the annotation empty.  Exact runtimes
    require the pinned Qwen method's annotation to mention ``Tensor``; a
    generic ``**kwargs`` is not treated as proof of sparse-logit support.
    """

    try:
        parameter = inspect.signature(model.forward).parameters.get(
            "logits_to_keep"
        )
    except (AttributeError, TypeError, ValueError):
        return False
    if parameter is None:
        return False
    annotation = parameter.annotation
    return (
        (
            not exact
            and annotation is inspect.Parameter.empty
        )
        or "Tensor" in str(annotation)
    )


def _parameter_binding(model: object) -> tuple[tuple[object, ...], ...]:
    try:
        parameters = model.named_parameters()
    except AttributeError as exc:
        raise MemorySafeQwenError(
            "memory_safe_model_parameters_unavailable"
        ) from exc
    rows: list[tuple[object, ...]] = []
    for name, parameter in parameters:
        rows.append(
            (
                name,
                int(parameter.data_ptr()),
                int(parameter._version),
                tuple(int(value) for value in parameter.shape),
                str(parameter.dtype),
                str(parameter.device),
            )
        )
    return tuple(rows)


def _callable_binding(
    model: object, tokenizer: object
) -> tuple[tuple[str, int], ...]:
    """Bind Python dispatch surfaces in addition to parameter storage."""

    candidates = (
        ("model_type", type(model)),
        ("model_forward", getattr(type(model), "forward", None)),
        ("tokenizer_type", type(tokenizer)),
        (
            "decoder_type",
            type(getattr(model, "model", None)),
        ),
        (
            "decoder_forward",
            getattr(
                type(getattr(model, "model", None)),
                "forward",
                None,
            ),
        ),
        (
            "lm_head_type",
            type(getattr(model, "lm_head", None)),
        ),
        (
            "lm_head_forward",
            getattr(
                type(getattr(model, "lm_head", None)),
                "forward",
                None,
            ),
        ),
    )
    return tuple((name, id(value)) for name, value in candidates)


def _source_sha256() -> str:
    try:
        raw = Path(__file__).read_bytes()
    except OSError as exc:
        raise MemorySafeQwenError(
            "memory_safe_source_closure_unavailable"
        ) from exc
    if len(raw) > 4 * 1024 * 1024:
        raise MemorySafeQwenError(
            "memory_safe_source_closure_oversized"
        )
    return hashlib.sha256(raw).hexdigest()


def _resource_policy_is_frozen() -> bool:
    return (
        SCORING_BATCH_SIZE
        == SCORING_SEMANTICS["maximum_batch_size"]
        and MAXIMUM_ANSWER_TOKENS
        == SCORING_SEMANTICS["maximum_answer_tokens"]
        and MAXIMUM_SPARSE_POSITIONS
        == SCORING_SEMANTICS[
            "maximum_sparse_positions_per_forward"
        ]
        and MAXIMUM_VOCABULARY_SIZE
        == SCORING_SEMANTICS["maximum_vocabulary_size"]
        and MAXIMUM_SPARSE_LOGIT_ELEMENTS
        == SCORING_SEMANTICS[
            "maximum_sparse_logit_elements"
        ]
    )


class _PrivateTeacherForcedBackend:
    """Ephemeral backend tied to exactly one active runtime nonce."""

    __slots__ = ("_marker", "_nonce", "_owner", "_v2_errors")

    def __init__(
        self,
        owner: "MemorySafeQwenRuntime",
        nonce: object,
        marker: object,
        *,
        v2_errors: bool = False,
    ) -> None:
        if (
            marker is not _BACKEND_MARKER
            or type(owner) is not MemorySafeQwenRuntime
            or type(v2_errors) is not bool
        ):
            raise MemorySafeQwenError(
                "memory_safe_backend_authority_invalid"
            )
        self._marker = marker
        self._nonce = nonce
        self._owner = owner
        self._v2_errors = v2_errors

    def _require_active(self) -> None:
        if (
            type(self) is not _PrivateTeacherForcedBackend
            or self._marker is not _BACKEND_MARKER
            or type(self._owner) is not MemorySafeQwenRuntime
            or self._owner._active_nonce is not self._nonce
        ):
            raise MemorySafeQwenError(
                "memory_safe_backend_authority_invalid"
            )

    @property
    def runtime_commitment(self) -> str:
        self._require_active()
        # Calling the public runtime property here would reject the deliberately
        # active nonce.  The backend has already proved that it is the unique
        # active session, so it reads the immutable commitment directly.
        return self._owner._runtime_commitment

    def score_batch(
        self,
        pairs: tuple[closed.PromptAnswer, ...],
    ) -> tuple[hierarchical.TeacherForcedScore, ...]:
        self._require_active()
        try:
            return MemorySafeQwenRuntime._score_batch_bound(
                self._owner,
                pairs,
                nonce=self._nonce,
            )
        except closed.ClosedChoiceAbstention as exc:
            if not self._v2_errors:
                raise
            raise ClosedChoiceV2Abstention(
                "V2_CONTEXT_TOKEN_LIMIT_EXCEEDED",
                before_model_forward=bool(
                    getattr(exc, "pre_model", False)
                ),
            ) from exc
        except MemorySafeQwenError as exc:
            if not self._v2_errors:
                raise
            issue = str(getattr(exc, "issue_id", exc))
            if "cuda" in issue:
                mapped = "V2_CUDA_RUNTIME_UNAVAILABLE"
            elif (
                "token" in issue
                or "boundary" in issue
                or "pad_" in issue
            ):
                mapped = "V2_TOKEN_BOUNDARY_INVALID"
            elif (
                "authority" in issue
                or "binding" in issue
                or "manifest" in issue
                or "configuration" in issue
            ):
                mapped = "V2_AUTHORITY_INVALID"
            elif (
                "context" in issue
                or "resource_bound" in issue
                or "group_bound" in issue
            ):
                raise ClosedChoiceV2Abstention(
                    "V2_CONTEXT_TOKEN_LIMIT_EXCEEDED",
                    before_model_forward=False,
                ) from exc
            else:
                mapped = "V2_MODEL_FORWARD_FAILED"
            raise ClosedChoiceV2Error(mapped) from exc

    def count_program_owned_completion_tokens(
        self, completion: str
    ) -> int:
        self._require_active()
        try:
            return len(
                MemorySafeQwenRuntime._token_ids(
                    self._owner, completion
                )
            )
        except MemorySafeQwenError as exc:
            if not self._v2_errors:
                raise
            raise ClosedChoiceV2Error(
                "V2_TOKEN_BOUNDARY_INVALID"
            ) from exc


class MemorySafeQwenRuntime:
    """Exact-type, non-generative, bounded Qwen scoring runtime."""

    __slots__ = (
        "_active_nonce",
        "_callable_binding_value",
        "_context_limit_value",
        "_device",
        "_exact",
        "_expected_declarations",
        "_last_sparse_chunk_count",
        "_manifest_commitment",
        "_marker",
        "_model",
        "_parameter_binding_value",
        "_runtime_commitment",
        "_source_sha256_value",
        "_strategy",
        "_tokenizer",
        "_torch",
        "_vocabulary_size",
    )

    def __init__(
        self,
        *,
        model: object,
        tokenizer: object,
        torch_module: object,
        device: str,
        exact: bool,
        manifest_commitment: str,
        strategy: str,
        expected_declarations: Mapping[str, object] | None,
        marker: object,
    ) -> None:
        if marker is not _CONSTRUCTION_MARKER:
            raise MemorySafeQwenError(
                "memory_safe_runtime_construction_forbidden"
            )
        if (
            not isinstance(device, str)
            or not device
            or not isinstance(exact, bool)
            or not isinstance(manifest_commitment, str)
            or len(manifest_commitment) != 64
            or strategy not in (
                SPARSE_STRATEGY,
                FALLBACK_STRATEGY,
            )
        ):
            raise MemorySafeQwenError(
                "memory_safe_runtime_configuration_invalid"
            )
        config = getattr(model, "config", None)
        vocabulary_size = getattr(config, "vocab_size", None)
        model_limit = getattr(
            config, "max_position_embeddings", None
        )
        tokenizer_limit = getattr(
            tokenizer, "model_max_length", None
        )
        limits = [
            value
            for value in (model_limit, tokenizer_limit)
            if isinstance(value, int)
            and not isinstance(value, bool)
            and 1 <= value < 10**8
        ]
        if (
            isinstance(vocabulary_size, bool)
            or not isinstance(vocabulary_size, int)
            or not 2 <= vocabulary_size
            <= MAXIMUM_VOCABULARY_SIZE
            or not limits
        ):
            raise MemorySafeQwenError(
                "memory_safe_model_dimensions_invalid"
            )
        if strategy == FALLBACK_STRATEGY and (
            not callable(getattr(model, "model", None))
            or not callable(getattr(model, "lm_head", None))
        ):
            raise MemorySafeQwenError(
                "memory_safe_fallback_surface_unavailable"
            )
        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch_module
        self._device = device
        self._exact = exact
        self._manifest_commitment = manifest_commitment
        self._strategy = strategy
        self._vocabulary_size = vocabulary_size
        self._context_limit_value = min(limits)
        self._expected_declarations = (
            MappingProxyType(dict(expected_declarations))
            if expected_declarations is not None
            else None
        )
        self._active_nonce = None
        self._last_sparse_chunk_count = 0
        self._parameter_binding_value = _parameter_binding(model)
        self._callable_binding_value = _callable_binding(
            model, tokenizer
        )
        self._source_sha256_value = _source_sha256()
        self._marker = (
            _EXACT_RUNTIME_MARKER
            if exact
            else _QUALIFICATION_RUNTIME_MARKER
        )
        self._runtime_commitment = contract.semantic_sha256(
            {
                "device": device,
                "dtype": (
                    DTYPE_NAME if exact else "qualification_fake"
                ),
                "manifest_commitment": manifest_commitment,
                "source_sha256": self._source_sha256_value,
                "scoring_semantics": dict(SCORING_SEMANTICS),
                "strategy": strategy,
                "vocabulary_size": vocabulary_size,
            }
        )
        self._validate_binding()

    @property
    def runtime_commitment(self) -> str:
        self._validate_binding()
        return self._runtime_commitment

    @property
    def strategy(self) -> str:
        self._validate_binding()
        return self._strategy

    def _validate_binding(self) -> None:
        expected = (
            _EXACT_RUNTIME_MARKER
            if self._exact
            else _QUALIFICATION_RUNTIME_MARKER
        )
        if (
            type(self) is not MemorySafeQwenRuntime
            or self._marker is not expected
            or self._active_nonce is not None
            or _parameter_binding(self._model)
            != self._parameter_binding_value
            or _callable_binding(self._model, self._tokenizer)
            != self._callable_binding_value
            or _source_sha256() != self._source_sha256_value
            or not _fixed_canary_closure_is_frozen()
            or not _resource_policy_is_frozen()
        ):
            raise MemorySafeQwenError(
                "memory_safe_runtime_binding_drifted"
            )
        if not self._exact:
            return
        torch = self._torch
        try:
            exact_state = (
                self._device == DEVICE
                and bool(torch.cuda.is_available())
                and not bool(
                    torch.backends.cuda.matmul.allow_tf32
                )
                and not bool(torch.backends.cudnn.allow_tf32)
                and bool(
                    torch.are_deterministic_algorithms_enabled()
                )
                and not bool(self._model.training)
                and self._loaded_declarations()
                == dict(self._expected_declarations or {})
                and all(
                    parameter.device.type == "cuda"
                    and (
                        not parameter.is_floating_point()
                        or parameter.dtype == torch.float16
                    )
                    for parameter in self._model.parameters()
                )
            )
        except (AttributeError, TypeError):
            exact_state = False
        if not exact_state:
            raise MemorySafeQwenError(
                "memory_safe_exact_cuda_fp16_binding_invalid"
            )

    def _loaded_declarations(self) -> dict[str, object]:
        config = self._model.config
        chat_template = getattr(
            self._tokenizer, "chat_template", None
        )
        if not isinstance(chat_template, str) or not chat_template:
            raise MemorySafeQwenError(
                "memory_safe_chat_template_unavailable"
            )
        return {
            "attention_implementation": str(
                getattr(config, "_attn_implementation", None)
            ),
            "chat_template_sha256": hashlib.sha256(
                chat_template.encode("utf-8")
            ).hexdigest(),
            "context_limit": self._context_limit_value,
            "critical_config": {
                key: getattr(config, key, None)
                for key in worker.QWEN_ARCHITECTURE
            },
            "loaded_config_sha256": contract.semantic_sha256(
                {
                    "architectures": list(
                        getattr(config, "architectures", [])
                    ),
                    "config_class": config.__class__.__name__,
                    "values": {
                        key: getattr(config, key, None)
                        for key in (
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
                    },
                }
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
            raise MemorySafeQwenError(
                "memory_safe_tokenizer_failed"
            ) from exc
        if (
            type(values) is not list
            or not values
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < self._vocabulary_size
                for value in values
            )
        ):
            raise MemorySafeQwenError(
                "memory_safe_tokenizer_output_invalid"
            )
        return values

    def _encode_pair(
        self, pair: closed.PromptAnswer
    ) -> _EncodedPair:
        if type(pair) is not closed.PromptAnswer:
            raise MemorySafeQwenError(
                "memory_safe_prompt_answer_invalid"
            )
        prompt = self._token_ids(pair.prompt)
        answer = self._token_ids(pair.answer)
        combined = prompt + answer
        if combined != self._token_ids(pair.prompt + pair.answer):
            raise MemorySafeQwenError(
                "memory_safe_token_boundary_invalid"
            )
        if (
            not 1 <= len(answer) <= MAXIMUM_ANSWER_TOKENS
            or len(combined)
            > hierarchical.MAXIMUM_CONTEXT_TOKENS
            or len(combined) > self._context_limit_value
        ):
            raise closed.ClosedChoiceAbstention(
                "memory_safe_context_or_answer_too_long",
                pre_model=False,
            )
        return _EncodedPair(
            tuple(prompt), tuple(answer), tuple(combined)
        )

    def _seed(self) -> None:
        self._torch.manual_seed(TORCH_SEED)
        cuda = getattr(self._torch, "cuda", None)
        manual_seed_all = getattr(cuda, "manual_seed_all", None)
        if callable(manual_seed_all):
            manual_seed_all(TORCH_SEED)

    def _pad_token_id(self) -> int:
        value = getattr(self._tokenizer, "pad_token_id", None)
        if value is None:
            value = getattr(self._tokenizer, "eos_token_id", None)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < self._vocabulary_size
        ):
            raise MemorySafeQwenError(
                "memory_safe_pad_token_invalid"
            )
        return value

    def _validate_logits(
        self,
        logits: object,
        *,
        batch: int,
        positions: int,
    ) -> None:
        torch = self._torch
        try:
            is_tensor = bool(torch.is_tensor(logits))
            shape = tuple(int(value) for value in logits.shape)
        except (AttributeError, TypeError, ValueError):
            is_tensor = False
            shape = ()
        if (
            not is_tensor
            or shape
            != (batch, positions, self._vocabulary_size)
        ):
            raise MemorySafeQwenError(
                "memory_safe_sparse_logits_shape_invalid"
            )
        if self._exact:
            try:
                exact = (
                    logits.dtype == torch.float16
                    and logits.device.type == "cuda"
                    and int(logits.device.index or 0) == 0
                )
            except AttributeError:
                exact = False
            if not exact:
                raise MemorySafeQwenError(
                    "memory_safe_sparse_logits_binding_invalid"
                )
        try:
            finite = bool(torch.isfinite(logits).all().item())
        except (AttributeError, RuntimeError, TypeError):
            finite = False
        if not finite:
            raise MemorySafeQwenError(
                "memory_safe_sparse_logits_nonfinite"
            )

    def _score_logit_rows(
        self,
        logits: object,
        targets: Sequence[int],
    ) -> int:
        """Score ``[answer, vocabulary]`` logits without widening a batch."""

        log_probabilities = self._target_logprob_rows(
            logits, targets
        )
        try:
            total = float(
                log_probabilities.sum().detach().cpu().item()
            )
        except Exception as exc:
            raise MemorySafeQwenError(
                "memory_safe_logprob_reduction_failed"
            ) from exc
        if not (-float("inf") < total < float("inf")):
            raise MemorySafeQwenError(
                "memory_safe_logprob_nonfinite"
            )
        return round(
            total * closed.LOGPROB_QUANTIZATION_SCALE
        )

    def _target_logprob_rows(
        self,
        logits: object,
        targets: Sequence[int],
    ) -> object:
        """Reduce ``[tokens, vocab]`` to one fp32 scalar per token."""

        torch = self._torch
        if len(targets) == 0:
            raise MemorySafeQwenError(
                "memory_safe_target_vector_empty"
            )
        try:
            widened = logits.float()
            if tuple(int(value) for value in widened.shape) != (
                len(targets),
                self._vocabulary_size,
            ):
                raise ValueError("target/logit shape mismatch")
            target_tensor = torch.tensor(
                list(targets),
                dtype=torch.long,
                device=self._device,
            )
            result = (
                widened.gather(
                    1, target_tensor.unsqueeze(1)
                ).squeeze(1)
                - torch.logsumexp(widened, dim=1)
            )
            finite = bool(torch.isfinite(result).all().item())
        except Exception as exc:
            raise MemorySafeQwenError(
                "memory_safe_logprob_reduction_failed"
            ) from exc
        if not finite:
            raise MemorySafeQwenError(
                "memory_safe_logprob_nonfinite"
            )
        return result

    def _target_logprob_vector(
        self,
        logits: object,
        targets: Sequence[int],
    ) -> object:
        """Reduce ``[batch, 1, vocab]`` immediately to ``[batch]``."""

        try:
            return self._target_logprob_rows(
                logits[:, 0, :], targets
            )
        except Exception as exc:
            if isinstance(exc, MemorySafeQwenError):
                raise
            raise MemorySafeQwenError(
                "memory_safe_logprob_reduction_failed"
            ) from exc

    def _score_sparse(
        self, encoded: tuple[_EncodedPair, ...]
    ) -> tuple[hierarchical.TeacherForcedScore, ...]:
        self._last_sparse_chunk_count = 0
        requested = sorted(
            {
                position
                for row in encoded
                for position in range(
                    len(row.prompt) - 1,
                    len(row.prompt) + len(row.answer) - 1,
                )
            }
        )
        per_forward_capacity = min(
            MAXIMUM_SPARSE_POSITIONS,
            MAXIMUM_SPARSE_LOGIT_ELEMENTS
            // (len(encoded) * self._vocabulary_size),
        )
        if not requested or per_forward_capacity < 1:
            raise closed.ClosedChoiceAbstention(
                "memory_safe_sparse_resource_bound_exceeded",
                pre_model=True,
            )
        maximum = max(len(row.combined) for row in encoded)
        pad = self._pad_token_id()
        input_rows = [
            list(row.combined)
            + [pad] * (maximum - len(row.combined))
            for row in encoded
        ]
        mask_rows = [
            [1] * len(row.combined)
            + [0] * (maximum - len(row.combined))
            for row in encoded
        ]
        torch = self._torch
        input_tensor = torch.tensor(
            input_rows, dtype=torch.long, device=self._device
        )
        mask_tensor = torch.tensor(
            mask_rows, dtype=torch.long, device=self._device
        )
        self._seed()
        per_row: list[list[tuple[int, object]]] = [
            [] for _ in encoded
        ]
        with torch.inference_mode():
            for offset in range(
                0, len(requested), per_forward_capacity
            ):
                self._last_sparse_chunk_count += 1
                chunk = requested[
                    offset : offset + per_forward_capacity
                ]
                chunk_lookup = {
                    position: index
                    for index, position in enumerate(chunk)
                }
                position_tensor = torch.tensor(
                    chunk,
                    dtype=torch.long,
                    device=self._device,
                )
                try:
                    output = self._model(
                        input_ids=input_tensor,
                        attention_mask=mask_tensor,
                        use_cache=False,
                        return_dict=True,
                        logits_to_keep=position_tensor,
                    )
                    logits = output.logits
                except MemorySafeQwenError:
                    raise
                except Exception as exc:
                    # A declared sparse API failure is not silently converted
                    # into a different numerical path.  Compatibility must be
                    # qualified before an exact runtime is frozen.
                    raise MemorySafeQwenError(
                        "memory_safe_sparse_model_forward_failed"
                    ) from exc
                self._validate_logits(
                    logits,
                    batch=len(encoded),
                    positions=len(chunk),
                )
                for row_index, row in enumerate(encoded):
                    start = len(row.prompt) - 1
                    row_positions = [
                        position
                        for position in range(
                            start,
                            start + len(row.answer),
                        )
                        if position in chunk_lookup
                    ]
                    if not row_positions:
                        continue
                    sparse_indices = torch.tensor(
                        [
                            chunk_lookup[position]
                            for position in row_positions
                        ],
                        dtype=torch.long,
                        device=self._device,
                    )
                    row_logits = logits[row_index].index_select(
                        0, sparse_indices
                    )
                    row_targets = [
                        row.answer[position - start]
                        for position in row_positions
                    ]
                    log_probabilities = (
                        self._target_logprob_rows(
                            row_logits, row_targets
                        )
                    )
                    per_row[row_index].extend(
                        (
                            position,
                            log_probabilities[index],
                        )
                        for index, position in enumerate(
                            row_positions
                        )
                    )
                del logits
                del output
        scores: list[hierarchical.TeacherForcedScore] = []
        for row_index, row in enumerate(encoded):
            expected_positions = list(
                range(
                    len(row.prompt) - 1,
                    len(row.prompt) + len(row.answer) - 1,
                )
            )
            observed = sorted(
                per_row[row_index], key=lambda item: item[0]
            )
            if [position for position, _ in observed] != (
                expected_positions
            ):
                raise MemorySafeQwenError(
                    "memory_safe_sparse_position_coverage_invalid"
                )
            try:
                total = float(
                    torch.stack(
                        [value for _, value in observed]
                    )
                    .sum()
                    .detach()
                    .cpu()
                    .item()
                )
            except Exception as exc:
                raise MemorySafeQwenError(
                    "memory_safe_logprob_reduction_failed"
                ) from exc
            quantized = round(
                total * closed.LOGPROB_QUANTIZATION_SCALE
            )
            scores.append(
                hierarchical.TeacherForcedScore(
                    total_logprob_microunits=quantized,
                    answer_token_count=len(row.answer),
                    context_and_answer_token_count=len(
                        row.combined
                    ),
                )
            )
        return tuple(scores)

    def _score_fixed_full_logits_reference(
        self,
        pair: closed.PromptAnswer,
        *,
        nonce: object,
    ) -> hierarchical.TeacherForcedScore:
        """Independently score the fixed short pair using full LM logits."""

        if self._active_nonce is not nonce:
            raise MemorySafeQwenError(
                "memory_safe_score_authority_invalid"
            )
        encoded = self._encode_pair(pair)
        if len(encoded.answer) > 64:
            raise MemorySafeQwenError(
                "memory_safe_full_reference_not_short"
            )
        torch = self._torch
        input_tensor = torch.tensor(
            [list(encoded.combined)],
            dtype=torch.long,
            device=self._device,
        )
        attention_mask = torch.ones(
            (1, len(encoded.combined)),
            dtype=torch.long,
            device=self._device,
        )
        kwargs: dict[str, object] = {
            "attention_mask": attention_mask,
            "input_ids": input_tensor,
            "return_dict": True,
            "use_cache": False,
        }
        if _explicit_tensor_logits_to_keep(
            self._model, exact=self._exact
        ):
            # In the pinned Qwen API, integer zero requests all positions;
            # this is deliberately distinct from the tensor sparse path.
            kwargs["logits_to_keep"] = 0
        self._seed()
        try:
            with torch.inference_mode():
                output = self._model(**kwargs)
                logits = output.logits
        except MemorySafeQwenError:
            raise
        except Exception as exc:
            raise MemorySafeQwenError(
                "memory_safe_full_reference_forward_failed"
            ) from exc
        self._validate_logits(
            logits,
            batch=1,
            positions=len(encoded.combined),
        )
        prediction_positions = torch.tensor(
            list(
                range(
                    len(encoded.prompt) - 1,
                    len(encoded.prompt)
                    + len(encoded.answer)
                    - 1,
                )
            ),
            dtype=torch.long,
            device=self._device,
        )
        try:
            selected = logits[0].index_select(
                0, prediction_positions
            )
            microunits = self._score_logit_rows(
                selected, encoded.answer
            )
        finally:
            del logits
            del output
        return hierarchical.TeacherForcedScore(
            total_logprob_microunits=microunits,
            answer_token_count=len(encoded.answer),
            context_and_answer_token_count=len(
                encoded.combined
            ),
        )

    def _fallback_hidden(
        self,
        output: object,
        *,
        batch: int,
        positions: int,
    ) -> object:
        hidden = getattr(output, "last_hidden_state", None)
        try:
            shape = tuple(int(value) for value in hidden.shape)
        except (AttributeError, TypeError, ValueError):
            shape = ()
        if len(shape) != 3 or shape[:2] != (batch, positions):
            raise MemorySafeQwenError(
                "memory_safe_fallback_hidden_shape_invalid"
            )
        return hidden

    def _fallback_last_logits(
        self, hidden: object, *, batch: int
    ) -> object:
        try:
            logits = self._model.lm_head(hidden[:, -1:, :])
        except Exception as exc:
            raise MemorySafeQwenError(
                "memory_safe_fallback_lm_head_failed"
            ) from exc
        self._validate_logits(logits, batch=batch, positions=1)
        return logits

    def _score_fallback_group(
        self,
        rows: tuple[tuple[int, _EncodedPair], ...],
    ) -> tuple[
        tuple[int, hierarchical.TeacherForcedScore], ...
    ]:
        torch = self._torch
        batch = len(rows)
        prompt_length = len(rows[0][1].prompt)
        answer_length = len(rows[0][1].answer)
        if any(
            len(row.prompt) != prompt_length
            or len(row.answer) != answer_length
            for _, row in rows
        ):
            raise MemorySafeQwenError(
                "memory_safe_fallback_group_invalid"
            )
        prompt_tensor = torch.tensor(
            [list(row.prompt) for _, row in rows],
            dtype=torch.long,
            device=self._device,
        )
        attention_mask = torch.ones(
            (batch, prompt_length),
            dtype=torch.long,
            device=self._device,
        )
        try:
            output = self._model.model(
                input_ids=prompt_tensor,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        except Exception as exc:
            raise MemorySafeQwenError(
                "memory_safe_fallback_decoder_failed"
            ) from exc
        hidden = self._fallback_hidden(
            output, batch=batch, positions=prompt_length
        )
        past = getattr(output, "past_key_values", None)
        if past is None:
            raise MemorySafeQwenError(
                "memory_safe_fallback_cache_unavailable"
            )
        first_logits = self._fallback_last_logits(
            hidden, batch=batch
        )
        log_probability_steps = [
            self._target_logprob_vector(
                first_logits,
                [row.answer[0] for _, row in rows],
            )
        ]
        del first_logits
        for step in range(1, answer_length):
            previous = torch.tensor(
                [
                    [row.answer[step - 1]]
                    for _, row in rows
                ],
                dtype=torch.long,
                device=self._device,
            )
            attention_mask = torch.ones(
                (batch, prompt_length + step),
                dtype=torch.long,
                device=self._device,
            )
            try:
                output = self._model.model(
                    input_ids=previous,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
            except Exception as exc:
                raise MemorySafeQwenError(
                    "memory_safe_fallback_decoder_failed"
                ) from exc
            hidden = self._fallback_hidden(
                output, batch=batch, positions=1
            )
            past = getattr(output, "past_key_values", None)
            if past is None:
                raise MemorySafeQwenError(
                    "memory_safe_fallback_cache_unavailable"
                )
            step_logits = self._fallback_last_logits(
                hidden, batch=batch
            )
            log_probability_steps.append(
                self._target_logprob_vector(
                    step_logits,
                    [row.answer[step] for _, row in rows],
                )
            )
            del step_logits
        try:
            # Only [batch, answer_tokens] fp32 scalars survive the loop.
            stacked = torch.stack(
                log_probability_steps, dim=1
            )
            total_values = [
                float(
                    stacked[row_index]
                    .sum()
                    .detach()
                    .cpu()
                    .item()
                )
                for row_index in range(batch)
            ]
        except Exception as exc:
            raise MemorySafeQwenError(
                "memory_safe_logprob_reduction_failed"
            ) from exc
        results: list[
            tuple[int, hierarchical.TeacherForcedScore]
        ] = []
        for grouped_index, (original_index, row) in enumerate(rows):
            quantized = round(
                total_values[grouped_index]
                * closed.LOGPROB_QUANTIZATION_SCALE
            )
            results.append(
                (
                    original_index,
                    hierarchical.TeacherForcedScore(
                        total_logprob_microunits=quantized,
                        answer_token_count=len(row.answer),
                        context_and_answer_token_count=len(
                            row.combined
                        ),
                    ),
                )
            )
        return tuple(results)

    def _score_fallback(
        self, encoded: tuple[_EncodedPair, ...]
    ) -> tuple[hierarchical.TeacherForcedScore, ...]:
        groups: dict[
            tuple[int, int],
            list[tuple[int, _EncodedPair]],
        ] = {}
        for index, row in enumerate(encoded):
            groups.setdefault(
                (len(row.prompt), len(row.answer)), []
            ).append((index, row))
        if len(groups) > MAXIMUM_FALLBACK_GROUPS:
            raise closed.ClosedChoiceAbstention(
                "memory_safe_fallback_group_bound_exceeded",
                pre_model=True,
            )
        self._seed()
        unordered: list[
            tuple[int, hierarchical.TeacherForcedScore]
        ] = []
        try:
            with self._torch.inference_mode():
                for key in sorted(groups):
                    unordered.extend(
                        self._score_fallback_group(
                            tuple(groups[key])
                        )
                    )
        except MemorySafeQwenError:
            raise
        except Exception as exc:
            raise MemorySafeQwenError(
                "memory_safe_fallback_model_forward_failed"
            ) from exc
        ordered = sorted(unordered, key=lambda row: row[0])
        if [index for index, _ in ordered] != list(
            range(len(encoded))
        ):
            raise MemorySafeQwenError(
                "memory_safe_fallback_result_order_invalid"
            )
        return tuple(score for _, score in ordered)

    def _score_batch_bound(
        self,
        pairs: tuple[closed.PromptAnswer, ...],
        *,
        nonce: object,
    ) -> tuple[hierarchical.TeacherForcedScore, ...]:
        if (
            self._active_nonce is not nonce
            or type(pairs) is not tuple
            or not 1 <= len(pairs) <= SCORING_BATCH_SIZE
            or any(type(pair) is not closed.PromptAnswer for pair in pairs)
        ):
            raise MemorySafeQwenError(
                "memory_safe_score_authority_invalid"
            )
        encoded = tuple(self._encode_pair(pair) for pair in pairs)
        if self._strategy == SPARSE_STRATEGY:
            return self._score_sparse(encoded)
        if self._strategy == FALLBACK_STRATEGY:
            return self._score_fallback(encoded)
        raise MemorySafeQwenError(
            "memory_safe_strategy_invalid"
        )

    def _with_backend(
        self,
        operation: object,
        *,
        v2_errors: bool = False,
    ) -> object:
        if self._active_nonce is not None:
            raise MemorySafeQwenError(
                "memory_safe_runtime_reentrant"
            )
        self._validate_binding()
        nonce = object()
        self._active_nonce = nonce
        backend = _PrivateTeacherForcedBackend(
            self,
            nonce,
            _BACKEND_MARKER,
            v2_errors=v2_errors,
        )
        try:
            return operation(backend)
        finally:
            self._active_nonce = None
            self._validate_binding()

    def score_batch_qualification_only(
        self,
        pairs: tuple[closed.PromptAnswer, ...],
    ) -> tuple[hierarchical.TeacherForcedScore, ...]:
        """Score fake qualification pairs; unavailable on exact runtimes."""

        if (
            self._exact
            or self._marker is not _QUALIFICATION_RUNTIME_MARKER
        ):
            raise MemorySafeQwenError(
                "memory_safe_qualification_api_forbidden"
            )
        result = self._with_backend(
            lambda backend: backend.score_batch(pairs)
        )
        if type(result) is not tuple:
            raise MemorySafeQwenError(
                "memory_safe_score_result_invalid"
            )
        return result

    def run_fixed_teacher_forced_canary(
        self,
    ) -> Mapping[str, object]:
        """Run the immutable sparse/fallback equivalence canary.

        The method accepts no prompt, answer, scorer, logits, or backend.
        Production uses it on the exact CUDA runtime; fake CPU runtimes may
        exercise the identical fixed control flow in source-free tests.
        """

        def operation(
            backend: _PrivateTeacherForcedBackend,
        ) -> Mapping[str, object]:
            nonce = self._active_nonce
            if nonce is None:
                raise MemorySafeQwenError(
                    "memory_safe_score_authority_invalid"
                )
            short_strategy = backend.score_batch(
                (_FIXED_SHORT_CANARY_PAIR,)
            )[0]
            short_reference = (
                self._score_fixed_full_logits_reference(
                    _FIXED_SHORT_CANARY_PAIR,
                    nonce=nonce,
                )
            )
            if short_strategy != short_reference:
                raise MemorySafeQwenError(
                    "memory_safe_fixed_canary_short_mismatch"
                )
            long_encoded = self._encode_pair(
                _FIXED_LONG_CANARY_PAIR
            )
            if len(long_encoded.answer) <= MAXIMUM_SPARSE_POSITIONS:
                raise MemorySafeQwenError(
                    "memory_safe_fixed_canary_not_chunked"
                )
            first_long = backend.score_batch(
                (_FIXED_LONG_CANARY_PAIR,)
            )[0]
            first_chunks = self._last_sparse_chunk_count
            second_long = backend.score_batch(
                (_FIXED_LONG_CANARY_PAIR,)
            )[0]
            second_chunks = self._last_sparse_chunk_count
            first_long_bytes = contract.canonical_json_bytes(
                {
                    "answer_token_count": (
                        first_long.answer_token_count
                    ),
                    "context_and_answer_token_count": (
                        first_long.context_and_answer_token_count
                    ),
                    "total_logprob_microunits": (
                        first_long.total_logprob_microunits
                    ),
                },
                newline=False,
            )
            second_long_bytes = contract.canonical_json_bytes(
                {
                    "answer_token_count": (
                        second_long.answer_token_count
                    ),
                    "context_and_answer_token_count": (
                        second_long.context_and_answer_token_count
                    ),
                    "total_logprob_microunits": (
                        second_long.total_logprob_microunits
                    ),
                },
                newline=False,
            )
            if first_long_bytes != second_long_bytes:
                raise MemorySafeQwenError(
                    "memory_safe_fixed_canary_long_mismatch"
                )
            if self._strategy == SPARSE_STRATEGY and (
                first_chunks < 2 or second_chunks != first_chunks
            ):
                raise MemorySafeQwenError(
                    "memory_safe_fixed_canary_chunk_count_invalid"
                )
            body: dict[str, object] = {
                "fallback_independent_full_reference_passed": (
                    self._strategy != FALLBACK_STRATEGY
                    or short_strategy == short_reference
                ),
                "free_form_generation_count": 0,
                "long_answer_position_count": len(
                    long_encoded.answer
                ),
                "long_context_and_answer_token_count": len(
                    long_encoded.combined
                ),
                "long_pair_sha256": (
                    FIXED_LONG_CANARY_PAIR_SHA256
                ),
                "long_repeat_byte_exact": (
                    first_long_bytes == second_long_bytes
                ),
                "long_score_sha256": hashlib.sha256(
                    first_long_bytes
                ).hexdigest(),
                "long_score_microunits": (
                    first_long.total_logprob_microunits
                ),
                "schema": FIXED_CANARY_SCHEMA,
                "short_answer_position_count": (
                    short_strategy.answer_token_count
                ),
                "short_full_reference_microunits": (
                    short_reference.total_logprob_microunits
                ),
                "short_pair_sha256": (
                    FIXED_SHORT_CANARY_PAIR_SHA256
                ),
                "short_strategy_microunits": (
                    short_strategy.total_logprob_microunits
                ),
                "short_strategy_vs_full_reference_exact": True,
                "sparse_chunk_count": (
                    first_chunks
                    if self._strategy == SPARSE_STRATEGY
                    else 0
                ),
                "strategy": self._strategy,
            }
            return MappingProxyType(
                {
                    **body,
                    "self_sha256": contract.semantic_sha256(
                        body
                    ),
                }
            )

        result = self._with_backend(operation)
        if not isinstance(result, Mapping):
            raise MemorySafeQwenError(
                "memory_safe_fixed_canary_result_invalid"
            )
        return result

    def select_story(
        self, story_text: str
    ) -> hierarchical.ClosedChoiceV2Decision:
        """Run the program-owned engine; exact runtime's only scoring ABI."""

        def operation(
            backend: _PrivateTeacherForcedBackend,
        ) -> hierarchical.ClosedChoiceV2Decision:
            engine = hierarchical._HierarchicalEngine(
                hierarchical._ENGINE_MARKER
            )
            return engine.select(
                story_text,
                backend=backend,
                narrative_parser=worker._independent_parser,
            )

        try:
            decision = self._with_backend(
                operation, v2_errors=True
            )
        except (ClosedChoiceV2Error, ClosedChoiceV2Abstention):
            raise
        except MemorySafeQwenError as exc:
            issue = str(getattr(exc, "issue_id", exc))
            if "cuda" in issue:
                mapped = "V2_CUDA_RUNTIME_UNAVAILABLE"
            elif "token" in issue or "boundary" in issue:
                mapped = "V2_TOKEN_BOUNDARY_INVALID"
            elif (
                "authority" in issue
                or "binding" in issue
                or "manifest" in issue
                or "configuration" in issue
            ):
                mapped = "V2_AUTHORITY_INVALID"
            else:
                mapped = "V2_MODEL_FORWARD_FAILED"
            raise ClosedChoiceV2Error(mapped) from exc
        if type(decision) is not hierarchical.ClosedChoiceV2Decision:
            raise ClosedChoiceV2Error(
                "V2_MODEL_FORWARD_FAILED"
            )
        return decision


def build_fake_runtime_qualification_only(
    *,
    model: object,
    tokenizer: object,
    torch_module: object,
    device: str = "cpu",
    strategy: str | None = None,
) -> MemorySafeQwenRuntime:
    """Build an explicitly non-formal runtime for source-free fake tests."""

    selected = strategy
    if selected is None:
        selected = (
            SPARSE_STRATEGY
            if _explicit_tensor_logits_to_keep(model)
            else FALLBACK_STRATEGY
        )
    return MemorySafeQwenRuntime(
        model=model,
        tokenizer=tokenizer,
        torch_module=torch_module,
        device=device,
        exact=False,
        manifest_commitment=hashlib.sha256(
            b"gscl-memory-safe-qwen-fake-qualification-only"
        ).hexdigest(),
        strategy=selected,
        expected_declarations=None,
        marker=_CONSTRUCTION_MARKER,
    )


def load_exact_cuda_fp16_runtime(
    *,
    model_root: Path,
    manifest: worker.ModelAssetManifest,
) -> MemorySafeQwenRuntime:
    """Load the exact offline Qwen asset into a CUDA-fp16 v2 runtime.

    This constructor accepts no model, tokenizer, logits, scorer, or parser
    injection.  The asset manifest must already carry v1's private verified
    marker.  No formal source is opened here.
    """

    if (
        type(manifest) is not worker.ModelAssetManifest
        or manifest._marker is not worker._VERIFIED_MANIFEST_MARKER
    ):
        raise MemorySafeQwenError(
            "memory_safe_model_manifest_not_verified"
        )
    if not isinstance(model_root, Path):
        raise MemorySafeQwenError(
            "memory_safe_model_root_invalid"
        )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise MemorySafeQwenError(
            "memory_safe_local_runtime_unavailable"
        ) from exc
    if not torch.cuda.is_available():
        raise MemorySafeQwenError(
            "memory_safe_cuda_unavailable"
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
        raise MemorySafeQwenError(
            "memory_safe_local_model_load_failed"
        ) from exc
    model.eval()
    strategy = (
        SPARSE_STRATEGY
        if _explicit_tensor_logits_to_keep(
            model, exact=True
        )
        else FALLBACK_STRATEGY
    )
    return MemorySafeQwenRuntime(
        model=model,
        tokenizer=tokenizer,
        torch_module=torch,
        device=DEVICE,
        exact=True,
        manifest_commitment=manifest.manifest_file_sha256,
        strategy=strategy,
        expected_declarations=manifest.declarations,
        marker=_CONSTRUCTION_MARKER,
    )
