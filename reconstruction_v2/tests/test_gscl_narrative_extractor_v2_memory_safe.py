from __future__ import annotations

import hashlib
import inspect
from types import SimpleNamespace

import pytest
import torch

from replication_runtime.gscl_narrative_extractor_v1 import (
    closed_choice_worker as closed,
)
from replication_runtime.gscl_narrative_extractor_v2 import (
    closed_choice as hierarchical,
)
from replication_runtime.gscl_narrative_extractor_v2 import (
    memory_safe_qwen as memory_safe,
)


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    model_max_length = closed.MAXIMUM_CONTEXT_TOKENS

    def __init__(self, vocabulary_size: int) -> None:
        self.vocabulary_size = vocabulary_size

    def __call__(
        self, text: str, **_: object
    ) -> dict[str, list[int]]:
        return {
            "input_ids": [
                ord(character) % (self.vocabulary_size - 1) + 1
                for character in text
            ]
        }


class WordTokenizer(CharacterTokenizer):
    def __call__(
        self, text: str, **_: object
    ) -> dict[str, list[int]]:
        return {
            "input_ids": [
                int(
                    hashlib.sha256(
                        token.encode("utf-8")
                    ).hexdigest()[:8],
                    16,
                )
                % (self.vocabulary_size - 1)
                + 1
                for token in text.split()
            ]
        }


class SparseFakeModel:
    def __init__(self, *, vocabulary_size: int = 19) -> None:
        self.config = SimpleNamespace(
            vocab_size=vocabulary_size,
            max_position_embeddings=closed.MAXIMUM_CONTEXT_TOKENS,
        )
        self.calls: list[dict[str, object]] = []
        self.training = False

    def named_parameters(self):
        return iter(())

    def parameters(self):
        return iter(())

    def _full_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, length = input_ids.shape
        vocabulary = torch.arange(
            self.config.vocab_size,
            dtype=torch.float32,
            device=input_ids.device,
        ).view(1, 1, -1)
        positions = torch.arange(
            length,
            dtype=torch.long,
            device=input_ids.device,
        ).view(1, -1)
        centres = (
            input_ids + positions
        ) % self.config.vocab_size
        return -(
            vocabulary - centres.unsqueeze(-1).float()
        ).square() / 7.0

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
        logits_to_keep: torch.Tensor | int,
    ) -> SimpleNamespace:
        assert use_cache is False
        assert return_dict is True
        if isinstance(logits_to_keep, int):
            assert logits_to_keep == 0
            positions = tuple(range(int(input_ids.shape[1])))
            selected = self._full_logits(input_ids)
        else:
            positions = tuple(
                int(value)
                for value in logits_to_keep.cpu().tolist()
            )
            selected = self._full_logits(input_ids).index_select(
                1, logits_to_keep
            )
        self.calls.append(
            {
                "input_shape": tuple(input_ids.shape),
                "mask_shape": tuple(attention_mask.shape),
                "positions": positions,
            }
        )
        return SimpleNamespace(logits=selected)

    __call__ = forward


class UniformSparseFakeModel(SparseFakeModel):
    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
        logits_to_keep: torch.Tensor,
    ) -> SimpleNamespace:
        assert use_cache is False
        assert return_dict is True
        self.calls.append(
            {
                "input_shape": tuple(input_ids.shape),
                "mask_shape": tuple(attention_mask.shape),
                "positions": tuple(
                    int(value)
                    for value in logits_to_keep.cpu().tolist()
                ),
            }
        )
        return SimpleNamespace(
            logits=torch.zeros(
                (
                    int(input_ids.shape[0]),
                    int(logits_to_keep.shape[0]),
                    self.config.vocab_size,
                ),
                dtype=torch.float32,
                device=input_ids.device,
            )
        )

    __call__ = forward


class FullSequenceIgnoringSparseModel(SparseFakeModel):
    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
        logits_to_keep: torch.Tensor,
    ) -> SimpleNamespace:
        self.calls.append(
            {"positions": tuple(logits_to_keep.cpu().tolist())}
        )
        return SimpleNamespace(logits=self._full_logits(input_ids))

    __call__ = forward


class NonfiniteSparseModel(SparseFakeModel):
    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
        logits_to_keep: torch.Tensor,
    ) -> SimpleNamespace:
        logits = self._full_logits(input_ids).index_select(
            1, logits_to_keep
        )
        logits[0, 0, 0] = float("nan")
        self.calls.append(
            {"positions": tuple(logits_to_keep.cpu().tolist())}
        )
        return SimpleNamespace(logits=logits)

    __call__ = forward


class FallbackDecoder:
    def __init__(self, owner: "FallbackFakeModel") -> None:
        self.owner = owner

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
        past_key_values: object | None = None,
    ) -> SimpleNamespace:
        assert use_cache is True
        assert return_dict is True
        batch, positions = input_ids.shape
        vocabulary = torch.arange(
            self.owner.config.vocab_size,
            dtype=torch.float32,
            device=input_ids.device,
        ).view(1, 1, -1)
        # The next-token distribution depends only on the immediately
        # preceding token.  This makes the full teacher-forced reference and
        # the incremental-cache path analytically identical.
        hidden = -(
            vocabulary - input_ids.unsqueeze(-1).float()
        ).square() / 5.0
        self.owner.decoder_calls.append(
            {
                "batch": batch,
                "positions": positions,
                "attention_length": int(attention_mask.shape[1]),
                "had_cache": past_key_values is not None,
            }
        )
        return SimpleNamespace(
            last_hidden_state=hidden,
            past_key_values=(
                "cache",
                len(self.owner.decoder_calls),
            ),
        )


class FallbackFakeModel:
    def __init__(self, *, vocabulary_size: int = 19) -> None:
        self.config = SimpleNamespace(
            vocab_size=vocabulary_size,
            max_position_embeddings=closed.MAXIMUM_CONTEXT_TOKENS,
        )
        self.training = False
        self.decoder_calls: list[dict[str, object]] = []
        self.lm_head_shapes: list[tuple[int, ...]] = []
        self.top_level_calls = 0
        self.model = FallbackDecoder(self)

    def named_parameters(self):
        return iter(())

    def parameters(self):
        return iter(())

    # No logits_to_keep parameter: capability discovery must select the
    # incremental Qwen decoder/lm_head fallback.
    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> object:
        self.top_level_calls += 1
        raise AssertionError("top-level causal-LM forward is forbidden")

    __call__ = forward

    def lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        self.lm_head_shapes.append(tuple(hidden.shape))
        return hidden


class CanaryFallbackFakeModel(FallbackFakeModel):
    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        assert use_cache is False
        assert return_dict is True
        assert tuple(attention_mask.shape) == tuple(
            input_ids.shape
        )
        self.top_level_calls += 1
        vocabulary = torch.arange(
            self.config.vocab_size,
            dtype=torch.float32,
            device=input_ids.device,
        ).view(1, 1, -1)
        return SimpleNamespace(
            logits=-(
                vocabulary - input_ids.unsqueeze(-1).float()
            ).square()
            / 5.0
        )

    __call__ = forward


class CachelessFallbackDecoder(FallbackDecoder):
    def __call__(self, **kwargs: object) -> SimpleNamespace:
        output = super().__call__(**kwargs)
        output.past_key_values = None
        return output


def _pair(
    key: str,
    *,
    prompt: str = "prompt:",
    answer: str = "answer",
) -> closed.PromptAnswer:
    return closed.PromptAnswer(
        candidate_key=key,
        prompt=prompt,
        answer=answer,
    )


def _runtime(
    model: object,
    *,
    vocabulary_size: int = 19,
    strategy: str | None = None,
) -> memory_safe.MemorySafeQwenRuntime:
    return memory_safe.build_fake_runtime_qualification_only(
        model=model,
        tokenizer=CharacterTokenizer(vocabulary_size),
        torch_module=torch,
        strategy=strategy,
    )


def _reference_sparse_score(
    model: SparseFakeModel,
    tokenizer: CharacterTokenizer,
    pair: closed.PromptAnswer,
) -> int:
    prompt = tokenizer(pair.prompt)["input_ids"]
    answer = tokenizer(pair.answer)["input_ids"]
    sequence = torch.tensor([prompt + answer], dtype=torch.long)
    full = model._full_logits(sequence)[0]
    positions = torch.arange(
        len(prompt) - 1,
        len(prompt) + len(answer) - 1,
    )
    selected = full.index_select(0, positions).float()
    targets = torch.tensor(answer, dtype=torch.long)
    total = (
        selected.gather(1, targets.unsqueeze(1)).squeeze(1)
        - torch.logsumexp(selected, dim=1)
    ).sum()
    return round(
        float(total.item())
        * closed.LOGPROB_QUANTIZATION_SCALE
    )


def _reference_fallback_score(
    tokenizer: CharacterTokenizer,
    pair: closed.PromptAnswer,
    vocabulary_size: int,
) -> int:
    prompt = tokenizer(pair.prompt)["input_ids"]
    answer = tokenizer(pair.answer)["input_ids"]
    predecessors = [prompt[-1], *answer[:-1]]
    vocabulary = torch.arange(
        vocabulary_size, dtype=torch.float32
    ).view(1, -1)
    logits = -(
        vocabulary
        - torch.tensor(predecessors).float().unsqueeze(1)
    ).square() / 5.0
    targets = torch.tensor(answer, dtype=torch.long)
    total = (
        logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        - torch.logsumexp(logits, dim=1)
    ).sum()
    return round(
        float(total.item())
        * closed.LOGPROB_QUANTIZATION_SCALE
    )


def test_sparse_path_requests_only_union_of_answer_positions() -> None:
    model = SparseFakeModel()
    tokenizer = CharacterTokenizer(model.config.vocab_size)
    runtime = _runtime(model)
    pairs = (
        _pair("a", prompt="short:", answer="xy"),
        _pair("b", prompt="longer:", answer="z"),
    )
    scores = runtime.score_batch_qualification_only(pairs)

    expected_positions = sorted(
        {
            *range(len("short:") - 1, len("short:") + 1),
            *range(len("longer:") - 1, len("longer:")),
        }
    )
    assert model.calls == [
        {
            "input_shape": (2, len("longer:z")),
            "mask_shape": (2, len("longer:z")),
            "positions": tuple(expected_positions),
        }
    ]
    assert [row.total_logprob_microunits for row in scores] == [
        _reference_sparse_score(model, tokenizer, pair)
        for pair in pairs
    ]
    assert runtime.strategy == memory_safe.SPARSE_STRATEGY


def test_sparse_path_is_byte_exact_across_repeated_calls() -> None:
    model = SparseFakeModel()
    runtime = _runtime(model)
    pairs = (
        _pair("a", prompt="same:", answer="one"),
        _pair("b", prompt="same:", answer="two"),
    )
    first = runtime.score_batch_qualification_only(pairs)
    second = runtime.score_batch_qualification_only(pairs)
    assert first == second
    assert len(model.calls) == 2
    assert model.calls[0] == model.calls[1]


def test_sparse_path_rejects_model_that_ignores_position_indices() -> None:
    model = FullSequenceIgnoringSparseModel()
    runtime = _runtime(model)
    with pytest.raises(memory_safe.MemorySafeQwenError) as error:
        runtime.score_batch_qualification_only(
            (_pair("a", answer="xy"),)
        )
    assert error.value.issue_id == (
        "memory_safe_sparse_logits_shape_invalid"
    )


def test_sparse_path_rejects_nonfinite_logits() -> None:
    model = NonfiniteSparseModel()
    runtime = _runtime(model)
    with pytest.raises(memory_safe.MemorySafeQwenError) as error:
        runtime.score_batch_qualification_only(
            (_pair("a", answer="xy"),)
        )
    assert error.value.issue_id == (
        "memory_safe_sparse_logits_nonfinite"
    )


def test_sparse_positions_are_chunked_without_narrowing_v1_domain() -> None:
    vocabulary_size = 1_000
    model = SparseFakeModel(vocabulary_size=vocabulary_size)
    runtime = _runtime(
        model, vocabulary_size=vocabulary_size
    )
    pairs = tuple(
        _pair(
            str(index),
            prompt="p" * prompt_length,
            answer="a" * 80,
        )
        for index, prompt_length in enumerate((100, 140, 180, 220))
    )
    scores = runtime.score_batch_qualification_only(pairs)
    assert len(scores) == 4
    requested = [
        position
        for call in model.calls
        for position in call["positions"]
    ]
    assert requested == list(range(99, 299))
    assert len(model.calls) == 2
    assert all(
        len(call["positions"])
        <= memory_safe.MAXIMUM_SPARSE_POSITIONS
        for call in model.calls
    )


def test_fallback_uses_batched_decoder_cache_and_one_position_head() -> None:
    model = FallbackFakeModel()
    tokenizer = CharacterTokenizer(model.config.vocab_size)
    runtime = _runtime(model)
    pairs = (
        _pair("a", prompt="same:", answer="one"),
        _pair("b", prompt="same:", answer="two"),
    )
    scores = runtime.score_batch_qualification_only(pairs)
    assert runtime.strategy == memory_safe.FALLBACK_STRATEGY
    assert model.top_level_calls == 0
    assert model.decoder_calls == [
        {
            "batch": 2,
            "positions": len("same:"),
            "attention_length": len("same:"),
            "had_cache": False,
        },
        {
            "batch": 2,
            "positions": 1,
            "attention_length": len("same:") + 1,
            "had_cache": True,
        },
        {
            "batch": 2,
            "positions": 1,
            "attention_length": len("same:") + 2,
            "had_cache": True,
        },
    ]
    assert all(shape[1] == 1 for shape in model.lm_head_shapes)
    assert [row.total_logprob_microunits for row in scores] == [
        _reference_fallback_score(
            tokenizer, pair, model.config.vocab_size
        )
        for pair in pairs
    ]


def test_fallback_rejects_missing_kv_cache() -> None:
    model = FallbackFakeModel()
    model.model = CachelessFallbackDecoder(model)
    runtime = _runtime(model)
    with pytest.raises(memory_safe.MemorySafeQwenError) as error:
        runtime.score_batch_qualification_only(
            (_pair("a", answer="xy"),)
        )
    assert error.value.issue_id == (
        "memory_safe_fallback_cache_unavailable"
    )
    assert model.top_level_calls == 0


def test_token_boundary_merge_is_rejected_before_model_call() -> None:
    class MergingTokenizer(CharacterTokenizer):
        def __call__(
            self, text: str, **kwargs: object
        ) -> dict[str, list[int]]:
            if text == "prompt:answer":
                return {"input_ids": [3]}
            return super().__call__(text, **kwargs)

    model = SparseFakeModel()
    runtime = memory_safe.build_fake_runtime_qualification_only(
        model=model,
        tokenizer=MergingTokenizer(model.config.vocab_size),
        torch_module=torch,
    )
    with pytest.raises(memory_safe.MemorySafeQwenError) as error:
        runtime.score_batch_qualification_only(
            (_pair("a"),)
        )
    assert error.value.issue_id == (
        "memory_safe_token_boundary_invalid"
    )
    assert model.calls == []


def test_private_construction_and_backend_authority_are_not_forgeable() -> None:
    model = SparseFakeModel()
    tokenizer = CharacterTokenizer(model.config.vocab_size)
    with pytest.raises(memory_safe.MemorySafeQwenError) as error:
        memory_safe.MemorySafeQwenRuntime(
            model=model,
            tokenizer=tokenizer,
            torch_module=torch,
            device="cpu",
            exact=False,
            manifest_commitment=hashlib.sha256(b"x").hexdigest(),
            strategy=memory_safe.SPARSE_STRATEGY,
            expected_declarations=None,
            marker=object(),
        )
    assert error.value.issue_id == (
        "memory_safe_runtime_construction_forbidden"
    )
    with pytest.raises(memory_safe.MemorySafeQwenError) as error:
        memory_safe._PrivateTeacherForcedBackend(
            object(), object(), object()
        )
    assert error.value.issue_id == (
        "memory_safe_backend_authority_invalid"
    )


def test_program_owned_selection_uses_private_backend_end_to_end() -> None:
    model = UniformSparseFakeModel()
    runtime = memory_safe.build_fake_runtime_qualification_only(
        model=model,
        tokenizer=WordTokenizer(model.config.vocab_size),
        torch_module=torch,
    )
    decision = runtime.select_story(
        "Aster guides Birch while Birch supports Cedar and Cedar follows "
        "Dune before Dune helps Elm today now."
    )
    assert decision.wire_completion
    assert decision.canonical_completion
    assert (
        decision.wire_completion
        != decision.canonical_completion
    )
    assert len(decision.extraction.mentions) == 3
    assert len(decision.extraction.generators) == 1
    assert set(
        decision.receipt["endpoint_selection_receipt_commitments"][
            "r00"
        ]
    ) == {"anchor", "object0", "object1"}
    assert set(decision.receipt["consumer_binding"]) == {
        "flat_label_no_verifier",
        "full",
        "legacy_keyword",
        "semantic_only",
    }
    assert decision.selected_answer_token_count > 0
    assert decision.wire_completion_token_count > 0
    assert model.calls


def test_fixed_sparse_canary_uses_full_reference_and_chunks() -> None:
    model = SparseFakeModel()
    runtime = _runtime(model)
    receipt = runtime.run_fixed_teacher_forced_canary()
    assert receipt["strategy"] == memory_safe.SPARSE_STRATEGY
    assert receipt[
        "short_strategy_vs_full_reference_exact"
    ] is True
    assert (
        receipt["short_strategy_microunits"]
        == receipt["short_full_reference_microunits"]
    )
    assert (
        receipt["long_answer_position_count"]
        > memory_safe.MAXIMUM_SPARSE_POSITIONS
    )
    assert receipt["sparse_chunk_count"] >= 2
    assert receipt["long_repeat_byte_exact"] is True
    assert receipt["free_form_generation_count"] == 0
    assert len(receipt["self_sha256"]) == 64
    assert any(
        len(call["positions"])
        > memory_safe.MAXIMUM_SPARSE_POSITIONS
        for call in model.calls
    ) is False


def test_fixed_fallback_canary_requires_independent_equivalence() -> None:
    model = CanaryFallbackFakeModel()
    runtime = _runtime(model)
    assert runtime.strategy == memory_safe.FALLBACK_STRATEGY
    receipt = runtime.run_fixed_teacher_forced_canary()
    assert receipt["strategy"] == memory_safe.FALLBACK_STRATEGY
    assert receipt[
        "fallback_independent_full_reference_passed"
    ] is True
    assert receipt[
        "short_strategy_vs_full_reference_exact"
    ] is True
    assert receipt["sparse_chunk_count"] == 0
    assert receipt["long_answer_position_count"] > 128
    assert receipt["long_repeat_byte_exact"] is True
    assert model.top_level_calls == 1


def test_contract_has_no_generation_call_and_fixed_memory_bounds() -> None:
    source = inspect.getsource(memory_safe)
    assert ".generate(" not in source
    assert "logits_to_keep=position_tensor" in source
    assert "hidden[:, -1:, :]" in source
    assert memory_safe.SCORING_BATCH_SIZE == 4
    assert (
        memory_safe.MAXIMUM_ANSWER_TOKENS
        == hierarchical.MAXIMUM_CONTEXT_TOKENS
    )
    assert memory_safe.MAXIMUM_SPARSE_POSITIONS == 128
    assert tuple(
        inspect.signature(
            memory_safe.load_exact_cuda_fp16_runtime
        ).parameters
    ) == ("model_root", "manifest")
    assert (
        memory_safe.SCORING_SEMANTICS[
            "free_form_generation_count"
        ]
        == 0
    )
    assert tuple(
        inspect.signature(
            memory_safe.MemorySafeQwenRuntime
            .run_fixed_teacher_forced_canary
        ).parameters
    ) == ("self",)
