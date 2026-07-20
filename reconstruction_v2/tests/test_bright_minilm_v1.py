from __future__ import annotations

import numpy as np
import pytest

from replication_runtime.bright_minilm_v1 import encoder


def test_quantized_scores_are_integer_and_stable() -> None:
    documents = np.zeros((3, encoder.EMBEDDING_DIMENSION), dtype=np.float32)
    query = np.zeros(encoder.EMBEDDING_DIMENSION, dtype=np.float32)
    documents[0, 0] = 1.0
    documents[1, 0] = 0.5
    documents[2, 1] = 1.0
    query[0] = 1.0
    scores = encoder.quantized_scores(documents, query)
    assert scores.dtype == np.int32
    assert scores.tolist() == [1_000_000, 500_000, 0]


def test_text_and_matrix_contract_fail_closed() -> None:
    with pytest.raises(encoder.BrightMiniLMError, match="count"):
        encoder._validate_texts([])
    with pytest.raises(encoder.BrightMiniLMError, match="text"):
        encoder._validate_texts(["\x00"])
    with pytest.raises(encoder.BrightMiniLMError, match="shape"):
        encoder.quantized_scores(np.zeros((2, 3)), np.zeros(3))
    with pytest.raises(encoder.BrightMiniLMError, match="shape"):
        encoder.float32_matrix_sha256(np.zeros((2, 3)))
