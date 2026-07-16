"""Frozen offline NLI runtime used by the QASC direct-action study."""

from .binding import (
    NLIWorkerPool,
    run_canary,
    score_pairs_in_subprocess,
    verify_runtime_asset,
    verify_runtime_binding,
)
from .contract import NLIPair, QASCNLIError

__all__ = [
    "NLIPair",
    "NLIWorkerPool",
    "QASCNLIError",
    "run_canary",
    "score_pairs_in_subprocess",
    "verify_runtime_asset",
    "verify_runtime_binding",
]
