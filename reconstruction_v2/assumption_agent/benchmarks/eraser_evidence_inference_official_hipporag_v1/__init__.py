"""Item-local, retrieve-only official HippoRAG adapter for Evidence Inference."""

from .adapter import run_item_local_official_hipporag_v1
from .contract import EraserEvidenceInferenceOfficialHippoRAGError

__all__ = [
    "EraserEvidenceInferenceOfficialHippoRAGError",
    "run_item_local_official_hipporag_v1",
]
