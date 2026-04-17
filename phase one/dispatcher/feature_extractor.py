"""
Phase 1: Problem Feature Extractor.
Converts natural language problem descriptions to structured feature vectors.
"""

import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phase zero" / "scripts"))
from llm_client import create_client, parse_json_from_llm


FEATURE_EXTRACTION_PROMPT = """你是一个问题分析专家。分析以下问题的结构特征。

问题描述：
{problem_description}

输出 JSON（不要代码块标记）：
{{"domain": "software_engineering/mathematics/science/business/daily_life/engineering",
"coupling_estimate": 0.0到1.0的浮点数,
"decomposability": 0.0到1.0的浮点数,
"has_baseline": true或false,
"randomness_level": 0.0到1.0的浮点数,
"information_completeness": 0.0到1.0的浮点数,
"component_count": 整数,
"constraint_count": 整数,
"reversibility": 0.0到1.0的浮点数,
"difficulty": "easy/medium/hard"}}"""


class FeatureExtractor:
    """
    Extract structured features from problem descriptions.
    Uses LLM for structural features + sentence embedding for text features.
    """

    def __init__(self, use_llm: bool = True, cache_path: str = None):
        self.use_llm = use_llm
        self._client = None
        self._encoder = None

        # Load pre-computed embedding cache if available
        self._embedding_cache = {}
        import _config as settings
        cp = cache_path or str(settings.PROJECT_ROOT / "cache" / "embeddings.npz")
        try:
            from pathlib import Path
            if Path(cp).exists():
                data = np.load(cp, allow_pickle=True)
                ids = data["ids"]
                embeddings = data["embeddings"]
                self._embedding_cache = {str(pid): emb for pid, emb in zip(ids, embeddings)}
                print(f"  Loaded {len(self._embedding_cache)} cached embeddings")
        except Exception:
            pass

    @property
    def client(self):
        if self._client is None:
            self._client = create_client()
        return self._client

    def extract(self, problem_description: str, problem_id: str = None) -> Dict:
        """
        Extract features from a problem description.
        Returns dict with structural features + text embedding.
        """
        if self.use_llm:
            structural = self._extract_structural_llm(problem_description)
        else:
            structural = self._extract_structural_heuristic(problem_description)

        embedding = self._get_embedding(problem_description, problem_id=problem_id)

        return {
            **structural,
            "embedding": embedding,
        }

    def extract_batch(self, descriptions: List[str]) -> List[Dict]:
        """Extract features for multiple problems."""
        return [self.extract(d) for d in descriptions]

    def _extract_structural_llm(self, description: str) -> Dict:
        """Use LLM to extract structural features."""
        prompt = FEATURE_EXTRACTION_PROMPT.format(
            problem_description=description[:1000]  # Truncate if too long
        )

        try:
            response = self.client.generate(prompt, max_tokens=256, temperature=0.1)
            features = parse_json_from_llm(response["text"])

            # Normalize and validate
            return {
                "domain": str(features.get("domain", "unknown")),
                "coupling_estimate": float(np.clip(features.get("coupling_estimate", 0.5), 0, 1)),
                "decomposability": float(np.clip(features.get("decomposability", 0.5), 0, 1)),
                "has_baseline": bool(features.get("has_baseline", False)),
                "randomness_level": float(np.clip(features.get("randomness_level", 0.5), 0, 1)),
                "information_completeness": float(np.clip(features.get("information_completeness", 0.5), 0, 1)),
                "component_count": int(max(1, features.get("component_count", 5))),
                "constraint_count": int(max(0, features.get("constraint_count", 3))),
                "reversibility": float(np.clip(features.get("reversibility", 0.5), 0, 1)),
                "difficulty": features.get("difficulty", "medium"),
            }
        except Exception:
            return self._extract_structural_heuristic(description)

    def _extract_structural_heuristic(self, description: str) -> Dict:
        """Fallback: rule-based feature extraction."""
        text = description.lower()
        return {
            "domain": "unknown",
            "coupling_estimate": 0.5,
            "decomposability": 0.6 if any(w in text for w in ["部分", "模块", "组件"]) else 0.4,
            "has_baseline": any(w in text for w in ["基准", "正常", "之前可以"]),
            "randomness_level": 0.5,
            "information_completeness": 0.5,
            "component_count": 5,
            "constraint_count": 3,
            "reversibility": 0.5,
            "difficulty": "medium",
        }

    def _get_embedding(self, text: str, problem_id: str = None) -> np.ndarray:
        """Get text embedding. Priority: cache > model > hash fallback."""
        import _config as settings

        # 1. Check pre-computed cache (instant)
        if problem_id and problem_id in self._embedding_cache:
            return self._embedding_cache[problem_id]

        # Also try matching by text hash in case problem_id not available
        text_key = str(hash(text) % (2**31))
        if text_key in self._embedding_cache:
            return self._embedding_cache[text_key]

        # 2. Cache miss: use zero vector (fast) instead of loading model (slow)
        # sentence-transformer takes 8+ seconds to load and blocks training
        # Run precompute_embeddings.py first to avoid this path
        return np.zeros(settings.EMBEDDING_DIM, dtype=np.float32)

    def features_to_vector(self, features: Dict, kb_match_scores: np.ndarray = None,
                           cross_problem_stats: np.ndarray = None) -> np.ndarray:
        """
        Convert extracted features to a flat numpy vector for the MLP dispatcher.
        """
        import _config as settings

        # 1. Text embedding (768)
        embedding = features.get("embedding", np.zeros(settings.EMBEDDING_DIM))
        if len(embedding) != settings.EMBEDDING_DIM:
            embedding = np.zeros(settings.EMBEDDING_DIM)

        # 2. Structural features (10)
        structural = np.array([
            features.get("coupling_estimate", 0.5),
            features.get("decomposability", 0.5),
            float(features.get("has_baseline", False)),
            features.get("randomness_level", 0.5),
            features.get("information_completeness", 0.5),
            min(features.get("component_count", 5) / 20.0, 1.0),
            min(features.get("constraint_count", 3) / 10.0, 1.0),
            features.get("reversibility", 0.5),
            {"easy": 0.0, "medium": 0.5, "hard": 1.0}.get(
                features.get("difficulty", "medium"), 0.5),
            0.5,  # placeholder
        ], dtype=np.float32)

        # 3. KB match scores (NUM_ACTIONS)
        if kb_match_scores is None:
            kb_match_scores = np.zeros(settings.NUM_ACTIONS, dtype=np.float32)

        # 4. Cross-problem context (NUM_ACTIONS)
        if cross_problem_stats is None:
            cross_problem_stats = np.full(settings.NUM_ACTIONS, 0.5, dtype=np.float32)

        return np.concatenate([embedding, structural, kb_match_scores, cross_problem_stats])
