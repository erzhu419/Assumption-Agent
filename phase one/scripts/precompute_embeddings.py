"""
Pre-compute sentence embeddings for all problems.
Run once, then training loads from cache (instant).

Usage:
    python precompute_embeddings.py
"""

import sys
import json
import numpy as np
import time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent / "phase zero" / "scripts"))

import _config as cfg


def main():
    print("Loading sentence-transformer model...")
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # Load all problems
    problems = []
    for f in sorted(cfg.PROBLEMS_DIR.glob("*.json")):
        if "error" in f.name:
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(data, list):
            problems.extend(data)
    print(f"  {len(problems)} problems loaded")

    # Encode all descriptions in one batch (much faster than one-by-one)
    print("Encoding all problem descriptions...")
    t0 = time.time()
    descriptions = [p["description"] for p in problems]
    embeddings = model.encode(descriptions, batch_size=64, show_progress_bar=True)
    print(f"  Encoded in {time.time()-t0:.1f}s, shape={embeddings.shape}")

    # Save as {description_hash: embedding} mapping
    # Use description text hash as key (robust to ID format changes)
    cache = {}
    for p, emb in zip(problems, embeddings):
        # Store by both problem_id AND description hash for robustness
        cache[p["problem_id"]] = emb
        desc_hash = str(hash(p["description"]) % (2**31))
        cache[desc_hash] = emb

    cache_path = PROJECT / "cache" / "embeddings.npz"
    cache_path.parent.mkdir(exist_ok=True)
    np.savez_compressed(str(cache_path),
                        ids=np.array(list(cache.keys())),
                        embeddings=np.array(list(cache.values())))
    print(f"  Saved to {cache_path} ({cache_path.stat().st_size / 1024:.0f} KB)")
    print(f"  {len(cache)} embeddings cached")


if __name__ == "__main__":
    main()
