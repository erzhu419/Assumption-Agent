"""
Build sentence-transformer embeddings for signal-level retrieval.

Pools:
  - All triggers from trigger_library_v6.json (flat, 301)
  - All wisdom entries from wisdom_library.json (aphorism + unpacked_for_llm combined)

Model: paraphrase-multilingual-MiniLM-L12-v2 (Chinese-capable, 384-dim, small)
Cached to phase two/analysis/cache/signal_embeddings.npz
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CACHE = Path("/home/erzhu419/mine_code/Asumption Agent/phase two/analysis/cache")
TRIG_PATH = CACHE / "trigger_library_v6.json"
WIS_PATH = CACHE / "wisdom_library.json"
OUT_PATH = CACHE / "signal_embeddings.npz"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def main():
    print(f"Loading model {MODEL_NAME}...")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"  loaded ({time.time()-t0:.1f}s)")

    triggers_db = json.loads(TRIG_PATH.read_text(encoding="utf-8"))
    trigger_texts = []
    trigger_cats = []
    for cat, ts in triggers_db.items():
        for t in ts:
            trigger_texts.append(t)
            trigger_cats.append(cat)
    print(f"  triggers: {len(trigger_texts)} flat")

    wisdom = json.loads(WIS_PATH.read_text(encoding="utf-8"))
    wisdom_texts = [f"{e['aphorism']}。{e['unpacked_for_llm']}" for e in wisdom]
    wisdom_ids = [e["id"] for e in wisdom]
    print(f"  wisdom: {len(wisdom_texts)} entries")

    print("Encoding...")
    trig_emb = model.encode(trigger_texts, normalize_embeddings=True,
                             show_progress_bar=True, batch_size=32)
    wis_emb = model.encode(wisdom_texts, normalize_embeddings=True,
                            show_progress_bar=True, batch_size=32)

    # Also embed sample problems
    sample = json.loads((CACHE / "sample_100.json").read_text(encoding="utf-8"))
    prob_texts = [p.get("description", "") for p in sample]
    prob_ids = [p["problem_id"] for p in sample]
    prob_emb = model.encode(prob_texts, normalize_embeddings=True,
                             show_progress_bar=True, batch_size=32)
    print(f"  problems: {len(prob_texts)} encoded")

    np.savez(OUT_PATH,
             trigger_emb=trig_emb,
             trigger_texts=np.array(trigger_texts, dtype=object),
             trigger_cats=np.array(trigger_cats, dtype=object),
             wisdom_emb=wis_emb,
             wisdom_ids=np.array(wisdom_ids, dtype=object),
             problem_emb=prob_emb,
             problem_ids=np.array(prob_ids, dtype=object))
    print(f"Saved to {OUT_PATH.name} ({time.time()-t0:.0f}s total)")
    print(f"  shapes: trig={trig_emb.shape} wis={wis_emb.shape} prob={prob_emb.shape}")


if __name__ == "__main__":
    main()
