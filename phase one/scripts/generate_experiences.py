"""
Generate experience records by running trained dispatchers on all problems.

Loads all three trained checkpoints (DQN / SAC / RE-SAC), runs each on every
problem once, and writes ExecutionRecords to phase zero/experience_log/executions/.

Using three dispatchers gives natural success/failure diversity per problem —
exactly the contrastive signal Phase 2's distiller needs.

Usage:
    python generate_experiences.py                    # all 3 algos, all splits
    python generate_experiences.py --algos resac      # only RE-SAC
    python generate_experiences.py --split test       # only test set
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent / "phase zero" / "scripts"))

import _config as cfg
from experience_writer.exp_writer import ExperienceWriter


def load_dispatcher(algo: str, ckpt_path: Path):
    if algo == "dqn":
        from dispatcher.torch_dispatcher import DQNDispatcher
        d = DQNDispatcher(input_dim=cfg.INPUT_DIM, num_actions=cfg.NUM_ACTIONS)
    elif algo == "sac":
        from dispatcher.torch_dispatcher import SACDiscreteDispatcher
        d = SACDiscreteDispatcher(input_dim=cfg.INPUT_DIM, num_actions=cfg.NUM_ACTIONS)
    elif algo == "resac":
        from dispatcher.resac_discrete import RESACDiscreteDispatcher
        d = RESACDiscreteDispatcher(
            input_dim=cfg.INPUT_DIM, num_actions=cfg.NUM_ACTIONS, ensemble_size=5
        )
    else:
        raise ValueError(algo)
    d.load(str(ckpt_path))
    d.training = False
    return d


def run_one(dispatcher, algo: str, problems, task_env, extractor, writer: ExperienceWriter,
            structural_cache=None):
    count = 0
    t0 = time.time()
    for p in problems:
        task_env._current_task = p
        desc = p.get("description", "")
        features = extractor.extract(desc, problem_id=p.get("problem_id"))
        vec = extractor.features_to_vector(features)

        # Sample stochastically for SAC/RE-SAC to produce diversity;
        # DQN uses argmax (epsilon=0 after training)
        action = dispatcher.select_action(vec, cfg.ACTION_SPACE)

        outcome = task_env.evaluate_strategy_selection(
            p["problem_id"], action.strategy_id, action.confidence
        )

        # complexity_features the distiller will reason about
        # Prefer the precomputed structural cache (richer signal) over the
        # dispatcher's feature extractor (which uses zero vectors when LLM off).
        cached = structural_cache.get(p["problem_id"]) if structural_cache else None
        if cached:
            complexity = dict(cached)
        else:
            complexity = {
                "domain": features.get("domain", "unknown"),
                "coupling_estimate": float(features.get("coupling_estimate", 0.5)),
                "decomposability": float(features.get("decomposability", 0.5)),
                "has_baseline": bool(features.get("has_baseline", False)),
                "randomness_level": float(features.get("randomness_level", 0.5)),
                "information_completeness": float(features.get("information_completeness", 0.5)),
                "component_count": int(features.get("component_count", 5)),
                "constraint_count": int(features.get("constraint_count", 3)),
                "reversibility": float(features.get("reversibility", 0.5)),
                "difficulty": features.get("difficulty", "medium"),
            }

        task_for_record = {
            "problem_id": p["problem_id"],
            "description": desc,
            "domain": p.get("domain", "unknown"),
            "difficulty": p.get("difficulty", "medium"),
            "complexity_features": complexity,
        }

        writer.write(
            task=task_for_record,
            selected_strategy=action.strategy_id,
            selector_confidence=float(action.confidence),
            alternatives=[],
            outcome_success=outcome.success,
            evaluation_score=outcome.evaluation_score,
            consistency_score=outcome.consistency_score,
            failure_reason=outcome.failure_reason,
            wall_clock_seconds=0.0,
            steps_taken=outcome.steps_taken,
            extra_metadata={
                "dispatcher_algo": algo,
                "partial_success": outcome.partial_success,
            },
        )
        count += 1

    elapsed = time.time() - t0
    print(f"  [{algo}] wrote {count} records in {elapsed:.0f}s")
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", default=["dqn", "sac", "resac"])
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading data...")
    strategy_kb = {}
    for f in sorted(cfg.KB_DIR.glob("S*.json")):
        d = json.load(open(f, encoding="utf-8"))
        strategy_kb[d["id"]] = d

    from task_env.base_env import TaskEnvironment
    task_env = TaskEnvironment(strategy_kb=strategy_kb)
    print(f"  {task_env.stats()}")

    from dispatcher.feature_extractor import FeatureExtractor
    extractor = FeatureExtractor(use_llm=False)

    # Load cached structural features (from precompute_features.py)
    feature_cache_path = PROJECT / "cache" / "features.json"
    structural_cache = {}
    if feature_cache_path.exists():
        structural_cache = json.loads(feature_cache_path.read_text(encoding="utf-8"))
        print(f"  Loaded {len(structural_cache)} cached structural features")
    else:
        print(f"  WARNING: {feature_cache_path} missing; run precompute_features.py first")

    # Output dir: phase zero/experience_log/executions/
    out_dir = cfg.PHASE0_DIR / "experience_log" / "executions"
    writer = ExperienceWriter(out_dir)
    print(f"Writing to {out_dir}")

    total = 0
    for algo in args.algos:
        ckpt = PROJECT / "checkpoints" / f"dispatcher_{algo}.pt"
        if not ckpt.exists():
            print(f"  [skip] checkpoint missing: {ckpt}")
            continue
        print(f"\nLoading {algo} from {ckpt.name}...")
        dispatcher = load_dispatcher(algo, ckpt)

        for split in args.splits:
            problems = task_env.get_all_problems(split)
            print(f"  split={split} ({len(problems)} problems)")
            total += run_one(dispatcher, algo, problems, task_env, extractor, writer,
                            structural_cache=structural_cache)

    print(f"\nTotal records: {total}")
    print(f"Saved under {out_dir}")


if __name__ == "__main__":
    main()
