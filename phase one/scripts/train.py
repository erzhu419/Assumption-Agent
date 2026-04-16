"""
Phase 1: Main training script.

Usage:
    python train.py                          # Train with world model
    python train.py --episodes 1000          # Short training run
    python train.py --no-world-model         # Train without world model (model-free)
    python train.py --eval-only              # Only run evaluation
    python train.py --skip-llm-features      # Use heuristic features (no LLM calls)
"""

import sys
import json
import argparse
import random
import numpy as np
from pathlib import Path

# Add all module paths
PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent / "phase zero" / "scripts"))
sys.path.insert(0, str(PROJECT.parent / "phase half"))

import _config as phase_one_config


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Train strategy dispatcher")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--no-world-model", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-llm-features", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default=str(PROJECT / "checkpoints" / "dispatcher.npz"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # --- Load strategy KB ---
    print("Loading strategy KB...")
    strategy_kb = {}
    for f in sorted(phase_one_config.KB_DIR.glob("S*.json")):
        d = json.load(open(f, encoding="utf-8"))
        strategy_kb[d["id"]] = d
    print(f"  Loaded {len(strategy_kb)} strategies")

    # --- Build task environment ---
    print("Loading task environment...")
    from task_env.base_env import TaskEnvironment
    task_env = TaskEnvironment(strategy_kb=strategy_kb)
    print(f"  {task_env.stats()}")

    # --- Build feature extractor ---
    print("Building feature extractor...")
    from dispatcher.feature_extractor import FeatureExtractor
    extractor = FeatureExtractor(use_llm=not args.skip_llm_features)

    # --- Build world model ---
    world_model = None
    if not args.no_world_model:
        print("Building world model...")
        from world_model import HybridWorldModel
        world_model = HybridWorldModel(strategy_kb=strategy_kb)
        # Seed with Phase 0 annotation data
        _seed_world_model(world_model, task_env)
        print(f"  World model stats: {world_model.get_stats()}")

    # --- Build dispatcher ---
    print("Building MLP dispatcher...")
    from dispatcher.mlp_dispatcher import MLPDispatcher
    dispatcher = MLPDispatcher(
        input_dim=phase_one_config.INPUT_DIM,
        num_actions=phase_one_config.NUM_ACTIONS,
    )
    print(f"  Parameters: {dispatcher.param_count():,}")

    if args.eval_only:
        # Load checkpoint if exists
        ckpt = Path(args.save_path)
        if ckpt.exists():
            dispatcher.load(str(ckpt))
            print(f"  Loaded checkpoint: {ckpt}")
        else:
            print("  No checkpoint found, evaluating untrained dispatcher")

        from evaluation.evaluate import run_full_evaluation
        results = run_full_evaluation(dispatcher, task_env, extractor)
        return

    # --- Train ---
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)

    from training.ppo_trainer import PPOTrainer
    trainer = PPOTrainer(
        dispatcher=dispatcher,
        task_env=task_env,
        feature_extractor=extractor,
        world_model=world_model,
        config={**phase_one_config.PPO_CONFIG,
                **({"max_episodes": args.episodes} if args.episodes else {})},
    )

    stats = trainer.train()

    # --- Save ---
    save_dir = Path(args.save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    dispatcher.save(args.save_path)
    print(f"\nDispatcher saved to {args.save_path}")

    # --- Evaluate ---
    print("\n" + "="*50)
    from evaluation.evaluate import run_full_evaluation
    results = run_full_evaluation(dispatcher, task_env, extractor)

    # Save results
    results_path = PROJECT / "analysis" / "training_results.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "training_stats": {
                "episodes": stats.episode,
                "avg_reward": stats.avg_reward_100,
                "val_accuracy": stats.top1_accuracy,
                "collapse_detected": stats.collapse_detected,
            },
            "evaluation": {
                "dispatcher_accuracy": results["dispatcher"]["top1_accuracy"],
                "random_baseline": results["baselines"]["random"]["accuracy"],
                "most_frequent_baseline": results["baselines"]["most_frequent"]["accuracy"],
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {results_path}")


def _seed_world_model(world_model, task_env):
    """Seed world model with Phase 0 annotation pseudo-experience."""
    annotations_dir = phase_one_config.ANNOTATIONS_DIR
    count = 0
    for f in sorted(annotations_dir.glob("*.json")):
        ann = json.load(open(f, encoding="utf-8"))
        domain = ann.get("domain", "unknown")
        features = _domain_to_features(domain)

        for a in ann.get("annotations", []):
            for sid in a.get("selected_strategies", [])[:1]:
                world_model.update(features, sid, success=True,
                                   selector_confidence=0.8)
                count += 1
    print(f"  Seeded world model with {count} pseudo-experiences")


def _domain_to_features(domain):
    return {
        "software_engineering": {"coupling_estimate": 0.5, "decomposability": 0.7,
                                 "has_baseline": True, "information_completeness": 0.6,
                                 "component_count": 6},
        "mathematics": {"coupling_estimate": 0.3, "decomposability": 0.6,
                        "has_baseline": False, "information_completeness": 0.8,
                        "component_count": 3},
        "science": {"coupling_estimate": 0.4, "decomposability": 0.5,
                     "has_baseline": True, "information_completeness": 0.5,
                     "component_count": 5},
        "business": {"coupling_estimate": 0.6, "decomposability": 0.4,
                     "has_baseline": False, "information_completeness": 0.4,
                     "component_count": 7},
        "daily_life": {"coupling_estimate": 0.3, "decomposability": 0.7,
                       "has_baseline": True, "information_completeness": 0.7,
                       "component_count": 3},
        "engineering": {"coupling_estimate": 0.5, "decomposability": 0.6,
                        "has_baseline": True, "information_completeness": 0.5,
                        "component_count": 8},
    }.get(domain, {"coupling_estimate": 0.5, "decomposability": 0.5,
                    "has_baseline": False, "information_completeness": 0.5,
                    "component_count": 5})


if __name__ == "__main__":
    main()
