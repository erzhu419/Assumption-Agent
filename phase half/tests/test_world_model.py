"""
Phase 0.5: Test world model using Phase 0 data as seed.

Tests:
1. Feature discretizer correctness
2. Statistical model with synthetic data
3. OOD detector behavior
4. Hybrid model integration (with LLM simulator if API available)
5. Cold-start validation against Phase 0 annotations
6. Binary reward consistency check

Usage:
    python test_world_model.py              # Run all tests
    python test_world_model.py --skip-llm   # Skip LLM-dependent tests
"""

import sys
import json
import argparse
import random
import numpy as np
from pathlib import Path

# Add module paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phase zero" / "scripts"))

from world_model import (
    HybridWorldModel, StatisticalWorldModel, WorldModelPrediction,
    OODDetector, discretize, DiscreteState, build_full_state_space,
    WorldModelCalibrator,
)

PHASE0_DIR = Path(__file__).parent.parent.parent / "phase zero"
KB_DIR = PHASE0_DIR / "kb" / "strategies"
PROBLEMS_DIR = PHASE0_DIR / "benchmark" / "problems"
ANNOTATIONS_DIR = PHASE0_DIR / "benchmark" / "annotations"

passed = 0
failed = 0

def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        print(f"  ✓ {name}")
        passed += 1
    else:
        print(f"  ✗ {name} — {detail}")
        failed += 1


# =========================================================================
# Test 1: Feature Discretizer
# =========================================================================

def test_discretizer():
    print("\n=== Test 1: Feature Discretizer ===")

    # Basic discretization
    state = discretize({
        "coupling_estimate": 0.2,
        "decomposability": 0.8,
        "has_baseline": True,
        "information_completeness": 0.5,
        "component_count": 3,
    })
    check("coupling low", state.coupling == "low")
    check("decomp high", state.decomposability == "high")
    check("baseline yes", state.has_baseline == "yes")
    check("info medium", state.info_completeness == "medium")
    check("components few", state.component_count == "few")

    # Full state space
    states = build_full_state_space()
    check("state space size", len(states) == 108, f"got {len(states)}")

    # Index uniqueness
    indices = [s.to_index() for s in states]
    check("index uniqueness", len(set(indices)) == 108)
    check("index range", min(indices) == 0 and max(indices) == 107)


# =========================================================================
# Test 2: Statistical Model
# =========================================================================

def test_statistical_model():
    print("\n=== Test 2: Statistical Model ===")

    model = StatisticalWorldModel()

    # Empty model returns 0.5 with zero confidence
    state = discretize({"coupling_estimate": 0.5, "decomposability": 0.5,
                        "has_baseline": False, "information_completeness": 0.5,
                        "component_count": 5})
    pred = model.predict(state, "S01")
    check("empty model prob", abs(pred.predicted_success_probability - 0.5) < 0.01)
    check("empty model low confidence", pred.prediction_confidence < 0.1)

    # Add some data
    features = {"coupling_estimate": 0.2, "decomposability": 0.8,
                "has_baseline": True, "information_completeness": 0.6,
                "component_count": 3}
    for _ in range(15):
        model.update(features, "S01", success=True)
    for _ in range(5):
        model.update(features, "S01", success=False)

    state = discretize(features)
    pred = model.predict(state, "S01")
    check("updated prob ~0.75", 0.6 < pred.predicted_success_probability < 0.9,
          f"got {pred.predicted_success_probability:.2f}")
    check("updated confidence > 0", pred.prediction_confidence > 0.2)

    # Quality filter: low confidence selector should be ignored
    model2 = StatisticalWorldModel()
    model2.update(features, "S02", success=True, selector_confidence=0.1)
    pred2 = model2.predict(discretize(features), "S02")
    check("quality filter blocks low confidence",
          pred2.prediction_confidence == 0.0,
          f"confidence={pred2.prediction_confidence}")

    # Quality filter: simulated data should be ignored
    model3 = StatisticalWorldModel()
    model3.update(features, "S03", success=True, is_simulated=True)
    check("quality filter blocks simulated",
          model3.get_coverage_stats()["total_records"] == 0)


# =========================================================================
# Test 3: OOD Detector
# =========================================================================

def test_ood_detector():
    print("\n=== Test 3: OOD Detector ===")

    ood = OODDetector()

    # Empty detector: everything is OOD
    result = ood.check({"coupling_estimate": 0.5, "decomposability": 0.5,
                        "has_baseline": False, "information_completeness": 0.5,
                        "component_count": 5})
    check("empty detector flags OOD", result.is_ood)

    # Add some observations
    for _ in range(30):
        ood.update({"coupling_estimate": random.uniform(0.1, 0.3),
                    "decomposability": random.uniform(0.6, 0.9),
                    "has_baseline": True,
                    "information_completeness": random.uniform(0.5, 0.8),
                    "component_count": random.randint(2, 4)})

    # Same region: should not be OOD
    result = ood.check({"coupling_estimate": 0.2, "decomposability": 0.7,
                        "has_baseline": True, "information_completeness": 0.6,
                        "component_count": 3})
    check("known region not OOD", not result.is_ood)

    # Very different region: should be OOD (different discrete state)
    result = ood.check({"coupling_estimate": 0.9, "decomposability": 0.1,
                        "has_baseline": False, "information_completeness": 0.1,
                        "component_count": 15})
    check("unknown region is OOD", result.is_ood)


# =========================================================================
# Test 4: Hybrid Model Binary Reward Consistency
# =========================================================================

def test_binary_reward():
    print("\n=== Test 4: Binary Reward Consistency ===")

    model = HybridWorldModel()

    # Add enough data so statistical model kicks in
    features = {"coupling_estimate": 0.2, "decomposability": 0.8,
                "has_baseline": True, "information_completeness": 0.6,
                "component_count": 3}
    for _ in range(20):
        model.update(features, "S01", success=True)

    # Simulate multiple times
    scores = []
    for _ in range(100):
        result = model.simulate_execution(features, "S01")
        scores.append(result["evaluation_score"])
        # Binary check: evaluation_score must be in {0.0, 0.5, 1.0}
        check_val = result["evaluation_score"] in (0.0, 0.5, 1.0)
        if not check_val:
            check("binary score", False, f"got {result['evaluation_score']}")
            return

    check("all scores binary", True)
    check("predicted_probability preserved",
          all("predicted_probability" in model.simulate_execution(features, "S01")
              for _ in range(3)))

    # Check that scores are not all the same (probabilistic)
    unique_scores = set(scores)
    check("score variety (not all same)", len(unique_scores) >= 2,
          f"unique scores: {unique_scores}")


# =========================================================================
# Test 5: Cold-Start with Phase 0 Data
# =========================================================================

def test_cold_start():
    print("\n=== Test 5: Cold-Start with Phase 0 Annotations ===")

    # Load strategies
    strategies = {}
    for f in sorted(KB_DIR.glob("S*.json")):
        d = json.load(open(f, encoding="utf-8"))
        strategies[d["id"]] = d

    check("loaded strategies", len(strategies) >= 25, f"got {len(strategies)}")

    # Load annotations to build a simple "prior" for the statistical model
    model = HybridWorldModel(strategy_kb=strategies)

    # Use annotations as pseudo-experience:
    # If annotators chose S_XX for a problem, treat it as "S_XX would succeed"
    seed_count = 0
    for f in sorted(ANNOTATIONS_DIR.glob("*.json")):
        ann = json.load(open(f, encoding="utf-8"))
        # We don't have real features, so use synthetic ones based on domain
        domain = ann.get("domain", "unknown")
        features = _domain_to_features(domain)

        for a in ann.get("annotations", []):
            for sid in a.get("selected_strategies", [])[:1]:  # top-1 only
                model.update(features, sid, success=True,
                             selector_confidence=0.8)
                seed_count += 1

    check("seeded records", seed_count > 100, f"got {seed_count}")

    stats = model.get_stats()
    check("statistical cells populated",
          stats["statistical"]["non_empty_cells"] > 10,
          f"got {stats['statistical']['non_empty_cells']}")

    # Test prediction for a known combination
    pred = model.predict(
        _domain_to_features("software_engineering"),
        "S02",  # S02 is most popular
        strategy_name="分而治之",
    )
    check("prediction for S02/software",
          pred.predicted_success_probability > 0.3,
          f"got {pred.predicted_success_probability:.2f}")
    check("prediction source not nodata",
          pred.source != "statistical_nodata",
          f"source={pred.source}")


def _domain_to_features(domain: str) -> dict:
    """Generate synthetic problem features from domain name."""
    domain_features = {
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
    }
    return domain_features.get(domain, {"coupling_estimate": 0.5, "decomposability": 0.5,
                                         "has_baseline": False, "information_completeness": 0.5,
                                         "component_count": 5})


# =========================================================================
# Test 6: LLM Simulator (optional)
# =========================================================================

def test_llm_simulator():
    print("\n=== Test 6: LLM Simulator ===")

    strategies = {}
    for f in sorted(KB_DIR.glob("S*.json")):
        d = json.load(open(f, encoding="utf-8"))
        strategies[d["id"]] = d

    model = HybridWorldModel(strategy_kb=strategies)

    state = discretize({
        "coupling_estimate": 0.2, "decomposability": 0.8,
        "has_baseline": True, "information_completeness": 0.7,
        "component_count": 3,
    })

    # LLM prediction (statistical has no data, so it should fall through to LLM)
    pred = model.predict(
        {"coupling_estimate": 0.2, "decomposability": 0.8,
         "has_baseline": True, "information_completeness": 0.7,
         "component_count": 3},
        "S01",
        strategy_name="控制变量法",
        strategy_description="固定其他条件，每次只改变一个因素",
    )

    check("LLM pred probability valid",
          0.0 <= pred.predicted_success_probability <= 1.0,
          f"got {pred.predicted_success_probability}")
    check("LLM pred has source",
          pred.source in ("llm", "statistical_nodata", "statistical_knn", "ood"),
          f"source={pred.source}")

    # Sequence simulation
    seq = model.simulate_strategy_sequence(
        {"coupling_estimate": 0.5, "decomposability": 0.5,
         "has_baseline": False, "information_completeness": 0.5,
         "component_count": 5},
        ["S06", "S01"],
        strategy_kb=strategies,
    )
    check("sequence has steps", len(seq.steps) >= 1)
    check("overall prob valid",
          0.0 <= seq.overall_success_probability <= 1.0)


# =========================================================================
# Main
# =========================================================================

def main():
    global passed, failed
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip tests that require LLM API calls")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    test_discretizer()
    test_statistical_model()
    test_ood_detector()
    test_binary_reward()
    test_cold_start()

    if not args.skip_llm:
        test_llm_simulator()
    else:
        print("\n=== Test 6: LLM Simulator (SKIPPED) ===")

    print(f"\n{'='*50}")
    print(f"Passed: {passed}, Failed: {failed}")
    if failed == 0:
        print("All tests passed!")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
