from __future__ import annotations

import argparse
import json
from pathlib import Path

from .events import JsonlEventSink
from .models import HypothesisKind, ResidualExample, SplitName
from .provider_chain import build_proposal_model
from .proposer import StructuredHypothesisProposer
from .secure_env import load_dotenv, map_legacy_model_env
from .validation import (
    EvaluatorEpochCheck,
    RecursiveValidationEngine,
    RuntimeCandidateKindCheck,
    RuntimeActionCheck,
    SchemaCheck,
    TrainingSupportCheck,
    TriggerVocabularyCheck,
    ValidationContext,
    build_trigger_feature_catalog,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a no-gold model proposal and recursive-validation canary.")
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    args = parser.parse_args()

    load_dotenv(args.env_file)
    presence = map_legacy_model_env()
    sink = JsonlEventSink(args.events)
    model = build_proposal_model(event_sink=sink)
    proposer = StructuredHypothesisProposer(model, event_sink=sink)
    residuals = tuple(
        ResidualExample(
            transition_id=f"canary-transition-{index}",
            task_id=f"canary-train-{index}",
            family="controlled_relation",
            split=SplitName.TRAIN,
            features={
                "relation_type": "controlled_comparison",
                "self_contained": True,
                "requires_live_source": False,
            },
            failure_type="baseline_missed_explicit_relation",
            evaluator_feedback=(
                "The selected path did not explicitly compare the controlled relation.",
                "A valid repair must preserve the baseline when its trigger is absent.",
            ),
            baseline_success=False,
        )
        for index in range(3)
    )
    capabilities = {
        "available_lanes": ["raw", "relation_solver"],
        "baseline_lane": "raw",
        "allowed_operator_actions": [
            "execute_step",
            "check_condition",
            "enable_lane",
            "prioritize_lane",
            "require_verifier",
        ],
        "runtime_trigger_contract": {
            "allowed_feature_catalog": build_trigger_feature_catalog(residuals),
            "forbidden_context_only_keys": ["task_instruction"],
            "context_is_for_action_design_only": True,
        },
    }
    root = proposer.propose(
        residuals,
        evaluator_epoch="canary-epoch-0",
        max_hypotheses=1,
        capabilities=capabilities,
        trace_id="live-proposal-canary",
    )[0]
    validator = RecursiveValidationEngine(
        [
            SchemaCheck(),
            RuntimeCandidateKindCheck(),
            TriggerVocabularyCheck(),
            TrainingSupportCheck(min_support=2),
            RuntimeActionCheck(),
            EvaluatorEpochCheck(),
        ],
        proposer=proposer,
        event_sink=sink,
    )
    tree = validator.validate(
        root,
        ValidationContext(
            evaluator_epoch="canary-epoch-0",
            residuals=residuals,
            available_lanes=frozenset({"raw", "relation_solver"}),
            baseline_lane="raw",
            allowed_runtime_kinds=frozenset(
                {HypothesisKind.TASK, HypothesisKind.POLICY}
            ),
            trigger_feature_catalog=build_trigger_feature_catalog(residuals),
        ),
        trace_id="live-proposal-canary",
    )
    payload = {
        "canary_version": "proposal_canary_v1",
        "model": presence["model"],
        "provider_chain": list(model.provider_ids),
        "provider_chain_hash": model.chain_hash,
        "root_hypothesis_id": root.id,
        "root_hypothesis_hash": root.payload_hash,
        "recursive_node_count": len(tree.nodes),
        "recursive_depth": tree.recursion_depth,
        "accepted": tree.accepted_program is not None,
        "accepted_program": tree.accepted_program.to_dict() if tree.accepted_program else None,
        "nodes": [
            {
                "hypothesis_id": node.program.id,
                "hypothesis_hash": node.program.payload_hash,
                "depth": node.depth,
                "passed": node.passed,
                "checks": [result.to_dict() for result in node.checks],
                "child_id": node.child_id,
            }
            for node in tree.nodes
        ],
        "api_key_present": presence["api_key_present"],
        "secret_value_persisted": False,
        "raw_content_persisted": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "accepted_program"}, indent=2))


if __name__ == "__main__":
    main()
