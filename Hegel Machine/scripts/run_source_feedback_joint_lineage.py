#!/usr/bin/env python3
"""Measure a shared multi-source successor without double-counting work.

Two distinct source ids carry the same learned response law in this small
fixture.  The proposal path must merge them into one executable lineage.  The
four masks then estimate the value of the joint lineage, each singleton, and
the empty baseline under the same paid prefix.
"""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "source_feedback_joint_lineage_20261004"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import Concept, KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig, SearchBudget
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_revisits import offer_successor_knowledge
from run_source_feedback_attribution import _earn_positive_return
from run_source_feedback_task_family import ParametricConflictLab, _fixtures, _previous


SEED = 20261009
GOOD = "concept_0002"
SECOND_GOOD = "concept_0003"
MASKS = ("keep_ab", "drop_a", "drop_b", "drop_ab")


class SessionLab(ParametricConflictLab):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def _config():
    return ResearchConfig(observation_budget=100, compute_budget=20_000,
        minimum_improvement=1e-7, max_rounds=1, max_candidates=2,
        active_probes=False, risk_learning=False, seed=SEED,
        knowledge_tie_tolerance=1e-8)


def _add_duplicate_source(library, source_id):
    source = next(item for item in library.concepts if item.concept_id == GOOD)
    library.concepts.append(Concept(source_id, source.program, source.risk_quantile,
        "duplicate_paid_repair", "representation"))
    for template in list(library.abstracts):
        if template.source_concept_id != GOOD:
            continue
        clone = deepcopy(template)
        clone.source_concept_id = source_id
        clone.template_id = clone.template_id.replace(GOOD, source_id, 1)
        clone.parents = [source_id]
        clone.transfer_history = []
        library.abstracts.append(clone)
    library._refresh_definitions()


def _delete_source_returns(library, source_ids):
    source_ids = set(source_ids)
    for concept in library.concepts:
        concept.reuse_history = [item for item in concept.reuse_history
            if not (item.get("kind") == "research_return"
                    and item.get("source_id") in source_ids)]


def _offer(library, case, spec, previous, source_ids):
    lab = ParametricConflictLab(spec, current_weight=.55, lag_weight=1.2,
        noise=0., seed=SEED)
    offer = offer_successor_knowledge(
        UnifiedResearchAgent(_config(), library=library, measurement_budget=0),
        lab, previous, set(source_ids), SearchBudget(4_000))
    if offer is None:
        raise RuntimeError(f"missing offer for {source_ids}")
    offer.update(task_position=0, source_task_position=0)
    return offer


def _value(library, case, spec, previous, offer):
    heldout_lab = ParametricConflictLab(spec, current_weight=.55, lag_weight=1.2,
        noise=0., seed=SEED + 100_000)
    heldout_rows = heldout_lab.observe(spec.available_probes[-1])
    baseline_loss = loss(Program.from_dict(previous["program"]), heldout_rows,
        previous["fallback"], .5)
    if offer is None:
        return {"selected_source_ids": [], "heldout_loss": baseline_loss,
            "value": 0., "visit_costs": {"observation_cost": 0,
                "measurement_cost": 0, "compute_cost": 0},
            "joint_returns": [], "local_returns": []}
    lab = SessionLab(spec, current_weight=.55, lag_weight=1.2, noise=0., seed=SEED)
    report = UnifiedResearchAgent(_config(), library=library,
        measurement_budget=0).run(lab, previous=deepcopy(previous),
        continuation_candidate=deepcopy(offer["candidate"]))
    program = Program.from_dict(report["program"])
    heldout_loss = loss(program, heldout_rows, report["fallback"], .5)
    returns = report.get("knowledge_return", {})
    return {"selected_source_ids": list(offer["candidate"]["source_concept_ids"]),
        "heldout_loss": heldout_loss, "value": baseline_loss - heldout_loss,
        "visit_costs": dict(report["visit_costs"]),
        "joint_returns": deepcopy(returns.get("joint_returns", [])),
        "local_returns": [{"source_id": item["source_id"],
            "comparison_compute": item["comparison_compute"],
            "adaptation_compute": item["adaptation_compute"],
            "lineage_id": item.get("lineage_id"),
            "joint_source_ids": item.get("joint_source_ids", [])}
            for item in returns.get("returns", [])],
        "status": report["status"], "external_model_calls": 0}


def main():
    case, good, warm, _cold = _fixtures()
    if good != GOOD:
        raise RuntimeError(f"fixture source changed: expected {GOOD}, got {good}")
    library = KnowledgeLibrary.from_dict(deepcopy(warm.to_dict()))
    _add_duplicate_source(library, SECOND_GOOD)
    positive_a = _earn_positive_return(case, GOOD, library)
    positive_b = _earn_positive_return(case, SECOND_GOOD, library)
    spec = replace(case["spec"], task_id="joint_lineage_target")
    previous = _previous(case, ParametricConflictLab(spec, current_weight=.55,
        lag_weight=1.2, noise=0., seed=SEED))
    # The negative source is not part of this value function. Its history is
    # removed before all masks so a source withdrawal cannot masquerade as a
    # joint contribution effect.
    bad = case["source_id"]
    _delete_source_returns(library, {bad})
    offers = {
        "joint": _offer(library, case, spec, previous, {GOOD, SECOND_GOOD}),
        "a": _offer(library, case, spec, previous, {GOOD}),
        "b": _offer(library, case, spec, previous, {SECOND_GOOD}),
    }
    if offers["joint"]["candidate"]["source_concept_ids"] != [GOOD, SECOND_GOOD]:
        raise RuntimeError("source union did not produce a joint lineage candidate")
    if len(offers["joint"]["candidate"]["transfer_evidence"]) != 2:
        raise RuntimeError("joint candidate lost one of its source origins")

    values = {}
    for mask in MASKS:
        arm_library = KnowledgeLibrary.from_dict(deepcopy(library.to_dict()))
        deleted = ({GOOD, SECOND_GOOD} if mask == "drop_ab"
                   else {GOOD} if mask == "drop_a"
                   else {SECOND_GOOD} if mask == "drop_b" else set())
        _delete_source_returns(arm_library, deleted)
        chosen = (offers["joint"] if mask == "keep_ab"
                  else offers["b"] if mask == "drop_a"
                  else offers["a"] if mask == "drop_b"
                  else None)
        values[mask] = _value(arm_library, case, spec, previous, chosen)
        values[mask]["deleted_source_ids"] = sorted(deleted)

    baseline = values["drop_ab"]["heldout_loss"]
    v_ab = values["keep_ab"]["value"]
    v_a = values["drop_b"]["value"]
    v_b = values["drop_a"]["value"]
    v_empty = values["drop_ab"]["value"]
    contrast = {"delta_a": v_ab - v_b, "delta_b": v_ab - v_a,
        "interaction": v_ab - v_a - v_b + v_empty,
        "baseline_heldout_loss": baseline}
    joint = values["keep_ab"]["joint_returns"]
    if len(joint) != 1 or not joint[0]["shared_cost_accounted_once"]:
        raise RuntimeError("joint return was not recorded once with shared cost")
    local_costs = values["keep_ab"]["local_returns"]
    assert len(local_costs) == 2
    assert joint[0]["comparison_compute"] == local_costs[0]["comparison_compute"]
    assert all(values[mask]["external_model_calls"] == 0 for mask in MASKS
               if "external_model_calls" in values[mask])
    payload = {"status": "complete", "lineage": {
        "package_version": "0.181.0",
        "parent": "natural near-tie source attribution",
        "stage": "joint source lineage and shared-cost attribution"},
        "protocol": {"seed": SEED, "masks": list(MASKS),
            "same_paid_prefix": True, "same_execution_limits": True,
            "negative_source_removed_before_masks": True,
            "source_ids": {"a": GOOD, "b": SECOND_GOOD, "bad": bad},
            "external_model_calls": 0},
        "positive_return_acquisition": {"a": positive_a, "b": positive_b},
        "offers": {key: {"source_concept_ids": value["candidate"]["source_concept_ids"],
            "expected_loss_reduction": value["expected_loss_reduction"],
            "lineage_ids": [item.get("lineage_id") for item in value["candidate"]["transfer_evidence"]]}
            for key, value in offers.items()},
        "values": values, "contrast": contrast,
        "interpretation": "The duplicated law makes A/B redundant; near-zero interaction is an identifiability result, not synergy.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": payload["status"], "masks": list(MASKS),
        "output": str((OUT / "results.json").relative_to(ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
