#!/usr/bin/env python3
"""Cross-task source-feedback study on a parameterized lag family.

The original conflict fixture hard-coded one transition law.  This receipt
keeps its real source concepts and paid-prefix protocol, but changes the
target transition coefficients in separate tasks.  It checks both sides of
the response-regime gate: a coefficient shift must block stale negative
transfer, while a noisy response must remain unresolved until replication
evidence is available.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from random import Random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "source_feedback_task_family_20261004"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


class ParametricConflictLab:
    channels = ()

    def __init__(self, spec, *, current_weight, lag_weight, noise=0., seed=0):
        self.spec = spec
        self.current_weight = current_weight
        self.lag_weight = lag_weight
        self.noise = noise
        self.rng = Random(seed)
        self.observed, self.queries = [], []

    def spec_for(self, channels=()):
        assert not channels
        return self.spec

    def observe(self, probe):
        self.queries.append(probe)
        history, rows = [], []
        for point in probe.sequence:
            target = self.current_weight * point["x"]
            if len(history) >= 2:
                target += self.lag_weight * history[-2]["x"]
            if self.noise:
                target += self.rng.gauss(0., self.noise)
            rows.append(self._row(point, target, history))
            history.append({**point, "__target__": target})
        self.observed.extend(rows)
        return rows

    def acquire(self, probe, channels=()):
        from hegel_machine.cognition.measurement_lab import MeasurementBatch
        assert not channels
        rows = self.observe(probe)
        return MeasurementBatch(rows, len(rows), 0, ())

    @staticmethod
    def _row(point, target, history):
        from hegel_machine.cognition.models import Observation
        return Observation(dict(point), target, tuple(history))


def _fixtures():
    from test_cognition_unified_source_returns import failed_successor, source_competition

    case = failed_successor.__wrapped__()
    _case, good, warm, cold = source_competition.__wrapped__(case)
    return case, good, warm.library, cold.library


def _previous(case, lab):
    from hegel_machine.cognition.evaluation import initial_prediction, loss
    from hegel_machine.cognition.language import Program

    batches, rows = [], []
    for probe in case["spec"].initial_probes:
        measured = lab.observe(probe)
        rows.extend(measured)
        batches.append({"probe": asdict(probe), "channels": [],
                        "observations": [asdict(row) for row in measured]})
    previous = deepcopy(case["previous"])
    previous["batches"] = batches
    previous["fallback"] = initial_prediction(rows, .5)
    # Keep a neutral unresolved incumbent so both source laws receive the same
    # real screening opportunity in every transition variant.
    previous["program"] = Program("const", value=previous["fallback"]).to_dict()
    previous["selection_loss"] = loss(
        Program.from_dict(previous["program"]), rows, previous["fallback"], .5)
    return previous


def _offers(case, good, library, lab, previous, seed):
    from hegel_machine.cognition.models import ResearchConfig, SearchBudget
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    from hegel_machine.cognition.unified_revisits import offer_successor_knowledge

    config = ResearchConfig(observation_budget=100, compute_budget=20_000,
        minimum_improvement=1e-7, max_rounds=1, active_probes=False,
        risk_learning=False, seed=seed, max_candidates=2)
    offers = []
    for position, source_id in enumerate((case["source_id"], good)):
        offer = offer_successor_knowledge(
            UnifiedResearchAgent(config, library=library, measurement_budget=0),
            lab, previous, {source_id}, SearchBudget(4_000))
        if offer is None:
            raise RuntimeError(f"missing source offer {source_id}")
        offer.update(task_position=0, source_task_position=position)
        offers.append(offer)
    return config, offers


def _run_variant(case, good, warm, cold, variant, *, arm, seed):
    from hegel_machine.cognition.evaluation import loss
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.memory import KnowledgeLibrary
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    from hegel_machine.cognition.unified_session import UnifiedResearchSession
    from test_cognition_conflict_transfer_refit import conflict_learning

    spec = replace(case["spec"], task_id=f"family_{variant['label']}")
    base = warm if arm == "feedback" else cold
    library = KnowledgeLibrary.from_dict(deepcopy(base.to_dict()))
    lab = ParametricConflictLab(spec, current_weight=variant["current_weight"],
        lag_weight=variant["lag_weight"], noise=variant["noise"], seed=seed)
    previous = _previous(case, lab)
    config, offers = _offers(case, good, library, lab, previous, seed)
    session = UnifiedResearchSession(config, tasks=[{"task_index": 0, "seed": seed}],
        library=library, measurement_budget=0,
        lab_factory=lambda _item: ParametricConflictLab(spec,
            current_weight=variant["current_weight"], lag_weight=variant["lag_weight"],
            noise=variant["noise"], seed=seed))
    if arm == "matched_no_return":
        payload = library.to_dict()
        for concept in payload["concepts"]:
            concept["reuse_history"] = [row for row in concept.get("reuse_history", [])
                if row.get("kind") != "research_return"]
        session.library = KnowledgeLibrary.from_dict(payload)
        library = session.library
    session.state["pending_revisits"] = deepcopy(offers)
    session.state["results"] = [{"task_position": 0, "research": deepcopy(previous)}]
    session._refresh_source_feedback()
    pending = list(session.state["pending_revisits"])
    if not pending:
        raise RuntimeError(f"no pending offer for {variant['label']}/{arm}")
    chosen = pending[0]
    report = UnifiedResearchAgent(config, library=library, measurement_budget=0).run(
        ParametricConflictLab(spec, current_weight=variant["current_weight"],
            lag_weight=variant["lag_weight"], noise=variant["noise"], seed=seed),
        previous=deepcopy(previous), continuation_candidate=deepcopy(chosen["candidate"]))
    heldout_lab = ParametricConflictLab(spec, current_weight=variant["current_weight"],
        lag_weight=variant["lag_weight"], noise=variant["noise"], seed=seed + 100_000)
    heldout_rows = heldout_lab.observe(case["spec"].available_probes[-1])
    program = Program.from_dict(report["program"])
    return {
        "variant": variant["label"], "arm": arm, "seed": seed,
        "transition": {key: variant[key] for key in ("current_weight", "lag_weight", "noise")},
        "selected_source": list(chosen["candidate"]["source_concept_ids"]),
        "pending_after": [list(item["candidate"]["source_concept_ids"]) for item in pending],
        "status": report["status"],
        "heldout_loss": loss(program, heldout_rows, report["fallback"], .5),
        "visit_costs": dict(report["costs"]),
        "feedback_events": deepcopy(session.state["scheduler_events"]),
        "external_model_calls": 0,
    }


def main():
    case, good, warm, cold = _fixtures()
    variants = (
        {"label": "base_law", "current_weight": .55, "lag_weight": 1.2, "noise": 0.},
        {"label": "shifted_current_law", "current_weight": .6, "lag_weight": 1.2, "noise": 0.},
        {"label": "noisy_base_law", "current_weight": .55, "lag_weight": 1.2, "noise": .02},
    )
    rows = [_run_variant(case, good, warm, cold, variant, arm=arm, seed=seed)
            for variant in variants for seed in (1237, 1238)
            for arm in ("feedback", "matched_no_return")]
    for variant in variants:
        for seed in (1237, 1238):
            pair = [row for row in rows if row["variant"] == variant["label"] and row["seed"] == seed]
            assert len(pair) == 2 and pair[0]["visit_costs"] == pair[1]["visit_costs"]
    for seed in (1237, 1238):
        base = [row for row in rows if row["variant"] == "base_law" and row["seed"] == seed]
        shifted = [row for row in rows if row["variant"] == "shifted_current_law" and row["seed"] == seed]
        assert base[0]["selected_source"] == [good] and base[1]["selected_source"] == [case["source_id"]]
        # The measured response signature blocks the stale negative return
        # after the current-input coefficient changes; both arms then choose
        # the source that actually fits this shifted law.
        assert shifted[0]["selected_source"] == [case["source_id"]]
        assert shifted[1]["selected_source"] == [case["source_id"]]
        assert shifted[0]["heldout_loss"] < 1e-10 and shifted[1]["heldout_loss"] < 1e-10
    payload = {"status": "complete", "protocol": {
        "variants": list(variants), "seeds": [1237, 1238],
        "same_screened_offers": True, "same_execution_limits": True,
        "independent_transition_parameters": True, "external_model_calls": 0},
        "results": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": payload["status"], "rows": len(rows),
                      "output": str((OUT / "results.json").relative_to(ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
