"""The joint source receipt must show complementarity and scoped withdrawal."""

import json
from pathlib import Path
import subprocess
import sys

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.knowledge_returns import return_priority
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.research_programs import _sources, propose_research_programs
from hegel_machine.cognition.memory import Concept, KnowledgeLibrary
from hegel_machine.cognition.unified_session import UnifiedResearchSession
from hegel_machine.cognition.transfer_condition_research import record_condition_coverages

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_shared_return_does_not_reprice_single_source_program():
    condition = {"objective": "predict", "data": {"boundary": "same"}}
    single = Program("feature", name="x")
    joint = Program("add", children=(single, Program("const", value=1.0)))
    shared_negative = {"kind": "research_return", "condition": condition,
        "joint_source_ids": ["a", "b"], "retained_normalized_gain": -.4,
        "adaptation_compute": 10, "comparison_compute": 10,
        "comparisons": [{"program": joint.to_dict(), "comparison_id": "joint"}]}
    history = [shared_negative]
    assert return_priority(history, condition, program=single) == (0.0, 0.0)
    assert return_priority(history, condition, program=joint)[0] > 0.0
    assert return_priority(history, condition) == (0.0, 0.0)


def test_production_composer_discovers_the_two_source_bundle():
    from run_source_feedback_complementary_bundle import _library, _rows, _spec

    library, _, _ = _library(_rows(20261212, 48, .6, 1.2))
    spec = _spec("production_composition")
    claims = propose_research_programs(spec, _rows(20261111, 48, .6, 1.2),
        SearchBudget(20_000), library.callable_atoms(spec, limit=8), max_candidates=8)
    assert any(claim["training_loss"] < 1e-12
               and set(_sources(claim["candidate"].program)) == {"concept_0001", "concept_0002"}
               for claim in claims)


def test_joint_return_does_not_withdraw_singleton_offer_scope():
    condition = {"objective": "predict", "data": {"boundary": "same"}}
    a = Program("feature", name="x")
    b = Program("feature", name="z")
    joint = Program("add", children=(a, b))
    shared = {"kind": "research_return", "condition": condition,
        "joint_source_ids": ["a", "b"], "retained_normalized_gain": -.4,
        "adaptation_compute": 10, "comparison_compute": 10,
        "confirmation_status": "rejected",
        "comparisons": [{"program": joint.to_dict(), "comparison_id": "joint"}]}
    library = KnowledgeLibrary(concepts=[
        Concept("a", a, .5, "source_a", "current"),
        Concept("b", b, .5, "source_b", "lag"),
    ])
    library.concepts[0].reuse_history.append(shared.copy())
    library.concepts[1].reuse_history.append(shared.copy())
    session = UnifiedResearchSession(ResearchConfig(observation_budget=8,
        compute_budget=100, active_probes=False), tasks=[{"task_index": 0, "seed": 1}],
        library=library, measurement_budget=8)
    single = {"candidate": {"program": a.to_dict(), "transfer_evidence": [
        {"source_id": "a", "condition": condition}]}}
    combined = {"candidate": {"program": joint.to_dict(), "transfer_evidence": [
        {"source_id": "a", "condition": condition},
        {"source_id": "b", "condition": condition}]}}
    assert session._source_feedback_state(single)[2] == ["unknown"]
    assert session._source_feedback_state(combined)[2] == ["negative", "negative"]


def test_singleton_negative_return_withdraws_only_its_actual_program():
    condition = {"objective": "predict", "data": {"boundary": "same"}}
    tested = Program("feature", name="x")
    other = Program("const", value=2.)
    record = {"kind": "research_return", "condition": condition,
        "source_id": "source", "joint_source_ids": ["source"],
        "retained_normalized_gain": -.4, "confirmation_status": "rejected",
        "adaptation_compute": 10, "comparison_compute": 10,
        "comparisons": [{"program": tested.to_dict(), "comparison_id": "paid"}]}
    source = Concept("source", tested, .5, "source_task", "refit")
    source.reuse_history.append(record)
    session = UnifiedResearchSession(ResearchConfig(),
        tasks=[{"task_index": 0, "seed": 1}], library=KnowledgeLibrary(concepts=[source]),
        measurement_budget=8)
    session.state["pending_revisits"] = [{"task_position": 0, "source_task_position": 0,
        "candidate": {"program": program.to_dict(), "transfer_evidence": [
            {"source_id": "source", "condition": condition}]}}
        for program in (tested, other)]

    session._refresh_source_feedback()

    remaining, = session.state["pending_revisits"]
    assert Program.from_dict(remaining["candidate"]["program"]) == other
    assert remaining["source_return_status"] == ["unknown"]
    assert return_priority([record], condition)[0] > 0.  # Source-level ranking still aggregates.
    equivalent = Program("add", children=(tested, Program("const", value=0.)))
    assert return_priority([record], condition, program=equivalent,
                           source_ids=["source"])[0] > 0.
    assert return_priority([record], condition, program=other,
                           source_ids=["source"]) == (0., 0.)


def test_same_visit_alternatives_do_not_confirm_each_other():
    from run_source_feedback_complementary_bundle import _library, _rows, _spec
    from hegel_machine.cognition.knowledge_returns import transfer_condition

    library, source_program, _ = _library(_rows(20261213, 24, .6, 1.2))
    source_id = "concept_0001"
    spec = _spec("same_visit_confirmation")
    rows = _rows(20261214, 12, .6, 1.2)
    condition = transfer_condition(spec, rows)
    condition["response_signature"]["residual_bin"] = 1
    candidate = Candidate("same-law", source_program, "reuse",
                          parents=(source_id,), memory_id=source_id)
    def visit(target):
        target.begin_transfer_episode(spec, rows)
        for template_id in ("template_a", "template_b"):
            target.record_transfer_result(candidate, spec, accepted=True, loss=.1,
                baseline_loss=1., evidence_rows=len(rows), comparison_compute=4,
                comparison_id=template_id, origin={"source_id": source_id,
                    "template_id": template_id, "mapping": {}, "mapping_roles": {},
                    "condition": condition, "adaptation_work": []})
        return target.finish_transfer_episode(spec, final_candidate=candidate,
            final_loss=.1, baseline_loss=1., compute_cost=20, observation_cost=12,
            validation_rows=12, final_status="completed")

    result = visit(library)
    assert len(result["returns"]) == 2
    assert all(item["confirmation_status"] == "provisional"
               and item["confirmation_replicates"] == 1 for item in result["returns"])

    # A second independent visit supplies one additional confirmation per
    # alternative.  Restoring the library in between also catches reuse of
    # an episode identifier, which would silently merge visits.
    restored = KnowledgeLibrary.from_dict(library.to_dict())
    result2 = visit(restored)
    assert len(result2["returns"]) == 2
    assert all(item["confirmation_status"] == "confirmed"
               and item["confirmation_replicates"] == 2 for item in result2["returns"])
    assert restored._transfer_serial == 2


def test_condition_coverage_receipt_is_deduplicated_and_serialized():
    x = Program("feature", name="x")
    predicate = Program("lt", children=(x, Program("const", value=0.)))
    call = Program("if", children=(predicate, x, Program("const", value=0.)))
    coverages = []
    retained = [(0, call, call.expanded())]
    scope = {"calls": [{"call_index": 0, "status": "covered_by_paid_evidence",
        "history_depth": 0, "conditions": [{"covered": True,
            "predicate": predicate.to_dict(), "paid_witnesses": [0]}]}]}
    record_condition_coverages(scope, retained, coverages, task_id="coverage-test")
    record_condition_coverages(scope, retained, coverages, task_id="coverage-test")
    assert len(coverages) == 1
    restored = KnowledgeLibrary(condition_coverages=coverages)
    round_trip = KnowledgeLibrary.from_dict(restored.to_dict())
    assert round_trip.condition_coverages == coverages


def test_condition_coverage_uses_original_index_after_unconditional_calls_are_filtered():
    conditional = Program("if", children=(
        Program("lt", children=(Program("feature", name="x"), Program("const", value=0.))),
        Program("feature", name="x"), Program("const", value=0.)))
    retained = [(3, conditional, conditional.expanded())]
    scope = {"calls": [{"call_index": 3, "status": "covered_by_paid_evidence",
        "history_depth": 0, "conditions": [{"covered": True,
            "predicate": conditional.children[0].to_dict(), "paid_witnesses": [0]}]}]}
    coverages = []
    record_condition_coverages(scope, retained, coverages)
    assert len(coverages) == 1
    assert coverages[0]["program"] == conditional.expanded().simplified().to_dict()


def test_without_removes_condition_coverage_for_withdrawn_sources():
    x = Program("feature", name="x")
    library = KnowledgeLibrary(concepts=[
        Concept("source_a", x, .5, "a", "refit"),
        Concept("source_b", x, .5, "b", "refit"),
    ], condition_coverages=[
        {"source_ids": ["source_a"], "program": x.to_dict(), "predicate": {}, "history_depth": 0},
        {"source_ids": ["source_a", "source_b"], "program": x.to_dict(), "predicate": {}, "history_depth": 0},
        {"source_ids": ["source_b"], "program": x.to_dict(), "predicate": {}, "history_depth": 0},
    ])
    restored = library.without({"source_a"})
    assert [record["source_ids"] for record in restored.condition_coverages] == [["source_b"]]


def test_complementary_bundle_and_scoped_counterexample():
    # This catches accidental metadata-only attribution, loss of one source
    # call, and broad source revocation after a changed mechanism.
    result = subprocess.run(
        [sys.executable, "scripts/run_source_feedback_complementary_bundle.py"],
        cwd=ROOT, env={"PYTHONPATH": "src"}, capture_output=True, text=True,
        check=True)
    assert '"status": "complete"' in result.stdout
    payload = json.loads((ROOT / "artifacts/source_feedback_complementary_bundle_20261004/results.json").read_text())
    assert payload["training_losses"]["ab"] < 1e-20
    assert payload["values"]["keep_ab"]["value"] > payload["values"]["drop_a"]["value"] + .1
    assert payload["values"]["keep_ab"]["value"] > payload["values"]["drop_b"]["value"] + .1
    joint = payload["joint_returns"]
    assert len(joint) == 1
    assert joint[0]["shared_cost_accounted_once"]
    shifted = payload["mechanism_shift"]
    assert shifted["joint_refuted"]
    assert not shifted["a_refuted"]
    assert not shifted["b_refuted"]
