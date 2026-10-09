"""A measured failed binding is not a refutation of its reusable procedure."""

from copy import deepcopy
from dataclasses import replace
import gzip
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import TaskSpec


@pytest.fixture(scope="module")
def paid_failure():
    return json.loads(gzip.decompress((Path(__file__).parent /
        "fixtures/draft_binding_counterexample_root10.json.gz").read_bytes()))


def prepared(case):
    spec = TaskSpec(**case["spec"])
    library = KnowledgeLibrary(research_drafts=[deepcopy(case["draft"])])
    program = Program.from_dict(case["entry"]["program"])
    return library, spec, program


def record(case, library, spec, program, *, evidence=None):
    return library.record_transfer_counterexample(spec, program, (),
        source_draft_ids=[case["entry"]["research_draft_id"]],
        evidence=deepcopy(case["evidence"] if evidence is None else evidence),
        fallback=case["entry"]["fallback"], probe_id=case["probe_id"])


def test_real_failed_binding_survives_json_without_refuting_source_or_revision(paid_failure):
    library, spec, program = prepared(paid_failure)
    original = deepcopy(library.research_drafts[0])
    assert not library.transfer_refuted(spec, program)
    event = record(paid_failure, library, spec, program)
    assert event["source_draft_ids"] == ["research_draft_0011"]
    assert not event["source_ids"]
    library = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    assert library.transfer_refuted(spec, program, fallback=paid_failure["entry"]["fallback"])
    assert not library.transfer_refuted(replace(spec, task_id="another_paid_target"), program)
    assert not library.transfer_refuted(replace(spec, context={**spec.context, "view": "new"}), program)
    revised = Program("add", children=(program, Program("const", value=.1)))
    assert not library.transfer_refuted(spec, revised, fallback=paid_failure["entry"]["fallback"])
    retained = deepcopy(library.research_drafts[0])
    retained.pop("outcomes")
    original.pop("outcomes", None)
    assert retained == original
    assert record(paid_failure, library, spec, program) is None


@pytest.mark.parametrize("evidence", [
    {"status": "unresolved"},
    {"status": "counterexample", "noise_replication": {"repeatable_error": False, "noise_supported": True}},
    {"status": "counterexample", "noise_replication": {"repeatable_error": False, "noise_supported": False}},
    {"status": "counterexample", "noise_replication": {"repeatable_error": True, "noise_supported": True}},
])
def test_unconfirmed_or_noise_supported_failure_is_not_withdrawn(paid_failure, evidence):
    library, spec, program = prepared(paid_failure)
    before = library.to_dict()
    assert record(paid_failure, library, spec, program, evidence=evidence) is None
    assert library.to_dict() == before


def test_history_fallback_is_part_of_the_failed_executable_binding(paid_failure):
    library, spec, _ = prepared(paid_failure)
    program = Program("lag", name=spec.feature_names[0], lag=1)
    assert program.minimum_history_depth
    record(paid_failure, library, spec, program)
    fallback = paid_failure["entry"]["fallback"]
    assert library.transfer_refuted(spec, program, fallback=fallback)
    assert not library.transfer_refuted(spec, program, fallback=fallback + 1.)
    assert not library.transfer_refuted(spec, program)


def test_draft_consumer_ablation_preserves_record_and_concept_behavior(paid_failure, monkeypatch):
    library, spec, program = prepared(paid_failure)
    record(paid_failure, library, spec, program)
    before = library.to_dict()
    monkeypatch.setattr(library, "draft_transfer_refuted", lambda *args, **kwargs: False)
    assert not library.transfer_refuted(spec, program, fallback=paid_failure["entry"]["fallback"])
    assert library.to_dict() == before


def test_paid_replay_removes_failed_binding_without_buying_new_search(paid_failure, monkeypatch, record_property):
    from hegel_machine.cognition.models import Observation, ResearchConfig, SearchBudget
    from hegel_machine.cognition.research_drafts import replay_numeric_research_drafts
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent

    batches = []
    for points in paid_failure["replay_training_batches"]:
        history, rows = [], []
        for point in points:
            rows.append(Observation(point["features"], point["target"], tuple(history)))
            history.append({**point["features"], "__target__": point["target"]})
        batches.append(rows)
    library, spec, failed = prepared(paid_failure)
    record(paid_failure, library, spec, failed)
    before = library.to_dict()
    agent = UnifiedResearchAgent(ResearchConfig(**paid_failure["config"]), library=library)
    rows = [row for batch in batches for row in batch]
    outputs = []
    for enabled in (False, True):
        budget, events = SearchBudget(80000), []
        with monkeypatch.context() as patch:
            if not enabled:
                patch.setattr(library, "draft_transfer_refuted", lambda *args, **kwargs: False)
            claims = replay_numeric_research_drafts(agent, spec, rows, budget, events,
                validation=batches[paid_failure["replay_validation_batch"]])
        outputs.append((claims, events, budget.spent))
    assert failed in [item["candidate"].program for item in outputs[0][0]]
    assert failed not in [item["candidate"].program for item in outputs[1][0]]
    assert any(e["action"] == "exclude_refuted_draft_candidate" for e in outputs[1][1])
    assert outputs[1][2] <= outputs[0][2]
    assert all(item["research_draft_id"] == "research_draft_0011" for item in outputs[1][0])
    assert library.to_dict() == before
    record_property("historical_prefix_compute_without_consumer", outputs[0][2])
    record_property("historical_prefix_compute_with_consumer", outputs[1][2])


def test_refuted_adequate_candidate_does_not_stop_later_draft_without_validation(paid_failure, monkeypatch):
    from hegel_machine.cognition import conditional_coordinates, decoded_conditional_coordinates
    from hegel_machine.cognition.models import Observation, ResearchConfig, SearchBudget
    from hegel_machine.cognition.research_drafts import _Fitter, replay_numeric_research_drafts
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent

    library, spec, failed = prepared(paid_failure)
    record(paid_failure, library, spec, failed)
    later = deepcopy(paid_failure["draft"])
    later["draft_id"] = "later_research_draft"
    output = later["research_program"]["graph"]["nodes"][later["research_program"]["output_node"]]
    feature = Program("feature", name=spec.feature_names[0])
    cube = Program("mul", children=(feature, Program("mul", children=(feature, feature))))
    output["answer"] = Program("add", children=(Program.from_dict(output["answer"]), cube)).to_dict()
    library.research_drafts.append(later)
    revised = Program("add", children=(failed, Program("const", value=.1)))
    fits = []

    def supplied_fit(self, terms):
        program = failed if not fits else revised
        fits.append(program)
        return {"program": program, "loss": 0., "fit_terms": (program,), "coefficients": (0., 1.)}

    def no_new_broad_search(*args, **kwargs):
        pytest.fail("Withdrawal must not purchase a new broad search")

    monkeypatch.setattr(_Fitter, "fit", supplied_fit)
    monkeypatch.setattr(conditional_coordinates, "propose_experimental_conditional_coordinates",
                        lambda *args, **kwargs: [])
    monkeypatch.setattr(conditional_coordinates, "propose_conditional_coordinates", no_new_broad_search)
    monkeypatch.setattr(decoded_conditional_coordinates, "propose_decoded_conditional_coordinates", no_new_broad_search)
    history, rows = [], []
    for point in paid_failure["replay_training_batches"][0]:
        rows.append(Observation(point["features"], point["target"], tuple(history)))
        history.append({**point["features"], "__target__": point["target"]})
    agent = UnifiedResearchAgent(ResearchConfig(**paid_failure["config"]), library=library)
    events = []
    claims = replay_numeric_research_drafts(agent, spec, rows, SearchBudget(80000), events,
        evidence=[{"control": "reading", "gain": 1.}])
    assert fits == [failed, revised]
    assert [item["research_draft_id"] for item in claims] == ["later_research_draft"]
    assert claims[0]["candidate"].program == revised
    replayed = [item for item in events if item["action"] == "replay_research_draft"]
    assert replayed[0]["status"] == "no_candidate"
    assert replayed[1]["draft_id"] == "later_research_draft"
    assert all(item["validation_loss"] is None for item in replayed)
