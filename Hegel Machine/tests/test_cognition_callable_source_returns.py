"""A fitted call earns only its actual binding and paid preparation cost."""

from copy import deepcopy

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.research_graph import ResearchGraph
from hegel_machine.cognition import unified_research
from test_cognition_unified_source_returns import failed_successor
from test_workflow_knowledge_returns import _setup


def research_call(power, monkeypatch):
    library, _, _, _ = _setup(quadratic=True)
    agent = unified_research.UnifiedResearchAgent(ResearchConfig(
        max_question_depth=0, state_search=False, risk_learning=False,
        compute_budget=12000), library=library)
    spec = TaskSpec("consumer", "Predict the observed response.", ("u",), (), (),
                    feature_bounds={"u": (-1., 1.)})
    training = [Observation({"u": value}, value ** power)
                for value in (-1., -.75, -.5, -.25, 0., .25, .5, .75, 1.)]
    # A changed validation response keeps the linear case in research, where
    # the fit actually considers and discards the inherited quadratic call.
    validation = [Observation({"u": value}, value ** power + (.5 if power == 1 else 0.))
                  for value in (-.9, -.4, .2, .7)]
    fits = []
    original_fit = unified_research._fit

    def counted_fit(terms, rows, budget, kind, **kwargs):
        before = budget.spent
        result = original_fit(terms, rows, budget, kind, **kwargs)
        if kind == "callable_transfer" and result is not None:
            fits.append({"candidate": result["candidate"], "compute_cost": budget.spent - before})
        return result

    monkeypatch.setattr(unified_research, "_fit", counted_fit)
    claims = agent._research_view(spec, training, validation, SearchBudget(12000),
                                  [], [], ResearchGraph())
    claim, = [item for item in claims if item["kind"] == "callable_transfer"]
    fit, = fits
    assert fit["candidate"] == claim["candidate"]
    return library, claim, fit, validation


def test_two_uses_of_one_recalled_call_keep_one_origin_and_one_actual_fit_bill(monkeypatch):
    library, claim, fit, validation = research_call(4, monkeypatch)
    program = claim["candidate"].program
    assert program.op == "mul" and program.children[0] == program.children[1]
    call = program.children[0]
    assert call.op == "call"
    origin, = claim["transfer_evidence"]
    assert origin["source_id"] == call.definition.source_concept_id
    assert origin["template_id"] == library.definition_bindings[call.name]["template_id"]
    assert origin["mapping"] == {"role_0": "u"}
    assert origin["mapping_roles"] == {"role_0": 0}
    work = origin["adaptation_work"]
    assert len({item["work_id"] for item in work}) == len(work)
    assert sum(item["compute_cost"] for item in work) == fit["compute_cost"] + len(validation)
    assert claim["validation_loss"] < 1e-12


def test_fitting_away_the_call_keeps_the_independent_answer_without_source_credit(monkeypatch):
    _, claim, fit, _ = research_call(1, monkeypatch)
    assert fit["compute_cost"] > 0
    assert claim["candidate"].program == Program("feature", name="u")
    assert not claim.get("transfer_evidence")


def test_memory_off_continuation_does_not_credit_a_prepared_source_offer(failed_successor):
    case = failed_successor
    library = KnowledgeLibrary.from_dict(case["library"].to_dict())
    before = deepcopy([source.reuse_history for source in library.concepts])
    config = ResearchConfig(observation_budget=len(case["old"]) + 24,
        compute_budget=20000, minimum_improvement=1e-7, max_rounds=1,
        active_probes=False, risk_learning=False, use_memory=False, seed=1237)
    lab = type(case["lab"])(case["spec"])
    assert case["offer"]["candidate"]["transfer_evidence"]
    report = unified_research.UnifiedResearchAgent(config, library=library,
        measurement_budget=0).run(lab, previous=case["previous"],
                                 continuation_candidate=case["offer"]["candidate"])
    assert lab.observed
    assert not report.get("knowledge_return")
    assert not any(event["action"] == "compare_successor_knowledge_on_fresh_response"
                   for event in report["actions"])
    assert [source.reuse_history for source in library.concepts] == before

