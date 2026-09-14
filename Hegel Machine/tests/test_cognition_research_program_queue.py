"""A newer procedure's extra binding must not hide an older useful procedure."""

from random import Random

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.research_programs import propose_research_programs
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


def observations(seed, mode, *, coefficients=(1.2, -.4)):
    rng = Random(seed)
    rows = []
    for length in (13, 23):
        history, recent, state, previous = [], [], 0., None
        for _ in range(length):
            features = {name: rng.uniform(-1, 1) for name in ("x", "y")}
            event = 0. if previous is None else features["y"] - previous
            recent.append(features["x"])
            state = .8 * state + features["x"]
            target = {"combined": coefficients[0] * features["x"] + coefficients[1] * event,
                      "window": sum(recent[-3:]), "state": state}[mode]
            rows.append(Observation(features, target, tuple(history)))
            history.append(dict(features))
            previous = features["y"]
    return rows


def test_main_controller_compares_distinct_methods_before_second_role_bindings():
    spec = TaskSpec("measured_process", "Predict the measured response", ("x", "y"), (), ())
    calls = []
    for name in spec.feature_names:
        definition = Definition(f"source_{name}", ("input",), Program("arg", name="input"), f"atom_{name}")
        calls.append(Program("call", name=definition.definition_id, definition=definition,
                             children=(Program("feature", name=name),)))
    library = KnowledgeLibrary()
    source_ids = []
    for mode in ("combined", "window", "state"):
        source_rows = observations(64, mode)
        proposals = propose_research_programs(spec, source_rows, SearchBudget(40000), calls)
        best = min(proposals, key=lambda item: item["training_loss"])
        concept_id, added = library.learn(best["candidate"], spec, source_rows)
        assert added
        source_ids.append(concept_id)
        library.meta_priors.append({"source_concept_id": concept_id,
                                   "research_program": best["research_program"]})
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7), library=library)
    training = observations(76, "combined", coefficients=(-.6, 1.8))
    budget, actions = SearchBudget(30000), []
    claims = agent._program_claims(spec, training, budget, agent._recall(spec), actions)
    replayed = [claim for claim in claims if claim["research_program"].get("reused_procedure")]
    assert {claim["method_source_ids"][0] for claim in replayed} == set(source_ids)
    future = observations(970, "combined", coefficients=(-.6, 1.8))
    assessed = agent._assess(claims, future, budget, 0., .5)
    selected = agent._best(assessed)
    assert selected["method_source_ids"] == [source_ids[0]]
    assert selected["research_program"]["reused_procedure"]
    assert selected["validation_loss"] < 1e-10
    assert len(replayed) <= 4
    assert budget.spent < 30000
