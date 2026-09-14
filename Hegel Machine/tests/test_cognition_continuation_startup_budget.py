"""Growing memory must leave a purchased continuation able to fit its data."""

from dataclasses import replace

import pytest

from hegel_machine.cognition import controller
from hegel_machine.cognition.continuation_plan import continuation_plan
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig, TaskSpec
from hegel_machine.cognition.workspace import ResearchWorkspace


@pytest.mark.parametrize("use_abstractions", [False, True])
def test_crowded_memory_cannot_spend_the_purchased_fit(monkeypatch, use_abstractions):
    class PolynomialTask:
        spec = TaskSpec("continued_polynomial", "Predict the measured response", ("x",), (), (),
                        feature_bounds={"x": (-2., 2.)}, max_probe_length=12)

        def observe(self, probe):
            rows, history = [], []
            for point in probe.sequence:
                target = .4 + .3 * point["x"] + .8 * point["x"] ** 2
                rows.append(Observation(point, target, tuple(history)))
                history.append({**point, "__target__": target})
            return rows

    environment, library = PolynomialTask(), KnowledgeLibrary()
    x = Program("feature", name="x")
    source_spec = replace(environment.spec, task_id="prior_linear_responses")
    for index in range(64):
        program = Program("mul", children=(Program("const", value=index / 10 + 1), x))
        support = [Observation({"x": value}, (index / 10 + 1) * value) for value in (-2., 0., 2.)]
        library.learn(Candidate(f"source_{index}", program, "refit"), source_spec, support)
    probes = [Probe(f"acquired_{index}", tuple({"x": -2 + 4 * j / 19} for j in range(20)))
              for index in range(3)]
    rows = [row for probe in probes for row in environment.observe(probe)]
    incumbent = Candidate("previous", x, "refit")
    workspace = ResearchWorkspace(environment.spec.task_id, observations=rows, reference=rows[-20:],
        queried=probes, incumbent=incumbent, bank=[incumbent], fallback=0., initial_loss=5.)
    config = ResearchConfig(seed=503, compute_budget=10000, use_abstractions=use_abstractions)
    plan = continuation_plan(config, environment.spec, workspace)
    workspace.research_intent = {"kind": "revision", "execution_plan": plan}
    allocated = replace(config, compute_budget=plan["compute_budget"],
        observation_budget=plan["observation_budget"], max_rounds=plan["max_rounds"])
    fit_budgets, sparse = [], controller.sparse_candidates

    def capture_fit(*args, **kwargs):
        fit_budgets.append(kwargs["budget"].remaining)
        return sparse(*args, **kwargs)

    monkeypatch.setattr(controller, "sparse_candidates", capture_fit)
    result = controller.CognitiveAgent(allocated, library).run(environment, research_state=workspace.to_dict())
    retrieval = next(action for action in result.actions if action.action == "retrieve")
    assert 0 < len(retrieval.details["tested"]) < 64
    assert retrieval.compute_cost <= retrieval.details["allocated_compute"]
    assert fit_budgets and fit_budgets[0] >= plan["requested_fit_compute"]
    assert result.validation_loss < 1e-10
    assert result.compute_cost <= allocated.compute_budget
    assert result.observation_cost <= allocated.observation_budget
