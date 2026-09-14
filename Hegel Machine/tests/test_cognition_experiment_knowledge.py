from hegel_machine.cognition.experiment_knowledge import ExperimentKnowledge
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, Probe, SearchBudget, TaskSpec
from hegel_machine.cognition.memory import KnowledgeLibrary


def test_actual_split_is_recalled_with_new_roles_and_control_scale():
    x = Program("feature", name="x")
    square = Program("mul", children=(x, x))
    candidates = [Candidate("linear", x, "refit"), Candidate("square", square, "composition")]
    spec = TaskSpec("source", "response", ("x",), (), (), feature_bounds={"x": (0., 2.)})
    old = [Observation({"x": value}, value) for value in (0., 1., 0., 1.)]
    probe = Probe("separate", ({"x": 2.}, {"x": 1.5}))
    new = [Observation(point, point["x"]**2) for point in probe.sequence]
    book = ExperimentKnowledge()
    budget = SearchBudget(200)
    learned = book.record(spec, candidates, old, probe, new, fallback=0., budget=budget)
    assert learned and learned["winner"] == 1 and learned["before_distance"] == 0
    assert budget.spent == 2 * (len(old) + len(new))

    target = TaskSpec("target", "response", ("u",), (), (), feature_bounds={"u": (0., 4.)})
    u = Program("feature", name="u")
    models = [Candidate("u", u, "refit"),
              Candidate("u2", Program("mul", children=(u, u)), "composition")]
    restored = ExperimentKnowledge(book.to_dict())
    offers = restored.propose(target, models, [Observation({"u": r.features["x"]}, r.target) for r in old],
                              fallback=0., budget=SearchBudget(200), remaining=8)
    assert offers and offers[0].sequence == ({"u": 4.}, {"u": 3.})
    restored.feedback(offers[0], supported=True)
    assert restored.records[0]["uses"] == 1
    assert restored.last_recall[0]["role_mapping"] == {"x": "u"}
    # Existing agreement in the target makes this old experiment irrelevant.
    assert not restored.propose(target, [models[0], models[0]], old,
                                fallback=0., budget=SearchBudget(100), remaining=8)


def test_an_outcome_that_does_not_distinguish_models_does_not_become_experiment_knowledge():
    spec = TaskSpec("s", "response", ("x",), (), (), feature_bounds={"x": (0., 2.)})
    x = Program("feature", name="x")
    models = [Candidate("x", x, "refit"), Candidate("xx", Program("mul", children=(x, x)), "composition")]
    rows = [Observation({"x": value}, value) for value in (0., 1.)]
    library = KnowledgeLibrary()
    assert library.experiments.record(spec, models, rows, Probe("same", ({"x": 1.},)), rows[-1:],
                                      fallback=0., budget=SearchBudget(100)) is None
    assert KnowledgeLibrary.from_dict(library.to_dict()).experiments.records == []
