"""World claims are executable fits, and can lose on actual unseen responses."""

from random import Random

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Definition, EvaluationContext, Program
from hegel_machine.cognition.meta_prior_claims import compile_claims
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec


def spec():
    return TaskSpec("unknown_response", "Predict observed outcomes", ("x", "z"), (), (),
                    feature_bounds={"x": (-2., 2.), "z": (-2., 2.)})


def rows(function, seed=117, count=32):
    rng = Random(seed)
    return [Observation(point, function(point["x"], point["z"]))
            for point in ({"x": rng.uniform(-2., 2.), "z": rng.uniform(-2., 2.)} for _ in range(count))]


def test_projection_is_fitted_from_unknown_quadratic_and_predicts_unseen_points():
    response = lambda x, z: .7 + .2*(.4*x-1.3*z) + 1.7*(.4*x-1.3*z)**2
    training, validation = rows(response), rows(response, seed=918)
    budget = SearchBudget(10000)
    claims = compile_claims(spec(), training, budget)
    projection = next(item for item in claims if item["kind"] == "sufficient_projection")
    additive = next(item for item in claims if item["kind"] == "additive_quadratic")
    assert projection["effective_dimension"] == 1
    assert loss(projection["candidate"].program, validation, 0., .5) < 1e-12
    assert loss(additive["candidate"].program, validation, 0., .5) > .1
    assert projection["counter_prediction"]["null_directions"]
    assert all(item["family_id"] in {"T02", "T03", "T05", "T08"} for item in claims)
    assert not any(item["kind"] == "finite_history" for item in claims)
    assert 0 < budget.spent <= budget.limit


def test_nullspace_intervention_refutes_a_previously_sufficient_projection():
    claims = compile_claims(spec(), rows(lambda x, z: (x+2*z)**2), SearchBudget(10000))
    prior = next(item for item in claims if item["kind"] == "sufficient_projection")
    nx, nz = prior["counter_prediction"]["null_directions"][0]
    points = [(.3 + d*nx, -.1 + d*nz) for d in (-1., -.5, 0., .5, 1.)]
    predictions = [prior["candidate"].program.predict(Observation({"x": x, "z": z}, 0.)) for x, z in points]
    assert max(predictions) - min(predictions) < 1e-7
    changed = [Observation({"x": x, "z": z}, (x+2*z)**2 + .8*(2*x-z)**2) for x, z in points]
    assert loss(prior["candidate"].program, changed, 0., .5) > .1
    expanded = compile_claims(spec(), rows(lambda x, z: (x+2*z)**2+.8*(2*x-z)**2), SearchBudget(10000))
    assert not any(item["kind"] == "sufficient_projection" for item in expanded)


def test_learned_call_executes_a_new_readout_under_budget_too_small_for_raw_search():
    arg = Program("arg", name="x")
    body = Program("mul", children=(arg, Program("mul", children=(arg, arg))))
    definition = Definition("learned_cube", ("x",), body, "observed_cube")
    call = Program("call", name="learned_cube", children=(Program("feature", name="x"),), definition=definition)
    training = rows(lambda x, z: .4 + 2*x**3 - .6*x**6)
    budget = SearchBudget(4*len(training))
    claims = compile_claims(spec(), training, budget, seed_programs=(call,))
    assert len(claims) == 1 and claims[0]["kind"] == "reuse_compiled"
    program = claims[0]["candidate"].program
    assert loss(program, rows(lambda x, z: .4+2*x**3-.6*x**6, seed=322), 0., .5) < 1e-12
    context = EvaluationContext(training[0])
    program.predict(training[0], context=context)
    assert context.call_evaluations == 1
    assert budget.spent == budget.limit


def test_finite_history_uses_observed_inputs_and_predicts_startup():
    training = []
    rng = Random(133)
    for _ in range(8):
        history = []
        for _ in range(8):
            point = {"x": rng.uniform(-1., 1.), "z": rng.uniform(-1., 1.)}
            target = .3*point["x"] + (history[-1]["z"] if history else 0.)
            training.append(Observation(point, target, tuple(history)))
            history.append(point)
    claims = compile_claims(spec(), training, SearchBudget(20000))
    historical = next(item for item in claims if item["kind"] == "finite_history" and item["history_depth"] == 1)
    program = historical["candidate"].program
    assert loss(program, [row for row in training if row.history], 0., .5) < 1e-12
    assert isinstance(program.predict(training[0]), float)
    assert all(term.name != "__target__" for term in historical["terms"])


def test_budget_does_not_spend_on_a_partial_candidate_fit():
    training = rows(lambda x, z: x-z)
    budget = SearchBudget(4*len(training)-1)
    assert compile_claims(spec(), training, budget) == []
    assert budget.spent == 0


def test_observed_nonlinear_slope_change_is_fitted_beyond_a_quadratic_response():
    response = lambda x, z: .2 + .4*x*x + .7*z - .3*x*z + .6*abs(x-z)
    training, validation = rows(response, count=48), rows(response, seed=313, count=80)
    claims = compile_claims(spec(), training, SearchBudget(12000))
    smooth = next(item for item in claims if item["kind"] == "coupled_quadratic")
    piecewise = next(item for item in claims if item["family_id"] == "T11")
    assert loss(smooth["candidate"].program, validation, 0., .5) > .01
    assert loss(piecewise["candidate"].program, validation, 0., .5) < 1e-12
    boundary = [Observation({"x": .3+offset, "z": .3}, response(.3+offset, .3))
                for offset in (-.04, -.01, 0., .01, .04)]
    assert loss(piecewise["candidate"].program, boundary, 0., .5) < 1e-12


def test_new_boundary_research_preserves_a_previously_learned_program():
    learned = compile_claims(spec(), rows(lambda x, z: (x+z)**2), SearchBudget(10000))
    old = next(item["candidate"].program for item in learned if item["kind"] == "sufficient_projection")
    before = old.to_dict()
    revised = compile_claims(spec(), rows(lambda x, z: (x+z)**2+.5*abs(x-z)),
                             SearchBudget(12000), seed_programs=(old,))
    assert old.to_dict() == before
    assert any(item["family_id"] == "T11" and item["training_loss"] < 1e-12 for item in revised)
    assert loss(old, rows(lambda x, z: (x+z)**2, seed=744), 0., .5) < 1e-12
