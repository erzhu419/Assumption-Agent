from random import Random
from statistics import mean

from hegel_machine.cognition.growth_models import abstract_program, bind_prior, discover
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget


def _feature(name):
    return Program("feature", name=name)


def _rows(seed, count, function):
    random = Random(seed)
    return [Observation({"x": (x := random.uniform(-2, 2)), "y": (y := random.uniform(-2, 2))},
        function(x, y)) for _ in range(count)]


def test_residual_growth_discovers_products_and_cubes_then_reuse_saves_compute():
    response = lambda x, y: x * y + x ** 3
    training, validation = _rows(71, 60, response), _rows(79, 30, response)
    budget = SearchBudget(50000)
    found = discover(training, validation, budget)
    assert found["loss"] < 1e-8
    assert any(event["action"] == "grow_residual_composition" and event["degree"] == 3
        for event in found["events"])
    terminal = _rows(101, 48, response)
    assert max(abs(found["program"].predict(row) - row.target) for row in terminal) < 1e-6
    assert sum(event["compute_cost"] for event in found["events"]) == budget.spent <= budget.limit
    next_budget = SearchBudget(50000)
    reused = discover(_rows(103, 60, response), _rows(107, 30, response), next_budget,
        priors=[found["program"]])
    assert reused["program"] == found["program"]
    assert reused["events"][-1]["action"] == "reuse_prior"
    assert next_budget.spent < budget.spent


def test_role_binding_recovers_reordered_names_from_observations():
    program = Program("add", children=(_feature("x"),
        Program("mul", children=(Program("const", value=2.), _feature("y")))))
    rows = _rows(43, 32, lambda x, y: x + 2 * y)
    renamed = [Observation({"alpha": row.features["y"], "zeta": row.features["x"]}, row.target)
        for row in rows]
    bindings = bind_prior(program, ("alpha", "zeta"), renamed, SearchBudget(1000))
    assert bindings[0]["mapping"] == {"x": "zeta", "y": "alpha"}
    assert bindings[0]["loss"] < 1e-12
    found = discover(renamed[:20], renamed[20:], SearchBudget(3000), priors=[program])
    assert found["loss"] < 1e-12
    assert found["events"][-1]["action"] == "reuse_bound_prior"
    definition = abstract_program(program, "learned_response")
    call = Program("call", name=definition.definition_id, definition=definition,
        children=(_feature("zeta"), _feature("alpha")))
    assert all(abs(call.predict(row) - row.target) < 1e-12 for row in renamed)
    assert definition.arguments == ("role_0", "role_1")
    source_call = Program("call", name=definition.definition_id, definition=definition,
        children=(_feature("x"), _feature("y")))
    rebound = bind_prior(source_call, ("alpha", "zeta"), renamed, SearchBudget(1000))[0]
    assert rebound["program"].op == "call"
    assert rebound["program"].definition == definition
    assert rebound["loss"] < 1e-12


def test_learned_scope_keeps_old_program_and_predicts_new_branch():
    x, y = _feature("x"), _feature("y")
    prior = Program("add", children=(Program("mul", children=(x, x)),
        Program("mul", children=(Program("const", value=.5), Program("mul", children=(x, y))))))
    def observations(xs, ys):
        return [Observation({"x": a, "y": b}, a * a + .5 * a * b + (.8 * a if a >= 0 else 0.))
            for a in xs for b in ys]
    train = observations((-2., -1., -.5, -.1, .1, .5, 1., 2.), (-1., -.4, .3, 1.2))
    validation = observations((-1.7, -.2, .2, 1.7), (-.8, .7))
    budget = SearchBudget(10000)
    result = discover(train, validation, budget, priors=[prior])
    assert result["loss"] < 1e-8
    assert result["program"].op == "if"
    predicate, old, new = result["program"].children
    assert old == prior
    assert predicate.children[1].value == 0.
    assert new != prior
    terminal = observations((-1.9, -.03, .03, 1.9), (-1.1, .9))
    assert max(abs(result["program"].predict(row) - row.target) for row in terminal) < 1e-6
    assert any(event["action"] == "learn_executable_scope" for event in result["events"])
    assert result["used_priors"] == [prior]
    assert result["scope_boundaries"] == [{"feature": "x", "threshold": 0., "lower": -.1, "upper": .1}]
    assert sum(event["compute_cost"] for event in result["events"]) == budget.spent <= budget.limit


def test_budget_exhaustion_retains_a_paid_candidate_deterministically():
    rows, validation = _rows(17, 24, lambda x, y: x * y), _rows(19, 12, lambda x, y: x * y)
    first_budget, second_budget = SearchBudget(220), SearchBudget(220)
    first = discover(rows, validation, first_budget)
    second = discover(rows, validation, second_budget)
    assert first["program"] == second["program"]
    assert first["events"] == second["events"]
    assert first_budget.spent == second_budget.spent <= 220
    assert first["program"] in first["alternatives"]
    assert first["loss"] >= 0


def test_supported_scoped_prior_can_be_simplified_on_the_current_evidence():
    x, y = _feature("x"), _feature("y")
    prior = Program("if", children=(Program("lt", children=(x, Program("const", value=0.))),
        y, Program("mul", children=(y, y))))
    random = Random(31)
    rows = [Observation({"x": random.uniform(-2., -.1), "y": (value := random.uniform(-2., 2.))}, value)
        for _ in range(48)]
    budget = SearchBudget(5000)
    result = discover(rows[:32], rows[32:], budget, priors=[prior])
    assert result["program"] == y
    assert result["loss"] < 1e-8
    assert result["used_priors"] == []
    assert result["events"][-1]["action"] == "simplify_supported_prior"
    assert sum(event["compute_cost"] for event in result["events"]) == budget.spent


def test_scratch_absolute_pair_relation_generalizes_beyond_the_fitting_rows():
    response = lambda x, y: .9 * x * x + .4 * x * y + .3 * abs(x - y)
    budget = SearchBudget(50000)
    result = discover(_rows(71,60,response), _rows(79,30,response), budget)
    terminal = _rows(101,48,response)
    assert max(abs(result["program"].predict(row) - row.target) for row in terminal) < 1e-6
    assert any(event["action"] == "grow_absolute_relation"
        and event["term"]["op"] == "abs" for event in result["events"])
    assert "abs(sub(x, y))" in result["program"].describe()
    assert result["fit_diagnostics"]["validation_rows"] == 30
    assert sum(event["compute_cost"] for event in result["events"]) == budget.spent


def test_absolute_source_extends_with_novel_cubic_residual_then_reuses_the_result():
    original = lambda x, y: .9 * abs(x) + .4 * x * y
    source = discover(_rows(71,24,original), _rows(79,24,original), SearchBudget(50000))
    definition = abstract_program(source["program"], "absolute_source")
    prior = Program("call", name=definition.definition_id, definition=definition,
        children=(_feature("x"), _feature("y")))
    changed = lambda x, y: original(x, y) + .35 * y ** 3
    budget = SearchBudget(50000)
    result = discover(_rows(81,24,changed), _rows(83,24,changed), budget, priors=[prior])
    assert result["used_priors"] == [prior]
    assert max(abs(result["program"].predict(row) - row.target)
        for row in _rows(101,48,changed)) < 1e-6
    repeat_budget = SearchBudget(50000)
    reused = discover(_rows(107,24,changed), _rows(109,24,changed), repeat_budget,
        priors=[result["program"]])
    assert reused["program"] == result["program"]
    assert reused["events"][-1]["action"] == "reuse_prior"
    assert repeat_budget.spent < budget.spent


def test_scratch_hinge_learns_nonzero_threshold_without_an_old_prior():
    response = lambda x, y: .9 * x * x + .4 * x * y + .8 * max(0., x - .17)
    budget = SearchBudget(100000)
    result = discover(_rows(71,24,response), _rows(79,24,response), budget)
    terminal = _rows(101,200,response)
    assert mean((result["program"].predict(row) - row.target) ** 2 for row in terminal) < 1e-6
    assert result["used_priors"] == []
    boundary = result["scope_boundaries"][0]
    assert boundary["feature"] == "x"
    assert abs(boundary["threshold"] - .17) < .003
    assert boundary["lower"] < boundary["threshold"] < boundary["upper"]
    assert any(event["action"] == "refine_hinge_threshold" for event in result["events"])
    assert sum(event["compute_cost"] for event in result["events"]) == budget.spent <= budget.limit


def test_scratch_cubic_hinge_uses_the_same_expression_family_without_prior_memory():
    response = lambda x, y: .9 * x ** 3 + .4 * x * y + .8 * max(0., x)
    budget = SearchBudget(100000)
    result = discover(_rows(71,24,response), _rows(79,24,response), budget)
    assert mean((result["program"].predict(row) - row.target) ** 2
        for row in _rows(101,200,response)) < 1e-6
    assert result["scope_boundaries"][0]["feature"] == "x"
    assert sum(event["compute_cost"] for event in result["events"]) == budget.spent <= budget.limit


def test_retained_normalized_nonlinear_call_combines_with_new_measured_coordinate():
    x, y = _feature("x"), _feature("y")
    source = Program("add", children=(Program("mul", children=(Program("mul", children=(x,x)),x)),
        Program("mul", children=(x,y))))
    definition = abstract_program(source, "cubic_source")
    normalized = lambda name, center, scale: Program("mul", children=(Program("const",value=scale),
        Program("sub", children=(_feature(name), Program("const",value=center)))))
    prior = Program("call", name=definition.definition_id, definition=definition,
        children=(normalized("pressure",40.,.05),normalized("field",300.,.01)))

    def observations(seed, count):
        random = Random(seed)
        result = []
        for _ in range(count):
            a,b,sensor = random.uniform(-1,1),random.uniform(-1,1),random.uniform(-.6,.6)
            result.append(Observation({"pressure":40.+20.*a,"field":300.+100.*b,"sensor":sensor},
                a ** 3 + a*b + .7*sensor + .2))
        return result

    budget = SearchBudget(10000)
    result = discover(observations(71,24), observations(79,24), budget, priors=[prior])
    assert result["events"][-1]["action"] == "fit_prior_residual"
    assert result["used_priors"] == [prior]
    assert max(abs(result["program"].predict(row)-row.target)
        for row in observations(101,48)) < 1e-6
    assert not any(event["action"] == "grow_residual_composition" for event in result["events"])
    assert sum(event["compute_cost"] for event in result["events"]) == budget.spent < 300
