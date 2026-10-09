"""Executable workflows remain subordinate to paid legacy research and confirmation."""

from dataclasses import replace

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.growth_methods import ResearchMethodBook, instantiate
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Probe, SearchBudget
from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab
from hegel_machine.cognition.workflow_execution import WorkflowEpisode


def _claim(program, training, validation):
    return {"candidate": Candidate("measured", program, "fit"),
            "training_loss": loss(program, training, 0., .5),
            "validation_loss": loss(program, validation, 0., .5)}


def _paid(episode, lab, name, values):
    probe = Probe(name, tuple({"u": value} for value in values))
    rows = lab.acquire(probe).observations
    episode.acquired(probe, (), rows, purpose=name)
    return rows


def _true_program():
    return Program("add", children=(Program("const", value=.15),
        Program("mul", children=(Program("const", value=.55), Program("feature", name="u")))))


def _learned_episode(training_values=(-.9, .1, .7, -.4)):
    lab = UnknownTaskLab(0)
    episode = WorkflowEpisode(ResearchMethodBook(), lab.spec, lab.channels)
    training = _paid(episode, lab, "fit", training_values)
    validation = _paid(episode, lab, "validation", (-.6, .8))
    program = _true_program()
    episode.fitted([_claim(program, training, validation)], training, validation, lab.spec)
    fresh = _paid(episode, lab, "confirmation", (-.3, .5))
    episode.tested(program, fresh, loss(program, fresh, 0., .5), accepted=True)
    method = episode.finish(program, confirmed=True, threshold=1e-7, actions=[])
    assert method is not None
    return episode, lab


def test_native_workflow_replays_paid_rows_and_leaves_confirmation_to_native_gate():
    source, _ = _learned_episode()
    target = UnknownTaskLab(0, seed=212)
    episode = WorkflowEpisode(source.book, target.spec, target.channels, attempt=1)
    actions, fit_rows = [], []
    compute = SearchBudget(1000)

    def acquire(channels, purpose, *, probe):
        rows = target.acquire(probe, channels).observations
        episode.acquired(probe, channels, rows, purpose=purpose)
        return rows

    def fit(channels, training, validation):
        fit_rows.append((list(training), list(validation)))
        assert compute.spend(len(training) + len(validation))
        # A fixed candidate is sufficient here: the test concerns actual data
        # routing and the native confirmation boundary, not model search.
        claims = [_claim(_true_program(), training, validation)]
        episode.fitted(claims, training, validation, target.spec)
        return claims

    result = episode.replay(acquire=acquire, project=lambda rows, *_: rows, fit=fit,
        legal_views=[(), ("sensor_a",), ("sensor_b",), ("sensor_a", "sensor_b")],
        remaining=lambda: {"observation_remaining": 20,
            "measurement_remaining": 0, "compute_remaining": compute.remaining},
        confirmation_length=2, threshold=1e-7, diagnostics={}, actions=actions)
    assert result["adequate"]
    assert target.observation_cost == 6
    assert [len(rows) for rows in fit_rows[0]] == [4, 2]
    assert len(source.book.methods[0]["examples"]) == 1
    assert episode.finish(_true_program(), confirmed=False, threshold=1e-7, actions=actions) is None
    fresh = _paid(episode, target, "native_fresh", (-.2, .6))
    episode.tested(_true_program(), fresh, loss(_true_program(), fresh, 0., .5), accepted=True)
    assert episode.finish(_true_program(), confirmed=True, threshold=1e-7, actions=actions)
    assert target.observation_cost == 8
    assert target.measurement_cost == 0
    assert any(item["action"] == "confirm_research_workflow" for item in actions)
    # A shorter public trajectory limit cannot silently truncate the method.
    short = WorkflowEpisode(source.book, replace(target.spec, max_probe_length=3), target.channels)
    before = target.observation_cost
    assert short.replay(acquire=acquire, project=lambda rows, *_: rows, fit=fit,
        legal_views=[()], remaining=lambda: {"observation_remaining": 20,
            "measurement_remaining": 0, "compute_remaining": 1000},
        confirmation_length=2, threshold=1e-7, diagnostics={}, actions=actions) is None
    assert target.observation_cost == before


def test_failed_confirmation_keeps_its_frozen_fit_when_same_program_is_refitted():
    source, _ = _learned_episode()
    lab = UnknownTaskLab(0, seed=301)
    episode = WorkflowEpisode(source.book, lab.spec, lab.channels)
    # This candidate predicts the correct response only at the measured u=0.
    constant = Program("const", value=.15)
    training = _paid(episode, lab, "initial_fit", (0., 0., 0., 0.))
    validation = _paid(episode, lab, "initial_check", (0., 0.))
    episode.fitted([_claim(constant, training, validation)], training, validation, lab.spec)
    first_fit = episode.fit_ids[constant]
    episode.replayed = source.book.methods[0]
    episode.replay_program = constant
    episode.replay_fit_id = first_fit
    failed = _paid(episode, lab, "failed_fresh", (-1., 1.))
    episode.tested(constant, failed, loss(constant, failed, 0., .5), accepted=False)
    # Paid counterexamples can feed a subsequent fit even if it returns the
    # identical AST. They must not become descendants of that later fit.
    episode.fitted([_claim(constant, [*training, *failed], validation)],
                   [*training, *failed], validation, lab.spec)
    later_fit = episode.fit_ids[constant]
    passed = _paid(episode, lab, "later_fresh", (0., 0.))
    episode.tested(constant, passed, loss(constant, passed, 0., .5), accepted=True)
    method = episode.finish(constant, confirmed=True, threshold=1e-7, actions=[])
    failure = next(event for event in episode.trace if event["recipe"].get("sequence") == [{"u": -1.}, {"u": 1.}])
    assert failure["details"]["model_confirmed"] is False
    assert failure["consumes"] == [first_fit]
    assert failure["consumes"] != [later_fit]
    assert source.book.methods[0]["examples"][-1]["success"] is False
    assert method["parent_method_id"] == source.book.methods[0]["method_id"]
    seen = set()
    for step in instantiate(method, lab.spec, lab.channels):
        assert set(step["depends_on"]) <= seen
        seen.add(step["node_id"])


def test_growth_comparison_runs_native_fits_on_one_legal_purchase():
    from hegel_machine.cognition.measurement_lab import MeasurementChannel
    from hegel_machine.cognition.models import Observation, TaskSpec

    channels = (MeasurementChannel("coarse", 1, "Absolute reading"),
                MeasurementChannel("fine", 2, "Signed reading"))
    spec = TaskSpec("comparison", "Predict response", ("u",), (), (),
                    feature_bounds={"u": (-1., 1.)}, max_probe_length=12)
    sequence = tuple({"u": value} for value in (-1., -.7, -.4, -.1, .2, .5, .8, 1.))

    def measured(probe):
        return [Observation({"u": point["u"], "coarse": abs(point["u"]),
                             "fine": point["u"]}, 2. * point["u"]) for point in probe.sequence]

    def program(name):
        return Program("mul", children=(Program("const", value=2.), Program("feature", name=name)))

    source_rows = measured(Probe("source", sequence))
    measured_losses = {name: loss(program(name), source_rows, 0., .5) for name in ("coarse", "fine")}
    assert measured_losses["coarse"] > .1 and measured_losses["fine"] == 0.
    book = ResearchMethodBook()
    book.learn([
        {"event_id": "pilot", "action": "measure", "produces": ["paid"],
         "recipe": {"sequence": sequence, "channel_binding": "compare_available"}},
        {"event_id": "fit", "action": "fit", "consumes": ["paid"], "produces": ["model"],
         "recipe": {"estimator": "residual_expression_search", "instrument_comparison": {
             "candidate_channels": ["coarse", "fine"], "losses": measured_losses}}},
        {"event_id": "confirm", "action": "confirm", "consumes": ["model"],
         "recipe": {"sequence": ({"u": -.3}, {"u": .6}), "channels": ["fine"]},
         "details": {"model_confirmed": True}},
    ], {"feature_bounds": spec.feature_bounds, "channel_names": ["coarse", "fine"],
        "feature_count": 1}, success=True, task_id="source")
    episode = WorkflowEpisode(book, spec, channels)
    purchases, fitted_views, actions = [], [], []

    def acquire(names, purpose, *, probe):
        assert names == ("fine",)  # Joint coarse+fine is not a legal purchase.
        purchases.append(probe)
        rows = measured(probe)
        episode.acquired(probe, names, rows, purpose=purpose)
        return rows

    def project(rows, source, names):
        projected = [Observation({"u": row.features["u"],
                    **{name: row.features[name] for name in names}}, row.target) for row in rows]
        episode.projected(rows, projected)
        return projected

    def fit(names, training, validation):
        fitted_views.append(names)
        claims = [_claim(program(names[0]), training, validation)]
        episode.fitted(claims, training, validation,
                       replace(spec, feature_names=("u", *names)))
        return claims

    result = episode.replay(acquire=acquire, project=project, fit=fit,
        legal_views=[(), ("coarse",), ("fine",)],
        projectable=lambda names: [(), ("coarse",), ("fine",)] if names == ("fine",) else [(), names],
        remaining=lambda: {"observation_remaining": 20, "measurement_remaining": 40,
                           "compute_remaining": 1000},
        confirmation_length=2, threshold=1e-7, diagnostics={}, actions=actions)
    assert result["adequate"] and result["selected"][1] == ("fine",)
    assert len(purchases) == 1 and purchases[0].cost == 8
    assert fitted_views == [("coarse",), ("fine",)]
    assert len(book.methods[0]["examples"]) == 1  # Native fresh check still required.


def test_legacy_temporal_research_compiles_and_executes_with_native_costs():
    from hegel_machine.cognition.instrument_method_lab import InstrumentMethodLab
    from hegel_machine.cognition.memory import KnowledgeLibrary
    from hegel_machine.cognition.models import ResearchConfig
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent

    config = ResearchConfig(observation_budget=192, compute_budget=120000,
                            minimum_improvement=1e-7, max_candidates=16)
    library = KnowledgeLibrary()
    source_lab = InstrumentMethodLab(0, seed=113)
    source = UnifiedResearchAgent(config, library=library).run(source_lab)
    assert source["status"] == "completed" and source["research_workflow_id"] is not None
    method = library.research_workflows.methods[0]
    assert any(node["action"] == "fit" for node in method["nodes"])
    assert method["nodes"][-1]["action"] == "confirm"
    assert all(step["probe"].cost <= source_lab.spec.max_probe_length
               for step in instantiate(method, source_lab.spec, source_lab.channels) if "probe" in step)
    produced = [production for concept in library.concepts for production in concept.production_methods
                if production["kind"] == "research_workflow"]
    assert produced and produced[0]["method_id"] == source["research_workflow_id"]
    assert produced[0]["source_concept_id"] in source["concepts_added"]
    assert produced[0]["diagnostics"]["transfer_condition"]

    target_lab = InstrumentMethodLab(2, seed=113)
    target = UnifiedResearchAgent(config, library=library).run(target_lab)
    assert any(item["action"] == "replay_research_workflow" for item in target["actions"])
    replayed = [item for item in target["actions"] if item["action"] == "acquire"
                and item["purpose"] == "replay_research_workflow"]
    assert replayed
    assert target["costs"]["observation_cost"] == target_lab.observation_cost <= config.observation_budget
    assert target["costs"]["measurement_cost"] == target_lab.measurement_cost
    assert target["costs"]["compute_cost"] <= config.compute_budget
    assert all(len(item["sequence"]) <= target_lab.spec.max_probe_length for item in target_lab.acquisitions)
    native_checks = [item for item in target["actions"] if item["action"] == "validate_frozen_program"]
    assert native_checks
    assert target["status"] == "completed"
    assert not native_checks[-1]["accepted"]
    reopening = next(index for index, item in enumerate(target["actions"])
                     if item["action"] == "reopen_workflow_observation_research")
    repaired = [item for item in target["actions"][reopening + 1:]
                if item["action"] == "confirm_observation_research" and item["accepted"]]
    assert repaired and repaired[-1]["after_loss"] <= config.minimum_improvement
    assert target["confirmation_loss"] <= config.minimum_improvement
    # A workflow failure is retained as a failure, never a terminal gate bypass.
    for action in target["actions"]:
        if action["action"] == "confirm_research_workflow":
            assert action["basis"] == "native_fresh_confirmation"



def test_unavailable_high_return_plan_yields_to_executable_alternative_and_carries_demand():
    from copy import deepcopy
    from test_workflow_knowledge_returns import _setup, _consume

    # Returns come from a real paid library comparison. The executable plans
    # are independently compiled from actual acquisitions by the same bridge.
    library, paid_diagnostics, high_return, methods = _setup()
    outcome = _consume(library)
    assert outcome["method_returns"][0]["retained_gain"] > 0.
    long, _ = _learned_episode()
    short, _ = _learned_episode(training_values=(-.9, .7))
    library.research_workflows.methods[0]["nodes"] = deepcopy(long.book.methods[0]["nodes"])
    library.research_workflows.methods[1]["nodes"] = deepcopy(short.book.methods[0]["nodes"])
    demand = {"task_id": "paid_unresolved_consumer", "paid_rows": 5,
              "condition": deepcopy(paid_diagnostics["transfer_condition"]), "status": "unresolved"}
    diagnostics = {**paid_diagnostics, "downstream_demands": [demand]}
    book = library.research_workflows
    assert book.propose(diagnostics)["method_id"] == high_return

    # The first run excludes the high-value plan by public probe length; the
    # second excludes it by the remaining full acquisition+confirmation bill.
    for maximum, available_rows, expected_reason in (
            (3, 20, "trajectory_view_or_solver_not_available"),
            (96, 7, "execution_and_confirmation_not_funded")):
        lab = UnknownTaskLab(0, seed=613)
        episode = WorkflowEpisode(book, replace(lab.spec, max_probe_length=maximum), lab.channels)
        actions, purchases = [], []

        def acquire(channels, purpose, *, probe):
            purchases.append(probe)
            rows = lab.acquire(probe, channels).observations
            episode.acquired(probe, channels, rows, purpose=purpose)
            return rows

        def fit(channels, training, validation):
            claims = [_claim(_true_program(), training, validation)]
            episode.fitted(claims, training, validation, lab.spec)
            return claims

        report = episode.replay(acquire=acquire, project=lambda rows, *_: rows, fit=fit,
            legal_views=[()], remaining=lambda: {"observation_remaining": available_rows,
                "measurement_remaining": 0, "compute_remaining": 1000},
            confirmation_length=2, threshold=1e-7, diagnostics=diagnostics, actions=actions)
        assert report["adequate"]
        assert lab.observation_cost == 4 and [probe.cost for probe in purchases] == [2, 2]
        chosen = next(item for item in actions if item["action"] == "replay_research_workflow")
        assert chosen["method_id"] == methods[1]["method_id"]
        assert [item["method_id"] for item in chosen["alternatives"]] == [methods[1]["method_id"]]
        assert chosen["diagnostics"]["downstream_demands"] == [demand]
        assert chosen["selection_value"] == book.knowledge_value(methods[1], diagnostics)
        assert any(item["action"] == "defer_research_workflow" and
            item["method_id"] == high_return and item["reason"] == expected_reason for item in actions)
        assert episode.diagnostics["downstream_demands"] is not diagnostics["downstream_demands"]
        assert len(book.methods[0]["examples"]) == 1
        assert len(book.methods[1]["examples"]) == 1
