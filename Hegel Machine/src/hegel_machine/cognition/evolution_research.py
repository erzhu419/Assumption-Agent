"""A runnable, shared-library study of mechanism changes and recursive questions."""

from dataclasses import replace

from .controller import CognitiveAgent
from .evaluation import initial_prediction, loss
from .knowledge_evolution import _calls
from .language import Program
from .memory import KnowledgeLibrary
from .models import Candidate, Observation, Probe, ResearchConfig, SearchBudget, TaskSpec
from .research_graph import ResearchGraph
from .synthesis import _fit_linear
from .workspace import ResearchWorkspace


def _op(name, *children):
    return Program(name, children=children)


def _feature(name):
    return Program("feature", name=name)


class _MeasuredProcess:
    """Expose observations and control trajectories through the ordinary interface."""

    def __init__(self, name, names, response, probes, sensor=None):
        self.response, self.sensor = response, sensor
        self.spec = TaskSpec(name, "Predict the acquired process response", tuple(names), (), tuple(probes),
                             max_probe_length=8)

    def observe(self, probe):
        result, history = [], []
        for control in probe.sequence:
            features = dict(control)
            if self.sensor:
                features.update(self.sensor(features))
            target = float(self.response(features))
            result.append(Observation(features, target, tuple(history)))
            history.append({**features, "__target__": target})
        return result


def _fit_source(library, name, variable, response, degree, budget):
    """Fit numerical coefficients from measurements in a polynomial vocabulary."""
    values = tuple(-2. + index / 6. for index in range(25))
    rows = [Observation({variable: value}, response(value)) for value in values]
    coordinate, terms = _feature(variable), []
    power = coordinate
    for _ in range(degree):
        terms.append(power)
        power = _op("mul", power, coordinate)
    if not budget.spend(len(rows) * (len(terms) + 1)):
        raise ValueError("source-learning budget cannot fund the coordinate fit")
    program = _fit_linear(tuple(terms), rows)
    if not budget.spend(len(rows)):
        raise ValueError("source-learning budget cannot fund its comparison")
    measured = loss(program, rows, 0., .5)
    spec = TaskSpec(name, "Fit an observed component response", (variable,), (), ())
    identifier, _ = library.learn(Candidate(name, program, "composition"), spec, rows)
    call = next(atom for atom in library.callable_atoms(spec, limit=64)
                if atom.definition.source_concept_id == identifier and atom.name.endswith(":abstract:0:definition"))
    return call, {"concept_id": identifier, "training_mse": measured, "rows": len(rows)}


def _workspace(task, program, rows, *, reference=(), graph=None):
    return ResearchWorkspace(task.spec.task_id, observations=list(rows), reference=list(reference),
        incumbent=Candidate("retained_model", program, "composition"), initial_loss=100., fallback=0.,
        graph=graph.to_dict() if graph else {}).to_dict()


def _episode(result):
    relevant = {"repair_mechanism", "merge_concepts", "split_concepts", "execute_research_graph",
                "supply_research_measurements", "revise", "rollback", "compare", "reuse_check",
                "merge_adopted", "merge_retained", "split_adopted", "split_rejected",
                "bind_evolution_knowledge", "knowledge_evolution_comparison"}
    return {"program": result.program.to_dict(), "expression": result.program.describe(),
            "compute_cost": result.compute_cost, "observation_cost": result.observation_cost,
            "selected_operation": result.selected_operation, "selected_memory_id": result.selected_memory_id,
            "actions": [{"action": action.action, "compute_cost": action.compute_cost, "details": action.details}
                        for action in result.actions if action.action in relevant],
            "research_state": result.research_state}


def _called_sources(program):
    sources = set()
    for call in _calls(program):
        if call.definition.source_concept_id:
            sources.add(call.definition.source_concept_id)
        sources.update(_called_sources(call.definition.body))
    return sorted(sources)


def _continue_with_callable_knowledge(library, task, acquired, config):
    """Offer eligible library calls together; ordinary comparison chooses one."""
    fallback = initial_prediction(acquired, task.spec.risk_quantile)
    calls = library.callable_atoms(task.spec, limit=64)
    bank = [Candidate(f"call:{index}:{call.name}", call, "reuse",
                      "Execute an eligible learned computation", memory_id=call.definition.source_concept_id)
            for index, call in enumerate(calls)]
    constant = Program("const", value=fallback)
    workspace = ResearchWorkspace(task.spec.task_id, observations=list(acquired),
        incumbent=Candidate("initial_prediction", constant, "refit"), bank=bank, fallback=fallback,
        initial_loss=loss(constant, acquired, fallback, task.spec.risk_quantile))
    result = CognitiveAgent(config, library).run(task, research_state=workspace.to_dict())
    return result, {"offered_calls": len(calls), "called_sources": _called_sources(result.program)}


def run_evolution_research(library=None):
    """Learn sources, repair/merge/split through the controller, and resolve a DAG.

    The supplied knowledge library is updated in place. Task responses are
    accessed through observe; candidate coefficients come from acquired rows.
    Each resumed controller visit owns a fresh bounded experiment budget.
    """
    library = library if library is not None else KnowledgeLibrary()
    source_work = SearchBudget(2000)
    square, quadratic_source = _fit_source(library, "evolution:quadratic", "x", lambda x: x*x, 2, source_work)
    linear, linear_source = _fit_source(library, "evolution:linear", "z", lambda z: 2*z, 1, source_work)
    config = ResearchConfig(compute_budget=22000, observation_budget=8, max_rounds=1,
        strategy="fixed", use_abstractions=False, active_probes=False, state_search=False,
        joint_planning=False, use_warm_start=False, transition_probes=False, learn_research_policy=False)

    original = _op("add", square, linear)
    acquired_probe = Probe("component_training", tuple({"x": x, "z": z, "regime": regime}
        for regime in (-1., 1.) for x in (-2., -1., 1., 2.) for z in (-2., -1., 1., 2.)))
    fresh_probe = Probe("component_fresh", tuple({"x": x, "z": z, "regime": regime}
        for regime in (-.5, .5) for x, z in ((-1.5, -.7), (-.2, .3), (.6, -1.1), (1.7, 1.4))))
    components = _MeasuredProcess("evolution:components", ("x", "z", "regime"),
        lambda row: row["x"]**2 + (2 if row["regime"] < 0 else 3)*row["z"], (fresh_probe,))
    acquired = components.observe(acquired_probe)
    supported = [row for row in acquired if row.features["regime"] < 0]
    library.learn(Candidate("source_composition", original, "composition"), components.spec, supported)
    repaired = CognitiveAgent(config, library).run(components,
        research_state=_workspace(components, original, acquired, reference=supported))

    delta = lambda value: value*(value*value-1)*(value*value-4)
    nonlinear, nonlinear_source = _fit_source(library, "evolution:distinguishable", "x",
        lambda x: x*x+delta(x), 5, source_work)
    separate = _op("add", nonlinear, square)
    agreement = Probe("merge_agreement", tuple({"x": x} for x in (-2., -1., 0., 1., 2., -2., 0., 2.)))
    merge_task = _MeasuredProcess("evolution:merge", ("x",),
        lambda row: 2*row["x"]**2+delta(row["x"]), (agreement,))
    merged = CognitiveAgent(config, library).run(merge_task,
        research_state=_workspace(merge_task, separate, merge_task.observe(agreement)))
    # Restoration uses exactly the ordinary persistence representation.
    merge_checkpoint = library.to_dict()
    persisted = KnowledgeLibrary.from_dict(merge_checkpoint)
    changed = Probe("merge_counterexample", tuple({"x": x} for x in (.2, .3, .4, .5, .6, .7, .8, .9)))
    split_fresh = Probe("split_prospective", tuple({"x": x}
        for x in (-1.8, -1.3, -.6, -.15, .15, .45, 1.25, 1.8)))
    split_task = _MeasuredProcess(merge_task.spec.task_id, ("x",), merge_task.response, (split_fresh,))
    continued = ResearchWorkspace.from_dict(merged.research_state)
    counterexamples = split_task.observe(changed)
    continued.observations.extend(counterexamples)
    continued.queried.append(changed)
    split = CognitiveAgent(config, persisted).run(split_task, research_state=continued.to_dict())
    # Keep the caller's shared library object, including the newly revised state.
    library.__dict__.update(persisted.__dict__)

    # The earlier merge also gets a control: its current shared law continues
    # to predict new inputs correctly. The old separate program still competes.
    control_probe = Probe("merge_still_supported", tuple({"x": x}
        for x in (-1.6, -1.1, -.7, -.25, .25, .55, 1.1, 1.6)))
    control_task = _MeasuredProcess(merge_task.spec.task_id, ("x",),
        lambda row: 2*row["x"]**2, (control_probe,))
    control_library = KnowledgeLibrary.from_dict(merge_checkpoint)
    merge_control = CognitiveAgent(config, control_library).run(control_task, research_state=merged.research_state)

    # A later problem uses the actual learned library, including restored
    # definitions and revised merge scope. No candidate is selected by its ID.
    transfer_probe = Probe("split_transfer_prospective", tuple({"x": x}
        for x in (-1.75, -1.15, -.65, -.05, .35, .65, 1.35, 1.75)))
    transfer_task = _MeasuredProcess("evolution:split_transfer", ("x",), merge_task.response, (transfer_probe,))
    transfer_initial = transfer_task.observe(Probe("split_transfer_initial", tuple({"x": x}
        for x in (-1.9, -1.4, -.8, -.35, .1, .4, 1.15, 1.9))))
    restored_library = KnowledgeLibrary.from_dict(library.to_dict())
    transferred, transfer_details = _continue_with_callable_knowledge(
        restored_library, transfer_task, transfer_initial, config)
    library.__dict__.update(restored_library.__dict__)

    sensor_probes = tuple(Probe(f"sensor_fresh:{index}", tuple({"x": x}
        for x in (-1.4, -.9, -.6, -.1, .3, .7, 1.1, 1.6))) for index in range(2))
    sensor_task = _MeasuredProcess("evolution:recursive_measurement", ("x", "sensor"),
        lambda row: row["sensor"]**2, sensor_probes, sensor=lambda row: {"sensor": row["x"]**3})
    sensor_training = sensor_task.observe(Probe("sensor_training", tuple({"x": x} for x in (-1.5, -.8, -.3, .2, .6, 1.2))))
    graph = ResearchGraph()
    graph.add_question("sensor", "Read the actual added measurement", required_measurements=("sensor",),
        expression=_feature("sensor"), observations=[Observation({"x": row.features["x"]}, row.target)
                                                    for row in sensor_training[:3]])
    graph.add_question("response", "Test the response formed from the measurement", depends_on=("sensor",),
        expression=_op("mul", Program("arg", name="sensor"), Program("arg", name="sensor")))
    sensor_result = CognitiveAgent(replace(config, observation_budget=16, max_rounds=2), library).run(sensor_task,
        research_state=_workspace(sensor_task, _op("mul", _feature("x"), _feature("x")), sensor_training, graph=graph))

    old_rows = supported
    new_rows = ResearchWorkspace.from_dict(repaired.research_state).observations[-fresh_probe.cost:]
    split_rows = ResearchWorkspace.from_dict(split.research_state).observations[-split_fresh.cost:]
    control_rows = ResearchWorkspace.from_dict(merge_control.research_state).observations[-control_probe.cost:]
    transfer_rows = ResearchWorkspace.from_dict(transferred.research_state).observations[-transfer_probe.cost:]
    sensor_rows = ResearchWorkspace.from_dict(sensor_result.research_state).observations[-sensor_probes[-1].cost:]
    scored = SearchBudget(sum(map(len, (old_rows, new_rows, counterexamples, split_rows, control_rows, transfer_rows, sensor_rows))))
    scored.spend(scored.limit)
    merge_comparison = next((action.details["execution_gain"] for action in merged.actions
                            if action.action == "compare" and action.details.get("execution_gain")), None)
    split_record = next((record for record in library.evolution.merges
                         if record.get("split_concept_id") and record.get("split_program") == split.program.to_dict()), None)
    split_concept = next((concept for concept in library.concepts if split_record
                          and concept.concept_id == split_record["split_concept_id"]), None)
    old_merge = next((concept for concept in library.concepts if split_record
                      and concept.concept_id == split_record.get("merged_concept_id")), None)
    summary = {"source_fitting_compute": source_work.spent, "source_fitting_budget": source_work.limit,
        "source_observations": sum(item["rows"] for item in (quadratic_source, linear_source, nonlinear_source)),
        "preparation_observations": len(acquired) + agreement.cost + len(counterexamples) + len(sensor_training) + len(transfer_initial),
        "local_repair_adopted": any(action.action == "revise" and action.details["operation"] == "mechanism_repair"
                                    for action in repaired.actions),
        "retained_old_domain_mse": loss(repaired.program, old_rows, 0., .5),
        "repaired_fresh_mse": loss(repaired.program, new_rows, 0., .5),
        "merge_execution_gain": merge_comparison,
        "split_counterexample_mse": loss(split.program, counterexamples, 0., .5),
        "split_proposed": any(action.action == "split_concepts" and action.details["tested"] for action in split.actions),
        "split_adopted": bool(split_record and any(event["operation"] == "split_adopted"
                              for event in split_record.get("comparisons", []))),
        "split_fresh_mse": loss(split.program, split_rows, 0., .5),
        "split_concept_id": split_concept.concept_id if split_concept else None,
        "split_parent_ids": split_concept.parents if split_concept else [],
        "merged_concept_status": old_merge.status if old_merge else None,
        "merged_concept_counterexamples": len(old_merge.counterexamples) if old_merge else 0,
        "split_transfer_called": bool(split_concept and split_concept.concept_id in transfer_details["called_sources"]),
        "split_transfer_fresh_mse": loss(transferred.program, transfer_rows, 0., .5),
        "split_transfer": transfer_details,
        "valid_merge_retained": merge_control.program == merged.program,
        "valid_merge_fresh_mse": loss(merge_control.program, control_rows, 0., .5),
        "measurement_unblocked": any(action.action == "supply_research_measurements" for action in sensor_result.actions),
        "recursive_answer_adopted": any(action.action == "revise" and action.details["operation"] == "research_answer"
                                       for action in sensor_result.actions),
        "recursive_fresh_mse": loss(sensor_result.program, sensor_rows, 0., .5),
        "report_comparison_compute": scored.spent,
        "controller_compute": sum(result.compute_cost for result in (repaired, merged, split, merge_control, transferred, sensor_result)),
        "controller_observations": sum(result.observation_cost for result in (repaired, merged, split, merge_control, transferred, sensor_result))}
    return {"schema": "hegel_machine_evolution_research_v1", "summary": summary,
        "source_models": [quadratic_source, linear_source, nonlinear_source],
        "episodes": {"repair": _episode(repaired), "merge": _episode(merged),
                     "split": _episode(split), "merge_control": _episode(merge_control),
                     "split_transfer": _episode(transferred), "recursive_measurement": _episode(sensor_result)},
        "evolution": library.evolution.to_dict()}
