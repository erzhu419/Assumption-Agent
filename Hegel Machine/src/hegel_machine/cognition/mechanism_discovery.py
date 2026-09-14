"""Discover and revise executable structure from a finite vector apparatus.

Matrix responses, held-parent interventions and new trajectories are separate
evidence. An adequate response predictor does not by itself establish a low
rank quotient, independent modules, stable evolution or persistent topology.
"""

from dataclasses import replace
from math import atan2, cos, sin, sqrt
from time import perf_counter

from .continuous_state import _linear_fit, _sum_terms
from .language import Definition, EvaluationContext, Program
from .memory import KnowledgeLibrary
from .models import Candidate, EpisodeResult, Observation, Probe, ResearchConfig, SearchBudget
from .workspace import ResearchWorkspace


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def _basis(rows, tolerance=1e-7):
    result = []
    for row in rows:
        vector = list(row)
        for direction in result:
            weight = _dot(vector, direction)
            vector = [a - weight * b for a, b in zip(vector, direction)]
        norm = sqrt(_dot(vector, vector))
        if norm > tolerance:
            result.append([value / norm for value in vector])
    return result


def _complement(basis):
    unit = [[float(i == j) for i in range(4)] for j in range(4)]
    return _basis([*basis, *unit])[len(basis):]


def _product(left, right):
    return [[_dot(row, column) for column in zip(*right)] for row in left]


def _power(matrix, exponent):
    value = [[float(i == j) for j in range(4)] for i in range(4)]
    for _ in range(exponent):
        value = _product(value, matrix)
    return value


def _eigen(matrix):
    """Jacobi rotations for the measured four-dimensional symmetric case."""
    value = [list(row) for row in matrix]
    vectors = [[float(i == j) for j in range(4)] for i in range(4)]
    for _ in range(48):
        i, j = max(((i, j) for i in range(4) for j in range(i + 1, 4)),
                   key=lambda pair: abs(value[pair[0]][pair[1]]))
        if abs(value[i][j]) < 1e-12:
            break
        angle = .5 * atan2(2 * value[i][j], value[j][j] - value[i][i])
        c, s = cos(angle), sin(angle)
        rotation = [[float(a == b) for b in range(4)] for a in range(4)]
        rotation[i][i] = rotation[j][j] = c
        rotation[i][j], rotation[j][i] = s, -s
        value = _product(_product(list(map(list, zip(*rotation))), value), rotation)
        vectors = _product(vectors, rotation)
    pairs = [(value[i][i], [vectors[j][i] for j in range(4)]) for i in range(4)]
    return sorted(pairs, key=lambda pair: -abs(pair[0]))


def _components(matrix):
    """Connected input/output mechanism blocks from measured nonzero effects."""
    edges = {(i, j + 4) for i, row in enumerate(matrix) for j, value in enumerate(row) if abs(value) > 1e-7}
    unseen, result = set(range(8)), []
    while unseen:
        found = {min(unseen)}
        while True:
            expanded = found | {a for a, b in edges if b in found} | {b for a, b in edges if a in found}
            if expanded == found:
                break
            found = expanded
        unseen -= found
        result.append({"outputs": sorted(node for node in found if node < 4),
                       "inputs": sorted(node - 4 for node in found if node >= 4)})
    return result


def _graph(matrix):
    edges = [[j, i] for i, row in enumerate(matrix) for j, value in enumerate(row) if abs(value) > 1e-7]
    reachable = [[i == j for j in range(4)] for i in range(4)]
    for source, target in edges:
        reachable[source][target] = True
    for via in range(4):
        for source in range(4):
            for target in range(4):
                reachable[source][target] |= reachable[source][via] and reachable[via][target]
    return {"edges": edges, "reachability": reachable,
            "cycles": [i for i in range(4) if any(source == i and reachable[target][i]
                                                   for source, target in edges)]}


def _call(program, source_id, name, inputs):
    def bind(node):
        if node.op == "feature" and node.name in inputs:
            return Program("arg", name=node.name)
        return Program(node.op, node.value, node.name, node.lag,
                       tuple(bind(child) for child in node.children), node.definition)
    definition = Definition(name, tuple(inputs), bind(program), source_id)
    return Program("call", name=name, definition=definition,
                   children=tuple(Program("feature", name=key) for key in inputs))


class _Unfunded(Exception):
    pass


def run_mechanism_research(laboratory, config=None, library=None, measurement_budget=4096):
    """Learn, experimentally challenge, execute and remember one mechanism.

    The returned workspace is also accepted by the ordinary CognitiveAgent.
    All fitting, structural comparisons and program applications draw from the
    supplied compute budget. No incomplete evidence is promoted to a prior.
    """
    if laboratory.family == "order":
        from .response_profile import run_profile_research
        return run_profile_research(laboratory, config=config, library=library,
                                    measurement_budget=measurement_budget)
    config = config or ResearchConfig(observation_budget=96, compute_budget=24000)
    library = library if library is not None else KnowledgeLibrary()
    started, initial = perf_counter(), laboratory.costs
    budget = SearchBudget(config.compute_budget)
    rows, batches, probes, experiments, assessments = [], [], [], [], []
    discovery, prior, episode = {}, None, None
    applied_sources = set()
    names = laboratory.feature_names
    features = [Program("feature", name=name) for name in names]

    def pay(units):
        if not budget.spend(units):
            raise _Unfunded

    def acquire(probe):
        spent = laboratory.costs
        if (spent["observation_cost"] - initial["observation_cost"] + probe.cost > config.observation_budget
                or spent["measurement_cost"] - initial["measurement_cost"]
                + probe.cost * laboratory.measurement_cost_per_row > measurement_budget):
            raise _Unfunded
        batch = laboratory.acquire(probe)
        rows.extend(batch.observations)
        batches.append(batch)
        probes.append(probe)
        return batch

    def predict(program, observations):
        # Program size is a complete upper bound for these finite expressions;
        # repeated callable bodies can share results in the evaluation context.
        # Lexical argument reads and calls add at most another two traversals
        # for the one-layer component calls constructed by this researcher.
        quote = 3 * program.expanded().size * len(observations)
        if quote > budget.remaining:
            raise _Unfunded
        predicted = []
        for row in observations:
            context = EvaluationContext(row)
            predicted.append(context.evaluate(program))
            pay(context.primitive_steps + context.call_evaluations)
            applied_sources.update(definition.source_concept_id for definition, _ in context._calls
                                   if definition.source_concept_id is not None)
        return predicted

    try:
        training = acquire(laboratory.random_probe(config.seed, 12, "mechanism-fit"))
        pay(12 * (4 * 5 * 5 + 4 * 5) + 4 * 5 ** 3)
        columns = [[row.features[name] for row in training.observations] for name in names]
        coefficients = [_linear_fit(columns, [values[i] for values in training.responses]) for i in range(4)]
        matrix = [values[1:] for values in coefficients]
        intercepts = [values[0] for values in coefficients]
        programs = [_sum_terms(features, row, intercept) for row, intercept in zip(matrix, intercepts)]
        basis = _basis(matrix)
        pay(4 ** 3 * 4)
        discovery = {"response_matrix": matrix, "intercepts": intercepts,
                     "rank": len(basis), "orthogonal_projection": basis,
                     "modules": _components(matrix), "graph": _graph(matrix),
                     "raw_dimension": 4, "effective_dimension": len(basis),
                     "hypotheses": [], "component_programs": [program.to_dict() for program in programs]}
        sources = [record for record in library.structural_priors
                   if record.get("kind") == "mechanism" and record["family"] == laboratory.family]
        source = sources[-1] if sources else None
        retained, source_ids = [], [None] * 4
        if source:
            for i, payload in enumerate(source["component_programs"]):
                observed = [Observation(row.features, values[i]) for row, values in
                            zip(training.observations, training.responses)]
                old = Program.from_dict(payload)
                errors = [(a - row.target) ** 2 for a, row in zip(predict(old, observed), observed)]
                keep = max(errors) < 1e-12
                if keep:
                    programs[i], source_ids[i] = old, source["component_concept_ids"][i]
                    retained.append(i)
                assessments.append({"record_id": source["record_id"], "component": i,
                    "status": "retained" if keep else "revised", "loss": sum(errors) / len(errors),
                    "retained_source_support": source["support_rows"],
                    "counterexample": None if keep else {"features": observed[errors.index(max(errors))].features,
                        "observed": observed[errors.index(max(errors))].target,
                        "prediction": old.predict(observed[errors.index(max(errors))])}})

        # Each proposal chooses a concrete experiment from its learned geometry.
        challenge = []
        proposed = source["discovery"] if source else discovery
        if laboratory.family == "low_rank":
            quotient = proposed["orthogonal_projection"]
            directions = _complement(quotient)
            challenge = [laboratory.point([sign * value for value in direction])
                         for direction in directions for sign in (-.8, .8)]
            proposal = {"kind": "sufficient_projection", "dimension": len(quotient),
                        "projection": quotient, "tested_directions": directions}
        elif laboratory.family == "modules":
            # Hold each learned parent set fixed; intervene on every omitted
            # input. Prior parent sets are challenged before retaining modules.
            parent_matrix = proposed["response_matrix"]
            for j in range(4):
                challenge.extend(laboratory.point([(.75 * sign if k == j else .2) for k in range(4)])
                                 for sign in (-1, 1))
            proposal = {"kind": "conditional_mechanisms", "parents":
                        [[j for j, value in enumerate(row) if abs(value) > 1e-7] for row in parent_matrix]}
        elif laboratory.family == "dynamics":
            pay(48 * 4 ** 3 * 2)
            modes = _eigen(matrix)
            discovery["modes"] = [{"rate": rate, "direction": direction} for rate, direction in modes]
            challenge = [laboratory.point([sign * value for value in direction])
                         for _, direction in modes for sign in (-.8, .8)]
            proposal = {"kind": "contraction_and_scale", "modes": discovery["modes"]}
        else:
            challenge = [laboratory.point([float(i == j) for i in range(4)], gain=gain)
                         for gain in (.85, 1.15) for j in range(4)]
            proposal = {"kind": "topological_persistence", "graph": proposed["graph"]}
        experiment = acquire(Probe("mechanism:structural-intervention", tuple(challenge)))
        pay(len(challenge) * 4 ** 2 * laboratory.horizon)
        if laboratory.family == "low_rank":
            response = max(abs(value) for values in experiment.responses for value in values)
            proposal.update(status="supported" if response < 1e-7 else "refuted", max_response=response,
                counterexample=None if response < 1e-7 else experiment.to_dict()["observations"][0])
            # The sufficient representation expands when an old null direction
            # acquires an observed effect; the full observed row space is kept.
            discovery["reexpanded"] = len(basis) > len(proposal["projection"])
            projections = [_sum_terms(features, direction) for direction in basis]
            programs = [_sum_terms(projections, [_dot(row, direction) for direction in basis], intercept)
                        if i not in retained else programs[i]
                        for i, (row, intercept) in enumerate(zip(matrix, intercepts))]
        elif laboratory.family == "modules":
            violations = []
            for j in range(4):
                difference = [b - a for a, b in zip(experiment.responses[2*j], experiment.responses[2*j+1])]
                for i, value in enumerate(difference):
                    if j not in proposal["parents"][i] and abs(value) > 1e-7:
                        violations.append({"output": i, "outside_input": j, "response_change": value})
            proposal.update(status="refuted" if violations else "supported", counterexamples=violations,
                            independent_blocks=len(discovery["modules"]), retained_components=retained)
        elif laboratory.family == "dynamics":
            ratios = []
            for i in range(0, len(experiment.trajectories), 2):
                left, right = experiment.trajectories[i:i+2]
                distance = [sqrt(sum((a-b)**2 for a, b in zip(x, y))) for x, y in zip(left, right)]
                ratios.extend(b / a for a, b in zip(distance, distance[1:]) if a > 1e-12)
            rate = max(ratios)
            gaps = [abs(modes[i][0]) / max(abs(modes[i+1][0]), 1e-12) for i in range(3)]
            split = max(range(3), key=gaps.__getitem__) + 1
            slow = modes[:split] if max(gaps) >= 3 else modes
            coarse = _sum_terms([_sum_terms(features, direction) for _, direction in slow],
                [direction[laboratory.target_index] * rate ** laboratory.horizon for rate, direction in slow])
            discovery.update(slow_dimension=len(slow), coarse_program=coarse.to_dict())
            proposal.update(status="supported" if rate < 1. else "refuted", maximum_distance_ratio=rate,
                            slow_dimension=len(slow), spectral_gap=max(gaps),
                            counterexample=None if rate < 1. else {"trajectory_pair": experiment.trajectories[:2]})
        else:
            graphs = [_graph([[experiment.responses[offset+j][i] for j in range(4)] for i in range(4)])
                      for offset in (0, 4)]
            matches = all(graph == proposal["graph"] for graph in graphs)
            proposal.update(status="supported" if matches else "refuted", perturbed_graphs=graphs,
                target_reachable=graphs[0]["reachability"][0][laboratory.target_index],
                counterexample=None if matches else {"previous": proposal["graph"], "observed": graphs[0]})
        experiments.append(proposal)
        discovery["hypotheses"].append(proposal)

        validation = acquire(laboratory.random_probe(config.seed + 7919, 12, "mechanism-prospective"))
        all_responses = [values for batch in batches for values in batch.responses]
        component_errors = []
        for i, program in enumerate(programs):
            relevant = [Observation(row.features, response[i] / row.features.get("gain", 1.))
                        for row, response in zip(rows, all_responses)]
            values = predict(program, relevant)
            component_errors.append(max((value - row.target) ** 2 for value, row in zip(values, relevant)))
        discovery["component_max_errors"] = component_errors
        if max(component_errors) > 1e-10:
            discovery["unresolved"] = "response model contradicted by intervention or new data"
            raise _Unfunded
        # Retained components become actual calls in the successor expression.
        executable = [_call(program, source_ids[i], f"{source_ids[i]}:mechanism:{i}", names)
                      if source_ids[i] else program for i, program in enumerate(programs)]
        if laboratory.horizon == 1:
            program = executable[laboratory.target_index]
        else:
            pay(laboratory.horizon * 4 ** 3)
            weights = _power(matrix, laboratory.horizon - 1)[laboratory.target_index]
            program = _sum_terms(executable, weights)
            if laboratory.family == "topology":
                for _ in range(laboratory.horizon):
                    program = Program("mul", children=(program, Program("feature", name="gain")))
        predictions = predict(program, validation.observations)
        error = sum((value - row.target) ** 2 for value, row in zip(predictions, validation.observations)) / len(predictions)
        if error > 1e-10:
            discovery["unresolved"] = "composed mechanism failed prospective task observations"
            raise _Unfunded
        if "coarse_program" in discovery:
            coarse_values = predict(Program.from_dict(discovery["coarse_program"]), validation.observations)
            discovery["coarse_validation_loss"] = sum((a-row.target)**2 for a, row in
                zip(coarse_values, validation.observations)) / len(coarse_values)
        discovery["triangulation"] = {"channels": ["response_fit", "targeted_intervention", "new_trajectory"],
            "component_agreement": True, "prospective_loss": error,
            "structural_claim_status": proposal["status"]}
        discovery["retained_components"] = retained
        discovery["applied_concept_ids"] = sorted(applied_sources)
        concepts = []
        for i, item in enumerate(programs):
            relevant = [Observation(row.features, response[i] / row.features.get("gain", 1.))
                        for row, response in zip(rows, all_responses)]
            if source_ids[i] is None:
                source_ids[i], created = library.learn(Candidate(f"mechanism:{i}", item, "mechanism_discovery"),
                    replace(laboratory.spec, task_id=f"{laboratory.spec.task_id}:output:{i}"), relevant)
                if created:
                    concepts.append(source_ids[i])
            else:
                component_spec = replace(laboratory.spec, task_id=f"{laboratory.spec.task_id}:output:{i}")
                library.record_use(source_ids[i], component_spec.task_id, spec=component_spec, observations=relevant)
        operation = "mechanism_reuse" if len(retained) == 4 else "local_mechanism_revision" if retained else "mechanism_discovery"
        candidate = Candidate("mechanism:task", program, operation, parents=tuple(source_ids),
                              memory_id=source_ids[laboratory.target_index] if retained else None)
        target_id, created = library.learn(candidate, laboratory.spec, rows)
        if created:
            concepts.append(target_id)
        bank = [candidate]
        if "coarse_program" in discovery:
            bank.append(Candidate("mechanism:slow-variables", Program.from_dict(discovery["coarse_program"]),
                                  "scale_selection", "Measured slower coordinates predict three-step evolution"))
        workspace = ResearchWorkspace(laboratory.spec.task_id, observations=rows,
            reference=validation.observations, queried=probes, incumbent=candidate, evidence_champion=candidate,
            bank=bank, fallback=0., initial_loss=sum(row.target**2 for row in rows) / len(rows))
        prior = {"kind": "mechanism", "record_id": f"mechanism-prior:{len(library.structural_priors)}",
                 "family": laboratory.family, "discovery": discovery, "program": program.to_dict(),
                 "component_programs": [item.to_dict() for item in programs],
                 "component_concept_ids": source_ids, "support_rows": len(rows),
                 "revises": [source["record_id"]] if source and len(retained) < 4 else []}
        library.structural_priors.append(prior)
        episode = EpisodeResult(laboratory.spec.task_id, program, 0., rows, [], [],
            len(rows), budget.spent, error, workspace.initial_loss, operation, candidate.memory_id, concepts,
            [{"kind": "component_reuse", "component": i, "concept_id": source_ids[i]}
             for i in retained if source_ids[i] in applied_sources],
            "mechanism_prospectively_applied", validation_rows=len(validation.observations),
            research_state=workspace.to_dict()).to_dict()
    except _Unfunded:
        workspace = ResearchWorkspace(laboratory.spec.task_id, observations=rows, queried=probes,
            experiment_frontier={"kind": "mechanism", "reason": discovery.get("unresolved", "additional funded evidence required")})
        episode = {"task_id": laboratory.spec.task_id, "program": Program("const", value=0.).to_dict(),
                   "research_state": workspace.to_dict(), "validation_loss": None, "validation_rows": 0,
                   "concepts_added": [], "stop_reason": workspace.experiment_frontier["reason"]}
    costs = laboratory.costs
    return {"task_id": laboratory.spec.task_id, "status": "completed" if prior else "unresolved",
            "episode": episode, "discovery": discovery, "experiments": experiments,
            "prior_assessments": assessments, "structural_prior": prior,
            "costs": {"observation_cost": costs["observation_cost"] - initial["observation_cost"],
                      "measurement_cost": costs["measurement_cost"] - initial["measurement_cost"],
                      "compute_cost": budget.spent, "wall_time_seconds": perf_counter() - started}}
