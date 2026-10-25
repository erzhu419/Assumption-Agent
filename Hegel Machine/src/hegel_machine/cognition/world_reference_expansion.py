"""Reuse the finite numerical grammar on each publicly measured coordinate."""

import ast
from copy import deepcopy
import json
from time import monotonic

from .language import Program
from .checkpoint_storage import write_text_incrementally
from .mechanism_revision import _signed_terms
from .models import Observation, SearchBudget
from .nonlinear_parameters import _coefficient
from .synthesis import synthesize
from .world_reference_hypothesis import _retain_declaration
from .world_response_consequences import QUERY_ALLOWANCE
from .world_response_constraints import query_response_family


COMPOSITION_ORIGIN = "finite_grammar_search/paid_counterexamples/seeded_composition"


def _basis(program):
    terms = []
    for signed in _signed_terms(program):
        _, term = _coefficient(signed)
        term = term.simplified()
        if term.op != "const" and term not in terms:
            terms.append(term)
    return sorted(terms, key=lambda term: term.describe())


def pending_reference_compositions(state, estimators):
    """Continue supplied intermediate coordinates once after a real research obligation."""
    response = state.get("response_research", {})
    if (response.get("status") != "family_research_needed"
            or not response.get("family_research", {}).get("research_obligation")):
        return []
    table = state.get("research_table", {})
    references = table.get("references", [])
    rows = references + table.get("targets", [])
    if len(references) < 2 or any("frequency" not in row for row in references):
        return []
    features = set.intersection(*(set(row.get("measurements", {})) for row in rows))
    attempted = {Program.from_dict(term).simplified()
        for result in state.get("basis_expansions", [])
        for attempt in result.get("composition_attempts", []) for term in attempt["seeds"]}
    seeds = {}
    for item in estimators:
        # Generated candidates must be tested, not immediately become an
        # unbounded tower of new polynomial coordinates in this same search.
        if item.get("origin") == COMPOSITION_ORIGIN:
            continue
        for payload in item["basis"]:
            term = Program.from_dict(payload).simplified()
            if not term.history_depth and len(term.required_features()) == 1:
                seeds.setdefault(term, []).append(item["id"])
    groups = []
    for feature in sorted(features):
        selected = [term for term in seeds if term.required_features() == {feature}]
        if any(term.op != "feature" and term not in attempted for term in selected):
            groups.append({"features": [feature], "seeds": [term.to_dict() for term in selected],
                "source_ids": list(dict.fromkeys(source for term in selected for source in seeds[term]))})
    return groups


def execute_reference_expansion(agent, laboratory, arguments):
    """Declare untested bases; coefficient fitting and physical tests stay downstream."""
    path = arguments["path"]
    source_path = agent.workspace / path
    source = source_path.read_text(encoding="utf-8")
    assignment = next(node for node in ast.parse(source).body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
            and target.id == "ESTIMATORS" for target in node.targets))
    estimators = list(ast.literal_eval(assignment.value))
    state_path = agent.workspace / "shared/reference_recalibration.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    table = state.get("research_table", {})
    references = table.get("references", [])
    table_rows = references + table.get("targets", [])
    features = (set.intersection(*(set(row.get("measurements", {})) for row in table_rows))
                if len(references) >= 2 and all("frequency" in row for row in references) else set())
    pending = sorted(features - set(state.get("expanded_features", [])))
    existing_bases = [sorted((Program.from_dict(term).simplified() for term in item["basis"]),
                             key=lambda term: term.describe()) for item in estimators]
    measurements = {feature: measurement for item in estimators
                    for feature, measurement in item.get("measurements", {}).items()}
    attempts = pending_reference_compositions(state, estimators) if not pending else []
    groups = [{"features": [feature], "seeds": []} for feature in pending] or attempts
    generated, checks, compute_cost, check_cost = [], {}, 0, 0
    for group in groups:
        feature = group["features"][0]
        rows = [Observation({feature: row["measurements"][feature]["value"]},
                            row["frequency"]["value"]) for row in references]
        budget = SearchBudget(20000)
        started = monotonic()
        try:
            candidates = synthesize(rows, operations=["refit", "composition"], budget=budget, max_history=0,
                **({"seed_programs": [Program.from_dict(term) for term in group["seeds"]]} if attempts else {}))
        finally:
            agent.costs["python_execution_seconds"] += monotonic() - started
        compute_cost += budget.spent
        for candidate in candidates:
            basis = _basis(candidate.program)
            if not basis or basis in existing_bases:
                continue
            serial = 1
            while any(item["id"] == "grammar_" + feature + "_" + str(serial) for item in estimators):
                serial += 1
            declaration = {"id": "grammar_" + feature + "_" + str(serial),
                "basis": [term.to_dict() for term in basis],
                "origin": "finite_grammar_search/public_calibration/old_refit_composition",
                "rationale": candidate.rationale}
            if attempts:
                declaration.update(origin=COMPOSITION_ORIGIN, source_ids=group["source_ids"])
            if feature in measurements:
                declaration["measurements"] = {feature: deepcopy(measurements[feature])}
            source = _retain_declaration(source, declaration)
            generated.append(declaration)
            estimators.append(declaration)
            existing_bases.append(basis)
            if attempts:
                # Activation labels remain inequalities. Only public activated
                # reference settings above supply the old grammar's fit labels.
                paid_features = {row["crystal"]: {name: value["value"]
                    for name, value in row["measurements"].items()} for row in table_rows}
                responses = state.get("response_settings", [])
                started = monotonic()
                try:
                    check = query_response_family(paid_features, responses, declaration,
                        queries=(), branch_limit=None, compute_limit=QUERY_ALLOWANCE)
                finally:
                    agent.costs["python_execution_seconds"] += monotonic() - started
                check_cost += check["compute_cost"]
                check.update(features={str(key): value for key, value in paid_features.items()},
                    responses=deepcopy(responses), evidence=sorted(
                        {index for row in responses for index in row.get("evidence", [])}
                        | {row["measurements"][feature]["evidence"] for row in table_rows}))
                checks[declaration["id"]] = check
                state.setdefault("family_history_checks", {})[declaration["id"]] = deepcopy(check)
    result = {"operator": "reference_expansion", "path": path,
              "generated_hypotheses": generated, "expanded_features": pending,
              "compute_cost": compute_cost + check_cost, "status": "untested" if generated else "no_new_basis"}
    if attempts:
        result.update(composition_attempts=attempts, paid_family_checks=checks)
        state["response_research"].pop("basis_expansion_action", None)
    if pending or attempts:
        source_path.write_text(source, encoding="utf-8")
        agent.files[path] = source
        state["expanded_features"] = sorted(set(state.get("expanded_features", [])) | set(pending))
        state.setdefault("basis_expansions", []).append(deepcopy(result))
        write_text_incrementally(state_path, json.dumps(state, indent=2))
    agent.costs["research_compute_units"] += compute_cost + check_cost
    return result
