"""Reuse the finite numerical grammar on each publicly measured coordinate."""

import ast
from copy import deepcopy
import json
from time import monotonic

from .language import Program
from .mechanism_revision import _signed_terms
from .models import Observation, SearchBudget
from .nonlinear_parameters import _coefficient
from .synthesis import synthesize
from .world_reference_hypothesis import _retain_declaration


def _basis(program):
    terms = []
    for signed in _signed_terms(program):
        _, term = _coefficient(signed)
        term = term.simplified()
        if term.op != "const" and term not in terms:
            terms.append(term)
    return sorted(terms, key=lambda term: term.describe())


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
    generated, compute_cost = [], 0
    for feature in pending:
        rows = [Observation({feature: row["measurements"][feature]["value"]},
                            row["frequency"]["value"]) for row in references]
        budget = SearchBudget(20000)
        started = monotonic()
        try:
            candidates = synthesize(rows, operations=["refit", "composition"], budget=budget, max_history=0)
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
            if feature in measurements:
                declaration["measurements"] = {feature: deepcopy(measurements[feature])}
            source = _retain_declaration(source, declaration)
            generated.append(declaration)
            estimators.append(declaration)
            existing_bases.append(basis)
    result = {"operator": "reference_expansion", "path": path,
              "generated_hypotheses": generated, "expanded_features": pending,
              "compute_cost": compute_cost, "status": "untested" if generated else "no_new_basis"}
    if pending:
        source_path.write_text(source, encoding="utf-8")
        agent.files[path] = source
        state["expanded_features"] = sorted(set(state.get("expanded_features", [])) | set(pending))
        state.setdefault("basis_expansions", []).append(deepcopy(result))
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    agent.costs["research_compute_units"] += compute_cost
    return result
