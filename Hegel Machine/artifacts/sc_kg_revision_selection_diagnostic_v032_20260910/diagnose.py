"""Compare hypotheses using paid training folds, then inspect held future rows."""

import json
from pathlib import Path
from statistics import mean

from hegel_machine.cognition.continuous_state import _linear_fit, _sum_terms
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.meta_prior_claims import compile_claims
from hegel_machine.cognition.models import SearchBudget, TaskSpec
from hegel_machine.cognition.workspace import observation_from_dict


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/sc_kg_joint_revision_batch_v032_20260910/runs"
SPEC = TaskSpec("paid_scope_diagnostic", "Predict measured responses", ("x1", "x2", "x3"),
                (), (), feature_bounds={f"x{i}": (0., 1.) for i in range(1, 4)})


def spend(budget, count):
    if not budget.spend(count):
        raise RuntimeError("diagnostic selection exhausted its explicit compute budget")


def model_key(claim):
    return f"{claim['kind']}:{len(claim['terms']) + 1}"


def quadratic_fit(terms, rows, budget):
    # Two coordinates, one fit and one in-sample comparison per row, exactly
    # the existing compiler's units. This deliberately does not assume caching.
    spend(budget, (len(terms) + 2) * len(rows))
    coefficients = _linear_fit([[term.predict(row) for row in rows] for term in terms],
                               [row.target for row in rows])
    program = _sum_terms(terms, coefficients[1:], coefficients[0]).simplified()
    training_mse = mean((program.predict(row) - row.target) ** 2 for row in rows)
    return program, training_mse, coefficients


reports = []
for path in sorted(SOURCE.glob("*mechanism_change_noise_0.03_joint_revision.json")):
    data = json.loads(path.read_text())
    recovery = data["recovery"]
    episode = recovery["episodes"][0]
    if episode["proposals"]:
        proposal = episode["proposals"][0]
        train_indices = proposal["training_indices"]
        future_indices = proposal["validation"]["observation_indices"]
        split = "Original proposal training rows and its original four future validation rows"
    else:
        train_indices = list(range(episode["trigger_indices"][0], 28))
        future_indices = list(range(28, 32))
        split = "Late-trigger offline replay: four trigger rows plus next four ordinary paid outcomes; final four rows remain held out. These were not newly chosen coverage controls."
    train = [observation_from_dict(recovery["observations"][i]["observation"]) for i in train_indices]
    future = [observation_from_dict(recovery["observations"][i]["observation"]) for i in future_indices]
    budget = SearchBudget(20000)
    claims = compile_claims(SPEC, train, budget)
    initial_compile = budget.spent
    folds = {}
    for held_index, held in enumerate(train):
        fitted = compile_claims(SPEC, train[:held_index] + train[held_index + 1:], budget)
        for claim in fitted:
            spend(budget, 1)
            error = (claim["candidate"].program.predict(held) - held.target) ** 2
            folds.setdefault(model_key(claim), []).append(error)
    compiler_loo = budget.spent - initial_compile
    candidates = []
    programs = {}
    for claim in claims:
        key = model_key(claim)
        losses = folds.get(key, [])
        programs[key] = claim["candidate"].program
        candidates.append({"model": key, "parameters": len(claim["terms"]) + 1,
                           "program_size": programs[key].size,
                           "training_mse": claim["training_loss"],
                           "loo_mse": mean(losses) if len(losses) == len(train) else None,
                           "complete_folds": len(losses)})
    before_sparse = budget.spent
    for name in SPEC.feature_names:
        coordinate = Program("feature", name=name)
        terms = [coordinate, Program("mul", children=(coordinate, coordinate))]
        program, training_mse, coefficients = quadratic_fit(terms, train, budget)
        losses = []
        for held_index, held in enumerate(train):
            fold, _, _ = quadratic_fit(terms, train[:held_index] + train[held_index + 1:], budget)
            spend(budget, 1)
            losses.append((fold.predict(held) - held.target) ** 2)
        key = f"sparse_quadratic:{name}"
        programs[key] = program
        candidates.append({"model": key, "parameters": 3, "program_size": program.size,
                           "training_mse": training_mse, "loo_mse": mean(losses),
                           "complete_folds": len(losses), "coefficients": coefficients})
    # Freeze the training-only winner before accessing any future target.
    winner = min((item for item in candidates if item["loo_mse"] is not None),
                 key=lambda item: (item["loo_mse"], item["program_size"]))["model"]
    for item in candidates:
        program = programs[item["model"]]
        item["held_future_mse"] = mean((program.predict(row) - row.target) ** 2 for row in future)
    reports.append({"source": str(path.relative_to(ROOT)), "split": split,
                    "training_indices": train_indices, "held_future_indices": future_indices,
                    "old_choice": episode["proposals"][0]["kind"] if episode["proposals"] else None,
                    "training_only_loo_winner": winner,
                    "compute_units": {"original_full_compile": initial_compile,
                        "all_compiler_training_folds_and_predictions": compiler_loo,
                        "three_sparse_quadratic_fits_folds_and_predictions": budget.spent - before_sparse,
                        "total_training_selection": budget.spent,
                        "separate_diagnostic_future_predictions": len(candidates) * len(future)},
                    "candidates": candidates})

report = {"new_simulator_calls": 0,
          "method": "Existing compiler is independently rerun inside every training leave-one-out fold, so sparse coordinate nominations are refitted. Additionally enumerate every public coordinate's univariate quadratic and refit fixed terms inside each fold. Select from training LOO only. Four later observed targets are read only after recording the winner.",
          "limitations": "Development diagnosis on already collected runs, not a new prospective optimizer comparison. The late case reuses later ordinary controls rather than simulating a new coverage policy. LOO has substantial variance with 11-12 training rows; univariate quadratics provide a lower-dimensional competitor but are not universally the best model.",
          "runs": reports}
Path(__file__).with_name("diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
for item in reports:
    choice = next(c for c in item["candidates"] if c["model"] == item["training_only_loo_winner"])
    print(Path(item["source"]).name, "winner", choice["model"],
          "loo", choice["loo_mse"], "future", choice["held_future_mse"],
          "compute", item["compute_units"]["total_training_selection"])
