"""Replay saved GP initialization and compare paid evidence; no simulator calls."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.sc_kg_environment import HegelProgramBasis, SCKGEnvironment

SOURCE = ROOT / "artifacts/sc_kg_optimizer_data_reuse_v030_20260910"
protocol = json.loads((SOURCE / "protocol.json").read_text())
output = {"source": str(SOURCE), "new_simulator_calls": 0,
          "evidence": "Saved initial research outputs and acquisition logs only. No final truth values enter this diagnostic.",
          "cohorts": {}}


def describe(values):
    return {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max())}


def forbidden(*args, **kwargs):
    raise RuntimeError("This diagnostic cannot call the simulator")


for cohort in protocol["source_protocol"]["cohorts"]:
    source = json.loads((SOURCE / f"{cohort}_research.json").read_text())
    runs = {arm: json.loads((SOURCE / f"{cohort}_{arm}.json").read_text())
            for arm in ("learned_with_data", "default_with_data")}
    environment = SCKGEnvironment(protocol["source_protocol"]["sc_kg_project_root"], **source["task"])
    environment._problem.simulate = forbidden
    import numpy as np
    from core.gpr import ParametricGPR
    from problems.single_objective import ScalarizedProblem
    from variance.orthogonal_hvd import OrthogonalHVD
    base = ScalarizedProblem(environment._problem)
    program = Program.from_dict(source["result"]["program"])
    basis = HegelProgramBasis(program, environment.spec.feature_names, base.int_bounds())
    X = [tuple(point) for point in runs["learned_with_data"]["initial_points"]]
    Y = np.asarray(runs["learned_with_data"]["initial_outputs"])
    config = runs["learned_with_data"]["config"]

    def fit(points, values, custom=None):
        model = ParametricGPR(base.d, config["lambda_i"], config["prior_var"],
                              normalize_func=base.normalize, basis_map=custom)
        matrix = model.basis_matrix(points)
        coefficients = np.linalg.lstsq(matrix, values, rcond=None)[0]
        residual = values - matrix @ coefficients
        deviation = max(float(np.var(residual)), 1e-6)
        prior = max(float(np.var(coefficients)), 1e-6)
        # The ordinary SC-KG empirical initialization fits means, then augments
        # the point inventory. It does not condition the covariance on these rows.
        model.set_parametric_prior(coefficients, deviation, prior)
        for point in points:
            model.dimension_augment(point)
        return model, {"basis_dim": model.p, "design_rank": int(np.linalg.matrix_rank(matrix)),
                       "empirical_coefficient_variance": prior, "deviation_variance": deviation}

    objective, _ = fit(X, Y[:, 0])
    observations = {}
    for point, vector in zip(X, Y):
        observations.setdefault(point, []).append(vector)
    arms, models = {}, {}
    for name, custom in (("default", None), ("learned", basis)):
        model, stats = fit(X, Y[:, 1], custom)
        models[name] = model
        mu = model.posterior_mean_many(X)
        stats.update(initial_constraint_mse=float(np.mean((mu - Y[:, 1]) ** 2)),
                     initial_constraint_mae=float(np.mean(abs(mu - Y[:, 1]))),
                     initial_predictive_variance=describe(model.posterior_var_many(X)))
        hvd = OrthogonalHVD(mode="pooled", n_outputs=2, floor=1e-8)
        hvd.initialize(list(observations), observations, [objective, model], base)
        stats["initial_observation_variance"] = hvd.predict_variance(1, X[0], base)
        prefix, prefix_stats = fit(X[:20], Y[:20, 1], custom)
        mu_next = prefix.posterior_mean_many(X[20:])
        stats["paid_last_probe_from_first_20"] = {
            **prefix_stats, "rows": len(X[20:]),
            "mse": float(np.mean((mu_next - Y[20:, 1]) ** 2)),
            "mean_predictive_variance": float(prefix.posterior_var_many(X[20:]).mean()),
            "raw_sign_disagreements": [
                {"observed": float(y), "predicted": float(mu)}
                for y, mu in zip(Y[20:, 1], mu_next) if (mu <= 0) != (y <= 0)]}
        first = runs[name + "_with_data"]["actions"][0]
        stats["first_log_mean_reconstruction_error"] = abs(
            model.posterior_mean(first["x_selected"]) - first["mu_before"][1])
        arms[name] = stats
    differences = [i for i, (left, right) in enumerate(zip(
        runs["learned_with_data"]["actions"], runs["default_with_data"]["actions"]))
        if left["x_selected"] != right["x_selected"]]
    first_difference = differences[0] if differences else None
    compared = models["learned"].posterior_mean_many(X) - models["default"].posterior_mean_many(X)
    row = {"paid_initial_rows": len(X), "hypothesis": program.describe(), "arms": arms,
           "initial_mean_difference": {"rmse": float(np.sqrt(np.mean(compared ** 2))),
                                       "max_absolute": float(np.max(abs(compared)))},
           "different_acquisitions": len(differences),
           "first_different_acquisition_index": first_difference,
           "first_different_acquisition": {name: runs[name + "_with_data"]["actions"][first_difference]
                for name in ("default", "learned")} if differences else None}
    output["cohorts"][cohort] = row

Path(__file__).with_name("diagnostic.json").write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(output, indent=2))
