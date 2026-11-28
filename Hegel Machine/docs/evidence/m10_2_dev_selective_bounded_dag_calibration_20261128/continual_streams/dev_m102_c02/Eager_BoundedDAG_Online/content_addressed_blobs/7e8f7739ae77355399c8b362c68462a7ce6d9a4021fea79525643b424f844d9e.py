"""Project affine trajectory coefficients onto two shared relaxation modes."""
import numpy as np
from scipy.optimize import least_squares


def run_node(inputs: dict) -> dict:
    upstream = inputs['node_01']
    beta = np.asarray(upstream['beta'], dtype=float)
    query = np.asarray(upstream['new_design'], dtype=float)
    t = np.asarray(upstream['t'], dtype=float)
    if t.size <= 2 or np.any(t < 0):
        return {'state_pred': query @ beta}

    def project(log_rates):
        basis = -np.expm1(-t[:, None] * np.exp(log_rates)[None, :])
        return (basis @ np.linalg.lstsq(basis, beta.T, rcond=None)[0]).T

    def residual(log_rates):
        return (project(log_rates) - beta).ravel()

    best = min((least_squares(residual, np.log(start), bounds=(np.log(.005), np.log(100.0)),
                              max_nfev=100)
                for start in ((.5, 2.), (1., 5.), (2., 12.))), key=lambda fit: fit.cost)
    _, _, right = np.linalg.svd(beta, full_matrices=False)
    modes = right[:2]
    unrestricted = (beta @ modes.T) @ modes
    physical = project(best.x)
    # Retain the data-driven basis if the two-mode relaxation law does not fit.
    if np.linalg.norm(beta - physical) > 0.004 * np.linalg.norm(beta) + 2 * np.linalg.norm(beta - unrestricted):
        physical = unrestricted
    return {'state_pred': query @ physical}
