"""Finite-volume Helmholtz inversion from calibrated polar telemetry."""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _line_laplacian(n):
    """Unit face conductances; exterior faces are handled separately."""
    off = np.ones(n - 1)
    diagonal = np.zeros(n)
    diagonal[:-1] += off
    diagonal[1:] += off
    return diags((-off, diagonal, -off), (-1, 0, 1), format="csc")


def _source_responses(problem, n_sq, basis):
    nx, ny = n_sq.shape
    hx, hy = problem["Lx"] / nx, problem["Ly"] / ny
    area = hx * hy
    k0 = problem["k0"]

    face_length = np.zeros((nx, ny))
    face_length[0, :] += hy
    face_length[-1, :] += hy
    face_length[:, 0] += hx
    face_length[:, -1] += hx
    diagonal = -area * k0**2 * n_sq - 1j * k0 * problem["beta_imp"] * face_length
    operator = (
        (hy / hx) * kron(_line_laplacian(nx), eye(ny), format="csc")
        + (hx / hy) * kron(eye(nx), _line_laplacian(ny), format="csc")
        + diags(diagonal.ravel(), format="csc")
    )
    rhs = area * basis.reshape(basis.shape[0], -1).T
    return splu(operator.tocsc()).solve(rhs.astype(np.complex128))


def _from_polar(telemetry):
    return np.asarray(telemetry["amplitude_obs"]) * np.exp(
        1j * np.asarray(telemetry["wrapped_phase_obs"])
    )


def _robust_real_fit(design, measured, alpha):
    """Fit real coefficients, downweighting each complex channel as a pair."""
    m = design.shape[1]
    target = np.r_[measured.real, measured.imag, np.zeros(m)]
    matrix = np.vstack((design.real, design.imag, np.sqrt(alpha) * np.eye(m)))

    def fit(weights):
        root = np.sqrt(weights)
        scale = np.r_[root, root, np.ones(m)]
        return np.linalg.lstsq(matrix * scale[:, None], target * scale, rcond=None)[0]

    coeff = fit(np.ones(measured.size))
    floor = max(np.finfo(float).tiny,
                1e-12 * np.median(np.abs(measured)))

    # Start with Huber weights to limit the influence of spikes without
    # discarding channels while the residual scale is still settling.
    for _ in range(30):
        residual = np.abs(design @ coeff - measured)
        cutoff = 1.5 * max(floor, np.median(residual))
        updated = fit(cutoff / np.maximum(cutoff, residual))
        if np.linalg.norm(updated - coeff) <= 1e-9 * max(1.0, np.linalg.norm(updated)):
            coeff = updated
            break
        coeff = updated

    # Freeze the robust scale and use a redescending fit so very large
    # amplitude/phase jumps have no leverage on the final coefficients.
    cutoff = 4.0 * max(floor, np.median(np.abs(design @ coeff - measured)))
    for _ in range(25):
        scaled = np.minimum(1.0, np.abs(design @ coeff - measured) / cutoff)
        updated = fit((1.0 - scaled**2)**2)
        if np.linalg.norm(updated - coeff) <= 1e-9 * max(1.0, np.linalg.norm(updated)):
            return updated
        coeff = updated
    return coeff


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    """Estimate oscillator phase, real source amplitudes, and field modulus."""
    n_sq = np.asarray(problem["n_sq"])
    basis = np.asarray(problem["basis"])
    pilot = problem["pilot_calibration"]
    known = np.asarray(pilot["known_field_re"]) + 1j * np.asarray(pilot["known_field_im"])
    phase = float(np.angle(np.vdot(known, _from_polar(pilot))))
    if phase <= -np.pi:
        phase = float(np.pi)

    response = _source_responses(problem, n_sq, basis)
    survey = problem["survey"]
    ij = np.asarray(survey["sensor_ij"])
    rows = np.ravel_multi_index((ij[:, 0], ij[:, 1]), n_sq.shape)
    measured = _from_polar(survey) * np.exp(-1j * phase)
    coeff = _robust_real_fit(response[rows], measured, survey["tikhonov_alpha"])
    return {
        "source_amplitudes": coeff,
        "global_phase_rad": phase,
        "predicted_field_abs": np.abs(response @ coeff).reshape(n_sq.shape),
    }
