"""Finite-volume Helmholtz inversion with a shared pilot phase reference."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _responses(problem):
    n_sq = np.asarray(problem['n_sq'])
    nx, ny = n_sq.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    k = problem['k0']
    impedance = -1j * k * problem['beta_imp']

    def axis_operator(size, coupling, wall):
        graph_diag = np.full(size, 2 * coupling, dtype=complex)
        graph_diag[[0, -1]] += wall - coupling
        return sparse.diags((-coupling, graph_diag, -coupling), (-1, 0, 1),
                            shape=(size, size), format='csc')

    # Each interior face enters both cell balances, and each corner has two
    # separate impedance faces. The real stiffness is not positive definite.
    operator = (sparse.kron(axis_operator(nx, hy / hx, hy * impedance), sparse.eye(ny), format='csc')
                + sparse.kron(sparse.eye(nx), axis_operator(ny, hx / hy, hx * impedance), format='csc')
                - sparse.diags(hx * hy * k**2 * n_sq.ravel(), format='csc'))
    basis = np.asarray(problem['basis'])
    rhs = hx * hy * basis.reshape(basis.shape[0], nx * ny).T
    return splu(operator).solve(rhs.astype(np.complex128)).reshape(nx, ny, -1)


def _phasors(telemetry):
    return np.asarray(telemetry['amplitude_obs']) * np.exp(1j * np.asarray(telemetry['wrapped_phase_obs']))


def _robust_sources(design, observed, alpha):
    # Treat each complex observation as one sensor: its two quadratures
    # receive the same robust weight and the source coefficients stay real.
    gram = np.einsum('ni,nj->nij', design.conj(), design).real
    rhs = (design.conj() * observed[:, None]).real
    penalty = alpha * np.eye(design.shape[1])
    a = np.linalg.solve(gram.sum(axis=0) + penalty, rhs.sum(axis=0))
    floor = np.finfo(float).eps * np.max(np.abs(observed))
    for _ in range(40):
        residual = np.abs(design @ a - observed)
        cutoff = max(2.5 * np.median(residual), floor)
        if cutoff == 0:
            break
        weights = 1 / (1 + (residual / cutoff)**2)
        new = np.linalg.solve(np.einsum('n,nij->ij', weights, gram) + penalty, weights @ rhs)
        if np.linalg.norm(new - a) <= 1e-10 * (1 + np.linalg.norm(new)):
            return new
        a = new
    return a


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    """Recover real source amplitudes, oscillator phase and pressure modulus."""
    responses = _responses(problem)
    pilot = problem['pilot_calibration']
    known = np.asarray(pilot['known_field_re']) + 1j * np.asarray(pilot['known_field_im'])
    phase = float(np.angle(np.vdot(known, _phasors(pilot))))
    if phase <= -np.pi:
        phase = float(np.pi)
    survey = problem['survey']
    i, j = np.asarray(survey['sensor_ij']).T
    a = _robust_sources(responses[i, j], _phasors(survey) * np.exp(-1j * phase), survey['tikhonov_alpha'])
    return {'source_amplitudes': a, 'global_phase_rad': phase,
            'predicted_field_abs': np.abs(responses @ a)}
