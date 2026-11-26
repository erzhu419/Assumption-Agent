"""Finite-volume Helmholtz inversion with pilot phase calibration."""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu


def _basis_fields(problem):
    """Solve once per source basis using a shared sparse LU factorization."""
    n_sq = np.asarray(problem['n_sq'])
    basis = np.asarray(problem['basis'])
    nx, ny = n_sq.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    area = hx * hy
    k0 = problem['k0']
    radiation = -1j * k0 * problem['beta_imp']
    cx, cy = hy / hx, hx / hy
    diagonal = np.asarray(2 * cx + 2 * cy - area * k0**2 * n_sq,
                          dtype=np.complex128)
    diagonal[[0, -1], :] += hy * radiation - cx
    diagonal[:, [0, -1]] += hx * radiation - cy

    # In C order x-neighbors are ny apart; y-neighbors never cross a row.
    x_links = np.full((nx - 1) * ny, -cx)
    y_links = np.full(nx * ny - 1, -cy)
    y_links[ny - 1::ny] = 0
    helmholtz = diags([x_links, y_links, diagonal.ravel(), y_links, x_links],
                      [-ny, -1, 0, 1, ny], format='csc')
    rhs = area * basis.reshape(basis.shape[0], nx * ny).T
    return splu(helmholtz).solve(rhs.astype(np.complex128)).reshape(nx, ny, -1)


def _complex_observations(telemetry):
    return telemetry['amplitude_obs'] * np.exp(1j * telemetry['wrapped_phase_obs'])


def _robust_real_fit(design, observed, alpha):
    """Fit real coefficients, assigning a single robust weight per receiver."""
    m = design.shape[1]
    ridge = np.sqrt(alpha) * np.eye(m)

    def solve(weights):
        w = np.sqrt(weights)
        d, z = design * w[:, None], observed * w
        return np.linalg.lstsq(np.vstack((d.real, d.imag, ridge)),
                               np.concatenate((z.real, z.imag, np.zeros(m))),
                               rcond=None)[0]

    coeff = solve(np.ones(len(observed)))
    # Concentrate the starting fit on the smallest residuals to keep large
    # impulses from inflating the initial inlier-scale estimate.
    keep = max(m, int(np.ceil(0.85 * len(observed))))
    for _ in range(8):
        residual = np.abs(design @ coeff - observed)
        weights = np.zeros(len(observed))
        weights[np.argpartition(residual, keep - 1)[:keep]] = 1
        trial = solve(weights)
        if np.linalg.norm(trial - coeff) <= 1e-9 * (1 + np.linalg.norm(coeff)):
            break
        coeff = trial

    # The median modulus of circular complex Gaussian noise is
    # sqrt(2*log(2)) times the standard deviation of either component.
    residual = np.abs(design @ coeff - observed)
    sigma = np.median(residual) / np.sqrt(2 * np.log(2))
    cutoff = 4.685 * max(sigma, np.finfo(float).eps * np.max(np.abs(observed)))
    if cutoff == 0:
        return coeff
    for _ in range(20):
        residual = np.abs(design @ coeff - observed)
        weights = (1 - np.minimum(residual / cutoff, 1)**2)**2
        trial = solve(weights)
        change = np.linalg.norm(trial - coeff)
        coeff = trial
        if change <= 1e-9 * (1 + np.linalg.norm(coeff)):
            break
    return coeff


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    """Return real source strengths, oscillator phase, and field magnitude."""
    pilot = problem['pilot_calibration']
    reference = pilot['known_field_re'] + 1j * pilot['known_field_im']
    phase = float(np.angle(np.vdot(reference, _complex_observations(pilot))))
    if phase <= -np.pi:
        phase = float(np.pi)

    fields = _basis_fields(problem)
    survey = problem['survey']
    i, j = np.asarray(survey['sensor_ij']).T
    observed = _complex_observations(survey) * np.exp(-1j * phase)
    coeff = _robust_real_fit(fields[i, j], observed, survey['tikhonov_alpha'])
    return {'source_amplitudes': coeff, 'global_phase_rad': phase,
            'predicted_field_abs': np.abs(fields @ coeff)}
