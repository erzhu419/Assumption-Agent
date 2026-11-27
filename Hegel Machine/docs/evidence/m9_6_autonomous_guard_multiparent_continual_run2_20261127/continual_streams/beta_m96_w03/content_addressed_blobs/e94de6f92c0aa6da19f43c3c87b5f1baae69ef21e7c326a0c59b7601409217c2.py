"""Calibrated multi-wing tracer inversion with a box-simplex certificate."""
import json
from pathlib import Path

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _build_1d(L, n, kappa, vel, robin, diag_add):
    h = L / n
    kappa, vel = np.asarray(kappa), np.asarray(vel)
    conductance = 2 * kappa[:-1] * kappa[1:] / (kappa[:-1] + kappa[1:]) / h**2
    diagonal = np.array(diag_add, dtype=float, copy=True)
    diagonal[:-1] += conductance
    diagonal[1:] += conductance
    for edge in (0, n - 1):
        k = kappa[edge]
        diagonal[edge] += robin * k / (h * (robin * h / 2 + k))
    positive = np.maximum(vel, 0) / h
    negative = np.maximum(-vel, 0) / h
    diagonal += positive + negative
    return diags((-conductance - positive[1:], diagonal,
                  -conductance - negative[:-1]), (-1, 0, 1), format='csc')


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _bank_sweeps(sweeps, bank):
    selected = np.asarray(sweeps['bank_id']) == bank
    return (np.asarray(sweeps['ref_signal'], dtype=float)[selected],
            np.asarray(sweeps['raw_readout'], dtype=float)[selected])


def _affine_candidates(x, y):
    """Find the largest set of checks consistent with one affine line."""
    if len(x) < 2:
        return []
    i, j = np.triu_indices(len(x), 1)
    spread = x[j] - x[i]
    # Prefix sweeps are reported to six decimals. Near-coincident checks
    # give a poor slope even when both raw readings are uncontaminated.
    pairs = np.flatnonzero(np.abs(spread) > 1e-4)
    tol = 2e-6
    best, candidates = 1, []
    for k in pairs:
        gain = (y[j[k]] - y[i[k]]) / spread[k]
        if gain <= 0:
            continue
        offset = y[i[k]] - gain * x[i[k]]
        good = np.abs(y - (gain * x + offset)) <= tol
        n = int(good.sum())
        if n < best:
            continue
        if n > best:
            candidates, best = [], n
        if not any(np.array_equal(good, mask) for mask in candidates):
            candidates.append(good)
    return [_fit_affine(x[mask], y[mask]) for mask in candidates]


def _fit_affine(x, y):
    centered = x - x.mean()
    gain = centered @ (y - y.mean()) / (centered @ centered)
    return np.array([gain, y.mean() - gain * x.mean()])


def _calibration_coefficients(sweeps):
    checks = [_bank_sweeps(sweeps, bank) for bank in range(4)]
    candidates = [_affine_candidates(x, y) for x, y in checks]
    if all(len(fits) == 1 for fits in candidates):
        return np.array([fits[0] for fits in candidates])

    # Keep the archived episodes separate: averaging their coefficients
    # would mix distinct sensor states. Use history only for unresolved banks.
    history = []
    if _HISTORY.is_file():
        for episode in json.loads(_HISTORY.read_text()):
            fits = [_affine_candidates(*_bank_sweeps(episode, bank)) for bank in range(4)]
            if all(len(f) == 1 for f in fits):
                history.append([f[0] for f in fits])
    if not history:
        history = [np.tile([1., 0.], (4, 1))]
    history = np.asarray(history)
    result = []
    for bank, fits in enumerate(candidates):
        if len(fits) == 1:
            result.append(fits[0])
            continue
        x, y = checks[bank]
        priors = history[:, bank]
        # If this bank has no adequate sweep, prefer matching states from its
        # wing partner, then the other wing when the partner is unresolved.
        partner = bank ^ 1
        peers = [partner] if len(candidates[partner]) == 1 else [
            k for k in range(4) if k != bank and len(candidates[k]) == 1]
        peer_error = np.zeros(len(history))
        for k in peers:
            peer_error += np.sum((history[:, k] - candidates[k][0])**2, axis=1)
        if fits:
            trial = np.asarray(fits)
            distances = np.sum((trial[:, None, :] - priors[None, :, :])**2, axis=2)
            selected = np.argmin(np.min(distances + peer_error[None, :], axis=1))
            result.append(trial[selected])
        elif len(x):
            residuals = y[None, :] - (priors[:, :1] * x[None, :] + priors[:, 1:])
            correction = np.median(residuals, axis=1)
            index = np.argmin(correction**2 + peer_error)
            gain, offset = priors[index]
            result.append([gain, offset + correction[index]])
        else:
            result.append(priors[np.argmin(peer_error)])
    return np.asarray(result)


def _calibrated_observations(study):
    gain, offset = _calibration_coefficients(study['sensor_calibration_sweeps']).T
    rows = np.asarray(study['row_bank_id'])
    return (np.asarray(study['y_obs_raw']) - offset[rows]) / gain[rows]


def _forward(study):
    nx, ny = study['nx'], study['ny']
    reaction = np.asarray(study['V_reaction_2d'], dtype=float)
    vx = reaction.mean(axis=1)
    vy = reaction.mean(axis=0) - reaction.mean()
    interaction = reaction - vx[:, None] - vy[None, :]
    nonseparable = np.linalg.norm(interaction) > 1e-6 * np.linalg.norm(reaction)
    vortex_memory = np.any(np.asarray(study['past_vortex_circulation_trace'])
                          > study['critical_vortex_circulation'])
    diffusivity = 1.45 if nonseparable or vortex_memory else 1.0
    ax = _build_1d(study['Lx'], nx, diffusivity * np.asarray(study['kappa_x']),
                   study['vel_x'], study['robin_x'], study['reaction_x'])
    ay = _build_1d(study['Ly'], ny, diffusivity * np.asarray(study['kappa_y']),
                   study['vel_y'], study['robin_y'], study['reaction_y'])
    base = (kron(ax, eye(ny), format='csc') + kron(eye(nx), ay, format='csc')
            + diags(reaction.ravel(), format='csc'))
    modes = np.asarray(study['source_modes'], dtype=float)
    rhs = modes.reshape(len(modes), nx * ny).T
    sensor = np.asarray(study['sensor_ij'])
    sensor_index = sensor[:, 0] * ny + sensor[:, 1]
    identity = eye(nx * ny, format='csc')
    samples = []
    for shift in study['mu_shifts']:
        terminal = splu(base + shift * identity).solve(rhs)
        samples.append(terminal[sensor_index])
    return np.vstack(samples), terminal


def _box_simplex(S, y, ridge, cap, budget):
    m = S.shape[1]
    H = S.T @ S + ridge * np.eye(m)
    c = S.T @ y
    w = np.full(m, budget / m)
    # -1 and +1 denote fixed lower and upper coordinates, respectively.
    fixed = np.zeros(m, dtype=int)
    if budget == 0:
        return w, float(np.min(-c))
    if budget == m * cap:
        return w, float(np.max(H @ w - c))
    for _ in range(20 * m * m):
        free = fixed == 0
        if not np.any(free):
            g = H @ w - c
            low = np.flatnonzero(fixed < 0)
            high = np.flatnonzero(fixed > 0)
            a = low[np.argmin(g[low])]
            b = high[np.argmax(g[high])]
            if g[a] >= g[b] - 1e-12:
                return w, float((g[a] + g[b]) / 2)
            fixed[[a, b]] = 0
            continue
        active = ~free
        rhs = c[free] - H[np.ix_(free, active)] @ w[active]
        k = np.count_nonzero(free)
        system = np.empty((k + 1, k + 1))
        system[:k, :k] = H[np.ix_(free, free)]
        system[:k, k] = -1
        system[k, :k] = 1
        system[k, k] = 0
        face, nu = np.split(np.linalg.solve(system, np.r_[rhs, budget - w[active].sum()]), [k])
        nu = float(nu[0])
        old = w[free]
        direction = face - old
        below, above = face < -1e-12, face > cap + 1e-12
        if np.any(below | above):
            step = np.full(k, np.inf)
            step[below] = -old[below] / direction[below]
            step[above] = (cap - old[above]) / direction[above]
            hit = int(np.argmin(step))
            w[free] = np.clip(old + step[hit] * direction, 0, cap)
            index = np.flatnonzero(free)[hit]
            fixed[index] = -1 if below[hit] else 1
            w[index] = 0 if below[hit] else cap
            continue
        w[free] = np.clip(face, 0, cap)
        violation = -fixed * (nu - (H @ w - c))
        release = int(np.argmax(violation))
        if violation[release] <= 1e-12:
            return w, nu
        fixed[release] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def _profile(study, samples, observations):
    Z = np.asarray(study['Z_nuisance'])
    data = np.column_stack((samples, observations))
    U, singular, _ = np.linalg.svd(Z, full_matrices=False)
    ridge = study['nuisance_ridge']
    if ridge > 0:
        fraction = singular**2 / (singular**2 + ridge)
    else:
        fraction = singular > np.finfo(float).eps * max(Z.shape) * np.max(singular, initial=0)
    profiled = data - U @ (fraction[:, None] * (U.T @ data))
    return profiled[:, :-1], profiled[:, -1]


def _certificate(S, y, ridge, cap, budget, w, nu):
    residual = S @ w - y
    gradient = S.T @ residual + ridge * w
    objective = float((residual @ residual + ridge * (w @ w)) / 2)
    shifted = gradient - nu
    lower_bound = objective - gradient @ w + nu * budget + cap * np.minimum(shifted, 0).sum()
    return {'weights': w.copy(), 'nu_sum': nu,
            'lam_lower': np.maximum(shifted, 0), 'lam_upper': np.maximum(-shifted, 0),
            'primal_objective': objective, 'dual_lower_bound': min(objective, float(lower_bound))}


def solve_beta_m96_w03_batch_04(study: dict) -> dict:
    samples, terminal = _forward(study)
    S, y = _profile(study, samples, _calibrated_observations(study))
    ridge, cap, budget = (float(study[k]) for k in ('source_ridge', 'u_box', 'total_budget'))
    weights, nu = _box_simplex(S, y, ridge, cap, budget)
    return {'weights': weights,
            'terminal_field': (terminal @ weights).reshape(study['nx'], study['ny']),
            'kkt_certificate': _certificate(S, y, ridge, cap, budget, weights, nu)}
