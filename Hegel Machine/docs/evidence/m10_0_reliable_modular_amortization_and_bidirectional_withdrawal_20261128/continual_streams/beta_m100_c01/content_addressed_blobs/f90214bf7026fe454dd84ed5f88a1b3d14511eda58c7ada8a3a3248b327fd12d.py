"""Calibrate the paired rosette readouts before the canonical PDE solve."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


_PARTNER = np.array([2, 3, 0, 1])


def _design(ref):
    partner = ref[:, _PARTNER]
    return np.stack((np.ones_like(ref), ref, np.sin(2.5 * ref),
                     partner, np.sin(partner)**2), axis=-1)


def _fit(ref, raw):
    # EMI corrupts all four readings of a sweep; reject at the row level.
    center = np.median(raw, axis=0)
    spread = np.median(np.abs(raw - center), axis=0)
    clean = np.all(np.abs(raw - center) <= 10 * np.maximum(spread, 1e-12), axis=1)
    design = _design(ref)

    def fit(mask):
        return np.stack([np.linalg.lstsq(design[mask, j], raw[mask, j], rcond=None)[0]
                         for j in range(4)])

    coef = fit(clean)
    for _ in range(3):
        residual = np.abs(np.einsum('njk,jk->nj', design, coef) - raw)
        scale = np.median(residual[clean], axis=0)
        refined = clean & np.all(residual <= 10 * np.maximum(scale, 1e-12), axis=1)
        if refined.sum() < 5 or np.array_equal(refined, clean):
            break
        clean = refined
        coef = fit(clean)
    return coef


def _inverse(raw, coef):
    raw = np.asarray(raw, dtype=float)
    offset, slope, sine, cross, even = coef.T
    diag = np.arange(4)

    def forward(x):
        partner = x[..., _PARTNER]
        return offset + slope*x + sine*np.sin(2.5*x) + cross*partner + even*np.sin(partner)**2

    def jacobian(x):
        jac = np.zeros(x.shape[:-1] + (4, 4))
        jac[..., diag, diag] = slope + 2.5*sine*np.cos(2.5*x)
        jac[..., diag, _PARTNER] = cross + even*np.sin(2*x[..., _PARTNER])
        return jac

    target = raw.reshape(-1, 4)
    start = np.linalg.solve(jacobian(np.zeros(4)), (target - offset).T).T
    estimate = start.copy()
    # Vectorized Newton on the central branch, with a trust-region solve for
    # rows for which a full Newton step does not improve the residual.
    active = np.arange(len(target))
    for _ in range(12):
        x = estimate[active]
        residual = forward(x) - target[active]
        pending = np.max(np.abs(residual), axis=1) > 1e-12
        active, x, residual = active[pending], x[pending], residual[pending]
        if not len(active):
            break
        jac = jacobian(x)
        regular = np.abs(np.linalg.det(jac)) > 1e-10
        active, x, residual, jac = active[regular], x[regular], residual[regular], jac[regular]
        step = np.linalg.solve(jac, residual[..., None])[..., 0]
        proposal = x - step
        improves = (np.linalg.norm(forward(proposal) - target[active], axis=1)
                    < np.linalg.norm(residual, axis=1))
        active = active[improves]
        estimate[active] = proposal[improves]
    for i in np.flatnonzero(np.max(np.abs(forward(estimate) - target), axis=1) > 1e-10):
        estimate[i] = least_squares(lambda x: forward(x) - target[i], start[i],
                                    jac=jacobian, gtol=1e-13, xtol=1e-13, ftol=1e-13).x
    return estimate.reshape(raw.shape)


def calibrate_upstream(study: dict) -> dict:
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    coef = _fit(ref, np.asarray(sweeps['raw_rosette'], dtype=float))
    result = dict(study, y_obs_raw=_inverse(study['y_obs_raw'], coef))
    if study.get('reference_tracer_pulses') is not None:
        result['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_inverse(pulse['y_pulse_raw'], coef))
            if pulse.get('y_pulse_raw') is not None else dict(pulse)
            for pulse in study['reference_tracer_pulses']
        ]
    banks = sweeps['single_bank_sweeps']
    result['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=ref.copy(),
        single_bank_sweeps=dict(banks, raw_readout=np.asarray(banks['ref_signal']).copy()),
    )
    return result
