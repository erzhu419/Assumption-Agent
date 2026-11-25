"""Control-calibrated Hill fits for a 384-well viability screen."""

def __hegel_load_skill_autogenous_prim_ridge_profiled_matching():
    import types as __hegel_types
    import numpy as np
    import numpy as np

    def fit_ridge_nuisance(design: 'np.ndarray', values: 'np.ndarray', *, ridge: 'float | np.ndarray'=1e-06) -> 'tuple[np.ndarray, np.ndarray, np.ndarray]':
        """Fit design (n, p) to values (n,) or (n, q) with scalar or (p,) ridge and return coefficients (p,) or (p, q), same-shape residuals, and an (r, p) Gram whitener."""
        design = np.asarray(design)
        values = np.asarray(values)
        if design.ndim != 2 or values.ndim not in (1, 2) or len(values) != len(design):
            raise ValueError('Design and values must share their observation axis')
        (n, p) = design.shape
        penalty = np.broadcast_to(np.asarray(ridge, dtype=float), (p,))
        if np.any(~np.isfinite(penalty)) or np.any(penalty < 0):
            raise ValueError('Ridge penalties must be finite and nonnegative')
        augmented = np.vstack((design, np.diag(np.sqrt(penalty))))
        (u, s, vh) = np.linalg.svd(augmented, full_matrices=False)
        cutoff = np.finfo(s.dtype).eps * max(augmented.shape) * s.max(initial=0.0)
        keep = s > cutoff
        whitener = vh[keep] / s[keep, None]
        coefficients = whitener.conj().T @ (u[:n, keep].conj().T @ values)
        return (coefficients, values - design @ coefficients, whitener)
    return __hegel_types.SimpleNamespace(fit_ridge_nuisance=fit_ridge_nuisance)
__hegel_skill_ns_autogenous_prim_ridge_profiled_matching = __hegel_load_skill_autogenous_prim_ridge_profiled_matching()

def __hegel_load_skill_autogenous_prim_robust_affine_projection():
    import types as __hegel_types
    import numpy as np
    import numpy as np

    def robust_residual_weights(residual: 'np.ndarray', scale: 'float | np.ndarray', *, loss: 'str'='cauchy') -> 'np.ndarray':
        """For residuals of shape S and a positive scalar or scale array broadcastable to S, return Huber or Cauchy IRLS weights of shape S."""
        r = np.asarray(residual, dtype=float)
        s = np.broadcast_to(np.asarray(scale, dtype=float), r.shape)
        if not np.all(np.isfinite(s) & (s > 0)):
            raise ValueError('scale must be finite and positive')
        with np.errstate(over='ignore'):
            u = np.abs(r) / s
        if loss == 'huber':
            return 1.0 / np.maximum(1.0, u)
        if loss == 'cauchy':
            inverse = 1.0 / np.hypot(1.0, u)
            return inverse * inverse
        raise ValueError('loss must be huber or cauchy')
    return __hegel_types.SimpleNamespace(robust_residual_weights=robust_residual_weights)
__hegel_skill_ns_autogenous_prim_robust_affine_projection = __hegel_load_skill_autogenous_prim_robust_affine_projection()
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import expit
fit_ridge_nuisance = __hegel_skill_ns_autogenous_prim_ridge_profiled_matching.fit_ridge_nuisance
robust_residual_weights = __hegel_skill_ns_autogenous_prim_robust_affine_projection.robust_residual_weights
_LOG10 = np.log(10.0)
_CONTROL_SCALE = 0.025

def _geometry(wells):
    row = np.array([well['row'] for well in wells], dtype=float)
    col = np.array([well['col'] for well in wells], dtype=float)
    depth = np.minimum.reduce([row, 15 - row, col, 23 - col])
    planar = np.column_stack([np.ones_like(row), (row - 7.5) / 7.5, (col - 11.5) / 11.5])
    return (planar, depth)

def _design(planar, depth, width):
    return np.column_stack([planar, np.exp(-depth / width)])

def _fit_gain(planar, depth, signal, width):
    design = _design(planar, depth, width)
    weights = np.ones_like(signal)
    for _ in range(7):
        root = np.sqrt(weights)
        (coef, _, _) = fit_ridge_nuisance(design * root[:, None], signal * root, ridge=1e-06)
        residual = signal - design @ coef
        weights = robust_residual_weights(residual, _CONTROL_SCALE)
    return (np.log1p((residual / _CONTROL_SCALE) ** 2).sum(), coef)

def _shared_width(controls):

    def objective(log_width):
        width = np.exp(log_width)
        return sum((_fit_gain(*block, width)[0] for block in controls))
    fit = minimize_scalar(objective, bounds=(np.log(0.35), np.log(12.0)), method='bounded', options={'xatol': 0.003})
    return np.exp(fit.x)

def _hill(dose, signal, gain):
    levels = np.unique(dose)
    median_viability = np.array([np.median((signal / gain)[dose == x]) for x in levels])
    initial = levels[np.argmin(np.abs(median_viability - 0.5))]

    def residual(params):
        (midpoint, log_slope) = params
        return gain * expit(_LOG10 * np.exp(log_slope) * (midpoint - dose)) - signal
    fit = least_squares(residual, [initial, 0.0], bounds=([dose.min() - 4, -3], [dose.max() + 4, 2]), loss='cauchy', f_scale=0.035)
    (midpoint, log_slope) = fit.x
    return {'log10_ic50_uM': float(midpoint), 'hill_slope': float(np.exp(log_slope))}

def estimate_compound_hill_parameters(plates: list[dict]) -> dict[str, dict[str, float]]:
    """Fit decreasing Hill curves after removing background and spatial gain."""
    (controls, prepared) = ([], [])
    for plate in plates:
        wells = plate['wells']
        kind = np.array([well['well_type'] for well in wells])
        fluorescence = np.array([well['fluorescence'] for well in wells], dtype=float)
        dmso = kind == 'DMSO'
        background = np.median(fluorescence[kind == 'POS_KILL'])
        scale = np.median(fluorescence[dmso]) - background
        signal = (fluorescence - background) / scale
        (planar, depth) = _geometry(wells)
        controls.append((planar[dmso], depth[dmso], signal[dmso]))
        prepared.append((wells, kind, planar, depth, signal))
    if not prepared:
        return {}
    width = _shared_width(controls)
    compounds = {}
    for (block, (wells, kind, planar, depth, signal)) in zip(controls, prepared):
        (_, coef) = _fit_gain(*block, width)
        idx = np.flatnonzero(kind == 'COMPOUND')
        gain = _design(planar[idx], depth[idx], width) @ coef
        for (i, g) in zip(idx, gain):
            well = wells[i]
            compounds.setdefault(well['compound_id'], []).append((well['log10_conc_uM'], signal[i], g))
    return {name: _hill(*np.asarray(values, dtype=float).T) for (name, values) in compounds.items()}
