"""Estimate per-compound Hill parameters from multi-plate viability assays."""
import numpy as np
from scipy.optimize import least_squares


def _surface(row, col):
    r = np.asarray(row, dtype=float) / 15.0 * 2.0 - 1.0
    c = np.asarray(col, dtype=float) / 23.0 * 2.0 - 1.0
    return np.column_stack((np.ones(r.size), r, c, r*r, c*c, r*c,
                            r**4, c**4, r**3*c, r*c**3, r*r*c*c))


def estimate_compound_hill_parameters(plates: list[dict]) -> dict[str, dict[str, float]]:
    """Fit shared compound Hill curves while estimating plate-specific assay tops.

    DMSO control wells identify each plate's smooth spatial multiplicative gain;
    POS_KILL wells set its additive background. Robust fitting limits pipetting
    outliers in both the control surface and compound dose-response data.
    """
    prepared = []
    compounds = set()
    for plate in plates:
        wells = plate['wells']
        dmso = [w for w in wells if w.get('well_type') == 'DMSO']
        pos = [w for w in wells if w.get('well_type') == 'POS_KILL']
        if len(dmso) < 12 or len(pos) < 3:
            continue
        bg = float(np.median([w['fluorescence'] for w in pos]))
        A = _surface([w['row'] for w in dmso], [w['col'] for w in dmso])
        y = np.asarray([w['fluorescence'] - bg for w in dmso], float)
        keep = np.isfinite(y)
        coef = np.linalg.lstsq(A[keep], y[keep], rcond=None)[0]
        for _ in range(6):
            resid = y - A @ coef
            med = np.median(resid[keep])
            mad = 1.4826 * np.median(np.abs(resid[keep] - med)) + 1e-8
            keep = np.isfinite(y) & (np.abs(resid - med) < 2.8 * mad)
            if keep.sum() < A.shape[1] + 2:
                keep = np.isfinite(y)
            coef = np.linalg.lstsq(A[keep], y[keep], rcond=None)[0]
        # Normalize surface scale; plate tops absorb the arbitrary global scale.
        grid = _surface(np.repeat(np.arange(16), 24), np.tile(np.arange(24), 16)) @ coef
        norm = float(np.mean(grid))
        if not np.isfinite(norm) or norm <= 0:
            continue
        coef = coef / norm
        prepared.append((wells, bg, coef))
        compounds.update(w['compound_id'] for w in wells if w.get('well_type') == 'COMPOUND' and w.get('compound_id') is not None)

    result = {}
    for compound in sorted(compounds):
        conc, obs, plate_idx, gain = [], [], [], []
        for pi, (wells, bg, coef) in enumerate(prepared):
            for w in wells:
                if w.get('well_type') != 'COMPOUND' or w.get('compound_id') != compound:
                    continue
                vals = (w.get('log10_conc_uM'), w.get('fluorescence'))
                if not all(v is not None and np.isfinite(v) for v in vals):
                    continue
                g = float(_surface([w['row']], [w['col']]) @ coef)
                if g <= 0 or not np.isfinite(g):
                    continue
                conc.append(float(vals[0])); obs.append(float(vals[1]) - bg)
                plate_idx.append(pi); gain.append(g)
        if len(conc) < 8:
            continue
        c = np.asarray(conc); y = np.asarray(obs); p = np.asarray(plate_idx, int); g = np.asarray(gain)
        nplates = len(prepared)
        def residual(q):
            ic, slope = q[:2]
            viability = 1.0 / (1.0 + np.power(10.0, np.clip(slope * (c - ic), -30, 30)))
            return y - g * q[2:][p] * viability
        init = np.r_[[-0.3, 1.05], np.full(nplates, 2600.0)]
        lower = np.r_[[-4.0, 0.2], np.full(nplates, 500.0)]
        upper = np.r_[[3.0, 3.0], np.full(nplates, 6000.0)]
        fit = least_squares(residual, init, bounds=(lower, upper), loss='soft_l1', f_scale=35.0, max_nfev=3000)
        result[compound] = {'log10_ic50_uM': float(fit.x[0]), 'hill_slope': float(fit.x[1])}
    return result
