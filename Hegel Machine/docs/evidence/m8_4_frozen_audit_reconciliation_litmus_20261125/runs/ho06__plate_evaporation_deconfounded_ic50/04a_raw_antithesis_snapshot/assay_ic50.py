import numpy as np
from scipy.optimize import curve_fit

def estimate_compound_hill_parameters(plates: list[dict]) -> dict[str, dict[str, float]]:
    # Naive pooled plate normalization (ignores 2D perimeter evaporation gradient G_plate(r, c) & outliers!)
    by_cmp = {}
    for p in plates:
        dmso = np.mean([w['fluorescence'] for w in p['wells'] if w['well_type'] == 'DMSO'])
        kill = np.mean([w['fluorescence'] for w in p['wells'] if w['well_type'] == 'POS_KILL'])
        for w in p['wells']:
            if w['well_type'] == 'COMPOUND':
                v = (w['fluorescence'] - kill) / max(dmso - kill, 1.0)
                by_cmp.setdefault(w['compound_id'], []).append((float(w['log10_conc_uM']), float(v)))
    out = {}
    for cid, pts in by_cmp.items():
        xs = np.array([a for a, _ in pts])
        ys = np.clip([b for _, b in pts], 0.01, 0.99)
        popt, _ = curve_fit(lambda c, ic50, sl: 1.0 / (1.0 + 10.0 ** (sl * (c - ic50))), xs, ys, p0=[0.0, 1.0], maxfev=2000)
        out[cid] = {'log10_ic50_uM': float(popt[0]), 'hill_slope': float(popt[1])}
    return out
