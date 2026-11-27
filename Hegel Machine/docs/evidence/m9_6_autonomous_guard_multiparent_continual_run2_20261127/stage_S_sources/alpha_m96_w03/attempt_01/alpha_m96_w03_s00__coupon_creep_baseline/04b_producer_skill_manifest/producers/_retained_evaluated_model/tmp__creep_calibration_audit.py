import itertools, pickle, runpy
import numpy as np


def robust_line(x, y):
    n = len(x)
    design = np.column_stack((x, np.ones(n)))
    keep_count = n - (n - 1) // 3
    best = None
    for a, b in itertools.combinations(range(n), 2):
        if abs(x[a] - x[b]) <= 1e-10:
            continue
        g = (y[a] - y[b]) / (x[a] - x[b])
        error = np.abs(y - (g*x + y[a] - g*x[a]))
        core = np.argsort(error)[:keep_count]
        fit = np.linalg.lstsq(design[core], y[core], rcond=None)[0]
        r = (y - design @ fit)[core]
        score = r @ r
        if best is None or score < best[0]:
            best = score, fit
    if best is None:
        return 1., np.median(y - x) if n else 0.
    _, fit = best
    r = y - design @ fit
    middle = np.median(r)
    keep = np.abs(r - middle) <= max(4 * 1.4826 * np.median(np.abs(r - middle)), 1e-8)
    if keep.sum() >= keep_count and np.ptp(x[keep]) > 1e-10:
        fit = np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
    return fit


def calibrate(checks):
    fits = []
    for lab in range(4):
        x, y = (np.asarray(checks[k]) for k in ('cmd_stress', 'true_stress'))
        m = (np.asarray(checks['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
        fits.append(robust_line(x[m], y[m]))
    return np.asarray(fits).T


if __name__ == '__main__':
    root = '/holdout_task/'
    s = runpy.run_path(root+'alpha_m96_w03_s00_solver.py'); old = s['_calibrate']
    e, = pickle.load(open(root+'public_alpha_m96_w03_s00.pkl','rb')); d = e['dataset']; c = d['calibration_load_cells']
    for i in range(4):
        m = c['lab_id'] == i; x = c['cmd_stress'][m]; y = c['true_stress'][m]
        base = old(c); sparse = []; jack = []
        for j in np.flatnonzero(m):
            sel = np.arange(len(m)) != j; g,b = calibrate({k:v[sel] for k,v in c.items()})
            jack.append(abs(g[i]*c['cmd_stress'][j]+b[i]-c['true_stress'][j]))
        for gtrue,btrue in [(base[0][i],base[1][i]),(.86+.085*i,.11-.08*i)]:
            yy0 = y + (gtrue-base[0][i])*x + (btrue-base[1][i])
            for ids in itertools.combinations(range(len(x)),4):
                xx = x[list(ids)]; yy1 = yy0[list(ids)]
                if np.ptp(xx) < .7*np.ptp(x): continue
                for j in range(4):
                    for spike in (-.8,.8):
                        yy = yy1.copy(); yy[j] += spike
                        g,b = robust_line(xx,yy)
                        sparse.append(np.max(abs((g-gtrue)*x + b-btrue)))
        print(i,'sparse EMI median/p95/max',np.quantile(sparse,[.5,.95,1]),'clean jackknife max',max(jack))
    for fn in [old, calibrate]:
        solve = s['solve_alpha_m96_w03_s00']; solve.__globals__['_calibrate'] = fn
        o = solve(d); lab = d['test']['lab_id']; print(fn.__name__,'worst lab public strain/rate',*[max(np.sqrt(np.mean((o[k][lab==i]-e['gt'][truth][lab==i])**2)) for i in range(4)) for k,truth in [('predicted_strain','te_clean'),('initial_rate_pred','te_true_rate0')]])
