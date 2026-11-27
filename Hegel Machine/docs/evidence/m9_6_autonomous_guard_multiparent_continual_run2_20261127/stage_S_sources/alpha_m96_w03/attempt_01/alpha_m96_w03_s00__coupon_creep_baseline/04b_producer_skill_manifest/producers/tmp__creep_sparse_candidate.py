import itertools
import pickle
import runpy
import numpy as np


def _calibration_options(x, y, limits, yield_stress, baseline):
    # The check noise on the public panel is about 1e-4 stress units.
    tol = 6e-4
    if len(x) < 4:
        return [baseline]
    a, b = np.triu_indices(len(x), 1)
    distinct = np.abs(x[a] - x[b]) > 1e-10
    a, b = a[distinct], b[distinct]
    g = (y[a] - y[b]) / (x[a] - x[b])
    r = y[None, :] - y[a, None] - g[:, None] * (x[None, :] - x[a, None])
    masks = np.unique(np.abs(r) <= tol, axis=0)
    design = np.column_stack((x, np.ones(len(x))))
    minimum = len(x) - (len(x) - 1) // 3
    options = []
    for m in masks:
        n = m.sum()
        if n < minimum or np.ptp(x[m]) <= 1e-10:
            continue
        fit = np.linalg.lstsq(design[m], y[m], rcond=None)[0]
        if np.max(np.abs(fit[0] * limits + fit[1])) > yield_stress + tol:
            continue
        err = y - design @ fit
        options.append((n, np.ptp(x[m]), err[m] @ err[m], fit))
    if not options:
        return [baseline]
    n = max(v[0] for v in options)
    if n == len(x):
        return [baseline]
    ranked = sorted((v for v in options if v[0] == n), key=lambda v: (-v[1], v[2]))
    return [v[-1] for v in ranked]


def _select_calibration(options, fit, block, x, dt, times, hold_cls):
    obs = block['instant_rate_obs']
    good = np.isfinite(obs)
    curves = block['strain_curves_obs']
    scores = []
    for gb in options:
        # Only the affine calibration varies; coefficients come from other labs.
        arrays = np.broadcast_to(np.asarray(gb)[:, None], (2, 4))
        hold = hold_cls(dt, times, block, x, arrays)
        pred = hold.design(fit[:2])[0] @ fit[2:]
        de = np.abs(pred[:, :-1] + hold.base[:, None] - curves)
        score = np.nanmedian(de)
        if good.any():
            score += np.median(np.abs(pred[good, -1] - obs[good]))
        scores.append(score)
    return options[np.argmin(scores)]


if __name__ == '__main__':
    root = '/holdout_task/'
    s = runpy.run_path(root + 'alpha_m96_w03_s00_solver.py')
    e, = pickle.load(open(root + 'public_alpha_m96_w03_s00.pkl', 'rb'))
    d = e['dataset']; tr, te = d['train'], d['test']; c = d['calibration_load_cells']
    cal = np.array(s['_calibrate'](c)); x, _ = s['_align'](tr, te); dt, t = d['past_dt'], d['eval_times']
    for lab in range(4):
        m = tr['lab_id'] == lab
        other = {k:v[~m] for k,v in tr.items()}; target = {k:v[m] for k,v in tr.items()}
        fit = s['_fit'](s['_Hold'](dt,t,other,x[~m],cal),other)
        cm = c['lab_id'] == lab; cmd, y = c['cmd_stress'][cm], c['true_stress'][cm]
        g0,b0 = cal[:,lab]
        commands = np.concatenate([d[k][v][d[k]['lab_id']==lab].ravel() for k in ('train','test') for v in ('past_stress_cmd','stress_hold_cmd')])
        baseline_err, new_err = [], []; decisions = 0
        for g,b in [(g0,b0),(.86+.085*lab,.11-.08*lab)]:
            base = y+(g-g0)*cmd+b-b0
            to_cmd = lambda z: (g0*z+b0-b)/g
            limits = to_cmd(np.array([commands.min(), commands.max()]))
            tb = {**target, **{v:to_cmd(target[v]) for v in ('past_stress_cmd','stress_hold_cmd')}}
            for ids in itertools.combinations(range(len(cmd)),4):
                xx = cmd[list(ids)]
                if np.ptp(xx) < .7*np.ptp(cmd): continue
                for j in range(4):
                    for spike in (-.8,.8):
                        yy = base[list(ids)].copy(); yy[j] += spike
                        old = np.array(s['_calibrate']({'lab_id':np.full(4,lab),'cmd_stress':xx,'true_stress':yy}))[:,lab]
                        opts = _calibration_options(xx,yy,limits,d['yield_stress'],old)
                        new = opts[0]
                        if np.max(np.abs((np.asarray(opts)-new) @ np.vstack((limits, np.ones(2))))) > .01:
                            decisions += 1
                            new = _select_calibration(opts,fit,tb,x[m],dt,t,s['_Hold'])
                        for z,r in [(old,baseline_err),(new,new_err)]:
                            r.append(np.max(abs((z[0]-g)*cmd+z[1]-b)))
        print(lab,'cross-lab tie breaks',decisions,'baseline p95/max',np.quantile(baseline_err,[.95,1]),'candidate p95/max',np.quantile(new_err,[.95,1]))
