import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import expit

spec = importlib.util.spec_from_file_location('assay_ic50', '/holdout_task/assay_ic50.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_screen.pkl', 'rb') as f:
    screen = pickle.load(f)
plates, truth = screen['plates'], screen['gt']

def coordinates(w):
    r,c,y = np.array([(a['row'], a['col'], a['fluorescence']) for a in w], float).T
    return r,c,y

def depth(r,c):
    return np.minimum.reduce((r,15-r,c,23-c))

def basis(r,c,t,mode):
    u,v = (r-7.5)/7.5, (c-11.5)/11.5
    d = depth(r,c)
    p = 1/(1+(d/t)**2) if mode == 'rational' else np.exp(-d/t)
    cols = [np.ones_like(r), p, u, v]
    if mode == 'edge-tilt':
        cols += [p*u, p*v]
    return np.column_stack(cols)

def fit(r,c,y,mode):
    if mode == 'exp':
        t,b = m._fit_gain(r,c,y)
        return lambda r,c: m._spatial_basis(r,c,t) @ b
    scale = np.median(y)
    def profile(t):
        a = basis(r,c,t,mode)
        z = y/scale
        init = np.linalg.lstsq(a,z,rcond=None)[0]
        return least_squares(lambda b:a@b-z,init,jac=lambda b:a,loss='soft_l1',f_scale=.009)
    t = minimize_scalar(lambda t:profile(t).cost,bounds=(.55,6),method='bounded').x
    b = profile(t).x * scale
    return lambda r,c: basis(r,c,t,mode) @ b

modes = ('exp','rational','edge-tilt')
control_rmse = []
for p in plates:
    w = p['wells']
    bg = np.median([a['fluorescence'] for a in w if a['well_type']=='POS_KILL'])
    r,c,y = coordinates([a for a in w if a['well_type']=='DMSO'])
    y -= bg
    scores = []
    for mode in modes:
        errs = []
        for j in range(len(y)):
            keep = np.arange(len(y)) != j
            f = fit(r[keep],c[keep],y[keep],mode)
            errs.append((f(r[j:j+1],c[j:j+1])[0]-y[j])/np.median(y[keep]))
        scores.append(np.sqrt(np.mean(np.square(errs))))
    control_rmse.append(scores)
control_rmse = np.array(control_rmse)
print('Control fractional RMSE by plate (exp, rational, edge-tilt):', control_rmse.round(5).tolist())
for i,p in enumerate(plates):
    train = [q for j,q in enumerate(plates) if j != i]
    mode = modes[np.argmin(np.delete(control_rmse,i,axis=0).max(axis=0))]
    pred = m.estimate_compound_hill_parameters(train)
    w = p['wells']
    bg = np.median([a['fluorescence'] for a in w if a['well_type']=='POS_KILL'])
    r,c,y = coordinates([a for a in w if a['well_type']=='DMSO'])
    f = fit(r,c,y-bg,mode)
    test = [a for a in w if a['well_type']=='COMPOUND']
    r,c,y = coordinates(test)
    d = depth(r,c)
    par = [pred[a['compound_id']] for a in test]
    v = expit(np.log(10)*np.array([q['hill_slope']*(q['log10_ic50_uM']-a['log10_conc_uM']) for a,q in zip(test,par)]))
    print(p['plate_id'], 'training-selected mode:',mode,'signed viability residual medians by depth (raw, spatial):',[(int(k),*[round(float(np.median(e[d==k])),4) for e in ((y-bg)/np.median([a['fluorescence']-bg for a in w if a['well_type']=='DMSO'])-v,(y-bg)/f(r,c)-v)]) for k in np.unique(d) if np.sum(d==k)>=8])
res = m.estimate_compound_hill_parameters(plates)
err = [max(abs(res[c][k]-truth[c][k]) for c in truth) for k in ('log10_ic50_uM','hill_slope')]
print('Final full-screen maxima:', err)
assert np.all(np.array(err) <= [.09,.18])
