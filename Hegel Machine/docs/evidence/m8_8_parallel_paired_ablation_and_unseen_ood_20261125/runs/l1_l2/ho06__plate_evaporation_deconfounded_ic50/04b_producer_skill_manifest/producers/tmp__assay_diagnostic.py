import pickle
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

with open('/holdout_task/public_screen.pkl', 'rb') as f:
    data = pickle.load(f)
plates, truth = data['plates'], data['gt']

def features(r, c, model, decay):
    x, y = (r-7.5)/7.5, (c-11.5)/11.5
    dr, dc = np.minimum(r, 15-r), np.minimum(c, 23-c)
    er, ec = np.exp(-dr/decay), np.exp(-dc/decay)
    base = [np.ones_like(x), x, y]
    if model == 'flat': return np.array(base[:1]).T
    if model == 'nearest': base += [np.maximum(er, ec)]
    if model == 'sum': base += [er+ec]
    if model == 'separate': base += [er, ec]
    if model == 'corner': base += [er, ec, er*ec]
    if model == 'cross': base += [np.maximum(er, ec), x*y]
    return np.array(base).T

def fit_gain(X, y, scale):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return least_squares(lambda b: X@b-y, beta, loss='cauchy', f_scale=scale).x

def calibrate(plate, model, decay):
    w = plate['wells']; t = np.array([a['well_type'] for a in w]); f = np.array([a['fluorescence'] for a in w])
    X = features(np.array([a['row'] for a in w]), np.array([a['col'] for a in w]), model, decay)
    b = np.median(f[t=='POS_KILL']); idx = np.flatnonzero(t=='DMSO'); scale = .015*np.median(f[idx]-b)
    g = X@fit_gain(X[idx], f[idx]-b, scale)
    loo = []
    for j in idx:
        k = idx[idx!=j]; loo.append((f[j]-b-X[j]@fit_gain(X[k],f[k]-b,scale))/np.median(g[idx]))
    return [(a['compound_id'], a['log10_conc_uM'], v) for a,v in zip(w,(f-b)/g) if a['well_type']=='COMPOUND'], loo

def errors(rows):
    grouped = {k:[] for k in truth}
    for k,x,y in rows: grouped[k].append((x,y))
    ans = {}
    for k, pairs in grouped.items():
        x,y = np.array(pairs).T
        sol = least_squares(lambda p: expit(np.log(10)*p[1]*(p[0]-x))-y, [x[np.argmin(abs(y-.5))], 1.], bounds=([min(x)-3,.08],[max(x)+3,6]),loss='cauchy',f_scale=.023)
        ans[k] = abs(sol.x - np.array([truth[k]['log10_ic50_uM'],truth[k]['hill_slope']]))
    return np.max(list(ans.values()),axis=0)

for model in ['flat','nearest','sum','separate','corner','cross']:
    for decay in ([2.] if model=='flat' else [1.5,2.5,4.,6.]):
        groups, cv = zip(*(calibrate(p,model,decay) for p in plates))
        reference = errors(sum(groups,[])); omit = np.max([errors(sum([g for j,g in enumerate(groups) if j!=i],[])) for i in range(len(groups))],axis=0)
        print(f'{model:9s} {decay:.1f} full {reference.round(4)} omit {omit.round(4)} control 90pct worst plate {max(np.quantile(np.abs(e),.9) for e in cv):.4f}')
