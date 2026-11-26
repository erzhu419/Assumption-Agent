import importlib.util
import pickle
import numpy as np
s = importlib.util.spec_from_file_location('candidate', '/holdout_task/alpha_ep04_solver.py')
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
with open('/holdout_task/public_alpha_ep04.pkl', 'rb') as f:
    case = pickle.load(f)[0]
d, g = case['dataset'], case['gt']
tr, te = d['train'], d['test']
t, dt = d['eval_times'], d['past_dt']
past, hold, x, complete = m._prepare(d)
n = len(tr['lab_id'])
Y = np.c_[tr['strain_curves_obs'] - tr['strain_0'][:, None], tr['instant_rate_obs']]
P = np.empty_like(Y)
for lab in np.unique(tr['lab_id']):
    r = np.flatnonzero((tr['lab_id'] != lab) & complete[:n]); v = np.flatnonzero(tr['lab_id'] == lab)
    q, w = m._fit(m._Design(past[r], hold[r], dt, t, x[r]), Y[r])
    P[v] = m._Design(past[v], hold[v], dt, t, x[v])(q) @ w
    print('LOLO', lab, 'taus', np.exp(q).round(5), 'weights', w.round(5))
out = m.solve_alpha_ep04(d)
print('estimated vs reference tau:', out['estimated_taus'], g['taus'])
print('estimated vs reference weights:', out['hereditary_weights'], g['theta'])
print('calibrated stress max abs error (history, hold)', np.max(abs(past[n:]-g['te_hist_true'])), np.max(abs(hold[n:]-g['te_hold_true'])))
def compare(name, lab, good, pred, curves, obs, clean_rate=None):
    rate_innov = obs - pred[:, -1]
    diff = curves - pred[:, :-1]
    drift_innov = diff @ t / (t @ t)
    print(name, 'rate innovation quantiles', np.quantile(abs(rate_innov), [0.25,0.5,0.9,0.99,1]))
    for scope, rows in [('complete',good),('imputed',~good)]:
        a, b = rate_innov[rows], drift_innov[rows]
        print(scope,'n',sum(rows),'least-squares correction',a@b/(a@a),'median |innovation|',np.median(abs(a)))
    for c, i in [(0,0), (0,0.5), (0,1), (0.25,1), (0.5,1), (1,1)]:
        z = np.where(good,c,i) * rate_innov
        err = diff - z[:, None] * t
        rmse = lambda a: np.sqrt(np.mean(a*a))
        print((c,i),'curves pooled/complete/imputed/worst-lab/worst-time',np.round([rmse(err),rmse(err[good]),rmse(err[~good]),max(rmse(err[lab==l]) for l in np.unique(lab)),max(rmse(col) for col in err.T)],6),'rate pooled/worst-lab',None if clean_rate is None else np.round([rmse(clean_rate-pred[:,-1]-z),max(rmse((clean_rate-pred[:,-1]-z)[lab==l]) for l in np.unique(lab))],6))
compare('LOLO observed',tr['lab_id'],complete[:n],P,Y[:,:-1],Y[:,-1])
compare('public test clean',te['lab_id'],complete[n:],np.c_[out['predicted_strain']-te['strain_0'][:,None],out['initial_rate_pred']],g['te_clean']-te['strain_0'][:,None],te['instant_rate_obs'],g['te_true_rate0'])
