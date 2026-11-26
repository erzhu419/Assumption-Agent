import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_ep03_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_ep03.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
a, _, y, complete = s._prepare(d)
complete = complete[:len(y)]
labs = d['train']['lab_id']
strata = d['train']['stratum']
cal = d['calibration_load_cells']
e = np.empty_like(y[:, :-1])
er = np.empty(len(y))
for lab in np.unique(labs):
    m = labs == lab
    lt, c = s._fit(a, y, ~m & complete)
    p = a.matrix(lt)[m] @ c
    er[m] = y[m,-1] - p[:,-1]
    e[m] = y[m,:-1] - p[:,:-1] - np.where(complete[m], 0, er[m])[:,None]*a.times
    q = cal['lab_id'] == lab
    cmd, obs = cal['cmd_stress'][q], cal['true_stress'][q]
    loo = [np.polyfit(np.delete(cmd,i), np.delete(obs,i),1) for i in range(len(cmd))]
    z = np.asarray(loo)
    print('lab', lab, 'load-cell max LOO error', np.max(np.abs(z[:,0]*cmd+z[:,1]-obs)), 'gain/bias LOO spans', np.ptp(z,axis=0))
    for k in np.unique(strata[m]):
        for high in [False, True]:
            j=m & (strata==k) & ((a.ov!=0)==high)
            if j.any():
                print('  stratum',k,'high',high,'n',j.sum(),'median residual',round(np.median(e[j]),5),'median abs',round(np.median(np.abs(e[j])),5),'RMSE',round(np.sqrt(np.mean(e[j]**2)),5))
print('absolute curve residual quantiles', np.quantile(np.abs(e),[.5,.9,.95,.98,.99,1]))
worst=np.argsort(np.max(np.abs(e),axis=1))[-10:]
for i in worst:
    print('large residual row', i, 'lab',labs[i], 'stratum',strata[i],'complete',complete[i],'rate residual',round(er[i],4),'curve residual',np.round(e[i],3))
print('eval times', a.times)
