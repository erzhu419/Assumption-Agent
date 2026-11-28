import importlib.util, itertools, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['dataset']
c = d['calibration_load_cells']; x, y = c['true_channel_loads'], c['bridge_raw']
# Keep complete axial sweeps and adjacent blocks of mixed sweeps together.
groups = np.where(np.arange(len(x)) < 16, np.arange(len(x)) // 4, 4 + (np.arange(len(x)) - 16) // 16)
for k in np.unique(groups):
    v = groups == k
    b, keep = m._fit_bridge(y[~v], x[~v]); a = x[~v][keep]; z = y[~v][keep]
    e0 = np.linalg.norm(y[v] - m._bridge_forward(x[v], b), axis=1)
    def design(a, j, kind):
        u, w = a[:,j], a[:,(j+1)%4]
        cols = [np.ones(len(a)), *a[:,[(j-1)%4,j,(j+1)%4]].T]
        if kind != 'affine': cols += [u*w,u**3]
        if kind == 'all_second': cols += [a[:,(j+2)%4], *(a[:,i]*a[:,l] for i,l in itertools.combinations_with_replacement(range(4),2) if (i,l) != tuple(sorted((j,(j+1)%4))))]
        if kind == 'rational': cols += [u/(1+(u-w)**2),w/(1+(u+w)**2)]
        return np.column_stack(cols)
    scores = {}
    clean = e0 < .01
    for kind in ('affine','cyclic','all_second','rational'):
        p = np.column_stack([design(x[v],j,kind) @ np.linalg.lstsq(design(a,j,kind),z[:,j],rcond=None)[0] for j in range(4)])
        e = np.linalg.norm(y[v]-p,axis=1)
        scores[kind] = tuple(np.round(np.quantile(e[clean],[.5,1]),7))
    print('bridge block',k,'clean',clean.sum(),'gross',len(clean)-clean.sum(),'median/max',scores)
tr,te = d['train'],d['test']; dt,times=d['past_dt'],d['eval_times']
a,_ = m._features(tr,te); b,_=m._fit_bridge(y,x)
h,s,f=m._calibrated_inputs(tr,a,b)
yc=tr['strain_curves_obs']-tr['strain_0'][:,None]; yr=tr['instant_rate_obs']
for lab in np.unique(tr['lab_id']):
    v=tr['lab_id']==lab; j=~v
    t,w=m._fit(h[j],s[j],f[j],dt,times,yc[j],yr[j]); cc,rr=m._design(t,h[v],s[v],f[v],dt,times)
    for stratum in np.unique(tr['stratum'][v]):
        q=tr['stratum'][v]==stratum; e=(cc@w-yc[v])[q]; er=(rr@w-yr[v])[q]
        print('coupon slice',lab,stratum,int(q.sum()),'signed median strain/rate',np.round([np.median(e),np.median(er)],6),'abs p90 strain/rate',np.round([np.quantile(abs(e),.9),np.quantile(abs(er),.9)],6))
