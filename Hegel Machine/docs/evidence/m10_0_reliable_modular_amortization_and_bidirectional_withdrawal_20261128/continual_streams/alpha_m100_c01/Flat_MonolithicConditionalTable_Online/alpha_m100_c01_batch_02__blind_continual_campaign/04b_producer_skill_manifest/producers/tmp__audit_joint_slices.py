import importlib.util, pickle, numpy as np
from scipy.optimize import least_squares, minimize_scalar
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; tr=d['train']; g=p['campaign_diagnostics']; dt=d['past_dt']; t=d['eval_times']; c=d['calibration_load_cells']; b=m._bridge(c['bridge_raw'],c['true_channel_loads']); x,_=m._features(tr,d['test']); xg,_=m._features(g,{k:v[:0] for k,v in g.items()}); P,H,F=m._inputs(tr,x,b); Pg,Hg,Fg=m._inputs(g,xg,b)
y=np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]; yg=np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]

def pred(a,w,P,H,F):
    C,R=m._design(a,P,H,F,dt,t); return np.c_[C@w,R@w]

def fit(i,loss,f,rate_weight):
    a,w=m._fit(P[i],H[i],F[i],dt,t,y[i,:-1],y[i,-1])
    if loss is None: return a,w
    def res(q):
        e=pred(np.exp(q[:2]),q[2:],P[i],H[i],F[i])-y[i]; e[:,-1]*=rate_weight; return e.ravel()
    q=least_squares(res,np.r_[np.log(a),w],loss=loss,f_scale=f,x_scale='jac').x
    return np.exp(q[:2]),q[2:]

for weight in (1, 2, 4):
    setting = (None, 0, 1) if weight == 1 else ('soft_l1', .008, weight)
    rows = []
    for lab in [-1, *np.unique(tr['lab_id'])]:
        a, w = fit(tr['lab_id'] != lab, *setting)
        base = pred(a, w, Pg, Hg, Fg)
        u = Fg @ w[2:]
        def err(tau):
            q = base - yg
            q[:, :-1] += u[:, None] * (-tau*np.expm1(-t/tau)-t)
            return q
        def tune(mask):
            return minimize_scalar(lambda tau: np.mean(err(tau)[mask, :-1]**2), bounds=(.5, 2), method='bounded').x
        if lab == -1:
            full = tune(np.ones(len(yg), bool))
            print('weight', weight, 'full material taus', a, 'full chamber time', full)
        for groups in (g['lab_id'], g['stratum'], np.isnan(g['coupon_features_raw']).any(axis=1)):
            for v in np.unique(groups):
                j = groups == v
                q = err(tune(~j))[j]
                rm = np.sqrt(np.mean(q*q, axis=0))
                rows.append((lab, rm[:-1].max(), rm[-1]))
    z = np.asarray(rows)
    print('weight', weight, 'full train worst chamber', z[z[:,0] == -1,1:].max(axis=0), 'train lab removal worst chamber', z[z[:,0] != -1,1:].max(axis=0), flush=True)
