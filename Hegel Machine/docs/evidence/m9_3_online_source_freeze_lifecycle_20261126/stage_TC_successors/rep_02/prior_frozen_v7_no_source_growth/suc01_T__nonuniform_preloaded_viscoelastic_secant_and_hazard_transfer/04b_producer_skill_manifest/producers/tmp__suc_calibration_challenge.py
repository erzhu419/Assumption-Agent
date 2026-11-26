import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'preloaded_hereditary_secant_hazard_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def subset(q, rows):
    return {k:v[rows] for k,v in q.items()}

def err(curve, te, t, gt):
    initial = te['strain_0']
    secant = (curve[:,-1]-initial)/np.maximum(abs(te['stress_hold_0']),.5)
    crossing = m._first_crossings(initial,curve,te['critical_strain'],t)
    return np.c_[abs(curve-gt['te_clean']).mean(axis=1),abs(secant-gt['te_secant']),abs(crossing-gt['te_cross'])]

for case in pickle.load(open(root+'public_suc01_T_datasets.pkl','rb')):
    data, gt = case['dataset'], case['gt']
    train, test, t = data['train'], data['test'], data['eval_times']
    for lab in [None,*np.unique(train['lab_id'])]:
        take = np.ones(len(test['lab_id']),dtype=bool) if lab is None else test['lab_id']==lab
        tr = train if lab is None else subset(train, train['lab_id']!=lab)
        te = subset(test,take)
        truth = {k:gt[k][take] for k in ('te_clean','te_secant','te_cross')}
        # Recover the original fitted parameters without changing the artifact.
        params = []
        original = m.least_squares
        def capture(*args,**kw):
            fit = original(*args,**kw)
            params.append(fit.x)
            return fit
        m.least_squares = capture
        try:
            out = m.solve_preloaded_hereditary_transfer(dict(data,train=tr,test=te))
        finally:
            m.least_squares = original
        p, tau = params[0], out['estimated_taus']
        (_, _), (v, complete) = m._coupon_design(tr,te)
        branch = m._branches(te,data['past_dt'],t,tau) @ p[2:4]
        drift = v @ p[4:]
        rate_drift = te['instant_rate_obs']-branch[:,0]
        corrections = {}
        for mode in ('base','offset','affine'):
            corrected = drift.copy()
            for a in np.unique(te['lab_id']):
                observed = (te['lab_id']==a)&complete
                rows = te['lab_id']==a
                if mode=='base' or not observed.any():
                    continue
                center = np.median(drift[observed])
                x = drift[observed]-center
                X = np.ones((len(x),1)) if mode=='offset' else np.c_[np.ones(len(x)),x]
                r = rate_drift[observed]-drift[observed]
                coef = least_squares(lambda z:X@z-r,np.zeros(X.shape[1]),loss='soft_l1',f_scale=.012).x
                corrected[rows] += coef[0]
                if len(coef)==2:
                    corrected[rows] += coef[1]*(drift[rows]-center)
            final = np.where(complete,corrected,rate_drift)
            curve = te['strain_0'][:,None]+branch[:,1:]+final[:,None]*t
            if mode=='base': np.testing.assert_allclose(curve,out['predicted_strain'])
            e = err(curve,te,t,truth)
            corrections[mode] = e.mean(0).round(5).tolist()
            if lab is None:
                groups = [(te['lab_id']==a)&(te['stratum']==s) for a in np.unique(te['lab_id']) for s in np.unique(te['stratum'])]
                groups.extend((complete,~complete))
                print('public',mode,'worst cell or missingness',np.max([e[g].mean(0) for g in groups if g.any()],axis=0).round(5))
        print('held lab',lab,'strain/secant/crossing',corrections)
