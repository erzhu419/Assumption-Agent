import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'preloaded_hereditary_secant_hazard_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def split(q, rows):
    return {k:v[rows] for k,v in q.items()}

def features(x):
    a = x[:,1:]
    pairs = [(i,j) for i in range(4) for j in range(i+1,4)]
    differences = np.column_stack([a[:,i]-a[:,j] for i,j in pairs])
    sums = np.column_stack([a[:,i]+a[:,j] for i,j in pairs])
    products = np.column_stack([a[:,i]*a[:,j] for i,j in pairs])
    triples = np.column_stack([np.prod(a[:,np.arange(4)!=i],axis=1) for i in range(4)])
    bounded = np.column_stack([a[:,i]/(1+a[:,j]**2) for i in range(4) for j in range(4)])
    return {'linear':x,'polynomial':np.c_[x,a*a,products,triples], 'rational':np.c_[x,bounded,differences/(1+sums**2)]}

for case in pickle.load(open(root+'public_suc01_T_datasets.pkl','rb')):
    d, gt = case['dataset'], case['gt']
    tr, te, t = d['train'], d['test'], d['eval_times']
    print('observed fields:', sorted(d), sorted(tr))
    base = m.solve_preloaded_hereditary_transfer(d)
    print('baseline tau max rel error:', np.max(np.abs(base['estimated_taus']/gt['taus']-1)))
    for lab in np.unique(tr['lab_id']):
        fit = split(tr,tr['lab_id']!=lab)
        take = te['lab_id']==lab
        val = split(te,take)
        fd = dict(d,train=fit,test=val)
        out = m.solve_preloaded_hereditary_transfer(fd)
        (x,g),(v,h) = m._coupon_design(fit,val)
        train_b = m._branches(fit,d['past_dt'],t,out['estimated_taus'])
        val_b = m._branches(val,d['past_dt'],t,out['estimated_taus'])
        g = g & np.isfinite(fit['instant_rate_obs'])
        y = np.c_[fit['instant_rate_obs'],fit['strain_curves_obs']-fit['strain_0'][:,None]][g]
        taug = np.r_[1.,t]
        errors = {}
        for name,a in features(x).items():
            b = features(v)[name]
            A = np.concatenate((train_b[g], a[g,None,:]*taug[None,:,None]),axis=2)
            A = A.reshape(-1,A.shape[-1]); yy = y.ravel()
            p = least_squares(lambda p:A@p-yy,np.linalg.lstsq(A,yy,rcond=None)[0],jac=lambda p:A,loss='cauchy',f_scale=.025,max_nfev=100).x
            curve = val['strain_0'][:,None]+val_b[:,1:]@p[:2]+(b@p[2:])[:,None]*t
            errors[name] = round(np.abs(curve[h]-gt['te_clean'][take][h]).mean(),5)
        print('held lab',lab,'complete-coupon curve MAE',errors)
    for lab in np.unique(te['lab_id']):
        for s in np.unique(te['stratum']):
            rows = (te['lab_id']==lab)&(te['stratum']==s)
            if rows.any():
                print('cell',lab,s,int(rows.sum()),np.round([np.abs(base['predicted_strain'][rows]-gt['te_clean'][rows]).mean(),np.abs(base['secant_compliance'][rows]-gt['te_secant'][rows]).mean(),np.abs(base['hazard_crossing_times'][rows]-gt['te_cross'][rows]).mean()],5))
    miss = ~np.isfinite(te['coupon_features_raw']).all(axis=1)
    print('missing coupons [observed rate error, true crossing, predicted crossing]:')
    print(np.round(np.c_[(te['instant_rate_obs']-gt['te_true_rate0'])[miss],gt['te_cross'][miss],base['hazard_crossing_times'][miss]],5))
