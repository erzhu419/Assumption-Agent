import runpy
from scipy.optimize import least_squares
p=runpy.run_path('/tmp/audit_preloaded.py',run_name='preload_audit') if False else {}
import importlib.util,pickle,numpy as np
path='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',path+'preloaded_hereditary_secant_hazard_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
d=pickle.load(open(path+'public_suc01_T_datasets.pkl','rb'))[0]['dataset']; a=d['train']; lab=a['lab_id']; st=a['stratum']; times=d['eval_times']; dt=d['past_dt']; n=len(lab)
def split(m): return {k:v[m] for k,v in a.items()}
rng=np.random.default_rng(419); fold=np.empty(n,int)
for i,j in np.unique(np.column_stack((lab,st)),axis=0):
    ix=rng.permutation(np.flatnonzero((lab==i)&(st==j))); fold[ix]=np.arange(len(ix))%5
for blocks in (fold,lab):
    errors={k:np.zeros(n) for k in ('baseline','lab_adjusted','lab_center_only','rate')}
    for label in np.unique(blocks):
        v=blocks==label; tr,te=split(~v),split(v)
        x,z,ok,good,w=s._coupons(tr,te); tau,amp,beta=s._fit(tr,dt,times,x,ok,w)
        def drift(q):
            c,r=s._branches(q,dt,times,tau)
            y=q['strain_curves_obs']-q['strain_0'][:,None]-c@amp
            return np.median(y[:,5:]/times[5:],axis=1),q['instant_rate_obs']-r@amp
        target,_=drift(tr); truth,rate=drift(te)
        coupon=z@beta; center=coupon.copy(); adj=coupon.copy()
        for group in np.unique(tr['lab_id']):
            m=(tr['lab_id']==group)&ok; out=te['lab_id']==group
            xx=x[m,1:]@beta[1:]; yy=target[m]-beta[0]-xx
            # Reference calibration errors produce an additive offset and a common gain on coupon drift.
            coef=least_squares(lambda q:(q[0]+q[1]*xx-yy)/.005,[0.,0.],loss='soft_l1',f_scale=1.5).x
            center[out]+=np.median(yy); adj[out]+=coef[0]+coef[1]*(z[out,1:]@beta[1:])
        for key,est in [('baseline',coupon),('lab_center_only',center),('lab_adjusted',adj)]:
            errors[key][v]=abs(np.where(good,.75*est+.25*rate,rate)-truth)
        errors['rate'][v]=abs(rate-truth)
    full=np.isfinite(a['coupon_features_raw']).all(axis=1)
    print('validation:', 'within-lab' if blocks is fold else 'held-out lab')
    for k,e in errors.items(): print(k,np.round([e[full].mean(),max(e[(lab==l)&full].mean() for l in np.unique(lab)),e[~full].mean()],6))
