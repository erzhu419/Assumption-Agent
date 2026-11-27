import importlib.util, pickle, numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_m97_c02_batch_01_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb'))[0]; d=i['dataset']; tr=d['train']; te=d['test']; c=d['calibration_load_cells']; full=m._calibration(d); x=m._align(d); n=len(tr['lab_id']); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; complete=np.isfinite(tr['coupon_features_raw']).all(axis=1)
def subset(a, sel): return tuple(v if j in (2,3) else v[sel] for j,v in enumerate(a))
for lab in range(4):
    fit=(tr['lab_id']!=lab)&complete; inv=(tr['lab_id']==lab)&complete; test=te['lab_id']==lab
    a=m._inputs(d,tr,full,x[:n]); p=m._fit(subset(a,fit),y[fit],r[fit]); cc,rr,*_=m._design(p[:2],*subset(a,fit)); sc=np.array([m._mad(v) for v in (cc@p[2:]-y[fit]).T]); sr=m._mad(rr@p[2:]-r[fit]); rows=np.flatnonzero(c['lab_id']==lab)
    for count in (0,1):
        keep=c['lab_id']!=lab; keep[rows[:count]]=True; cal=m._calibration(dict(d,calibration_load_cells={k:v[keep] for k,v in c.items()})); base=cal.copy(); u,v=(c[k][rows[0]] for k in ('cmd_stress','true_stress'))
        def set_pair(q):
            cal[lab]=[q[0],q[1] if not count else v-u*q[0]]
        def residual(q):
            set_pair(q); aa=subset(m._inputs(d,tr,cal,x[:n]),inv); cc,rr,*_=m._design(p[:2],*aa)
            return np.r_[((cc@p[2:]-y[inv])/sc).ravel(),(rr@p[2:]-r[inv])/sr]
        q0=base[lab,:2-count]; opt=least_squares(residual,q0,bounds=(np.r_[1e-5,[-np.inf]*(1-count)],np.full(2-count,np.inf)),loss='soft_l1',max_nfev=120); set_pair(opt.x)
        def test_rmse(coef):
            aa=subset(m._inputs(d,te,coef,x[n:]),test); cc,rr,*_=m._design(p[:2],*aa)
            return [np.sqrt(np.mean(z*z)) for z in (te['strain_0'][test,None]+cc@p[2:]-i['gt']['te_clean'][test],rr@p[2:]-i['gt']['te_true_rate0'][test])]
        print(lab,count,'archive/refined calibration max error', np.round([np.max(abs(coef[lab]-full[lab])) for coef in (base,cal)],5),'archive/refined held-out curve/rate RMSE', np.round([test_rmse(coef) for coef in (base,cal)],5).tolist(),flush=True)
