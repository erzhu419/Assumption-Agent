import importlib.util, pickle
import numpy as np
p='/holdout_task/alpha_w01_batch_08_solver.py'
s=importlib.util.spec_from_file_location('coupon08',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_alpha_w01_batch_08.pkl','rb'))[0]['dataset']
tr=d['train']; dt,t,sy=(d[k] for k in ('past_dt','eval_times','yield_stress'))
cal=m._calibrate(d['calibration_load_cells']); print('calibration:',cal.tolist())
def eval_cv(cal, lab_holdout=False):
    h,u=m._stress(tr,cal); cross=(np.abs(h).max(axis=1)>sy)|(np.abs(u)>sy)
    complete=np.isfinite(tr['coupon_features_raw']).all(axis=1); lab=tr['lab_id']; fold=np.empty(len(lab),int)
    rng=np.random.default_rng(808)
    for a in range(4):
        for b in (False,True):
            for c in (False,True):
                idx=rng.permutation(np.flatnonzero((lab==a)&(cross==b)&(complete==c)))
                fold[idx]=np.arange(len(idx))%4
    if lab_holdout: fold=lab
    errs={}
    for f in range(4):
        mask=fold==f
        train,test=({k:v[sel] for k,v in tr.items()} for sel in (~mask,mask))
        x,xt,c,ct=m._align(train,test)
        tau,theta=m._fit(train,x,c,cal,dt,t,sy)
        if not lab_holdout: print('fold',f,'taus',tau.round(5),'weights',theta.round(5))
        hh,uu=h[mask],u[mask]; A,B=m._design(tau,hh,uu,xt,dt,t,*m._plastic(hh,uu,dt,sy))
        V=m._lab_drift(train,x,c,tau,theta,cal,dt,t,sy)[test['lab_id']]
        drift=np.where(ct,V[:,0]+V[:,1]*(xt[:,1:]@theta[3:7]),0.)
        for drift_label,delta in [('none',np.zeros(len(ct))),('lab',drift)]:
            for fraction in (0.,.25,.5,1.):
                y0=B@theta+delta
                fix=np.where(ct,fraction,1.)*(test['instant_rate_obs']-y0)
                curve=test['strain_0'][:,None]+A@theta+(delta+fix)[:,None]*t
                er,ec=errs.setdefault((drift_label,fraction),(np.zeros(len(lab)),np.zeros((len(lab),len(t)))))
                er[mask]=y0+fix-test['instant_rate_obs']; ec[mask]=curve-test['strain_curves_obs']
    groups=[(lab==a)&(cross==b) for a in range(4) for b in (False,True)]
    for k,(er,ec) in errs.items():
        print('LOLO' if lab_holdout else 'OOF',k,'curve/rate RMSE %.5f %.5f' % (np.sqrt(np.mean(ec**2)), np.sqrt(np.mean(er**2))), 'worst lab×yield curve/rate RMSE %.5f %.5f' % (max(np.sqrt(np.mean(ec[s]**2)) for s in groups),max(np.sqrt(np.mean(er[s]**2)) for s in groups)), 'worst endpoint %.5f' % max(np.sqrt(np.mean(ec[s,-1]**2)) for s in groups),'missing curve %.5f' % np.sqrt(np.mean(ec[~complete]**2)))
eval_cv(cal); eval_cv(cal,True)
