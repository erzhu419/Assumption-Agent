import sys, pickle
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, '/holdout_task')
from _verified_upstream import calibrate_upstream
from batch_solver import solve_batch
from downstream_solver import _features, _loads, _fit, _design, _predict_chamber

with open('/holdout_task/public_batch.pkl','rb') as f:
    item=pickle.load(f)[0]
d=calibrate_upstream(dict(item['dataset'], chamber_verification_trials=item['campaign_diagnostics']))
tr,te,q=(d[k] for k in ('train','test','chamber_verification_trials'))
t,dt=d['eval_times'],d['past_dt']
curve=lambda s:s['strain_curves_obs']-s['strain_0'][:,None]
x=_features(tr,te)[:len(tr['lab_id'])]
p,h=_loads(tr)
truth=curve(tr)
rate=tr['instant_rate_obs']
lab=tr['lab_id']
full=solve_batch(item['dataset']); tau=full['estimated_taus']; theta=full['hereditary_weights']
for g in np.unique(lab):
    m=lab!=g
    a,b=_fit(x[m],p[m],h[m],dt,t,truth[m],rate[m]); A,B=_design(a,x[~m],p[~m],h[~m],dt,t)
    ec=(A@b-truth[~m]).ravel(); er=B@b-rate[~m]
    print('intrinsic lab',g,'tau relative change',np.round(a/tau-1,5),'max weight change',round(abs(b-theta).max(),5),'absolute error median/p90 curve and rate',np.round([*np.quantile(abs(ec),[.5,.9]),*np.quantile(abs(er),[.5,.9])],5))

p,h=_loads(q); xq=_features(q,q)[:len(h)]; lab=q['lab_id']; miss=np.isnan(q['coupon_features_raw']).any(axis=1)
z=np.zeros((len(h),2))
for j,s in enumerate(dt):
    a=np.exp(-s/tau); z=a*z+(1-a)*p[:,j,None]
E=np.vstack((-np.expm1(-t[:,None]/tau),1/tau)); L=np.r_[t,1.]; D=E*tau-L[:,None]
base=h[:,None]*(E@theta[:2]); r=theta[2]+xq@theta[3:]
Y=np.c_[curve(q),q['instant_rate_obs']]
# Test retention of each signed gap h-z, and rational attenuation of routed drift.
# Q is the signed hold-to-last stress contrast, scaled using the training fold only.
contrast=h-p[:,-1]
slices=[('lab '+str(g), lab==g) for g in np.unique(lab)]+[('missing',miss),('complete',~miss),('withdrawal',contrast<0),('loading',contrast>=0)]
fixed_c,fixed_r=_predict_chamber(tau,theta,xq,h,t,q['strain_0']); fixed=np.c_[fixed_c-q['strain_0'][:,None],fixed_r]
vals=lambda err,m:[np.sqrt(np.mean(e**2)) for e in (err[m,:-1],err[m,-1],err[m,-2])]
for kind in ('fixed artifact','conserved','retained memory','rational routing','lab additive nuisance'):
    pred=fixed.copy()
    if kind!='fixed artifact':
        for g in np.unique(lab):
            m=lab!=g
            scale=max(np.median(abs(contrast[m])),1e-12)
            Q=contrast/scale
            retained=-(z*theta[:2])@E.T
            n=(2 if kind=='retained memory' else int(kind in ('rational routing','lab additive nuisance')))
            def predict(v):
                G=r[:,None]*(D@v[:2]); pred0=base+r[:,None]*L
                if kind=='retained memory': return pred0+G+(z*theta[:2]*(-v[2:]))@E.T
                if kind=='rational routing': return pred0+G/(1+v[2]*Q[:,None]**2)
                if kind=='lab additive nuisance': return pred0+G+v[2]*(L+D@v[:2])
                return pred0+G
            lo=np.r_[0.,0.,np.full(n,0. if kind=='rational routing' else -1.)]; hi=np.ones(2+n)
            v=least_squares(lambda v:(predict(v)[m]-Y[m]).ravel(),np.r_[.21451,.51507,np.zeros(n)],bounds=(lo,hi),loss='soft_l1',f_scale=.004).x
            pred[~m]=predict(v)[~m]
    err=pred-Y
    print(kind,'worst curve/rate/final',np.round(np.max([vals(err,m) for _,m in slices if m.any()],axis=0),6),'complete',np.round(vals(err,~miss),6),'lab signed rate means',np.round([err[lab==g,-1].mean() for g in np.unique(lab)],6))
