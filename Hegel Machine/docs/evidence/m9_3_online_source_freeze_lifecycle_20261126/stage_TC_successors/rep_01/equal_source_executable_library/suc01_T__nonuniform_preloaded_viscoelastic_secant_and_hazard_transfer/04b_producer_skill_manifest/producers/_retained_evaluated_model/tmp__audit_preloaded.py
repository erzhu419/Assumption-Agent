import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
P='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',P+'preloaded_hereditary_secant_hazard_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
d=pickle.load(open(P+'public_suc01_T_datasets.pkl','rb'))[0]['dataset']; a=d['train']; t=d['eval_times']; dt=d['past_dt']; lab=a['lab_id']; st=a['stratum']; n=len(lab)

def subsets(mask): return {k:v[mask] for k,v in a.items()}
def calibrate(tr,va,kind):
    if kind=='baseline': return s._coupons(tr,va)
    labels=np.r_[tr['lab_id'],va['lab_id']]; panel=np.r_[tr['ref_coupons'],va['ref_coupons']]; raw=np.r_[tr['coupon_features_raw'],va['coupon_features_raw']].copy(); nt=len(tr['lab_id'])
    for label in np.unique(labels):
        m=labels==label; r=panel[m]; center,scale=np.nanmean(r),np.nanstd(r)
        raw[m]=(raw[m]-center)/scale
    valid=np.isfinite(raw); w=s._cell_weights(tr['lab_id'],tr['stratum']); fill=(w@np.where(valid[:nt],raw[:nt],0))/(w@valid[:nt]); x=np.column_stack((np.ones(len(raw)),np.where(valid,raw,fill))); complete=valid.all(axis=1)
    return x[:nt],x[nt:],complete[:nt],complete[nt:],w

def terms(x):
    u=x[:,1:]; i,j=np.triu_indices(u.shape[1],1)
    return np.column_stack((u*u,u[:,i]*u[:,j],u/(1+.25*u*u)))

def evaluate(kind, folds):
    names=('coupon','blend','rate','augmented','masked_equal_blend'); errors={k:np.full(n,np.nan) for k in names}; onset=np.full(n,np.nan); ts=[]
    for m in folds:
        tr,va=subsets(~m),subsets(m); x,z,ok,vok,w=calibrate(tr,va,kind); tau,amp,beta=s._fit(tr,dt,t,x,ok,w); ts.append(tau)
        ac,ar=s._branches(tr,dt,t,tau); bc,br=s._branches(va,dt,t,tau)
        # Estimate a reference drift from held-out strain measurements only, not the onset sensor.
        yy=va['strain_curves_obs']-va['strain_0'][:,None]-bc@amp
        target=np.median(yy[:,5:]/t[5:],axis=1); dr=va['instant_rate_obs']-br@amp; dc=z@beta; onset[m]=dr-dc
        yt=tr['strain_curves_obs']-tr['strain_0'][:,None]-ac@amp; v=terms(x); vtest=terms(z)
        def err(q): return np.r_[((x[ok]@q[:5]+v[ok]@q[5:])[:,None]*t-yt[ok]).ravel()/.018, q[5:]/.002]
        p=least_squares(err,np.r_[beta,np.zeros(v.shape[1])],loss='cauchy',f_scale=2.).x
        zz=z.copy(); wv=np.average(x[:,1:],axis=0,weights=w)
        # One masked coupon per complete specimen; fixed rotation covers all feature positions independently of outcomes.
        f=np.flatnonzero(m)%4; zz[np.arange(len(z)),f+1]=wv[f]; dm=zz@beta
        predictions=(np.where(vok,dc,dr),np.where(vok,.75*dc+.25*dr,dr),dr,np.where(vok,.75*(z@p[:5]+vtest@p[5:])+.25*dr,dr),np.where(vok,.5*dm+.5*dr,dr))
        for k,pred in zip(names,predictions): errors[k][m]=abs(pred-target)
    print(kind,'tau ranges',np.round(np.min(ts,axis=0),5),np.round(np.max(ts,axis=0),5),'median onset innovation by lab',[(int(q),round(float(np.nanmedian(onset[(lab==q)&full])),5)) for q in np.unique(lab)])
    for k,e in errors.items():
        print(k,'terminal drift proxy MAE complete, worst complete lab, incomplete',np.round([np.mean(e[full]),max(np.mean(e[full&(lab==q)]) for q in np.unique(lab)),np.mean(e[~full])],5))

full=np.isfinite(a['coupon_features_raw']).all(axis=1)
for kind in ('baseline','pooled_moments'): evaluate(kind,[lab==q for q in np.unique(lab)])
