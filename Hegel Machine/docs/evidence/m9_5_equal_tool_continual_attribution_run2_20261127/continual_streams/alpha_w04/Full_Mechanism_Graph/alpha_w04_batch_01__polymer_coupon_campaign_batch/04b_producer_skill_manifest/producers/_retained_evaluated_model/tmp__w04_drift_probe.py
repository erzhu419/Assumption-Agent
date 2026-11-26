import sys,pickle
sys.path.insert(0,'/holdout_task')
import numpy as np
from scipy.optimize import least_squares
import alpha_w04_batch_01_solver as s

def profile(dt,t,h,u,x,z,y,r):
    # Remove the per-coupon linear drift without estimating it from features.
    # Two long-time readings supply a robust slope pilot; the projection below
    # instead uses all horizons with the instantaneous-rate channel weighted in strain units.
    v=np.r_[t,.4]; vv=v@v
    target=np.column_stack((y,.4*r))
    def proj(a):
        return a-v[None,:,None]*np.einsum('k,nkm->nm',v,a)[:,None,:]/vv
    target=proj(target[:,:,None])[:,:,0]
    def model(p):
        a,b=s._design(p[:2],dt,t,h,u,x,z)
        q=proj(np.concatenate((a[:,:,:2], .4*b[:,None,:2]),axis=1))
        return q@p[2:]-target
    def fit(p,scale,loss):
        return least_squares(lambda w:(model(w)/scale).ravel(),p,bounds=([.05,1.,-np.inf,-np.inf],[2.,25.,np.inf,np.inf]),loss=loss,x_scale='jac')
    p=fit([.6,3.,.7,.5],.03,'soft_l1').x
    e=model(p); scale=np.maximum(.001,1.4826*np.median(abs(e-np.median(e,axis=0)),axis=0)); p=fit(p,2.385*scale,'cauchy').x
    return p

c=pickle.load(open('/holdout_task/public_alpha_w04_batch_01.pkl','rb'))[0]
d,g=c['dataset'],c['gt']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; cal=s._calibrate(d['calibration_load_cells']); x,xt=s._align(tr,te); h,u=s._stress(tr,cal); ht,ut=s._stress(te,cal); z,zt=tr['initial_memory_states'],te['initial_memory_states']; y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; miss=np.isnan(tr['coupon_features_raw']).any(axis=1); mt=np.isnan(te['coupon_features_raw']).any(axis=1)
rmse=lambda e:np.sqrt(np.mean(e**2))
for lab in [0,1,2,3,-1]:
    train=(tr['lab_id']!=lab); test=~train
    taus,w=s._fit(dt,t,h[train],u[train],x[train],z[train],y[train],r[train]); p=profile(dt,t,h[train],u[train],x[train],z[train],y[train],r[train]); print('fold',lab,'base taus',taus,'projected params',p,flush=True)
    if lab<0:
        a,b=s._design(taus,dt,t,ht,ut,xt,zt); ap,bp=s._design(p[:2],dt,t,ht,ut,xt,zt); Y=g['te_clean']-te['strain_0'][:,None]; R=te['instant_rate_obs']; M=mt; labs=te['lab_id']
    else:
        a,b=s._design(taus,dt,t,h[test],u[test],x[test],z[test]); ap,bp=s._design(p[:2],dt,t,h[test],u[test],x[test],z[test]); Y=y[test]; R=r[test]; M=miss[test]; labs=tr['lab_id'][test]
    prior=a@w; corr=(R-b@w)[:,None]*t; prof=ap[:,:,:2]@p[2:]+(R-bp[:,:2]@p[2:])[:,None]*t
    for k in np.unique(labs):
        for missing in (False,True):
            m=(labs==k)&(M==missing)
            if m.any():
                print(' lab/missing/n',k,int(missing),m.sum(),'endpoint RMSE [base, half, anchor, prof]',np.round([rmse((A-Y)[m,-1]) for A in (prior,prior+.5*corr,prior+corr,prof)],5),'median corrections',np.round(np.median(corr[m,-1]),5),flush=True)
