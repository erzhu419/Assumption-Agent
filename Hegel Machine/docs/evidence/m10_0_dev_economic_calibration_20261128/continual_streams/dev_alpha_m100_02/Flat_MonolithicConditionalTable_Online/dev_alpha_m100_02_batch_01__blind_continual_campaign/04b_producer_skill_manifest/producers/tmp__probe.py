import pickle, numpy as np, importlib.util,glob
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('old',glob.glob('/tmp/hegel_continual_archive/*solver.py')[0]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
a=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=a['dataset']; tr,te,di=ds['train'],ds['test'],a['campaign_diagnostics']; t,dt=ds['eval_times'],ds['past_dt']; np.set_printoptions(precision=4,suppress=True,linewidth=140)
c=ds['calibration_load_cells']; u,v=c['bridge_raw'],c['true_channel_loads']; B=m._bridge(c); e=np.c_[np.ones(len(u)),u]@B-v
print('bridge',B,'err',np.sqrt((e**2).mean(0)),np.max(abs(e),axis=0)); print('sweeps',[ (k,np.shape(q),np.asarray(q).ravel()[:10]) for k,q in c['single_channel_sweeps'].items()]);
xt,xe=m._features(tr,te); st=m._prepare(tr,xt,B); tau,th=m._fit(st,dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('material',tau,th)
lab,refs=(np.concatenate((tr[k],te[k])) for k in ['lab_id','ref_coupons']); xd=di['coupon_features_raw'].copy()
for l in np.unique(lab):
 r=refs[lab==l]; xd[di['lab_id']==l]=(xd[di['lab_id']==l]-np.median(r.mean(0)))/np.median(r.std(0)); print('ref',l,np.sum(lab==l),np.sum(di['lab_id']==l),np.median(r.mean(0)),np.median(r.std(0)),np.median(di['ref_coupons'][di['lab_id']==l].mean(0)),np.median(di['ref_coupons'][di['lab_id']==l].std(0)))
labx=np.r_[tr['lab_id'],te['lab_id']]; strata=np.r_[tr['stratum'],te['stratum']]; raw=np.concatenate([tr['coupon_features_raw'],te['coupon_features_raw']]); x=np.r_[xt,xe]; _,cc,n=np.unique(np.c_[labx,strata],axis=0,return_inverse=True,return_counts=True); w=1/n[cc.ravel(),None]; mu=(np.where(np.isnan(raw),0,x)*w).sum(0)/(~np.isnan(raw)*w).sum(0); xd=np.where(np.isnan(xd),mu,xd); sd=m._prepare(di,xd,B)
def Y(g):return np.column_stack((g['instant_rate_obs'],g['strain_curves_obs']-g['strain_0'][:,None]))
def base(ss):
 C,R=m._design(np.log(tau),ss,dt,t);return np.column_stack((R@th,C@th))
for g,ss in [(tr,st),(di,sd)]:
 e=Y(g)-base(ss); print('res mean',e.mean(0),'mad',np.median(abs(e),axis=0),'max',abs(e).max(0),'sing',np.linalg.svd(e,compute_uv=False)); print('range stress',ss[0].min(),ss[0].max(),ss[1].min(),ss[1].max())
D=np.vstack((np.r_[1/tau,1],np.c_[-np.expm1(-t[:,None]/tau),t])); et=Y(di)-base(sd); coef=np.linalg.lstsq(D,et.T,rcond=None)[0].T; print('native basis resid med/max',np.median(abs(et-coef@D.T),0),abs(et-coef@D.T).max(0)); print('diag lab stratum hold last drift offset fit and residual',np.c_[di['lab_id'],di['stratum'],sd[1],sd[0][:,-1], sd[2]@th[2:],coef,np.linalg.norm(et-coef@D.T,axis=1)])
P=np.column_stack((np.ones(len(u)),u,np.column_stack([u[:,i]*u[:,j] for i in range(4) for j in range(i,4)]))); print('bridge quadratic error',np.sqrt(np.mean((P@np.linalg.lstsq(P,v,rcond=None)[0]-v)**2,0)))
