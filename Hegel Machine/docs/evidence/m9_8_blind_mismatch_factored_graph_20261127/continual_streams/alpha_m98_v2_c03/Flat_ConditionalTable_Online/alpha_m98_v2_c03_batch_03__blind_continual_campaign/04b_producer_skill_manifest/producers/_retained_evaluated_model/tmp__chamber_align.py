import importlib.util,pickle,numpy as np
s=importlib.util.spec_from_file_location('base','/tmp/hegel_continual_archive/alpha_m98_v2_c03_batch_02_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; c=i['campaign_diagnostics']; tr=d['train']; dt,t=d['past_dt'],d['eval_times']; train,diag=m._prepare(dict(d,test=c)); y=np.column_stack([tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]); tau,theta=m._fit(train,dt,t,y)
yc=np.column_stack([c['strain_curves_obs']-c['strain_0'][:,None],c['instant_rate_obs']]); pred=m._design(diag,dt,t,tau)@theta; r=yc-pred; past,h,f=diag; drift=f@theta[2:]; g=m._design(diag,dt,t,tau)[:,-1,:2]*tau; z=h[:,None]-g
rm=lambda a:np.sqrt(np.mean(a*a,axis=0))
from scipy.optimize import least_squares
keys=['lab_id','stratum','coupon_features_raw','ref_coupons']; lab=c['lab_id']; n=len(h); ng=len(tr['lab_id']); te=d['test']; offset,total=m._calibration(d['calibration_load_cells']); past=(c['past_channel_raw']-offset)@total; h=(c['hold_channel_raw']-offset)@total
cat=lambda *a:{k:np.concatenate([s[k] for s in a]) for k in keys}; actual_train,_=m._prepare(d); tau0,theta0=m._fit(actual_train,dt,t,y)
E=lambda x:np.r_[-np.expm1(-x*t)/x,1.]; rmse=lambda a:np.sqrt(np.mean(a*a)); good=~np.isnan(c['coupon_features_raw']).any(axis=1)
for name,panel in [('diag',c),('train+diag',cat(tr,c)),('test+diag',cat(te,c)),('all',cat(tr,te,c)),('train',tr),('test',te),('train+test',cat(tr,te))]:
 X=c['coupon_features_raw'].copy(); ref=panel['ref_coupons']; pl=panel['lab_id']; stats=[]
 for l in np.unique(lab):
  M=np.median(ref[pl==l].mean(axis=0)); S=np.median(ref[pl==l].std(axis=0)); X[lab==l]=(X[lab==l]-M)/S; stats.append((M,S))
 if name in ('diag','train+diag','test+diag','all'):
  e={k:c[k][:0] for k in keys}; X=m._features(panel,e)[-n:]
 else:
  X=np.where(np.isnan(X),np.mean(m._features(panel,{k:c[k][:0] for k in keys}),axis=0),X)
 for source,U,V in [('actual',tau0,theta0),('diag-swap',tau,theta)]:
  F=np.c_[np.ones(n),X]; D=F@V[2:]; B=m._design((past,h,np.zeros_like(F)),dt,t,U)@V; C=yc-B
  for kind,A in [('drift',np.c_[np.ones(n),D]),('features',F),('drift2',np.c_[np.ones(n),D,D**2]),('drift-lab',np.c_[np.eye(4)[lab],np.eye(4)[lab]*D[:,None]])]:
   def infer(q):
    def res(p, everything=False):
     s=E(np.exp(p[0])); z=C[q]@s/(s@s); b=np.linalg.lstsq(A[q],z,rcond=None)[0]; r=A@b
     return (r[:,None]*s-C,b) if everything else (r[:,None]*s-C)[q].ravel()
    p=least_squares(res,[0.],bounds=(-3,3)).x; e,b=res(p,True); return e,np.r_[np.exp(p),b]
   e,p=infer(good); folds=[infer(good&(lab!=l))[0][lab==l] for l in np.unique(lab)] if kind!='drift-lab' else [infer(good&(np.arange(n)!=j))[0][[j]] for j in np.flatnonzero(good)]; q=~good
   print(name,source,kind,'good/miss/worst-curve/worst-rate',np.round([rmse(e[good]),rmse(e[q]),max(rmse(s[:,:-1]) for s in folds),max(rmse(s[:,-1]) for s in folds)],4),'p',p.round(3))
 print('panel center,scale',np.round(stats,3).tolist())
