import numpy as np, sys, pickle
sys.path.insert(0,'/tmp/hegel_continual_archive'); import alpha_m99_c02_s00_solver as s
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; tr,te,di=d['train'],d['test'],i['campaign_diagnostics']; dt,t=d['past_dt'],d['eval_times']; b=s._bridge(d['calibration_load_cells']); x,xt=s._features(tr,te); _,xd=s._features(tr,di); taus,w=s._material(tr,b,x,dt,t); p,h=s._loads(di,b); C,R=s._basis(taus,p,h,xd,dt,t); g=R[:,:2]*taus; y=di['strain_curves_obs']-di['strain_0'][:,None]; r=di['instant_rate_obs']; Y=np.c_[y,r]; B=np.vstack((np.c_[-np.expm1(-t[:,None]/taus),t],np.r_[1/taus,1])); n=len(h); lab=di['lab_id']; drift=R[:,2:]@w[2:]; one=np.ones(n); g0,g1=g.T; z0,z1=(h[:,None]-g).T
from scipy.optimize import least_squares
keys=('coupon_features_raw','ref_coupons','lab_id','stratum')
def join(*parts):
 return {k:np.concatenate([a[k] for a in parts]) for k in keys}
empty={k:tr[k][:0] for k in keys}
for name,other in [('train-only',empty),('train+test',te),('train+test+diag',join(te,di))]:
 xx=s._features(tr,other)[0]; ts,ww=s._material(tr,b,xx,dt,t); a,q=s._basis(ts,*s._loads(tr,b),xx,dt,t); e=a@ww+tr['strain_0'][:,None]-tr['strain_curves_obs']; er=q@ww-tr['instant_rate_obs']; print(name,'tau',ts.round(6),'weights',ww.round(6),'median abs curve/rate',np.median(abs(e)),np.median(abs(er)))
P=np.vstack((-np.expm1(-t[:,None]/taus),1/taus)); base=(h[:,None]*w[:2])@P.T; D=Y-base
for label,xx in [('diag only',s._features(empty,di)[1]),('joint train+diag',xd)]:
 v=w[2]+xx@w[3:]; print(label,'drift range',np.round([v.min(),v.max()],4))
 for mode in ('slope','slope+offset','slope+history','slope+history+offset','free drift','shared power'):
  if mode=='slope': F=v[:,None]
  elif mode=='slope+offset': F=np.c_[v,one]
  elif mode=='slope+history': F=np.c_[v,z0,z1]
  elif mode=='slope+history+offset': F=np.c_[v,z0,z1,one]
  elif mode=='free drift': F=np.c_[one,xx]
  else: F=np.c_[v,v*np.abs(v)]
  N=(P[None,:,:,None]*F[:,None,None,:]).reshape(n,len(t)+1,-1); E=np.zeros_like(D); worst=[]
  for l in np.unique(lab):
   m=lab!=l; fit=np.linalg.lstsq(N[m].reshape(-1,N.shape[-1]),D[m].ravel(),rcond=None)[0]; E[~m]=N[~m]@fit-D[~m]; worst.append(np.sqrt(np.mean(E[~m]**2,axis=0)).max())
  fit=np.linalg.lstsq(N.reshape(-1,N.shape[-1]),D.ravel(),rcond=None)[0]; print(mode,'fit/OOF strain/rate/worst',np.round([np.sqrt(np.mean((N@fit-D)**2)),np.sqrt(np.mean(E[:,:-1]**2)),np.sqrt(np.mean(E[:,-1]**2)),max(worst)],5),'coeff',fit.round(4))
